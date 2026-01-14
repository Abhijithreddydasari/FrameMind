"""CLIP-based frame embedding and relevance scoring.

This module provides CLIP model integration for computing frame embeddings
and scoring frame relevance against text queries. It's the core of the
intelligent frame selection pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from src.core.config import settings
from src.core.exceptions import MLModelError
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CLIPConfig:
    """Configuration for CLIP model."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"
    batch_size: int = 32
    cache_embeddings: bool = True


@dataclass
class FrameEmbedding:
    """Frame embedding with metadata."""

    frame_index: int
    embedding: NDArray[np.float32]
    timestamp_ms: int = 0


class CLIPScorer:
    """CLIP-based frame embedding and relevance scoring.
    
    Provides functionality for:
    - Computing frame embeddings using CLIP vision encoder
    - Computing text embeddings for queries
    - Scoring frame relevance against queries
    - Batch processing for efficiency
    
    Example:
        scorer = CLIPScorer()
        await scorer.load_model()
        
        # Get frame embeddings
        embeddings = scorer.embed_frames(frames)
        
        # Score against query
        scores = scorer.score_relevance(embeddings, "a person speaking")
    """

    def __init__(self, config: CLIPConfig | None = None) -> None:
        self.config = config or CLIPConfig(
            model_name=settings.clip_model,
            device=settings.clip_device,
        )
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    async def load_model(self) -> None:
        """Load CLIP model and processor.
        
        Uses lazy loading and caches the model in memory.
        """
        if self._loaded:
            return

        try:
            # Import here to avoid loading torch on startup
            from transformers import CLIPModel, CLIPProcessor

            logger.info(
                "Loading CLIP model",
                model=self.config.model_name,
                device=self.config.device,
            )

            self._processor = CLIPProcessor.from_pretrained(self.config.model_name)
            self._model = CLIPModel.from_pretrained(self.config.model_name)
            self._model.to(self.config.device)
            self._model.eval()

            self._loaded = True

            logger.info("CLIP model loaded successfully")

        except Exception as e:
            logger.error("Failed to load CLIP model", error=str(e))
            raise MLModelError(f"Failed to load CLIP model: {e}")

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._loaded:
            raise MLModelError("CLIP model not loaded. Call load_model() first.")

    def embed_frames(
        self,
        frames: list[NDArray[np.uint8]] | list[Image.Image],
        frame_indices: list[int] | None = None,
        timestamps_ms: list[int] | None = None,
    ) -> list[FrameEmbedding]:
        """Compute CLIP embeddings for a batch of frames.
        
        Args:
            frames: List of frames (numpy arrays or PIL Images)
            frame_indices: Optional frame indices for tracking
            timestamps_ms: Optional timestamps for each frame
            
        Returns:
            List of FrameEmbedding objects
        """
        self._ensure_loaded()

        if not frames:
            return []

        # Convert numpy arrays to PIL Images
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                # Assume BGR from OpenCV, convert to RGB
                import cv2
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(rgb_frame))
            else:
                pil_frames.append(frame)

        # Default indices and timestamps
        if frame_indices is None:
            frame_indices = list(range(len(frames)))
        if timestamps_ms is None:
            timestamps_ms = [0] * len(frames)

        embeddings: list[FrameEmbedding] = []

        # Process in batches
        for batch_start in range(0, len(pil_frames), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(pil_frames))
            batch_frames = pil_frames[batch_start:batch_end]
            batch_indices = frame_indices[batch_start:batch_end]
            batch_timestamps = timestamps_ms[batch_start:batch_end]

            # Process batch
            inputs = self._processor(
                images=batch_frames,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy
            batch_embeddings = image_features.cpu().numpy().astype(np.float32)

            for i, (idx, ts, emb) in enumerate(
                zip(batch_indices, batch_timestamps, batch_embeddings)
            ):
                embeddings.append(
                    FrameEmbedding(
                        frame_index=idx,
                        embedding=emb,
                        timestamp_ms=ts,
                    )
                )

        logger.debug("Computed frame embeddings", count=len(embeddings))

        return embeddings

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Compute CLIP text embedding for a query.
        
        Args:
            text: Query text
            
        Returns:
            Normalized text embedding
        """
        self._ensure_loaded()

        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP max token length
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().astype(np.float32)[0]

    def score_relevance(
        self,
        frame_embeddings: list[FrameEmbedding],
        query: str,
    ) -> list[tuple[int, float]]:
        """Score frame relevance against a text query.
        
        Args:
            frame_embeddings: List of frame embeddings
            query: Text query
            
        Returns:
            List of (frame_index, relevance_score) tuples, sorted by score descending
        """
        if not frame_embeddings:
            return []

        # Get text embedding
        text_embedding = self.embed_text(query)

        # Stack frame embeddings
        frame_matrix = np.stack([fe.embedding for fe in frame_embeddings])

        # Compute cosine similarity (embeddings are normalized)
        similarities = frame_matrix @ text_embedding

        # Create scored list
        scored = [
            (frame_embeddings[i].frame_index, float(similarities[i]))
            for i in range(len(frame_embeddings))
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def compute_frame_similarities(
        self,
        embeddings: list[FrameEmbedding],
    ) -> NDArray[np.float32]:
        """Compute pairwise similarity matrix between frames.
        
        Useful for clustering and diversity sampling.
        
        Args:
            embeddings: List of frame embeddings
            
        Returns:
            NxN similarity matrix
        """
        if not embeddings:
            return np.array([])

        frame_matrix = np.stack([fe.embedding for fe in embeddings])
        similarities = frame_matrix @ frame_matrix.T

        return similarities.astype(np.float32)

    def find_diverse_frames(
        self,
        embeddings: list[FrameEmbedding],
        n_frames: int,
    ) -> list[int]:
        """Select diverse frames using max-min distance sampling.
        
        Greedy algorithm that iteratively selects frames that are
        maximally different from already selected frames.
        
        Args:
            embeddings: List of frame embeddings
            n_frames: Number of frames to select
            
        Returns:
            List of selected frame indices
        """
        if len(embeddings) <= n_frames:
            return [e.frame_index for e in embeddings]

        # Compute similarity matrix
        similarities = self.compute_frame_similarities(embeddings)

        # Convert to distance (1 - similarity)
        distances = 1 - similarities

        # Greedy max-min selection
        n = len(embeddings)
        selected_mask = np.zeros(n, dtype=bool)
        selected_indices: list[int] = []

        # Start with the frame most different from all others (lowest avg similarity)
        avg_distances = distances.mean(axis=1)
        first_idx = int(np.argmax(avg_distances))
        selected_mask[first_idx] = True
        selected_indices.append(embeddings[first_idx].frame_index)

        for _ in range(n_frames - 1):
            # Find minimum distance to any selected frame for each unselected frame
            min_distances = np.full(n, -np.inf)

            for i in range(n):
                if not selected_mask[i]:
                    # Minimum distance to any selected frame
                    min_dist = np.min(distances[i, selected_mask])
                    min_distances[i] = min_dist

            # Select frame with maximum minimum distance
            next_idx = int(np.argmax(min_distances))
            selected_mask[next_idx] = True
            selected_indices.append(embeddings[next_idx].frame_index)

        return selected_indices

    async def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._loaded:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False

            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("CLIP model unloaded")
