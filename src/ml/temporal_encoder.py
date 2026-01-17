
"""X-CLIP temporal encoder for video understanding.

Provides clip-level temporal embeddings using X-CLIP model,
with sliding window extraction and batched GPU encoding.
"""
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.core.config import settings
from src.core.exceptions import MLModelError, ProcessingError
from src.core.logging import get_logger
from src.ml.parallel_encoder import DeviceAllocation, get_device_manager

logger = get_logger(__name__)


@dataclass
class ClipConfig:
    """Configuration for video clip extraction."""

    window_frames: int = 16  # Frames per clip
    clip_fps: float = 8.0  # FPS for clip extraction
    stride: float = 0.5  # Overlap ratio (0.5 = 50% overlap)
    max_clips: int = 500  # Maximum clips per video

    @property
    def window_duration_ms(self) -> int:
        """Duration of each clip in milliseconds."""
        return int((self.window_frames / self.clip_fps) * 1000)

    @property
    def stride_frames(self) -> int:
        """Number of frames to advance between clips."""
        return max(1, int(self.window_frames * (1 - self.stride)))


@dataclass
class VideoClip:
    """Extracted video clip with metadata."""

    index: int
    frames: NDArray[np.uint8]  # Shape: (T, H, W, C)
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int


@dataclass
class ClipEmbedding:
    """Temporal embedding for a video clip."""

    clip_index: int
    embedding: NDArray[np.float32]
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int


class VideoClipExtractor:
    """Extracts sliding window clips from video.
    
    Uses OpenCV or Decord for efficient video decoding.
    
    Example:
        extractor = VideoClipExtractor(config)
        clips = extractor.extract("video.mp4")
    """

    def __init__(self, config: ClipConfig | None = None) -> None:
        self.config = config or ClipConfig(
            window_frames=settings.temporal_window_frames,
            clip_fps=settings.temporal_fps,
            stride=settings.temporal_stride,
        )
        self._use_decord = self._check_decord()

    def _check_decord(self) -> bool:
        """Check if decord is available."""
        try:
            import decord
            return True
        except ImportError:
            logger.info("Decord not available, using OpenCV for video decoding")
            return False

    def extract(self, video_path: str | Path) -> list[VideoClip]:
        """Extract clips from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of extracted video clips
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ProcessingError(f"Video not found: {video_path}")

        if self._use_decord:
            return self._extract_decord(video_path)
        else:
            return self._extract_opencv(video_path)

    def _extract_opencv(self, video_path: Path) -> list[VideoClip]:
        """Extract clips using OpenCV."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ProcessingError(f"Could not open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if video_fps <= 0:
                video_fps = 30.0

            # Calculate frame sampling interval
            sample_interval = max(1, int(video_fps / self.config.clip_fps))

            clips: list[VideoClip] = []
            clip_index = 0
            current_frame = 0

            while current_frame < total_frames and clip_index < self.config.max_clips:
                frames: list[NDArray[np.uint8]] = []
                start_frame = current_frame

                # Extract frames for this clip
                for _ in range(self.config.window_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    ret, frame = cap.read()

                    if not ret:
                        break

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

                    current_frame += sample_interval

                # Only add if we got enough frames
                if len(frames) >= self.config.window_frames // 2:
                    # Pad if necessary
                    while len(frames) < self.config.window_frames:
                        frames.append(frames[-1])

                    end_frame = start_frame + len(frames) * sample_interval
                    start_ms = int((start_frame / video_fps) * 1000)
                    end_ms = int((end_frame / video_fps) * 1000)

                    clips.append(
                        VideoClip(
                            index=clip_index,
                            frames=np.stack(frames),
                            start_frame=start_frame,
                            end_frame=end_frame,
                            start_ms=start_ms,
                            end_ms=end_ms,
                        )
                    )

                    clip_index += 1

                # Move to next clip position (with stride)
                current_frame = start_frame + self.config.stride_frames * sample_interval

            logger.info(
                "Clips extracted",
                video=str(video_path),
                clip_count=len(clips),
                total_frames=total_frames,
            )

            return clips

        finally:
            cap.release()

    def _extract_decord(self, video_path: Path) -> list[VideoClip]:
        """Extract clips using Decord (faster)."""
        import decord
        from decord import VideoReader, cpu

        decord.bridge.set_bridge("native")

        vr = VideoReader(str(video_path), ctx=cpu(0))
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)

        if video_fps <= 0:
            video_fps = 30.0

        sample_interval = max(1, int(video_fps / self.config.clip_fps))

        clips: list[VideoClip] = []
        clip_index = 0
        current_frame = 0

        while current_frame < total_frames and clip_index < self.config.max_clips:
            # Calculate frame indices for this clip
            frame_indices = []
            for i in range(self.config.window_frames):
                idx = current_frame + i * sample_interval
                if idx < total_frames:
                    frame_indices.append(idx)
                else:
                    # Pad with last valid frame
                    frame_indices.append(min(idx, total_frames - 1))

            if len(frame_indices) >= self.config.window_frames // 2:
                # Batch read frames
                frames = vr.get_batch(frame_indices).asnumpy()

                start_frame = frame_indices[0]
                end_frame = frame_indices[-1]
                start_ms = int((start_frame / video_fps) * 1000)
                end_ms = int((end_frame / video_fps) * 1000)

                clips.append(
                    VideoClip(
                        index=clip_index,
                        frames=frames,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )

                clip_index += 1

            current_frame += self.config.stride_frames * sample_interval

        logger.info(
            "Clips extracted (decord)",
            video=str(video_path),
            clip_count=len(clips),
        )

        return clips

    async def extract_async(self, video_path: str | Path) -> list[VideoClip]:
        """Async wrapper for clip extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract, video_path)


class XCLIPEncoder:
    """X-CLIP encoder for temporal video understanding.
    
    Wraps the X-CLIP model from Hugging Face for batched
    video clip encoding with GPU optimization.
    
    Example:
        encoder = XCLIPEncoder()
        await encoder.load_model()
        
        embeddings = encoder.encode_clips(clips)
        text_embedding = encoder.encode_text("person running")
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.xclip_model
        self._device = device
        self._batch_size = batch_size or settings.temporal_batch_size

        self._model: Any = None
        self._processor: Any = None
        self._loaded = False

    @property
    def device(self) -> str:
        """Get device string."""
        if self._device:
            return self._device

        # Auto-detect
        dm = get_device_manager()
        allocation = dm.allocate("xclip")
        self._device = allocation.device
        return self._device

    @property
    def batch_size(self) -> int:
        """Get optimal batch size."""
        if self._batch_size and self._batch_size > 0:
            return self._batch_size

        # Auto-detect based on device
        dm = get_device_manager()
        allocation = dm.allocate("xclip")
        return allocation.max_batch_size

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    async def load_model(self) -> None:
        """Load X-CLIP model and processor."""
        if self._loaded:
            return

        try:
            from transformers import XCLIPModel, XCLIPProcessor

            logger.info(
                "Loading X-CLIP model",
                model=self.model_name,
                device=self.device,
            )

            # Load in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self._processor, self._model = await loop.run_in_executor(
                None, self._load_model_sync
            )

            self._loaded = True
            logger.info("X-CLIP model loaded successfully")

        except Exception as e:
            logger.error("Failed to load X-CLIP model", error=str(e))
            raise MLModelError(f"Failed to load X-CLIP model: {e}")

    def _load_model_sync(self) -> tuple[Any, Any]:
        """Synchronous model loading."""
        from transformers import XCLIPModel, XCLIPProcessor

        processor = XCLIPProcessor.from_pretrained(self.model_name)
        model = XCLIPModel.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()

        return processor, model

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._loaded:
            raise MLModelError("X-CLIP model not loaded. Call load_model() first.")

    def encode_clips(
        self,
        clips: list[VideoClip],
    ) -> list[ClipEmbedding]:
        """Encode video clips to embeddings.
        
        Args:
            clips: List of video clips
            
        Returns:
            List of clip embeddings
        """
        self._ensure_loaded()

        if not clips:
            return []

        embeddings: list[ClipEmbedding] = []

        # Process in batches
        for batch_start in range(0, len(clips), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(clips))
            batch_clips = clips[batch_start:batch_end]

            batch_embeddings = self._encode_batch(batch_clips)
            embeddings.extend(batch_embeddings)

        logger.debug("Encoded clips", count=len(embeddings))
        return embeddings

    def _encode_batch(self, clips: list[VideoClip]) -> list[ClipEmbedding]:
        """Encode a batch of clips."""
        # Prepare video tensors
        # X-CLIP expects: list of (T, C, H, W) tensors
        videos = []
        for clip in clips:
            # clip.frames is (T, H, W, C), need to convert to list of PIL or tensor
            # X-CLIP processor handles numpy arrays
            videos.append(list(clip.frames))

        # Process through X-CLIP
        inputs = self._processor(
            videos=videos,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            video_features = self._model.get_video_features(**inputs)
            # Normalize
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        embeddings_np = video_features.cpu().numpy().astype(np.float32)

        # Build ClipEmbedding objects
        results: list[ClipEmbedding] = []
        for i, clip in enumerate(clips):
            results.append(
                ClipEmbedding(
                    clip_index=clip.index,
                    embedding=embeddings_np[i],
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    start_ms=clip.start_ms,
                    end_ms=clip.end_ms,
                )
            )

        return results

    def encode_text(self, text: str) -> NDArray[np.float32]:
        """Encode text query for temporal matching.
        
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
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().astype(np.float32)[0]

    async def extract_and_encode(
        self,
        video_path: str | Path,
        config: ClipConfig | None = None,
    ) -> list[ClipEmbedding]:
        """Extract clips and encode in one step.
        
        Args:
            video_path: Path to video file
            config: Optional clip extraction config
            
        Returns:
            List of clip embeddings
        """
        # Extract clips
        extractor = VideoClipExtractor(config)
        clips = await extractor.extract_async(video_path)

        if not clips:
            return []

        # Ensure model is loaded
        await self.load_model()

        # Encode clips
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.encode_clips, clips
        )

        return embeddings

    def score_relevance(
        self,
        clip_embeddings: list[ClipEmbedding],
        query: str,
    ) -> list[tuple[int, float]]:
        """Score clip relevance against a text query.
        
        Args:
            clip_embeddings: List of clip embeddings
            query: Text query
            
        Returns:
            List of (clip_index, score) sorted by score descending
        """
        if not clip_embeddings:
            return []

        # Get text embedding
        text_emb = self.encode_text(query)

        # Stack clip embeddings
        clip_matrix = np.stack([ce.embedding for ce in clip_embeddings])

        # Compute cosine similarity
        similarities = clip_matrix @ text_emb

        # Create scored list
        scored = [
            (clip_embeddings[i].clip_index, float(similarities[i]))
            for i in range(len(clip_embeddings))
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    async def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._loaded:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._loaded = False

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("X-CLIP model unloaded")


# Convenience function
async def create_temporal_encoder(
    device: str | None = None,
) -> XCLIPEncoder:
    """Create and initialize a temporal encoder."""
    encoder = XCLIPEncoder(device=device)
    await encoder.load_model()
    return encoder
