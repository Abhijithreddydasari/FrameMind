"""Intelligent frame selection combining multiple signals.

This module orchestrates the frame selection pipeline, combining:
- Shot boundary detection for temporal structure
- CLIP embeddings for visual content
- Clustering for diversity
- Query relevance scoring

The goal is to reduce thousands of frames to 10-20 key frames for VLM analysis.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from src.core.config import settings
from src.core.exceptions import FrameExtractionError
from src.core.logging import get_logger
from src.core.models import Frame, FrameType, SceneBoundary
from src.ml.clip_scorer import CLIPScorer, FrameEmbedding
from src.ml.shot_detector import ShotDetector

logger = get_logger(__name__)


@dataclass
class SelectionConfig:
    """Configuration for frame selection."""

    target_frames: int = 15
    min_frames: int = 5
    max_frames: int = 30

    # Weights for frame scoring
    temporal_weight: float = 0.3  # Coverage across time
    scene_boundary_weight: float = 0.3  # Prefer scene boundaries
    diversity_weight: float = 0.2  # Visual diversity
    query_weight: float = 0.2  # Query relevance (when available)

    # Temporal sampling
    ensure_start_end: bool = True  # Always include first and last frames
    min_temporal_gap_ms: int = 2000  # Minimum 2s between selected frames


@dataclass
class SelectionResult:
    """Result of frame selection."""

    selected_indices: list[int]
    selected_frames: list[Frame]
    embeddings: list[FrameEmbedding]
    scene_boundaries: list[SceneBoundary]
    selection_scores: dict[int, float]  # frame_index -> score


class FrameSelector:
    """Orchestrates intelligent frame selection from video.
    
    Combines multiple signals to select the most informative frames
    while ensuring temporal coverage and visual diversity.
    
    Example:
        selector = FrameSelector()
        await selector.initialize()
        
        result = await selector.select_from_video(
            video_path="video.mp4",
            query="What is happening in the video?",
        )
        
        print(f"Selected {len(result.selected_indices)} frames")
    """

    def __init__(self, config: SelectionConfig | None = None) -> None:
        self.config = config or SelectionConfig(
            target_frames=settings.target_keyframes,
        )
        self.shot_detector = ShotDetector()
        self.clip_scorer = CLIPScorer()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize ML models."""
        if not self._initialized:
            await self.clip_scorer.load_model()
            self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure selector is initialized."""
        if not self._initialized:
            raise RuntimeError("FrameSelector not initialized. Call initialize() first.")

    async def select_from_video(
        self,
        video_path: str | Path,
        query: str | None = None,
        sample_fps: float | None = None,
    ) -> SelectionResult:
        """Select key frames from a video file.
        
        Args:
            video_path: Path to video file
            query: Optional query to boost relevant frames
            sample_fps: Frame extraction rate (default: settings.frame_extraction_fps)
            
        Returns:
            SelectionResult with selected frames and metadata
        """
        self._ensure_initialized()

        video_path = Path(video_path)
        sample_fps = sample_fps or settings.frame_extraction_fps

        # Extract frames
        frames, frame_indices, timestamps_ms, video_fps = self._extract_frames(
            video_path, sample_fps
        )

        if not frames:
            raise FrameExtractionError(f"No frames extracted from {video_path}")

        logger.info(
            "Frames extracted",
            total_frames=len(frames),
            video_path=str(video_path),
        )

        # Detect shot boundaries
        boundaries = self.shot_detector.detect_from_frames(frames)
        boundary_indices = {b.frame_index for b in boundaries}

        logger.info("Shot boundaries detected", count=len(boundaries))

        # Compute CLIP embeddings
        embeddings = self.clip_scorer.embed_frames(
            frames,
            frame_indices=frame_indices,
            timestamps_ms=timestamps_ms,
        )

        # Score and select frames
        selected_indices, scores = self._score_and_select(
            embeddings=embeddings,
            boundary_indices=boundary_indices,
            timestamps_ms=timestamps_ms,
            query=query,
        )

        # Build Frame objects
        selected_frames = self._build_frame_objects(
            video_path,
            selected_indices,
            timestamps_ms,
            boundary_indices,
            embeddings,
        )

        return SelectionResult(
            selected_indices=selected_indices,
            selected_frames=selected_frames,
            embeddings=embeddings,
            scene_boundaries=boundaries,
            selection_scores=scores,
        )

    def _extract_frames(
        self,
        video_path: Path,
        sample_fps: float,
    ) -> tuple[list[NDArray[np.uint8]], list[int], list[int], float]:
        """Extract frames from video at specified rate.
        
        Returns:
            Tuple of (frames, frame_indices, timestamps_ms, video_fps)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Could not open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if video_fps <= 0:
                video_fps = 30.0  # Default fallback

            frame_interval = max(1, int(video_fps / sample_fps))

            frames: list[NDArray[np.uint8]] = []
            frame_indices: list[int] = []
            timestamps_ms: list[int] = []

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frames.append(frame)
                    frame_indices.append(frame_count)
                    timestamp = int((frame_count / video_fps) * 1000)
                    timestamps_ms.append(timestamp)

                frame_count += 1

                if len(frames) >= settings.max_frames_per_video:
                    logger.warning(
                        "Frame limit reached",
                        limit=settings.max_frames_per_video,
                    )
                    break

            return frames, frame_indices, timestamps_ms, video_fps

        finally:
            cap.release()

    def _score_and_select(
        self,
        embeddings: list[FrameEmbedding],
        boundary_indices: set[int],
        timestamps_ms: list[int],
        query: str | None,
    ) -> tuple[list[int], dict[int, float]]:
        """Score frames and select top candidates.
        
        Returns:
            Tuple of (selected_indices, score_dict)
        """
        n_frames = len(embeddings)
        if n_frames == 0:
            return [], {}

        scores: dict[int, float] = {}

        # Query relevance scores (if query provided)
        query_scores: dict[int, float] = {}
        if query:
            relevance = self.clip_scorer.score_relevance(embeddings, query)
            # Normalize to 0-1
            if relevance:
                max_score = max(s for _, s in relevance)
                min_score = min(s for _, s in relevance)
                score_range = max_score - min_score + 1e-7
                for idx, score in relevance:
                    query_scores[idx] = (score - min_score) / score_range

        # Temporal scores (favor even distribution)
        total_duration = timestamps_ms[-1] - timestamps_ms[0] + 1
        temporal_scores: dict[int, float] = {}

        for i, emb in enumerate(embeddings):
            # Temporal position (0 to 1)
            position = timestamps_ms[i] / total_duration if total_duration > 0 else 0
            # Score favors diversity in position
            temporal_scores[emb.frame_index] = 1.0 - abs(position - 0.5) * 0.5

        # Scene boundary bonus
        boundary_scores: dict[int, float] = {
            idx: 1.0 if idx in boundary_indices else 0.0
            for emb in embeddings
            for idx in [emb.frame_index]
        }

        # Compute combined scores
        for emb in embeddings:
            idx = emb.frame_index

            temporal = temporal_scores.get(idx, 0.0)
            boundary = boundary_scores.get(idx, 0.0)
            query_rel = query_scores.get(idx, 0.5) if query else 0.5

            score = (
                self.config.temporal_weight * temporal
                + self.config.scene_boundary_weight * boundary
                + self.config.query_weight * query_rel
            )

            scores[idx] = score

        # Select frames
        target = min(self.config.target_frames, n_frames)

        # Always include first and last if configured
        selected: list[int] = []
        if self.config.ensure_start_end and n_frames >= 2:
            selected.append(embeddings[0].frame_index)
            selected.append(embeddings[-1].frame_index)
            target = max(0, target - 2)

        # Add scene boundaries
        boundary_list = sorted(boundary_indices)
        for bidx in boundary_list:
            if len(selected) >= self.config.max_frames:
                break
            if bidx not in selected:
                selected.append(bidx)
                target -= 1

        # Fill remaining with diversity-based selection
        if target > 0:
            # Filter out already selected
            remaining_embeddings = [
                e for e in embeddings if e.frame_index not in selected
            ]

            if remaining_embeddings:
                diverse_indices = self.clip_scorer.find_diverse_frames(
                    remaining_embeddings,
                    n_frames=target,
                )
                selected.extend(diverse_indices)

        # Sort by temporal order
        selected = sorted(set(selected))

        # Enforce temporal gap
        if self.config.min_temporal_gap_ms > 0:
            selected = self._enforce_temporal_gap(
                selected, timestamps_ms, embeddings
            )

        logger.info(
            "Frame selection complete",
            total_frames=n_frames,
            selected_frames=len(selected),
            scene_boundaries=len(boundary_indices),
        )

        return selected, scores

    def _enforce_temporal_gap(
        self,
        selected: list[int],
        timestamps_ms: list[int],
        embeddings: list[FrameEmbedding],
    ) -> list[int]:
        """Remove frames that are too close together temporally."""
        if len(selected) <= 1:
            return selected

        # Build index -> timestamp map
        idx_to_ts = {e.frame_index: e.timestamp_ms for e in embeddings}

        filtered: list[int] = [selected[0]]
        last_ts = idx_to_ts.get(selected[0], 0)

        for idx in selected[1:]:
            current_ts = idx_to_ts.get(idx, 0)
            if current_ts - last_ts >= self.config.min_temporal_gap_ms:
                filtered.append(idx)
                last_ts = current_ts

        return filtered

    def _build_frame_objects(
        self,
        video_path: Path,
        selected_indices: list[int],
        timestamps_ms: list[int],
        boundary_indices: set[int],
        embeddings: list[FrameEmbedding],
    ) -> list[Frame]:
        """Build Frame objects for selected frames."""
        from uuid import uuid4

        # Map frame index to embedding
        idx_to_embedding = {e.frame_index: e for e in embeddings}

        frames: list[Frame] = []
        video_id = uuid4()  # Would come from job in production

        for idx in selected_indices:
            emb = idx_to_embedding.get(idx)

            frame_type = FrameType.REGULAR
            if idx in boundary_indices:
                frame_type = FrameType.SCENE_BOUNDARY

            ts = emb.timestamp_ms if emb else 0

            frames.append(
                Frame(
                    video_id=video_id,
                    index=idx,
                    timestamp_ms=ts,
                    frame_type=frame_type,
                    path=str(video_path),  # Would be actual frame path in production
                    embedding=emb.embedding.tolist() if emb else None,
                )
            )

        return frames

    async def select_from_frames(
        self,
        frames: Sequence[NDArray[np.uint8]],
        frame_indices: list[int] | None = None,
        timestamps_ms: list[int] | None = None,
        query: str | None = None,
    ) -> SelectionResult:
        """Select key frames from pre-extracted frames.
        
        Useful when frames are already extracted or coming from a stream.
        """
        self._ensure_initialized()

        n_frames = len(frames)
        if n_frames == 0:
            return SelectionResult(
                selected_indices=[],
                selected_frames=[],
                embeddings=[],
                scene_boundaries=[],
                selection_scores={},
            )

        # Default indices and timestamps
        if frame_indices is None:
            frame_indices = list(range(n_frames))
        if timestamps_ms is None:
            # Assume 2 FPS
            timestamps_ms = [i * 500 for i in range(n_frames)]

        # Detect shot boundaries
        boundaries = self.shot_detector.detect_from_frames(list(frames))
        boundary_indices = {b.frame_index for b in boundaries}

        # Compute embeddings
        embeddings = self.clip_scorer.embed_frames(
            list(frames),
            frame_indices=frame_indices,
            timestamps_ms=timestamps_ms,
        )

        # Score and select
        selected_indices, scores = self._score_and_select(
            embeddings=embeddings,
            boundary_indices=boundary_indices,
            timestamps_ms=timestamps_ms,
            query=query,
        )

        # Build minimal Frame objects
        from uuid import uuid4

        video_id = uuid4()
        selected_frames = [
            Frame(
                video_id=video_id,
                index=idx,
                timestamp_ms=timestamps_ms[frame_indices.index(idx)] if idx in frame_indices else 0,
                frame_type=FrameType.SCENE_BOUNDARY if idx in boundary_indices else FrameType.REGULAR,
                path="",
            )
            for idx in selected_indices
        ]

        return SelectionResult(
            selected_indices=selected_indices,
            selected_frames=selected_frames,
            embeddings=embeddings,
            scene_boundaries=boundaries,
            selection_scores=scores,
        )

    async def cleanup(self) -> None:
        """Release resources."""
        if self._initialized:
            await self.clip_scorer.unload_model()
            self._initialized = False
