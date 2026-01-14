"""Shot/scene boundary detection using histogram analysis.

This module implements a production-grade shot detector that identifies
scene changes in videos using color histogram differences with adaptive
thresholding. This reduces redundant VLM calls by avoiding analysis of
frames within the same visual scene.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from src.core.config import settings
from src.core.exceptions import FrameExtractionError
from src.core.logging import get_logger
from src.core.models import SceneBoundary

logger = get_logger(__name__)


@dataclass
class HistogramConfig:
    """Configuration for histogram-based shot detection."""

    bins: int = 64
    threshold: float = 0.3  # Chi-squared distance threshold
    min_scene_length: int = 10  # Minimum frames between boundaries
    use_hsv: bool = True  # Use HSV color space
    adaptive_threshold: bool = True  # Adapt threshold based on video statistics


class ShotDetector:
    """Detects scene boundaries in video using histogram analysis.
    
    Uses chi-squared histogram comparison with adaptive thresholding
    to identify significant visual changes between frames.
    
    Example:
        detector = ShotDetector()
        boundaries = detector.detect_from_video("video.mp4")
        # Or from pre-extracted frames:
        boundaries = detector.detect_from_frames(frame_paths)
    """

    def __init__(self, config: HistogramConfig | None = None) -> None:
        self.config = config or HistogramConfig(
            threshold=settings.shot_threshold,
            min_scene_length=settings.min_scene_length,
        )
        self._prev_histogram: NDArray[np.float32] | None = None

    def compute_histogram(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Compute normalized color histogram for a frame.
        
        Args:
            frame: BGR image array from OpenCV
            
        Returns:
            Flattened, normalized histogram array
        """
        if self.config.use_hsv:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Compute histogram for each channel
        histograms = []
        for channel in range(3):
            hist = cv2.calcHist(
                [frame],
                [channel],
                None,
                [self.config.bins],
                [0, 256],
            )
            histograms.append(hist.flatten())

        # Concatenate and normalize
        combined = np.concatenate(histograms)
        normalized = combined / (combined.sum() + 1e-7)

        return normalized.astype(np.float32)

    def compare_histograms(
        self,
        hist1: NDArray[np.float32],
        hist2: NDArray[np.float32],
    ) -> float:
        """Compare two histograms using chi-squared distance.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
            
        Returns:
            Chi-squared distance (0 = identical, higher = more different)
        """
        # Chi-squared distance with epsilon for numerical stability
        epsilon = 1e-10
        diff = hist1 - hist2
        sum_hist = hist1 + hist2 + epsilon
        chi_squared = np.sum((diff ** 2) / sum_hist) / 2

        return float(chi_squared)

    def detect_from_frames(
        self,
        frames: Sequence[NDArray[np.uint8]],
    ) -> list[SceneBoundary]:
        """Detect shot boundaries from a sequence of frames.
        
        Args:
            frames: List of BGR frame arrays
            
        Returns:
            List of detected scene boundaries with confidence scores
        """
        if len(frames) < 2:
            return []

        boundaries: list[SceneBoundary] = []
        distances: list[float] = []
        histograms: list[NDArray[np.float32]] = []

        # Compute all histograms
        for frame in frames:
            histograms.append(self.compute_histogram(frame))

        # Compute pairwise distances
        for i in range(1, len(histograms)):
            distance = self.compare_histograms(histograms[i - 1], histograms[i])
            distances.append(distance)

        # Adaptive thresholding
        if self.config.adaptive_threshold and distances:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            threshold = max(
                self.config.threshold,
                mean_dist + 1.5 * std_dist,
            )
        else:
            threshold = self.config.threshold

        # Detect boundaries with minimum scene length constraint
        last_boundary = -self.config.min_scene_length

        for i, distance in enumerate(distances):
            frame_idx = i + 1  # Boundary is at frame i+1

            if distance > threshold and (frame_idx - last_boundary) >= self.config.min_scene_length:
                # Normalize confidence to 0-1 range
                confidence = min(1.0, distance / (threshold * 3))

                boundaries.append(
                    SceneBoundary(
                        frame_index=frame_idx,
                        confidence=confidence,
                    )
                )
                last_boundary = frame_idx

        logger.info(
            "Shot detection complete",
            total_frames=len(frames),
            boundaries_found=len(boundaries),
            threshold=threshold,
        )

        return boundaries

    def detect_from_video(
        self,
        video_path: str | Path,
        sample_fps: float | None = None,
    ) -> list[SceneBoundary]:
        """Detect shot boundaries directly from a video file.
        
        Args:
            video_path: Path to video file
            sample_fps: Sample rate (default: settings.frame_extraction_fps)
            
        Returns:
            List of detected scene boundaries
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FrameExtractionError(f"Video not found: {video_path}")

        sample_fps = sample_fps or settings.frame_extraction_fps

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Could not open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(video_fps / sample_fps))

            frames: list[NDArray[np.uint8]] = []
            frame_indices: list[int] = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frames.append(frame)
                    frame_indices.append(frame_count)

                frame_count += 1

                # Safety limit
                if len(frames) >= settings.max_frames_per_video:
                    logger.warning(
                        "Frame limit reached",
                        max_frames=settings.max_frames_per_video,
                    )
                    break

        finally:
            cap.release()

        # Detect boundaries
        boundaries = self.detect_from_frames(frames)

        # Map back to original frame indices
        for boundary in boundaries:
            if boundary.frame_index < len(frame_indices):
                boundary.frame_index = frame_indices[boundary.frame_index]

        return boundaries

    def detect_streaming(
        self,
        frame: NDArray[np.uint8],
        frame_index: int,
    ) -> SceneBoundary | None:
        """Detect shot boundary in streaming mode (frame by frame).
        
        Useful for real-time processing or memory-constrained scenarios.
        
        Args:
            frame: Current BGR frame
            frame_index: Index of current frame
            
        Returns:
            SceneBoundary if a boundary is detected, None otherwise
        """
        current_hist = self.compute_histogram(frame)

        if self._prev_histogram is None:
            self._prev_histogram = current_hist
            return None

        distance = self.compare_histograms(self._prev_histogram, current_hist)
        self._prev_histogram = current_hist

        if distance > self.config.threshold:
            confidence = min(1.0, distance / (self.config.threshold * 3))
            return SceneBoundary(
                frame_index=frame_index,
                confidence=confidence,
            )

        return None

    def reset_streaming_state(self) -> None:
        """Reset streaming detection state."""
        self._prev_histogram = None
