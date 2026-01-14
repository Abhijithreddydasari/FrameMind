"""Frame extraction from video files."""
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from numpy.typing import NDArray

from src.core.config import settings
from src.core.exceptions import FrameExtractionError
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedFrame:
    """Extracted frame with metadata."""

    frame: NDArray[np.uint8]
    index: int
    timestamp_ms: int


class FrameExtractor:
    """Extracts frames from video at configurable rate.
    
    Supports both batch and streaming extraction modes.
    
    Example:
        extractor = FrameExtractor(fps=2.0)
        
        # Batch extraction
        frames = extractor.extract_all("video.mp4")
        
        # Streaming extraction
        for frame in extractor.extract_stream("video.mp4"):
            process(frame)
    """

    def __init__(
        self,
        fps: float | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.fps = fps or settings.frame_extraction_fps
        self.max_frames = max_frames or settings.max_frames_per_video

    def extract_all(
        self,
        video_path: str | Path,
    ) -> list[ExtractedFrame]:
        """Extract all frames at configured FPS.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of extracted frames with metadata
        """
        return list(self.extract_stream(video_path))

    def extract_stream(
        self,
        video_path: str | Path,
    ) -> Generator[ExtractedFrame, None, None]:
        """Stream frames from video.
        
        Memory-efficient generator for large videos.
        
        Args:
            video_path: Path to video file
            
        Yields:
            ExtractedFrame objects
        """
        path = Path(video_path)
        
        if not path.exists():
            raise FrameExtractionError(f"Video not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Could not open video: {path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30.0  # Default fallback

            # Calculate frame interval
            frame_interval = max(1, int(video_fps / self.fps))

            frame_count = 0
            extracted_count = 0

            logger.info(
                "Starting frame extraction",
                video=str(path),
                target_fps=self.fps,
                interval=frame_interval,
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    timestamp_ms = int((frame_count / video_fps) * 1000)

                    yield ExtractedFrame(
                        frame=frame,
                        index=frame_count,
                        timestamp_ms=timestamp_ms,
                    )

                    extracted_count += 1
                    if extracted_count >= self.max_frames:
                        logger.warning(
                            "Frame limit reached",
                            limit=self.max_frames,
                        )
                        break

                frame_count += 1

            logger.info(
                "Frame extraction complete",
                total_frames=frame_count,
                extracted=extracted_count,
            )

        finally:
            cap.release()

    def extract_at_timestamps(
        self,
        video_path: str | Path,
        timestamps_ms: list[int],
    ) -> list[ExtractedFrame]:
        """Extract frames at specific timestamps.
        
        Args:
            video_path: Path to video file
            timestamps_ms: List of timestamps in milliseconds
            
        Returns:
            List of extracted frames
        """
        path = Path(video_path)

        if not path.exists():
            raise FrameExtractionError(f"Video not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FrameExtractionError(f"Could not open video: {path}")

        frames: list[ExtractedFrame] = []

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0:
                video_fps = 30.0

            for timestamp_ms in sorted(timestamps_ms):
                # Seek to timestamp
                frame_index = int((timestamp_ms / 1000) * video_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

                ret, frame = cap.read()
                if ret:
                    frames.append(
                        ExtractedFrame(
                            frame=frame,
                            index=frame_index,
                            timestamp_ms=timestamp_ms,
                        )
                    )

            return frames

        finally:
            cap.release()

    def save_frames(
        self,
        frames: list[ExtractedFrame],
        output_dir: str | Path,
        format: str = "jpg",
        quality: int = 95,
    ) -> list[Path]:
        """Save extracted frames to disk.
        
        Args:
            frames: List of extracted frames
            output_dir: Output directory
            format: Image format (jpg, png)
            quality: JPEG quality (1-100)
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []

        for frame_data in frames:
            filename = f"frame_{frame_data.index:06d}.{format}"
            filepath = output_dir / filename

            if format.lower() == "jpg":
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            else:
                params = []

            cv2.imwrite(str(filepath), frame_data.frame, params)
            saved_paths.append(filepath)

        logger.info(
            "Frames saved",
            count=len(saved_paths),
            directory=str(output_dir),
        )

        return saved_paths
