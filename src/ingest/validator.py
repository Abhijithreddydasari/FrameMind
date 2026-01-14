"""Video format validation and metadata extraction."""
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.core.config import settings
from src.core.exceptions import UnsupportedFormatError, VideoTooLargeError, VideoValidationError
from src.core.logging import get_logger
from src.core.models import VideoMetadata

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of video validation."""

    valid: bool
    metadata: VideoMetadata | None = None
    error: str | None = None


class VideoValidator:
    """Validates video files for processing.
    
    Checks:
    - File exists and is readable
    - Format is supported
    - Size is within limits
    - Video is decodable
    
    Example:
        validator = VideoValidator()
        result = validator.validate("video.mp4")
        
        if result.valid:
            print(f"Duration: {result.metadata.duration_ms}ms")
    """

    def __init__(self) -> None:
        self.allowed_formats = settings.allowed_video_formats
        self.max_size_bytes = settings.max_video_size_mb * 1024 * 1024

    def validate(self, video_path: str | Path) -> ValidationResult:
        """Validate a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            ValidationResult with metadata if valid
        """
        path = Path(video_path)

        # Check file exists
        if not path.exists():
            return ValidationResult(
                valid=False,
                error=f"File not found: {path}",
            )

        # Check format
        extension = path.suffix.lower().lstrip(".")
        if extension not in self.allowed_formats:
            return ValidationResult(
                valid=False,
                error=f"Unsupported format: {extension}",
            )

        # Check size
        size_bytes = path.stat().st_size
        if size_bytes > self.max_size_bytes:
            return ValidationResult(
                valid=False,
                error=f"File too large: {size_bytes} bytes (max: {self.max_size_bytes})",
            )

        # Try to open and read metadata
        try:
            metadata = self._extract_metadata(path, size_bytes)
            
            logger.info(
                "Video validated",
                path=str(path),
                duration_ms=metadata.duration_ms,
                resolution=f"{metadata.width}x{metadata.height}",
            )
            
            return ValidationResult(valid=True, metadata=metadata)

        except Exception as e:
            logger.error("Video validation failed", path=str(path), error=str(e))
            return ValidationResult(
                valid=False,
                error=f"Could not read video: {e}",
            )

    def _extract_metadata(self, path: Path, size_bytes: int) -> VideoMetadata:
        """Extract video metadata using OpenCV."""
        cap = cv2.VideoCapture(str(path))
        
        if not cap.isOpened():
            raise VideoValidationError(f"Could not open video: {path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            codec_code = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Decode codec
            codec = "".join([chr((codec_code >> 8 * i) & 0xFF) for i in range(4)])

            # Calculate duration
            duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0

            return VideoMetadata(
                filename=path.name,
                format=path.suffix.lower().lstrip("."),
                duration_ms=duration_ms,
                width=width,
                height=height,
                fps=fps,
                codec=codec,
                size_bytes=size_bytes,
                frame_count=frame_count,
            )

        finally:
            cap.release()

    def validate_or_raise(self, video_path: str | Path) -> VideoMetadata:
        """Validate video and raise exception if invalid.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoMetadata if valid
            
        Raises:
            VideoValidationError: If validation fails
            UnsupportedFormatError: If format not supported
            VideoTooLargeError: If file too large
        """
        path = Path(video_path)

        if not path.exists():
            raise VideoValidationError(f"File not found: {path}")

        extension = path.suffix.lower().lstrip(".")
        if extension not in self.allowed_formats:
            raise UnsupportedFormatError(
                f"Format '{extension}' not supported",
                details={"allowed": self.allowed_formats},
            )

        size_bytes = path.stat().st_size
        if size_bytes > self.max_size_bytes:
            raise VideoTooLargeError(
                f"File exceeds {settings.max_video_size_mb}MB limit",
                details={"size_bytes": size_bytes, "max_bytes": self.max_size_bytes},
            )

        return self._extract_metadata(path, size_bytes)
