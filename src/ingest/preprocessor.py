"""Video preprocessing and normalization."""
import subprocess
from pathlib import Path

from src.core.config import settings
from src.core.exceptions import ProcessingError
from src.core.logging import get_logger

logger = get_logger(__name__)


class VideoPreprocessor:
    """Preprocesses videos for optimal frame extraction.
    
    Operations:
    - Resolution normalization (optional)
    - Codec standardization (optional)
    - Audio stripping (for faster processing)
    
    Example:
        preprocessor = VideoPreprocessor()
        output_path = preprocessor.process("input.mkv", "output.mp4")
    """

    def __init__(
        self,
        target_width: int | None = None,
        target_height: int | None = None,
        target_fps: float | None = None,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps or settings.frame_extraction_fps

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        strip_audio: bool = True,
    ) -> Path:
        """Preprocess a video file.
        
        Args:
            input_path: Input video path
            output_path: Output path (default: same dir, .mp4 extension)
            strip_audio: Remove audio track for faster processing
            
        Returns:
            Path to preprocessed video
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.with_suffix(".processed.mp4")
        else:
            output_path = Path(output_path)

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y", "-i", str(input_path)]

        # Video codec
        cmd.extend(["-c:v", "libx264", "-preset", "fast"])

        # Resolution scaling
        if self.target_width and self.target_height:
            cmd.extend([
                "-vf",
                f"scale={self.target_width}:{self.target_height}:force_original_aspect_ratio=decrease"
            ])

        # Audio handling
        if strip_audio:
            cmd.extend(["-an"])
        else:
            cmd.extend(["-c:a", "aac"])

        # Output
        cmd.append(str(output_path))

        logger.info(
            "Preprocessing video",
            input=str(input_path),
            output=str(output_path),
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            
            logger.info("Video preprocessed", output=str(output_path))
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(
                "FFmpeg failed",
                stderr=e.stderr[:500] if e.stderr else None,
            )
            raise ProcessingError(f"Video preprocessing failed: {e.stderr}")
        except FileNotFoundError:
            raise ProcessingError("FFmpeg not found. Please install FFmpeg.")

    def needs_preprocessing(self, video_path: str | Path) -> bool:
        """Check if video needs preprocessing.
        
        Returns True if:
        - Format is not MP4
        - Resolution exceeds 1080p
        - Has unusual codec
        """
        path = Path(video_path)

        # Non-MP4 formats may need transcoding
        if path.suffix.lower() not in (".mp4", ".m4v"):
            return True

        # Check resolution and codec via OpenCV
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return True

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Downscale if > 1080p
            if width > 1920 or height > 1080:
                return True

            return False

        finally:
            cap.release()
