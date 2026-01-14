"""Context-aware prompt construction for VLM queries."""
from dataclasses import dataclass

from src.core.models import Frame


@dataclass
class PromptContext:
    """Context for building VLM prompts."""

    query: str
    video_duration_ms: int | None = None
    total_frames: int | None = None
    frames: list[Frame] | None = None


class PromptBuilder:
    """Builds context-aware prompts for VLM queries.
    
    Constructs prompts that provide temporal context and
    guide the VLM to produce structured responses.
    
    Example:
        builder = PromptBuilder()
        prompt = builder.build_analysis_prompt(
            query="What is happening in the video?",
            frames=selected_frames,
        )
    """

    def __init__(self) -> None:
        self.system_context = """You are an expert video analyst. You will be shown key frames from a video 
and asked to answer questions about the video content.

Guidelines:
- Analyze the frames in temporal order (earlier frames come first)
- Consider the timestamps to understand the video timeline
- Be specific about what you observe
- If uncertain, indicate your confidence level
- Focus on answering the specific question asked"""

    def build_analysis_prompt(
        self,
        query: str,
        frames: list[Frame] | None = None,
        include_timestamps: bool = True,
    ) -> str:
        """Build prompt for video analysis.
        
        Args:
            query: User's question about the video
            frames: Selected frames with metadata
            include_timestamps: Whether to include temporal context
            
        Returns:
            Formatted prompt string
        """
        parts = [self.system_context, ""]

        # Add temporal context
        if frames and include_timestamps:
            parts.append("Frame Information:")
            for i, frame in enumerate(frames):
                timestamp = self._format_timestamp(frame.timestamp_ms)
                frame_type = frame.frame_type.value
                parts.append(f"  Frame {i+1}: {timestamp} ({frame_type})")
            parts.append("")

        # Add the query
        parts.append(f"Question: {query}")
        parts.append("")
        parts.append("Please analyze the frames and answer the question:")

        return "\n".join(parts)

    def build_summary_prompt(
        self,
        frames: list[Frame] | None = None,
        video_duration_ms: int | None = None,
    ) -> str:
        """Build prompt for video summarization.
        
        Args:
            frames: Selected frames with metadata
            video_duration_ms: Total video duration
            
        Returns:
            Formatted prompt string
        """
        parts = [
            self.system_context,
            "",
            "Task: Provide a comprehensive summary of this video.",
            "",
        ]

        if video_duration_ms:
            duration = self._format_timestamp(video_duration_ms)
            parts.append(f"Video Duration: {duration}")

        if frames:
            parts.append(f"Key Frames Shown: {len(frames)}")

            # Add frame timeline
            parts.append("")
            parts.append("Timeline:")
            for i, frame in enumerate(frames):
                timestamp = self._format_timestamp(frame.timestamp_ms)
                parts.append(f"  {timestamp} - Frame {i+1}")

        parts.extend([
            "",
            "Please provide:",
            "1. A brief overview of the video content",
            "2. Key events or moments in chronological order",
            "3. Notable objects, people, or activities",
            "4. The overall theme or purpose of the video",
        ])

        return "\n".join(parts)

    def build_comparison_prompt(
        self,
        query: str,
        frame_groups: list[list[Frame]],
    ) -> str:
        """Build prompt for comparing different parts of the video.
        
        Args:
            query: Comparison question
            frame_groups: Groups of frames to compare
            
        Returns:
            Formatted prompt string
        """
        parts = [
            self.system_context,
            "",
            f"Question: {query}",
            "",
        ]

        for group_idx, frames in enumerate(frame_groups):
            parts.append(f"Segment {group_idx + 1}:")
            for frame in frames:
                timestamp = self._format_timestamp(frame.timestamp_ms)
                parts.append(f"  {timestamp}")
            parts.append("")

        parts.append("Please compare these segments and answer the question:")

        return "\n".join(parts)

    def _format_timestamp(self, ms: int) -> str:
        """Format milliseconds as HH:MM:SS.mmm."""
        seconds = ms // 1000
        millis = ms % 1000
        minutes = seconds // 60
        seconds = seconds % 60
        hours = minutes // 60
        minutes = minutes % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
        else:
            return f"{minutes:02d}:{seconds:02d}.{millis:03d}"
