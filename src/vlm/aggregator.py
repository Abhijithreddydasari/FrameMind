"""Multi-frame response aggregation for VLM outputs."""
from dataclasses import dataclass, field

from src.core.logging import get_logger
from src.core.models import Frame, FrameSource

logger = get_logger(__name__)


@dataclass
class AggregatedResponse:
    """Aggregated response from multiple VLM calls."""

    answer: str
    confidence: float
    sources: list[FrameSource]
    raw_responses: list[str] = field(default_factory=list)


class ResponseAggregator:
    """Aggregates and synthesizes multi-frame VLM responses.
    
    When analyzing multiple frames separately, this class
    combines the responses into a coherent answer.
    
    Example:
        aggregator = ResponseAggregator()
        
        # Collect responses from individual frame analyses
        aggregator.add_response(frame1, "Person entering room")
        aggregator.add_response(frame2, "Person sitting at desk")
        
        # Get combined result
        result = aggregator.aggregate("What does the person do?")
    """

    def __init__(self) -> None:
        self.responses: list[tuple[Frame, str, float]] = []

    def add_response(
        self,
        frame: Frame,
        response: str,
        relevance_score: float = 1.0,
    ) -> None:
        """Add a frame analysis response.
        
        Args:
            frame: Analyzed frame
            response: VLM response text
            relevance_score: Frame relevance to query (0-1)
        """
        self.responses.append((frame, response, relevance_score))

    def aggregate(
        self,
        query: str,
        combine_mode: str = "synthesize",
    ) -> AggregatedResponse:
        """Aggregate collected responses.
        
        Args:
            query: Original query for context
            combine_mode: How to combine responses:
                - "synthesize": Create coherent narrative
                - "list": List individual observations
                - "vote": Find consensus answer
                
        Returns:
            Aggregated response with sources
        """
        if not self.responses:
            return AggregatedResponse(
                answer="No frames were analyzed.",
                confidence=0.0,
                sources=[],
            )

        # Sort by timestamp
        sorted_responses = sorted(
            self.responses,
            key=lambda x: x[0].timestamp_ms,
        )

        # Build sources
        sources = [
            FrameSource(
                frame_index=frame.index,
                timestamp_ms=frame.timestamp_ms,
                relevance_score=score,
                description=response[:200],  # Truncate for storage
            )
            for frame, response, score in sorted_responses
        ]

        # Calculate confidence as weighted average of relevance scores
        total_weight = sum(s.relevance_score for s in sources)
        confidence = total_weight / len(sources) if sources else 0.0

        # Combine responses based on mode
        if combine_mode == "synthesize":
            answer = self._synthesize(sorted_responses, query)
        elif combine_mode == "list":
            answer = self._list_observations(sorted_responses)
        else:
            answer = self._simple_concat(sorted_responses)

        return AggregatedResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            raw_responses=[r for _, r, _ in sorted_responses],
        )

    def _synthesize(
        self,
        responses: list[tuple[Frame, str, float]],
        query: str,
    ) -> str:
        """Synthesize responses into coherent narrative."""
        if len(responses) == 1:
            return responses[0][1]

        # Build temporal narrative
        parts = []

        for i, (frame, response, _) in enumerate(responses):
            timestamp = self._format_time(frame.timestamp_ms)

            if i == 0:
                parts.append(f"At the beginning ({timestamp}): {response}")
            elif i == len(responses) - 1:
                parts.append(f"Finally ({timestamp}): {response}")
            else:
                parts.append(f"At {timestamp}: {response}")

        return " ".join(parts)

    def _list_observations(
        self,
        responses: list[tuple[Frame, str, float]],
    ) -> str:
        """List observations with timestamps."""
        lines = []

        for frame, response, _ in responses:
            timestamp = self._format_time(frame.timestamp_ms)
            lines.append(f"[{timestamp}] {response}")

        return "\n".join(lines)

    def _simple_concat(
        self,
        responses: list[tuple[Frame, str, float]],
    ) -> str:
        """Simply concatenate responses."""
        return " ".join(r for _, r, _ in responses)

    def _format_time(self, ms: int) -> str:
        """Format milliseconds to MM:SS."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def clear(self) -> None:
        """Clear collected responses."""
        self.responses.clear()


class BatchAggregator:
    """Aggregates responses from batch VLM calls.
    
    When multiple frames are sent to VLM in a single call,
    this extracts and structures the combined response.
    """

    def extract_frame_references(
        self,
        response: str,
        frames: list[Frame],
    ) -> list[FrameSource]:
        """Extract frame references from VLM response.
        
        Attempts to match mentions of frames/timestamps to
        the actual frames for proper attribution.
        
        Args:
            response: VLM response text
            frames: Frames that were analyzed
            
        Returns:
            List of frame sources with relevance scores
        """
        sources: list[FrameSource] = []

        # Simple heuristic: all frames contribute equally
        # In production, use NLP to extract specific frame mentions
        for frame in frames:
            sources.append(
                FrameSource(
                    frame_index=frame.index,
                    timestamp_ms=frame.timestamp_ms,
                    relevance_score=1.0 / len(frames),
                    description=None,
                )
            )

        return sources

    def structure_response(
        self,
        response: str,
        frames: list[Frame],
    ) -> AggregatedResponse:
        """Structure a batch VLM response.
        
        Args:
            response: Raw VLM response
            frames: Analyzed frames
            
        Returns:
            Structured aggregated response
        """
        sources = self.extract_frame_references(response, frames)

        return AggregatedResponse(
            answer=response,
            confidence=0.8,  # Default confidence for batch responses
            sources=sources,
            raw_responses=[response],
        )
