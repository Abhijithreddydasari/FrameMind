"""Domain exceptions for FrameMind."""
from typing import Any


class FrameMindError(Exception):
    """Base exception for all FrameMind errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


# Ingestion errors
class IngestionError(FrameMindError):
    """Error during video ingestion."""

    pass


class VideoValidationError(IngestionError):
    """Invalid video format or content."""

    pass


class VideoTooLargeError(IngestionError):
    """Video exceeds size limit."""

    pass


class UnsupportedFormatError(IngestionError):
    """Unsupported video format."""

    pass


# Processing errors
class ProcessingError(FrameMindError):
    """Error during video processing pipeline."""

    pass


class FrameExtractionError(ProcessingError):
    """Failed to extract frames from video."""

    pass


class MLModelError(ProcessingError):
    """Error in ML model inference."""

    pass


# VLM errors
class VLMError(FrameMindError):
    """Error communicating with VLM."""

    pass


class VLMRateLimitError(VLMError):
    """VLM rate limit exceeded."""

    pass


class VLMTimeoutError(VLMError):
    """VLM request timed out."""

    pass


class VLMResponseError(VLMError):
    """Invalid VLM response."""

    pass


# Storage errors
class StorageError(FrameMindError):
    """Error in storage operations."""

    pass


class VideoNotFoundError(StorageError):
    """Video not found in storage."""

    pass


class JobNotFoundError(StorageError):
    """Job not found."""

    pass


# Cache errors
class CacheError(FrameMindError):
    """Error in cache operations."""

    pass


class RateLimitExceededError(CacheError):
    """Rate limit exceeded."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(
            message="Rate limit exceeded",
            details={"retry_after": retry_after},
        )
        self.retry_after = retry_after
