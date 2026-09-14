"""Domain models for FrameMind."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class JobStatus(StrEnum):
    """Video processing job status."""

    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FrameType(StrEnum):
    """Type of extracted frame."""

    REGULAR = "regular"
    SCENE_BOUNDARY = "scene_boundary"
    KEYFRAME = "keyframe"


# ============ Request/Response Models ============


class VideoUploadRequest(BaseModel):
    """Request to upload a video for processing."""

    filename: str
    content_type: str = "video/mp4"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoUploadResponse(BaseModel):
    """Response after video upload."""

    job_id: UUID
    status: JobStatus
    message: str
    created_at: datetime


class QueryRequest(BaseModel):
    """Semantic query against a processed video."""

    query: str = Field(..., min_length=1, max_length=2000)
    max_frames: int = Field(default=10, ge=1, le=50)
    include_timestamps: bool = True
    use_cache: bool = True
    analysis_backend: Literal["default", "none", "nvila_autogaze"] = "default"


class InspectRequest(QueryRequest):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    crop: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "InspectRequest":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.end_ms - self.start_ms > 60_000:
            raise ValueError("Inspection intervals cannot exceed 60 seconds")
        if self.crop:
            x1, y1, x2, y2 = self.crop
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("crop must be normalized x1,y1,x2,y2 coordinates")
        return self


class QueryResponse(BaseModel):
    """Response to a semantic query."""

    job_id: UUID
    query: str
    answer: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    frames_analyzed: int
    processing_time_ms: int
    sources: list["FrameSource"]
    backend_used: str = "none"
    analysis_status: Literal["complete", "retrieval_only", "failed", "insufficient_evidence"] = (
        "retrieval_only"
    )
    index_generation: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class FrameSource(BaseModel):
    """Source frame used in query response."""

    frame_index: int | None = None
    timestamp_ms: int
    relevance_score: float
    description: str | None = None
    evidence_id: str = ""
    source_kind: Literal["frame", "clip"] = "frame"
    start_ms: int | None = None
    end_ms: int | None = None
    frame_timestamps_ms: list[int] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    """Job status response."""

    job_id: UUID
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    message: str | None = None
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    result: dict[str, Any] | None = None


# ============ Internal Domain Models ============


class VideoMetadata(BaseModel):
    """Video file metadata."""

    filename: str
    format: str
    duration_ms: int
    width: int
    height: int
    fps: float
    codec: str
    size_bytes: int
    frame_count: int


class Frame(BaseModel):
    """Extracted video frame."""

    id: UUID = Field(default_factory=uuid4)
    video_id: UUID
    index: int
    timestamp_ms: int
    frame_type: FrameType = FrameType.REGULAR
    path: str
    embedding: list[float] | None = None
    scene_score: float | None = None  # shot boundary confidence


class VideoJob(BaseModel):
    """Video processing job."""

    id: UUID = Field(default_factory=uuid4)
    status: JobStatus = JobStatus.PENDING
    video_path: str | None = None
    metadata: VideoMetadata | None = None
    frames: list[Frame] = Field(default_factory=list)
    keyframe_indices: list[int] = Field(default_factory=list)
    progress: float = 0.0
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def update_status(self, status: JobStatus, message: str | None = None) -> None:
        """Update job status and timestamp."""
        self.status = status
        self.updated_at = datetime.utcnow()
        if status == JobStatus.COMPLETE:
            self.completed_at = datetime.utcnow()
        if status == JobStatus.FAILED and message:
            self.error = message


class SceneBoundary(BaseModel):
    """Detected scene boundary."""

    frame_index: int
    confidence: float
    prev_histogram: list[float] | None = None


class FrameCluster(BaseModel):
    """Cluster of similar frames."""

    centroid_index: int
    frame_indices: list[int]
    average_embedding: list[float] | None = None


# Resolve forward references
QueryResponse.model_rebuild()
