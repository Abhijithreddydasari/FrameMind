"""Domain models for FrameMind."""
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Video processing job status."""

    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FrameType(str, Enum):
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


class QueryResponse(BaseModel):
    """Response to a semantic query."""

    job_id: UUID
    query: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    frames_analyzed: int
    processing_time_ms: int
    sources: list["FrameSource"]


class FrameSource(BaseModel):
    """Source frame used in query response."""

    frame_index: int
    timestamp_ms: int
    relevance_score: float
    description: str | None = None


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
