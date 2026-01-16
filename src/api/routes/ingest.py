"""Video ingestion endpoints."""
import shutil
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.deps import MetadataStoreDep, RedisCacheDep, SettingsDep
from src.core.exceptions import UnsupportedFormatError, VideoTooLargeError
from src.core.logging import get_logger
from src.core.models import JobStatus, VideoUploadResponse

logger = get_logger(__name__)

router = APIRouter()


class IngestStatusResponse(BaseModel):
    """Status of an ingestion job."""

    job_id: UUID
    status: JobStatus
    progress: float
    message: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime


@router.post("/upload", response_model=VideoUploadResponse, status_code=202)
async def upload_video(
    settings: SettingsDep,
    cache: RedisCacheDep,
    metadata_store: MetadataStoreDep,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> VideoUploadResponse:
    """Upload a video for processing.
    
    The video will be validated, stored, and queued for async processing.
    Returns immediately with a job_id to track progress.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    extension = Path(file.filename).suffix.lower().lstrip(".")
    if extension not in settings.allowed_video_formats:
        raise UnsupportedFormatError(
            f"Format '{extension}' not supported",
            details={"allowed": settings.allowed_video_formats},
        )

    # Check file size (if content-length header is available)
    if file.size and file.size > settings.max_video_size_mb * 1024 * 1024:
        raise VideoTooLargeError(
            f"Video exceeds {settings.max_video_size_mb}MB limit",
            details={"size_bytes": file.size, "max_bytes": settings.max_video_size_mb * 1024 * 1024},
        )

    # Generate job ID and storage path
    job_id = uuid4()
    video_dir = settings.storage_path / "videos" / str(job_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"source.{extension}"

    # Stream file to disk
    try:
        async with aiofiles.open(video_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await out_file.write(chunk)
    except Exception as e:
        logger.error("Failed to save video", job_id=str(job_id), error=str(e))
        shutil.rmtree(video_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Failed to save video")

    # Verify final file size
    actual_size = video_path.stat().st_size
    if actual_size > settings.max_video_size_mb * 1024 * 1024:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise VideoTooLargeError(
            f"Video exceeds {settings.max_video_size_mb}MB limit",
            details={"size_bytes": actual_size},
        )

    # Store job state in Redis
    job_data = {
        "status": JobStatus.PENDING.value,
        "video_path": str(video_path),
        "filename": file.filename,
        "progress": 0.0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    await cache.set_job(job_id, job_data)
    await metadata_store.create_job(job_id, job_data)

    # Queue for async processing
    background_tasks.add_task(enqueue_processing, job_id, cache)

    logger.info("Video uploaded", job_id=str(job_id), filename=file.filename)

    return VideoUploadResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Video uploaded successfully. Processing will begin shortly.",
        created_at=datetime.utcnow(),
    )


async def enqueue_processing(job_id: UUID, cache: RedisCacheDep) -> None:
    """Enqueue video for async processing via ARQ."""
    try:
        # In production, this would enqueue to ARQ
        # For now, update status to show it's queued
        await cache.update_job_status(job_id, JobStatus.PROCESSING)
        logger.info("Job enqueued for processing", job_id=str(job_id))
    except Exception as e:
        logger.error("Failed to enqueue job", job_id=str(job_id), error=str(e))
        await cache.update_job_status(job_id, JobStatus.FAILED, error=str(e))


@router.get("/status/{job_id}", response_model=IngestStatusResponse)
async def get_ingest_status(
    job_id: UUID,
    cache: RedisCacheDep,
) -> IngestStatusResponse:
    """Get the status of a video ingestion job."""
    job_data = await cache.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    return IngestStatusResponse(
        job_id=job_id,
        status=JobStatus(job_data.get("status", "pending")),
        progress=float(job_data.get("progress", 0.0)),
        message=job_data.get("message"),
        error=job_data.get("error"),
        created_at=datetime.fromisoformat(job_data["created_at"]),
        updated_at=datetime.fromisoformat(job_data["updated_at"]),
    )


@router.delete("/{job_id}", status_code=204)
async def cancel_job(
    job_id: UUID,
    settings: SettingsDep,
    cache: RedisCacheDep,
) -> None:
    """Cancel a pending or processing job."""
    job_data = await cache.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    current_status = JobStatus(job_data.get("status", "pending"))

    if current_status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in {current_status.value} state",
        )

    # Update status
    await cache.update_job_status(job_id, JobStatus.CANCELLED)

    # Clean up files
    video_dir = settings.storage_path / "videos" / str(job_id)
    if video_dir.exists():
        shutil.rmtree(video_dir, ignore_errors=True)

    logger.info("Job cancelled", job_id=str(job_id))
