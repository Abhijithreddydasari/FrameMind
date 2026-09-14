"""Stream uploads, preserve original recordings, and enqueue durable jobs."""

from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import aiofiles
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.api.deps import MetadataStoreDep, SettingsDep
from src.workers.orchestrator import JobOrchestrator

router = APIRouter()


@router.post("/upload", status_code=202)
async def upload_video(
    request: Request,
    settings: SettingsDep,
    store: MetadataStoreDep,
    file: Annotated[UploadFile, File(...)],
):
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in settings.allowed_video_formats:
        raise HTTPException(415, "Unsupported video format")
    limit = settings.max_video_size_mb * 1024**2
    if file.size and file.size > limit:
        raise HTTPException(413, "Video exceeds configured upload limit; use local ingestion")
    job_id = uuid4()
    directory = settings.storage_path / "videos" / str(job_id)
    directory.mkdir(parents=True)
    path = directory / f"source.{suffix}"
    size = 0
    try:
        async with aiofiles.open(path, "wb") as handle:
            while block := await file.read(1024**2):
                size += len(block)
                if size > limit:
                    raise HTTPException(413, "Video exceeds configured upload limit")
                await handle.write(block)
    except BaseException:
        path.unlink(missing_ok=True)
        directory.rmdir()
        raise
    finally:
        await file.close()
    if not size:
        path.unlink()
        directory.rmdir()
        raise HTTPException(422, "Empty upload")
    await store.create_job(
        job_id,
        {
            "filename": file.filename,
            "video_path": str(path.resolve()),
            "result_data": {"needs_reindex": False},
        },
    )
    try:
        await request.app.state.queue.enqueue_job(
            "process_video", str(job_id), str(path.resolve()), _job_id=f"ingest:{job_id}:new:0"
        )
    except Exception as exc:
        await store.update_job(job_id, {"status": "failed", "error": "Unable to enqueue ingestion"})
        raise HTTPException(
            503, {"job_id": str(job_id), "message": "Queue unavailable; recording retained"}
        ) from exc
    return {"job_id": str(job_id), "status": "pending", "message": "Recording queued"}


@router.get("/status/{job_id}")
async def get_ingest_status(job_id: UUID, store: MetadataStoreDep):
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    state = job["pipeline"]
    return {
        "job_id": str(job_id),
        "status": job["status"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "error": job["error"],
        "duration_ms": job["duration_ms"],
        "coverage_ms": state.get("checkpoint_ms", 0),
        "remaining_ms": max(0, (job["duration_ms"] or 0) - state.get("checkpoint_ms", 0)),
        "streams": state.get("streams", {}),
        "warnings": state.get("warnings", []),
        "needs_reindex": state.get("needs_reindex", False),
    }


@router.delete("/{job_id}", status_code=204)
async def cancel_job(job_id: UUID, request: Request, store: MetadataStoreDep):
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("complete", "failed", "cancelled"):
        raise HTTPException(409, "Job is already terminal")
    await JobOrchestrator(store, request.app.state.queue).update(job_id, status="cancelled")


@router.post("/{job_id}/reindex", status_code=202)
async def reindex_job(job_id: UUID, request: Request, store: MetadataStoreDep):
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] not in ("complete", "failed"):
        raise HTTPException(409, "Only complete or failed recordings can be reindexed")
    # A new job fences old tasks and preserves the previous searchable generation.
    new_id = uuid4()
    await store.create_job(
        new_id,
        {
            "filename": job["filename"],
            "video_path": job["video_path"],
            "result_data": {"needs_reindex": False, "replaces": str(job_id)},
        },
    )
    try:
        await request.app.state.queue.enqueue_job(
            "process_video", str(new_id), job["video_path"], _job_id=f"ingest:{new_id}:new:0"
        )
    except Exception as exc:
        await store.update_job(new_id, {"status": "failed", "error": "Queue unavailable"})
        raise HTTPException(503, "Unable to enqueue reindex") from exc
    return {"job_id": str(new_id), "replaces": str(job_id), "status": "pending"}
