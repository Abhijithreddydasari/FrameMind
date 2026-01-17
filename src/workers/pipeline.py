"""ARQ task definitions for video processing pipeline.

Defines async tasks for each stage of the video processing pipeline:
1. Preprocessing - validate and normalize video
2. Frame extraction - extract frames at configured FPS
3. Frame selection - ML-based intelligent selection (spatial + temporal)
4. VLM analysis - send selected frames to VLM
5. Aggregation - combine results

Supports parallel two-stream extraction (CLIP + X-CLIP).
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from arq import ArqRedis, cron
from arq.connections import RedisSettings

from src.core.config import settings
from src.core.logging import get_logger
from src.core.models import JobStatus
from src.ingest.extractor import FrameExtractor
from src.storage.metadata import MetadataStore

logger = get_logger(__name__)


def get_redis_settings() -> RedisSettings:
    """Get Redis settings for ARQ."""
    # Parse Redis URL
    url = settings.redis_url_str
    # arq expects host/port, not URL
    # Simple parsing for redis://host:port/db format
    if url.startswith("redis://"):
        url = url[8:]
    host_port, _, db = url.partition("/")
    host, _, port = host_port.partition(":")

    return RedisSettings(
        host=host or "localhost",
        port=int(port) if port else 6379,
        database=int(db) if db else 0,
    )


# ============ Pipeline Tasks ============


async def process_video(
    ctx: dict[str, Any],
    job_id: str,
    video_path: str,
) -> dict[str, Any]:
    """Main video processing task.
    
    Orchestrates the full pipeline from preprocessing to frame selection.
    
    Args:
        ctx: ARQ context (contains Redis connection)
        job_id: Unique job identifier
        video_path: Path to uploaded video
        
    Returns:
        Processing result with selected frames and metadata
    """
    redis: ArqRedis = ctx["redis"]
    metadata_store: MetadataStore = ctx["metadata_store"]
    job_uuid = UUID(job_id)

    logger.info("Starting video processing", job_id=job_id)

    try:
        # Update status
        await _update_job_status(redis, job_uuid, JobStatus.PROCESSING, progress=0.1)

        # Step 1: Validate and preprocess
        await _update_job_status(redis, job_uuid, JobStatus.PROCESSING, progress=0.2)
        validated_path = await preprocess_video(ctx, job_id, video_path)
        await metadata_store.update_job(
            job_uuid,
            {"video_path": validated_path, "status": JobStatus.PROCESSING.value},
        )

        # Step 2: Parallel extraction (spatial + temporal)
        await _update_job_status(redis, job_uuid, JobStatus.EXTRACTING, progress=0.3)
        
        # Run both streams concurrently
        spatial_task = extract_and_select_frames(ctx, job_id, validated_path)
        
        if settings.use_temporal:
            temporal_task = extract_temporal_clips(ctx, job_id, validated_path)
            spatial_result, temporal_result = await asyncio.gather(
                spatial_task, temporal_task, return_exceptions=True
            )
        else:
            spatial_result = await spatial_task
            temporal_result = None

        # Handle exceptions
        if isinstance(spatial_result, Exception):
            raise spatial_result
        
        if isinstance(temporal_result, Exception):
            logger.warning("Temporal extraction failed, continuing with spatial only", 
                          error=str(temporal_result))
            temporal_result = None

        # Step 3: Cache embeddings
        await _update_job_status(redis, job_uuid, JobStatus.ANALYZING, progress=0.7)
        await cache_embeddings(ctx, job_id, spatial_result, temporal_result)
        await metadata_store.update_job(
            job_uuid,
            {
                "status": JobStatus.COMPLETE.value,
                "keyframe_count": len(selection_result.get("selected_indices", [])),
                "scene_count": len(selection_result.get("scene_boundaries", [])),
                "completed_at": datetime.utcnow(),
            },
        )

        # Mark complete
        await _update_job_status(redis, job_uuid, JobStatus.COMPLETE, progress=1.0)

        frames_selected = len(spatial_result.get("selected_indices", []))
        clips_extracted = len(temporal_result.get("clip_embeddings", [])) if temporal_result else 0

        logger.info(
            "Video processing complete",
            job_id=job_id,
            frames_selected=frames_selected,
            clips_extracted=clips_extracted,
        )

        return {
            "status": "complete",
            "frames_selected": frames_selected,
            "clips_extracted": clips_extracted,
            "scene_boundaries": len(spatial_result.get("scene_boundaries", [])),
        }

    except Exception as e:
        logger.exception("Video processing failed", job_id=job_id)
        await _update_job_status(
            redis, job_uuid, JobStatus.FAILED, error=str(e)
        )
        raise


async def preprocess_video(
    ctx: dict[str, Any],
    job_id: str,
    video_path: str,
) -> str:
    """Validate and preprocess video.
    
    Checks format, extracts metadata, and optionally transcodes.
    
    Returns:
        Path to preprocessed video (may be same as input)
    """
    logger.info("Preprocessing video", job_id=job_id, video_path=video_path)

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Validate format using ffprobe
    # In production: run ffprobe and check codec, resolution, etc.
    # For now, just verify the file exists and is readable

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    logger.info(
        "Video validated",
        job_id=job_id,
        fps=fps,
        resolution=f"{width}x{height}",
        frames=frame_count,
    )

    # Store metadata
    redis: ArqRedis = ctx["redis"]
    metadata_store: MetadataStore = ctx["metadata_store"]
    await redis.hset(
        f"job:{job_id}:metadata",
        mapping={
            "fps": str(fps),
            "width": str(width),
            "height": str(height),
            "frame_count": str(frame_count),
            "duration_ms": str(int((frame_count / fps) * 1000)) if fps > 0 else "0",
        },
    )

    # Persist metadata to database
    await metadata_store.update_job(
        UUID(job_id),
        {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_ms": int((frame_count / fps) * 1000) if fps > 0 else 0,
        },
    )

    return video_path


async def extract_and_select_frames(
    ctx: dict[str, Any],
    job_id: str,
    video_path: str,
) -> dict[str, Any]:
    """Extract frames and perform intelligent selection.
    
    Uses CLIP embeddings and shot detection for selection.
    
    Returns:
        Selection result with frame indices and embeddings
    """
    logger.info("Extracting and selecting frames", job_id=job_id)

    from src.ml.frame_selector import FrameSelector

    selector = FrameSelector()
    await selector.initialize()

    try:
        result = await selector.select_from_video(video_path)

        selection_payload = {
            "selected_indices": result.selected_indices,
            "scene_boundaries": [
                {"frame_index": b.frame_index, "confidence": b.confidence}
                for b in result.scene_boundaries
            ],
            "scores": result.selection_scores,
            "embeddings": [
                {
                    "frame_index": e.frame_index,
                    "embedding": e.embedding.tolist(),
                    "timestamp_ms": e.timestamp_ms,
                    "is_selected": e.frame_index in result.selected_indices,
                }
                for e in result.embeddings
            ],
        }
        await _persist_frame_data(ctx, job_id, video_path, selection_payload)

        return selection_payload

    finally:
        await selector.cleanup()


async def extract_temporal_clips(
    ctx: dict[str, Any],
    job_id: str,
    video_path: str,
) -> dict[str, Any]:
    """Extract temporal clips and compute X-CLIP embeddings.
    
    Uses sliding window extraction and batched GPU encoding.
    
    Returns:
        Dict with clip_embeddings and clip metadata
    """
    logger.info("Extracting temporal clips", job_id=job_id)

    from src.ml.temporal_encoder import XCLIPEncoder, ClipConfig

    config = ClipConfig(
        window_frames=settings.temporal_window_frames,
        clip_fps=settings.temporal_fps,
        stride=settings.temporal_stride,
    )

    encoder = XCLIPEncoder()

    try:
        clip_embeddings = await encoder.extract_and_encode(video_path, config)

        logger.info(
            "Temporal clips extracted",
            job_id=job_id,
            clip_count=len(clip_embeddings),
        )

        return {
            "clip_embeddings": [
                {
                    "clip_index": ce.clip_index,
                    "embedding": ce.embedding.tolist(),
                    "start_frame": ce.start_frame,
                    "end_frame": ce.end_frame,
                    "start_ms": ce.start_ms,
                    "end_ms": ce.end_ms,
                }
                for ce in clip_embeddings
            ],
        }

    finally:
        await encoder.unload_model()


async def cache_embeddings(
    ctx: dict[str, Any],
    job_id: str,
    spatial_result: dict[str, Any],
    temporal_result: dict[str, Any] | None = None,
) -> None:
    """Cache spatial and temporal embeddings."""
    redis: ArqRedis = ctx["redis"]
    metadata_store: MetadataStore = ctx["metadata_store"]
    job_uuid = UUID(job_id)

    # Save spatial embeddings
    embeddings = spatial_result.get("embeddings", [])
    selected_indices = set(spatial_result.get("selected_indices", []))

    spatial_count = 0
    for emb in embeddings:
        if emb["frame_index"] in selected_indices:
            key = f"job:{job_id}:spatial:{emb['frame_index']}"
            await redis.set(
                key,
                str(emb["embedding"]),
                ex=86400 * 7,  # 7 days TTL
            )
            spatial_count += 1

    # Save to database
    frame_embs = [
        {
            "frame_index": emb["frame_index"],
            "timestamp_ms": emb.get("timestamp_ms", 0),
            "embedding": emb.get("embedding", []),
            "is_selected": emb["frame_index"] in selected_indices,
            "frame_path": emb.get("frame_path"),
        }
        for emb in embeddings
        if emb["frame_index"] in selected_indices
    ]
    await metadata_store.save_embeddings(job_uuid, frame_embs)

    # Save temporal embeddings if available
    temporal_count = 0
    if temporal_result:
        clip_embs = temporal_result.get("clip_embeddings", [])
        
        for ce in clip_embs:
            key = f"job:{job_id}:temporal:{ce['clip_index']}"
            await redis.set(
                key,
                str(ce["embedding"]),
                ex=86400 * 7,
            )
            temporal_count += 1

        # Save to DB
        temp_emb_data = [
            {
                "clip_index": ce["clip_index"],
                "start_frame": ce["start_frame"],
                "end_frame": ce["end_frame"],
                "start_ms": ce["start_ms"],
                "end_ms": ce["end_ms"],
                "embedding_path": f"job:{job_id}:temporal:{ce['clip_index']}",
            }
            for ce in clip_embs
        ]
        await metadata_store.save_temporal_embeddings(job_uuid, temp_emb_data)

    logger.info(
        "Embeddings cached",
        job_id=job_id,
        spatial_count=spatial_count,
        temporal_count=temporal_count,
    )


async def _persist_frame_data(
    ctx: dict[str, Any],
    job_id: str,
    video_path: str,
    selection_result: dict[str, Any],
) -> None:
    """Persist embeddings and selected frame assets."""
    metadata_store: MetadataStore = ctx["metadata_store"]
    embeddings = selection_result.get("embeddings", [])
    selected_indices = set(selection_result.get("selected_indices", []))

    # Persist embeddings to database
    await metadata_store.save_embeddings(UUID(job_id), embeddings)

    # Extract and store selected frames
    selected_timestamps = [
        emb["timestamp_ms"]
        for emb in embeddings
        if emb["frame_index"] in selected_indices
    ]

    if not selected_timestamps:
        return

    extractor = FrameExtractor()
    extracted = extractor.extract_at_timestamps(video_path, selected_timestamps)

    output_dir = settings.storage_path / "frames" / job_id
    saved_paths = extractor.save_frames(extracted, output_dir)

    # Map frame_index to saved path
    frame_paths: dict[int, str] = {}
    for frame_data, path in zip(extracted, saved_paths):
        frame_paths[frame_data.index] = str(path)

    await metadata_store.update_frame_paths(UUID(job_id), frame_paths)


async def query_video(
    ctx: dict[str, Any],
    job_id: str,
    query_id: str,
    query_text: str,
    max_frames: int = 10,
) -> dict[str, Any]:
    """Process a semantic query against a video.
    
    1. Load cached embeddings
    2. Score frames against query
    3. Send top frames to VLM
    4. Aggregate response
    
    Returns:
        Query result with answer and sources
    """
    logger.info(
        "Processing query",
        job_id=job_id,
        query_id=query_id,
        query_length=len(query_text),
    )

    # In production:
    # 1. Load embeddings from cache
    # 2. Use CLIP to encode query
    # 3. Find top-k similar frames
    # 4. Load frame images
    # 5. Send to VLM with prompt
    # 6. Aggregate and return

    # Placeholder for scaffolding
    return {
        "query_id": query_id,
        "job_id": job_id,
        "answer": "[VLM integration pending]",
        "confidence": 0.0,
        "frames_analyzed": 0,
        "sources": [],
    }


# ============ Helper Functions ============


async def _update_job_status(
    redis: ArqRedis,
    job_id: UUID,
    status: JobStatus,
    progress: float = 0.0,
    error: str | None = None,
) -> None:
    """Update job status in Redis."""
    data = {
        "status": status.value,
        "progress": str(progress),
        "updated_at": datetime.utcnow().isoformat(),
    }
    if error:
        data["error"] = error

    await redis.hset(f"job:{job_id}", mapping=data)


# ============ Cleanup Tasks ============


async def cleanup_old_jobs(ctx: dict[str, Any]) -> int:
    """Periodic task to clean up old completed/failed jobs.
    
    Runs daily and removes jobs older than 7 days.
    """
    redis: ArqRedis = ctx["redis"]
    cutoff = datetime.utcnow() - timedelta(days=7)
    cleaned = 0

    # In production: scan for old jobs and delete
    logger.info("Running job cleanup", cutoff=cutoff.isoformat())

    return cleaned


# ============ Worker Configuration ============


async def startup(ctx: dict[str, Any]) -> None:
    """Worker startup hook."""
    logger.info("Worker starting up")

    # Initialize metadata store
    metadata_store = MetadataStore()
    await metadata_store.initialize()
    ctx["metadata_store"] = metadata_store

    # Pre-load ML models if configured
    if not settings.is_production:
        logger.info("Development mode: ML models will be loaded on demand")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Worker shutdown hook."""
    logger.info("Worker shutting down")
    metadata_store: MetadataStore | None = ctx.get("metadata_store")
    if metadata_store:
        await metadata_store.close()


class WorkerSettings:
    """ARQ worker settings."""

    functions = [
        process_video,
        preprocess_video,
        extract_and_select_frames,
        extract_temporal_clips,
        cache_embeddings,
        query_video,
    ]

    cron_jobs = [
        cron(cleanup_old_jobs, hour=3, minute=0),  # Run at 3 AM daily
    ]

    on_startup = startup
    on_shutdown = shutdown

    max_jobs = settings.worker_concurrency
    job_timeout = settings.job_timeout

    # Retry configuration
    max_tries = 3
    retry_defer = timedelta(seconds=30)

    redis_settings = get_redis_settings()
