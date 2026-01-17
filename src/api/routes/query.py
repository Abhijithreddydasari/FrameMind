"""Semantic query endpoints with dual-stream (spatial + temporal) retrieval."""
import time
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.deps import MetadataStoreDep, RedisCacheDep, SettingsDep
from src.core.config import settings as app_config
from src.core.logging import get_logger
from src.core.models import FrameSource, JobStatus, QueryRequest, QueryResponse
from src.ingest.extractor import FrameExtractor
from src.ml.clip_scorer import CLIPScorer
from src.ml.embeddings import DualStreamIndex, FaissEmbeddingIndex, top_k_similar
from src.vlm.aggregator import BatchAggregator
from src.vlm.client import VLMClient
from src.vlm.prompt_builder import PromptBuilder

logger = get_logger(__name__)

router = APIRouter()

_clip_scorer: CLIPScorer | None = None
_xclip_encoder: Any = None


async def get_clip_scorer() -> CLIPScorer:
    """Lazy-load CLIP model for query scoring."""
    global _clip_scorer
    if _clip_scorer is None:
        _clip_scorer = CLIPScorer()
        await _clip_scorer.load_model()
    return _clip_scorer


async def get_xclip_encoder() -> Any:
    """Lazy-load X-CLIP model for temporal query encoding."""
    global _xclip_encoder
    if _xclip_encoder is None and app_config.use_temporal:
        try:
            from src.ml.temporal_encoder import XCLIPEncoder
            _xclip_encoder = XCLIPEncoder()
            await _xclip_encoder.load_model()
        except Exception as e:
            logger.warning("Failed to load X-CLIP encoder", error=str(e))
            return None
    return _xclip_encoder


class QueryJobResponse(BaseModel):
    """Response when submitting a query."""

    query_id: UUID
    job_id: UUID
    status: str
    message: str


class QueryResultResponse(BaseModel):
    """Full query result."""

    query_id: UUID
    job_id: UUID
    query: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    frames_analyzed: int
    processing_time_ms: int
    sources: list[FrameSource]
    created_at: datetime


@router.post("/{job_id}", response_model=QueryResponse)
async def query_video(
    job_id: UUID,
    query: QueryRequest,
    settings: SettingsDep,
    cache: RedisCacheDep,
    metadata_store: MetadataStoreDep,
) -> QueryResponse:
    """Query a processed video with natural language.
    
    The query will be matched against video frames using CLIP embeddings,
    and relevant frames will be sent to the VLM for analysis.
    """
    start_time = time.perf_counter()
    query_id = uuid4()

    # Verify job exists and is complete
    job_data = await cache.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = JobStatus(job_data.get("status", "pending"))

    if job_status != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready for queries. Current status: {job_status.value}",
        )

    # Check query cache
    cache_key = f"query:{job_id}:{hash(query.query)}"
    cached_result = await cache.get(cache_key)

    if cached_result:
        logger.info("Query cache hit", job_id=str(job_id))
        # Return cached response
        return QueryResponse(**cached_result)

    # Load spatial embeddings from metadata store
    spatial_embeddings = await metadata_store.get_embeddings(job_id)
    
    # Load temporal embeddings if available
    temporal_embeddings: list[dict[str, Any]] = []
    if app_config.use_temporal:
        temporal_embeddings = await metadata_store.get_temporal_embeddings(job_id)

    if not spatial_embeddings and not temporal_embeddings:
        raise HTTPException(status_code=400, detail="No embeddings available for this job")

    # Build dual-stream index
    dual_index = DualStreamIndex()

    # Add spatial embeddings
    for emb in spatial_embeddings:
        dual_index.add_spatial(
            np.array(emb["embedding"], dtype=np.float32),
            {"frame_index": emb["frame_index"], "timestamp_ms": emb.get("timestamp_ms", 0)}
        )

    # Add temporal embeddings (load from Redis cache)
    for emb in temporal_embeddings:
        emb_key = emb.get("embedding_path", f"job:{job_id}:temporal:{emb['clip_index']}")
        cached = await cache.client.get(emb_key)
        if cached:
            try:
                embedding = eval(cached)  # Parse from string (safe - we control format)
                dual_index.add_temporal(
                    np.array(embedding, dtype=np.float32),
                    {
                        "clip_index": emb["clip_index"],
                        "start_ms": emb.get("start_ms", 0),
                        "end_ms": emb.get("end_ms", 0),
                    }
                )
            except Exception as e:
                logger.warning("Failed to load temporal embedding", clip_index=emb.get("clip_index"), error=str(e))

    logger.info(
        "Dual index built",
        spatial_count=dual_index.spatial_count,
        temporal_count=dual_index.temporal_count,
    )

    # Encode query with both encoders
    clip_scorer = await get_clip_scorer()
    spatial_query = clip_scorer.embed_text(query.query)

    temporal_query = None
    if app_config.use_temporal and temporal_embeddings:
        xclip = await get_xclip_encoder()
        if xclip:
            try:
                temporal_query = xclip.encode_text(query.query)
            except Exception as e:
                logger.warning("Temporal query encoding failed", error=str(e))

    # Fused search across both indexes
    top_k = min(query.max_frames, max(len(spatial_embeddings), len(temporal_embeddings)))
    fused_results = dual_index.search_fused(
        spatial_query=spatial_query,
        temporal_query=temporal_query,
        k=top_k,
        alpha=app_config.fusion_alpha,
    )

    logger.info(
        "Fused search complete",
        results_count=len(fused_results),
        alpha=app_config.fusion_alpha,
    )

    # Build sources from fused results
    sources: list[FrameSource] = []
    emb_map = {e["frame_index"]: e for e in spatial_embeddings}
    
    for stream, idx, score, meta in fused_results[:query.max_frames]:
        if stream == "spatial":
            sources.append(
                FrameSource(
                    frame_index=meta.get("frame_index", idx),
                    timestamp_ms=int(meta.get("timestamp_ms", 0)),
                    relevance_score=float(score),
                    description="Matched via spatial (frame) similarity",
                )
            )
        else:
            sources.append(
                FrameSource(
                    frame_index=meta.get("clip_index", idx),
                    timestamp_ms=int(meta.get("start_ms", 0)),
                    relevance_score=float(score),
                    description=f"Matched via temporal (clip) similarity ({meta.get('start_ms', 0)}-{meta.get('end_ms', 0)}ms)",
                )
            )

    # Build sources and ensure frame assets exist
    job_data = await metadata_store.get_job(job_id)
    if not job_data or not job_data.get("video_path"):
        raise HTTPException(status_code=400, detail="Video path not available for this job")

    video_path = job_data["video_path"]
    extractor = FrameExtractor()

    # Get timestamps for frames we need to extract
    needed_timestamps = []
    for source in sources:
        frame_info = emb_map.get(source.frame_index)
        if frame_info and not frame_info.get("frame_path"):
            needed_timestamps.append(source.timestamp_ms)

    # Extract missing frames on demand
    frame_paths: dict[int, str] = {}
    if needed_timestamps:
        try:
            extracted = extractor.extract_at_timestamps(video_path, needed_timestamps)
            output_dir = settings.storage_path / "frames" / str(job_id)
            saved_paths = extractor.save_frames(extracted, output_dir)
            for frame_data, path in zip(extracted, saved_paths):
                frame_paths[frame_data.index] = str(path)

            if frame_paths:
                await metadata_store.update_frame_paths(job_id, frame_paths)
        except Exception as e:
            logger.warning("Frame extraction failed", error=str(e))

    # Collect image paths for VLM
    image_paths = []
    for source in sources:
        frame_info = emb_map.get(source.frame_index)
        if frame_info:
            frame_path = frame_info.get("frame_path") or frame_paths.get(source.frame_index)
            if frame_path:
                image_paths.append(frame_path)

    # Call VLM if configured
    answer = "VLM not configured. Provide VLM_API_KEY to enable semantic answers."
    confidence = 0.0

    if settings.vlm_api_key and image_paths:
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build_analysis_prompt(
            query=query.query,
            frames=None,
            include_timestamps=query.include_timestamps,
        )
        vlm_client = VLMClient.create()
        response = await vlm_client.analyze_images(image_paths, prompt)

        aggregator = BatchAggregator()
        aggregated = aggregator.structure_response(response.content, frames=[])

        answer = aggregated.answer
        confidence = aggregated.confidence

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    response = QueryResponse(
        job_id=job_id,
        query=query.query,
        answer=answer,
        confidence=confidence,
        frames_analyzed=len(sources),
        processing_time_ms=processing_time_ms,
        sources=sources,
    )

    # Persist query result
    await metadata_store.save_query(
        query_id=query_id,
        job_id=job_id,
        query_text=query.query,
        result=response.model_dump(mode="json"),
    )

    # Cache the result (with TTL)
    await cache.set(cache_key, response.model_dump(mode="json"), ttl=3600)

    logger.info(
        "Query processed",
        job_id=str(job_id),
        query_length=len(query.query),
        processing_time_ms=processing_time_ms,
    )

    return response


@router.post("/{job_id}/async", response_model=QueryJobResponse, status_code=202)
async def query_video_async(
    job_id: UUID,
    query: QueryRequest,
    cache: RedisCacheDep,
) -> QueryJobResponse:
    """Submit an async query for a processed video.
    
    For long-running queries or when rate limits are a concern.
    Poll /query/result/{query_id} for results.
    """
    # Verify job exists
    job_data = await cache.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = JobStatus(job_data.get("status", "pending"))

    if job_status != JobStatus.COMPLETE:
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready for queries. Current status: {job_status.value}",
        )

    # Create query job
    query_id = uuid4()
    query_data = {
        "query_id": str(query_id),
        "job_id": str(job_id),
        "query": query.query,
        "max_frames": query.max_frames,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
    }

    await cache.set(f"query_job:{query_id}", query_data, ttl=86400)

    # In production: Enqueue to ARQ for async processing

    logger.info("Async query submitted", query_id=str(query_id), job_id=str(job_id))

    return QueryJobResponse(
        query_id=query_id,
        job_id=job_id,
        status="pending",
        message="Query submitted. Poll /query/result/{query_id} for results.",
    )


@router.get("/result/{query_id}", response_model=QueryResultResponse)
async def get_query_result(
    query_id: UUID,
    cache: RedisCacheDep,
) -> QueryResultResponse:
    """Get the result of an async query."""
    query_data = await cache.get(f"query_job:{query_id}")

    if not query_data:
        raise HTTPException(status_code=404, detail="Query not found")

    if query_data.get("status") == "pending":
        raise HTTPException(
            status_code=202,
            detail="Query still processing",
            headers={"Retry-After": "5"},
        )

    if query_data.get("status") == "failed":
        raise HTTPException(
            status_code=500,
            detail=query_data.get("error", "Query processing failed"),
        )

    # Return completed result
    return QueryResultResponse(
        query_id=query_id,
        job_id=UUID(query_data["job_id"]),
        query=query_data["query"],
        answer=query_data.get("answer", ""),
        confidence=query_data.get("confidence", 0.0),
        frames_analyzed=query_data.get("frames_analyzed", 0),
        processing_time_ms=query_data.get("processing_time_ms", 0),
        sources=[FrameSource(**s) for s in query_data.get("sources", [])],
        created_at=datetime.fromisoformat(query_data["created_at"]),
    )
