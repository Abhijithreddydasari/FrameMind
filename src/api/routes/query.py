"""Semantic query endpoints."""
import time
from datetime import datetime
from uuid import UUID, uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.deps import MetadataStoreDep, RedisCacheDep, SettingsDep
from src.core.logging import get_logger
from src.core.models import FrameSource, JobStatus, QueryRequest, QueryResponse
from src.ingest.extractor import FrameExtractor
from src.ml.clip_scorer import CLIPScorer
from src.ml.embeddings import FaissEmbeddingIndex, top_k_similar
from src.vlm.aggregator import BatchAggregator
from src.vlm.client import VLMClient
from src.vlm.prompt_builder import PromptBuilder

logger = get_logger(__name__)

router = APIRouter()

_clip_scorer: CLIPScorer | None = None


async def get_clip_scorer() -> CLIPScorer:
    """Lazy-load CLIP model for query scoring."""
    global _clip_scorer
    if _clip_scorer is None:
        _clip_scorer = CLIPScorer()
        await _clip_scorer.load_model()
    return _clip_scorer


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

    # Load embeddings from metadata store
    embeddings = await metadata_store.get_embeddings(job_id)
    if not embeddings:
        raise HTTPException(status_code=400, detail="No embeddings available for this job")

    # Encode query and score frames
    clip_scorer = await get_clip_scorer()
    query_embedding = clip_scorer.embed_text(query.query)

    embedding_matrix = np.array([e["embedding"] for e in embeddings], dtype=np.float32)
    indices = [e["frame_index"] for e in embeddings]
    top_k = min(query.max_frames, len(indices))

    if settings.use_faiss:
        try:
            faiss_index = FaissEmbeddingIndex(dim=embedding_matrix.shape[1])
            faiss_index.add_batch(
                embedding_matrix,
                metadata_list=[{"frame_index": idx} for idx in indices],
                ids=indices,
            )
            ranked_raw = faiss_index.search(query_embedding, k=top_k)
            ranked = [(item[0], item[1]) for item in ranked_raw]
        except Exception:
            ranked = top_k_similar(
                query_embedding,
                embedding_matrix,
                k=top_k,
                indices=indices,
            )
    else:
        ranked = top_k_similar(
            query_embedding,
            embedding_matrix,
            k=top_k,
            indices=indices,
        )

    # Build sources and ensure frame assets exist
    job_data = await metadata_store.get_job(job_id)
    if not job_data or not job_data.get("video_path"):
        raise HTTPException(status_code=400, detail="Video path not available for this job")

    video_path = job_data["video_path"]
    extractor = FrameExtractor()

    # Map frame_index -> embedding info
    emb_map = {e["frame_index"]: e for e in embeddings}
    needed_timestamps = []
    for frame_index, _score in ranked:
        frame_info = emb_map.get(frame_index)
        if frame_info and not frame_info.get("frame_path"):
            needed_timestamps.append(frame_info.get("timestamp_ms", 0))

    # Extract missing frames on demand
    frame_paths: dict[int, str] = {}
    if needed_timestamps:
        extracted = extractor.extract_at_timestamps(video_path, needed_timestamps)
        output_dir = settings.storage_path / "frames" / str(job_id)
        saved_paths = extractor.save_frames(extracted, output_dir)
        for frame_data, path in zip(extracted, saved_paths):
            frame_paths[frame_data.index] = str(path)

        if frame_paths:
            await metadata_store.update_frame_paths(job_id, frame_paths)

    # Build sources and VLM inputs
    sources: list[FrameSource] = []
    image_paths = []
    for frame_index, score in ranked:
        frame_info = emb_map.get(frame_index)
        if not frame_info:
            continue
        frame_path = frame_info.get("frame_path") or frame_paths.get(frame_index)
        if frame_path:
            image_paths.append(frame_path)
        sources.append(
            FrameSource(
                frame_index=frame_index,
                timestamp_ms=int(frame_info.get("timestamp_ms", 0)),
                relevance_score=float(score),
                description=None,
            )
        )

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
        frames_analyzed=len(ranked),
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
