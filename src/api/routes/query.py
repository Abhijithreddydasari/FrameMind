"""Semantic query endpoints."""
import time
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.deps import RedisCacheDep, SettingsDep
from src.core.logging import get_logger
from src.core.models import FrameSource, JobStatus, QueryRequest, QueryResponse

logger = get_logger(__name__)

router = APIRouter()


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
) -> QueryResponse:
    """Query a processed video with natural language.
    
    The query will be matched against video frames using CLIP embeddings,
    and relevant frames will be sent to the VLM for analysis.
    """
    start_time = time.perf_counter()

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

    # In production: 
    # 1. Load frame embeddings from cache/storage
    # 2. Encode query text with CLIP
    # 3. Find top-K similar frames
    # 4. Send frames to VLM with context
    # 5. Aggregate responses

    # Placeholder response for scaffolding
    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    response = QueryResponse(
        job_id=job_id,
        query=query.query,
        answer="[VLM integration pending] This is a placeholder response. "
               "The full implementation will analyze video frames using CLIP "
               "embeddings and generate answers via GPT-4V or Claude.",
        confidence=0.0,
        frames_analyzed=0,
        processing_time_ms=processing_time_ms,
        sources=[],
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
