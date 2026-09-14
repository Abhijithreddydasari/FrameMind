"""Shared synchronous, asynchronous, and interval-inspection APIs."""

from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.deps import MetadataStoreDep
from src.core.models import InspectRequest, QueryRequest, QueryResponse

router = APIRouter()


@router.get("/result/{query_id}")
async def get_query_result(query_id: UUID, store: MetadataStoreDep):
    task = await store.get_query_task(str(query_id))
    if not task:
        raise HTTPException(404, "Query not found")
    if task["status"] in ("pending", "processing"):
        return JSONResponse(
            {"query_id": str(query_id), "status": task["status"]},
            status_code=202,
            headers={"Retry-After": "5"},
        )
    if task["status"] == "failed":
        raise HTTPException(500, task["error"])
    return {"query_id": str(query_id), "status": "complete", **task["result"]}


@router.post("/{job_id}", response_model=QueryResponse)
async def query_video(job_id: UUID, query: QueryRequest, request: Request):
    return await request.app.state.query_service.query(job_id, query)


@router.post("/{job_id}/inspect", response_model=QueryResponse)
async def inspect_video(job_id: UUID, query: InspectRequest, request: Request):
    return await request.app.state.query_service.query(job_id, query)


@router.post("/{job_id}/async", status_code=202)
async def query_video_async(
    job_id: UUID, query: QueryRequest, request: Request, store: MetadataStoreDep
):
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "complete" or job["pipeline"].get("needs_reindex"):
        raise HTTPException(409, "Recording is not ready")
    query_id = str(uuid4())
    await store.put_query_task(query_id, str(job_id), query.model_dump(mode="json"))
    try:
        await request.app.state.queue.enqueue_job(
            "query_video", query_id, _job_id=f"query:{query_id}"
        )
    except Exception as exc:
        await store.update_query_task(query_id, status="failed", error="Queue unavailable")
        raise HTTPException(503, "Unable to enqueue query") from exc
    return {"query_id": query_id, "job_id": str(job_id), "status": "pending"}
