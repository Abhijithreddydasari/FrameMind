"""Checkpointed ARQ tasks. Each invocation encodes at most one chunk."""

import asyncio
import hashlib
import json
from datetime import datetime
from uuid import UUID, uuid4

from arq import Retry, cron
from arq.connections import RedisSettings

from src.core.concurrency import run_blocking
from src.core.config import settings
from src.core.logging import get_logger
from src.ingest.stream import fingerprint, probe
from src.storage.indexes import atomic_json, write_shard
from src.storage.metadata import MetadataStore
from src.workers.orchestrator import JobOrchestrator

logger = get_logger(__name__)


def get_redis_settings():
    return RedisSettings.from_dsn(settings.redis_url_str)


def encoding_config(config):
    names = (
        "clip_model",
        "xclip_model",
        "model_revision",
        "frame_extraction_fps",
        "temporal_fps",
        "temporal_stride",
        "decode_size",
        "use_temporal",
        "chunk_seconds",
    )
    return {name: getattr(config, name) for name in names}


async def get_encoders(ctx):
    if "spatial" not in ctx:
        from src.ml.clip_scorer import CLIPScorer

        scorer = CLIPScorer()
        await scorer.load_model()
        ctx["spatial"] = scorer
    if settings.use_temporal and "temporal" not in ctx:
        from src.ml.temporal_encoder import XCLIPEncoder

        encoder = XCLIPEncoder(device=settings.clip_device)
        await encoder.load_model()
        ctx["temporal"] = encoder
    return ctx["spatial"], ctx.get("temporal")


async def process_video(ctx, job_id: str, video_path: str):
    token = uuid4().hex
    store = ctx["metadata_store"]
    if not await store.acquire_lease(job_id, token, settings.job_timeout + 60):
        return {"status": "busy"}
    try:
        return await _process_video(ctx, job_id, video_path)
    finally:
        await store.release_lease(job_id, token)


async def _process_video(ctx, job_id: str, video_path: str):
    store = ctx["metadata_store"]
    lifecycle = JobOrchestrator(store, ctx["redis"])
    job = await store.get_job(UUID(job_id))
    if not job or job["status"] in ("cancelled", "complete"):
        return {"status": job["status"] if job else "missing"}
    try:
        state = dict(job["pipeline"])
        signature = await asyncio.to_thread(fingerprint, video_path)
        if state.get("source") and state["source"] != signature:
            raise ValueError("Source recording changed; explicit reindex required")
        profile = encoding_config(settings)
        if state.get("config") and state["config"] != profile:
            raise ValueError("Encoding configuration changed; explicit reindex required")
        if not state.get("generation"):
            metadata = await asyncio.to_thread(probe, video_path)
            state = {
                **state,
                "generation": uuid4().hex,
                "checkpoint_ms": 0,
                "source": signature,
                "config": profile,
                "needs_reindex": False,
                "warnings": [],
                "streams": {
                    "spatial": "pending",
                    "temporal": "pending" if settings.use_temporal else "disabled",
                },
            }
            await lifecycle.update(job_id, **metadata, status="processing", result_data=state)
            job.update(metadata)
        start = state["checkpoint_ms"]
        duration = job["duration_ms"]
        end = min(duration, start + settings.chunk_seconds * 1000)
        root = settings.storage_path / "indexes" / job_id / state["generation"]
        await lifecycle.update(job_id, status="extracting", result_data=state)
        if start < duration:
            try:
                spatial, temporal = await get_encoders(ctx)
            except Exception as exc:
                if "spatial" not in ctx:
                    raise
                spatial, temporal = ctx["spatial"], None
                state["streams"]["temporal"] = "failed"
                state["warnings"] = list(
                    dict.fromkeys(
                        [*state["warnings"], f"Temporal model unavailable: {type(exc).__name__}"]
                    )
                )
            from src.ml.chunk_encoder import encode_chunk

            revisions = {
                "spatial": getattr(spatial._model.config, "_commit_hash", None)
                or settings.model_revision
            }
            if temporal:
                revisions["temporal"] = (
                    getattr(temporal._model.config, "_commit_hash", None) or settings.model_revision
                )
            if any(state.get("model_revisions", {}).get(k, v) != v for k, v in revisions.items()):
                raise ValueError("Checkpoint revision changed between chunks; reindex required")
            state["model_revisions"] = {**state.get("model_revisions", {}), **revisions}
            try:
                outputs = await run_blocking(
                    encode_chunk, video_path, start, end, duration, settings, spatial, temporal
                )
            except Exception:
                if temporal is None:
                    raise
                outputs = await run_blocking(
                    encode_chunk, video_path, start, end, duration, settings, spatial, None
                )
                state["streams"]["temporal"] = "failed"
                state["warnings"] = list(
                    dict.fromkeys(
                        [*state["warnings"], "Temporal encoding failed for one or more chunks"]
                    )
                )
            if not outputs["spatial"]:
                raise ValueError(f"No decodable spatial samples in chunk {start}-{end}ms")
            if await lifecycle.cancelled(job_id):
                return {"status": "cancelled"}
            artifact = await asyncio.to_thread(
                write_shard, root, start, outputs, settings.use_faiss
            )
            await store.save_chunk(job_id, state["generation"], start, artifact)
            state["checkpoint_ms"] = end
            state["streams"]["spatial"] = "complete" if end == duration else "processing"
            if temporal and state["streams"]["temporal"] != "failed":
                state["streams"]["temporal"] = "complete" if end == duration else "processing"
            await lifecycle.update(job_id, result_data=state, progress=end / duration)
        if await lifecycle.cancelled(job_id):
            return {"status": "cancelled"}
        if end < duration:
            await ctx["redis"].enqueue_job(
                "process_video",
                job_id,
                video_path,
                _job_id=f"ingest:{job_id}:{state['generation']}:{end}",
            )
            return {"status": "extracting", "coverage_ms": end}
        paths = await store.chunks(job_id, state["generation"])
        manifest = {
            **state,
            "duration_ms": duration,
            "shards": paths,
            "config_hash": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        }
        await asyncio.to_thread(atomic_json, root / "manifest.json", manifest)
        state["manifest"] = str(root / "manifest.json")
        await lifecycle.update(
            job_id,
            status="complete",
            progress=1.0,
            result_data=state,
            completed_at=datetime.utcnow(),
            error=None,
        )
        return {"status": "complete", "generation": state["generation"]}
    except Exception as exc:
        if await lifecycle.cancelled(job_id):
            return {"status": "cancelled"}
        logger.exception("Chunk processing failed", job_id=job_id)
        if ctx.get("job_try", 1) < 3:
            await lifecycle.update(job_id, error=str(exc))
            raise Retry(defer=30) from exc
        await lifecycle.update(job_id, status="failed", error=str(exc))
        raise


async def query_video(ctx, query_id):
    token = uuid4().hex
    store = ctx["metadata_store"]
    lease_id = f"query:{query_id}"
    if not await store.acquire_lease(lease_id, token, settings.job_timeout + 60):
        return
    try:
        return await _query_video(ctx, query_id)
    finally:
        await store.release_lease(lease_id, token)


async def _query_video(ctx, query_id):
    from src.core.models import QueryRequest
    from src.services.query import QueryService

    store = ctx["metadata_store"]
    task = await store.get_query_task(query_id)
    if not task or task["status"] in ("complete", "failed"):
        return
    try:
        await store.update_query_task(query_id, status="processing")
        service = ctx.setdefault("query_service", QueryService(store, settings))
        result = await service.query(UUID(task["job_id"]), QueryRequest(**task["request"]))
        await store.update_query_task(
            query_id, status="complete", result=result.model_dump(mode="json")
        )
    except Exception as exc:
        await store.update_query_task(query_id, status="failed", error=str(exc))
        raise


async def recover_jobs(ctx):
    for job in await ctx["metadata_store"].unfinished_jobs():
        state = job["pipeline"]
        await ctx["redis"].enqueue_job(
            "process_video",
            job["id"],
            job["video_path"],
            _job_id=f"recover:{job['id']}:{state.get('generation', 'new')}:{state.get('checkpoint_ms', 0)}:{int(datetime.utcnow().timestamp()) // 300}",
        )
    for query_id in await ctx["metadata_store"].unfinished_queries():
        await ctx["redis"].enqueue_job(
            "query_video",
            query_id,
            _job_id=f"recover-query:{query_id}:{int(datetime.utcnow().timestamp()) // 300}",
        )


async def startup(ctx):
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    store = MetadataStore()
    await store.initialize()
    ctx["metadata_store"] = store
    await recover_jobs(ctx)


async def shutdown(ctx):
    if "query_service" in ctx:
        await ctx["query_service"].close()
    await ctx["metadata_store"].close()
    for key in ("spatial", "temporal"):
        if key in ctx:
            await ctx[key].unload_model()


class WorkerSettings:
    functions = [process_video, query_video]
    cron_jobs = [cron(recover_jobs, minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55})]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 1
    job_timeout = settings.job_timeout
    max_tries = 3
    redis_settings = get_redis_settings()
