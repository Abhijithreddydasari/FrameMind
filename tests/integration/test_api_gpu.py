import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from src.api.deps import get_metadata_store, get_settings_dep
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.core.config import Settings
from src.gpu.app import read_bundle
from src.storage.metadata import MetadataStore
from src.workers import pipeline


@pytest.fixture
async def api(tmp_path):
    store = MetadataStore(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    await store.initialize()
    app = FastAPI()
    app.include_router(ingest_router, prefix="/ingest")
    app.include_router(query_router, prefix="/query")
    app.state.queue = SimpleNamespace(enqueue_job=AsyncMock(), hset=AsyncMock())
    app.dependency_overrides[get_metadata_store] = lambda: store
    app.dependency_overrides[get_settings_dep] = lambda: Settings(
        storage_path=tmp_path, max_video_size_mb=1
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, store, app
    await store.close()


async def test_upload_queue_failure_is_visible(api):
    client, store, app = api
    app.state.queue.enqueue_job.side_effect = ConnectionError()
    result = await client.post(
        "/ingest/upload", files={"file": ("video.mp4", b"test-video", "video/mp4")}
    )
    assert result.status_code == 503
    job_id = result.json()["detail"]["job_id"]
    job = await store.get_job(UUID(job_id))
    assert job["status"] == "failed"
    assert Path(job["video_path"]).exists()


async def test_oversize_upload_leaves_no_source(api, tmp_path):
    client, _, app = api
    result = await client.post(
        "/ingest/upload", files={"file": ("video.mp4", b"x" * (1024**2 + 1), "video/mp4")}
    )
    assert result.status_code == 413
    app.state.queue.enqueue_job.assert_not_called()
    assert not list(tmp_path.rglob("source.*"))


async def test_async_query_persists_and_completes(api, monkeypatch):
    client, store, app = api
    job_id = uuid4()
    await store.create_job(job_id, {"status": "complete", "result_data": {"needs_reindex": False}})
    result = await client.post(
        f"/query/{job_id}/async", json={"query": "car", "analysis_backend": "none"}
    )
    assert result.status_code == 202
    query_id = result.json()["query_id"]
    assert (await client.get(f"/query/result/{query_id}")).status_code == 202
    from src.core.models import QueryResponse

    response = QueryResponse(
        job_id=job_id,
        query="car",
        answer="Matches",
        frames_analyzed=0,
        processing_time_ms=1,
        sources=[],
    )
    service = SimpleNamespace(query=AsyncMock(return_value=response))
    await pipeline.query_video({"metadata_store": store, "query_service": service}, query_id)
    completed = await client.get(f"/query/result/{query_id}")
    assert completed.status_code == 200 and completed.json()["answer"] == "Matches"
    await pipeline.query_video({"metadata_store": store, "query_service": service}, query_id)
    assert service.query.await_count == 1


async def test_query_queue_failure_persisted(api):
    client, store, app = api
    job_id = uuid4()
    await store.create_job(job_id, {"status": "complete", "result_data": {"needs_reindex": False}})
    app.state.queue.enqueue_job.side_effect = ConnectionError()
    result = await client.post(f"/query/{job_id}/async", json={"query": "car"})
    assert result.status_code == 503
    assert await store.unfinished_queries() == []


async def test_competing_tasks_cannot_acquire_same_lease(api):
    _, store, _ = api
    assert await store.acquire_lease("job", "first", 30)
    assert not await store.acquire_lease("job", "second", 30)
    await store.release_lease("job", "second")
    assert not await store.acquire_lease("job", "third", 30)
    await store.release_lease("job", "first")
    assert await store.acquire_lease("job", "second", 30)


def bundle(path, entries):
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def test_gpu_bundle_rejects_paths_and_preserves_timestamps(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(image, "JPEG")
    metadata = {"timestamps_ms": [12000], "prompt": "question", "gaze": True, "max_tiles": 12}
    entries = {"metadata.json": json.dumps(metadata), "frame-0000.jpg": image.getvalue()}
    archive = tmp_path / "bundle.zip"
    bundle(archive, entries)
    result, paths = read_bundle(archive, tmp_path)
    assert result["timestamps_ms"] == [12000] and len(paths) == 1
    entries["../escape.jpg"] = image.getvalue()
    bundle(archive, entries)
    with pytest.raises(ValueError, match="Unexpected archive entries"):
        read_bundle(archive, tmp_path)


async def test_gpu_auth_no_model_load(monkeypatch):
    from src.gpu.app import app

    monkeypatch.setenv("AUTOGAZE_TOKEN", "test-token")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/health")).status_code == 401
