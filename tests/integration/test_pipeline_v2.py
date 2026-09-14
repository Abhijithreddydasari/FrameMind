"""Real decoding/storage, deterministic encoder doubles, no external services."""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import av
import numpy as np
import pytest

from src.core.config import Settings
from src.core.models import FrameSource, InspectRequest, QueryRequest
from src.ingest.stream import frame_at, probe, samples
from src.ml.clip_scorer import FrameEmbedding
from src.ml.temporal_encoder import ClipEmbedding
from src.services.query import QueryService, cache_key, validate_answer
from src.storage.indexes import IndexRepository, fuse_intervals
from src.storage.metadata import MetadataStore
from src.workers import pipeline


def video(path, seconds=4, fps=8, start_seconds=0, width=64, height=48):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        for i in range(seconds * fps):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[:, :, 0 if i >= (seconds - 1) * fps else 2] = 255
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i + start_seconds * fps
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


class Spatial:
    _model = SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision"))
    max_batch = 0

    def embed_frames(self, frames, frame_indices, timestamps_ms):
        self.max_batch = max(self.max_batch, len(frames))
        return [
            FrameEmbedding(idx, np.asarray(image).mean(axis=(0, 1)).astype(np.float32) + 0.01, ts)
            for image, idx, ts in zip(frames, frame_indices, timestamps_ms, strict=False)
        ]

    def embed_text(self, query):
        return np.array([1, 0, 0], np.float32)


class Temporal:
    _model = SimpleNamespace(
        config=SimpleNamespace(
            _commit_hash="test-revision", vision_config=SimpleNamespace(num_frames=8)
        )
    )
    max_batch = 0

    def encode_clips(self, clips):
        self.max_batch = max(self.max_batch, len(clips))
        return [
            ClipEmbedding(
                c.index,
                c.frames.mean(axis=(0, 1, 2)).astype(np.float32) + 0.01,
                c.start_frame,
                c.end_frame,
                c.start_ms,
                c.end_ms,
            )
            for c in clips
        ]

    def encode_text(self, query):
        return np.array([1, 0, 0], np.float32)


@pytest.fixture
async def store(tmp_path):
    instance = MetadataStore(f"sqlite+aiosqlite:///{tmp_path / 'metadata.db'}")
    await instance.initialize()
    yield instance
    await instance.close()


@pytest.fixture
def config(tmp_path):
    return Settings(
        storage_path=tmp_path,
        chunk_seconds=1,
        decode_size=32,
        spatial_batch_size=2,
        temporal_batch_size=2,
        model_revision="test-revision",
    )


async def indexed(store, config, monkeypatch, path, ctx=None):
    monkeypatch.setattr(pipeline, "settings", config)
    job_id = uuid4()
    await store.create_job(
        job_id, {"video_path": str(path), "result_data": {"needs_reindex": False}}
    )
    ctx = ctx or {
        "metadata_store": store,
        "redis": SimpleNamespace(hset=AsyncMock(), enqueue_job=AsyncMock()),
        "spatial": Spatial(),
        "temporal": Temporal(),
        "job_try": 3,
    }
    for _ in range(10000):
        result = await pipeline.process_video(ctx, str(job_id), str(path))
        if result["status"] == "complete":
            break
    else:
        raise AssertionError("Job failed to complete")
    return job_id, ctx


def test_pts_decoding_and_tail(tmp_path):
    path = video(tmp_path / "offset.mp4", start_seconds=5)
    assert probe(path)["duration_ms"] == 4000
    frames = list(samples(path, 0, 4000, 2, 32))
    assert [f.timestamp_ms for f in frames] == list(range(0, 4000, 500))
    assert frames[-1].rgb.shape == (32, 32, 3)
    assert frame_at(path, 3050).timestamp_ms == 3125
    assert list(samples(path, 3000, 4000, 8))[-1].timestamp_ms == 3875


def test_variable_frame_rate_uses_presentation_times(tmp_path):
    path = tmp_path / "variable.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for pts in (50, 52, 59, 80):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded = list(samples(path, 0, 4000, 60, 32))
    assert [f.timestamp_ms for f in decoded] == [0, 200, 900, 3000]
    assert frame_at(path, 100).timestamp_ms == 200


async def test_full_index_restart_and_late_retrieval(store, config, monkeypatch, tmp_path):
    path = video(tmp_path / "test.mp4")
    job_id, ctx = await indexed(store, config, monkeypatch, path)
    job = await store.get_job(job_id)
    assert job["pipeline"]["checkpoint_ms"] == 4000
    assert job["pipeline"]["streams"] == {"spatial": "complete", "temporal": "complete"}
    manifest = json.loads(Path(job["pipeline"]["manifest"]).read_text())
    assert len(manifest["shards"]) == 4
    ids = []
    for shard in manifest["shards"]:
        ids.extend(
            row["evidence_id"]
            for row in json.loads((Path(shard) / "temporal.json").read_text())["rows"]
        )
    assert len(ids) == len(set(ids))
    repository = IndexRepository()
    query = np.array([1, 0, 0], dtype=np.float32)
    hits = repository.search(manifest, "spatial", query, 2)
    assert all(h["timestamp_ms"] >= 3000 for h in hits)
    cached = list(repository.cache.values())
    assert repository.search(manifest, "spatial", query, 2) == hits
    assert list(repository.cache.values())[0] is cached[0]
    assert IndexRepository().search(manifest, "spatial", query, 2) == hits
    assert ctx["spatial"].max_batch <= 2 and ctx["temporal"].max_batch <= 2
    assert (await pipeline.process_video(ctx, str(job_id), str(path)))["status"] == "complete"
    assert len(await store.chunks(str(job_id), manifest["generation"])) == 4


async def test_checkpoint_retry_after_enqueue_failure(store, config, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "settings", config)
    path = video(tmp_path / "retry.mp4")
    job_id = uuid4()
    await store.create_job(
        job_id, {"video_path": str(path), "result_data": {"needs_reindex": False}}
    )
    queue = SimpleNamespace(hset=AsyncMock(), enqueue_job=AsyncMock(side_effect=ConnectionError()))
    ctx = {
        "metadata_store": store,
        "redis": queue,
        "spatial": Spatial(),
        "temporal": Temporal(),
        "job_try": 1,
    }
    from arq import Retry

    with pytest.raises(Retry):
        await pipeline.process_video(ctx, str(job_id), str(path))
    state = (await store.get_job(job_id))["pipeline"]
    assert state["checkpoint_ms"] == 1000
    first = await store.chunks(str(job_id), state["generation"])
    queue.enqueue_job.side_effect = None
    await pipeline.process_video(ctx, str(job_id), str(path))
    assert len(await store.chunks(str(job_id), state["generation"])) == 2
    assert (await store.chunks(str(job_id), state["generation"]))[0] == first[0]


async def test_cancellation_is_terminal_and_retains_source(store, config, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "settings", config)
    path = video(tmp_path / "cancel.mp4")
    job_id = uuid4()
    await store.create_job(job_id, {"video_path": str(path)})
    await store.update_job(job_id, {"status": "cancelled"})
    await store.update_job(job_id, {"status": "complete", "progress": 1.0})
    assert (await store.get_job(job_id))["status"] == "cancelled"
    assert (
        await pipeline.process_video(
            {"metadata_store": store, "redis": None}, str(job_id), str(path)
        )
    )["status"] == "cancelled"
    assert path.exists()


async def test_temporal_failure_is_explicit(store, config, monkeypatch, tmp_path):
    temporal = Temporal()
    temporal.encode_clips = lambda _: (_ for _ in ()).throw(RuntimeError("encoder failure"))
    ctx = {
        "metadata_store": store,
        "redis": SimpleNamespace(hset=AsyncMock(), enqueue_job=AsyncMock()),
        "spatial": Spatial(),
        "temporal": temporal,
        "job_try": 3,
    }
    job_id, _ = await indexed(store, config, monkeypatch, video(tmp_path / "failure.mp4"), ctx)
    job = await store.get_job(job_id)
    assert job["pipeline"]["checkpoint_ms"] == 4000
    assert job["pipeline"]["streams"]["temporal"] == "failed"
    assert job["pipeline"]["warnings"]


async def test_legacy_metadata_requires_reindex_and_upsert(store):
    job_id = uuid4()
    await store.create_job(job_id, {"video_path": "legacy.mp4"})
    assert (await store.get_job(job_id))["pipeline"]["needs_reindex"]
    row = {"frame_index": 15, "timestamp_ms": 500, "embedding": [1.0, 0.0]}
    await store.save_embeddings(job_id, [row])
    await store.save_embeddings(job_id, [row])
    assert len(await store.get_embeddings(job_id)) == 1
    await store.initialize()


async def test_inspection_supplies_true_timestamps(store, config, monkeypatch, tmp_path):
    job_id, _ = await indexed(store, config, monkeypatch, video(tmp_path / "query.mp4"))
    config.vlm_api_key = "test"

    async def analyze(paths, prompt):
        assert len(paths) == 2
        assert "3000" in prompt and "3500" in prompt
        assert all(Path(p).exists() for p in paths)
        return SimpleNamespace(
            content=json.dumps({"answer": "Red vehicle", "evidence_ids": ["e0-f0", "e0-f1"]})
        )

    from src.vlm.client import VLMClient

    monkeypatch.setattr(VLMClient, "create", lambda: SimpleNamespace(analyze_images=analyze))
    service = QueryService(store, config)
    result = await service.query(
        job_id, InspectRequest(query="car", start_ms=3000, end_ms=4000, max_frames=2)
    )
    assert result.analysis_status == "complete" and result.frames_analyzed == 2
    assert [s.timestamp_ms for s in result.sources] == [3000, 3500]
    assert result.confidence is None


async def test_gpu_failure_does_not_call_hosted_vlm(store, config, monkeypatch, tmp_path):
    job_id, _ = await indexed(store, config, monkeypatch, video(tmp_path / "gpu.mp4"))
    config.autogaze_enabled = True
    config.autogaze_url = "https://gpu.example"
    config.autogaze_token = "test"
    config.video_max_frames = 16
    from src.vlm.client import VLMClient
    from src.vlm.video_client import RemoteVideoClient

    monkeypatch.setattr(
        RemoteVideoClient, "analyze", AsyncMock(side_effect=TimeoutError("GPU timeout"))
    )
    hosted = AsyncMock()
    monkeypatch.setattr(VLMClient, "create", hosted)
    result = await QueryService(store, config).query(
        job_id,
        InspectRequest(query="car", start_ms=3000, end_ms=4000, analysis_backend="nvila_autogaze"),
    )
    assert result.analysis_status == "failed" and result.sources
    hosted.assert_not_called()


def test_cache_and_citation_contract():
    base = QueryRequest(query="car")
    assert cache_key("job", base, "one", {}) == cache_key("job", base, "one", {})
    assert cache_key("job", base, "one", {}) != cache_key(
        "job", base.model_copy(update={"max_frames": 2}), "one", {}
    )
    assert cache_key("job", base, "one", {}) != cache_key("job", base, "two", {})
    source = FrameSource(evidence_id="e0", timestamp_ms=12000, relevance_score=1)
    with pytest.raises(ValueError):
        validate_answer('{"answer":"car","evidence_ids":["e999"]}', [source])


def test_fusion_keeps_clip_identity_and_bounds():
    spatial = [{"evidence_id": "f15", "timestamp_ms": 500}]
    temporal = [{"evidence_id": "c15", "timestamp_ms": 12000, "start_ms": 12000, "end_ms": 14000}]
    result = fuse_intervals(spatial, temporal, 15000)
    assert len(result) == 2
    assert result[1]["start_ms"] == 10000 and result[1]["end_ms"] == 15000
    assert result[1]["hits"][0]["evidence_id"] == "c15"
