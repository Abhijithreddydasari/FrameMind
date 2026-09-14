"""Opt-in resource acceptance with real two-hour 1080p decoding and deterministic encoders."""

# ruff: noqa: F811 -- imported fixtures are intentionally injected by pytest
import json
import os
import threading
import time

import av
import numpy as np
import pytest

from tests.integration.test_pipeline_v2 import config, indexed, store  # noqa: F401


@pytest.mark.benchmark
@pytest.mark.skipif(
    os.environ.get("FM_RUN_RESOURCE_TEST") != "1", reason="Opt-in two-hour resource test"
)
async def test_two_hour_1080p(store, config, tmp_path, monkeypatch):
    import psutil

    path = tmp_path / "two-hours.mp4"
    # 2 FPS source keeps this reproducible on CPU; this is not a 30 FPS throughput claim.
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=2)
        stream.width, stream.height = 1920, 1080
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "30", "threads": "2"}
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        rgb[:, :, 2] = 255
        for i in range(14400):
            if i == 14380:
                rgb[:, :, 2] = 0
                rgb[:, :, 0] = 255
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    del rgb, frame
    config.chunk_seconds = 60
    config.decode_size = 224
    config.spatial_batch_size = 32
    config.temporal_batch_size = 8
    peak = [0]
    stop = threading.Event()
    process = psutil.Process()

    def watch():
        while not stop.wait(0.05):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=watch)
    thread.start()
    started = time.perf_counter()
    try:
        job_id, _ = await indexed(store, config, monkeypatch, path)
    finally:
        stop.set()
        thread.join()
    job = await store.get_job(job_id)
    assert job["pipeline"]["checkpoint_ms"] == 7200000
    from pathlib import Path

    from src.storage.indexes import IndexRepository

    manifest = json.loads(Path(job["pipeline"]["manifest"]).read_text())
    hits = IndexRepository().search(manifest, "spatial", np.array([1, 0, 0], np.float32), 10)
    assert all(hit["timestamp_ms"] >= 7190000 for hit in hits)
    assert peak[0] < 4 * 1024**3
    report = {
        "duration_seconds": 7200,
        "source_fps": 2,
        "resolution": "1920x1080",
        "encoders": "deterministic test doubles",
        "peak_rss_bytes": peak[0],
        "ingestion_seconds": round(time.perf_counter() - started, 2),
        "coverage_ms": job["pipeline"]["checkpoint_ms"],
        "late_event_retrieved": True,
    }
    output = Path(os.environ.get("FM_RESOURCE_REPORT", "data/resource-report.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
