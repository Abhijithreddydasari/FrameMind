"""Opt-in, genuine checkpoint shape/retrieval smoke test; does not score accuracy."""

import os

import numpy as np
import pytest
from PIL import Image


@pytest.mark.models
@pytest.mark.skipif(
    os.environ.get("FM_RUN_MODEL_TESTS") != "1", reason="Explicit checkpoint download opt-in"
)
async def test_real_encoders(tmp_path):
    from src.ml.clip_scorer import CLIPScorer
    from src.ml.temporal_encoder import VideoClip, XCLIPEncoder

    clip, xclip = CLIPScorer(), XCLIPEncoder(device="cpu", batch_size=1)
    try:
        await clip.load_model()
        await xclip.load_model()
        spatial = clip.embed_frames([Image.new("RGB", (224, 224), "red")])[0].embedding
        text = clip.embed_text("a red image")
        assert spatial.shape == text.shape and np.isfinite(spatial).all()
        count = xclip._model.config.vision_config.num_frames
        frames = np.zeros((count, 224, 224, 3), dtype=np.uint8)
        temporal = xclip.encode_clips([VideoClip(0, frames, 0, count - 1, 0, 1000)])[0].embedding
        assert temporal.shape == xclip.encode_text("a still scene").shape
        from src.core.config import Settings
        from src.ml.chunk_encoder import encode_chunk
        from src.storage.indexes import IndexRepository, write_shard
        from tests.integration.test_pipeline_v2 import video

        path = video(tmp_path / "real-model-input.mp4", seconds=2)
        config = Settings(temporal_batch_size=1, spatial_batch_size=2)
        outputs = encode_chunk(path, 0, 2000, 2000, config, clip, xclip)
        assert len(outputs["spatial"]) == 4 and len(outputs["temporal"]) > 0
        shard = write_shard(tmp_path / "index", 0, outputs)
        repository = IndexRepository()
        for stream, vector in (("spatial", text), ("temporal", xclip.encode_text("a red scene"))):
            hits = repository.search({"shards": [shard]}, stream, vector, 5)
            assert hits and all(0 <= h["timestamp_ms"] < 2000 for h in hits)
    finally:
        await clip.unload_model()
        await xclip.unload_model()
