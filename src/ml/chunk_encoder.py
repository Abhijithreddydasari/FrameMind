"""Encode one bounded chunk, with a rolling temporal window and small spatial batches."""

from collections import deque

import numpy as np
from PIL import Image

from src.ingest.stream import prefetched_samples
from src.ml.temporal_encoder import VideoClip


def encode_chunk(path, start_ms, end_ms, duration_ms, config, spatial, temporal=None):
    window = (
        temporal._model.config.vision_config.num_frames
        if temporal
        else config.temporal_window_frames
    )
    stride = max(1, int(window * (1 - config.temporal_stride)))
    window_ms = round(window / config.temporal_fps * 1000)
    begin = max(0, start_ms - window_ms)
    finish = min(duration_ms, end_ms + window_ms)
    rate = max(config.frame_extraction_fps, config.temporal_fps if temporal else 0)
    recent = deque()
    images, timestamps, clips = [], [], []
    output = {"spatial": [], "temporal": []}
    next_spatial = start_ms
    next_temporal = begin
    previous = None

    def flush_spatial():
        if images:
            embeddings = spatial.embed_frames(
                images, frame_indices=timestamps, timestamps_ms=timestamps
            )
            output["spatial"].extend(
                {
                    "evidence_id": f"f{e.timestamp_ms}",
                    "timestamp_ms": e.timestamp_ms,
                    "frame_index": None,
                    "embedding": e.embedding,
                }
                for e in embeddings
            )
            images.clear()
            timestamps.clear()

    def flush_temporal():
        if clips:
            embeddings = temporal.encode_clips(clips)
            output["temporal"].extend(
                {
                    "evidence_id": f"c{e.start_ms}",
                    "timestamp_ms": e.start_ms,
                    "start_ms": e.start_ms,
                    "end_ms": e.end_ms,
                    "embedding": e.embedding,
                }
                for e in embeddings
            )
            clips.clear()

    def add_clip():
        if not recent or not (start_ms <= recent[0][0] < end_ms):
            return
        frames = list(recent)
        first = frames[0][0]
        last = min(duration_ms, frames[-1][0] + round(1000 / config.temporal_fps))
        while len(frames) < window:
            frames.append(frames[-1])
        clips.append(
            VideoClip(
                index=first,
                frames=np.stack([f[1] for f in frames]),
                start_frame=first,
                end_frame=last,
                start_ms=first,
                end_ms=last,
            )
        )
        if len(clips) >= config.temporal_batch_size:
            flush_temporal()

    def temporal_slot(slot_ms, rgb):
        # Slot times describe clip intervals; source-frame PTS remain authoritative for evidence.
        recent.append((round(slot_ms), rgb))
        if len(recent) == window:
            add_clip()
            for _ in range(stride):
                recent.popleft()

    for sample in prefetched_samples(
        path,
        begin,
        finish,
        rate,
        config.decode_size,
        capacity=max(1, config.prefetch_batches * config.spatial_batch_size),
    ):
        ts = sample.timestamp_ms
        if start_ms <= ts < end_ms and ts >= next_spatial:
            images.append(Image.fromarray(sample.rgb))
            timestamps.append(ts)
            next_spatial = (
                start_ms
                + (int((ts - start_ms) * config.frame_extraction_fps / 1000) + 1)
                * 1000
                / config.frame_extraction_fps
            )
            if len(images) >= config.spatial_batch_size:
                flush_spatial()
        if temporal:
            while next_temporal <= ts:
                rgb = sample.rgb if previous is None or next_temporal == ts else previous.rgb
                temporal_slot(next_temporal, rgb)
                next_temporal += 1000 / config.temporal_fps
        previous = sample
    if temporal and previous is not None:
        while next_temporal < finish:
            temporal_slot(next_temporal, previous.rgb)
            next_temporal += 1000 / config.temporal_fps
    if temporal and finish == duration_ms:
        add_clip()
    flush_spatial()
    flush_temporal()
    return output
