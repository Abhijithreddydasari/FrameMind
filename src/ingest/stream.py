"""Bounded, PTS-based decoding. All timestamps are relative to the video stream start."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread

import av
import numpy as np


@dataclass
class Sample:
    timestamp_ms: int
    rgb: np.ndarray


def probe(path: str | Path) -> dict:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration
            else (container.duration / av.time_base if container.duration else 0)
        )
        if duration <= 0:
            raise ValueError("Video duration is unavailable; remux the recording before ingestion")
        return {
            "duration_ms": round(duration * 1000),
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate or 0),
            "frame_count": stream.frames,
        }


def samples(
    path: str | Path, start_ms: int, end_ms: int, fps: float, size: int | None = 224
) -> Iterator[Sample]:
    if fps <= 0 or end_ms <= start_ms:
        raise ValueError("Invalid sample interval or FPS")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = stream.start_time or 0
        container.seek(
            int(start_ms / 1000 / stream.time_base) + origin, stream=stream, backward=True
        )
        period = 1000 / fps
        next_ms = float(start_ms)
        last_ms = -1
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Decoded frame lacks a presentation timestamp")
            timestamp = round(float((frame.pts - origin) * stream.time_base) * 1000)
            if timestamp >= end_ms:
                break
            if timestamp < start_ms or timestamp < next_ms or timestamp <= last_ms:
                continue
            if size:
                # Bound decoded samples before they enter queues or clip windows.
                frame = frame.reformat(width=size, height=size)
            yield Sample(timestamp, frame.to_ndarray(format="rgb24"))
            last_ms = timestamp
            next_ms = start_ms + (int((timestamp - start_ms) / period) + 1) * period


def frame_at(path: str | Path, timestamp_ms: int, crop=None) -> Sample:
    sample = next(samples(path, timestamp_ms, timestamp_ms + 2000, 1, None), None)
    if sample is None:
        raise ValueError(f"No frame available at {timestamp_ms}ms")
    if crop:
        h, w = sample.rgb.shape[:2]
        x1, y1, x2, y2 = crop
        sample.rgb = sample.rgb[
            int(y1 * h) : max(int(y2 * h), int(y1 * h) + 1),
            int(x1 * w) : max(int(x2 * w), int(x1 * w) + 1),
        ]
    return sample


def fingerprint(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prefetched_samples(*args, capacity=32):
    queue = Queue(maxsize=capacity)
    stop = Event()
    sentinel = object()

    def put(value):
        while not stop.is_set():
            try:
                queue.put(value, timeout=0.1)
                return
            except Full:
                continue

    def produce():
        try:
            for sample in samples(*args):
                if stop.is_set():
                    break
                put(sample)
        except Exception as exc:
            put(exc)
        finally:
            put(sentinel)

    thread = Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            value = queue.get()
            if value is sentinel:
                break
            if isinstance(value, Exception):
                raise value
            yield value
    finally:
        stop.set()
        thread.join()
