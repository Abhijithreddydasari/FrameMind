"""Immutable vector shards and atomically published index generations."""

import json
import os
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

import numpy as np


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_shard(root: Path, start_ms: int, streams: dict, use_faiss=True) -> str:
    directory = root / f"chunk-{start_ms:012d}-{uuid4().hex}"
    directory.mkdir(parents=True)
    counts = {}
    for name, rows in streams.items():
        counts[name] = len(rows)
        if not rows:
            continue
        vectors = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Invalid embedding matrix")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Zero embedding vector")
        vectors /= norms
        np.save(directory / f"{name}.npy", vectors, allow_pickle=False)
        atomic_json(
            directory / f"{name}.json",
            {"rows": [{k: v for k, v in row.items() if k != "embedding"} for row in rows]},
        )
        if use_faiss:
            import faiss

            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            faiss.write_index(index, str(directory / f"{name}.faiss"))
    atomic_json(directory / "shard.json", {"counts": counts, "start_ms": start_ms})
    return str(directory)


class IndexRepository:
    def __init__(self, cache_bytes=256 * 1024**2):
        self.cache_bytes = cache_bytes
        self.cache = OrderedDict()
        self.bytes = 0

    def _load(self, directory: Path, stream: str):
        key = (str(directory), stream)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        path = directory / f"{stream}.npy"
        if not path.exists():
            return None
        vectors = np.load(path, mmap_mode="r", allow_pickle=False)
        rows = json.loads((directory / f"{stream}.json").read_text())["rows"]
        index = None
        if (directory / f"{stream}.faiss").exists():
            import faiss

            index = faiss.read_index(str(directory / f"{stream}.faiss"))
        value = (vectors, rows, index)
        size = path.stat().st_size * 2 + (directory / f"{stream}.json").stat().st_size * 4
        while self.cache and self.bytes + size > self.cache_bytes:
            _, (_, old_size) = self.cache.popitem(last=False)
            self.bytes -= old_size
        if size <= self.cache_bytes:
            self.cache[key] = (value, size)
            self.bytes += size
        return value

    def search(self, manifest: dict, stream: str, query: np.ndarray, k=50) -> list[dict]:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or norm <= 0:
            raise ValueError("Invalid query embedding")
        query /= norm
        hits = []
        for path in manifest["shards"]:
            loaded = self._load(Path(path), stream)
            if loaded is None:
                continue
            vectors, rows, index = loaded
            if vectors.shape[1] != query.size:
                raise ValueError("Query/index model dimension mismatch; reindex required")
            count = min(k, len(rows))
            if index is not None:
                scores, ids = index.search(query[None], count)
                pairs = zip(ids[0], scores[0], strict=False)
            else:
                similarities = vectors @ query
                ids = np.argsort(similarities)[-count:][::-1]
                pairs = ((i, similarities[i]) for i in ids)
            hits.extend(
                {**rows[int(i)], "score": float(score), "stream": stream} for i, score in pairs
            )
            hits.sort(key=lambda h: (-h["score"], h["evidence_id"]))
            hits = hits[:k]
        return hits


def fuse_intervals(
    spatial: list[dict], temporal: list[dict], duration_ms: int, alpha=0.5, limit=5, context_ms=2000
) -> list[dict]:
    candidates = []
    for hits, weight in (
        (spatial, alpha if temporal else 1),
        (temporal, 1 - alpha if spatial else 1),
    ):
        for rank, hit in enumerate(hits, 1):
            if weight <= 0:
                continue
            start = hit.get("start_ms", hit["timestamp_ms"])
            end = hit.get("end_ms", start + 1)
            candidates.append(
                {
                    "start_ms": max(0, start - context_ms),
                    "end_ms": min(duration_ms, end + context_ms),
                    "score": weight / (60 + rank),
                    "hits": [hit],
                    "stream_scores": {
                        hit.get("stream", "temporal" if hits is temporal else "spatial"): weight
                        / (60 + rank)
                    },
                }
            )
    # Merge temporal overlap before ranking, so duplicate frames cannot consume the budget.
    candidates.sort(key=lambda c: c["start_ms"])
    merged = []
    for candidate in candidates:
        if (
            merged
            and candidate["start_ms"] <= merged[-1]["end_ms"]
            and max(candidate["end_ms"], merged[-1]["end_ms"]) - merged[-1]["start_ms"] <= 60000
        ):
            previous = merged[-1]
            previous["end_ms"] = max(previous["end_ms"], candidate["end_ms"])
            for stream, score in candidate["stream_scores"].items():
                previous["stream_scores"][stream] = max(
                    previous["stream_scores"].get(stream, 0), score
                )
            previous["score"] = sum(previous["stream_scores"].values())
            previous["hits"].extend(candidate["hits"])
        else:
            merged.append(candidate)
    return sorted(merged, key=lambda c: (-c["score"], c["start_ms"]))[:limit]
