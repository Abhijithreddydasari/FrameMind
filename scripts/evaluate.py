"""Manifest-driven evaluation. No downloads, GPU provisioning, or automatic promotion.

Modes: retrieve, default, pipeline-gaze, gaze, no-gaze. Gaze/no-gaze use the same annotated
intervals and exactly the same sampled evidence. Score answers with reviewed JSON.
"""

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from src.core.config import settings
from src.core.models import FrameSource
from src.ingest.stream import frame_at
from src.services.query import evidence_prompt, validate_answer
from src.vlm.video_client import RemoteVideoClient


def interval_hit(sources, intervals):
    return any(
        max(
            source.get("start_ms")
            if source.get("start_ms") is not None
            else source["timestamp_ms"],
            start,
        )
        < min(
            source.get("end_ms")
            if source.get("end_ms") is not None
            else source["timestamp_ms"] + 1,
            end,
        )
        for source in sources[:10]
        for start, end in intervals
    )


async def run(args):
    manifest = json.loads(args.manifest.read_text())
    if not 1 <= args.frames <= 1024:
        raise ValueError("frames must be between 1 and 1024")
    cases = manifest["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Case IDs must be unique")
    if any(
        len({c["split"] for c in cases if c["video_path"] == path}) > 1
        for path in {c["video_path"] for c in cases}
    ):
        raise ValueError("Split by recording: one video cannot appear in both splits")
    rows = []
    async with httpx.AsyncClient(timeout=1800) as client:
        for case in manifest["cases"]:
            if case["split"] != args.split:
                continue
            started = time.perf_counter()
            row = {
                "id": case["id"],
                "mode": args.mode,
                "recording": case["video_path"],
                "critical": case.get("critical", False),
                "split": case["split"],
            }
            try:
                if args.mode in ("retrieve", "default", "pipeline-gaze"):
                    response = await client.post(
                        args.api.rstrip("/") + f"/api/v1/query/{case['job_id']}",
                        json={
                            "query": case["question"],
                            "max_frames": 10,
                            "analysis_backend": {
                                "retrieve": "none",
                                "default": "default",
                                "pipeline-gaze": "nvila_autogaze",
                            }[args.mode],
                            "use_cache": False,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()
                    row.update(
                        result=result,
                        recall_at_10=interval_hit(result["sources"], case["intervals"]),
                    )
                    if args.mode != "retrieve" and result["analysis_status"] not in (
                        "complete",
                        "insufficient_evidence",
                    ):
                        raise ValueError(f"Analysis did not complete: {result['analysis_status']}")
                else:
                    # Fixed annotated intervals isolate inference; these are NOT retrieval results.
                    answers, metrics = [], []
                    for i, (start, end) in enumerate(case["intervals"]):
                        if end - start > 60000:
                            raise ValueError("Benchmark intervals must be at most 60 seconds")
                        source = FrameSource(
                            evidence_id=f"e{i}",
                            timestamp_ms=start,
                            start_ms=start,
                            end_ms=end,
                            source_kind="clip",
                            relevance_score=1,
                        )
                        with tempfile.TemporaryDirectory() as temp:
                            paths = []
                            for target in np.linspace(
                                start, end, args.frames, endpoint=False
                            ).astype(int):
                                frame = frame_at(case["video_path"], int(target))
                                if (
                                    frame.timestamp_ms >= end
                                    or frame.timestamp_ms in source.frame_timestamps_ms
                                ):
                                    continue
                                path = Path(temp) / f"{frame.timestamp_ms}.jpg"
                                Image.fromarray(frame.rgb).save(path, quality=95)
                                paths.append(path)
                                source.frame_timestamps_ms.append(frame.timestamp_ms)
                            result = await RemoteVideoClient(settings).analyze(
                                paths,
                                source,
                                evidence_prompt(case["question"], [source]),
                                gaze=args.mode == "gaze",
                            )
                            answer, citations = validate_answer(result["content"], [source])
                            answers.append(
                                {"answer": answer, "sources": [s.model_dump() for s in citations]}
                            )
                            metrics.append(
                                {
                                    **result["metrics"],
                                    "timestamps_ms": source.frame_timestamps_ms,
                                    "max_tiles": settings.video_max_tiles,
                                }
                            )
                    row.update(answers=answers, metrics=metrics)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            rows.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"mode": args.mode, "rows": rows}, indent=2))
    return rows


def compare(baseline, accelerated, scores, expected_modes):
    left, right = {r["id"]: r for r in baseline}, {r["id"]: r for r in accelerated}
    if len(left) != len(baseline) or len(right) != len(accelerated):
        raise ValueError("Duplicate case IDs in run")
    if left.keys() != right.keys() or not left:
        raise ValueError("Paired runs must contain exactly the same nonempty case set")
    ids = list(left)
    for i in ids:
        if (left[i]["mode"], right[i]["mode"]) != expected_modes:
            raise ValueError("Incorrect comparison modes")
        if any(left[i].get(k) != right[i].get(k) for k in ("recording", "split", "critical")):
            raise ValueError("Paired case metadata differs")
    fields = ("baseline_correct", "gaze_correct", "baseline_missed", "gaze_missed")
    reviewed = all(i in scores and all(type(scores[i].get(k)) is bool for k in fields) for i in ids)
    failures = [i for i in ids if left[i].get("error") or right[i].get("error")]
    held_out = all(left[i]["split"] == "held_out" for i in ids)
    recordings = len({left[i]["recording"] for i in ids})
    speedup = float(
        np.percentile([left[i]["elapsed_ms"] for i in ids], 95)
        / max(1, np.percentile([right[i]["elapsed_ms"] for i in ids], 95))
    )
    drop = (
        statistics.mean(
            int(scores[i]["baseline_correct"]) - int(scores[i]["gaze_correct"]) for i in ids
        )
        if reviewed
        else None
    )
    critical_misses = [
        i
        for i in ids
        if right[i].get("critical")
        and reviewed
        and not scores[i]["baseline_missed"]
        and scores[i]["gaze_missed"]
    ]
    matched_evidence = True
    if expected_modes == ("no-gaze", "gaze"):
        keys = (
            "timestamps_ms",
            "max_tiles",
            "sampled_frames",
            "model_revision",
            "autogaze_revision",
        )
        for i in ids:
            lm, rm = left[i].get("metrics", []), right[i].get("metrics", [])
            if not lm or len(lm) != len(rm):
                matched_evidence = False
                continue
            for before, after in zip(lm, rm, strict=True):
                if any(k not in before or k not in after or before[k] != after[k] for k in keys):
                    matched_evidence = False
    passes = (
        len(ids) >= 100
        and recordings >= 3
        and held_out
        and reviewed
        and not failures
        and not critical_misses
        and matched_evidence
        and speedup >= 1.25
        and drop <= 0.02
    )
    return {
        "paired_queries": len(ids),
        "recordings": recordings,
        "held_out_only": held_out,
        "p95_speedup": speedup,
        "accuracy_drop": drop,
        "failures": failures,
        "additional_critical_misses": critical_misses,
        "manual_review_complete": reviewed,
        "matched_evidence_and_model": matched_evidence,
        "passes": bool(passes),
    }


def summarize(args):
    def rows(path):
        return json.loads(path.read_text())["rows"]

    ablation = compare(
        rows(args.baseline),
        rows(args.accelerated),
        json.loads(args.scores.read_text()),
        ("no-gaze", "gaze"),
    )
    pipeline = None
    paths = (args.pipeline_baseline, args.pipeline_accelerated, args.pipeline_scores)
    if any(paths) and not all(paths):
        raise ValueError("Supply all three pipeline comparison files")
    if all(paths):
        pipeline = compare(
            rows(args.pipeline_baseline),
            rows(args.pipeline_accelerated),
            json.loads(args.pipeline_scores.read_text()),
            ("default", "pipeline-gaze"),
        )
    result = {
        "inference_ablation": ablation,
        "end_to_end": pipeline,
        "eligible_for_manual_promotion": bool(
            ablation["passes"] and pipeline and pipeline["passes"]
        ),
        "note": "Eligibility never changes configuration. Review failures, per-question accuracy, footage coverage and the checkpoint license before manual promotion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("manifest", type=Path)
    runner.add_argument(
        "--mode", choices=["retrieve", "default", "pipeline-gaze", "gaze", "no-gaze"], required=True
    )
    runner.add_argument("--split", choices=["tuning", "held_out"], default="held_out")
    runner.add_argument("--api", default="http://127.0.0.1:8000")
    runner.add_argument("--frames", type=int, default=128)
    runner.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    for name in ("baseline", "accelerated", "scores", "output"):
        report.add_argument(f"--{name}", type=Path, required=True)
    for name in ("pipeline-baseline", "pipeline-accelerated", "pipeline-scores"):
        report.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args))
    else:
        summarize(args)


if __name__ == "__main__":
    main()
