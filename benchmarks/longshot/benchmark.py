"""Prepare official annotations and run a small, unscored visual pilot."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from uuid import UUID

import httpx

REPO = "MBZUAI/longshot-bench"
REVISION = "5a16213fadcbbbacd40b6feb6e5b048e5c180ddc"
FILENAME = "postvalid_v2_test.jsonl"
ROOT = Path("data/benchmarks/longshot")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def positive(value):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("Limits must be finite and positive")
    return value


def select(rows, max_videos=1, max_questions=3, max_minutes=45):
    for limit in (max_videos, max_questions, max_minutes):
        positive(limit)
    if type(max_videos) is not int or type(max_questions) is not int:
        raise ValueError("Video and question limits must be integers")
    eligible = []
    seen = set()
    for row in rows:
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(row["sample_id"])
        users = [t for t in row["conversations"] if t["role"] == "user"]
        if (
            row["sample_type"] == "single_turn"
            and len(users) == 1
            and set(users[0].get("modalities", [])) == {"visual"}
        ):
            positive(row["duration"])
            eligible.append((row, users[0]))
    cases, references, videos = [], {}, {}
    # Prefer consistent visual task labels, then shorter recordings. Stable ties by ID.
    eligible.sort(
        key=lambda item: (
            "audio" in str(item[0].get("task", "")).lower(),
            item[0]["duration"],
            item[0]["sample_id"],
        )
    )
    for row, user in eligible:
        vid = row["video_id"]
        if vid not in videos and (
            len(videos) >= max_videos or sum(videos.values()) + row["duration"] > max_minutes * 60
        ):
            continue
        if len(cases) >= max_questions:
            break
        videos[vid] = row["duration"]
        cases.append(
            {
                "id": row["sample_id"],
                "video_id": vid,
                "duration_seconds": row["duration"],
                "question": user["content"],
            }
        )
        references[row["sample_id"]] = {
            "task": row.get("task"),
            "metadata_warning": "audio task labelled visual"
            if "audio" in str(row.get("task", "")).lower()
            else None,
            "reference_turns": [t for t in row["conversations"] if t["role"] == "assistant"],
        }
    return (
        cases,
        references,
        {
            "source_samples": len(rows),
            "eligible_visual_single_turn": len(eligible),
            "selected_questions": len(cases),
            "selected_videos": len(videos),
            "source_minutes": sum(videos.values()) / 60,
        },
    )


def estimate(cases, frames=10, usd_per_attempt=None):
    positive(frames)
    videos = {c["video_id"]: c["duration_seconds"] for c in cases}
    if usd_per_attempt is not None:
        positive(usd_per_attempt)
    return {
        "questions": len(cases),
        "videos": len(videos),
        "source_minutes": sum(videos.values()) / 60,
        "max_selected_frames": len(cases) * frames,
        "hosted_attempt_allowance": len(cases) * 3,
        "assumed_usd_per_attempt": usd_per_attempt,
        "hosted_scenario_usd": None
        if usd_per_attempt is None
        else len(cases) * 3 * usd_per_attempt,
        "note": "Scenario only, not a billing cap. Excludes indexing, GPU hosting and grading. Retrieval mode makes no VLM calls.",
    }


def prepare(args):
    raw = args.annotations.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    cases, references, audit = select(rows, args.max_videos, args.max_questions, args.max_minutes)
    provenance = {
        "dataset": REPO,
        "revision": args.revision,
        "annotations_sha256": hashlib.sha256(raw).hexdigest(),
    }
    write_json(args.output / "manifest.json", {"provenance": provenance, "cases": cases})
    write_json(args.output / "references.json", references)
    write_json(args.output / "audit.json", {**provenance, **audit})
    write_json(args.output / "estimate.json", estimate(cases, args.frames, args.usd_per_attempt))
    print(json.dumps(audit, indent=2))


def run(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    positive(args.max_questions)
    if not cases or len(cases) > args.max_questions:
        raise ValueError("Empty pilot or question cap exceeded; prepare a smaller subset")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    if args.backend != "none" and not args.allow_paid:
        raise ValueError("Analysis requires --allow-paid after reviewing the estimate")
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    # Validate all mappings before making any inference requests.
    ids = {c["video_id"]: str(UUID(jobs[c["video_id"]])) for c in cases}
    rows = []
    with httpx.Client(timeout=1800) as client:
        for job in set(ids.values()):
            status = client.get(f"{args.api.rstrip('/')}/api/v1/ingest/status/{job}")
            status.raise_for_status()
            if status.json()["status"] != "complete":
                raise ValueError(f"Job {job} is not complete")
        for case in cases:
            started = time.perf_counter()
            row = {"id": case["id"], "video_id": case["video_id"], "score": None}
            try:
                response = client.post(
                    f"{args.api.rstrip('/')}/api/v1/query/{ids[case['video_id']]}",
                    json={
                        "query": case["question"],
                        "max_frames": args.frames,
                        "analysis_backend": args.backend,
                        "use_cache": False,
                    },
                )
                response.raise_for_status()
                row["result"] = response.json()
            except httpx.HTTPError as exc:
                row["error"] = str(exc)
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            rows.append(row)
            write_json(
                args.output,
                {
                    "provenance": manifest.get("provenance"),
                    "backend": args.backend,
                    "official_score": None,
                    "rows": rows,
                },
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--output", type=Path, default=ROOT / "download")
    fetch.add_argument("--revision", default=REVISION)
    prep = commands.add_parser("prepare")
    prep.add_argument("--annotations", type=Path, default=ROOT / "download" / FILENAME)
    prep.add_argument("--revision", default=REVISION, help="Revision of the input annotations")
    prep.add_argument("--output", type=Path, default=ROOT / "pilot")
    prep.add_argument("--max-videos", type=int, default=1)
    prep.add_argument("--max-questions", type=int, default=3)
    prep.add_argument("--max-minutes", type=float, default=45)
    prep.add_argument("--usd-per-attempt", type=float)
    runner = commands.add_parser("run")
    runner.add_argument("--manifest", type=Path, default=ROOT / "pilot/manifest.json")
    runner.add_argument("--jobs", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--api", default="http://127.0.0.1:8000")
    runner.add_argument("--backend", choices=["none", "default", "nvila_autogaze"], default="none")
    runner.add_argument("--allow-paid", action="store_true")
    runner.add_argument("--max-questions", type=int, default=3)
    for command in (prep, runner):
        command.add_argument("--frames", type=int, choices=range(1, 51), default=10)
    args = parser.parse_args()
    if args.command == "fetch":
        from huggingface_hub import get_token, hf_hub_download

        print(
            hf_hub_download(
                REPO,
                FILENAME,
                repo_type="dataset",
                revision=args.revision,
                local_dir=args.output,
                token=get_token() or False,
            )
        )
    elif args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
