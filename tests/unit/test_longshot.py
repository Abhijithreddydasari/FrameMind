"""Selection and spending safeguards for the public benchmark adapter."""

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.longshot.benchmark import estimate, run, select


def sample(identifier="a", modality="visual", duration=60, task="event_understanding"):
    return {
        "sample_id": identifier,
        "video_id": identifier,
        "duration": duration,
        "sample_type": "single_turn",
        "task": task,
        "conversations": [
            {"role": "user", "content": "What happened?", "modalities": [modality]},
            {"role": "assistant", "content": "SECRET REFERENCE", "criteria": [{"score": 0}]},
        ],
    }


def test_visual_selection_and_reference_separation():
    cases, references, audit = select([sample(), sample("b", "speech")])
    assert len(cases) == 1
    assert "SECRET" not in json.dumps(cases)
    assert "SECRET" in json.dumps(references)
    assert audit["eligible_visual_single_turn"] == 1


def test_caps_and_inconsistent_task_priority():
    rows = [sample("a", task="audio_understanding"), sample("b", duration=120)]
    cases, _, _ = select(rows)
    assert cases[0]["id"] == "b"
    cases, _, _ = select(rows, max_minutes=0.5)
    assert cases == []
    cases, _, _ = select(rows, max_videos=2, max_questions=1)
    assert len(cases) == 1


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), True])
def test_invalid_caps(limit):
    with pytest.raises(ValueError):
        select([sample()], max_minutes=limit)


def test_duplicate_and_multiturn():
    with pytest.raises(ValueError, match="Duplicate"):
        select([sample(), sample()])
    row = sample()
    row["sample_type"] = "multi_turn"
    assert select([row])[0] == []


def test_estimate_does_not_invent_price():
    cases = select([sample()])[0]
    assert estimate(cases)["hosted_scenario_usd"] is None
    assert estimate(cases, usd_per_attempt=0.01)["hosted_scenario_usd"] == 0.03


def test_runner_rejects_paid_and_oversized_before_network(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": select([sample()])[0]}))
    args = argparse.Namespace(
        manifest=manifest, max_questions=3, backend="default", allow_paid=False
    )
    with pytest.raises(ValueError, match="allow-paid"):
        run(args)
    manifest.write_text(json.dumps({"cases": [{"id": str(i)} for i in range(4)]}))
    with pytest.raises(ValueError, match="cap exceeded"):
        run(args)


def test_runner_sends_only_question_and_keeps_unscored_results(tmp_path):
    manifest, jobs = tmp_path / "manifest.json", tmp_path / "jobs.json"
    cases = select([sample()])[0]
    cases[0]["reference"] = "NEVER SEND"
    manifest.write_text(json.dumps({"cases": cases}))
    jobs.write_text(json.dumps({"a": "00000000-0000-0000-0000-000000000001"}))
    args = argparse.Namespace(
        manifest=manifest,
        jobs=jobs,
        max_questions=3,
        backend="none",
        allow_paid=False,
        api="http://localhost:8000",
        frames=10,
        output=tmp_path / "result.json",
    )
    client = MagicMock()
    client.get.return_value.json.return_value = {"status": "complete"}
    client.post.return_value.json.return_value = {"analysis_status": "retrieval_only"}
    with patch("benchmarks.longshot.benchmark.httpx.Client") as factory:
        factory.return_value.__enter__.return_value = client
        rows = run(args)
    payload = client.post.call_args.kwargs["json"]
    assert payload == {
        "query": "What happened?",
        "max_frames": 10,
        "analysis_backend": "none",
        "use_cache": False,
    }
    assert rows[0]["score"] is None
    assert json.loads(args.output.read_text())["official_score"] is None
