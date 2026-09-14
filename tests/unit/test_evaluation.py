import copy
import json
from types import SimpleNamespace

import pytest

from scripts.evaluate import compare, interval_hit, summarize


def pair():
    baseline, accelerated, scores = [], [], {}
    for i in range(100):
        case = str(i)
        row = {
            "id": case,
            "mode": "no-gaze",
            "recording": f"recording-{i % 3}",
            "split": "held_out",
            "critical": True,
            "elapsed_ms": 1000,
            "metrics": [
                {
                    "timestamps_ms": [0, 100],
                    "max_tiles": 12,
                    "sampled_frames": 2,
                    "model_revision": "a",
                    "autogaze_revision": "b",
                }
            ],
        }
        baseline.append(row)
        accelerated.append({**copy.deepcopy(row), "mode": "gaze", "elapsed_ms": 500})
        scores[case] = {
            "baseline_correct": True,
            "gaze_correct": True,
            "baseline_missed": False,
            "gaze_missed": False,
        }
    return baseline, accelerated, scores


def test_ablation_requires_matching_evidence():
    baseline, accelerated, scores = pair()
    assert compare(baseline, accelerated, scores, ("no-gaze", "gaze"))["passes"]
    accelerated[0]["metrics"][0]["timestamps_ms"] = [500]
    assert not compare(baseline, accelerated, scores, ("no-gaze", "gaze"))["passes"]


@pytest.mark.parametrize("failure", ["critical_miss", "unreviewed", "tuning", "error"])
def test_promotion_rejects_failed_gates(failure):
    baseline, accelerated, scores = pair()
    if failure == "critical_miss":
        scores["0"]["gaze_missed"] = True
    elif failure == "unreviewed":
        scores["0"]["gaze_correct"] = "true"
    elif failure == "tuning":
        baseline[0]["split"] = accelerated[0]["split"] = "tuning"
    else:
        accelerated[0]["error"] = "OOM"
    assert not compare(baseline, accelerated, scores, ("no-gaze", "gaze"))["passes"]


def test_ablation_alone_cannot_promote(tmp_path):
    baseline, accelerated, scores = pair()
    paths = {}
    for name, value in (
        ("baseline", {"rows": baseline}),
        ("accelerated", {"rows": accelerated}),
        ("scores", scores),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    output = tmp_path / "report.json"
    args = SimpleNamespace(
        **paths,
        pipeline_baseline=None,
        pipeline_accelerated=None,
        pipeline_scores=None,
        output=output,
    )
    summarize(args)
    assert not json.loads(output.read_text())["eligible_for_manual_promotion"]


def test_interval_hit_handles_nullable_frame_bounds():
    assert interval_hit([{"timestamp_ms": 123, "start_ms": None, "end_ms": None}], [[120, 130]])
