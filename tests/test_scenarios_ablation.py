"""Tests for deterministic scenarios and report comparisons."""

from pathlib import Path

import pytest

from evals.ablation import compare_reports
from evals.scenarios import run_scenarios


def test_bundled_scenarios_all_pass():
    directory = Path(__file__).resolve().parents[1] / "evals" / "scenarios"
    report = run_scenarios(directory)
    assert report["total"] >= 8
    assert report["failed"] == 0
    assert report["pass_rate"] == 1


def test_ablation_reports_real_deltas_without_claiming_significance():
    baseline = {
        "aggregate": {
            "games": 10,
            "completion_rate": 0.8,
            "fallback_rate": 0.2,
        }
    }
    candidate = {
        "aggregate": {
            "games": 10,
            "completion_rate": 0.9,
            "fallback_rate": 0.1,
        }
    }
    comparison = compare_reports(
        baseline,
        candidate,
        metrics=("completion_rate", "fallback_rate"),
    )
    assert comparison["metrics"]["completion_rate"]["delta"] == pytest.approx(0.1)
    assert comparison["metrics"]["fallback_rate"]["delta"] == pytest.approx(-0.1)
    assert "significance is not claimed" in comparison["warning"]
