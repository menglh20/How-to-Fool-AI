"""Compare real evaluation reports without inventing conclusions."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_METRICS = (
    "completion_rate",
    "first_call_legal_rate",
    "repair_success_rate",
    "fallback_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "llm_errors",
    "input_tokens",
    "output_tokens",
    "estimated_cost",
    "judge_decision_score",
    "judge_action_score",
    "intelligence_score",
)


def compare_reports(
    baseline: dict,
    candidate: dict,
    metrics=DEFAULT_METRICS,
) -> dict:
    left = baseline["aggregate"]
    right = candidate["aggregate"]
    comparisons = {}
    for metric in metrics:
        before = left.get(metric)
        after = right.get(metric)
        delta = (
            after - before
            if isinstance(before, (int, float))
            and isinstance(after, (int, float))
            else None
        )
        relative = (
            delta / abs(before)
            if delta is not None and before not in {0, None}
            else None
        )
        comparisons[metric] = {
            "baseline": before,
            "candidate": after,
            "delta": delta,
            "relative_change": relative,
        }
    return {
        "baseline_games": left.get("games"),
        "candidate_games": right.get("games"),
        "metrics": comparisons,
        "warning": (
            "Deltas are descriptive only; statistical significance is not "
            "claimed."
        ),
    }


def compare_files(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    return compare_reports(baseline, candidate)
