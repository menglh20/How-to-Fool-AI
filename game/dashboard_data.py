"""Pure data preparation for the Streamlit observability dashboard."""

from __future__ import annotations

from collections import Counter

from evals.metrics import TraceEvaluator
from game.replay import ReplayService


def build_dashboard_snapshot(
    replay: ReplayService,
    *,
    include_sensitive: bool = False,
) -> dict:
    timeline = replay.timeline(include_sensitive=include_sensitive)
    llm = [
        event for event in timeline if event["event_type"] == "llm_response"
    ]
    tools = [
        event for event in timeline if event["event_type"] == "tool_execution"
    ]
    tool_counts = Counter(
        event["payload"].get("tool") or "(missing)" for event in tools
    )
    state_types = {
        "phase_changed",
        "round_configured",
        "choice_submitted",
        "score_updated",
        "send_budgets_reset",
    }
    return {
        "summary": replay.summary(),
        "metrics": TraceEvaluator().evaluate(replay),
        "timeline": timeline,
        "latency_rows": [
            {
                "sequence": event["sequence"],
                "agent": event["actor_id"],
                "phase": event["payload"].get("phase"),
                "latency_ms": event["payload"].get("latency_ms") or 0,
                "input_tokens": (
                    event["payload"].get("usage", {}).get("input_tokens") or 0
                ),
                "output_tokens": (
                    event["payload"].get("usage", {}).get("output_tokens") or 0
                ),
            }
            for event in llm
        ],
        "tool_counts": dict(tool_counts),
        "state_changes": [
            event for event in timeline if event["event_type"] in state_types
        ],
    }


def compare_replays(
    baseline: ReplayService,
    candidate: ReplayService,
) -> list[dict]:
    left = TraceEvaluator().evaluate(baseline)
    right = TraceEvaluator().evaluate(candidate)
    metrics = (
        "complete",
        "actions",
        "first_call_legal_rate",
        "repair_success_rate",
        "fallback_rate",
        "latency_p50_ms",
        "latency_p95_ms",
        "input_tokens",
        "output_tokens",
    )
    rows = []
    for metric in metrics:
        before = left.get(metric)
        after = right.get(metric)
        delta = (
            after - before
            if isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
            else None
        )
        rows.append({
            "metric": metric,
            "baseline": before,
            "candidate": after,
            "delta": delta,
        })
    return rows
