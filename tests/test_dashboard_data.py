"""Tests for dashboard data preparation without Streamlit rendering."""

from game.dashboard_data import build_dashboard_snapshot, compare_replays
from game.replay import ReplayService
from game.trace import TraceRecorder


def replay_with(latency: int, fallback: bool):
    recorder = TraceRecorder()
    recorder.record(
        "llm_response",
        payload={
            "latency_ms": latency,
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
    )
    recorder.record(
        "tool_execution",
        payload={
            "action_id": "one",
            "tool": "choose_bottle",
            "attempt": 3 if fallback else 1,
            "fallback": fallback,
            "is_error": False,
        },
    )
    return ReplayService(recorder.snapshot())


def test_dashboard_snapshot_builds_charts_and_redacted_timeline():
    replay = replay_with(50, False)
    snapshot = build_dashboard_snapshot(replay)
    assert snapshot["tool_counts"] == {"choose_bottle": 1}
    assert snapshot["latency_rows"][0]["latency_ms"] == 50
    assert snapshot["metrics"]["actions"] == 1


def test_dashboard_comparison_uses_trace_metrics():
    rows = compare_replays(
        replay_with(100, True),
        replay_with(50, False),
    )
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["latency_p95_ms"]["delta"] == -50
    assert by_metric["fallback_rate"]["delta"] == -1
