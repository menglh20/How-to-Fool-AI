"""Tests for trace-derived batch evaluation metrics."""

import json
import threading

import pytest

from evals.metrics import Pricing, TraceEvaluator
from evals.runner import (
    BatchEvaluationRunner,
    EvaluationCancelled,
    write_report,
)
from game.replay import ReplayRepository, ReplayService
from game.trace import TraceRecorder


def evaluation_trace(tmp_path):
    recorder = TraceRecorder.persistent(tmp_path)
    for latency, input_tokens, output_tokens in [
        (10, 100, 10),
        (20, 200, 20),
        (100, 300, 30),
    ]:
        recorder.record(
            "llm_response",
            payload={
                "latency_ms": latency,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                "error": None,
            },
        )
    recorder.record(
        "tool_execution",
        payload={
            "action_id": "a",
            "attempt": 1,
            "is_error": False,
            "fallback": False,
        },
    )
    recorder.record(
        "tool_execution",
        payload={
            "action_id": "b",
            "attempt": 1,
            "is_error": True,
            "fallback": False,
        },
    )
    recorder.record(
        "tool_execution",
        payload={
            "action_id": "b",
            "attempt": 2,
            "is_error": False,
            "fallback": False,
        },
    )
    recorder.record(
        "tool_execution",
        payload={
            "action_id": "c",
            "attempt": 1,
            "is_error": True,
            "fallback": False,
        },
    )
    recorder.record(
        "tool_execution",
        payload={
            "action_id": "c",
            "attempt": 3,
            "is_error": False,
            "fallback": True,
        },
    )
    recorder.record(
        "game_event",
        payload={
            "event_type": "game_over",
            "event_payload": {"final_scores": {"human": 2, "ai_0": 1}},
        },
    )
    return recorder


def test_trace_evaluator_computes_real_metrics(tmp_path):
    recorder = evaluation_trace(tmp_path)
    report = TraceEvaluator(Pricing(1.0, 2.0)).evaluate(
        ReplayService.from_jsonl(recorder.path)
    )
    assert report["complete"]
    assert report["first_call_legal_rate"] == pytest.approx(1 / 3)
    assert report["repair_success_rate"] == pytest.approx(1 / 2)
    assert report["fallback_rate"] == pytest.approx(1 / 3)
    assert report["latency_p50_ms"] == 20
    assert report["latency_p95_ms"] == 100
    assert report["input_tokens"] == 600
    assert report["output_tokens"] == 60
    assert report["estimated_cost"] == pytest.approx(0.00072)
    assert report["winners"] == ["human"]


def test_batch_runner_and_report_files(tmp_path):
    recorder = evaluation_trace(tmp_path / "traces")
    runner = BatchEvaluationRunner(ReplayRepository(tmp_path / "traces"))
    report = runner.run(
        [recorder.game_id],
        experiment_label="structured-tools",
        metadata={"commit": "abc123"},
    )
    assert report["aggregate"]["games"] == 1
    assert report["aggregate"]["completion_rate"] == 1
    assert report["experiment"]["label"] == "structured-tools"
    assert report["experiment"]["metadata"]["commit"] == "abc123"
    assert report["config"]["game_ids"] == [recorder.game_id]
    json_path, markdown_path = write_report(
        report, tmp_path / "report.json"
    )
    assert json.loads(json_path.read_text())["schema_version"] == 1
    assert "no values are fabricated" in markdown_path.read_text()


def test_batch_runner_honours_cancellation(tmp_path):
    runner = BatchEvaluationRunner(ReplayRepository(tmp_path))
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(EvaluationCancelled):
        runner.run(["a" * 32], cancel_event=cancelled)
