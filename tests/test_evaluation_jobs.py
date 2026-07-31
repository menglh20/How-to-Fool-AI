"""Tests for bounded asynchronous evaluation operations."""

import time

from evals.jobs import EvaluationManager
from game.mcp_tools import MCPToolService
from game.replay import ReplayRepository
from game.trace import TraceRecorder


def wait_terminal(manager: EvaluationManager, run_id: str) -> dict:
    deadline = time.time() + 3
    while time.time() < deadline:
        status = manager.get(run_id)
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(0.01)
    raise AssertionError("evaluation did not finish")


def completed_game(tmp_path):
    recorder = TraceRecorder.persistent(tmp_path)
    recorder.record(
        "game_event",
        payload={
            "event_type": "game_over",
            "event_payload": {"final_scores": {"human": 1}},
        },
    )
    return recorder


def test_evaluation_job_is_idempotent_and_persists_report(tmp_path):
    recorder = completed_game(tmp_path / "traces")
    manager = EvaluationManager(
        ReplayRepository(tmp_path / "traces"),
        tmp_path / "reports",
    )
    first = manager.start(
        game_ids=[recorder.game_id],
        idempotency_key="same-request",
        experiment_label="candidate",
        metadata={"variant": "memory-on"},
    )
    second = manager.start(
        game_ids=[recorder.game_id],
        idempotency_key="same-request",
    )
    assert first["run_id"] == second["run_id"]
    final = wait_terminal(manager, first["run_id"])
    assert final["status"] == "completed"
    report = manager.report(first["run_id"])
    assert report["aggregate"]["completion_rate"] == 1
    assert report["run_id"] == first["run_id"]
    assert report["experiment"] == {
        "label": "candidate",
        "metadata": {"variant": "memory-on"},
    }


def test_evaluation_mcp_operations_do_not_block(tmp_path):
    recorder = completed_game(tmp_path / "traces")
    repository = ReplayRepository(tmp_path / "traces")
    manager = EvaluationManager(repository, tmp_path / "reports")
    service = MCPToolService(repository, manager)
    started = service.start_evaluation(game_ids=[recorder.game_id])
    assert started["status"] in {"queued", "running", "completed"}
    final = wait_terminal(manager, started["run_id"])
    assert service.get_evaluation_status(started["run_id"]) == final
    assert service.get_evaluation_report(
        started["run_id"]
    )["aggregate"]["games"] == 1


def test_evaluation_validates_limits(tmp_path):
    manager = EvaluationManager(
        ReplayRepository(tmp_path / "traces"),
        tmp_path / "reports",
    )
    try:
        manager.start(game_ids=["a" * 32] * 201)
    except ValueError as exc:
        assert "200" in str(exc)
    else:
        raise AssertionError("oversized evaluation should fail")
