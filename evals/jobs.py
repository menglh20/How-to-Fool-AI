"""Bounded asynchronous evaluation jobs for MCP and dashboards."""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.metrics import Pricing
from evals.ablation import compare_reports
from evals.runner import (
    BatchEvaluationRunner,
    EvaluationCancelled,
    write_report,
)
from game.llm_client import create_client
from game.replay import ReplayRepository


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvaluationRun:
    run_id: str
    status: str
    created_at: str
    updated_at: str
    config: dict[str, Any]
    progress_completed: int = 0
    progress_total: int = 0
    report_path: str | None = None
    error: str | None = None


class EvaluationManager:
    STATUSES = {"queued", "running", "completed", "failed", "cancelled"}

    def __init__(
        self,
        repository: ReplayRepository,
        report_directory: Path,
        *,
        max_concurrent: int = 2,
        max_queued: int = 20,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.repository = repository
        self.report_directory = report_directory
        self.report_directory.mkdir(parents=True, exist_ok=True)
        self.max_queued = max_queued
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.RLock()
        self._runs: dict[str, EvaluationRun] = {}
        self._idempotency: dict[str, str] = {}
        self._cancel: dict[str, threading.Event] = {}

    def start(
        self,
        *,
        game_ids: list[str] | None = None,
        input_price: float = 0.0,
        output_price: float = 0.0,
        idempotency_key: str | None = None,
        experiment_label: str | None = None,
        metadata: dict[str, Any] | None = None,
        judge_model: str | None = None,
        judge_max_actions: int = 50,
    ) -> dict:
        if input_price < 0 or output_price < 0:
            raise ValueError("Prices cannot be negative")
        if game_ids is not None and len(game_ids) > 200:
            raise ValueError("At most 200 games may be evaluated per run")
        if experiment_label is not None and len(experiment_label) > 80:
            raise ValueError("experiment_label cannot exceed 80 characters")
        if metadata is not None and len(str(metadata)) > 2000:
            raise ValueError("metadata is too large")
        if judge_model is not None and len(judge_model) > 200:
            raise ValueError("judge_model is too long")
        if judge_max_actions < 1 or judge_max_actions > 200:
            raise ValueError("judge_max_actions must be between 1 and 200")
        with self._lock:
            if idempotency_key and idempotency_key in self._idempotency:
                return self.get(self._idempotency[idempotency_key])
            active = sum(
                run.status in {"queued", "running"}
                for run in self._runs.values()
            )
            if active >= self.max_queued:
                raise RuntimeError("Evaluation queue is full")
            run_id = uuid.uuid4().hex
            config = {
                "game_ids": list(game_ids) if game_ids is not None else None,
                "input_price": input_price,
                "output_price": output_price,
                "idempotency_key": idempotency_key,
                "experiment_label": experiment_label,
                "metadata": metadata or {},
                "judge_model": judge_model,
                "judge_max_actions": judge_max_actions,
            }
            run = EvaluationRun(
                run_id=run_id,
                status="queued",
                created_at=_now(),
                updated_at=_now(),
                config=config,
            )
            self._runs[run_id] = run
            self._cancel[run_id] = threading.Event()
            if idempotency_key:
                self._idempotency[idempotency_key] = run_id
        thread = threading.Thread(
            target=self._execute,
            args=(run_id,),
            daemon=True,
            name=f"Evaluation-{run_id[:8]}",
        )
        thread.start()
        return self.get(run_id)

    def _execute(self, run_id: str) -> None:
        cancel_event = self._cancel[run_id]
        with self._semaphore:
            if cancel_event.is_set():
                self._set_status(run_id, "cancelled")
                return
            self._set_status(run_id, "running")
            with self._lock:
                config = dict(self._runs[run_id].config)

            def progress(completed: int, total: int) -> None:
                with self._lock:
                    run = self._runs[run_id]
                    run.progress_completed = completed
                    run.progress_total = total
                    run.updated_at = _now()

            try:
                judge_client = (
                    create_client(config["judge_model"])
                    if config["judge_model"]
                    else None
                )
                if judge_client is not None:
                    judge_client.record_calls = False
                runner = BatchEvaluationRunner(
                    self.repository,
                    pricing=Pricing(
                        config["input_price"],
                        config["output_price"],
                    ),
                    scenario_directory=Path(__file__).resolve().parent
                    / "scenarios",
                    judge_client=judge_client,
                    judge_max_actions=config["judge_max_actions"],
                )
                report = runner.run(
                    config["game_ids"],
                    cancel_event=cancel_event,
                    progress=progress,
                    experiment_label=config["experiment_label"],
                    metadata=config["metadata"],
                )
                report["run_id"] = run_id
                output = self.report_directory / f"{run_id}.json"
                write_report(report, output)
                with self._lock:
                    run = self._runs[run_id]
                    run.report_path = str(output)
                self._set_status(run_id, "completed")
            except EvaluationCancelled:
                self._set_status(run_id, "cancelled")
            except Exception as exc:
                with self._lock:
                    self._runs[run_id].error = str(exc)
                self._set_status(run_id, "failed")

    def _set_status(self, run_id: str, status: str) -> None:
        if status not in self.STATUSES:
            raise ValueError(f"Unknown status: {status}")
        with self._lock:
            run = self._runs[run_id]
            run.status = status
            run.updated_at = _now()

    def get(self, run_id: str) -> dict:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Unknown evaluation run: {run_id}")
            return asdict(self._runs[run_id])

    def list(self) -> list[dict]:
        with self._lock:
            return [
                asdict(run)
                for run in sorted(
                    self._runs.values(),
                    key=lambda item: item.created_at,
                    reverse=True,
                )
            ]

    def cancel(self, run_id: str) -> dict:
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"Unknown evaluation run: {run_id}")
            run = self._runs[run_id]
            if run.status not in {"queued", "running"}:
                return asdict(run)
            self._cancel[run_id].set()
            if run.status == "queued":
                run.status = "cancelled"
                run.updated_at = _now()
            return asdict(run)

    def report(self, run_id: str) -> dict:
        snapshot = self.get(run_id)
        if snapshot["status"] != "completed" or not snapshot["report_path"]:
            raise RuntimeError("Evaluation report is not ready")
        import json

        return json.loads(
            Path(snapshot["report_path"]).read_text(encoding="utf-8")
        )

    def compare(self, baseline_run_id: str, candidate_run_id: str) -> dict:
        return compare_reports(
            self.report(baseline_run_id),
            self.report(candidate_run_id),
        )
