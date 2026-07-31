"""Batch evaluation over persisted, real game traces."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from evals.intelligence import IntelligenceJudge, aggregate_intelligence
from evals.metrics import Pricing, TraceEvaluator, aggregate_reports
from evals.scenarios import run_scenarios
from game.llm_client import LLMClient
from game.replay import ReplayRepository


class EvaluationCancelled(RuntimeError):
    pass


class BatchEvaluationRunner:
    def __init__(
        self,
        repository: ReplayRepository,
        *,
        pricing: Pricing | None = None,
        scenario_directory: Path | None = None,
        judge_client: LLMClient | None = None,
        judge_max_actions: int = 50,
    ) -> None:
        if judge_max_actions < 1 or judge_max_actions > 200:
            raise ValueError("judge_max_actions must be between 1 and 200")
        self.repository = repository
        self.evaluator = TraceEvaluator(pricing)
        self.scenario_directory = scenario_directory
        self.judge = (
            IntelligenceJudge(judge_client)
            if judge_client is not None
            else None
        )
        self.judge_max_actions = judge_max_actions

    def run(
        self,
        game_ids: Iterable[str] | None = None,
        *,
        cancel_event: threading.Event | None = None,
        progress: Callable[[int, int], None] | None = None,
        experiment_label: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        ids = list(game_ids) if game_ids is not None else [
            row["game_id"]
            for row in self.repository.list_replays(limit=200)
        ]
        reports = []
        errors = []
        remaining_judge_actions = self.judge_max_actions
        for index, game_id in enumerate(ids, 1):
            if cancel_event is not None and cancel_event.is_set():
                raise EvaluationCancelled("Evaluation was cancelled")
            try:
                replay = self.repository.get(game_id)
                game_report = self.evaluator.evaluate(replay)
                if self.judge is not None and remaining_judge_actions > 0:
                    intelligence = self.judge.evaluate(
                        replay,
                        max_actions=remaining_judge_actions,
                    )
                    game_report["intelligence"] = intelligence
                    remaining_judge_actions -= intelligence[
                        "candidate_actions"
                    ]
                reports.append(game_report)
            except Exception as exc:
                errors.append({"game_id": game_id, "error": str(exc)})
            if progress:
                progress(index, len(ids))
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment": {
                "label": experiment_label,
                "metadata": metadata or {},
            },
            "config": {
                "game_ids": ids,
                "pricing_per_million_tokens": {
                    "input": self.evaluator.pricing.input_per_million,
                    "output": self.evaluator.pricing.output_per_million,
                },
            },
            "aggregate": aggregate_reports(reports),
            "games": reports,
            "errors": errors,
        }
        if self.judge is not None:
            report["config"]["judge"] = {
                "provider": self.judge.client.provider,
                "model": self.judge.client.model,
                "max_actions": self.judge_max_actions,
            }
            intelligence = aggregate_intelligence(reports)
            report["aggregate"]["intelligence"] = intelligence
            report["aggregate"]["judge_decision_score"] = intelligence[
                "decision_score"
            ]
            report["aggregate"]["judge_action_score"] = intelligence[
                "action_score"
            ]
            report["aggregate"]["intelligence_score"] = intelligence[
                "intelligence_score"
            ]
        if self.scenario_directory is not None:
            report["scenarios"] = run_scenarios(self.scenario_directory)
        return report


def write_report(report: dict, json_path: Path) -> tuple[Path, Path]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(json_path)
    markdown_path = json_path.with_suffix(".md")
    aggregate = report["aggregate"]
    lines = [
        "# Evaluation Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Experiment: {report.get('experiment', {}).get('label') or '(unlabelled)'}",
        f"- Games: {aggregate['games']}",
        f"- Completion rate: {aggregate['completion_rate']}",
        f"- First-call legal rate: {aggregate['first_call_legal_rate']}",
        f"- Repair success rate: {aggregate['repair_success_rate']}",
        f"- Fallback rate: {aggregate['fallback_rate']}",
        f"- P50 latency: {aggregate['latency_p50_ms']} ms",
        f"- P95 latency: {aggregate['latency_p95_ms']} ms",
        f"- Input tokens: {aggregate['input_tokens']}",
        f"- Output tokens: {aggregate['output_tokens']}",
        f"- Estimated cost: {aggregate['estimated_cost']}",
    ]
    intelligence = aggregate.get("intelligence")
    if intelligence is not None:
        lines.extend([
            f"- Judge decision score: {intelligence['decision_score']}",
            f"- Judge action score: {intelligence['action_score']}",
            f"- Judge intelligence score: {intelligence['intelligence_score']}",
            f"- Judge-scored actions: {intelligence['scored_actions']}",
            f"- Judge tokens: {intelligence['usage']['total_tokens']}",
        ])
    lines.extend([
        "",
        "Metrics are computed from recorded traces; no values are fabricated.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
