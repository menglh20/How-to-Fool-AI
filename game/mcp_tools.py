"""Application services exposed through the local MCP server."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from game.replay import ReplayRepository
from game.security import redact_trace_payload

if TYPE_CHECKING:
    from evals.jobs import EvaluationManager


def _page(offset: int, limit: int, *, maximum: int = 500) -> None:
    if offset < 0:
        raise ValueError("offset cannot be negative")
    if limit < 1 or limit > maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")


class MCPToolService:
    """Stable, transport-independent implementation of MCP tools."""

    def __init__(
        self,
        replay_repository: ReplayRepository,
        evaluation_manager: EvaluationManager | None = None,
    ) -> None:
        self.replays = replay_repository
        self.evaluations = evaluation_manager

    def list_replays(self, offset: int = 0, limit: int = 50) -> dict:
        rows = self.replays.list_replays(offset=offset, limit=limit)
        return {"items": rows, "offset": offset, "limit": limit}

    def get_replay_summary(self, game_id: str) -> dict:
        return self.replays.get(game_id).summary()

    def get_replay_events(
        self,
        game_id: str,
        event_type: str | None = None,
        actor_id: str | None = None,
        round: int | None = None,
        offset: int = 0,
        limit: int = 100,
        include_sensitive: bool = False,
    ) -> dict:
        _page(offset, limit)
        replay = self.replays.get(game_id)
        events = replay.events(
            event_type=event_type,
            actor_id=actor_id,
            round=round,
        )
        items = []
        for event in events[offset:offset + limit]:
            item = asdict(event)
            item["payload"] = redact_trace_payload(
                event.event_type,
                event.payload,
                include_sensitive=include_sensitive,
            )
            items.append(item)
        return {
            "game_id": game_id,
            "items": items,
            "total": len(events),
            "offset": offset,
            "limit": limit,
            "sensitive": include_sensitive,
        }

    def get_agent_trace(
        self,
        game_id: str,
        agent_id: str,
        offset: int = 0,
        limit: int = 100,
        include_sensitive: bool = False,
    ) -> dict:
        return self.get_replay_events(
            game_id,
            actor_id=agent_id,
            offset=offset,
            limit=limit,
            include_sensitive=include_sensitive,
        )

    def get_tool_failures(
        self,
        game_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> dict:
        _page(offset, limit)
        replay = self.replays.get(game_id)
        failures = [
            event
            for event in replay.events(event_type="tool_execution")
            if event.payload.get("is_error")
        ]
        items = []
        for event in failures[offset:offset + limit]:
            item = asdict(event)
            item["payload"] = redact_trace_payload(
                event.event_type,
                event.payload,
                include_sensitive=False,
            )
            items.append(item)
        return {
            "game_id": game_id,
            "items": items,
            "total": len(failures),
            "offset": offset,
            "limit": limit,
        }

    def start_evaluation(
        self,
        game_ids: list[str] | None = None,
        input_price: float = 0.0,
        output_price: float = 0.0,
        idempotency_key: str | None = None,
        experiment_label: str | None = None,
        metadata: dict | None = None,
        judge_model: str | None = None,
        judge_max_actions: int = 50,
    ) -> dict:
        if self.evaluations is None:
            raise RuntimeError("Evaluation service is not configured")
        return self.evaluations.start(
            game_ids=game_ids,
            input_price=input_price,
            output_price=output_price,
            idempotency_key=idempotency_key,
            experiment_label=experiment_label,
            metadata=metadata,
            judge_model=judge_model,
            judge_max_actions=judge_max_actions,
        )

    def get_evaluation_status(self, run_id: str) -> dict:
        if self.evaluations is None:
            raise RuntimeError("Evaluation service is not configured")
        return self.evaluations.get(run_id)

    def get_evaluation_report(self, run_id: str) -> dict:
        if self.evaluations is None:
            raise RuntimeError("Evaluation service is not configured")
        return self.evaluations.report(run_id)

    def cancel_evaluation(self, run_id: str) -> dict:
        if self.evaluations is None:
            raise RuntimeError("Evaluation service is not configured")
        return self.evaluations.cancel(run_id)

    def compare_evaluations(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
    ) -> dict:
        if self.evaluations is None:
            raise RuntimeError("Evaluation service is not configured")
        return self.evaluations.compare(
            baseline_run_id,
            candidate_run_id,
        )
