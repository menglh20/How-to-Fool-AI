"""Metrics computed exclusively from recorded game traces."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from game.replay import ReplayService


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1),
    )
    return round(ordered[index], 3)


@dataclass(frozen=True)
class Pricing:
    input_per_million: float = 0.0
    output_per_million: float = 0.0


class TraceEvaluator:
    def __init__(self, pricing: Pricing | None = None) -> None:
        self.pricing = pricing or Pricing()

    def evaluate(self, replay: ReplayService) -> dict:
        tool_events = replay.events(event_type="tool_execution")
        llm_events = replay.events(event_type="llm_response")
        groups: dict[str, list] = defaultdict(list)
        for event in tool_events:
            action_id = event.payload.get("action_id") or (
                f"legacy-{event.sequence}"
            )
            groups[str(action_id)].append(event)

        first_legal = 0
        repaired = 0
        fallbacks = 0
        completed_actions = 0
        invalid_first = 0
        for events in groups.values():
            ordered = sorted(
                events,
                key=lambda event: (
                    int(event.payload.get("attempt", 0)),
                    event.sequence,
                ),
            )
            first = ordered[0]
            if not first.payload.get("is_error"):
                first_legal += 1
            else:
                invalid_first += 1
            non_fallback_success = any(
                not event.payload.get("is_error")
                and not event.payload.get("fallback")
                for event in ordered[1:]
            )
            if first.payload.get("is_error") and non_fallback_success:
                repaired += 1
            if any(event.payload.get("fallback") for event in ordered):
                fallbacks += 1
            if any(not event.payload.get("is_error") for event in ordered):
                completed_actions += 1

        latencies = [
            float(event.payload["latency_ms"])
            for event in llm_events
            if event.payload.get("latency_ms") is not None
        ]
        input_tokens = sum(
            int(event.payload.get("usage", {}).get("input_tokens") or 0)
            for event in llm_events
        )
        output_tokens = sum(
            int(event.payload.get("usage", {}).get("output_tokens") or 0)
            for event in llm_events
        )
        cost = (
            input_tokens / 1_000_000 * self.pricing.input_per_million
            + output_tokens / 1_000_000 * self.pricing.output_per_million
        )
        game_over = [
            event
            for event in replay.events(event_type="game_event")
            if event.payload.get("event_type") == "game_over"
        ]
        final_scores = (
            game_over[-1]
            .payload.get("event_payload", {})
            .get("final_scores", {})
            if game_over
            else {}
        )
        winners = []
        if final_scores:
            top = max(final_scores.values())
            winners = sorted(
                player for player, score in final_scores.items()
                if score == top
            )

        actions = len(groups)
        return {
            "game_id": replay.game_id,
            "complete": bool(game_over),
            "rounds": replay.summary()["rounds"],
            "actions": actions,
            "completed_actions": completed_actions,
            "first_call_legal_rate": (
                first_legal / actions if actions else None
            ),
            "repair_success_rate": (
                repaired / invalid_first if invalid_first else None
            ),
            "fallback_rate": fallbacks / actions if actions else None,
            "llm_calls": len(llm_events),
            "llm_errors": sum(
                bool(event.payload.get("error")) for event in llm_events
            ),
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": round(cost, 8),
            "final_scores": final_scores,
            "winners": winners,
        }


def aggregate_reports(reports: Iterable[dict]) -> dict:
    rows = list(reports)
    total = len(rows)

    def average(name: str) -> float | None:
        values = [row[name] for row in rows if row.get(name) is not None]
        return round(sum(values) / len(values), 6) if values else None

    winner_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for winner in row.get("winners", []):
            winner_counts[winner] += 1
    return {
        "games": total,
        "complete_games": sum(bool(row.get("complete")) for row in rows),
        "completion_rate": (
            sum(bool(row.get("complete")) for row in rows) / total
            if total else None
        ),
        "first_call_legal_rate": average("first_call_legal_rate"),
        "repair_success_rate": average("repair_success_rate"),
        "fallback_rate": average("fallback_rate"),
        "latency_p50_ms": average("latency_p50_ms"),
        "latency_p95_ms": average("latency_p95_ms"),
        "llm_calls": sum(row.get("llm_calls", 0) for row in rows),
        "llm_errors": sum(row.get("llm_errors", 0) for row in rows),
        "input_tokens": sum(row.get("input_tokens", 0) for row in rows),
        "output_tokens": sum(row.get("output_tokens", 0) for row in rows),
        "estimated_cost": round(
            sum(row.get("estimated_cost", 0.0) for row in rows), 8
        ),
        "winner_counts": dict(winner_counts),
    }
