"""Read-only game-trace queries used by replay UIs and MCP tools."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from game.security import redact_trace_payload, validate_game_id
from game.trace import TraceEvent


class ReplayService:
    def __init__(self, events: Iterable[TraceEvent]) -> None:
        self._events = sorted(events, key=lambda event: event.sequence)

    @classmethod
    def from_jsonl(cls, path: Path) -> "ReplayService":
        events: list[TraceEvent] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    events.append(TraceEvent(**json.loads(line)))
                except Exception as exc:
                    raise ValueError(
                        f"Invalid trace line {line_number}: {exc}"
                    ) from exc
        return cls(events)

    @property
    def game_id(self) -> str | None:
        return self._events[0].game_id if self._events else None

    def events(
        self,
        *,
        event_type: str | None = None,
        actor_id: str | None = None,
        round: int | None = None,
    ) -> list[TraceEvent]:
        return [
            event
            for event in self._events
            if (event_type is None or event.event_type == event_type)
            and (actor_id is None or event.actor_id == actor_id)
            and (round is None or event.round == round)
        ]

    def timeline(self, *, include_sensitive: bool = True) -> list[dict]:
        timeline = []
        for event in self._events:
            item = asdict(event)
            item["payload"] = redact_trace_payload(
                event.event_type,
                event.payload,
                include_sensitive=include_sensitive,
            )
            timeline.append(item)
        return timeline

    def summary(self) -> dict:
        counts = Counter(event.event_type for event in self._events)
        tool_events = self.events(event_type="tool_execution")
        rounds = sorted({
            event.round for event in self._events if event.round is not None
        })
        return {
            "game_id": self.game_id,
            "event_count": len(self._events),
            "event_types": dict(counts),
            "rounds": rounds,
            "tool_calls": len(tool_events),
            "tool_errors": sum(
                bool(event.payload.get("is_error")) for event in tool_events
            ),
            "tool_fallbacks": sum(
                bool(event.payload.get("fallback")) for event in tool_events
            ),
        }


class ReplayRepository:
    """Constrained access to trace files under one configured directory."""

    def __init__(self, trace_directory: Path) -> None:
        self.trace_directory = trace_directory.resolve()
        self.trace_directory.mkdir(parents=True, exist_ok=True)

    def list_replays(
        self, *, offset: int = 0, limit: int = 50
    ) -> list[dict]:
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        files = sorted(
            self.trace_directory.glob("*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        rows = []
        for path in files[offset:offset + limit]:
            try:
                game_id = validate_game_id(path.stem)
                replay = ReplayService.from_jsonl(path)
                rows.append({
                    **replay.summary(),
                    "game_id": game_id,
                    "modified_at": path.stat().st_mtime,
                })
            except (ValueError, OSError):
                continue
        return rows

    def get(self, game_id: str) -> ReplayService:
        validated = validate_game_id(game_id)
        path = (self.trace_directory / f"{validated}.jsonl").resolve()
        if path.parent != self.trace_directory:
            raise ValueError("Trace path escapes the configured directory")
        if not path.is_file():
            raise FileNotFoundError(f"Replay not found: {validated}")
        return ReplayService.from_jsonl(path)
