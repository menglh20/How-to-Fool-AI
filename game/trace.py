"""Thread-safe per-game event tracing for replay and evaluation."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from game.security import redact_value


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return repr(value)


@dataclass(frozen=True)
class TraceEvent:
    game_id: str
    sequence: int
    timestamp: str
    event_type: str
    actor_id: str | None
    round: int | None
    game_type: str | None
    payload: dict[str, Any]


class TraceRecorder:
    """Stores an ordered in-memory trace and optionally appends JSONL."""

    def __init__(
        self,
        game_id: str | None = None,
        path: Path | None = None,
    ) -> None:
        self.game_id = game_id or uuid.uuid4().hex
        self.path = path
        self._lock = threading.Lock()
        self._sequence = 0
        self._events: list[TraceEvent] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def persistent(cls, directory: Path) -> "TraceRecorder":
        game_id = uuid.uuid4().hex
        return cls(game_id, directory / f"{game_id}.jsonl")

    def record(
        self,
        event_type: str,
        *,
        actor_id: str | None = None,
        round: int | None = None,
        game_type: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> TraceEvent:
        safe_payload = _json_safe(redact_value(payload or {}))
        with self._lock:
            self._sequence += 1
            event = TraceEvent(
                game_id=self.game_id,
                sequence=self._sequence,
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                actor_id=actor_id,
                round=round,
                game_type=game_type,
                payload=safe_payload,
            )
            self._events.append(event)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(asdict(event), ensure_ascii=False) + "\n"
                    )
            return event

    def snapshot(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)
