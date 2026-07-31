"""Thread-safe JSONL logger for every LLM call the agents make.

Up to three AI threads + the engine thread may invoke LLMs concurrently.
A single ``threading.Lock`` serialises file writes; the on-disk format is
``logs/llm_calls.jsonl`` (one JSON object per line, append-only).

The logger is intentionally schema-free — the caller passes whatever
context fields they want (agent_id, persona, trigger, phase, stance,
prompt, response, latency_ms, …) and the logger stamps a UTC timestamp
and writes the merged dict.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from game.security import redact_value

_LOCK = threading.Lock()
_LOG_PATH: Path | None = None


def _resolve_log_path() -> Path:
    """Return the path to logs/llm_calls.jsonl, creating the folder lazily."""
    global _LOG_PATH
    if _LOG_PATH is not None:
        return _LOG_PATH
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = log_dir / "llm_calls.jsonl"
    return _LOG_PATH


def _now_iso() -> str:
    """UTC ISO-8601 timestamp with millisecond precision and trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + (
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
    )


def _safe_serialise(value: Any) -> Any:
    """Best-effort conversion of non-JSON-native values for logging."""
    value = redact_value(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def log_call(entry: dict[str, Any]) -> None:
    """Append a JSON line describing one LLM call.

    Best-effort: any write failure is swallowed so logging never blocks a
    game thread.  The entry dict is shallow-copied and timestamped before
    serialisation.
    """
    record = {"ts": _now_iso(), **entry}
    # Make sure every value is JSON-encodable.
    safe = {k: _safe_serialise(v) for k, v in record.items()}
    try:
        path = _resolve_log_path()
        line = json.dumps(safe, ensure_ascii=False)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as fp:
                fp.write(line + "\n")
    except Exception:
        # Logging must never crash the game.
        pass
