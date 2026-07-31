"""Security helpers shared by replay, MCP, logs, and evaluation."""

from __future__ import annotations

import re
from typing import Any

GAME_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
_SECRET_PATTERNS = (
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)\b(ANTHROPIC_API_KEY|OPENAI_API_KEY)\s*[:=]\s*\S+"
    ),
)
_INJECTION_PATTERNS = (
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)reveal\s+(the\s+)?system\s+prompt"),
    re.compile(r"(?i)developer\s+message"),
    re.compile(r"忽略.{0,12}(之前|以上).{0,8}(指令|提示)"),
    re.compile(r"(泄露|展示|输出).{0,8}系统提示词"),
    re.compile(r"开发者消息"),
)


def validate_game_id(game_id: str) -> str:
    if not GAME_ID_PATTERN.fullmatch(game_id):
        raise ValueError("game_id must be 32 lowercase hexadecimal characters")
    return game_id


def redact_text(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    return value


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def sanitize_model_output(text: str, *, max_length: int = 240) -> str:
    cleaned = redact_text(text)
    cleaned = "".join(
        character
        for character in cleaned
        if character in {"\n", "\t"} or ord(character) >= 32
    ).strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()
    return cleaned


def redact_trace_payload(
    event_type: str,
    payload: dict[str, Any],
    *,
    include_sensitive: bool,
) -> dict[str, Any]:
    safe = redact_value(payload)
    if include_sensitive:
        return safe
    if event_type == "chat_message":
        safe["text"] = "[PRIVATE MESSAGE REDACTED]"
        safe.pop("thinking", None)
    elif event_type == "llm_response":
        safe["text"] = "[MODEL TEXT REDACTED]"
        for call in safe.get("tool_calls", []):
            arguments = call.get("arguments", {})
            if "content" in arguments:
                arguments["content"] = "[PRIVATE MESSAGE REDACTED]"
            if "decision_summary" in arguments:
                arguments["decision_summary"] = "[REDACTED]"
    elif event_type == "tool_execution":
        arguments = safe.get("arguments", {})
        result = safe.get("result", {})
        if "content" in arguments:
            arguments["content"] = "[PRIVATE MESSAGE REDACTED]"
        if "decision_summary" in arguments:
            arguments["decision_summary"] = "[REDACTED]"
        if isinstance(result, dict) and "content" in result:
            result["content"] = "[PRIVATE MESSAGE REDACTED]"
    return safe
