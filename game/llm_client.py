"""LLM client for AI agent calls.

API key resolution order:
1. st.secrets["ANTHROPIC_API_KEY"]  (Streamlit deployment / local secrets.toml)
2. os.environ["ANTHROPIC_API_KEY"]  (CI / shell environment)

Model assignment:
- Every call routes to ``claude-opus-4-8`` — the latest Opus.  The two
  function names (``call_llm`` / ``call_strategic``) are retained for
  backward compatibility with existing test stubs but both hit the same
  Opus model now.

Every successful or fallback call writes one entry to the LLM logger so
downstream tooling can audit who called what, with what context, when.
"""

from __future__ import annotations

import os
import time
from typing import Any

import anthropic

from game.llm_logger import log_call

_DEFAULT_MODEL = "claude-opus-4-8"

# Runtime-configurable so the setup page can switch to a cheaper model for
# fast / low-cost test runs.  Single-session app, so a module-level variable
# is sufficient.
_active_model: str = _DEFAULT_MODEL


def set_active_model(model: str) -> None:
    """Change the model used by every subsequent ``call_llm`` / ``call_strategic``."""
    global _active_model
    _active_model = model


def get_active_model() -> str:
    return _active_model


# Convenience map for the setup-page selector.
MODEL_CHOICES: dict[str, str] = {
    "Opus 4.8 — most capable (default)": "claude-opus-4-8",
    "Sonnet 4.6 — balanced": "claude-sonnet-4-6",
    "Haiku 4.5 — fast & cheap": "claude-haiku-4-5-20251001",
}


def _get_api_key() -> str:
    """Return the Anthropic API key from Streamlit secrets or env."""
    try:
        import streamlit as st  # imported lazily — not available in tests
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key


def _make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=_get_api_key())


def _log(
    *,
    agent_id: str | None,
    persona: str | None,
    trigger: str | None,
    phase: str | None,
    stance: str | None,
    system_prompt: str,
    messages: list[dict[str, str]],
    response: str,
    latency_ms: int,
    error: str | None = None,
) -> None:
    """Forward one LLM-call record to the logger.

    Caller must always pass log context kwargs; if they're missing we log
    what we have so the entry isn't lost.
    """
    entry: dict[str, Any] = {
        "agent_id": agent_id,
        "persona": persona,
        "model": _active_model,
        "trigger": trigger,
        "phase": phase,
        "stance": stance,
        "latency_ms": latency_ms,
        "system_prompt": system_prompt,
        "user_messages": messages,
        "response": response,
    }
    if error:
        entry["error"] = error
    log_call(entry)


def _call(
    system_prompt: str,
    messages: list[dict[str, str]],
    default_reply: str,
    *,
    max_tokens: int,
    agent_id: str | None,
    persona: str | None,
    trigger: str | None,
    phase: str | None,
    stance: str | None,
) -> str:
    """Shared Opus call body used by both public entry points."""
    started = time.monotonic()
    try:
        client = _make_client()
        response = client.messages.create(
            model=_active_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        text = response.content[0].text.strip()
        latency = int((time.monotonic() - started) * 1000)
        _log(
            agent_id=agent_id,
            persona=persona,
            trigger=trigger,
            phase=phase,
            stance=stance,
            system_prompt=system_prompt,
            messages=messages,
            response=text,
            latency_ms=latency,
        )
        return text
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        _log(
            agent_id=agent_id,
            persona=persona,
            trigger=trigger,
            phase=phase,
            stance=stance,
            system_prompt=system_prompt,
            messages=messages,
            response=default_reply,
            latency_ms=latency,
            error=repr(exc),
        )
        return default_reply


def call_llm(
    system_prompt: str,
    messages: list[dict[str, str]],
    default_reply: str,
    *,
    max_tokens: int = 300,
    agent_id: str | None = None,
    persona: str | None = None,
    trigger: str | None = None,
    phase: str | None = None,
    stance: str | None = None,
) -> str:
    """Call Opus for a chat-style reply.

    Falls back to *default_reply* on any API error so the agent never crashes.
    """
    return _call(
        system_prompt,
        messages,
        default_reply,
        max_tokens=max_tokens,
        agent_id=agent_id,
        persona=persona,
        trigger=trigger,
        phase=phase,
        stance=stance,
    )


def call_strategic(
    system_prompt: str,
    messages: list[dict[str, str]],
    default_reply: str,
    *,
    max_tokens: int = 300,
    agent_id: str | None = None,
    persona: str | None = None,
    trigger: str | None = None,
    phase: str | None = None,
    stance: str | None = None,
) -> str:
    """Call Opus for a strategic in-game decision.

    Identical to :func:`call_llm` — both names exist for backward compat.
    """
    return _call(
        system_prompt,
        messages,
        default_reply,
        max_tokens=max_tokens,
        agent_id=agent_id,
        persona=persona,
        trigger=trigger,
        phase=phase,
        stance=stance,
    )
