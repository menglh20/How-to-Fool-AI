"""Pluggable LLM backends for development, evaluation, and online play.

The game depends on the small :class:`LLMClient` interface instead of a
provider SDK.  This keeps unit tests offline and makes model providers a
runtime choice:

* ``MockLLMClient`` returns the caller-provided fallback without using tokens.
* ``ScriptedLLMClient`` consumes deterministic responses for evals/replays.
* ``AnthropicLLMClient`` and ``OpenAILLMClient`` power online demos.

Provider SDKs are imported lazily, so importing the game never requires them.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from game.llm_logger import log_call

_REQUEST_TIMEOUT_SECONDS = 30.0

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Environment variables and Streamlit secrets still work without dotenv.
    pass


@dataclass(frozen=True)
class ToolDefinition:
    """Provider-neutral function/tool declaration."""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if self.parameters.get("type") != "object":
            raise ValueError("Tool parameters must be a JSON Schema object")


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by a model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Result returned by the application after executing a tool."""

    tool_call_id: str
    output: Any
    is_error: bool = False


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Normalized result returned by every backend."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    provider: str = "unknown"
    model: str = "unknown"
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: int = 0
    finish_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class LLMRequest:
    system_prompt: str
    messages: list[dict[str, Any]]
    default_reply: str
    max_tokens: int = 300
    tools: tuple[ToolDefinition, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    tool_choice: str = "auto"
    agent_id: str | None = None
    persona: str | None = None
    trigger: str | None = None
    phase: str | None = None
    stance: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tool_choice not in {"auto", "required", "none"}:
            raise ValueError(
                "tool_choice must be one of: auto, required, none"
            )


class LLMClient(ABC):
    """Provider-neutral interface used by every AI agent."""

    provider = "unknown"

    def __init__(self, model: str, *, record_calls: bool = True) -> None:
        self.model = model
        self.record_calls = record_calls

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a normalized response, falling back rather than raising."""

    def _record(
        self,
        request: LLMRequest,
        response: LLMResponse,
    ) -> None:
        if not self.record_calls:
            return
        entry: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "agent_id": request.agent_id,
            "persona": request.persona,
            "trigger": request.trigger,
            "phase": request.phase,
            "stance": request.stance,
            "latency_ms": response.latency_ms,
            "system_prompt": request.system_prompt,
            "user_messages": request.messages,
            "tools": [asdict(tool) for tool in request.tools],
            "tool_results": [asdict(item) for item in request.tool_results],
            "tool_choice": request.tool_choice,
            "response": response.text,
            "tool_calls": [asdict(call) for call in response.tool_calls],
            "usage": asdict(response.usage),
            "finish_reason": response.finish_reason,
            "metadata": dict(request.metadata),
        }
        if response.error:
            entry["error"] = response.error
        log_call(entry)


class MockLLMClient(LLMClient):
    """Zero-token backend for local development and unit tests."""

    provider = "mock"

    def __init__(self, *, record_calls: bool = False) -> None:
        super().__init__("mock", record_calls=record_calls)

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = LLMResponse(
            text=request.default_reply,
            provider=self.provider,
            model=self.model,
            finish_reason="mock",
        )
        self._record(request, response)
        return response


class ScriptedLLMClient(LLMClient):
    """Deterministic backend for batch evaluations and exact replays."""

    provider = "scripted"

    def __init__(
        self,
        responses: Iterable[str | LLMResponse],
        *,
        record_calls: bool = False,
    ) -> None:
        super().__init__("scripted", record_calls=record_calls)
        self._responses = deque(responses)
        self._lock = threading.Lock()

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            scripted = (
                self._responses.popleft()
                if self._responses
                else request.default_reply
            )
        if isinstance(scripted, str):
            response = LLMResponse(
                text=scripted,
                provider=self.provider,
                model=self.model,
                finish_reason="scripted",
            )
        else:
            # Accept response objects created before an importlib.reload too.
            response = scripted
        self._record(request, response)
        return response


def _secret(name: str) -> str:
    """Resolve a secret from Streamlit first, then the environment."""
    try:
        import streamlit as st

        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name, "")


class AnthropicLLMClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(model)
        self.api_key = api_key

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.monotonic()
        try:
            import anthropic

            client = anthropic.Anthropic(
                api_key=self.api_key or _secret("ANTHROPIC_API_KEY"),
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": request.max_tokens,
                "system": request.system_prompt,
                "messages": list(request.messages),
            }
            if request.tool_results:
                import json

                kwargs["messages"].append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": item.tool_call_id,
                            "content": json.dumps(
                                item.output, ensure_ascii=False
                            ),
                            "is_error": item.is_error,
                        }
                        for item in request.tool_results
                    ],
                })
            if request.tools:
                kwargs["tools"] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    }
                    for tool in request.tools
                ]
                anthropic_choice = (
                    "any" if request.tool_choice == "required"
                    else request.tool_choice
                )
                kwargs["tool_choice"] = {"type": anthropic_choice}
            result = client.messages.create(
                **kwargs
            )
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in result.content:
                block_type = getattr(block, "type", "")
                if block_type == "text":
                    text_parts.append(getattr(block, "text", ""))
                elif block_type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    ))
            input_tokens = getattr(result.usage, "input_tokens", None)
            output_tokens = getattr(result.usage, "output_tokens", None)
            response = LLMResponse(
                text="\n".join(text_parts).strip(),
                tool_calls=tuple(tool_calls),
                provider=self.provider,
                model=self.model,
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=(
                        input_tokens + output_tokens
                        if input_tokens is not None
                        and output_tokens is not None
                        else None
                    ),
                ),
                latency_ms=int((time.monotonic() - started) * 1000),
                finish_reason=getattr(result, "stop_reason", None),
            )
            self._record(request, response)
            return response
        except Exception as exc:
            response = LLMResponse(
                text=request.default_reply,
                provider=self.provider,
                model=self.model,
                latency_ms=int((time.monotonic() - started) * 1000),
                finish_reason="fallback",
                error=repr(exc),
            )
            self._record(request, response)
            return response


class OpenAILLMClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(model)
        self.api_key = api_key

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.monotonic()
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key or _secret("OPENAI_API_KEY"),
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            kwargs: dict[str, Any] = {
                "model": self.model,
                "instructions": request.system_prompt,
                "input": list(request.messages),
                "max_output_tokens": request.max_tokens,
            }
            if request.tool_results:
                import json

                kwargs["input"].extend(
                    {
                        "type": "function_call_output",
                        "call_id": item.tool_call_id,
                        "output": json.dumps(item.output, ensure_ascii=False),
                    }
                    for item in request.tool_results
                )
            if request.tools:
                kwargs["tools"] = [
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    }
                    for tool in request.tools
                ]
                kwargs["tool_choice"] = request.tool_choice
            result = client.responses.create(**kwargs)
            tool_calls: list[ToolCall] = []
            for item in getattr(result, "output", []):
                if getattr(item, "type", "") != "function_call":
                    continue
                import json

                raw_arguments = getattr(item, "arguments", "{}")
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else dict(raw_arguments)
                )
                tool_calls.append(ToolCall(
                    id=str(
                        getattr(item, "call_id", None)
                        or getattr(item, "id", "")
                    ),
                    name=str(getattr(item, "name", "")),
                    arguments=arguments,
                ))
            usage = getattr(result, "usage", None)
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            response = LLMResponse(
                text=result.output_text.strip(),
                tool_calls=tuple(tool_calls),
                provider=self.provider,
                model=self.model,
                usage=TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                ),
                latency_ms=int((time.monotonic() - started) * 1000),
                finish_reason=getattr(result, "status", None),
            )
            self._record(request, response)
            return response
        except Exception as exc:
            response = LLMResponse(
                text=request.default_reply,
                provider=self.provider,
                model=self.model,
                latency_ms=int((time.monotonic() - started) * 1000),
                finish_reason="fallback",
                error=repr(exc),
            )
            self._record(request, response)
            return response


# The value is ``provider:model``. Mock is the safe default and needs no key.
MODEL_CHOICES: dict[str, str] = {
    "Mock — free, offline development": "mock:mock",
    "Anthropic Opus 4.8 — most capable": "anthropic:claude-opus-4-8",
    "Anthropic Sonnet 4.6 — balanced": "anthropic:claude-sonnet-4-6",
    "Anthropic Haiku 4.5 — fast & cheap": (
        "anthropic:claude-haiku-4-5-20251001"
    ),
    "OpenAI — model from OPENAI_MODEL": "openai:env",
}

_ACTIVE_LOCK = threading.RLock()
_active_client: LLMClient = MockLLMClient()


def create_client(spec: str) -> LLMClient:
    """Create a backend from ``provider:model`` configuration."""
    provider, separator, model = spec.partition(":")
    if not separator:
        # Backward compatibility: historical values were Anthropic model IDs.
        provider, model = "anthropic", spec
    if provider == "mock":
        return MockLLMClient()
    if provider == "scripted":
        raise ValueError("ScriptedLLMClient requires an explicit response list")
    if provider == "anthropic":
        key = _secret("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be configured for the Anthropic backend"
            )
        return AnthropicLLMClient(model, api_key=key)
    if provider == "openai":
        selected = _secret("OPENAI_MODEL") if model == "env" else model
        if not selected:
            raise ValueError(
                "OPENAI_MODEL must be configured when using the OpenAI backend"
            )
        key = _secret("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY must be configured for the OpenAI backend"
            )
        return OpenAILLMClient(selected, api_key=key)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def set_active_client(client: LLMClient) -> None:
    global _active_client
    with _ACTIVE_LOCK:
        _active_client = client


def get_active_client() -> LLMClient:
    with _ACTIVE_LOCK:
        return _active_client


def set_active_model(spec: str) -> None:
    """Compatibility entry point used by the Streamlit setup screen."""
    set_active_client(create_client(spec))


def get_active_model() -> str:
    return get_active_client().model


def get_active_backend() -> str:
    return get_active_client().provider


def _request(
    system_prompt: str,
    messages: list[dict[str, Any]],
    default_reply: str,
    **kwargs: Any,
) -> LLMRequest:
    return LLMRequest(
        system_prompt=system_prompt,
        messages=messages,
        default_reply=default_reply,
        **kwargs,
    )


def call_llm(
    system_prompt: str,
    messages: list[dict[str, Any]],
    default_reply: str,
    **kwargs: Any,
) -> str:
    """Backward-compatible functional facade."""
    return get_active_client().complete(
        _request(system_prompt, messages, default_reply, **kwargs)
    ).text


def call_strategic(
    system_prompt: str,
    messages: list[dict[str, Any]],
    default_reply: str,
    **kwargs: Any,
) -> str:
    """Strategic calls share the same configured backend."""
    return call_llm(system_prompt, messages, default_reply, **kwargs)


def _get_api_key() -> str:
    """Backward-compatible Anthropic key helper."""
    return _secret("ANTHROPIC_API_KEY")
