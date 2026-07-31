"""Offline tests for provider-neutral LLM backends."""

from __future__ import annotations

import pytest

from game.llm_client import (
    LLMRequest,
    LLMResponse,
    MockLLMClient,
    ScriptedLLMClient,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
    create_client,
    get_active_backend,
    get_active_model,
    set_active_client,
)


def request(default: str = "fallback") -> LLMRequest:
    return LLMRequest(
        system_prompt="system",
        messages=[{"role": "user", "content": "hello"}],
        default_reply=default,
    )


def test_mock_uses_fallback_without_provider_sdk():
    client = MockLLMClient()
    response = client.complete(request())
    assert response.text == "fallback"
    assert response.provider == "mock"
    assert response.tool_calls == ()
    assert client.provider == "mock"


def test_scripted_client_is_deterministic_and_falls_back():
    client = ScriptedLLMClient(["first", "second"])
    assert client.complete(request()).text == "first"
    assert client.complete(request()).text == "second"
    assert client.complete(request()).text == "fallback"
    assert client.remaining == 0


def test_scripted_client_can_return_tool_calls():
    scripted_response = LLMResponse(
        tool_calls=(
            ToolCall(
                id="call-1",
                name="choose_bottle",
                arguments={"bottle": "Red"},
            ),
        ),
        provider="scripted",
        model="scripted",
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        finish_reason="tool_call",
    )
    client = ScriptedLLMClient([scripted_response])
    response = client.complete(request())
    assert response.tool_calls[0].name == "choose_bottle"
    assert response.tool_calls[0].arguments == {"bottle": "Red"}
    assert response.usage.total_tokens == 15


def test_request_accepts_provider_neutral_tool_schema():
    tool = ToolDefinition(
        name="send_message",
        description="Send a private message",
        parameters={
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
            "additionalProperties": False,
        },
    )
    tool_request = LLMRequest(
        system_prompt="system",
        messages=[],
        default_reply="",
        tools=(tool,),
        tool_results=(ToolResult("previous-call", {"ok": True}),),
        tool_choice="required",
        metadata={"round": 2},
    )
    assert tool_request.tools[0].name == "send_message"
    assert tool_request.tool_results[0].output == {"ok": True}
    assert tool_request.metadata["round"] == 2


def test_invalid_tool_schema_and_choice_are_rejected():
    with pytest.raises(ValueError, match="JSON Schema object"):
        ToolDefinition("bad", "bad", {"type": "string"})
    with pytest.raises(ValueError, match="tool_choice"):
        LLMRequest("", [], "", tool_choice="sometimes")


def test_mock_factory_and_active_client():
    client = create_client("mock:mock")
    set_active_client(client)
    assert get_active_backend() == "mock"
    assert get_active_model() == "mock"


def test_anthropic_factory_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        create_client("anthropic:test-model")


def test_openai_factory_requires_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        create_client("openai:env")
