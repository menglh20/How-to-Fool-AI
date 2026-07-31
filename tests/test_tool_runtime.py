"""Tests for safe provider-neutral tool execution."""

from game.llm_client import ToolCall, ToolDefinition
from game.game_tools import (
    MAX_CHAT_CHARACTERS,
    STAY_SILENT,
    send_message_tool,
)
from game.tool_runtime import ToolRegistry


def bottle_tool() -> ToolDefinition:
    return ToolDefinition(
        name="choose_bottle",
        description="Choose one bottle",
        parameters={
            "type": "object",
            "properties": {
                "bottle": {"type": "string", "enum": ["Red", "Blue"]}
            },
            "required": ["bottle"],
            "additionalProperties": False,
        },
    )


def test_registry_executes_valid_tool_call():
    registry = ToolRegistry()
    registry.register(bottle_tool(), lambda bottle: {"chosen": bottle})
    result = registry.execute(
        ToolCall("call-1", "choose_bottle", {"bottle": "Red"})
    )
    assert not result.is_error
    assert result.output == {"chosen": "Red"}


def test_registry_rejects_unknown_tool():
    result = ToolRegistry().execute(ToolCall("x", "delete_all", {}))
    assert result.is_error
    assert "Unknown tool" in result.output["error"]


def test_registry_validates_arguments_before_execution():
    called = False

    def handler(bottle):
        nonlocal called
        called = True

    registry = ToolRegistry()
    registry.register(bottle_tool(), handler)
    result = registry.execute(
        ToolCall("call-2", "choose_bottle", {"bottle": "Green"})
    )
    assert result.is_error
    assert not called


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry()
    registry.register(bottle_tool(), lambda bottle: bottle)
    try:
        registry.register(bottle_tool(), lambda bottle: bottle)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate registration should fail")


def test_registry_enforces_string_and_array_constraints():
    registry = ToolRegistry()
    definition = ToolDefinition(
        "attribute_words",
        "Attribute words",
        {
            "type": "object",
            "properties": {
                "authors": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["a", "b"]},
                    "minItems": 2,
                    "maxItems": 2,
                }
            },
            "required": ["authors"],
            "additionalProperties": False,
        },
    )
    registry.register(definition, lambda authors: authors)
    too_short = registry.execute(
        ToolCall("x", "attribute_words", {"authors": ["a"]})
    )
    invalid_item = registry.execute(
        ToolCall("y", "attribute_words", {"authors": ["a", "c"]})
    )
    assert too_short.is_error
    assert invalid_item.is_error


def test_registry_truncates_optional_metadata_but_not_required_content():
    captured = {}
    registry = ToolRegistry()
    registry.register(
        STAY_SILENT,
        lambda reason="": captured.update(reason=reason),
    )
    result = registry.execute(
        ToolCall("silent", "stay_silent", {"reason": "x" * 121})
    )
    assert not result.is_error
    assert captured["reason"] == "x" * 120

    messages = ToolRegistry()
    messages.register(
        send_message_tool(["human"]),
        lambda target, content, decision_summary="": None,
    )
    too_long = messages.execute(
        ToolCall(
            "message",
            "send_message",
            {
                "target": "human",
                "content": "x" * (MAX_CHAT_CHARACTERS + 1),
            },
        )
    )
    assert too_long.is_error
