"""JSON-Schema tools exposed to game agents, scoped by action phase."""

from __future__ import annotations

from game.llm_client import ToolDefinition

MAX_CHAT_CHARACTERS = 60


def _summary_property() -> dict:
    return {
        "type": "string",
        "description": "A brief, user-safe explanation of the decision.",
        "maxLength": 120,
    }


def tick_action_tool(
    game_tool: ToolDefinition | None,
    chat_targets: list[str],
) -> ToolDefinition:
    """One tick may contain one game action and one private message."""
    properties: dict = {
        "decision_summary": _summary_property(),
    }
    if game_tool is not None:
        properties["game_action"] = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [game_tool.name]},
                "arguments": game_tool.parameters,
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
    if chat_targets:
        properties["chat_action"] = {
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": chat_targets},
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_CHAT_CHARACTERS,
                },
            },
            "required": ["target", "content"],
            "additionalProperties": False,
        }
    return ToolDefinition(
        name="act_in_tick",
        description=(
            "Submit this tick's optional game action and optional private "
            "message. The game action is executed first."
        ),
        parameters={
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        },
        # Nested optional action objects are validated again by ToolRegistry.
        strict=False,
    )


def send_message_tool(targets: list[str]) -> ToolDefinition:
    return ToolDefinition(
        name="send_message",
        description="Send one private in-game message to an allowed player.",
        parameters={
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": targets},
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_CHAT_CHARACTERS,
                },
                "decision_summary": _summary_property(),
            },
            "required": ["target", "content"],
            "additionalProperties": False,
        },
    )


STAY_SILENT = ToolDefinition(
    name="stay_silent",
    description="Do not send a message during this opportunity.",
    parameters={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "maxLength": 120},
        },
        "additionalProperties": False,
    },
)


def word_tool(name: str, description: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "pattern": "^[A-Za-z][A-Za-z'-]*$",
                    "maxLength": 40,
                },
                "decision_summary": _summary_property(),
            },
            "required": ["word"],
            "additionalProperties": False,
        },
    )


SUBMIT_WORD = word_tool("submit_word", "Submit the secret English word.")
GUESS_WORD = word_tool("guess_word", "Guess the writer's English word.")


def attribute_words_tool(candidates: list[str], count: int) -> ToolDefinition:
    return ToolDefinition(
        name="attribute_words",
        description=(
            "Attribute each displayed word to an author. Authors must be in "
            "the same order as the displayed words."
        ),
        parameters={
            "type": "object",
            "properties": {
                "authors": {
                    "type": "array",
                    "items": {"type": "string", "enum": candidates},
                    "minItems": count,
                    "maxItems": count,
                },
                "decision_summary": _summary_property(),
            },
            "required": ["authors"],
            "additionalProperties": False,
        },
    )


def choose_bottle_tool(bottles: list[str]) -> ToolDefinition:
    return ToolDefinition(
        name="choose_bottle",
        description="Choose one currently available bottle.",
        parameters={
            "type": "object",
            "properties": {
                "bottle": {"type": "string", "enum": bottles},
                "decision_summary": _summary_property(),
            },
            "required": ["bottle"],
            "additionalProperties": False,
        },
    )
