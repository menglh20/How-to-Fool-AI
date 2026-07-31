"""Provider-neutral registration, validation, and execution of LLM tools."""

from __future__ import annotations

from collections.abc import Callable
import re
from typing import Any

from game.llm_client import ToolCall, ToolDefinition, ToolResult

ToolHandler = Callable[..., Any]


class ToolRegistry:
    """Maps declared tools to trusted application handlers."""

    def __init__(self) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, ToolHandler] = {}

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._definitions.values())

    def register(
        self, definition: ToolDefinition, handler: ToolHandler
    ) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._definitions[definition.name] = definition
        self._handlers[definition.name] = handler

    def execute(self, call: ToolCall) -> ToolResult:
        definition = self._definitions.get(call.name)
        if definition is None:
            return ToolResult(
                call.id, {"error": f"Unknown tool: {call.name}"}, True
            )
        try:
            arguments = dict(call.arguments)
            properties = definition.parameters.get("properties", {})
            required = set(definition.parameters.get("required", []))
            for name, value in list(arguments.items()):
                field = properties.get(name, {})
                maximum = field.get("maxLength")
                if (
                    name not in required
                    and isinstance(value, str)
                    and maximum is not None
                    and len(value) > maximum
                ):
                    arguments[name] = value[:maximum]
            self._validate(arguments, definition.parameters)
            output = self._handlers[call.name](**arguments)
            return ToolResult(call.id, output)
        except Exception as exc:
            return ToolResult(call.id, {"error": str(exc)}, True)

    @staticmethod
    def _validate(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            raise ValueError(f"Missing required arguments: {missing}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(arguments) - set(properties))
            if extra:
                raise ValueError(f"Unexpected arguments: {extra}")

        python_types = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        for name, value in arguments.items():
            field_schema = properties.get(name)
            if not field_schema:
                continue
            schema_type = field_schema.get("type")
            expected = python_types.get(schema_type)
            wrong_type = (
                schema_type == "integer" and type(value) is not int
            ) or (
                schema_type == "number"
                and type(value) not in {int, float}
            ) or (
                schema_type not in {"integer", "number"}
                and expected
                and not isinstance(value, expected)
            )
            if wrong_type:
                raise ValueError(
                    f"Argument {name!r} has the wrong type"
                )
            allowed = field_schema.get("enum")
            if allowed is not None and value not in allowed:
                raise ValueError(
                    f"Argument {name!r} must be one of {allowed}"
                )
            if isinstance(value, str):
                minimum = field_schema.get("minLength")
                maximum = field_schema.get("maxLength")
                pattern = field_schema.get("pattern")
                if minimum is not None and len(value) < minimum:
                    raise ValueError(f"Argument {name!r} is too short")
                if maximum is not None and len(value) > maximum:
                    raise ValueError(f"Argument {name!r} is too long")
                if pattern and re.fullmatch(pattern, value) is None:
                    raise ValueError(
                        f"Argument {name!r} has an invalid format"
                    )
            if isinstance(value, list):
                minimum = field_schema.get("minItems")
                maximum = field_schema.get("maxItems")
                if minimum is not None and len(value) < minimum:
                    raise ValueError(f"Argument {name!r} has too few items")
                if maximum is not None and len(value) > maximum:
                    raise ValueError(f"Argument {name!r} has too many items")
                item_schema = field_schema.get("items", {})
                item_type = python_types.get(item_schema.get("type"))
                item_enum = item_schema.get("enum")
                for item in value:
                    if item_type and not isinstance(item, item_type):
                        raise ValueError(
                            f"Argument {name!r} contains a wrong type"
                        )
                    if item_enum is not None and item not in item_enum:
                        raise ValueError(
                            f"Argument {name!r} contains an invalid value"
                        )
