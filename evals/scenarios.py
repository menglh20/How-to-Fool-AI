"""Deterministic safety and tool scenarios executable without model tokens."""

from __future__ import annotations

import json
from pathlib import Path

from game.game_tools import choose_bottle_tool
from game.llm_client import ToolCall
from game.memory import AgentMemory
from game.security import (
    contains_prompt_injection,
    redact_text,
    validate_game_id,
)
from game.tool_runtime import ToolRegistry


def load_scenarios(directory: Path) -> list[dict]:
    scenarios = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Scenario file must contain a list: {path}")
        for item in data:
            scenarios.append({**item, "source": path.name})
    return scenarios


def _run_one(scenario: dict) -> tuple[bool, str]:
    kind = scenario["kind"]
    if kind == "game_id_validation":
        try:
            validate_game_id(scenario["input"])
            actual = True
        except ValueError:
            actual = False
        expected = bool(scenario["expected_valid"])
        return actual == expected, f"valid={actual}"

    if kind == "secret_redaction":
        output = redact_text(scenario["input"])
        leaked = any(value in output for value in scenario["must_not_contain"])
        return not leaked, output

    if kind == "prompt_injection_detection":
        actual = contains_prompt_injection(scenario["input"])
        expected = bool(scenario["expected_detected"])
        return actual == expected, f"detected={actual}"

    if kind == "choose_bottle_tool":
        registry = ToolRegistry()
        registry.register(
            choose_bottle_tool(scenario["bottles"]),
            lambda bottle, decision_summary="": bottle,
        )
        result = registry.execute(
            ToolCall(
                scenario["id"],
                "choose_bottle",
                scenario["arguments"],
            )
        )
        actual = not result.is_error
        expected = bool(scenario["expected_valid"])
        return actual == expected, str(result.output)

    if kind == "memory_verification":
        memory = AgentMemory("evaluator")
        claim = memory.ingest_message(
            source=scenario["source_agent"],
            text=scenario["message"],
            round=scenario["round"],
            game_type=scenario["game_type"],
        )
        memory.verify_round(
            round=scenario["round"],
            game_type=scenario["game_type"],
            results=scenario["results"],
        )
        expected = bool(scenario["expected_truth"])
        return claim.verified and claim.truth is expected, str(claim.truth)

    raise ValueError(f"Unknown scenario kind: {kind}")


def run_scenarios(directory: Path) -> dict:
    results = []
    for scenario in load_scenarios(directory):
        try:
            passed, detail = _run_one(scenario)
            error = None
        except Exception as exc:
            passed, detail, error = False, "", str(exc)
        results.append({
            "id": scenario["id"],
            "kind": scenario["kind"],
            "source": scenario["source"],
            "passed": passed,
            "detail": detail,
            "error": error,
        })
    passed = sum(result["passed"] for result in results)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else None,
        "results": results,
    }
