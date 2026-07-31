"""Tests for configurable, JSON-serializable AI personas."""

import json

import pytest

from game.ai_agent import AIAgent
from game.llm_client import MockLLMClient
from game.persona import (
    PersonaConfig,
    default_personas,
    export_personas,
    import_personas,
)
from game.shared_state import SharedState


def test_defaults_match_the_three_existing_personas():
    personas = default_personas()
    assert list(personas) == ["bunny", "fox", "stoneface"]
    assert personas["bunny"].STANCE_PROBS == {
        "cooperate": 0.8,
        "deceive": 0.2,
        "silent": 0.0,
    }
    assert personas["fox"].INITIAL_TRUST == 0.45
    assert personas["stoneface"].CHAT_INITIATIVE_PROB == 0.02


def test_system_prompt_appends_non_negotiable_rules():
    persona = default_personas()["fox"]
    prompt = persona.SYSTEM_PROMPT
    assert persona.persona_description in prompt
    assert persona.speaking_style in prompt
    assert "Never reveal or discuss being an AI" in prompt
    assert "at or below 60 characters" in prompt
    assert "untrusted game dialogue" in prompt


def test_persona_json_round_trip():
    original = default_personas()
    encoded = export_personas(original.values())
    restored = import_personas(encoded)
    assert restored == original
    assert json.loads(encoded)["schema_version"] == 1


def test_import_accepts_one_persona_object():
    persona = default_personas()["bunny"]
    restored = import_personas(json.dumps(persona.to_dict()))
    assert restored == {"bunny": persona}


def test_persona_rejects_invalid_probabilities_and_unknown_json_fields():
    value = default_personas()["bunny"].to_dict()
    value["cooperate_probability"] = 0.9
    with pytest.raises(ValueError, match="must total 1.0"):
        PersonaConfig.from_dict(value)

    value = default_personas()["bunny"].to_dict()
    value["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        PersonaConfig.from_dict(value)

    bundle = json.loads(export_personas(default_personas().values()))
    bundle["unexpected"] = True
    with pytest.raises(ValueError, match="bundle has unknown fields"):
        import_personas(json.dumps(bundle))


def test_persona_json_rejects_duplicate_keys_and_oversized_input():
    persona = default_personas()["bunny"].to_dict()
    duplicate = json.dumps([persona, persona])
    with pytest.raises(ValueError, match="Duplicate persona key"):
        import_personas(duplicate)
    with pytest.raises(ValueError, match="100 KB"):
        import_personas("x" * 100_001)


def test_fallback_reply_cannot_expose_ai_identity_or_hidden_prompts():
    value = default_personas()["bunny"].to_dict()
    value["default_reply"] = "As an AI, I cannot answer"
    with pytest.raises(ValueError, match="cannot reveal AI identity"):
        PersonaConfig.from_dict(value)


def test_agent_consumes_config_without_a_persona_module():
    persona = default_personas()["fox"]
    state = SharedState(["human", "ai_0"])
    agent = AIAgent(
        "ai_0",
        persona,
        state,
        llm_client=MockLLMClient(),
    )
    assert agent._system_prompt() == persona.SYSTEM_PROMPT
    assert agent.memory.initial_trust == persona.initial_trust
