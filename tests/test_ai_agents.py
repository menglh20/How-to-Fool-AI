"""Tests for game/ai_agent.py, game/llm_client.py, and prompts/.

These tests do NOT require a live Anthropic API key — all LLM calls are
patched with lightweight stubs that return default persona replies.

Run with:
    python -m pytest tests/test_ai_agents.py -v
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from game.shared_state import ChatMessage, GameEvent, SharedState
from game.ai_agent import AIAgent
from game.llm_client import MockLLMClient
import prompts.bai as bai
import prompts.fox as fox
import prompts.ironface as ironface

PLAYERS = ["ai_0", "ai_1", "ai_2"]
PERSONAS = {
    "ai_0": bai,
    "ai_1": fox,
    "ai_2": ironface,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    return SharedState(PLAYERS, initial_score=0, initial_send_budget=20)


def _make_agent(player_id: str, state: SharedState) -> AIAgent:
    return AIAgent(
        player_id=player_id,
        persona=PERSONAS[player_id],
        state=state,
        llm_client=MockLLMClient(),
        tick_seconds=0.01,
    )


# ---------------------------------------------------------------------------
# 1. Three AI threads can start independently
# ---------------------------------------------------------------------------

def test_three_agents_start_independently(state):
    """All three AIAgent threads start without error and are alive."""
    agents = [_make_agent(pid, state) for pid in PLAYERS]
    try:
        for a in agents:
            a.start()
        time.sleep(0.1)
        for a in agents:
            assert a.is_alive(), f"{a.name} is not alive after start()"
    finally:
        for a in agents:
            a.stop()
        for a in agents:
            a.join(timeout=5)


def test_agents_stop_cleanly(state):
    """Agents exit within 5 seconds after stop() is called."""
    agents = [_make_agent(pid, state) for pid in PLAYERS]
    for a in agents:
        a.start()
    time.sleep(0.1)
    for a in agents:
        a.stop()
    for a in agents:
        a.join(timeout=5)
        assert not a.is_alive(), f"{a.name} did not stop cleanly"


# ---------------------------------------------------------------------------
# 2. Reply on the next tick (LLM stubbed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("player_id,persona,expected_reply", [
    ("ai_0", bai,       bai.DEFAULT_REPLY),
    ("ai_1", fox,       fox.DEFAULT_REPLY),
    ("ai_2", ironface,  ironface.DEFAULT_REPLY),
])
def test_agent_replies_on_next_tick(player_id, persona, expected_reply, state):
    """Send a message to an agent; it should reply on the next test tick."""
    sender = next(p for p in PLAYERS if p != player_id)

    with patch.object(
        AIAgent,
        "_pick_stance",
        return_value="cooperate",
    ):
        agent = AIAgent(
            player_id=player_id,
            persona=persona,
            state=state,
            llm_client=MockLLMClient(),
            tick_seconds=0.01,
        )
        agent.start()
        try:
            # Deliver a message to the agent's inbox.
            state.send_message(sender, player_id, "Hello, want to chat?")

            deadline = time.time() + 3.0
            reply_received = False
            while time.time() < deadline:
                msgs = state.get_my_messages(sender)
                for m in msgs:
                    if isinstance(m, ChatMessage) and m.sender == player_id:
                        reply_received = True
                        assert m.text == expected_reply, (
                            f"{player_id} replied {m.text!r}, expected {expected_reply!r}"
                        )
                        break
                if reply_received:
                    break
                time.sleep(0.05)

            assert reply_received, (
                f"{player_id} did not reply within 3 seconds"
            )
        finally:
            agent.stop()
            agent.join(timeout=5)


# ---------------------------------------------------------------------------
# 3. API key resolution (st.secrets → env var)
# ---------------------------------------------------------------------------

def test_api_key_from_env(monkeypatch):
    """llm_client reads the key from ANTHROPIC_API_KEY env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-from-env")

    # Streamlit not available in tests; ensure env fallback is used.
    import sys
    # Remove cached streamlit module so import fails cleanly inside _get_api_key
    st_mock = sys.modules.get("streamlit")
    sys.modules["streamlit"] = None  # make `import streamlit` raise ImportError

    try:
        from game import llm_client
        # Reload to pick up patched environment.
        import importlib
        importlib.reload(llm_client)
        key = llm_client._get_api_key()
        assert key == "test-key-from-env"
    finally:
        if st_mock is None:
            del sys.modules["streamlit"]
        else:
            sys.modules["streamlit"] = st_mock


def test_api_key_from_streamlit_secrets(monkeypatch):
    """llm_client prefers st.secrets["ANTHROPIC_API_KEY"] over env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

    fake_secrets = {"ANTHROPIC_API_KEY": "streamlit-key"}
    fake_st = MagicMock()
    fake_st.secrets = fake_secrets

    import sys
    import importlib
    sys.modules["streamlit"] = fake_st
    from game import llm_client
    importlib.reload(llm_client)

    try:
        key = llm_client._get_api_key()
        assert key == "streamlit-key"
    finally:
        del sys.modules["streamlit"]
        importlib.reload(llm_client)


# ---------------------------------------------------------------------------
# 4. Persona constants sanity checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("persona,expected_id", [
    (bai,       "ai_0"),
    (fox,       "ai_1"),
    (ironface,  "ai_2"),
])
def test_persona_ids(persona, expected_id):
    assert persona.PERSONA_ID == expected_id


@pytest.mark.parametrize("persona", [bai, fox, ironface])
def test_persona_has_required_attributes(persona):
    for attr in ("PERSONA_ID", "NAME", "EMOJI", "SYSTEM_PROMPT",
                 "DEFAULT_REPLY", "CHAT_INITIATIVE_PROB", "INITIAL_TRUST",
                 "TRUTH_REWARD", "LIE_PENALTY"):
        assert hasattr(persona, attr), f"{persona.__name__} missing {attr!r}"


def test_initiative_probs_ordering():
    """Bunny initiative > Fox > Stoneface (personality hierarchy)."""
    assert bai.CHAT_INITIATIVE_PROB > fox.CHAT_INITIATIVE_PROB
    assert fox.CHAT_INITIATIVE_PROB > ironface.CHAT_INITIATIVE_PROB


def test_personas_use_distinct_trust_profiles(state):
    memories = {
        pid: _make_agent(pid, state).memory
        for pid in PLAYERS
    }
    assert memories["ai_0"].initial_trust > memories["ai_1"].initial_trust
    assert memories["ai_1"].initial_trust > memories["ai_2"].initial_trust
    assert memories["ai_2"].lie_penalty > memories["ai_0"].lie_penalty


def test_visible_scores_hide_other_players(state):
    state.update_score("ai_0", 2)
    state.update_score("ai_1", 5)
    visible = _make_agent("ai_0", state)._visible_scores()
    assert visible["ai_0"] == 2
    assert visible["ai_1"] == "hidden"
    assert visible["ai_2"] == "hidden"


def test_direct_chat_stance_is_stable_within_a_round(state):
    state.set_round_info(1, 3, "poison_bottle")
    agent = _make_agent("ai_1", state)
    with patch(
        "game.ai_agent.random.choices",
        return_value=["deceive"],
    ) as choose:
        first = agent._pick_stance(
            trigger="chat_reply",
            counterpart="ai_0",
        )
        second = agent._pick_stance(
            trigger="chat_reply",
            counterpart="ai_0",
        )
        state.set_round_info(2, 3, "poison_bottle")
        third = agent._pick_stance(
            trigger="chat_reply",
            counterpart="ai_0",
        )
    assert first == second == third == "deceive"
    assert choose.call_count == 2


def test_private_game_action_never_uses_deception_stance(state):
    agent = _make_agent("ai_1", state)
    assert agent._pick_stance(trigger="game_decision") == "cooperate"


def test_decision_prompt_requests_summary_not_chain_of_thought():
    from prompts.templates import build_chat_reply_decision_prompt

    prompt = build_chat_reply_decision_prompt(
        sender_name="Bunny",
        incoming_text="Blue is safe",
        score=1,
        send_budget=5,
    )
    assert "GOAL:" in prompt
    assert "BELIEF:" in prompt
    assert "INTENT:" in prompt
    assert "REASONING:" not in prompt


def test_poison_picker_event_can_trigger_outbound(state):
    agent = _make_agent("ai_1", state)
    agent._round_state = {
        "game_type": "poison_bottle",
        "selection_order": PLAYERS,
        "picked_so_far": [],
    }
    with patch.object(
        agent,
        "_maybe_outbound_for_event",
    ) as outbound:
        agent._handle_event(GameEvent(
            "poison_picker_change",
            {
                "picker": "ai_0",
                "position": 1,
                "selection_order": PLAYERS,
            },
        ))
    assert outbound.call_args.kwargs["trigger"] == "poison_picker_change"


def test_private_poison_result_can_trigger_outbound(state):
    agent = _make_agent("ai_1", state)
    with patch.object(
        agent,
        "_maybe_outbound_for_event",
    ) as outbound:
        agent._handle_event(GameEvent(
            "poison_result",
            {"choice": "Green", "is_poisoned": False},
        ))
    assert outbound.call_args.kwargs["trigger"] == "private_poison_result"
    assert "Green bottle was safe" in outbound.call_args.kwargs["summary"]


def test_disproved_claim_can_trigger_round_end_outbound(state):
    state.set_round_info(1, 3, "poison_bottle")
    agent = _make_agent("ai_1", state)
    agent.memory.ingest_message(
        source="ai_0",
        text="I picked Red",
        round=1,
        game_type="poison_bottle",
    )
    with patch.object(
        agent,
        "_maybe_outbound_for_event",
    ) as outbound:
        agent._handle_event(GameEvent(
            "round_end",
            {
                "round": 1,
                "game_type": "poison_bottle",
                "results": {"choices": {"ai_0": "Blue"}},
                "score_deltas": {},
            },
        ))
    assert outbound.call_args.kwargs["trigger"] == "round_evidence_revealed"
    assert "Verified false claims" in outbound.call_args.kwargs["summary"]


# ---------------------------------------------------------------------------
# 5. Budget respected — agent stops sending when budget exhausted
# ---------------------------------------------------------------------------

def test_agent_stops_sending_when_budget_zero():
    """An agent with send_budget=0 must not attempt to send any messages."""
    tight_state = SharedState(PLAYERS, initial_send_budget=0)
    sender = "ai_1"

    with patch.object(AIAgent, "_pick_stance", return_value="cooperate"):
        agent = AIAgent(
            player_id="ai_0",
            persona=bai,
            state=tight_state,
            llm_client=MockLLMClient(),
            tick_seconds=0.01,
        )
        agent.start()
        try:
            # Send a message to ai_0 — it has 0 budget and should not reply.
            tight_state._inboxes["ai_0"].put(
                ChatMessage(sender=sender, recipient="ai_0", text="hello")
            )
            time.sleep(1.5)
            # ai_1's inbox should be empty — ai_0 couldn't reply.
            replies = tight_state.get_my_messages(sender)
            assert replies == [], f"Unexpected replies: {replies}"
        finally:
            agent.stop()
            agent.join(timeout=5)


def test_reserved_budget_blocks_proactive_when_low(state):
    """Agent should keep reserved budget for future replies."""
    agent = AIAgent(player_id="ai_0", persona=bai, state=state)
    # Spend ai_0 down to the reserved threshold.
    for _ in range(17):  # 20 -> 3
        state.send_message("ai_0", "ai_1", "x")
    assert state.get_send_budget("ai_0") == 3
    assert not agent._can_initiate_proactive()


def test_reserved_budget_does_not_block_evidence_trigger(state):
    """Meaningful events may use budget reserved from random initiative."""
    agent = AIAgent(player_id="ai_0", persona=bai, state=state)
    for _ in range(17):  # 20 -> 3
        state.send_message("ai_0", "ai_1", "x")
    with (
        patch.object(agent, "_pick_stance", return_value="cooperate"),
        patch.object(agent, "_outbound_two_step") as outbound,
    ):
        agent._maybe_outbound_for_event("round_evidence_revealed", "new fact")
    outbound.assert_called_once()
