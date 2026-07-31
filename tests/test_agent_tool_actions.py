"""End-to-end Agent tests for structured game and chat tools."""

import time
from unittest.mock import patch

from game.ai_agent import AIAgent
from game.llm_client import (
    LLMRequest,
    LLMResponse,
    ScriptedLLMClient,
    ToolCall,
)
from game.shared_state import ChatMessage, GameEvent, SharedState
import prompts.fox as fox


PLAYERS = ["human", "ai_0", "ai_1", "ai_2"]


class CapturingScriptedLLMClient(ScriptedLLMClient):
    def __init__(self, responses):
        super().__init__(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return super().complete(request)


def tool_response(name: str, arguments: dict, call_id: str = "call-1"):
    return LLMResponse(
        tool_calls=(ToolCall(call_id, name, arguments),),
        provider="scripted",
        model="scripted",
    )


def agent_with(responses) -> tuple[AIAgent, SharedState]:
    state = SharedState(
        PLAYERS,
        initial_send_budget=10,
        human_player_id="human",
    )
    state.set_round_info(1, 3, "poison_bottle")
    state.set_phase("playing")
    client = CapturingScriptedLLMClient(responses)
    agent = AIAgent(
        "ai_1",
        fox,
        state,
        llm_client=client,
        tick_seconds=0.01,
    )
    return agent, state


def test_choose_bottle_executes_structured_tool():
    agent, state = agent_with([
        "Pick Blue based on the current evidence.",
        tool_response(
            "choose_bottle",
            {"bottle": "Blue", "decision_summary": "Blue looks safer"},
        ),
    ])
    agent._decide_choose_bottle({
        "game_type": "poison_bottle",
        "available_bottles": ["Red", "Blue"],
    })
    assert state.get_choice("ai_1") == "Blue"


def test_action_prompt_receives_the_actual_decision():
    decision = "GOAL: survive\nBELIEF: Blue is safer\nPLAN: choose Blue"
    agent, state = agent_with([
        decision,
        tool_response("choose_bottle", {"bottle": "Blue"}),
    ])
    agent._decide_choose_bottle({
        "game_type": "poison_bottle",
        "available_bottles": ["Red", "Blue", "Green", "Yellow"],
    })
    action_request = agent.llm_client.requests[1]
    assert decision in action_request.messages[0]["content"]
    assert "(decision supplied in prior phase)" not in (
        action_request.messages[0]["content"]
    )


def test_invalid_tool_call_is_repaired_once():
    agent, state = agent_with([
        "Choose a legal bottle.",
        tool_response("choose_bottle", {"bottle": "Green"}, "bad-call"),
        tool_response("choose_bottle", {"bottle": "Red"}, "fixed-call"),
    ])
    agent._decide_choose_bottle({
        "game_type": "poison_bottle",
        "available_bottles": ["Red", "Blue"],
    })
    assert state.get_choice("ai_1") == "Red"


def test_submit_and_guess_word_tools_normalize_words():
    writer, writer_state = agent_with([
        "Choose a distinctive word.",
        tool_response("submit_word", {"word": "Mirror"}),
    ])
    writer_state.set_round_info(1, 3, "guess_the_word")
    writer._decide_write_word({"game_type": "guess_the_word"})
    assert writer_state.get_choice("ai_1") == "mirror"

    guesser, guess_state = agent_with([
        "Guess from the available evidence.",
        tool_response("guess_word", {"word": "Thunder"}),
    ])
    guess_state.set_round_info(1, 3, "guess_the_word")
    guesser._decide_guess_word({"game_type": "guess_the_word"})
    assert guess_state.get_choice("ai_1") == "thunder"


def test_attribute_words_tool_submits_ordered_authors():
    agent, state = agent_with([
        "Map each word to one candidate.",
        tool_response(
            "attribute_words",
            {"authors": ["human", "ai_0", "ai_2"]},
        ),
    ])
    state.set_round_info(1, 3, "who_wrote_it")
    agent._decide_guess_authors({
        "game_type": "who_wrote_it",
        "words": ["echo", "apple", "stone"],
        "candidate_authors": ["human", "ai_0", "ai_2"],
    })
    assert state.get_choice("ai_1") == "human,ai_0,ai_2"


def test_chat_reply_uses_send_message_tool():
    agent, state = agent_with([
        "Reply briefly and probe for information.",
        tool_response(
            "send_message",
            {
                "target": "human",
                "content": "Which bottle did you choose?",
                "decision_summary": "Verify the human's claim",
            },
        ),
    ])
    agent._handle_chat(
        ChatMessage("human", "ai_1", "I picked the safe bottle")
    )
    messages = state.get_my_messages("human")
    assert len(messages) == 1
    assert messages[0].text == "Which bottle did you choose?"
    history = state.get_chat_history()
    assert history[-1]["thinking"] == "Verify the human's claim"


def test_direct_chat_sends_without_typing_wait():
    content = "Check the blue bottle"
    agent, state = agent_with([
        "GOAL: verify the claim",
        tool_response(
            "send_message",
            {"target": "human", "content": content},
        ),
    ])
    with patch.object(
        agent._stop_event,
        "wait",
        return_value=False,
    ) as wait:
        agent._handle_chat(ChatMessage("human", "ai_1", "Blue is safe"))
    wait.assert_not_called()
    assert state.get_my_messages("human")[0].text == content


def test_default_tick_is_ten_seconds():
    state = SharedState(PLAYERS)
    agent = AIAgent("ai_1", fox, state, llm_client=ScriptedLLMClient([]))
    assert agent.tick_seconds == 10


def test_chat_cannot_send_to_unexposed_target():
    agent, state = agent_with([
        "Try another target.",
        tool_response(
            "send_message",
            {"target": "ai_0", "content": "Leak"},
            "bad-target",
        ),
        tool_response("stay_silent", {"reason": "target not allowed"}),
    ])
    agent._handle_chat(ChatMessage("human", "ai_1", "hello"))
    assert state.get_my_messages("human") == []
    assert state.get_my_messages("ai_0") == []


def test_expired_game_action_is_rejected_even_by_fallback():
    agent, state = agent_with([
        "Choose after the deadline.",
        tool_response("choose_bottle", {"bottle": "Blue"}),
        tool_response("choose_bottle", {"bottle": "Red"}),
    ])
    agent._decide_choose_bottle({
        "game_type": "poison_bottle",
        "available_bottles": ["Red", "Blue"],
        "deadline_ts": time.time() - 1,
    })
    assert state.get_choice("ai_1") is None
    assert agent.llm_client.requests == []


def test_pending_decision_from_previous_round_is_discarded_without_llm_calls():
    agent, state = agent_with([
        "This response must never be consumed.",
    ])
    state.set_round_info(1, 3, "guess_the_word")
    agent._handle_event(GameEvent(
        "make_decision",
        {
            "game_type": "guess_the_word",
            "action": "guess_word",
            "round": 1,
            "writer": "human",
            "deadline_ts": time.time() + 5,
        },
    ))
    assert "guess_word" in agent._pending_decisions

    state.set_round_info(2, 3, "poison_bottle")
    agent._resolve_pending_decisions()

    assert agent.llm_client.requests == []
    assert "guess_word" not in agent._pending_decisions


def test_deadline_expiring_during_decision_skips_action_llm_call():
    agent, state = agent_with([
        "Choose Blue before time runs out.",
        tool_response("choose_bottle", {"bottle": "Blue"}),
    ])
    checks = iter([None, ValueError("Decision deadline has passed")])

    def check_window(_payload):
        result = next(checks)
        if isinstance(result, Exception):
            raise result

    with patch.object(agent, "_require_decision_window", side_effect=check_window):
        agent._decide_choose_bottle({
            "game_type": "poison_bottle",
            "available_bottles": ["Red", "Blue"],
            "deadline_ts": time.time() + 10,
        })

    assert len(agent.llm_client.requests) == 1
    assert state.get_choice("ai_1") is None


def test_game_action_is_rejected_in_wrong_game_type():
    agent, state = agent_with([
        "Try a bottle action in another game.",
        tool_response("choose_bottle", {"bottle": "Blue"}),
        tool_response("choose_bottle", {"bottle": "Red"}),
    ])
    state.set_round_info(1, 3, "guess_the_word")
    agent._decide_choose_bottle({
        "game_type": "poison_bottle",
        "available_bottles": ["Red", "Blue"],
    })
    assert state.get_choice("ai_1") is None


def test_multiple_tool_calls_are_rejected_and_repaired():
    multiple = LLMResponse(
        tool_calls=(
            ToolCall("one", "choose_bottle", {"bottle": "Red"}),
            ToolCall("two", "choose_bottle", {"bottle": "Blue"}),
        ),
        provider="scripted",
        model="scripted",
    )
    agent, state = agent_with([
        "Choose exactly once.",
        multiple,
        tool_response("choose_bottle", {"bottle": "Blue"}),
    ])
    agent._decide_choose_bottle({
        "game_type": "poison_bottle",
        "available_bottles": ["Red", "Blue"],
    })
    assert state.get_choice("ai_1") == "Blue"


def test_tick_batches_observations_and_sends_at_most_one_message():
    agent, state = agent_with([
        tool_response(
            "act_in_tick",
            {
                "decision_summary": "Answer the latest human question",
                "chat_action": {
                    "target": "human",
                    "content": "Blue still looks safest.",
                },
            },
        ),
    ])
    state.send_message("ai_0", "ai_1", "I chose Red")
    state.send_message("human", "ai_1", "Which bottle is safest?")

    with patch.object(agent, "_pick_stance", return_value="cooperate"):
        agent._tick()

    messages = state.get_my_messages("human")
    assert [message.text for message in messages] == ["Blue still looks safest."]
    assert len(agent.llm_client.requests) == 1
    prompt = agent.llm_client.requests[0].messages[0]["content"]
    assert "I chose Red" in prompt
    assert "Which bottle is safest?" in prompt


def test_tick_executes_game_and_chat_from_one_model_call():
    agent, state = agent_with([
        tool_response(
            "act_in_tick",
            {
                "decision_summary": "Submit a word and answer the human",
                "game_action": {
                    "name": "submit_word",
                    "arguments": {"word": "Mirror"},
                },
                "chat_action": {
                    "target": "human",
                    "content": "I have made my choice.",
                },
            },
        ),
    ])
    state.set_round_info(1, 3, "guess_the_word")
    state._inboxes["ai_1"].put(GameEvent(
        "make_decision",
        {
            "game_type": "guess_the_word",
            "action": "write_word",
            "round": 1,
            "deadline_ts": time.time() + 30,
        },
    ))
    state.send_message("human", "ai_1", "Have you chosen?")

    with patch.object(agent, "_pick_stance", return_value="cooperate"):
        agent._tick()

    assert state.get_choice("ai_1") == "mirror"
    assert state.get_my_messages("human")[0].text == "I have made my choice."
    assert len(agent.llm_client.requests) == 1


def test_invalid_tick_chat_does_not_block_game_action():
    agent, state = agent_with([
        tool_response(
            "act_in_tick",
            {
                "game_action": {
                    "name": "submit_word",
                    "arguments": {"word": "Compass"},
                },
                "chat_action": {
                    "target": "ai_0",
                    "content": "This target is not allowed.",
                },
            },
        ),
    ])
    state.set_round_info(1, 3, "guess_the_word")
    state._inboxes["ai_1"].put(GameEvent(
        "make_decision",
        {
            "game_type": "guess_the_word",
            "action": "write_word",
            "round": 1,
            "deadline_ts": time.time() + 30,
        },
    ))
    state.send_message("human", "ai_1", "Tell me your plan")

    with patch.object(agent, "_pick_stance", return_value="cooperate"):
        agent._tick()

    assert state.get_choice("ai_1") == "compass"
    assert state.get_my_messages("human") == []
    assert state.get_my_messages("ai_0") == []


def test_idle_tick_skips_llm_call():
    agent, _ = agent_with([])
    with patch("game.ai_agent.random.random", return_value=1.0):
        agent._tick()
    assert agent.llm_client.requests == []
