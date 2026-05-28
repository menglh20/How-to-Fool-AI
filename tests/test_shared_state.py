"""Tests for game/shared_state.py"""

import threading
import pytest

from game.shared_state import ChatMessage, GameEvent, SharedState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PLAYERS = ["ai_0", "ai_1", "ai_2"]


@pytest.fixture
def state():
    return SharedState(PLAYERS, initial_score=0, initial_send_budget=50)


# ---------------------------------------------------------------------------
# Concurrent score updates
# ---------------------------------------------------------------------------

def test_concurrent_update_score_no_error(state):
    """Parallel threads updating scores must not raise or corrupt state."""
    errors = []

    def worker(player_id, delta, n):
        for _ in range(n):
            try:
                state.update_score(player_id, delta)
            except Exception as exc:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(pid, 1, 200))
        for pid in PLAYERS
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent updates: {errors}"
    # Each player did 200 increments of 1 → score must be exactly 200.
    for pid in PLAYERS:
        assert state.get_score(pid) == 200


# ---------------------------------------------------------------------------
# Message isolation
# ---------------------------------------------------------------------------

def test_message_isolation(state):
    """ai_0 must NOT receive messages exchanged between ai_1 and ai_2."""
    state.send_message("ai_1", "ai_2", "hello from 1 to 2")
    state.send_message("ai_2", "ai_1", "hello from 2 to 1")

    ai_0_messages = state.get_my_messages("ai_0")
    assert ai_0_messages == [], (
        f"ai_0 received unexpected messages: {ai_0_messages}"
    )

    # Sanity: the correct recipients got their messages.
    ai_1_msgs = state.get_my_messages("ai_1")
    ai_2_msgs = state.get_my_messages("ai_2")
    assert len(ai_1_msgs) == 1
    assert len(ai_2_msgs) == 1
    assert isinstance(ai_1_msgs[0], ChatMessage)
    assert ai_1_msgs[0].sender == "ai_2"
    assert isinstance(ai_2_msgs[0], ChatMessage)
    assert ai_2_msgs[0].sender == "ai_1"


def test_message_isolation_self_send(state):
    """A player sending to themselves doesn't pollute others' inboxes."""
    state.send_message("ai_0", "ai_0", "note to self")
    assert state.get_my_messages("ai_1") == []
    assert state.get_my_messages("ai_2") == []
    msgs = state.get_my_messages("ai_0")
    assert len(msgs) == 1
    assert msgs[0].text == "note to self"


# ---------------------------------------------------------------------------
# send_budget is deducted on send, not on receive
# ---------------------------------------------------------------------------

def test_send_deducts_budget(state):
    """Each send_message call costs 1 from the sender's budget."""
    initial = state.get_send_budget("ai_0")
    state.send_message("ai_0", "ai_1", "msg1")
    state.send_message("ai_0", "ai_2", "msg2")
    assert state.get_send_budget("ai_0") == initial - 2


def test_receive_does_not_deduct_budget(state):
    """get_my_messages must not change any player's send_budget."""
    state.send_message("ai_1", "ai_0", "incoming")
    budget_before = state.get_send_budget("ai_0")

    state.get_my_messages("ai_0")

    assert state.get_send_budget("ai_0") == budget_before


def test_budget_exhaustion_raises(state):
    """Sending when budget is 0 must raise RuntimeError."""
    tight = SharedState(PLAYERS, initial_send_budget=1)
    tight.send_message("ai_0", "ai_1", "last message")
    with pytest.raises(RuntimeError):
        tight.send_message("ai_0", "ai_1", "this should fail")


def test_unlimited_budget(state):
    """initial_send_budget=None means unlimited sends."""
    unlimited = SharedState(PLAYERS, initial_send_budget=None)
    for i in range(100):
        unlimited.send_message("ai_0", "ai_1", f"msg {i}")
    assert unlimited.get_send_budget("ai_0") is None


# ---------------------------------------------------------------------------
# broadcast_event
# ---------------------------------------------------------------------------

def test_broadcast_event_reaches_all_players(state):
    event = GameEvent(event_type="round_start", payload={"round": 1})
    state.broadcast_event(event)

    for pid in PLAYERS:
        items = state.get_my_messages(pid)
        assert len(items) == 1
        assert isinstance(items[0], GameEvent)
        assert items[0].event_type == "round_start"


def test_broadcast_event_independent_objects(state):
    """Each player's inbox gets the same event object (put by reference)."""
    event = GameEvent(event_type="ping", payload=None)
    state.broadcast_event(event)

    items = [state.get_my_messages(pid)[0] for pid in PLAYERS]
    # All references point to the same object — that is acceptable and expected.
    assert all(item is event for item in items)


# ---------------------------------------------------------------------------
# Unknown player errors
# ---------------------------------------------------------------------------

def test_unknown_player_send_raises():
    s = SharedState(["ai_0"])
    with pytest.raises(KeyError):
        s.send_message("ghost", "ai_0", "hi")


def test_unknown_player_get_messages_raises():
    s = SharedState(["ai_0"])
    with pytest.raises(KeyError):
        s.get_my_messages("ghost")
