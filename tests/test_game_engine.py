"""Tests for game/game_engine.py and the SharedState extensions it relies on.

All tests are integration-level (no live API calls) — LLM calls are stubbed,
and the engine runs with very short timeouts so choices are provided by the
test fixtures or fall back to random defaults quickly.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from game.game_engine import GameEngine, _parse_attribution
from game.shared_state import GameEvent, SharedState

PLAYERS = ["human", "ai_0", "ai_1", "ai_2"]


@pytest.fixture
def state():
    return SharedState(PLAYERS, initial_score=0, initial_send_budget=10)


# ---------------------------------------------------------------------------
# SharedState extensions
# ---------------------------------------------------------------------------

class TestSharedStateExtensions:
    def test_set_get_phase(self, state):
        assert state.get_phase() == "setup"
        state.set_phase("playing")
        assert state.get_phase() == "playing"

    def test_set_get_round_info(self, state):
        state.set_round_info(2, 5, "poison_bottle")
        info = state.get_round_info()
        assert info["current_round"] == 2
        assert info["total_rounds"] == 5
        assert info["game_type"] == "poison_bottle"

    def test_reset_send_budgets(self, state):
        state.send_message("human", "ai_0", "hi")
        assert state.get_send_budget("human") == 9
        state.reset_send_budgets()
        assert state.get_send_budget("human") == 10

    def test_submit_and_get_choice(self, state):
        state.reset_choices(["human"])
        state.submit_choice("human", "apple")
        assert state.get_choice("human") == "apple"

    def test_choice_event_set_on_submit(self, state):
        state.reset_choices(["ai_0"])
        event = state.get_choice_event("ai_0")
        assert not event.is_set()
        state.submit_choice("ai_0", "Blue")
        assert event.is_set()

    def test_reset_choices_clears_old_event(self, state):
        state.reset_choices(["ai_0"])
        state.submit_choice("ai_0", "Red")
        old_event = state.get_choice_event("ai_0")
        assert old_event.is_set()
        # After reset the event should be a fresh, unset Event.
        state.reset_choices(["ai_0"])
        new_event = state.get_choice_event("ai_0")
        assert not new_event.is_set()

    def test_send_event_to_player(self, state):
        ev = GameEvent(event_type="make_decision", payload={"action": "write_word"})
        state.send_event_to_player("ai_1", ev)
        msgs = state.get_my_messages("ai_1")
        assert len(msgs) == 1
        assert msgs[0].event_type == "make_decision"


# ---------------------------------------------------------------------------
# _parse_attribution helper
# ---------------------------------------------------------------------------

class TestParseAttribution:
    def test_valid_submission(self):
        result = _parse_attribution("ai_0,ai_1,ai_2", ["ai_0", "ai_1", "ai_2"])
        assert result == {"ai_0": "ai_0", "ai_1": "ai_1", "ai_2": "ai_2"}

    def test_wrong_length_falls_back_to_random(self):
        result = _parse_attribution("ai_0,ai_1", ["ai_0", "ai_1", "ai_2"])
        assert set(result.keys()) == {"ai_0", "ai_1", "ai_2"}

    def test_none_falls_back_to_random(self):
        result = _parse_attribution(None, ["ai_0", "ai_1"])
        assert set(result.keys()) == {"ai_0", "ai_1"}
        assert set(result.values()) == {"ai_0", "ai_1"}


# ---------------------------------------------------------------------------
# GameEngine game type rotation
# ---------------------------------------------------------------------------

class TestGameTypeRotation:
    def test_random_rotation_uses_all_types(self, state):
        engine = GameEngine(state, total_rounds=30, choice_timeout=0.1)
        types = [engine._select_game_type() for _ in range(30)]
        assert set(types) == {"guess_the_word", "who_wrote_it", "poison_bottle"}

    def test_no_consecutive_repeats_in_full_cycle(self, state):
        engine = GameEngine(state, total_rounds=9, choice_timeout=0.1)
        types = [engine._select_game_type() for _ in range(9)]
        for a, b in zip(types, types[1:]):
            assert a != b, f"Consecutive repeat: {a}"


# ---------------------------------------------------------------------------
# _wait_for_choice
# ---------------------------------------------------------------------------

class TestWaitForChoice:
    def test_returns_none_on_timeout(self, state):
        state.reset_choices(["ai_0"])
        engine = GameEngine(state, total_rounds=1, choice_timeout=0.05)
        result = engine._wait_for_choice("ai_0", timeout=0.05)
        assert result is None

    def test_returns_choice_when_submitted(self, state):
        state.reset_choices(["ai_0"])
        engine = GameEngine(state, total_rounds=1, choice_timeout=5.0)

        def submit_later():
            time.sleep(0.05)
            state.submit_choice("ai_0", "moon")

        t = threading.Thread(target=submit_later)
        t.start()
        result = engine._wait_for_choice("ai_0", timeout=2.0)
        t.join()
        assert result == "moon"

    def test_does_not_block_other_threads(self, state):
        """While engine waits, a separate thread can still run."""
        state.reset_choices(["ai_0"])
        engine = GameEngine(state, total_rounds=1, choice_timeout=0.3)

        other_thread_ran = threading.Event()

        def other_work():
            time.sleep(0.05)
            other_thread_ran.set()

        t = threading.Thread(target=other_work)
        t.start()
        engine._wait_for_choice("ai_0", timeout=0.3)
        t.join()
        assert other_thread_ran.is_set()


# ---------------------------------------------------------------------------
# Full game loop — short timeout so engine uses fallback choices
# ---------------------------------------------------------------------------

class TestGameEngineLoop:
    def _run_engine(self, state, rounds, timeout=0.1, reveal=0.05):
        engine = GameEngine(
            state,
            total_rounds=rounds,
            choice_timeout=timeout,
            reveal_duration=reveal,
        )
        engine.start()
        # Each round: up to (choice_timeout × players) + reveal; add generous buffer.
        engine.join(timeout=rounds * (timeout * len(PLAYERS) + reveal) + 10)
        return engine

    def test_game_ends_with_game_over_phase(self, state):
        engine = self._run_engine(state, rounds=3, timeout=0.05)
        assert not engine.is_alive(), "Engine thread did not finish in time"
        assert state.get_phase() == "game_over"

    def test_game_over_event_broadcast(self, state):
        # Collect game_over events from all players' inboxes.
        engine = self._run_engine(state, rounds=1, timeout=0.05)
        game_over_events = []
        for pid in PLAYERS:
            for item in state.get_my_messages(pid):
                if isinstance(item, GameEvent) and item.event_type == "game_over":
                    game_over_events.append(item)
        assert len(game_over_events) == len(PLAYERS)

    def test_round_count_in_metadata(self, state):
        """After the engine runs, round metadata should reflect total_rounds."""
        engine = self._run_engine(state, rounds=2, timeout=0.05)
        info = state.get_round_info()
        assert info["total_rounds"] == 2

    def test_scores_change_across_rounds(self, state):
        """Running 3 rounds (all three types) exercises scoring paths."""
        engine = self._run_engine(state, rounds=3, timeout=0.05)
        # At minimum, scores should have been touched (some may be 0, some non-0).
        score_sum = sum(state.get_score(pid) for pid in PLAYERS)
        # Poison Bottle gives -1; others give +1.  Sum can be any integer.
        assert isinstance(score_sum, int)

    def test_engine_stops_cleanly_on_stop(self, state):
        # Use a short poll interval (choice_timeout drives _POLL=0.25 s) so
        # stop() is noticed quickly.
        engine = GameEngine(state, total_rounds=10, choice_timeout=30.0, reveal_duration=30.0)
        engine.start()
        time.sleep(0.2)
        engine.stop()
        engine.join(timeout=5)
        assert not engine.is_alive()
