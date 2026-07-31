"""Tests for per-game tracing and read-only replay queries."""

from concurrent.futures import ThreadPoolExecutor

from game.replay import ReplayService
from game.shared_state import GameEvent, SharedState
from game.trace import TraceRecorder


def test_trace_recorder_orders_concurrent_events():
    recorder = TraceRecorder("game-test")

    def record(index: int):
        recorder.record("tick", actor_id=f"ai_{index % 3}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(40)))

    events = recorder.snapshot()
    assert [event.sequence for event in events] == list(range(1, 41))
    assert {event.game_id for event in events} == {"game-test"}


def test_shared_state_records_replayable_events(tmp_path):
    recorder = TraceRecorder.persistent(tmp_path)
    state = SharedState(
        ["human", "ai_0"],
        human_player_id="human",
        trace_recorder=recorder,
    )
    state.set_round_info(1, 3, "guess_the_word")
    state.set_phase("playing")
    state.send_message("ai_0", "human", "mirror")
    state.submit_choice("ai_0", "mirror")
    state.update_score("ai_0", 1)
    state.broadcast_event(GameEvent("round_end", {"round": 1}))

    replay = ReplayService.from_jsonl(recorder.path)
    assert replay.game_id == recorder.game_id
    assert len(replay.events(actor_id="ai_0")) == 3
    assert replay.events(event_type="choice_submitted")[0].payload == {
        "choice": "mirror"
    }
    assert replay.events(event_type="score_updated")[0].payload["score"] == 1


def test_replay_summary_counts_tool_outcomes():
    recorder = TraceRecorder("summary-game")
    recorder.record(
        "tool_execution",
        round=1,
        payload={"is_error": True, "fallback": False},
    )
    recorder.record(
        "tool_execution",
        round=1,
        payload={"is_error": False, "fallback": True},
    )
    recorder.record("phase_changed", round=2, payload={"phase": "playing"})

    summary = ReplayService(recorder.snapshot()).summary()
    assert summary["rounds"] == [1, 2]
    assert summary["tool_calls"] == 2
    assert summary["tool_errors"] == 1
    assert summary["tool_fallbacks"] == 1
    assert summary["event_count"] == 3


def test_replay_rejects_invalid_jsonl(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    try:
        ReplayService.from_jsonl(path)
    except ValueError as exc:
        assert "line 1" in str(exc)
    else:
        raise AssertionError("invalid replay should fail")
