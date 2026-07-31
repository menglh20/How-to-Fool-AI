"""Tests for transport-independent read-only MCP operations."""

import pytest

from game.mcp_tools import MCPToolService
from game.replay import ReplayRepository
from game.trace import TraceRecorder


def service_with_trace(tmp_path):
    recorder = TraceRecorder.persistent(tmp_path)
    recorder.record(
        "chat_message",
        actor_id="ai_0",
        round=1,
        payload={
            "recipient": "human",
            "text": "secret plan",
            "thinking": "private",
        },
    )
    recorder.record(
        "tool_execution",
        actor_id="ai_0",
        round=1,
        payload={
            "tool": "send_message",
            "arguments": {
                "target": "human",
                "content": "private rejected text",
            },
            "is_error": True,
            "fallback": False,
        },
    )
    return MCPToolService(ReplayRepository(tmp_path)), recorder


def test_mcp_replay_queries_default_to_redacted(tmp_path):
    service, recorder = service_with_trace(tmp_path)
    result = service.get_replay_events(recorder.game_id)
    chat = result["items"][0]["payload"]
    assert chat["text"] == "[PRIVATE MESSAGE REDACTED]"
    assert "thinking" not in chat

    sensitive = service.get_agent_trace(
        recorder.game_id, "ai_0", include_sensitive=True
    )
    assert sensitive["items"][0]["payload"]["text"] == "secret plan"


def test_mcp_lists_summarizes_and_filters_failures(tmp_path):
    service, recorder = service_with_trace(tmp_path)
    assert service.list_replays()["items"][0]["game_id"] == recorder.game_id
    assert service.get_replay_summary(recorder.game_id)["tool_errors"] == 1
    failures = service.get_tool_failures(recorder.game_id)
    assert failures["total"] == 1
    assert failures["items"][0]["payload"]["tool"] == "send_message"
    assert (
        failures["items"][0]["payload"]["arguments"]["content"]
        == "[PRIVATE MESSAGE REDACTED]"
    )


@pytest.mark.parametrize(
    "game_id",
    ["../secret", "ABC", "a" * 31, "g" * 32],
)
def test_mcp_rejects_invalid_game_ids(tmp_path, game_id):
    service = MCPToolService(ReplayRepository(tmp_path))
    with pytest.raises(ValueError):
        service.get_replay_summary(game_id)


def test_mcp_pagination_is_bounded(tmp_path):
    service, recorder = service_with_trace(tmp_path)
    with pytest.raises(ValueError):
        service.get_replay_events(recorder.game_id, limit=501)
    page = service.get_replay_events(
        recorder.game_id, offset=1, limit=1
    )
    assert page["total"] == 2
    assert len(page["items"]) == 1
