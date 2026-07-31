"""Local MCP server for replay inspection and evaluation operations."""

from __future__ import annotations

import os
from pathlib import Path

from evals.jobs import EvaluationManager
from game.mcp_tools import MCPToolService
from game.replay import ReplayRepository

try:
    # MCP SDK 2.x renamed the high-level server while retaining the same
    # decorator-based registration API.
    from mcp.server import MCPServer as MCPServerClass
except (ImportError, ModuleNotFoundError):
    try:
        # Keep the project usable with MCP SDK 1.x installations.
        from mcp.server.fastmcp import FastMCP as MCPServerClass
    except (ImportError, ModuleNotFoundError):
        MCPServerClass = None


def build_server(trace_directory: Path | None = None):
    if MCPServerClass is None:
        raise RuntimeError(
            "The MCP SDK is not installed; run pip install -r requirements.txt"
        )
    directory = trace_directory or Path(
        os.environ.get("REPLAY_TRACE_DIR", "logs/traces")
    )
    repository = ReplayRepository(directory)
    report_directory = Path(
        os.environ.get(
            "EVALUATION_REPORT_DIR",
            "reports/evaluation/runs",
        )
    )
    service = MCPToolService(
        repository,
        EvaluationManager(repository, report_directory),
    )
    server = MCPServerClass("How to Fool AI")

    server.tool()(service.list_replays)
    server.tool()(service.get_replay_summary)
    server.tool()(service.get_replay_events)
    server.tool()(service.get_agent_trace)
    server.tool()(service.get_tool_failures)
    server.tool()(service.start_evaluation)
    server.tool()(service.get_evaluation_status)
    server.tool()(service.get_evaluation_report)
    server.tool()(service.cancel_evaluation)
    server.tool()(service.compare_evaluations)
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
