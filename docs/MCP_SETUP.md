# MCP Setup

The MCP server uses stdio and exposes replay inspection plus asynchronous
evaluation operations.

## Start manually

```bash
pip install -r requirements.txt
python mcp_server.py
```

Python 3.11+ is supported. The adapter accepts both the MCP Python SDK 1.x
`FastMCP` surface and the MCP 2.x `MCPServer` surface.

Optional environment variables:

```text
REPLAY_TRACE_DIR=/absolute/path/to/logs/traces
EVALUATION_REPORT_DIR=/absolute/path/to/reports/evaluation/runs
```

Configure an MCP client to launch the same command with the repository as its
working directory. Keep the server local unless authentication and transport
security are added.

## Replay tools

- `list_replays`
- `get_replay_summary`
- `get_replay_events`
- `get_agent_trace`
- `get_tool_failures`

Private messages and model text are redacted by default. Reading them requires
the explicit `include_sensitive=true` argument.

## Evaluation tools

- `start_evaluation`
- `get_evaluation_status`
- `get_evaluation_report`
- `cancel_evaluation`
- `compare_evaluations`

`start_evaluation` returns immediately with a `run_id`. Poll status instead of
holding one MCP call open. An idempotency key prevents duplicate submissions.

## Security boundary

The MCP does not expose live `SharedState` or internal game mutation tools.
Replay IDs must be 32 lowercase hexadecimal characters, trace paths are
confined to the configured directory, pagination is bounded, and sensitive
content is hidden by default.
