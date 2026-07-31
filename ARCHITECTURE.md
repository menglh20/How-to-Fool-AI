# Architecture

## Runtime

The game has five concurrent execution units:

1. Streamlit main thread for human input and rendering.
2. One `GameEngine` daemon thread for rounds, phases, and scoring.
3. Three `AIAgent` daemon threads for independent AI players.

All game coordination passes through a single `SharedState`. Its mutable state
is protected by `threading.RLock`; messages and events use per-player queues;
human and AI decisions signal per-player `threading.Event` objects.

## LLM boundary

`AIAgent` depends on the provider-neutral
`LLMClient.complete(LLMRequest) -> LLMResponse` interface. It does not import
or construct an Anthropic/OpenAI SDK client.

```text
AIAgent
  └── LLMClient
       ├── MockLLMClient       local development and unit tests
       ├── ScriptedLLMClient   deterministic evals and replay
       ├── AnthropicLLMClient  online Anthropic API
       └── OpenAILLMClient     online OpenAI Responses API
```

The Streamlit setup page configures one client before starting the agent
threads. Each agent captures that client for the lifetime of the game, avoiding
mid-game provider changes and making tests explicit.

Provider SDKs are lazy imports. Mock and Scripted modes therefore work without
network access, API keys, or provider packages. Online backends fail fast at
setup when required credentials or model configuration are absent; individual
request failures return the supplied safe fallback and are written to the LLM
log.

### Persona boundary

`PersonaConfig` in `game/persona.py` is the validated boundary between the
setup UI and `AIAgent`. The player chooses three distinct configs before
starting; each Agent receives one immutable config for the lifetime of that
game. Bunny, Fox, and Stoneface are ordinary default configs, not hard-coded
Agent-slot behavior.

The Streamlit session holds an editable persona library and supports JSON
import/export. Imported objects have an explicit schema version, size limit,
exact fields, bounded text and numeric values, and stance weights that total
1.0. No database is involved.

The final system prompt concatenates the player-defined persona, speaking style
and message-length guidance, then appends non-negotiable identity, security,
state-integrity, content-safety, and chat-length rules. Numeric stance, trust,
and initiative controls are consumed directly by the Agent rather than
translated into prompt prose.

### Unified protocol

`LLMRequest` contains:

- system prompt and provider-neutral conversation messages;
- maximum output tokens and a safe fallback;
- zero or more JSON Schema `ToolDefinition` objects;
- `tool_choice` (`auto`, `required`, or `none`);
- prior `ToolResult` objects for a tool continuation turn;
- trace metadata such as agent, trigger, phase, and stance.

`LLMResponse` contains:

- generated text;
- normalized `ToolCall` objects;
- input, output, and total token usage;
- provider, model, latency, finish reason, and optional error.

Anthropic tool-use blocks and OpenAI function-call items are converted into
the same `ToolCall(id, name, arguments)` representation. Tool results are
converted back to each provider's native continuation format.

### Tool runtime

`ToolRegistry` in `game/tool_runtime.py` separates untrusted model output from
trusted application handlers. Before dispatch it verifies:

- the tool is registered;
- all required arguments are present;
- unknown arguments are rejected when the schema forbids them;
- primitive argument types and enum constraints match the schema.

Execution always produces a `ToolResult`; unknown tools, invalid arguments,
and handler exceptions become error results instead of escaping into the game
thread.

### Game tool migration

Every AI action now uses a structured tool rather than parsing free-form model
text:

| Action scope | Allowed tools |
|---|---|
| Reply to a DM | `send_message` to the sender, `stay_silent` |
| Proactive DM | `send_message` to another player, `stay_silent` |
| Write-word phase | `submit_word` |
| Guess-word phase | `guess_word` |
| Attribution phase | `attribute_words` |
| Poison phase | `choose_bottle` |

The tool schemas are built in `game/game_tools.py`. Dynamic values such as
legal message targets, candidate authors, and available bottles are encoded as
schema enums, so the provider sees only actions valid for that decision.

The live tick path validates each nested action independently. An invalid or
missing required game action uses a deterministic fallback through the same
registry; an invalid optional chat action is discarded. Focused direct-action
helpers retain one repair attempt for debugging. Game handlers additionally
check the current phase and decision deadline; chat handlers rely on
`SharedState` for send-budget enforcement.

Online model requests have a 30-second provider timeout. Tool execution records
include the tool, arguments, result, attempt number, validation outcome, and
whether a fallback was used.

## Agent flow

The agent keeps recent conversational context, round history, and a live
snapshot of the current round. Every 10 seconds it drains and batches the
events received since its previous tick, updates memory and state locally, and
uses one `act_in_tick` request only when a game or chat action is available.
That structured result contains at most one game action and one chat action.
The game action is validated and executed first; either action can fail without
cancelling the other. Game actions are submitted to `SharedState`; the engine
remains the authority that validates timing, computes results, and updates
scores.

Exact opponent scores are never included in model context. Poison Bottle uses
four returned bottles and exposes only score-derived selection order. Incoming
DMs are prioritized over round evidence, private poison results, other poison
pickers, phase changes, and periodic persona initiative. A tick can send at
most one 60-character DM. Event-driven opportunities may use the three-message
reply reserve; periodic initiative may not. Idle ticks make no LLM call, and
the three Agent schedules are staggered by one-third of a tick to avoid bursts.
Typing simulation is intentionally absent: the tick cadence plus provider
latency controls response frequency without blocking decision deadlines.

### Structured memory

`AgentMemory` separates short-lived working state from cross-round knowledge:

- working state tracks the current phase and active task;
- claim memory records who asserted a bottle, poison, or answer fact;
- episodic memory stores compact round outcomes;
- trust scores are updated only after a claim can be checked against round
  truth.

Each persona supplies a distinct initial trust, truth reward, and lie penalty.
Trust also adjusts the cooperate/deceive/silent stance distribution. A stance
toward the same counterpart remains stable within a round and is reconsidered
when new round evidence arrives.

Retrieval is relevance- and trust-aware and returns a bounded prompt summary.
Messages that resemble prompt injection are retained for auditing but marked
untrusted and down-weighted. This keeps memory inspectable and testable rather
than hiding it inside an ever-growing conversation string.

## Security boundaries

Model output and replay data are treated as untrusted:

- provider API keys are recursively redacted from logs and traces;
- model-generated chat is length checked and stripped of obvious secret
  material;
- prompt-injection-like claims cannot directly change tool permissions;
- each tick accepts one outer tool call containing at most one game action and
  one chat action;
- every action is checked against its JSON Schema and the current game state;
- malformed required game actions use a deterministic fallback; malformed
  optional chat actions are discarded;
- private replay content is hidden by default in external inspection tools.

Opaque replay IDs must match the generated 32-character hexadecimal form.
`ReplayRepository` resolves those IDs inside its configured trace directory,
so MCP callers cannot request arbitrary filesystem paths.

## Observability

Every online backend writes the same JSONL call record: provider, model,
agent/persona, trigger, phase, stance, latency, prompt, response, timestamp,
and optional error. Mock and Scripted clients disable logging by default so
tests do not mutate the workspace; evaluations can opt in with
`record_calls=True`.

### Per-game trace and replay

Every game created by the Streamlit application receives an opaque `game_id`
and a thread-safe `TraceRecorder`. Unlike the provider-oriented LLM log, this
trace joins the complete product flow under one ordered sequence:

- round and phase changes;
- targeted and broadcast game events;
- private messages and remaining send budget;
- submitted choices and score changes;
- normalized LLM responses, token usage, latency, and errors;
- validated tool executions, repair attempts, and fallbacks.

Events are kept in memory and appended to
`logs/traces/<game_id>.jsonl`. `ReplayService` loads either representation,
sorts by sequence, filters by event type, actor, or round, produces a timeline,
and summarizes tool errors and fallbacks. The game-over screen exposes these
metrics and lets the user download the JSONL replay.

Trace persistence is disabled unless a `TraceRecorder` is explicitly supplied
to `SharedState`, so unit tests and embedded uses do not mutate the workspace.
Trace files may contain private game messages and must not be served before
the post-game reveal without access control.

### Replay and evaluation MCP

`mcp_server.py` exposes a deliberately narrow, transport-independent service:

- read-only replay listing, summaries, filtered events, agent traces, and tool
  failures;
- asynchronous evaluation submission, progress, report retrieval,
  cancellation, and report comparison.

The MCP adapter contains no game mutation tools. `EvaluationManager` bounds
concurrency and queue size, assigns an opaque `run_id`, supports idempotency
keys, and persists completed reports under `reports/evaluation/runs`.

### Offline evaluation

`evals/runner.py` evaluates persisted traces without calling a model. It
measures completion, first-call tool legality, repair success, fallback rate,
LLM errors, latency percentiles, tokens, scores, and winners. Aggregation is a
macro-average across games. Pricing is opt-in; when no verified price is
provided the report does not invent a cost.

Deterministic scenario suites cover tool boundaries, security, and memory.
`evals/ablation.py` compares two real reports and labels deltas without claiming
statistical significance. JSON is the source artifact; Markdown is a
human-readable rendering.

### Developer dashboard

The Streamlit multipage dashboard in `pages/1_Observability.py` consumes the
same replay and comparison services as tests. It supports timeline filters,
tool failure inspection, latency/token summaries, state changes, and a
side-by-side comparison. Sensitive fields remain opt-in.

## Delivery pipeline

GitHub Actions runs on Python 3.12, executes the full pytest suite, then runs
the deterministic scenario evaluation. The same commands are documented in
`README.md` and `EVALUATION.md`; MCP setup and role-capability evidence are
kept in `docs/`.

## Tests

- `test_shared_state.py`: concurrency, isolation, budgets, and events.
- `test_game_engine.py`: phase transitions, rotation, choices, and loop.
- `test_scoring.py`: deterministic rules for all mini-games.
- `test_ai_agents.py`: agent lifecycle, reply behavior, and personas.
- `test_llm_clients.py`: zero-token Mock and deterministic Scripted backends,
  unified requests/responses, tool calls, configuration validation, and
  active-client selection.
- `test_tool_runtime.py`: tool registration, schema validation, safe dispatch,
  and error conversion.
- `test_agent_tool_actions.py`: structured chat/game actions, one-call batched
  ticks, independent action failure, target isolation, and deadlines.
- `test_replay.py`: concurrent trace ordering, SharedState integration,
  persistent replay loading, filtering, summaries, and malformed input.
- `test_mcp_tools.py`: constrained replay access, pagination, privacy defaults,
  and path validation.
- `test_evaluation.py`: trace metrics, aggregation, and report persistence.
- `test_evaluation_jobs.py`: asynchronous jobs, idempotency, limits, status,
  and reports.
- `test_scenarios_ablation.py`: deterministic scenario suites and honest
  report comparison.
- `test_memory.py`: claim extraction, verification, trust, injection handling,
  and bounded retrieval.
- `test_security.py`: secret redaction, output sanitation, and ID validation.
- `test_persona.py`: defaults, prompt hard rules, JSON validation/round-trip,
  and Agent integration.
- `test_dashboard_data.py`: dashboard projections and comparisons.
