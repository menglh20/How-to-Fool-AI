# Game AI Agent Role - Project Evidence

This file maps implementation evidence to the target role. Replace every
`pending` result with data from `EVALUATION.md` before using it in a resume.

| Role capability | Project evidence |
|---|---|
| Task/action planning | Two-stage decision and structured action execution |
| Memory management | Working state, episodic results, claims, verification, trust |
| Tool calling | Six JSON Schema game tools with Claude/OpenAI adapters |
| Multi-turn interaction | Private human/AI and AI/AI messaging across rounds |
| State management | Thread-safe SharedState, phase/deadline/target permissions |
| Evaluation | Trace-derived batch metrics, deterministic scenarios, comparisons |
| Debugging/replay | Ordered game traces, ReplayService, Dashboard, JSONL download |
| AI toolchain | Read-only replay MCP and asynchronous evaluation MCP |
| Safety/control | Schema validation, redaction, bounded MCP, injection flags |
| Multi-agent game AI | Three autonomous persona agents in social deduction games |

## Resume bullet template

Use only verified measurements:

> Built a multi-agent social-deduction system with provider-neutral
> Claude/OpenAI Tool Calling, structured cross-round memory, and thread-safe
> game-state arbitration; developed trace replay, MCP debugging tools, and an
> offline evaluation pipeline measuring Tool legality, repair, fallback,
> latency, token usage, and game completion.

Quantified bullet after experiments:

> Replaced free-text action parsing with six schema-validated tools, improving
> first-call legality from `[baseline]` to `[candidate]` and reducing fallback
> rate by `[measured delta]` across `[N]` comparable games.

Do not include bracketed placeholders in an application.
