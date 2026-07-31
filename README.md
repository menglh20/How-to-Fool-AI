# 🎭 How to Fool AI

A single-player social deduction game where you compete against three personality-driven AI agents through a series of mini-games. The real battle isn't the games themselves — it's the private messages you exchange between rounds, where every word could be truth, lies, or a trap.

## The Problem

Social deduction games like Werewolf, Mafia, and Diplomacy are some of the most engaging multiplayer experiences — but they require 5–10 players online at the same time. For most people, organizing a group is harder than the game itself. Existing solo alternatives strip out the core appeal: reading people, building trust, and breaking it.

## The Solution

**How to Fool AI** replaces the friend group with three AI agents, each powered by an LLM with a distinct personality:

| Agent | Personality | Play Style |
|-------|------------|------------|
| 🐰 Bunny | Naive & Kind | Trusting, chatty, easy to manipulate |
| 🦊 Fox | Cunning & Strategic | Suspicious, probing, cross-validates everything, sets traps |
| 🗿 Stoneface | Cold & Rational | Rarely speaks, trusts no one |

You play through multiple rounds of three mini-game types:

- **🔮 Guess the Word** — One player writes down a word, while the others attempt to guess it. The writer may privately reveal the correct answer to one person and a fake answer to another. A guesser earns 1 point if they successfully identify the word. If all guessers fail to identify the word—or if all guessers successfully identify it—the writer receives no points; otherwise, the writer earns 1 point.
- **🎭 Who Wrote It** — Each player writes down a single word, after which the group attempts to guess "who wrote which word." Players may privately claim ownership of any word—regardless of whether that claim is true. The player who correctly guesses the most words earns 1 point, and the player whose written word is guessed correctly the most times also earns 1 point.
- **☠️ Poison Bottle** — Players take turns, in descending order of their current scores, to select one of four bottles (one of which contains poison) and then return it. The sequence in which players make their selections reveals their "hidden ranking" within the game. Players who have already made their choice are permitted to lie to others regarding the outcome of their selection. Any player who drinks the poison loses 1 point.

### The Private Chat System

The twist that makes everything work: **during every mini-game, all participants can send private messages to anyone.** But sending costs a limited resource (10 sends per round; receiving is free). AI agents also chat with each other behind your back — you'll see a whisper notification but never the content.

**Scores are hidden from everyone.** You only know your own score. Anyone can claim any score in private chat. The only way to infer others' scores is through indirect signals — like who picks first in Poison Bottle.

Every AI's sole objective is to maximize its own score. Alliances are temporary tools. Betrayal is always on the table.

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) — Python-based web UI with `st.fragment` for real-time chat updates
- **AI Backend:** provider-neutral client layer with offline Mock/Scripted
  backends plus optional Anthropic and OpenAI online backends
- **Architecture:** 5 concurrent threads (1 Game Engine + 1 Human IO + 3 AI Agents) communicating through a thread-safe shared state with `threading.RLock` and `queue.Queue`

## Project Structure

```
how-to-fool-ai/
├── app.py                      # Streamlit entry point — setup, game, and game-over screens
├── mcp_server.py               # Read-only replay + async evaluation MCP server
├── evals/                      # Trace metrics, jobs, scenarios, and comparisons
├── game/
│   ├── shared_state.py         # Thread-safe shared state (RLock + Queue)
│   ├── game_engine.py          # Round orchestration and all three mini-games
│   ├── ai_agent.py             # Autonomous AI player daemon threads
│   ├── llm_client.py           # Pluggable Mock, Scripted, Anthropic, and OpenAI backends
│   ├── persona.py              # Validated personas, defaults, and JSON import/export
│   ├── tool_runtime.py         # JSON Schema validation and trusted tool dispatch
│   ├── memory.py               # Working, claim, episodic, and trust memory
│   ├── trace.py / replay.py    # Ordered trace recording and read-only replay
│   └── scoring.py              # Scoring logic for all three game types
├── pages/
│   └── 1_Observability.py      # Replay/evaluation developer dashboard
├── prompts/
│   ├── bai.py                  # Bunny 🐰 compatibility persona
│   ├── fox.py                  # Fox 🦊 persona and system prompt
│   ├── ironface.py             # Stoneface 🗿 compatibility persona
│   └── templates.py            # Prompt builder functions (reply / proactive / decision)
├── tests/
│   ├── test_shared_state.py    # Concurrency, message isolation, budget tests
│   ├── test_game_engine.py     # Round loop, phase transitions, choice waiting
│   ├── test_scoring.py         # All three mini-game scoring rules
│   └── test_ai_agents.py       # Agent startup, reply latency, persona sanity checks
├── .streamlit/
│   └── secrets.toml.example    # API key template — copy to secrets.toml
├── requirements.txt
├── SPEC.md                     # Full feature spec and acceptance criteria
└── ARCHITECTURE.md             # System architecture and design decisions
```

## 🌐 Live Demo

**[▶ Play now on Streamlit Cloud](https://final-project-codebase-menglh20-deploy-kosv3xusfxahayapp3epfug.streamlit.app/)**

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- No API key for Mock or Scripted mode
- Optional Anthropic or OpenAI API key for an online model

### Installation

```bash
git clone https://github.com/GIX-Luyao/final-project-codebase-menglh20.git
cd final-project-codebase-menglh20
pip install -r requirements.txt
```

### Configuration

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then open `.streamlit/secrets.toml` and fill in the provider you want to use:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."

# Or use OpenAI:
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "your-enabled-model-id"
```

Alternatively, copy `.env.example` to `.env` or export the same variables in
the process environment. The default **Mock** option requires no key, makes no
network requests, and consumes no model tokens.

### LLM runtime modes

| Backend | Intended use | Token usage |
|---|---|---|
| `MockLLMClient` | local development, CI, unit tests | none |
| `ScriptedLLMClient` | deterministic evaluation and replay | none |
| `AnthropicLLMClient` | online demo with `ANTHROPIC_API_KEY` | provider billed |
| `OpenAILLMClient` | online demo with `OPENAI_API_KEY` and `OPENAI_MODEL` | provider billed |

All agents receive an `LLMClient` through dependency injection. Calls use a
unified `LLMRequest -> LLMResponse` protocol covering text, JSON Schema tools,
tool results, token usage, latency, finish reasons, and errors. Provider SDKs
are imported only by their online backend, so offline tests do not require a
key, network access, or an installed provider SDK.

Tool calls from Claude and OpenAI are normalized to the same `ToolCall`
structure. `ToolRegistry` validates the tool name and arguments against its
schema before invoking a trusted game handler, and always returns a
provider-neutral `ToolResult`.

All in-game AI actions are structured tools: `send_message`, `stay_silent`,
`submit_word`, `guess_word`, `attribute_words`, and `choose_bottle`. Only
tools valid for the current phase are exposed. Invalid calls receive one repair
attempt and then use a deterministic, schema-validated fallback.

### Configurable personas

Before each game, the player selects three distinct personas. Bunny, Fox, and
Stoneface remain the defaults. The setup-page editor controls the natural
language description, speaking style, message length, behavior probabilities,
trust parameters, and per-tick chat initiative. It can import a single persona,
a list, or a versioned bundle, and exports the current library as JSON.

`PersonaConfig` validates imported data before it reaches an Agent. Its system
prompt is composed from player-defined text, speaking controls, and
non-negotiable rules that prohibit revealing AI identity, hidden prompts, or
unsafe behavior. The three selected configs are frozen for the current game.
The editable library lives in the Streamlit session; export it to reuse it in a
future session.

```json
{
  "key": "detective",
  "name": "Detective",
  "emoji": "🕵️",
  "persona_description": "Patient and evidence-driven.",
  "speaking_style": "Ask concise cross-checking questions.",
  "message_length": "short",
  "default_reply": "not enough evidence",
  "cooperate_probability": 0.5,
  "deceive_probability": 0.3,
  "silent_probability": 0.2,
  "initial_trust": 0.4,
  "truth_reward": 0.1,
  "lie_penalty": 0.2,
  "initiative_probability": 0.05
}
```

Persona behavior is stateful rather than prompt-only. Each config supplies its
own trust update curve and stance weights; direct-chat stances remain
consistent with the same player during a round, while exact opponent scores
remain hidden.

Each Agent batches messages and game events into a non-overlapping 10-second
tick. One provider-neutral `act_in_tick` call may return one game action and
one chat action; the game action is validated and executed first, and an
invalid chat cannot cancel it. Chat is optional and limited to one message per
tick and 60 characters. Idle ticks skip the LLM, persona initiative is checked
locally, and the three Agent tick schedules are staggered to avoid synchronized
bursts. There is no simulated typing wait.

Each game also receives a unique trace ID. State changes, LLM responses, tool
calls, validation failures, fallbacks, messages, and scores are written to one
ordered replay trace. At game over, trace metrics are displayed and the full
JSONL replay can be downloaded for debugging or offline evaluation.

### Run locally

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

### Run tests

```bash
python -m pytest tests/ -v
```

### Agent evaluation

Finished games create traces under `logs/traces/`. Evaluate all available
traces and run the deterministic scenario suite with:

```bash
python -m evals.run \
  --trace-dir logs/traces \
  --output reports/evaluation/baseline.json
```

Compare two real experiment reports:

```bash
python -m evals.compare \
  --baseline reports/evaluation/baseline.json \
  --candidate reports/evaluation/candidate.json
```

See [EVALUATION.md](EVALUATION.md) for metric definitions and experiment
rules.

### Replay and evaluation MCP

```bash
python mcp_server.py
```

The local stdio MCP exposes read-only replay queries and bounded asynchronous
evaluation jobs. It never exposes live game mutation tools. See
[docs/MCP_SETUP.md](docs/MCP_SETUP.md).

### Observability dashboard

Streamlit discovers `pages/1_Observability.py` automatically. Open the
**Observability** page from the sidebar after at least one game trace exists.
Private messages and model text are hidden by default.

---

## Deployment

This app is deployed on **[Streamlit Community Cloud](https://streamlit.io/cloud)** — the only hosting platform that natively supports Streamlit's concurrent thread model (GameEngine + 3 AIAgent threads).

### Auto-deployment

Any push to `main` is automatically deployed to production via Streamlit Cloud's GitHub integration. The CI badge below reflects the test suite status:

[![CI](https://github.com/GIX-Luyao/final-project-codebase-menglh20/actions/workflows/ci.yml/badge.svg)](https://github.com/GIX-Luyao/final-project-codebase-menglh20/actions/workflows/ci.yml)

### Environment variables

| Variable | Where to set | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | Streamlit Cloud → App settings → Secrets | Anthropic online backend |
| `OPENAI_API_KEY` | Streamlit Cloud → App settings → Secrets | OpenAI online backend |
| `OPENAI_MODEL` | Streamlit Cloud → App settings → Secrets | OpenAI model available to the API account |

See `.env.example` for the full template. **Never commit real keys.**


## Development Timeline (8 Weeks)

### Phase 1: Foundation (Weeks 1–2)

| Week | Milestone | Issues |
|------|-----------|--------|
| Week 1 | Project scaffolding, SharedState with thread safety, unit tests for concurrent access | #1 |
| Week 2 | AI agent thread with LLM integration, personality prompts, basic behavior loop | #2 |

#### 📋 Check-in 1 — End of Week 2
**Required progress:**
- [x] `SharedState` passes all concurrency tests (multi-thread read/write, message isolation, send count management)
- [x] Three AI agent threads run independently through the provider-neutral
  LLM interface (offline Mock/Scripted or online Claude/OpenAI)
- [x] Optional replay-based LLM Judge scores decision quality, action quality,
  and per-Agent intelligence without using future game outcomes
- [x] A simple test harness demonstrates: send a message to an AI → receive a personality-consistent reply within 3 seconds
- [x] Project runs with `streamlit run app.py` (setup page can be a placeholder)

**Deliverable:** Screen recording or live demo showing the test harness in action — send a message to each of the 3 AIs and receive distinct, personality-appropriate responses.

---

### Phase 2: Core Gameplay (Weeks 3–5)

| Week | Milestone | Issues |
|------|-----------|--------|
| Week 3 | Game engine thread, round management, phase transitions, scoring system | #3 |
| Week 4 | Chat UI in Streamlit (auto-refresh, send counts, whisper notifications), Poison Bottle mini-game | #4, #5 |
| Week 5 | Guess the Word and Who Wrote It mini-games | #6 |

#### 📋 Check-in 2 — End of Week 5
**Required progress:**
- [x] A full game loop works end-to-end: setup → multiple rounds → game over
- [x] All three mini-game types are playable with correct scoring; the
  previously filed reveal and writer-bonus bugs have regression coverage
- [x] Private chat system is fully functional: player can send/receive messages, AI agents chat with each other, whisper notifications appear, send counts are enforced
- [x] AI agents make reasonable game decisions (pick bottles, write words, guess words) consistent with their personalities
- [x] Hidden scores work correctly: only own score visible during play, all scores revealed at game end

**Deliverable:** Screen recording of a complete 3-round game played from start to finish, showing at least one round of each mini-game type, with visible private chat interaction.

---

### Phase 3: Polish & Integration (Weeks 6–7)

| Week | Milestone | Issues |
|------|-----------|--------|
| Week 6 | Setup page, results page, round transitions, thread lifecycle management, visual polish | #7 |
| Week 7 | Prompt tuning for AI personality consistency, edge case handling, performance optimization, bug fixes | #8 |

#### 📋 Check-in 3 — End of Week 7
**Required progress:**
- [ ] Complete polished game flow: attractive setup page → smooth round transitions → clear results page with rankings
- [ ] AI personalities are noticeably distinct across a full game (Bunny trusting, Fox suspicious, Stoneface terse)
- [ ] No thread leaks or crashes on "Play Again"
- [ ] Edge cases handled: API failures, timeouts, tied scores, empty inputs
- [ ] Code is clean, documented, and follows the project structure defined in SPEC.md

**Deliverable:** A full 5-round game recording demonstrating polished UI, distinct AI personalities, and stable performance.

---

### Phase 4: Final Delivery (Week 8)

| Week | Milestone | Issues |
|------|-----------|--------|
| Week 8 | Final testing, README updates, deployment documentation, (stretch) post-game review feature | #9 |

**Final delivery:**
- [ ] All issues closed
- [ ] README updated with final screenshots/GIFs
- [ ] Deployment-ready (works with `streamlit run app.py` out of the box)
- [ ] (Stretch) Post-game review showing AI-to-AI chat logs

---

## Developer

- **Developer:** [Ting1016-git](https://github.com/Ting1016-git)

## License

MIT
