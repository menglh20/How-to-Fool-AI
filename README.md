# 🎭 How to Fool AI

A single-player social deduction game where you compete against three personality-driven AI agents through a series of mini-games. The real battle isn't the games themselves — it's the private messages you exchange between rounds, where every word could be truth, lies, or a trap.

## The Problem

Social deduction games like Werewolf, Mafia, and Diplomacy are some of the most engaging multiplayer experiences — but they require 5–10 players online at the same time. For most people, organizing a group is harder than the game itself. Existing solo alternatives strip out the core appeal: reading people, building trust, and breaking it.

## The Solution

**How to Fool AI** replaces the friend group with three AI agents, each powered by an LLM with a distinct personality:

| Agent | Personality | Play Style |
|-------|------------|------------|
| 🐰 Xiaobai | Naive & Kind | Trusting, chatty, easy to manipulate |
| 🦊 Fox | Cunning & Strategic | Suspicious, probing, cross-validates everything, sets traps |
| 🗿 Tiemian | Cold & Rational | Rarely speaks, trusts no one |

You play through multiple rounds of three mini-game types:

- **🔮 Guess the Word** — One player writes down a word, while the others attempt to guess it. The writer may privately reveal the correct answer to one person and a fake answer to another. A guesser earns 1 point if they successfully identify the word. If all guessers fail to identify the word—or if all guessers successfully identify it—the writer receives no points; otherwise, the writer earns 1 point.
- **🎭 Who Wrote It** — Each player writes down a single word, after which the group attempts to guess "who wrote which word." Players may privately claim ownership of any word—regardless of whether that claim is true. The player who correctly guesses the most words earns 1 point, and the player whose written word is guessed correctly the most times also earns 1 point.
- **☠️ Poison Bottle** — Players take turns, in descending order of their current scores, to select one of four bottles (one of which contains poison) and then return it. The sequence in which players make their selections reveals their "hidden ranking" within the game. Players who have already made their choice are permitted to lie to others regarding the outcome of their selection. Any player who drinks the poison loses 1 point.

### The Private Chat System

The twist that makes everything work: **during every mini-game, all participants can send private messages to anyone.** But sending costs a limited resource (10 sends per round; receiving is free). AI agents also chat with each other behind your back — you'll see "🤫 Xiaobai and Fox are whispering..." but never the content.

**Scores are hidden from everyone.** You only know your own score. Anyone can claim any score in private chat. The only way to infer others' scores is through indirect signals — like who picks first in Poison Bottle.

Every AI's sole objective is to maximize its own score. Alliances are temporary tools. Betrayal is always on the table.

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/) — Python-based web UI with `st.fragment` for real-time chat updates
- **AI Backend:** [Anthropic Claude API](https://docs.anthropic.com/) — Haiku for fast chat replies, Sonnet for strategic decisions
- **Architecture:** 5 concurrent threads (1 Game Engine + 1 Human IO + 3 AI Agents) communicating through a thread-safe shared state with `threading.RLock` and `queue.Queue`

## Project Structure

```
how-to-fool-ai/
├── app.py                      # Streamlit entry point — setup, game, and game-over screens
├── game/
│   ├── shared_state.py         # Thread-safe shared state (RLock + Queue)
│   ├── game_engine.py          # Round orchestration and all three mini-games
│   ├── ai_agent.py             # Autonomous AI player daemon threads
│   ├── llm_client.py           # Anthropic API wrappers (Haiku for chat, Sonnet for decisions)
│   └── scoring.py              # Scoring logic for all three game types
├── prompts/
│   ├── bai.py                  # Xiaobai 🐰 persona and system prompt
│   ├── fox.py                  # Fox 🦊 persona and system prompt
│   ├── ironface.py             # Tiemian 🗿 persona and system prompt
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
- An [Anthropic API key](https://console.anthropic.com/)

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

Then open `.streamlit/secrets.toml` and fill in your key:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Alternatively, copy `.env.example` to `.env` — the app will read the key from the environment as a fallback.

### Run locally

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Deployment

This app is deployed on **[Streamlit Community Cloud](https://streamlit.io/cloud)** — the only hosting platform that natively supports Streamlit's concurrent thread model (GameEngine + 3 AIAgent threads).

### Auto-deployment

Any push to `main` is automatically deployed to production via Streamlit Cloud's GitHub integration. The CI badge below reflects the test suite status:

[![CI](https://github.com/GIX-Luyao/final-project-codebase-menglh20/actions/workflows/ci.yml/badge.svg)](https://github.com/GIX-Luyao/final-project-codebase-menglh20/actions/workflows/ci.yml)

### Environment variables

| Variable | Where to set | Description |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | Streamlit Cloud → App settings → Secrets | Claude API key |

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
- [x] Three AI agent threads run independently and respond to messages via Claude API
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
- [ ] All three mini-game types are playable with correct scoring — ⚠️ two bugs filed: [#TODO-bug1] reveal screen shows blank results; [#TODO-bug2] Who Wrote It writer-bonus logic error
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
- [ ] AI personalities are noticeably distinct across a full game (Xiaobai trusting, Fox suspicious, Tiemian terse)
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