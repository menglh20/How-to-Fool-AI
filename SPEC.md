# SPEC.md — How to Fool AI

## Project Overview

**How to Fool AI** is a single-player social deduction game where one human player competes against three AI agents in a series of mini-games. The core mechanic is a private messaging system that runs concurrently with gameplay — players and AI agents exchange true or false information to manipulate each other's decisions. All scores are hidden; all alliances are temporary.

**Developer:** [Ting1016-git](https://github.com/Ting1016-git)

**Tech Stack:** Python + Streamlit + Anthropic Claude API

**Architecture:** 5-thread concurrent model (1 Game Engine + 1 Human IO via Streamlit + 3 AI Agent threads)

**Agreed Development Fee:** 40 GIX Bucks

---

## User Stories

### US-1: Game Setup
**As a** player,
**I want to** enter my name and choose the number of rounds (3, 5, 7, or 10),
**So that** I can start a game customized to my available time.

**Acceptance Criteria:**
- Player can input a display name (required, non-empty).
- Player can select round count from preset options (3/5/7/10).
- Player sees a brief introduction to the three AI opponents (Xiaobai, Fox, Tiemian) with their personality descriptions.
- Clicking "Start Game" initializes the game and transitions to the game view.

---

### US-2: Mini-Game — Guess the Word
**As a** player,
**I want to** participate in a word-guessing game where the writer can selectively leak information via private chat,
**So that** I can experience bluffing and trust-based decision making.

**Acceptance Criteria:**
- A random participant (human or AI) is selected as the word writer.
- The writer submits a word (Chinese, 2-4 characters) that is hidden from others.
- All non-writers simultaneously submit their guess (one word each).
- During the guessing window, all participants can send private messages (e.g., the writer can privately tell one person the real answer and another person a fake answer).
- Scoring: +1 for each correct guesser; +1 for the writer if at least one but not all guessers are correct.
- Results are revealed showing each guess and whether it was correct.

---

### US-3: Mini-Game — Who Wrote It
**As a** player,
**I want to** write a word that reflects my style and then guess who wrote each of the other words,
**So that** I can test my ability to read personality through language.

**Acceptance Criteria:**
- All four participants simultaneously write one word each.
- Each participant sees the three words written by others (not their own) and guesses the author of each.
- During the guessing window, participants can privately claim authorship of any word (truthfully or not).
- Scoring: +1 for the player(s) who correctly identify the most authors; +1 for a writer whose word is guessed by some but not all.
- Results reveal the true author of each word.

---

### US-4: Mini-Game — Poison Bottle
**As a** player,
**I want to** choose one of four colored bottles (one is poisoned) in score-ranked order,
**So that** I can use deduction, private chat, and bluffing to avoid the poison.

**Acceptance Criteria:**
- Four bottles (Red, Blue, Green, Yellow) are presented; one is randomly poisoned.
- Selection order is determined by current score (highest first), which implicitly reveals relative rankings.
- Each player has a time window (configurable, default 60 seconds) to pick a bottle. During this window, all participants can freely send private messages.
- Players who have already chosen can privately claim their bottle was poisoned or safe (true or false).
- Scoring: -1 for the player who drinks the poisoned bottle.
- Results reveal which bottle was poisoned and who chose it.

---

### US-5: Private Chat System
**As a** player,
**I want to** send private messages to any AI agent during gameplay and see notifications when AIs are chatting with each other,
**So that** I can trade information, bluff, and influence AI decisions.

**Acceptance Criteria:**
- A chat panel is available at all times during gameplay (sidebar or right column).
- Player can select a chat target (Xiaobai, Fox, or Tiemian) and send a message.
- Sending a message costs 1 of 10 per-round send opportunities. Receiving messages is free.
- The remaining send count is clearly displayed and updated in real time.
- When the send count reaches 0, the input is disabled with a clear message.
- AI messages appear in the chat within 1-3 seconds of being sent.
- When two AIs are privately chatting with each other, the player sees a notification like "🤫 Xiaobai and Fox are whispering..." but cannot see the content.
- All chat content is strictly isolated: A-B conversations are invisible to C and D.

---

### US-6: Hidden Scores
**As a** player,
**I want** all scores to be hidden from everyone except the score owner,
**So that** players must rely on private communication (which may be lies) to assess the competitive landscape.

**Acceptance Criteria:**
- The scoreboard shows only the human player's own score. Other players' scores are displayed as "???".
- Players (human and AI) can claim any score in private chats — the system never reveals the truth.
- At the end of the game (game over screen), all scores are revealed with final rankings.
- Selection order in Poison Bottle implicitly reveals relative rankings, which is intentional and should not be hidden.

---

### US-7: AI Agent Behavior
**As a** player,
**I want** the three AI agents to behave according to their distinct personalities — making independent decisions about when to chat, whom to trust, and when to lie,
**So that** each game feels dynamic and each AI presents a different challenge.

**Acceptance Criteria:**
- **Xiaobai (🐰 Naive & Kind):** Tends to trust others easily, responds enthusiastically, uses up chat sends quickly, is susceptible to manipulation.
- **Fox (🦊 Cunning & Strategic):** Suspicious by default, asks probing questions, cross-validates information, strategically conserves chat sends for key moments.
- **Tiemian (🗿 Cold & Rational):** Rarely initiates conversation, gives minimal responses, almost never trusts claims, hoards chat sends and strikes precisely when it matters.
- Each AI independently decides whether to reply to messages, initiate chats, or stay silent.
- Each AI's core objective is to maximize its own score. Alliances are temporary tools.
- AI agents chat with each other (invisible to the human player), exchanging real or fake information.
- AI behavior must remain consistent with personality across the entire game. The system prompt must reinforce personality in every API call.

---

### US-8: Game Results & Review
**As a** player,
**I want to** see final rankings and optionally review what happened behind the scenes,
**So that** I can understand how the AIs strategized and improve my play.

**Acceptance Criteria:**
- Game over screen shows all four players ranked by score, with scores revealed.
- A winner is declared (with appropriate messaging for human win vs. AI win).
- Player can start a new game from the results screen.
- (Stretch goal) Post-game review showing AI-to-AI private chat logs, letting the player see what was discussed behind their back.

---

## AI Agent Architecture

### Personality Definitions

| Agent | Name | Trait | System Prompt Emphasis |
|-------|------|-------|----------------------|
| ai_0 | Xiaobai 🐰 | Naive & Kind | "You are trusting and straightforward. You tend to believe what others tell you. You speak in a warm, simple manner." |
| ai_1 | Fox 🦊 | Cunning & Strategic | "You are suspicious and analytical. You like to set traps and test others. You are skilled at disguise and misdirection." |
| ai_2 | Tiemian 🗿 | Cold & Rational | "You trust almost no one. You are logical, concise, and direct. You speak only when necessary." |

### Decision Model

Each AI action is resolved through a single LLM API call. The prompt includes:
1. **Identity & personality** (system prompt, constant per agent)
2. **Core objective** ("Your only goal is to score the highest. Any cooperation is a temporary tool.")
3. **Current game context** (round number, game type, own score, remaining sends)
4. **Known information** (facts from own experience + received messages with credibility tags)
5. **Observations** (who is whispering with whom, selection order hints)
6. **Action options** (reply / initiate chat / make game decision / do nothing)

### Model Selection
- **Private chat replies:** Claude Haiku (fast, cheap, ~0.5-1s)
- **Strategic game decisions:** Claude Sonnet (stronger reasoning, ~1-2s)
- **Memory summaries:** Claude Haiku

### Context Management
- After each round, generate a structured memory summary per AI agent.
- Subsequent rounds receive only the summary, not full chat history.
- Keep the most recent 10 messages in context for the current round.

---

## Concurrency Architecture

### 5-Thread Model

```
Thread 1: GameEngine    — State arbitration, rule enforcement, phase progression
Thread 2: HumanIO      — Streamlit main process (UI rendering + input collection)
Thread 3: AIAgent[0]    — Xiaobai's autonomous behavior loop
Thread 4: AIAgent[1]    — Fox's autonomous behavior loop
Thread 5: AIAgent[2]    — Tiemian's autonomous behavior loop
```

### Shared State
- A single `SharedState` object protected by `threading.RLock`.
- Per-player message inboxes using `queue.Queue` (thread-safe).
- Game events broadcast via per-player event queues.
- Whisper notifications for the UI via a separate queue.

### Streamlit Integration
- `st.fragment(run_every=2)` for the chat panel — auto-refreshes every 2 seconds to show new AI messages.
- `st.fragment(run_every=3)` for the game area — auto-refreshes to reflect game state changes.
- Backend threads are started once and run as daemons, surviving Streamlit reruns.
- All state lives in `st.session_state` pointing to the shared `SharedState` instance.

---

## Scoring Rules Summary

| Game | Who Scores | Condition | Points |
|------|-----------|-----------|--------|
| Guess the Word | Correct guesser | Guessed the word correctly | +1 |
| Guess the Word | Writer | ≥1 but not all guessers correct | +1 |
| Who Wrote It | Best guesser(s) | Most correct author attributions | +1 |
| Who Wrote It | Writer | Word attributed by some but not all | +1 |
| Poison Bottle | Poisoned player | Chose the poisoned bottle | -1 |

---

## API Call Budget Estimate

| Call Type | Per Round | Model | Latency |
|-----------|-----------|-------|---------|
| Chat replies | 10–20 | Haiku | 0.5–1s |
| Proactive decisions | 6–12 | Haiku | 0.5–1s |
| Game decisions | 3–6 | Sonnet | 1–2s |
| Memory summaries | 3 | Haiku | 0.5–1s |
| **Per round total** | **22–41** | — | — |
| **10-round game** | **220–410** | — | — |

---

## Non-Functional Requirements

- **Response time:** AI chat replies should appear within 1-3 seconds.
- **Thread safety:** All shared state access must be protected by locks. No race conditions in score updates or message routing.
- **Graceful degradation:** If an API call fails, the AI should fall back to a personality-consistent default response (e.g., "Hmm... let me think" for Xiaobai).
- **Secrets management:** API keys stored in `st.secrets` (or `.streamlit/secrets.toml` for local dev). Never hardcoded.
- **Language:** All in-game content in English. Code comments and documentation in English.
