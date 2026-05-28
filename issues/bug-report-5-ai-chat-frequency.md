# Bug Report

**Issue Title:** [Bug] AI agents exhaust their send budget on AI-to-AI chats, failing to reply when the human player sends a message

## Steps to Reproduce

1. Start a game and wait for Round 1 to begin
2. Watch the chat panel for 10–20 seconds without doing anything
3. Observe AI agents proactively messaging each other — the chat panel fills with whisper notifications ("🤫 小白 and 狐狸 are whispering…") and direct messages
4. Send a private message to 小白 (Xiaobai), who has the highest chat initiative probability
5. Observe that 小白 frequently does not reply, or replies are delayed until a later tick when budget may already be zero

## Expected Behavior

- When the human player sends a message to an AI, the AI should reliably reply within a few seconds
- AI-to-AI proactive chat should be throttled so that each AI retains enough budget to respond to the human player at least once per round
- Human thinking time (reading messages, composing a response) should be factored in — the AI should not burn its entire budget in the first 10 seconds of a round

## Actual Behavior

The three AI agents tick every 1–3 seconds. Xiaobai's `CHAT_INITIATIVE_PROB` is **0.65**, meaning in a 60-second round she attempts to send roughly 20+ proactive messages. With only 10 sends per round and a random target selection (equal probability of human or other AIs), the budget is consumed almost entirely by AI-to-AI traffic in the early part of the round.

By the time the human sends a message, the receiving AI often has 0 remaining budget and silently drops the reply (`_handle_chat` returns early when `_has_budget()` is False). The human gets no response and no explanation.

## Root Cause (for developer)

Three compounding problems in `game/ai_agent.py`:

**1. Initiative probability too high for Xiaobai:**
```python
# prompts/bai.py
CHAT_INITIATIVE_PROB = 0.65   # fires ~20× per 60s round at 2s tick interval
```

**2. Proactive target is fully random — no preference for the human:**
```python
# ai_agent.py _proactive_chat()
target = random.choice(others)   # human and AIs weighted equally
```

**3. Reply is silently dropped when budget is zero — no fallback:**
```python
# ai_agent.py _handle_chat()
if not self._has_budget():
    return   # human message is received but ignored with no notification
```

**Suggested fixes:**
- Reduce `CHAT_INITIATIVE_PROB` for all agents (e.g. Xiaobai: 0.35, Fox: 0.15, Tiemian: 0.03)
- Reserve a minimum budget (e.g. 3 sends) for human replies; only use surplus for AI-to-AI chats
- When an AI cannot reply due to zero budget, send a fallback system notification to the human (e.g. "🐰 小白 has used up all their messages this round")
- Add a human-reply cooldown: after the human sends a message, the receiving AI should prioritize replying on its next tick before initiating new proactive chats

## Severity

- [ ] **Blocker** — App is unusable or data is corrupted. Must fix before Demo Day.
- [x] **Major** — Core feature is broken but app still works for other tasks. Should fix.
- [ ] **Minor** — Cosmetic issue, typo, or edge case. Fix if time allows.

Private chat is described as the central mechanic of the game. If AI agents cannot reliably reply to the human player, the core loop breaks down.

## Evidence

Start a game and wait ~15 seconds before sending any message. Check the remaining send budget displayed in the chat panel for each AI (not directly visible, but inferrable from whisper frequency). Then send a message to Xiaobai and count how many ticks pass before a reply arrives.

## Environment (if relevant)

| Detail | Value |
|--------|-------|
| Browser | Any |
| Device | Desktop |
| OS | Any |
| Deployed or local? | Reproducible on localhost |

## Related Issue

Related to #2 (AI agent behavior loop), #5 (Private chat system), #8 (Prompt tuning)
