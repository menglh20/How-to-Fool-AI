# Bug Report

**Issue Title:** [Bug] AI chat messages use raw player IDs ("ai_1", "human") instead of character names, and frequently produce off-topic or empty responses

## Steps to Reproduce

1. Start a game and wait for AI agents to begin proactive chatting
2. Read the chat messages sent by the AI agents
3. Observe messages that reference "ai_1", "ai_0", or "human" literally — e.g. "ai_1 你怎么看？" or "我想和 human 聊聊"
4. Also observe messages that are generic, repetitive, or unrelated to the current game context (e.g. repeated "嗯……让我想想" default replies appearing in chat)

## Expected Behavior

- AI agents should always refer to other players by their **character names**: 小白, 狐狸, 铁面, and the human player's display name (stored in `st.session_state.player_name`)
- Chat messages should feel like natural in-game social deduction dialogue: probing questions, strategic misdirection, alliance proposals — not generic filler
- Each AI's messages should be clearly distinguishable in tone: 小白 enthusiastic and trusting, 狐狸 suspicious and indirect, 铁面 terse and cold

## Actual Behavior

- Raw IDs appear in chat because the prompt templates pass `sender` and `target` as internal player ID strings (`"ai_0"`, `"ai_1"`, `"human"`) directly into the LLM prompt, without mapping them to display names first
- The LLM then reproduces these IDs verbatim in its output, breaking immersion
- Default replies (e.g. `"嗯……让我想想"`, `"有意思……"`) appear in the visible chat when the API call fails silently, making conversations feel hollow

## Root Cause (for developer)

In `prompts/templates.py`, player IDs are injected raw into the user-turn prompt:

```python
def build_reply_prompt(persona_name, sender, incoming_text, ...):
    lines = [
        sender + " 对你说：「" + incoming_text + "」",   # "ai_1 对你说：..."
    ]

def build_proactive_prompt(persona_name, target, ...):
    lines.append("你决定主动联系 " + target + "。")      # "你决定主动联系 ai_0。"
```

**Suggested fixes:**

1. **Name mapping** — add a `PLAYER_NAMES` dict and resolve IDs to display names before building prompts:
```python
PLAYER_NAMES = {
    "human": "{player_name}",   # injected at call site from session state
    "ai_0": "小白",
    "ai_1": "狐狸",
    "ai_2": "铁面",
}
```

2. **Prompt quality** — add game-context hints to the proactive prompt so the LLM generates strategically relevant messages rather than filler:
   - Current round number and game type
   - A reminder of what information is valuable to share or extract right now
   - An explicit instruction: "不要只说寒暄，你的每句话都应该有战略意图"

3. **Default reply suppression** — default replies (API fallback) should not be sent into the visible chat. If the API call fails, silently skip the send rather than delivering a meaningless message.

## Severity

- [ ] **Blocker** — App is unusable or data is corrupted. Must fix before Demo Day.
- [x] **Major** — Core feature is broken but app still works for other tasks. Should fix.
- [ ] **Minor** — Cosmetic issue, typo, or edge case. Fix if time allows.

Immersion is a core design goal of the game. Seeing "ai_1 对你说" destroys the fiction that you are playing against distinct personalities, and low-quality messages undermine the social deduction mechanic.

## Evidence

Start a game and observe the chat panel over 2–3 rounds. Look for raw IDs in message text and for repetitive or context-free messages. The prompt strings in `prompts/templates.py` lines 13–27 and 30–47 show the injection points clearly.

## Environment (if relevant)

| Detail | Value |
|--------|-------|
| Browser | Any |
| Device | Desktop |
| OS | Any |
| Deployed or local? | Reproducible on localhost |

## Related Issue

Related to #2 (AI agent personas), #8 (Prompt tuning for personality consistency)
