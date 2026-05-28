# Bug Report

**Issue Title:** [Bug] "Your score" only updates on full page rerender (button click or chat send), not automatically — creates false impression that actions award unearned points

## Steps to Reproduce

1. Run the app with `streamlit run app.py` and start a game with 3+ rounds
2. Play through Round 1 (Guess the Word) — earn at least 1 point by guessing correctly or as the writer
3. Observe the **Your score** metric at the top right: it does **not** update after the round ends
4. In any subsequent round, either:
   - **Send a private chat message** to any AI opponent, OR
   - **Click a bottle button** during a Poison Bottle round
5. Observe that **Your score** jumps immediately — triggered by the chat send or button click, not by any scoring event

## Expected Behavior

The **Your score** metric should update automatically within a few seconds whenever the score changes — regardless of whether the player clicks a button. A player who earns +1 in Round 1 should see their score update during or immediately after that round's reveal phase, not several rounds later.

## Actual Behavior

The score metric is frozen at its last-seen value until the player accidentally triggers a full page rerender. This happens in two ways:

- **Sending a chat message** via `st.chat_input` — Streamlit treats `st.chat_input` as a page-level widget; submitting it inside a fragment always triggers a full app rerun.
- **Clicking a bottle button** — the bottle button explicitly calls `st.rerun()`, which also causes a full app rerun.

This means the score can jump unexpectedly at any moment the player interacts with chat or selects a bottle — even when no scoring event just occurred. Players naturally assume whichever action they just took caused the score change, leading to confusion about the game rules.

## Root Cause (for developer)

In `app.py`, `st.metric("Your score", score)` is rendered **outside** any auto-refreshing fragment:

```python
def render_game() -> None:
    score = state.get_score(HUMAN_ID)   # read only on full-page rerender
    ...
    st.metric("Your score", score)      # outside _game_area_fragment

    with col_game:
        _game_area_fragment()           # run_every=3 — only this part auto-refreshes
    with col_chat:
        _chat_fragment()                # run_every=2
```

`st.fragment(run_every=3)` keeps `_game_area_fragment` refreshing, but the score metric above it is static. A full app rerun — which is the only thing that updates the metric — is triggered by:

| Player action | Why it causes a full rerun |
|---------------|---------------------------|
| Submit chat message | `st.chat_input` inside a fragment always triggers a full app rerun (Streamlit design) |
| Click bottle button | Explicitly calls `st.rerun()` inside the fragment |
| Submit word / guess buttons | Only reruns the fragment — score does **not** update |
| Fragment auto-refresh (every 2–3 s) | Only reruns the fragment — score does **not** update |

**Suggested fix** — move the score metric inside `_game_area_fragment` so it refreshes every 3 seconds automatically:

```python
@st.fragment(run_every=3)
def _game_area_fragment() -> None:
    state: SharedState = st.session_state.state
    # Show own score here so it auto-refreshes with the fragment
    st.metric("Your score", state.get_score(HUMAN_ID))
    ...
```

And remove the `st.metric` call from `render_game()`.

## Severity

- [ ] **Blocker** — App is unusable or data is corrupted. Must fix before Demo Day.
- [ ] **Major** — Core feature is broken but app still works for other tasks. Should fix.
- [x] **Minor** — Cosmetic issue, typo, or edge case. Fix if time allows.

The score is computed and stored correctly — only the display timing is wrong. However, the symptom (score jumping after a bottle pick) actively misleads players about how the scoring rules work, which undermines the game experience.

## Evidence

Reproduce by playing a game and earning points in Round 1. Watch the score at the top right — it will not update after the round ends. Then send a private chat message to any AI in a later round: the score will jump immediately, even though sending a message has no effect on scoring.

## Environment (if relevant)

| Detail | Value |
|--------|-------|
| Browser | Any (Streamlit renders server-side) |
| Device | Desktop |
| OS | Any |
| Deployed or local? | Reproducible on localhost |

## Related Issue

Related to #4 (Chat UI and game screens — `render_game()` layout was introduced in this issue)
