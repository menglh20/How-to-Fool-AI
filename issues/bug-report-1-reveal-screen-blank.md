# Bug Report

**Issue Title:** [Bug] Round reveal screen shows "?" for all fields after every mini-game

## Steps to Reproduce

1. Run the app with `streamlit run app.py`
2. Enter a player name and select any number of rounds, then click **Start Game**
3. Wait for the first mini-game round to complete (the Phase indicator changes to "🔍 Reveal")
4. Observe the **Round Results** panel in the game area

## Expected Behavior

The reveal screen should display the actual round outcome. For example, after a **Guess the Word** round:
- **Writer:** the player who wrote the word (e.g., "🐰 小白")
- **Word:** the secret word that was written (e.g., "苹果")
- **Guesses:** each player's guess and whether it was correct (✅ / ❌)
- **Score changes this round:** a list of non-zero score deltas

## Actual Behavior

Every field in the results panel shows its fallback placeholder:
- Writer shows `?`
- Word shows `?`
- All guesses show `?`
- Score changes section is empty (no deltas displayed)

The reveal phase appears but conveys no information to the player.

## Root Cause (for developer)

The `round_end` event payload has a **two-level structure**:

```python
# game_engine.py — what is actually broadcast:
{
    "round": round_num,
    "game_type": game_type,
    "results": {            # ← actual data is ONE level deeper
        "writer": ...,
        "word": ...,
        "guesses": ...,
        "score_deltas": ...,
    }
}
```

In `app.py`, `_game_area_fragment` stores the entire payload:

```python
st.session_state.last_round_results = item.payload   # the outer dict
```

Then `_render_round_results(results)` calls `results.get("writer", "?")` — but `"writer"` is not a top-level key; it lives under `results["results"]`. Every `.get()` returns the `"?"` default.

**Suggested fix** in `app.py` line 238:
```python
# Change:
st.session_state.last_round_results = item.payload
# To:
st.session_state.last_round_results = {**item.payload, **item.payload.get("results", {})}
```

## Severity

- [ ] **Blocker** — App is unusable or data is corrupted. Must fix before Demo Day.
- [x] **Major** — Core feature is broken but app still works for other tasks. Should fix.
- [ ] **Minor** — Cosmetic issue, typo, or edge case. Fix if time allows.

The game loop itself continues to run correctly; only the results display is broken. However, the reveal phase is a core part of the player experience — without it, players cannot see what happened each round.

## Evidence

Reproduce locally: after any round completes, the reveal panel will consistently show:

```
Writer: ?  |  Word: ?
Guesses:
  (empty or ? entries)
Score changes this round:
  (nothing shown)
```

## Environment (if relevant)

| Detail | Value |
|--------|-------|
| Browser | Any (Streamlit renders server-side) |
| Device | Desktop |
| OS | Any |
| Deployed or local? | Reproducible on localhost |

## Related Issue

Related to #4 (Chat UI and game screens — `_render_round_results` was added in this issue)
