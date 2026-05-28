# Bug Report

**Issue Title:** [Bug] "Who Wrote It" writer bonus is always awarded whenever any guesser is correct, ignoring the "not all" condition

## Steps to Reproduce

1. Run the app with `streamlit run app.py` and start a game
2. Wait for a **🎭 Who Wrote It** round to begin
3. Have **all three non-writer guessers** correctly identify the same writer's word (this can be simulated by checking `tests/test_scoring.py` directly — see below)
4. After the round ends, observe the score delta for the writer of that word

Alternatively, reproduce with a unit test:

```python
from game.scoring import score_who_wrote_it

# All 3 eligible guessers correctly identify ai_0 as the author of their word.
words = {"human": "梦想", "ai_0": "苹果", "ai_1": "天空", "ai_2": "月亮"}
attributions = {
    "human": {"ai_0": "ai_0", "ai_1": "ai_1", "ai_2": "ai_2"},   # human guesses all correct
    "ai_1":  {"human": "human", "ai_0": "ai_0", "ai_2": "ai_2"}, # ai_1 guesses ai_0 correct
    "ai_2":  {"human": "human", "ai_0": "ai_0", "ai_1": "ai_1"}, # ai_2 guesses ai_0 correct
    "ai_0":  {"human": "human", "ai_1": "ai_1", "ai_2": "ai_2"}, # ai_0 guesses others (not itself)
}
deltas = score_who_wrote_it(words, attributions)
print(deltas["ai_0"])  # Prints 1 — should be 0
```

## Expected Behavior

Per the game rules and SPEC.md (US-3):

> +1 for a writer whose word is guessed by **some but not all** guessers.

When **all eligible guessers** correctly identify a writer, the writer should receive **0 bonus points** (their word was too easy / everyone got it — no strategic value in being identified by everyone).

In the example above, `ai_0` should receive **0** from the writer-bonus rule (though may still earn from the guesser-bonus rule separately).

## Actual Behavior

The writer receives **+1** even when every eligible guesser identified their word correctly. The "not all" condition never triggers.

`ai_0` receives `1` when it should receive `0`.

## Root Cause (for developer)

In `game/scoring.py`, lines 102–109:

```python
total_guessers = len(attributions)   # ← always 4 (all players, including the writer)

for author_id in all_players:
    correct_for_author = sum(
        1 for their_guesses in attributions.values()
        if their_guesses.get(author_id) == author_id
    )
    if 0 < correct_for_author < total_guessers:   # ← compares against 4
        deltas[author_id] += 1
```

Each writer's word is only shown to the **other 3 players** — the writer does not guess their own word. So `correct_for_author` can be at most **3**. But `total_guessers` is **4**, making `correct_for_author < total_guessers` always `True` (3 < 4). The "not all" guard never fires.

**Suggested fix** — count only the guessers who actually received that author's word:

```python
for author_id in all_players:
    eligible_guessers = [
        g for g, guesses in attributions.items()
        if author_id in guesses          # only guessers who saw this author's word
    ]
    correct_for_author = sum(
        1 for g in eligible_guessers
        if attributions[g][author_id] == author_id
    )
    n_eligible = len(eligible_guessers)
    if 0 < correct_for_author < n_eligible:
        deltas[author_id] += 1
```

## Severity

- [ ] **Blocker** — App is unusable or data is corrupted. Must fix before Demo Day.
- [x] **Major** — Core feature is broken but app still works for other tasks. Should fix.
- [ ] **Minor** — Cosmetic issue, typo, or edge case. Fix if time allows.

The game runs without crashing, but the scoring rule is incorrect in every Who Wrote It round. Writers are systematically over-rewarded, which skews the competitive balance.

## Evidence

Run the reproduction snippet above in a Python shell with the project on your `PYTHONPATH`. The assertion `deltas["ai_0"] == 0` will fail, returning `1` instead.

The existing test `test_writer_partial_guess_bonus` in `tests/test_scoring.py` passes (it only checks the partial case), but there is no test covering the "all correct → no bonus" boundary, which is where the bug hides.

## Environment (if relevant)

| Detail | Value |
|--------|-------|
| Browser | N/A (logic bug in Python) |
| Device | Any |
| OS | Any |
| Deployed or local? | Reproducible in unit tests without running the app |

## Related Issue

Related to #3 (Game engine, scoring system) and #6 (Who Wrote It mini-game)
