# Bug Report

**Issue Title:** [Bug] Mini-game type rotation is fixed and predictable, not random — players can anticipate every round in advance

## Steps to Reproduce

1. Start a game with 3 or more rounds
2. Round 1 is always **🔮 Guess the Word**
3. Round 2 is always **🎭 Who Wrote It**
4. Round 3 is always **☠️ Poison Bottle**
5. Round 4 (if playing 5+ rounds) is always **🔮 Guess the Word** again
6. The sequence repeats indefinitely in the same fixed order

## Expected Behavior

For a social deduction game where unpredictability is central to the experience, the mini-game type for each round should be selected **randomly** (or at minimum, shuffled each game). Players should not be able to predict — from Round 1 onward — exactly what game type is coming next.

## Actual Behavior

The game type is determined by a fixed modulo cycle:

```python
# game/game_engine.py
_GAME_TYPES = ["guess_the_word", "who_wrote_it", "poison_bottle"]

def _select_game_type(self) -> str:
    game_type = _GAME_TYPES[self._type_index % len(_GAME_TYPES)]
    self._type_index += 1
    return game_type
```

A player who has played once immediately knows the full schedule for every future game. This eliminates surprise and reduces strategic tension — particularly for Poison Bottle, where knowing it is always Round 3 (and Round 6, Round 9…) allows players to calibrate their chat strategy in advance.

## Root Cause (for developer)

`_select_game_type` cycles through `_GAME_TYPES` in a fixed order. A minimal fix is to shuffle the order each game, or draw randomly while avoiding consecutive repeats:

```python
import random

def _select_game_type(self) -> str:
    """Pick a random game type, avoiding the same type two rounds in a row."""
    choices = [t for t in _GAME_TYPES if t != self._last_game_type]
    game_type = random.choice(choices)
    self._last_game_type = game_type
    return game_type
```

This preserves the guarantee that no two consecutive rounds share the same type (which the original comment in the test suite checks for) while making the sequence unpredictable across games.

Note: the existing test `test_no_consecutive_repeats_in_full_cycle` in `tests/test_game_engine.py` would still pass with this change. The test `test_cycles_through_all_types` asserts a specific fixed sequence — that test would need to be updated to only check that all types appear within N rounds, not that they appear in a particular order.

## Severity

- [ ] **Blocker** — App is unusable or data is corrupted. Must fix before Demo Day.
- [ ] **Major** — Core feature is broken but app still works for other tasks. Should fix.
- [x] **Minor** — Cosmetic issue, typo, or edge case. Fix if time allows.

The game functions correctly; only replayability and strategic unpredictability are affected. However, for a social deduction game whose pitch is "every alliance is temporary" and "every word could be truth or lies," a deterministic round schedule is a design contradiction.

## Evidence

Play two separate 3-round games. Both will follow the identical sequence: Guess the Word → Who Wrote It → Poison Bottle. The sequence is hard-coded and does not change between sessions.

## Environment (if relevant)

| Detail | Value |
|--------|-------|
| Browser | Any |
| Device | Desktop |
| OS | Any |
| Deployed or local? | Reproducible on localhost |

## Related Issue

Related to #3 (Game engine and round management)
