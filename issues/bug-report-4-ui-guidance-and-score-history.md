# Feature Request

**Issue Title:** [Enhancement] In-game UI lacks action guidance; post-game results don't show per-round score history

## Problem Description

Two related UX gaps make the game feel disorienting:

### 1. During gameplay — player doesn't know what to do

The game area shows round/phase/mini-game labels, but when it's not the player's turn the screen just shows a passive message ("Use the chat panel to exchange messages while waiting for your turn"). When it is the player's turn, the decision widget appears without enough context — what are the stakes, what have others already done, what is the optimal window for sending chat messages?

Players unfamiliar with the rules have no in-UI guidance to orient them.

### 2. Post-game — only final scores, no score path

The game-over screen shows final rankings and total scores, but gives no information about *when* or *how* each player earned their points. A player who finishes second has no way to reconstruct what went wrong or what the winner did differently.

## Steps to Reproduce

1. Start a game and play through at least 2 rounds
2. During the "waiting for your turn" phase — observe that there is no indication of what is happening or what you should be doing
3. After the game ends — observe that the results screen only shows final totals with no per-round breakdown

## Expected Behavior

**During gameplay:**
- A status bar or sidebar panel showing the current phase, whose turn it is, and a one-line prompt for the player (e.g. "📝 Write your word now — you have 60 seconds" or "⏳ Waiting for others to guess — use this time to send private messages")
- Clear visual distinction between "your turn to act" vs. "waiting" states (e.g. highlight the decision panel, dim the waiting message)

**Post-game:**
- A per-round score breakdown showing each player's score delta per round (e.g. a table or animated timeline)
- Ideally: a replay / review mode that shows what happened in each round — who was the writer, what was the word, how each player guessed, how AI-to-AI conversations influenced decisions
- Optional stretch: animate the score path as a step chart so players can see momentum shifts round by round

## Severity

- [ ] **Blocker** — App is unusable or data is corrupted. Must fix before Demo Day.
- [x] **Major** — Core feature is broken but app still works for other tasks. Should fix.
- [ ] **Minor** — Cosmetic issue, typo, or edge case. Fix if time allows.

The game is playable but the lack of guidance significantly hurts the first-time player experience, and the absence of a score history removes a key source of post-game satisfaction and replayability.

## Evidence

Play through a full game as a first-time player. Note how often the screen is ambiguous about what to do next, and observe the game-over screen — there is no way to learn from the result.

## Environment (if relevant)

| Detail | Value |
|--------|-------|
| Browser | Any |
| Device | Desktop |
| OS | Any |
| Deployed or local? | Reproducible on localhost |

## Related Issue

Related to #4 (Chat UI and game screens), #7 (Polish & Integration)
