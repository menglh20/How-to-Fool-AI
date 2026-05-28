"""GameEngine — orchestrates rounds, selects mini-game types, manages phase
transitions, and applies scoring.

Concurrency model
-----------------
The engine runs as a daemon thread alongside the Streamlit human-IO thread
and three AIAgent daemon threads.  All coordination happens through
``SharedState``:

* ``broadcast_event`` / ``send_event_to_player`` deliver ``GameEvent``
  objects to player inboxes so AI agents and the UI can react.
* ``submit_choice`` (called by AI agents and the human-IO thread) unblocks
  ``_wait_for_choice`` via a per-player ``threading.Event``.
* ``threading.Event.wait(timeout=...)`` is used throughout so the engine
  blocks without holding any lock and without starving the GIL — other
  threads continue running freely while the engine waits.

Round sequence
--------------
For each round::

    select game type → reset send budgets → broadcast phase_change(playing)
    → run mini-game → apply scores → broadcast round_end
    → broadcast phase_change(reveal) → pause _REVEAL_DURATION seconds

After all rounds::

    set phase = "game_over" → broadcast game_over event
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Dict, List, Optional

from game.shared_state import GameEvent, SharedState

# ── constants ────────────────────────────────────────────────────────────────

# Rotation order for mini-game types.
_GAME_TYPES: List[str] = ["guess_the_word", "who_wrote_it", "poison_bottle"]

# Bottle colours used in Poison Bottle.  Two bottles: one poisoned, one safe.
# Bottles are returned to the pool after each pick, so every player faces the
# same two-bottle choice; only the selection order leaks score ranking.
_BOTTLES: List[str] = ["Red", "Blue"]

# Fall-back English words used when a player times out on a write-word prompt.
_FALLBACK_WORDS: List[str] = [
    "apple", "sky", "moon", "star", "ocean", "mountain", "flower", "cloud",
    "time", "dream", "hope", "freedom",
]

# Maximum seconds the engine waits on the reveal phase before forcing the
# next round.  The reveal is normally advanced by the human clicking
# "Next Round"; this timeout is just a safety net for stuck clients.
_REVEAL_DURATION: float = 600.0

# Per-phase wall-clock durations (seconds).  Everyone — human and AI —
# operates on the same clock; the AI is instructed to defer its first
# decision until the last 10 s of the window (or when it has used up its
# chat budget).
_WRITE_WORD_DURATION: float = 30.0
_GUESS_DURATION: float = 120.0           # guess_word & guess_authors (2 min)
_POISON_PICK_DURATION: float = 30.0      # per player, sequential


def _random_word() -> str:
    return random.choice(_FALLBACK_WORDS)


# ── engine ───────────────────────────────────────────────────────────────────

class GameEngine(threading.Thread):
    """Daemon thread that drives the full game loop.

    The engine now operates phase-by-phase on a single wall-clock budget
    shared by all players — no per-player timeouts.  Players (AI and human
    alike) may submit and re-submit their choice at any time within the
    phase window; at deadline, the engine reads the latest submission for
    each player and falls back to a random default if missing.
    """

    def __init__(
        self,
        state: SharedState,
        total_rounds: int,
        reveal_duration: float = _REVEAL_DURATION,
        write_word_duration: float = _WRITE_WORD_DURATION,
        guess_duration: float = _GUESS_DURATION,
        poison_pick_duration: float = _POISON_PICK_DURATION,
        # Back-compat shim — older tests pass ``choice_timeout`` to override
        # the per-phase windows for fast execution.  When given, it replaces
        # all three.
        choice_timeout: Optional[float] = None,
    ) -> None:
        super().__init__(daemon=True, name="GameEngine")
        self.state = state
        self.total_rounds = total_rounds
        self.reveal_duration = reveal_duration
        if choice_timeout is not None:
            self.write_word_duration = choice_timeout
            self.guess_duration = choice_timeout
            self.poison_pick_duration = choice_timeout
        else:
            self.write_word_duration = write_word_duration
            self.guess_duration = guess_duration
            self.poison_pick_duration = poison_pick_duration
        self._stop_event = threading.Event()
        self._last_game_type: Optional[str] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the engine to exit after the current wait completes."""
        self._stop_event.set()

    def run(self) -> None:
        self.state.set_round_info(0, self.total_rounds, "")
        self.state.set_phase("playing")

        for round_num in range(1, self.total_rounds + 1):
            if self._stop_event.is_set():
                break
            self._run_round(round_num)

        # ── game over ────────────────────────────────────────────────────────
        self.state.set_phase("game_over")
        final_scores = {
            pid: self.state.get_score(pid) for pid in self.state.player_ids
        }
        self.state.broadcast_event(
            GameEvent(
                event_type="game_over",
                payload={"final_scores": final_scores},
            )
        )

    # ── round orchestration ──────────────────────────────────────────────────

    def _run_round(self, round_num: int) -> None:
        game_type = self._select_game_type()

        # Fresh send budgets for this round.
        self.state.reset_send_budgets()

        self.state.set_round_info(round_num, self.total_rounds, game_type)
        self.state.set_phase("playing")

        # Pre-assign public roles BEFORE broadcasting phase_change, so every
        # listener (AI agents in particular) learns role info in the very
        # event that announces the round.  Otherwise an AI's public-info
        # trigger fires before role info has reached them and the LLM
        # reasons under "role: unknown" — which is how we ended up with
        # Bunny asking Eric for a hint while she was the writer.
        phase_payload: Dict[str, Any] = {
            "phase": "playing",
            "round": round_num,
            "total_rounds": self.total_rounds,
            "game_type": game_type,
        }
        writer_id: Optional[str] = None
        selection_order: Optional[List[str]] = None
        if game_type == "guess_the_word":
            all_players = self.state.player_ids
            writer_id = random.choice(all_players)
            phase_payload["writer"] = writer_id
            phase_payload["guessers"] = [
                p for p in all_players if p != writer_id
            ]
        elif game_type == "poison_bottle":
            # The pick order is purely a function of current scores
            # (already public) — fold it into the round announcement so
            # every AI knows from t=0 who picks 1st/2nd/3rd/4th.
            from game.scoring import get_poison_bottle_order
            all_players = self.state.player_ids
            scores = {pid: self.state.get_score(pid) for pid in all_players}
            selection_order = get_poison_bottle_order(all_players, scores)
            phase_payload["selection_order"] = selection_order

        self.state.broadcast_event(
            GameEvent(event_type="phase_change", payload=phase_payload)
        )

        # Dispatch to the appropriate mini-game runner.
        if game_type == "guess_the_word":
            results = self._run_guess_the_word(round_num, writer_id=writer_id)
        elif game_type == "who_wrote_it":
            results = self._run_who_wrote_it(round_num)
        else:
            results = self._run_poison_bottle(
                round_num, selection_order=selection_order
            )

        # Apply score deltas atomically.
        for pid, delta in results.get("score_deltas", {}).items():
            if delta != 0:
                self.state.update_score(pid, delta)
        scores_after_round = {
            pid: self.state.get_score(pid) for pid in self.state.player_ids
        }

        self.state.broadcast_event(
            GameEvent(
                event_type="round_end",
                payload={
                    "round": round_num,
                    "game_type": game_type,
                    "results": results,
                    "score_deltas": results.get("score_deltas", {}),
                    "scores_after_round": scores_after_round,
                },
            )
        )

        # Reveal phase — UI shows results, players can still chat.
        self.state.reset_continue()
        self.state.set_phase("reveal")
        self.state.broadcast_event(
            GameEvent(
                event_type="phase_change",
                payload={"phase": "reveal", "round": round_num},
            )
        )

        # Wait until the human clicks "Next Round", reveal_duration elapses,
        # or stop() is called.  Poll in short slices so stop() can interrupt
        # promptly without busy-spinning.
        _POLL = 0.5
        deadline = time.monotonic() + self.reveal_duration
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self.state.wait_for_continue(timeout=min(remaining, _POLL)):
                break

    # ── game type rotation ───────────────────────────────────────────────────

    def _select_game_type(self) -> str:
        """Select a random game type, avoiding consecutive repeats when possible."""
        choices = [g for g in _GAME_TYPES if g != self._last_game_type]
        game_type = random.choice(choices or _GAME_TYPES)
        self._last_game_type = game_type
        return game_type

    # ── core blocking primitives ──────────────────────────────────────────────

    def _wait_phase(
        self,
        deadline_unix_ts: float,
        done_check=None,
    ) -> None:
        """Block until *deadline_unix_ts* (wall clock) passes or stop().

        If *done_check* is provided, the wait exits early as soon as it
        returns ``True`` — used by write-word phases to advance the moment
        everybody has submitted.  Guess phases pass no done_check so the
        full window is used for chat development.
        """
        _POLL = 0.5
        while not self._stop_event.is_set():
            if done_check is not None and done_check():
                break
            remaining = deadline_unix_ts - time.time()
            if remaining <= 0:
                break
            if self._stop_event.wait(timeout=min(remaining, _POLL)):
                break

    def _wait_for_choice(self, player_id: str, timeout: float) -> Any:
        """Block until *player_id* submits a choice, *timeout* expires, or
        the engine is stopped.

        Retained for the sequential poison-bottle picks and the legacy
        tests.  Phase-window mini-games use :meth:`_wait_phase` instead.
        """
        event = self.state.get_choice_event(player_id)
        deadline = time.monotonic() + timeout
        _POLL = 0.25
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if event.wait(timeout=min(remaining, _POLL)):
                break
        return self.state.get_choice(player_id)

    # ── mini-game: Guess the Word ─────────────────────────────────────────────

    def _run_guess_the_word(
        self,
        round_num: int,
        writer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one round of Guess the Word and return a results dict.

        If *writer_id* is provided (the common path — pre-assigned in
        :meth:`_run_round` so it can be put into the phase_change broadcast),
        use it; otherwise fall back to a random pick.
        """
        from game.scoring import score_guess_the_word

        all_players = self.state.player_ids
        if writer_id is None:
            writer_id = random.choice(all_players)
        guessers = [p for p in all_players if p != writer_id]

        # Phase 1 — writer submits a secret word (locked once submitted).
        self.state.reset_choices([writer_id])
        deadline = time.time() + self.write_word_duration
        self.state.send_event_to_player(
            writer_id,
            GameEvent(
                event_type="make_decision",
                payload={
                    "game_type": "guess_the_word",
                    "action": "write_word",
                    "round": round_num,
                    "your_role": "writer",
                    "deadline_ts": deadline,
                    "phase_duration": self.write_word_duration,
                },
            ),
        )
        # Write-word is locked after submit; advance as soon as the writer
        # has committed instead of dragging out the full 30 s.
        self._wait_phase(
            deadline,
            done_check=lambda: self.state.get_choice(writer_id) is not None,
        )
        writer_word = self.state.get_choice(writer_id) or _random_word()

        # Phase 2 — all guessers submit simultaneously (3 min window).
        self.state.reset_choices(guessers)
        deadline = time.time() + self.guess_duration
        for guesser in guessers:
            self.state.send_event_to_player(
                guesser,
                GameEvent(
                    event_type="make_decision",
                    payload={
                        "game_type": "guess_the_word",
                        "action": "guess_word",
                        "round": round_num,
                        "writer": writer_id,
                        "your_role": "guesser",
                        "deadline_ts": deadline,
                        "phase_duration": self.guess_duration,
                    },
                ),
            )
        self._wait_phase(deadline)
        guesses: Dict[str, str] = {
            g: (self.state.get_choice(g) or _random_word()) for g in guessers
        }

        deltas = score_guess_the_word(writer_id, writer_word, guesses)
        return {
            "writer": writer_id,
            "word": writer_word,
            "guesses": guesses,
            "score_deltas": deltas,
        }

    # ── mini-game: Who Wrote It ───────────────────────────────────────────────

    def _run_who_wrote_it(self, round_num: int) -> Dict[str, Any]:
        """Run one round of Who Wrote It and return a results dict."""
        from game.scoring import score_who_wrote_it

        all_players = self.state.player_ids

        # Phase 1 — everyone writes a word simultaneously (locked on submit).
        self.state.reset_choices(all_players)
        deadline = time.time() + self.write_word_duration
        for pid in all_players:
            self.state.send_event_to_player(
                pid,
                GameEvent(
                    event_type="make_decision",
                    payload={
                        "game_type": "who_wrote_it",
                        "action": "write_word",
                        "round": round_num,
                        "deadline_ts": deadline,
                        "phase_duration": self.write_word_duration,
                    },
                ),
            )
        # Advance as soon as every player has committed.
        self._wait_phase(
            deadline,
            done_check=lambda: all(
                self.state.get_choice(p) is not None for p in all_players
            ),
        )
        words: Dict[str, str] = {
            pid: (self.state.get_choice(pid) or _random_word()) for pid in all_players
        }

        # Phase 2 — everyone guesses authors simultaneously (3 min window).
        self.state.reset_choices(all_players)
        deadline = time.time() + self.guess_duration
        author_order: Dict[str, List[str]] = {}
        for guesser in all_players:
            other_authors = sorted(p for p in all_players if p != guesser)
            author_order[guesser] = other_authors
            self.state.send_event_to_player(
                guesser,
                GameEvent(
                    event_type="make_decision",
                    payload={
                        "game_type": "who_wrote_it",
                        "action": "guess_authors",
                        "round": round_num,
                        "words": [words[a] for a in other_authors],
                        "candidate_authors": other_authors,
                        "deadline_ts": deadline,
                        "phase_duration": self.guess_duration,
                    },
                ),
            )
        self._wait_phase(deadline)

        attributions: Dict[str, Dict[str, str]] = {}
        for guesser in all_players:
            choice = self.state.get_choice(guesser)
            attributions[guesser] = _parse_attribution(
                choice, author_order[guesser]
            )

        deltas = score_who_wrote_it(words, attributions)
        return {
            "words": words,
            "attributions": attributions,
            "score_deltas": deltas,
        }

    # ── mini-game: Poison Bottle ──────────────────────────────────────────────

    def _run_poison_bottle(
        self,
        round_num: int,
        selection_order: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run one round of Poison Bottle and return a results dict.

        Two bottles (one poisoned, one safe) are presented to every player.
        Players choose in score-ranked order (highest first); bottles are
        returned to the pool after each pick, so every player faces the same
        two-bottle choice.  Each chooser is informed immediately, via a
        targeted ``poison_result`` event, whether their pick was poisoned.

        *selection_order* is pre-computed in :meth:`_run_round` so it can be
        included in the phase_change broadcast — fall back to local
        computation if the caller didn't supply one.
        """
        from game.scoring import get_poison_bottle_order, score_poison_bottle

        poisoned = random.choice(_BOTTLES)
        all_players = self.state.player_ids
        if selection_order is None:
            scores = {pid: self.state.get_score(pid) for pid in all_players}
            selection_order = get_poison_bottle_order(all_players, scores)

        choices: Dict[str, str] = {}

        # Players choose one at a time in score-ranked order (highest first).
        # Bottles are NOT depleted; every player sees both available.  Each
        # player has ``poison_pick_duration`` seconds to commit.  Drinking is
        # final — the engine advances as soon as a choice is submitted (no
        # Change Decision for poison).  Order itself is the information leak.
        for position, pid in enumerate(selection_order):
            self.state.reset_choices([pid])
            deadline = time.time() + self.poison_pick_duration

            # Tell every player who is up to pick this turn so spectators
            # can see "X is choosing now" in real time.
            self.state.broadcast_event(
                GameEvent(
                    event_type="poison_picker_change",
                    payload={
                        "round": round_num,
                        "picker": pid,
                        "position": position + 1,
                        "total": len(selection_order),
                        "selection_order": selection_order,
                        "deadline_ts": deadline,
                        "phase_duration": self.poison_pick_duration,
                    },
                )
            )

            self.state.send_event_to_player(
                pid,
                GameEvent(
                    event_type="make_decision",
                    payload={
                        "game_type": "poison_bottle",
                        "action": "choose_bottle",
                        "round": round_num,
                        "available_bottles": list(_BOTTLES),
                        "selection_order": selection_order,
                        "your_position": position + 1,
                        "deadline_ts": deadline,
                        "phase_duration": self.poison_pick_duration,
                    },
                ),
            )

            # Event-driven wait — exit immediately on the picker's submit.
            choice = self._wait_for_choice(pid, timeout=self.poison_pick_duration)
            if not isinstance(choice, str) or choice not in _BOTTLES:
                choice = random.choice(_BOTTLES)
            choices[pid] = choice

            # Immediate per-player feedback so the chooser knows their fate
            # before the round-end reveal.
            is_poisoned = choice.strip().lower() == poisoned.strip().lower()
            self.state.send_event_to_player(
                pid,
                GameEvent(
                    event_type="poison_result",
                    payload={
                        "round": round_num,
                        "choice": choice,
                        "is_poisoned": is_poisoned,
                    },
                ),
            )

        deltas = score_poison_bottle(choices, poisoned)
        return {
            "poisoned_bottle": poisoned,
            "choices": choices,
            "selection_order": selection_order,
            "score_deltas": deltas,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_attribution(
    submission: Optional[str],
    expected_authors: List[str],
) -> Dict[str, str]:
    """Parse a comma-separated attribution string into a mapping.

    Expected format: ``"author_id1,author_id2,author_id3"`` — one guessed
    author per position matching *expected_authors*.  Falls back to a random
    shuffle when the submission is missing or malformed (e.g. timeout).

    Parameters
    ----------
    submission:
        Raw string from the agent/player, or ``None`` on timeout.
    expected_authors:
        The ordered list of true author IDs that the guesser was shown words
        for.

    Returns
    -------
    ``{true_author_id: guessed_author_id}``
    """
    if submission:
        parts = [p.strip() for p in submission.split(",")]
        if len(parts) == len(expected_authors):
            return dict(zip(expected_authors, parts))
    # Timeout or malformed — random attribution.
    shuffled = list(expected_authors)
    random.shuffle(shuffled)
    return dict(zip(expected_authors, shuffled))
