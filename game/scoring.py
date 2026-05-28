"""Scoring logic for all three mini-game types.

Scoring rules (from SPEC.md):
  Guess the Word:
    +1 per correct guesser
    +1 for writer if at least 1 but not all guessers are correct

  Who Wrote It:
    +1 for guesser(s) who identify the most authors correctly
    +1 for any writer whose word is identified by some but not all guessers

  Poison Bottle:
    -1 for the player who chose the poisoned bottle
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional


# Bottle names used in Poison Bottle.
BOTTLES: List[str] = ["Red", "Blue", "Green", "Yellow"]


def score_guess_the_word(
    writer_id: str,
    word: str,
    guesses: Dict[str, str],
) -> Dict[str, int]:
    """Return score deltas for one round of Guess the Word.

    Parameters
    ----------
    writer_id:
        The player who wrote the secret word.
    word:
        The secret word (case-insensitive comparison).
    guesses:
        ``{player_id: guessed_word}`` for every non-writer player.

    Returns
    -------
    ``{player_id: delta}`` covering the writer and all guessers.
    """
    word_norm = word.strip().lower()
    correct = [pid for pid, g in guesses.items() if g.strip().lower() == word_norm]
    total = len(guesses)

    deltas: Dict[str, int] = {pid: 0 for pid in list(guesses.keys()) + [writer_id]}

    for pid in correct:
        deltas[pid] += 1

    # Writer earns +1 only when at least one—but not every—guesser is correct.
    if 0 < len(correct) < total:
        deltas[writer_id] += 1

    return deltas


def score_who_wrote_it(
    words: Dict[str, str],
    attributions: Dict[str, Dict[str, str]],
) -> Dict[str, int]:
    """Return score deltas for one round of Who Wrote It.

    Parameters
    ----------
    words:
        ``{player_id: word_written}`` — the word each player submitted.
    attributions:
        ``{guesser_id: {true_author_id: guessed_author_id}}``
        For each guesser, a mapping from the *actual* author of a word to the
        guesser's claim of who wrote it.

    Returns
    -------
    ``{player_id: delta}`` for all players.
    """
    all_players = list(words.keys())
    deltas: Dict[str, int] = {pid: 0 for pid in all_players}

    # Count how many authors each guesser identified correctly.
    correct_counts: Dict[str, int] = {
        guesser: sum(
            1 for true_author, guessed in their_guesses.items()
            if guessed == true_author
        )
        for guesser, their_guesses in attributions.items()
    }

    # Best guesser(s) get +1 (ties all rewarded).
    if correct_counts:
        best = max(correct_counts.values())
        if best > 0:
            for pid, count in correct_counts.items():
                if count == best:
                    deltas[pid] += 1

    # Any writer whose word was guessed by *some* (not zero, not all) gets +1.
    # Important: "all" means all *eligible* guessers (players who actually
    # received that author's word), not len(attributions) blindly.
    for author_id in all_players:
        eligible_guessers = [
            guesser_id
            for guesser_id, their_guesses in attributions.items()
            if author_id in their_guesses
        ]
        total_guessers = len(eligible_guessers)
        if total_guessers == 0:
            continue
        correct_for_author = sum(
            1 for guesser_id in eligible_guessers
            if attributions[guesser_id].get(author_id) == author_id
        )
        if 0 < correct_for_author < total_guessers:
            deltas[author_id] += 1

    return deltas


def score_poison_bottle(
    choices: Dict[str, str],
    poisoned_bottle: str,
) -> Dict[str, int]:
    """Return score deltas for one round of Poison Bottle.

    Parameters
    ----------
    choices:
        ``{player_id: bottle_color}`` — every player's selection.
    poisoned_bottle:
        The color of the poisoned bottle (case-insensitive).

    Returns
    -------
    ``{player_id: delta}`` — ``-1`` for the player who drank poison, else ``0``.
    """
    poisoned_norm = poisoned_bottle.strip().lower()
    return {
        pid: (-1 if bottle.strip().lower() == poisoned_norm else 0)
        for pid, bottle in choices.items()
    }


def get_poison_bottle_order(
    player_ids: List[str],
    scores: Dict[str, int],
) -> List[str]:
    """Sort players by current score (highest first) for Poison Bottle.

    Ties are broken randomly so there is no systematic advantage for any
    tied player.

    Parameters
    ----------
    player_ids:
        All players participating in this round.
    scores:
        Current score for each player (missing players default to 0).
    """
    pairs = [(pid, scores.get(pid, 0)) for pid in player_ids]
    random.shuffle(pairs)           # randomise first so equal scores are shuffled
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in pairs]
