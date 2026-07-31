"""Decision/action prompt builders for AI agents.

The live Agent loop batches observations into one structured tick action.
Legacy single-action helpers still use TWO serial calls:
  - Phase A (decision): given the full context, produce a brief strategic
    intent — 1-3 short lines, no chat text, no game answer.
  - Phase B (action):   given the same context + the decision text, produce
    the actual output (chat reply, DM target+text, or game answer).

A persona's state- and trust-adjusted stance (cooperate / deceive / silent)
is shared by both phases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Stance guidance ──────────────────────────────────────────────────────────

_STANCE_GUIDANCE: Dict[str, str] = {
    "cooperate": (
        "Be honest and helpful in this exchange. Share information truthfully. "
        "If you have nothing useful, you may [skip], but do not deliberately mislead."
    ),
    "deceive": (
        "Mislead in this exchange. Plant false information, deny facts, deflect, "
        "or set a trap. Stay believable — over-the-top lies get caught."
    ),
    "silent": (
        "Stay silent this exchange. Output [skip] in the action phase."
    ),
}


def _stance_line(stance: Optional[str]) -> str:
    if not stance:
        return ""
    guidance = _STANCE_GUIDANCE.get(stance, "")
    return f"Your stance for this exchange: {stance.upper()} — {guidance}"


# ── Game rules ───────────────────────────────────────────────────────────────

GAME_FRAMEWORK: str = (
    "Game framework (applies to every mini-game):\n"
    "- Each round, every player has a send budget of 10 private DMs. Receiving is free.\n"
    "- All chat is 1-to-1 private DM. Nobody overhears your DMs with others.\n"
    "- Exact scores are private. You know your own score, not anyone else's.\n"
    "- Poison Bottle selection order publicly leaks relative score ranking, but not exact scores.\n"
    "- Any player may lie freely in chat. There is NO verification mechanism during play.\n"
    "- A player's own game choices (the secret word they wrote, the bottle they picked, "
    "the authors they guessed, etc.) are PRIVATE until the round-end reveal.\n"
    "- The only information you can rely on during a round is: (a) your own score and choices, "
    "(b) public roles/order/events, (c) what someone tells you in DM — which may be a lie.\n"
    "- After the round ends, the engine reveals every player's actual choices for that round, "
    "so past-round results are ground truth and a reliable source for spotting liars."
)


GAME_RULES: Dict[str, str] = {
    "guess_the_word": (
        "One player is the writer; everyone else is a guesser.\n"
        "- The writer secretly types one English word. PRIVATE — no other player ever sees it during the round.\n"
        "- Each guesser secretly types their guess. PRIVATE — no other player sees their guess during the round.\n"
        "- All choices are revealed only at round end.\n"
        "Scoring:\n"
        "- Each guesser who matches the writer's word: +1.\n"
        "- Writer: +1 only when at least one guesser is correct but NOT all of them. "
        "Writer's optimal strategy is a word that's guessable for some but not all."
    ),
    "who_wrote_it": (
        "Every player secretly writes one English word (PRIVATE). Then every player is shown "
        "the OTHER players' words (without authorship) and must attribute each word to a player.\n"
        "- Your own attribution choices are PRIVATE — no other player sees them during the round.\n"
        "- The words themselves become visible to all guessers in the attribution phase, but the "
        "authorship mapping is hidden until reveal.\n"
        "Scoring:\n"
        "- The guesser(s) with the most correct attributions: +1 (ties share, only if > 0 correct).\n"
        "- An author whose word is identified by SOME but NOT ALL eligible guessers: +1."
    ),
    "poison_bottle": (
        "Four bottles, one poisoned. Which bottle is poisoned is RANDOM per round and hidden from all "
        "players until reveal.\n"
        "- Players pick one bottle at a time in score-rank order (highest score first). "
        "Bottles are RETURNED to the pool, so every player faces the same four-bottle choice.\n"
        "- Selection ORDER is public (everyone sees who is up next). "
        "But the bottle each player picks is PRIVATE — only that player is told immediately whether "
        "their pick was poisoned. NO other player observes your pick.\n"
        "- The ONLY way another player learns what you picked is if you tell them in DM, "
        "and you may lie. Likewise, anyone telling you what they picked may be lying.\n"
        "- Selection order leaks relative score ranking even though exact scores are hidden.\n"
        "Scoring:\n"
        "- Drinking the poisoned bottle: -1. Safe bottle: 0."
    ),
}


# ── Formatters reused across builders ────────────────────────────────────────

def _format_scores(scores: Optional[Dict[str, int | str]]) -> str:
    if not scores:
        return "unknown"
    return ", ".join(f"{name}: {score}" for name, score in scores.items())


def _format_round_history(history: Optional[List[Dict[str, Any]]]) -> str:
    if not history:
        return "(none — this is the first round)"
    lines = []
    for h in history[-5:]:
        rn = h.get("round", "?")
        gt = h.get("game_type", "?")
        deltas = h.get("score_deltas", {})
        results = h.get("results", {}) or {}
        delta_str = ", ".join(
            f"{k}{('+' + str(v)) if v > 0 else v}"
            for k, v in deltas.items() if v != 0
        ) or "no change"
        if gt == "guess_the_word":
            extra = f"writer={results.get('writer')}, word={results.get('word')}, guesses={results.get('guesses')}"
        elif gt == "who_wrote_it":
            extra = f"words={results.get('words')}, attributions={results.get('attributions')}"
        elif gt == "poison_bottle":
            extra = f"poisoned={results.get('poisoned_bottle')}, choices={results.get('choices')}"
        else:
            extra = ""
        lines.append(f"- R{rn} ({gt}): {extra} | score Δ: {delta_str}")
    return "\n".join(lines)


def _format_recent_chat(messages: Optional[List[Dict[str, str]]]) -> str:
    if not messages:
        return "(no recent chat with this counterpart)"
    return "\n".join(
        f"  [{m['sender']} → {m['recipient']}] {m['text']}"
        for m in messages[-10:]
    )


def _common_context_block(
    *,
    score: int,
    send_budget: int | None,
    scores: Optional[Dict[str, int | str]],
    current_round: int | None,
    total_rounds: int | None,
    round_state_text: Optional[str],
    round_history: Optional[List[Dict[str, Any]]],
    recent_chat: Optional[List[Dict[str, str]]],
    game_type: str | None = None,
) -> List[str]:
    """Return the standard context lines every decision prompt shares."""
    budget_str = str(send_budget) if send_budget is not None else "unlimited"
    round_str = (
        f"{current_round}/{total_rounds}"
        if current_round is not None and total_rounds is not None
        else (str(current_round) if current_round is not None else "?")
    )

    lines = [
        GAME_FRAMEWORK,
        "",
        f"ROUND: {round_str}    YOUR SCORE: {score}    SENDS REMAINING: {budget_str}",
        f"VISIBLE SCORE INFORMATION: {_format_scores(scores)}",
    ]
    if game_type:
        rules = GAME_RULES.get(game_type, "(rules not provided)")
        lines.append(f"CURRENT MINI-GAME: {game_type}")
        lines.append(f"RULES:\n{rules}")
    if round_state_text:
        lines.append("")
        lines.append("CURRENT ROUND STATE (ground truth — do not contradict it):")
        lines.append(round_state_text)
    lines.append("")
    lines.append("PAST ROUNDS (use to spot patterns and past deception):")
    lines.append(_format_round_history(round_history))
    lines.append("")
    lines.append("RECENT DMs with the relevant counterpart:")
    lines.append(_format_recent_chat(recent_chat))
    return lines


# ── Chat reply — decision + action ───────────────────────────────────────────

def build_chat_reply_decision_prompt(
    *,
    sender_name: str,
    incoming_text: str,
    score: int,
    send_budget: int | None,
    scores: Optional[Dict[str, int | str]] = None,
    current_round: int | None = None,
    total_rounds: int | None = None,
    round_state_text: Optional[str] = None,
    round_history: Optional[List[Dict[str, Any]]] = None,
    recent_chat: Optional[List[Dict[str, str]]] = None,
    game_type: str | None = None,
    stance: Optional[str] = None,
) -> str:
    lines = _common_context_block(
        score=score,
        send_budget=send_budget,
        scores=scores,
        current_round=current_round,
        total_rounds=total_rounds,
        round_state_text=round_state_text,
        round_history=round_history,
        recent_chat=recent_chat,
        game_type=game_type,
    )
    lines.append("")
    lines.append(f'INCOMING DM from {sender_name}: "{incoming_text}"')
    lines.append("")
    sl = _stance_line(stance)
    if sl:
        lines.append(sl)
        lines.append("")
    lines.append(
        "Phase: DECISION. Return only a compact strategy summary:\n"
        "GOAL: <what this reply should accomplish>\n"
        "BELIEF: <the one fact, uncertainty, or past behavior that matters most>\n"
        "INTENT: <truth, deception, challenge, or silence and why>\n\n"
        "Do NOT write the reply yet. NO hidden chain-of-thought, quotes, or greeting. "
        "Use display names (Bunny / Fox / Stoneface / the human's name) — never internal IDs like ai_0."
    )
    return "\n".join(lines)


def build_chat_reply_action_prompt(
    *,
    sender_name: str,
    incoming_text: str,
    decision_text: str,
    score: int,
    send_budget: int | None,
    scores: Optional[Dict[str, int | str]] = None,
    current_round: int | None = None,
    total_rounds: int | None = None,
    round_state_text: Optional[str] = None,
    game_type: str | None = None,
    stance: Optional[str] = None,
) -> str:
    budget_str = str(send_budget) if send_budget is not None else "unlimited"
    round_str = (
        f"{current_round}/{total_rounds}"
        if current_round is not None and total_rounds is not None
        else (str(current_round) if current_round is not None else "?")
    )
    lines = [
        f"ROUND: {round_str}    YOUR SCORE: {score}    SENDS REMAINING: {budget_str}",
    ]
    if game_type:
        lines.append(f"Mini-game: {game_type}")
    if round_state_text:
        lines.append("")
        lines.append("CURRENT ROUND STATE (ground truth — do not contradict it):")
        lines.append(round_state_text)
    sl = _stance_line(stance)
    if sl:
        lines.append("")
        lines.append(sl)
    lines.append("")
    lines.append(f'INCOMING DM from {sender_name}: "{incoming_text}"')
    lines.append("YOUR DECISION (what you already decided to do):")
    lines.append(decision_text or "(no decision text)")
    lines.append("")
    lines.append(
        "Phase: ACTION. Produce the REPLY MESSAGE that executes your decision. "
        "Speak in your persona's voice. Keep it concise: usually 3-10 words "
        "and never more than 60 characters. "
        "Do not use internal IDs (e.g. ai_0/human). Use names. "
        "If after re-reading the situation a reply is not worth sending, output EXACTLY [skip]."
    )
    return "\n".join(lines)


# ── Proactive / public-info DM — decision + action ───────────────────────────

def build_outbound_decision_prompt(
    *,
    trigger_summary: str,
    candidate_targets: List[str],
    score: int,
    send_budget: int | None,
    scores: Optional[Dict[str, int | str]] = None,
    current_round: int | None = None,
    total_rounds: int | None = None,
    round_state_text: Optional[str] = None,
    round_history: Optional[List[Dict[str, Any]]] = None,
    recent_chat_summary: Optional[str] = None,
    game_type: str | None = None,
    stance: Optional[str] = None,
) -> str:
    """Decision prompt for AI-initiated DM (proactive or public-info-triggered).

    ``trigger_summary`` is plain text describing what just happened (e.g.
    "Round 2 ended. Results: …" or "(no trigger — proactive tick)").
    """
    lines = _common_context_block(
        score=score,
        send_budget=send_budget,
        scores=scores,
        current_round=current_round,
        total_rounds=total_rounds,
        round_state_text=round_state_text,
        round_history=round_history,
        recent_chat=None,  # broader summary below instead
        game_type=game_type,
    )
    if recent_chat_summary:
        lines.append("")
        lines.append("RECENT CHAT SUMMARY (all DMs you have memory of):")
        lines.append(recent_chat_summary)
    lines.append("")
    lines.append(f"TRIGGER: {trigger_summary}")
    lines.append(f"Eligible DM targets: {', '.join(candidate_targets) or '(nobody)'}")
    sl = _stance_line(stance)
    if sl:
        lines.append("")
        lines.append(sl)
    lines.append("")
    lines.append(
        "Phase: DECISION. Return only a compact strategy summary:\n"
        "GOAL: <what this trigger makes worth doing, if anything>\n"
        "BELIEF: <the key evidence, contradiction, or uncertainty>\n"
        "INTENT: <'DM <name>: <goal>' or 'hold: <reason>'>\n\n"
        "No hidden chain-of-thought and no chat text yet. "
        "Use display names — never internal IDs like ai_0."
    )
    return "\n".join(lines)


def build_outbound_action_prompt(
    *,
    decision_text: str,
    candidate_targets: List[str],
    score: int,
    send_budget: int | None,
    current_round: int | None = None,
    total_rounds: int | None = None,
    round_state_text: Optional[str] = None,
    game_type: str | None = None,
    stance: Optional[str] = None,
) -> str:
    budget_str = str(send_budget) if send_budget is not None else "unlimited"
    round_str = (
        f"{current_round}/{total_rounds}"
        if current_round is not None and total_rounds is not None
        else (str(current_round) if current_round is not None else "?")
    )
    lines = [
        f"ROUND: {round_str}    YOUR SCORE: {score}    SENDS REMAINING: {budget_str}",
    ]
    if game_type:
        lines.append(f"Mini-game: {game_type}")
    if round_state_text:
        lines.append("")
        lines.append("CURRENT ROUND STATE (ground truth — do not contradict it):")
        lines.append(round_state_text)
    sl = _stance_line(stance)
    if sl:
        lines.append("")
        lines.append(sl)
    lines.append("")
    lines.append("YOUR DECISION (what you already decided to do):")
    lines.append(decision_text or "(no decision text)")
    lines.append(f"Eligible DM targets (use one of these names): {', '.join(candidate_targets) or '(nobody)'}")
    lines.append("")
    lines.append(
        "Phase: ACTION. Produce the DM that executes your decision. "
        "Keep the message concise and never exceed 60 characters. "
        "Output format MUST be exactly TWO lines:\n"
        "  Line 1: TARGET: <one of the names above>\n"
        "  Line 2: <the message text in your persona's voice>\n"
        "If you decided not to send a DM, output EXACTLY: [skip]"
    )
    return "\n".join(lines)


# ── Batched Agent tick ───────────────────────────────────────────────────────

def build_tick_action_prompt(
    *,
    observations: List[str],
    game_action_name: str | None,
    game_payload: Optional[Dict[str, Any]],
    game_instruction: str | None,
    chat_trigger: str | None,
    chat_targets: List[str],
    chat_summary: str | None,
    score: int,
    send_budget: int | None,
    scores: Optional[Dict[str, int | str]] = None,
    current_round: int | None = None,
    total_rounds: int | None = None,
    round_state_text: Optional[str] = None,
    round_history: Optional[List[Dict[str, Any]]] = None,
    recent_chat_summary: Optional[str] = None,
    game_type: str | None = None,
    stance: Optional[str] = None,
) -> str:
    lines = _common_context_block(
        score=score,
        send_budget=send_budget,
        scores=scores,
        current_round=current_round,
        total_rounds=total_rounds,
        round_state_text=round_state_text,
        round_history=round_history,
        recent_chat=None,
        game_type=game_type,
    )
    lines.extend([
        "",
        "OBSERVATIONS SINCE THE PREVIOUS 10-SECOND TICK:",
        *(observations or ["(none)"]),
    ])
    if recent_chat_summary:
        lines.extend(["", "RECENT PRIVATE CHAT:", recent_chat_summary])

    lines.extend(["", "ACTION SLOTS FOR THIS TICK:"])
    if game_action_name:
        lines.append(
            f"- GAME (required): {game_action_name} with situation "
            f"{game_payload or {}}"
        )
        if game_instruction:
            lines.append(f"  Instruction: {game_instruction}")
    else:
        lines.append("- GAME: unavailable; omit game_action.")

    if chat_trigger and chat_targets:
        lines.append(
            f"- CHAT (optional, at most one): trigger={chat_trigger}; "
            f"context={chat_summary}; allowed target IDs={chat_targets}."
        )
    else:
        lines.append("- CHAT: unavailable; omit chat_action.")

    sl = _stance_line(stance)
    if sl:
        lines.extend(["", f"CHAT {sl}"])
    lines.extend([
        "",
        "Call act_in_tick exactly once. Include game_action whenever the GAME "
        "slot is required. Include chat_action only when one concise message "
        "(3-10 words, at most 60 characters) is worth sending. Game and chat "
        "may coexist. decision_summary must be brief and user-safe; do not "
        "include hidden chain-of-thought.",
    ])
    return "\n".join(lines)


# ── Game decision — decision + action ────────────────────────────────────────

def build_game_decision_prompt(
    *,
    game_type: str,
    situation: Dict[str, Any],
    score: int,
    send_budget: int | None,
    scores: Optional[Dict[str, int | str]] = None,
    current_round: int | None = None,
    total_rounds: int | None = None,
    round_state_text: Optional[str] = None,
    round_history: Optional[List[Dict[str, Any]]] = None,
    recent_chat_summary: Optional[str] = None,
    stance: Optional[str] = None,
) -> str:
    lines = _common_context_block(
        score=score,
        send_budget=send_budget,
        scores=scores,
        current_round=current_round,
        total_rounds=total_rounds,
        round_state_text=round_state_text,
        round_history=round_history,
        recent_chat=None,
        game_type=game_type,
    )
    if recent_chat_summary:
        lines.append("")
        lines.append("RECENT CHAT SUMMARY:")
        lines.append(recent_chat_summary)
    situation_str = "\n".join("  " + k + ": " + str(v) for k, v in situation.items())
    lines.append("")
    lines.append("SITUATION (what the engine sent you for this decision):")
    lines.append(situation_str)
    sl = _stance_line(stance)
    if sl:
        lines.append("")
        lines.append(sl)
    lines.append("")
    lines.append(
        "Phase: DECISION. Return only a compact strategy summary:\n"
        "GOAL: <the scoring outcome to optimize>\n"
        "BELIEF: <the strongest current-round evidence and relevant trust signal>\n"
        "PLAN: <one-line plan for the action phase>\n\n"
        "No hidden chain-of-thought. Do NOT output the final answer value yet. "
        "Use display names — never internal IDs like ai_0."
    )
    return "\n".join(lines)


def build_game_action_prompt(
    *,
    game_type: str,
    situation: Dict[str, Any],
    decision_text: str,
    score: int,
    send_budget: int | None,
    round_state_text: Optional[str] = None,
    stance: Optional[str] = None,
) -> str:
    budget_str = str(send_budget) if send_budget is not None else "unlimited"
    situation_str = "\n".join("  " + k + ": " + str(v) for k, v in situation.items())
    lines = [
        f"Mini-game: {game_type}",
        f"YOUR SCORE: {score}    SENDS REMAINING: {budget_str}",
    ]
    if round_state_text:
        lines.append("")
        lines.append("CURRENT ROUND STATE (ground truth — do not contradict it):")
        lines.append(round_state_text)
    sl = _stance_line(stance)
    if sl:
        lines.append("")
        lines.append(sl)
    lines.append("")
    lines.append("SITUATION:")
    lines.append(situation_str)
    lines.append("")
    lines.append("YOUR DECISION (what you already decided to do):")
    lines.append(decision_text or "(no decision text)")
    lines.append("")
    lines.append(
        "Phase: ACTION. Output ONLY the final answer value the action instruction "
        "asks for — no reasoning, no prose, no labels."
    )
    return "\n".join(lines)
