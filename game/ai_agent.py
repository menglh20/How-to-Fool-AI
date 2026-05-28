"""AIAgent — autonomous AI player running as a daemon thread.

Each LLM-driven moment now runs as TWO serial Opus calls:
  Phase A — decision: brief strategic intent (no chat text)
  Phase B — action:   the actual output (chat reply, DM, or game answer)

A stance (cooperate / deceive / silent) is rolled fresh at each trigger
using the persona's ``STANCE_PROBS``.  ``silent`` short-circuits without
spending tokens and is logged as a ``silent_skip`` entry.

Triggers that fire the two-step flow:
  - chat reply        (incoming ChatMessage)
  - public info       (phase_change/poison_picker_change/round_end events)
  - game decision     (deferred-then-committed make_decision payload)
  - proactive chat    (legacy tick-based dice roll, lowered probability)
"""

from __future__ import annotations

import random
import re
import threading
import time
import types
from typing import TYPE_CHECKING, Any, Optional

from game.shared_state import ChatMessage, GameEvent

if TYPE_CHECKING:
    from game.shared_state import SharedState

# Maximum number of (role, content) pairs kept as rolling context per agent.
_MAX_CONTEXT_PAIRS = 10
_HUMAN_ID = "human"
_RESERVED_REPLY_BUDGET = 3

# AI commit-trigger: act in the last N seconds of a phase if not already.
_DECIDE_TAIL_SECONDS = 10.0

# Re-evaluation cooldown for Change Decision.
_CHANGE_DECISION_COOLDOWN = 20.0
_CHANGE_DECISION_MSG_THRESHOLD = 2

# Token caps per phase.  Decision needs room for explicit REASONING bullets;
# action stays compact since it's just the chat line or game answer.
_DECISION_MAX_TOKENS = 500
_ACTION_CHAT_MAX_TOKENS = 200
_ACTION_GAME_MAX_TOKENS = 120


class AIAgent(threading.Thread):
    """Autonomous AI player.

    Parameters
    ----------
    player_id:
        Must match one of the IDs registered in *state* (e.g. "ai_0").
    persona:
        A module or object exposing: NAME, EMOJI, SYSTEM_PROMPT,
        DEFAULT_REPLY, CHAT_INITIATIVE_PROB, STANCE_PROBS.
    state:
        The shared SharedState instance.
    """

    def __init__(
        self,
        player_id: str,
        persona: types.ModuleType,
        state: "SharedState",
    ) -> None:
        super().__init__(daemon=True, name=f"AIAgent-{player_id}")
        self.player_id = player_id
        self.persona = persona
        self.state = state
        self._stop_event = threading.Event()
        # Rolling chat context — list of {"role": ..., "content": ...} dicts.
        self._context: list[dict[str, str]] = []
        # Pending game decisions deferred for "chat first / commit late".
        self._pending_decisions: dict[str, dict] = {}
        # Last ~10 round_end payloads — used by decision prompts to spot lies.
        self._round_history: list[dict] = []
        # Live snapshot of the CURRENT round — roles, who has picked, etc.
        self._round_state: dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────
    # Thread lifecycle
    # ──────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the agent to exit its run loop after the current tick."""
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._tick()
            time.sleep(random.uniform(1.0, 3.0))

    # ──────────────────────────────────────────────────────────────────
    # Main tick
    # ──────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        inbox_items = self.state.get_my_messages(self.player_id)
        chats = [i for i in inbox_items if isinstance(i, ChatMessage)]
        events = [i for i in inbox_items if isinstance(i, GameEvent)]
        chats.sort(key=lambda m: m.sender != _HUMAN_ID)

        if chats:
            for p in self._pending_decisions.values():
                if p.get("decided"):
                    p["new_msgs_since_decision"] = (
                        p.get("new_msgs_since_decision", 0) + len(chats)
                    )

        for item in chats:
            self._handle_chat(item)
        for item in events:
            self._handle_event(item)

        # Drive deferred game decisions through Make / Change.
        self._resolve_pending_decisions()

        # Lowered proactive tick — still optional, less spammy.
        if self._can_initiate_proactive():
            if random.random() < self.persona.CHAT_INITIATIVE_PROB:
                self._proactive_chat()

    # ──────────────────────────────────────────────────────────────────
    # Stance + two-step LLM helpers
    # ──────────────────────────────────────────────────────────────────

    def _pick_stance(self) -> str:
        """Roll a stance (cooperate / deceive / silent) using the persona's probabilities."""
        probs = getattr(self.persona, "STANCE_PROBS", {"cooperate": 1.0})
        keys = list(probs.keys())
        weights = list(probs.values())
        return random.choices(keys, weights=weights, k=1)[0]

    def _log_silent_skip(self, trigger: str) -> None:
        """Record a token-free silent skip in the LLM log for auditing."""
        from game.llm_logger import log_call
        from game.llm_client import get_active_model
        log_call({
            "agent_id": self.player_id,
            "persona": self.persona.NAME,
            "model": get_active_model(),
            "trigger": trigger,
            "phase": "silent_skip",
            "stance": "silent",
            "latency_ms": 0,
            "system_prompt": "",
            "user_messages": [],
            "response": "",
        })

    def _two_step(
        self,
        *,
        trigger: str,
        stance: str,
        decision_prompt: str,
        action_prompt_builder,
        action_max_tokens: int = _ACTION_CHAT_MAX_TOKENS,
        use_strategic: bool = False,
    ) -> tuple[str, str]:
        """Run phase A (decision) then phase B (action), with logging.

        ``action_prompt_builder`` is a callable taking the decision string
        and returning the phase-B user message text.

        Returns ``(decision_text, action_text)``.
        """
        from game.llm_client import call_llm, call_strategic
        caller = call_strategic if use_strategic else call_llm

        # ── Phase A — decision ──
        decision = caller(
            system_prompt=self._system_prompt(),
            messages=self._recent_context() + [
                {"role": "user", "content": decision_prompt}
            ],
            default_reply="",
            max_tokens=_DECISION_MAX_TOKENS,
            agent_id=self.player_id,
            persona=self.persona.NAME,
            trigger=trigger,
            phase="decision",
            stance=stance,
        )
        # The decision text is stored as the "thinking" payload alongside
        # the outbound chat message and revealed in the end-of-game
        # timeline.  Scrub any leaked internal IDs (ai_0/human/...) before
        # it reaches the user or the action prompt.
        decision = self._sanitize_chat_text(decision)
        self._append_context("user", decision_prompt)
        self._append_context("assistant", decision)

        # ── Phase B — action ──
        action_prompt = action_prompt_builder(decision)
        action = caller(
            system_prompt=self._system_prompt(),
            messages=[{"role": "user", "content": action_prompt}],
            default_reply=self.persona.DEFAULT_REPLY,
            max_tokens=action_max_tokens,
            agent_id=self.player_id,
            persona=self.persona.NAME,
            trigger=trigger,
            phase="action",
            stance=stance,
        )
        return decision, action

    # ──────────────────────────────────────────────────────────────────
    # Inbox handlers
    # ──────────────────────────────────────────────────────────────────

    def _handle_chat(self, msg: ChatMessage) -> None:
        """Reply (or stay silent) to an incoming ChatMessage via two-step Opus."""
        if not self._has_budget():
            if msg.sender == _HUMAN_ID:
                self.state.send_system_message(
                    sender=self.player_id,
                    recipient=_HUMAN_ID,
                    text=(
                        f"{self.persona.NAME} has used up this round's send budget; "
                        "it will refresh next round."
                    ),
                )
            return

        stance = self._pick_stance()
        if stance == "silent":
            self._log_silent_skip(trigger="chat_reply")
            return

        from prompts.templates import (
            build_chat_reply_decision_prompt,
            build_chat_reply_action_prompt,
        )
        round_info = self.state.get_round_info()
        sender_name = self._player_label(msg.sender)

        decision_prompt = build_chat_reply_decision_prompt(
            sender_name=sender_name,
            incoming_text=msg.text,
            score=self.state.get_score(self.player_id),
            send_budget=self.state.get_send_budget(self.player_id),
            scores=self._all_scores(),
            current_round=round_info.get("current_round"),
            total_rounds=round_info.get("total_rounds"),
            round_state_text=self._round_state_text(),
            round_history=self._round_history,
            recent_chat=self._recent_chat_with(msg.sender),
            game_type=round_info.get("game_type"),
            stance=stance,
        )

        def build_action(decision_text: str) -> str:
            return build_chat_reply_action_prompt(
                sender_name=sender_name,
                incoming_text=msg.text,
                decision_text=decision_text,
                score=self.state.get_score(self.player_id),
                send_budget=self.state.get_send_budget(self.player_id),
                scores=self._all_scores(),
                current_round=round_info.get("current_round"),
                total_rounds=round_info.get("total_rounds"),
                round_state_text=self._round_state_text(),
                game_type=round_info.get("game_type"),
                stance=stance,
            )

        decision, raw = self._two_step(
            trigger="chat_reply",
            stance=stance,
            decision_prompt=decision_prompt,
            action_prompt_builder=build_action,
        )
        reply = raw.strip()
        if not reply or reply.lower() == "[skip]":
            return

        reply = self._sanitize_chat_text(reply)
        self._append_context("user", "(action turn — see decision above)")
        self._append_context("assistant", reply)
        try:
            self.state.send_message(
                self.player_id, msg.sender, reply, thinking=decision or None
            )
        except RuntimeError:
            pass

    def _handle_event(self, event: GameEvent) -> None:
        """React to a GameEvent.  Pure-state events keep round_state honest;
        public-info events also fire a two-step LLM outbound."""
        # ── Round / phase lifecycle ──
        if event.event_type == "round_end":
            # State-only event — log analysis showed 100% skip on the
            # outbound LLM trigger, so we no longer pay for it.
            payload = event.payload or {}
            self._round_history.append(payload)
            if len(self._round_history) > 10:
                self._round_history = self._round_history[-10:]
            self._pending_decisions.clear()
            self._round_state = {}
            return

        if event.event_type == "phase_change":
            payload = event.payload or {}
            phase = payload.get("phase")
            if phase == "playing":
                self._round_state = {
                    "round": payload.get("round"),
                    "total_rounds": payload.get("total_rounds"),
                    "game_type": payload.get("game_type"),
                    "selection_order": [],
                    "picked_so_far": [],
                }
                # Capture public role info that came on the broadcast — for
                # guess_the_word the writer's identity is public from the
                # start of the round, so we MUST seed it before any LLM
                # trigger fires.  Otherwise the AI would reason under
                # "role: unknown" and ask weird questions (e.g. the writer
                # asking guessers what they wrote).
                gt = payload.get("game_type")
                if gt == "guess_the_word":
                    writer = payload.get("writer")
                    if writer:
                        self._round_state["writer_id"] = writer
                        self._round_state["your_role"] = (
                            "writer" if writer == self.player_id else "guesser"
                        )
                elif gt == "who_wrote_it":
                    # Every player both writes (phase 1) and guesses authors
                    # (phase 2).  Phase 1 is the first thing that starts so
                    # we seed the role to writer right away.
                    self._round_state["your_role"] = "writer"
                elif gt == "poison_bottle":
                    # Selection order is public from t=0 (it's just sorted
                    # by the public scoreboard).  Seed it so the AI doesn't
                    # ask "who picks first" of someone in the chat.
                    order = payload.get("selection_order") or []
                    if order:
                        self._round_state["selection_order"] = list(order)
                        try:
                            self._round_state["your_position"] = (
                                order.index(self.player_id) + 1
                            )
                        except ValueError:
                            pass
                    self._round_state["your_role"] = "picker"
                # phase_playing has ~56% productive yield — keep the trigger.
                self._maybe_outbound_for_public_event(
                    trigger="public_info_phase_playing",
                    summary=(
                        f"Phase changed to playing — Round "
                        f"{payload.get('round')}, mini-game "
                        f"{payload.get('game_type', self._round_state.get('game_type'))}"
                    ),
                )
            elif phase == "reveal":
                # State-only — 100% skip in log analysis. Drop the LLM call.
                self._pending_decisions.clear()
            return

        # ── Poison-bottle picker tracking ──
        if event.event_type == "poison_picker_change":
            # State-only — 100% skip on outbound trigger.  The proactive
            # tick will still fire DMs during the picking window.
            payload = event.payload or {}
            picker = payload.get("picker")
            prev = self._round_state.get("current_picker_id")
            if prev and prev != picker:
                self._round_state.setdefault("picked_so_far", []).append(prev)
            self._round_state["current_picker_id"] = picker
            self._round_state["selection_order"] = payload.get(
                "selection_order", self._round_state.get("selection_order", [])
            )
            return

        if event.event_type == "poison_result":
            payload = event.payload or {}
            self._round_state["your_pick"] = payload.get("choice")
            self._round_state["your_pick_result"] = payload.get("is_poisoned")
            return

        if event.event_type != "make_decision":
            return

        # ── make_decision → role bookkeeping + immediate or deferred decide ──
        payload = event.payload or {}
        action = payload.get("action", "")

        if action == "write_word":
            self._round_state["your_role"] = payload.get("your_role", "writer")
        elif action == "guess_word":
            self._round_state["your_role"] = "guesser"
            self._round_state["writer_id"] = payload.get("writer")
        elif action == "guess_authors":
            self._round_state["your_role"] = "guesser"
        elif action == "choose_bottle":
            self._round_state["your_role"] = "picker"
            self._round_state["selection_order"] = payload.get(
                "selection_order", self._round_state.get("selection_order", [])
            )
            self._round_state["your_position"] = payload.get("your_position")

        if action == "write_word":
            # 30 s window — no info to gather, decide right away.
            self._decide_write_word(payload)
            return

        self._pending_decisions[action] = {
            "payload": payload,
            "deadline_ts": payload.get("deadline_ts"),
            "decided": False,
            "new_msgs_since_decision": 0,
            "last_reconsider_ts": 0.0,
        }

    # ──────────────────────────────────────────────────────────────────
    # Deferred game decisions
    # ──────────────────────────────────────────────────────────────────

    def _resolve_pending_decisions(self) -> None:
        if not self._pending_decisions:
            return
        now = time.time()
        budget = self.state.get_send_budget(self.player_id)
        budget_empty = budget is not None and budget <= 0

        for action in list(self._pending_decisions.keys()):
            p = self._pending_decisions[action]
            deadline = p.get("deadline_ts") or 0
            time_left = deadline - now

            if not p["decided"]:
                if budget_empty or time_left <= _DECIDE_TAIL_SECONDS:
                    self._dispatch_action(action, p["payload"])
                    p["decided"] = True
                    p["new_msgs_since_decision"] = 0
                    p["last_reconsider_ts"] = now
                    if action == "choose_bottle":
                        del self._pending_decisions[action]
                continue

            if time_left <= 1.0:
                del self._pending_decisions[action]
                continue

            enough_new_info = (
                p["new_msgs_since_decision"] >= _CHANGE_DECISION_MSG_THRESHOLD
            )
            cool = (now - p["last_reconsider_ts"]) >= _CHANGE_DECISION_COOLDOWN
            if enough_new_info and cool:
                self._dispatch_action(action, p["payload"], reconsider=True)
                p["new_msgs_since_decision"] = 0
                p["last_reconsider_ts"] = now

    def _dispatch_action(
        self,
        action: str,
        payload: dict,
        *,
        reconsider: bool = False,
    ) -> None:
        if reconsider:
            prev = self.state.get_choice(self.player_id)
            payload = {**payload, "previous_choice": prev, "reconsider": True}
        if action == "guess_word":
            self._decide_guess_word(payload)
        elif action == "guess_authors":
            self._decide_guess_authors(payload)
        elif action == "choose_bottle":
            self._decide_choose_bottle(payload)

    # ──────────────────────────────────────────────────────────────────
    # Outbound DM trigger (proactive + public-info-driven)
    # ──────────────────────────────────────────────────────────────────

    def _maybe_outbound_for_public_event(self, trigger: str, summary: str) -> None:
        """Roll a stance + try a two-step outbound based on a public event."""
        if not self._has_budget() or not self._can_initiate_proactive():
            return
        stance = self._pick_stance()
        if stance == "silent":
            self._log_silent_skip(trigger=trigger)
            return
        self._outbound_two_step(trigger=trigger, stance=stance, summary=summary)

    def _proactive_chat(self) -> None:
        """Legacy tick-based dice-roll proactive DM (lowered probability)."""
        stance = self._pick_stance()
        if stance == "silent":
            self._log_silent_skip(trigger="proactive_chat")
            return
        self._outbound_two_step(
            trigger="proactive_chat",
            stance=stance,
            summary="(no specific trigger — periodic proactive tick)",
        )

    def _outbound_two_step(self, *, trigger: str, stance: str, summary: str) -> None:
        from prompts.templates import (
            build_outbound_decision_prompt,
            build_outbound_action_prompt,
        )

        others = [p for p in self.state.player_ids if p != self.player_id]
        target_names = [self._player_label(p) for p in others]
        round_info = self.state.get_round_info()

        decision_prompt = build_outbound_decision_prompt(
            trigger_summary=summary,
            candidate_targets=target_names,
            score=self.state.get_score(self.player_id),
            send_budget=self.state.get_send_budget(self.player_id),
            scores=self._all_scores(),
            current_round=round_info.get("current_round"),
            total_rounds=round_info.get("total_rounds"),
            round_state_text=self._round_state_text(),
            round_history=self._round_history,
            recent_chat_summary=self._recent_chat_summary(),
            game_type=round_info.get("game_type"),
            stance=stance,
        )

        def build_action(decision_text: str) -> str:
            return build_outbound_action_prompt(
                decision_text=decision_text,
                candidate_targets=target_names,
                score=self.state.get_score(self.player_id),
                send_budget=self.state.get_send_budget(self.player_id),
                current_round=round_info.get("current_round"),
                total_rounds=round_info.get("total_rounds"),
                round_state_text=self._round_state_text(),
                game_type=round_info.get("game_type"),
                stance=stance,
            )

        decision, raw = self._two_step(
            trigger=trigger,
            stance=stance,
            decision_prompt=decision_prompt,
            action_prompt_builder=build_action,
        )

        target, message_text = self._parse_outbound_action(raw, target_names, others)
        if target is None or not message_text:
            return  # [skip] or malformed
        message_text = self._sanitize_chat_text(message_text)

        self._append_context("user", "(action turn — see decision above)")
        self._append_context("assistant", message_text)
        try:
            self.state.send_message(
                self.player_id, target, message_text, thinking=decision or None
            )
        except RuntimeError:
            pass

    def _parse_outbound_action(
        self,
        raw: str,
        target_names: list[str],
        target_ids: list[str],
    ) -> tuple[Optional[str], str]:
        """Parse a phase-B outbound response into (target_id, text).

        Expected format:
            TARGET: <name>
            <message body...>
        Returns ``(None, "")`` for ``[skip]`` or unparseable output.
        """
        text = raw.strip()
        if not text or text.lower() == "[skip]":
            return None, ""

        # Look for TARGET: directive on first non-empty line.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None, ""
        first = lines[0].strip()
        m = re.match(r"(?i)\s*target\s*:\s*(.+)", first)
        if not m:
            return None, ""
        target_name = m.group(1).strip().strip(",.;")
        # Resolve case-insensitive.
        matched_idx = None
        for i, n in enumerate(target_names):
            if n.lower() == target_name.lower():
                matched_idx = i
                break
        if matched_idx is None:
            return None, ""
        target_id = target_ids[matched_idx]
        body = "\n".join(lines[1:]).strip()
        if not body:
            return None, ""
        if body.lower() == "[skip]":
            return None, ""
        return target_id, body

    # ──────────────────────────────────────────────────────────────────
    # Game decisions — write/guess/bottle (Phase A + B with stance)
    # ──────────────────────────────────────────────────────────────────

    def _decide_write_word(self, payload: dict) -> None:
        game_type = payload.get("game_type", "")
        # Words that already appeared in past rounds — feed back so the AI
        # actively diversifies instead of converging on the LLM's prior
        # ("apple" / "tree" / "sky" / "love" / etc.).
        avoid_pool = self._past_words_pool()
        avoid_clause = (
            f" Avoid repeating any of these previously-played words: {avoid_pool}." if avoid_pool else ""
        )
        if game_type == "guess_the_word":
            instruction = (
                "You are the WRITER. Pick a single English word at the SWEET SPOT — "
                "guessable for some but not for all (so the writer-bonus rule pays off). "
                "Skip both clichés ('apple', 'dog', 'sky', 'love') and ultra-obscure picks. "
                "Examples of the sweet spot: 'mirror', 'thunder', 'whisper', 'shadow', 'compass'. "
                "Lean into something a person of your persona would naturally pick."
                f"{avoid_clause} "
                "Output only the word itself, lowercase, no punctuation."
            )
        else:  # who_wrote_it
            instruction = (
                "Pick a single English word that REVEALS something about your persona but is "
                "still plausible coming from anyone — variety is what makes the attribution game work. "
                "Strongly avoid the most obvious defaults ('apple', 'dog', 'sky', 'love', 'tree') "
                "since other players will gravitate to them too."
                f"{avoid_clause} "
                "Lean into a word that matches your persona's flavour. "
                "Output only the word itself, lowercase, no punctuation."
            )
        raw = self._make_game_decision(payload, instruction)
        word = self._first_word_token(raw) or random.choice(
            ["mirror", "thunder", "whisper", "shadow", "compass", "echo"]
        )
        self.state.submit_choice(self.player_id, word)
        self._round_state["my_submission"] = word

    def _past_words_pool(self) -> str:
        """Return a comma-joined list of words played in past rounds.

        Used to discourage the LLM from picking the same word it (or anyone)
        already used — directly addresses the "2nd and 3rd words are often
        identical" complaint, which is LLM prior convergence.
        """
        seen: list[str] = []
        for h in self._round_history[-5:]:
            results = h.get("results", {}) or {}
            if h.get("game_type") == "guess_the_word":
                w = results.get("word")
                if w:
                    seen.append(str(w))
            elif h.get("game_type") == "who_wrote_it":
                words = results.get("words", {}) or {}
                for w in words.values():
                    if w:
                        seen.append(str(w))
        # De-dup while preserving order.
        deduped: list[str] = []
        for w in seen:
            wl = w.strip().lower()
            if wl and wl not in deduped:
                deduped.append(wl)
        return ", ".join(deduped[:20])

    def _decide_guess_word(self, payload: dict) -> None:
        instruction = (
            "Guess the single English word the writer chose. "
            "Output only the word itself, lowercase, no punctuation."
        )
        raw = self._make_game_decision(payload, instruction)
        word = self._first_word_token(raw) or random.choice(
            ["apple", "sky", "moon", "star"]
        )
        self.state.submit_choice(self.player_id, word)
        self._round_state["my_submission"] = word

    def _decide_guess_authors(self, payload: dict) -> None:
        words = payload.get("words", [])
        candidates = payload.get("candidate_authors", [])
        instruction = (
            f"Guess which player wrote each of the following words. Words: {words}. "
            f"Candidate authors: {candidates}. "
            f"Output the guessed author IDs in the same order as the words, "
            f"separated by commas. For example: ai_0,ai_1,ai_2"
        )
        raw = self._make_game_decision(payload, instruction)
        self.state.submit_choice(self.player_id, raw.strip())
        self._round_state["my_submission"] = raw.strip()

    def _decide_choose_bottle(self, payload: dict) -> None:
        available = payload.get("available_bottles", ["Red", "Blue", "Green", "Yellow"])
        instruction = (
            f"Choose one bottle from the following list: {available}. "
            f"Output only the bottle color in English, e.g. Red"
        )
        raw = self._make_game_decision(payload, instruction)
        choice_token = self._first_word_token(raw)
        match = next(
            (b for b in available if choice_token and b.lower() == choice_token.lower()),
            None,
        )
        choice = match or random.choice(available)
        self.state.submit_choice(self.player_id, choice)
        self._round_state["your_pick"] = choice

    def _make_game_decision(self, payload: dict, instruction: str) -> str:
        """Run two-step game decision for a single action.

        Returns the raw phase-B action string for downstream parsing.
        """
        from prompts.templates import (
            build_game_decision_prompt,
            build_game_action_prompt,
        )
        situation = {**payload, "instruction": instruction}

        stance = self._pick_stance()
        if stance == "silent":
            self._log_silent_skip(trigger="game_decision")
            # Silent on a game decision still needs an answer — fall back.
            stance = "cooperate"  # downgrade so we still produce SOMETHING

        round_info = self.state.get_round_info()
        game_type = situation.get("game_type", "unknown_game")

        decision_prompt = build_game_decision_prompt(
            game_type=game_type,
            situation=situation,
            score=self.state.get_score(self.player_id),
            send_budget=self.state.get_send_budget(self.player_id),
            scores=self._all_scores(),
            current_round=round_info.get("current_round"),
            total_rounds=round_info.get("total_rounds"),
            round_state_text=self._round_state_text(),
            round_history=self._round_history,
            recent_chat_summary=self._recent_chat_summary(),
            stance=stance,
        )

        def build_action(decision_text: str) -> str:
            return build_game_action_prompt(
                game_type=game_type,
                situation=situation,
                decision_text=decision_text,
                score=self.state.get_score(self.player_id),
                send_budget=self.state.get_send_budget(self.player_id),
                round_state_text=self._round_state_text(),
                stance=stance,
            )

        decision, raw = self._two_step(
            trigger="game_decision",
            stance=stance,
            decision_prompt=decision_prompt,
            action_prompt_builder=build_action,
            action_max_tokens=_ACTION_GAME_MAX_TOKENS,
            use_strategic=True,
        )
        # Context append: only the decision narrative — the action is the
        # game choice string, which is uninteresting for chat continuity.
        self._append_context("user", "(game action turn — answer extracted)")
        self._append_context("assistant", decision[:300])
        return raw

    # ──────────────────────────────────────────────────────────────────
    # Context / state helpers
    # ──────────────────────────────────────────────────────────────────

    def _recent_context(self) -> list[dict[str, str]]:
        return self._context[-(2 * _MAX_CONTEXT_PAIRS):]

    def _append_context(self, role: str, content: str) -> None:
        self._context.append({"role": role, "content": content})
        if len(self._context) > 2 * _MAX_CONTEXT_PAIRS + 4:
            self._context = self._context[-(2 * _MAX_CONTEXT_PAIRS):]

    def _recent_chat_with(self, other_id: str) -> list[dict[str, str]]:
        """Return the recent DM exchange between *other_id* and self.

        Pulled from SharedState's chat_history so we slice the actual
        canonical message stream rather than our own rolling context.
        """
        history = self.state.get_chat_history()
        out: list[dict[str, str]] = []
        for entry in history:
            if entry.get("type") != "chat":
                continue
            s = entry.get("sender")
            r = entry.get("recipient")
            if (s == self.player_id and r == other_id) or (
                s == other_id and r == self.player_id
            ):
                out.append({
                    "sender": self._player_label(s),
                    "recipient": self._player_label(r),
                    "text": entry.get("text", ""),
                })
        return out[-10:]

    def _recent_chat_summary(self) -> str:
        """Compact recent-chat summary across ALL counterparts for outbound prompts."""
        history = self.state.get_chat_history()
        lines: list[str] = []
        for entry in history[-15:]:
            if entry.get("type") != "chat":
                continue
            s = entry.get("sender")
            r = entry.get("recipient")
            if self.player_id not in (s, r):
                continue
            lines.append(
                f"[{self._player_label(s)} → {self._player_label(r)}] {entry.get('text', '')}"
            )
        return "\n".join(lines) if lines else "(no DMs yet)"

    def _all_scores(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for pid in self.state.player_ids:
            try:
                name = self.state.get_display_name(pid)
            except Exception:
                name = pid
            out[name] = self.state.get_score(pid)
        return out

    def _summarise_round_end(self, payload: dict) -> str:
        rn = payload.get("round")
        gt = payload.get("game_type")
        deltas = payload.get("score_deltas", {})
        scored = ", ".join(
            f"{self._player_label(k)}: {v:+d}"
            for k, v in deltas.items() if v
        ) or "no score changes"
        return f"Round {rn} ({gt}) just ended. Score deltas: {scored}."

    def _round_state_text(self) -> str:
        """Render the current-round snapshot as a plain-text block for prompts."""
        rs = self._round_state
        if not rs or not rs.get("game_type"):
            return "(round just starting — no role info yet)"

        rn = rs.get("round")
        total = rs.get("total_rounds")
        gt = rs.get("game_type")
        my_role = rs.get("your_role", "unknown")
        my_name = self._player_label(self.player_id)

        lines = [
            f"Round {rn}/{total} · Mini-game: {gt}",
            f"YOU ({my_name}) — current role: {my_role}",
        ]

        if gt == "guess_the_word":
            writer_id = rs.get("writer_id")
            writer_name = (
                self._player_label(writer_id) if writer_id else "(unknown)"
            )
            if my_role == "writer":
                lines.append(
                    "You are the WRITER. The other players are GUESSERS — they "
                    "do NOT write a word; they each try to guess yours."
                )
                if "my_submission" in rs:
                    lines.append(f"You already submitted your secret word: {rs['my_submission']}")
            else:
                lines.append(
                    f"Writer this round: {writer_name}. {writer_name} secretly "
                    "wrote ONE word. All other players (including you) are GUESSERS — "
                    "you do NOT write a word, you each try to guess the writer's word."
                )
                if "my_submission" in rs:
                    lines.append(f"You already submitted your guess: {rs['my_submission']}")

        elif gt == "who_wrote_it":
            lines.append(
                "Phase 1: every player writes ONE word in secret. "
                "Phase 2: every player attributes the other 3 players' words to who wrote each. "
                "Both authoring and guessing apply to everyone."
            )
            if "my_submission" in rs:
                lines.append(f"You already submitted this turn: {rs['my_submission']}")

        elif gt == "poison_bottle":
            order = rs.get("selection_order") or []
            picked = rs.get("picked_so_far") or []
            picker = rs.get("current_picker_id")
            order_names = ", ".join(self._player_label(p) for p in order) or "?"
            picked_names = (
                ", ".join(self._player_label(p) for p in picked) if picked
                else "(nobody yet)"
            )
            picker_name = self._player_label(picker) if picker else "?"
            lines.append(f"Selection order (highest score first): {order_names}")
            lines.append(f"Already picked: {picked_names}")
            lines.append(f"Currently choosing right now: {picker_name}")
            your_pick = rs.get("your_pick")
            your_res = rs.get("your_pick_result")
            if your_pick:
                tag = (
                    "POISONED" if your_res is True
                    else "SAFE" if your_res is False
                    else "result pending"
                )
                lines.append(f"YOUR pick this round: {your_pick} ({tag}).")
            else:
                if picker == self.player_id:
                    lines.append("YOU are picking right now (not yet submitted).")
                elif self.player_id in picked:
                    lines.append("YOU have already picked (no result recorded).")
                else:
                    lines.append(
                        f"YOU have NOT picked yet — your turn comes later in the order. "
                        "Anyone you DM may not have picked either; do not ask them what they 'drank' if it's not their turn yet."
                    )
            lines.append(
                "Reminder: each player's pick is PRIVATE. The only way to learn what "
                "someone else picked is by asking them in DM, and they may lie."
            )

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # Misc helpers
    # ──────────────────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        return self.persona.SYSTEM_PROMPT

    def _has_budget(self) -> bool:
        budget = self.state.get_send_budget(self.player_id)
        return budget is None or budget > 0

    def _can_initiate_proactive(self) -> bool:
        budget = self.state.get_send_budget(self.player_id)
        if budget is None:
            return True
        return budget > _RESERVED_REPLY_BUDGET

    def _first_word_token(self, raw: str) -> str:
        cleaned = re.sub(r"<[^>]*>", " ", raw)
        match = re.search(r"[A-Za-z][A-Za-z'-]*", cleaned)
        return match.group(0).lower() if match else ""

    def _sanitize_chat_text(self, text: str) -> str:
        for pid in self.state.player_ids:
            try:
                name = self.state.get_display_name(pid)
            except Exception:
                continue
            if not name or name == pid:
                continue
            text = re.sub(rf"\b{re.escape(pid)}\b", name, text)
        return text

    def _player_label(self, player_id: str) -> str:
        try:
            return self.state.get_display_name(player_id)
        except Exception:
            fallback = {
                "ai_0": "Bunny",
                "ai_1": "Fox",
                "ai_2": "Stoneface",
                _HUMAN_ID: "You",
            }
            return fallback.get(player_id, player_id)
