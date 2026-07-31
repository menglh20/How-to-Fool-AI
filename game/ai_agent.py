"""AIAgent — autonomous AI player running as a daemon thread.

The live loop batches observations every ten seconds and asks the model for
one structured tick action containing at most one game action and one chat
action. Game actions execute first. Direct single-action helpers retain the
two-call Decision → Action flow for focused tests and debugging.

A stance (cooperate / deceive / silent) is adjusted by persona, trust, and
trigger context. Direct conversations keep a stable stance for the round.
``silent`` short-circuits without spending tokens and is logged.

Observations considered by the tick:
  - chat reply        (incoming ChatMessage)
  - game evidence     (phase, poison-picker/result, and round-end events)
  - game decision     (deferred-then-committed make_decision payload)
  - proactive chat    (local persona dice roll)
"""

from __future__ import annotations

import random
import re
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional

from game.shared_state import ChatMessage, GameEvent
from game.memory import AgentMemory

if TYPE_CHECKING:
    from game.llm_client import LLMClient
    from game.shared_state import SharedState

# Maximum number of (role, content) pairs kept as rolling context per agent.
_MAX_CONTEXT_PAIRS = 10
_HUMAN_ID = "human"
_RESERVED_REPLY_BUDGET = 3

# AI commit-trigger: act in the last N seconds of a phase if not already.
_DECIDE_TAIL_SECONDS = 10.0
_TICK_SECONDS = 10.0

# Re-evaluation cooldown for Change Decision.
_CHANGE_DECISION_COOLDOWN = 20.0
_CHANGE_DECISION_MSG_THRESHOLD = 2

# Decision is a compact strategy summary; action is the chat line/game answer.
_DECISION_MAX_TOKENS = 180
_ACTION_CHAT_MAX_TOKENS = 200
_ACTION_GAME_MAX_TOKENS = 120


class AIAgent(threading.Thread):
    """Autonomous AI player.

    Parameters
    ----------
    player_id:
        Must match one of the IDs registered in *state* (e.g. "ai_0").
    persona:
        A config or compatible object exposing: NAME, EMOJI, SYSTEM_PROMPT,
        DEFAULT_REPLY, CHAT_INITIATIVE_PROB, STANCE_PROBS.
    state:
        The shared SharedState instance.
    """

    def __init__(
        self,
        player_id: str,
        persona: Any,
        state: "SharedState",
        llm_client: "LLMClient | None" = None,
        tick_seconds: float = _TICK_SECONDS,
    ) -> None:
        super().__init__(daemon=True, name=f"AIAgent-{player_id}")
        self.player_id = player_id
        self.persona = persona
        self.state = state
        if llm_client is None:
            from game.llm_client import get_active_client
            llm_client = get_active_client()
        if tick_seconds <= 0:
            raise ValueError("tick_seconds must be positive")
        self.llm_client = llm_client
        self.tick_seconds = float(tick_seconds)
        self._stop_event = threading.Event()
        self._collecting_tick = False
        self._tick_event_opportunities: list[dict[str, Any]] = []
        # Rolling chat context — list of {"role": ..., "content": ...} dicts.
        self._context: list[dict[str, str]] = []
        # Pending game decisions deferred for "chat first / commit late".
        self._pending_decisions: dict[str, dict] = {}
        # Last ~10 round_end payloads — used by decision prompts to spot lies.
        self._round_history: list[dict] = []
        # Live snapshot of the CURRENT round — roles, who has picked, etc.
        self._round_state: dict[str, Any] = {}
        self._round_stances: dict[tuple[int, str], str] = {}
        self.memory = AgentMemory(
            player_id,
            initial_trust=getattr(persona, "INITIAL_TRUST", 0.5),
            truth_reward=getattr(persona, "TRUTH_REWARD", 0.1),
            lie_penalty=getattr(persona, "LIE_PENALTY", 0.15),
        )

    # ──────────────────────────────────────────────────────────────────
    # Thread lifecycle
    # ──────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the agent to exit its run loop after the current tick."""
        self._stop_event.set()

    def run(self) -> None:
        try:
            index = int(self.player_id.rsplit("_", 1)[-1])
        except ValueError:
            index = 0
        if self._stop_event.wait((index % 3) * self.tick_seconds / 3):
            return
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            self._tick()
            next_tick += self.tick_seconds
            now = time.monotonic()
            if next_tick <= now:
                skipped = int((now - next_tick) // self.tick_seconds) + 1
                next_tick += skipped * self.tick_seconds
            if self._stop_event.wait(next_tick - now):
                return

    # ──────────────────────────────────────────────────────────────────
    # Main tick
    # ──────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        inbox_items = self.state.get_my_messages(self.player_id)
        chats = [i for i in inbox_items if isinstance(i, ChatMessage)]
        events = [i for i in inbox_items if isinstance(i, GameEvent)]
        observations = [
            (
                f"DM from {self._player_label(msg.sender)}: "
                f"{msg.text[:300]}"
            )
            for msg in chats
        ]
        observations.extend(
            f"{event.event_type}: {str(event.payload)[:600]}"
            for event in events
        )

        if chats:
            for p in self._pending_decisions.values():
                if p.get("decided"):
                    p["new_msgs_since_decision"] = (
                        p.get("new_msgs_since_decision", 0) + len(chats)
                    )

        for item in chats:
            self._ingest_chat(item)

        self._collecting_tick = True
        self._tick_event_opportunities = []
        try:
            for item in events:
                self._handle_event(item)
        finally:
            self._collecting_tick = False

        due_game = self._next_due_decision()
        chat = self._choose_tick_chat(chats)
        if due_game is None and chat is None:
            return

        game_succeeded = self._act_on_tick(
            observations=observations,
            due_game=due_game,
            chat=chat,
        )
        if due_game is not None:
            action, _, reconsider = due_game
            self._finish_due_decision(
                action,
                succeeded=game_succeeded,
                reconsider=reconsider,
            )

    # ──────────────────────────────────────────────────────────────────
    # Stance + two-step LLM helpers
    # ──────────────────────────────────────────────────────────────────

    def _pick_stance(
        self,
        *,
        trigger: str = "",
        counterpart: str | None = None,
    ) -> str:
        """Choose a persona- and trust-weighted stance.

        Private game choices are always strategic rather than deceptive:
        nobody else sees the action. Direct-chat stances persist for a round
        so one character does not randomly flip tone with the same player.
        """
        if trigger == "game_decision":
            return "cooperate"

        round_number = self.state.get_round_info().get("current_round", 0)
        stance_key = (round_number, counterpart) if counterpart else None
        if stance_key and stance_key in self._round_stances:
            return self._round_stances[stance_key]

        probabilities = dict(
            getattr(self.persona, "STANCE_PROBS", {"cooperate": 1.0})
        )
        if counterpart:
            trust = self.memory.trust_for(counterpart)
            probabilities["cooperate"] = (
                probabilities.get("cooperate", 0) * (0.5 + trust)
            )
            distrust = 1.5 - trust
            probabilities["deceive"] = (
                probabilities.get("deceive", 0) * distrust
            )
            probabilities["silent"] = (
                probabilities.get("silent", 0) * distrust
            )

        keys = list(probabilities)
        weights = list(probabilities.values())
        stance = random.choices(keys, weights=weights, k=1)[0]
        if stance_key:
            self._round_stances[stance_key] = stance
        return stance

    def _log_silent_skip(self, trigger: str) -> None:
        """Record a token-free silent skip in the LLM log for auditing."""
        from game.llm_logger import log_call
        from game.llm_client import get_active_backend, get_active_model
        record = {
            "agent_id": self.player_id,
            "persona": self.persona.NAME,
            "provider": get_active_backend(),
            "model": get_active_model(),
            "trigger": trigger,
            "phase": "silent_skip",
            "stance": "silent",
            "latency_ms": 0,
            "system_prompt": "",
            "user_messages": [],
            "response": "",
        }
        self.state.record_trace(
            "silent_skip",
            actor_id=self.player_id,
            payload={"trigger": trigger, **record},
        )
        if getattr(self.llm_client, "record_calls", False):
            log_call(record)

    def _tool_two_step(
        self,
        *,
        trigger: str,
        stance: str,
        decision_prompt: str,
        action_prompt: Callable[[str], str],
        registry,
        fallback_call,
        action_max_tokens: int = _ACTION_CHAT_MAX_TOKENS,
    ):
        """Decide, request one validated tool call, retry once, then fallback."""
        from game.llm_client import LLMRequest, ToolCall, ToolResult

        action_id = uuid.uuid4().hex
        decision_response = self.llm_client.complete(LLMRequest(
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
        ))
        self._trace_llm_response(
            decision_response,
            trigger=trigger,
            phase="decision",
            action_id=action_id,
        )
        decision = self._sanitize_chat_text(decision_response.text)
        self._append_context("user", decision_prompt)
        self._append_context("assistant", decision)

        try:
            rendered_action_prompt = action_prompt(decision)
        except ValueError as exc:
            self.state.record_trace(
                "stale_decision_discarded",
                actor_id=self.player_id,
                payload={
                    "action": fallback_call.name,
                    "reason": str(exc),
                    "stage": "after_decision",
                },
            )
            return decision, fallback_call, ToolResult(
                fallback_call.id,
                {"error": str(exc)},
                True,
            )
        prompt = (
            f"{rendered_action_prompt}\n\n"
            "Use exactly one of the provided tools. Do not answer with plain "
            "text. Keep decision_summary brief and safe to show the player."
        )
        last_error = "No tool call returned"
        for attempt in range(2):
            response = self.llm_client.complete(LLMRequest(
                system_prompt=self._system_prompt(),
                messages=[{"role": "user", "content": prompt}],
                default_reply="",
                max_tokens=action_max_tokens,
                tools=registry.definitions,
                tool_choice="required",
                agent_id=self.player_id,
                persona=self.persona.NAME,
                trigger=trigger,
                phase="tool_action",
                stance=stance,
                metadata={"attempt": attempt + 1},
            ))
            self._trace_llm_response(
                response,
                trigger=trigger,
                phase="tool_action",
                attempt=attempt + 1,
                action_id=action_id,
            )
            if len(response.tool_calls) == 1:
                call = response.tool_calls[0]
                result = registry.execute(call)
            elif len(response.tool_calls) > 1:
                call = ToolCall("", "", {})
                result = ToolResult(
                    "",
                    {"error": "Exactly one tool call is allowed per action"},
                    True,
                )
            else:
                call = ToolCall("", "", {})
                result = ToolResult("", {"error": last_error}, True)
            self._log_tool_execution(
                call,
                result,
                attempt=attempt + 1,
                fallback=False,
                action_id=action_id,
            )
            if not result.is_error:
                return decision, call, result
            last_error = str(result.output.get("error", "Invalid tool call"))
            if last_error.startswith((
                "Decision deadline has passed",
                "Game actions are not allowed",
                "Action for ",
            )):
                return decision, call, result
            prompt += (
                f"\n\nPrevious tool call was rejected: {last_error}. "
                "Correct the arguments and call one allowed tool."
            )

        result = registry.execute(fallback_call)
        self._log_tool_execution(
            fallback_call,
            result,
            attempt=3,
            fallback=True,
            action_id=action_id,
        )
        return decision, fallback_call, result

    def _log_tool_execution(
        self,
        call,
        result,
        *,
        attempt: int,
        fallback: bool,
        action_id: str,
    ) -> None:
        self.state.record_trace(
            "tool_execution",
            actor_id=self.player_id,
            payload={
                "provider": self.llm_client.provider,
                "model": self.llm_client.model,
                "tool": call.name,
                "arguments": call.arguments,
                "result": result.output,
                "is_error": result.is_error,
                "attempt": attempt,
                "fallback": fallback,
                "action_id": action_id,
            },
        )
        if not getattr(self.llm_client, "record_calls", False):
            return
        from game.llm_logger import log_call

        round_info = self.state.get_round_info()
        log_call({
            "event_type": "tool_execution",
            "provider": self.llm_client.provider,
            "model": self.llm_client.model,
            "agent_id": self.player_id,
            "round": round_info.get("current_round"),
            "game_type": round_info.get("game_type"),
            "tool": call.name,
            "arguments": call.arguments,
            "result": result.output,
            "is_error": result.is_error,
            "attempt": attempt,
            "fallback": fallback,
            "action_id": action_id,
        })

    def _trace_llm_response(
        self,
        response,
        *,
        trigger: str,
        phase: str,
        attempt: int | None = None,
        action_id: str | None = None,
    ) -> None:
        self.state.record_trace(
            "llm_response",
            actor_id=self.player_id,
            payload={
                "provider": response.provider,
                "model": response.model,
                "trigger": trigger,
                "phase": phase,
                "attempt": attempt,
                "action_id": action_id,
                "text": response.text,
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                    for call in response.tool_calls
                ],
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency_ms": response.latency_ms,
                "finish_reason": response.finish_reason,
                "error": response.error,
            },
        )

    def _require_decision_window(self, payload: dict) -> None:
        deadline = payload.get("deadline_ts")
        if deadline is not None and time.time() > float(deadline):
            raise ValueError("Decision deadline has passed")
        if self.state.get_phase() != "playing":
            raise ValueError("Game actions are not allowed in this phase")
        active_game = self.state.get_round_info().get("game_type")
        requested_game = payload.get("game_type")
        if active_game and requested_game and active_game != requested_game:
            raise ValueError(
                f"Action for {requested_game!r} is not allowed during "
                f"{active_game!r}"
            )

    # ──────────────────────────────────────────────────────────────────
    # Inbox handlers
    # ──────────────────────────────────────────────────────────────────

    def _ingest_chat(self, msg: ChatMessage) -> None:
        """Store every message observed during a tick before choosing a reply."""
        round_info = self.state.get_round_info()
        claim = self.memory.ingest_message(
            source=msg.sender,
            text=msg.text,
            round=round_info.get("current_round", 0),
            game_type=round_info.get("game_type", ""),
        )
        self.state.record_trace(
            "memory_claim_added",
            actor_id=self.player_id,
            payload={
                "memory_id": claim.memory_id,
                "source": claim.source,
                "fact_key": claim.fact_key,
                "value": claim.value,
                "confidence": claim.confidence,
                "injection_flagged": claim.injection_flagged,
            },
        )

    def _handle_chat(self, msg: ChatMessage) -> None:
        """Reply (or stay silent) using the direct two-step debug flow."""
        self._ingest_chat(msg)
        round_info = self.state.get_round_info()
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

        stance = self._pick_stance(
            trigger="chat_reply",
            counterpart=msg.sender,
        )
        if stance == "silent":
            self._log_silent_skip(trigger="chat_reply")
            return

        from prompts.templates import (
            build_chat_reply_decision_prompt,
            build_chat_reply_action_prompt,
        )
        sender_name = self._player_label(msg.sender)

        decision_prompt = build_chat_reply_decision_prompt(
            sender_name=sender_name,
            incoming_text=msg.text,
            score=self.state.get_score(self.player_id),
            send_budget=self.state.get_send_budget(self.player_id),
            scores=self._visible_scores(),
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
                scores=self._visible_scores(),
                current_round=round_info.get("current_round"),
                total_rounds=round_info.get("total_rounds"),
                round_state_text=self._round_state_text(),
                game_type=round_info.get("game_type"),
                stance=stance,
            )

        from game.game_tools import (
            MAX_CHAT_CHARACTERS,
            STAY_SILENT,
            send_message_tool,
        )
        from game.llm_client import ToolCall
        from game.tool_runtime import ToolRegistry

        registry = ToolRegistry()

        def send_reply(
            target: str,
            content: str,
            decision_summary: str = "",
        ) -> dict:
            if target != msg.sender:
                raise ValueError("Replies may only target the message sender")
            from game.security import sanitize_model_output

            reply = sanitize_model_output(
                self._sanitize_chat_text(content),
                max_length=MAX_CHAT_CHARACTERS,
            )
            if not reply:
                raise ValueError("Message cannot be empty")
            self.state.send_message(
                self.player_id,
                target,
                reply,
                thinking=decision_summary or None,
            )
            return {"sent": True, "target": target, "content": reply}

        registry.register(send_message_tool([msg.sender]), send_reply)
        registry.register(
            STAY_SILENT,
            lambda reason="": {"sent": False, "reason": reason},
        )

        def action_prompt(decision_text: str) -> str:
            return (
                build_action(decision_text)
                + f"\nFor the send_message tool, target must be the ID "
                f"{msg.sender!r} (display name: {sender_name})."
            )
        fallback = ToolCall(
            id="fallback-chat-reply",
            name="send_message",
            arguments={
                "target": msg.sender,
                "content": self.persona.DEFAULT_REPLY,
            },
        )
        decision, call, result = self._tool_two_step(
            trigger="chat_reply",
            stance=stance,
            decision_prompt=decision_prompt,
            action_prompt=action_prompt,
            registry=registry,
            fallback_call=fallback,
        )
        if result.is_error or call.name == "stay_silent":
            return
        reply = str(result.output.get("content", ""))
        self._append_context("user", "(action turn — see decision above)")
        self._append_context("assistant", reply)

    def _handle_event(self, event: GameEvent) -> None:
        """React to a GameEvent.  Pure-state events keep round_state honest;
        public-info events also fire a two-step LLM outbound."""
        # ── Round / phase lifecycle ──
        if event.event_type == "round_end":
            payload = event.payload or {}
            results = payload.get("results", {}) or {}
            verified = self.memory.verify_round(
                round=payload.get("round", 0),
                game_type=payload.get("game_type", ""),
                results=results,
            )
            self.memory.add_episode(
                round=payload.get("round", 0),
                game_type=payload.get("game_type", ""),
                results=results,
            )
            for claim in verified:
                self.state.record_trace(
                    "memory_claim_verified",
                    actor_id=self.player_id,
                    payload={
                        "memory_id": claim.memory_id,
                        "source": claim.source,
                        "truth": claim.truth,
                        "claimed_value": claim.value,
                        "actual_value": claim.actual_value,
                        "trust": self.memory.trust_for(claim.source),
                    },
                )
            self._round_history.append(payload)
            if len(self._round_history) > 10:
                self._round_history = self._round_history[-10:]
            disproved = [
                self._player_label(claim.source)
                for claim in verified
                if claim.truth is False
            ]
            evidence_summary = self._summarise_round_end(payload)
            if disproved:
                evidence_summary += (
                    " Verified false claims from: "
                    + ", ".join(sorted(set(disproved)))
                    + "."
                )
            self._round_stances.clear()
            self._maybe_outbound_for_event(
                trigger="round_evidence_revealed",
                summary=evidence_summary,
            )
            self._pending_decisions.clear()
            self._round_state = {}
            self.memory.clear_working()
            return

        if event.event_type == "phase_change":
            payload = event.payload or {}
            phase = payload.get("phase")
            if phase == "playing":
                self.memory.clear_working()
                self.memory.set_working(
                    round=payload.get("round"),
                    game_type=payload.get("game_type"),
                )
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
                    # Exact scores are hidden; their derived selection order
                    # is the intentional public ranking signal.
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
                self._maybe_outbound_for_event(
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
            payload = event.payload or {}
            picker = payload.get("picker")
            prev = self._round_state.get("current_picker_id")
            if prev and prev != picker:
                self._round_state.setdefault("picked_so_far", []).append(prev)
            self._round_state["current_picker_id"] = picker
            self._round_state["selection_order"] = payload.get(
                "selection_order", self._round_state.get("selection_order", [])
            )
            if picker and picker != self.player_id:
                self._maybe_outbound_for_event(
                    trigger="poison_picker_change",
                    summary=(
                        f"{self._player_label(picker)} is now choosing bottle "
                        f"#{payload.get('position', '?')}."
                    ),
                )
            return

        if event.event_type == "poison_result":
            payload = event.payload or {}
            self._round_state["your_pick"] = payload.get("choice")
            self._round_state["your_pick_result"] = payload.get("is_poisoned")
            outcome = "poisoned" if payload.get("is_poisoned") else "safe"
            self._maybe_outbound_for_event(
                trigger="private_poison_result",
                summary=(
                    f"Private new evidence: your {payload.get('choice')} "
                    f"bottle was {outcome}."
                ),
            )
            return

        if event.event_type != "make_decision":
            return

        # ── make_decision → role bookkeeping + batched tick decision ──
        payload = event.payload or {}
        action = payload.get("action", "")
        self.memory.set_working(action=action, decision_payload=payload)

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

        self._pending_decisions[action] = {
            "payload": payload,
            "deadline_ts": payload.get("deadline_ts"),
            "decided": False,
            "force": action == "write_word",
            "new_msgs_since_decision": 0,
            "last_reconsider_ts": 0.0,
        }

    # ──────────────────────────────────────────────────────────────────
    # Deferred game decisions
    # ──────────────────────────────────────────────────────────────────

    def _next_due_decision(self) -> tuple[str, dict, bool] | None:
        if not self._pending_decisions:
            return None
        now = time.time()
        budget = self.state.get_send_budget(self.player_id)
        budget_empty = budget is not None and budget <= 0

        for action in list(self._pending_decisions.keys()):
            p = self._pending_decisions[action]
            deadline = p.get("deadline_ts") or 0
            time_left = deadline - now
            payload_round = p.get("payload", {}).get("round")
            current_round = self.state.get_round_info().get("current_round")

            if payload_round is not None and payload_round != current_round:
                del self._pending_decisions[action]
                self.state.record_trace(
                    "stale_decision_discarded",
                    actor_id=self.player_id,
                    payload={
                        "action": action,
                        "decision_round": payload_round,
                        "current_round": current_round,
                        "reason": "round_changed",
                    },
                )
                continue
            if deadline and time_left <= 0:
                del self._pending_decisions[action]
                self.state.record_trace(
                    "stale_decision_discarded",
                    actor_id=self.player_id,
                    payload={
                        "action": action,
                        "decision_round": payload_round,
                        "current_round": current_round,
                        "reason": "deadline_passed",
                    },
                )
                continue

            if not p["decided"]:
                decision_window = _DECIDE_TAIL_SECONDS + self.tick_seconds
                if (
                    p.get("force")
                    or budget_empty
                    or time_left <= decision_window
                ):
                    return action, p["payload"], False
                continue

            if time_left <= 1.0:
                del self._pending_decisions[action]
                continue

            enough_new_info = (
                p["new_msgs_since_decision"] >= _CHANGE_DECISION_MSG_THRESHOLD
            )
            cool = (now - p["last_reconsider_ts"]) >= _CHANGE_DECISION_COOLDOWN
            if enough_new_info and cool:
                return action, p["payload"], True
        return None

    def _finish_due_decision(
        self,
        action: str,
        *,
        succeeded: bool,
        reconsider: bool,
    ) -> None:
        pending = self._pending_decisions.get(action)
        if pending is None or not succeeded:
            return
        pending["decided"] = True
        pending["new_msgs_since_decision"] = 0
        pending["last_reconsider_ts"] = time.time()
        pending["force"] = False
        if action in {"write_word", "choose_bottle"}:
            del self._pending_decisions[action]

    def _resolve_pending_decisions(self) -> None:
        """Direct single-action driver retained for tests and diagnostics."""
        due = self._next_due_decision()
        if due is None:
            return
        action, payload, reconsider = due
        self._dispatch_action(action, payload, reconsider=reconsider)
        self._finish_due_decision(
            action,
            succeeded=self.state.get_choice(self.player_id) is not None,
            reconsider=reconsider,
        )

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
        elif action == "write_word":
            self._decide_write_word(payload)

    def _choose_tick_chat(
        self,
        chats: list[ChatMessage],
    ) -> dict[str, Any] | None:
        """Choose at most one chat opportunity after ingesting the full batch."""
        if chats:
            if not self._has_budget():
                if any(msg.sender == _HUMAN_ID for msg in chats):
                    self.state.send_system_message(
                        sender=self.player_id,
                        recipient=_HUMAN_ID,
                        text=(
                            f"{self.persona.NAME} has used up this round's "
                            "send budget; it will refresh next round."
                        ),
                    )
                return None
            human_chats = [msg for msg in chats if msg.sender == _HUMAN_ID]
            msg = (human_chats or chats)[-1]
            stance = self._pick_stance(
                trigger="chat_reply",
                counterpart=msg.sender,
            )
            if stance == "silent":
                self._log_silent_skip("chat_reply")
                return None
            return {
                "trigger": "chat_reply",
                "summary": (
                    f"Reply to {self._player_label(msg.sender)}'s latest DM: "
                    f"{msg.text}"
                ),
                "targets": [msg.sender],
                "stance": stance,
            }

        if self._tick_event_opportunities and self._has_budget():
            opportunity = max(
                self._tick_event_opportunities,
                key=lambda item: item["priority"],
            )
            stance = self._pick_stance(trigger=opportunity["trigger"])
            if stance == "silent":
                self._log_silent_skip(opportunity["trigger"])
                return None
            return {
                **opportunity,
                "targets": [
                    pid
                    for pid in self.state.player_ids
                    if pid != self.player_id
                ],
                "stance": stance,
            }

        if (
            self._can_initiate_proactive()
            and random.random() < self.persona.CHAT_INITIATIVE_PROB
        ):
            stance = self._pick_stance(trigger="proactive_chat")
            if stance == "silent":
                self._log_silent_skip("proactive_chat")
                return None
            return {
                "trigger": "proactive_chat",
                "summary": "Periodic opportunity to exchange useful information.",
                "targets": [
                    pid
                    for pid in self.state.player_ids
                    if pid != self.player_id
                ],
                "stance": stance,
            }
        return None

    def _send_tick_chat(
        self,
        *,
        target: str,
        content: str,
        allowed_targets: list[str],
        decision_summary: str,
    ) -> dict:
        from game.game_tools import MAX_CHAT_CHARACTERS
        from game.security import sanitize_model_output

        if target not in allowed_targets:
            raise ValueError("Target is not allowed in this tick")
        message = sanitize_model_output(
            self._sanitize_chat_text(content),
            max_length=MAX_CHAT_CHARACTERS,
        )
        if not message:
            raise ValueError("Message cannot be empty")
        self.state.send_message(
            self.player_id,
            target,
            message,
            thinking=decision_summary or None,
        )
        return {"sent": True, "target": target, "content": message}

    def _act_on_tick(
        self,
        *,
        observations: list[str],
        due_game: tuple[str, dict, bool] | None,
        chat: dict[str, Any] | None,
    ) -> bool:
        """Make one model call, then independently execute game and chat."""
        from game.game_tools import send_message_tool, tick_action_tool
        from game.llm_client import LLMRequest, ToolCall
        from game.tool_runtime import ToolRegistry
        from prompts.templates import build_tick_action_prompt

        game_action = None
        game_payload = None
        game_instruction = None
        game_definition = None
        game_handler = None
        game_fallback = None
        if due_game is not None:
            game_action, game_payload, reconsider = due_game
            if reconsider:
                game_payload = {
                    **game_payload,
                    "previous_choice": self.state.get_choice(self.player_id),
                    "reconsider": True,
                }
            (
                game_instruction,
                game_definition,
                game_handler,
                game_fallback,
            ) = self._game_tool_spec(game_action, game_payload)

        chat_targets = list(chat["targets"]) if chat else []
        round_info = self.state.get_round_info()
        prompt = build_tick_action_prompt(
            observations=observations,
            game_action_name=(
                game_definition.name if game_definition is not None else None
            ),
            game_payload=game_payload,
            game_instruction=game_instruction,
            chat_trigger=chat["trigger"] if chat else None,
            chat_targets=chat_targets,
            chat_summary=chat["summary"] if chat else None,
            score=self.state.get_score(self.player_id),
            send_budget=self.state.get_send_budget(self.player_id),
            scores=self._visible_scores(),
            current_round=round_info.get("current_round"),
            total_rounds=round_info.get("total_rounds"),
            round_state_text=self._round_state_text(),
            round_history=self._round_history,
            recent_chat_summary=self._recent_chat_summary(),
            game_type=round_info.get("game_type"),
            stance=chat["stance"] if chat else None,
        )
        self.state.record_trace(
            "agent_tick",
            actor_id=self.player_id,
            payload={
                "observation_count": len(observations),
                "game_action": game_action,
                "chat_trigger": chat["trigger"] if chat else None,
            },
        )
        response = self.llm_client.complete(LLMRequest(
            system_prompt=self._system_prompt(),
            messages=[{"role": "user", "content": prompt}],
            default_reply="",
            max_tokens=_ACTION_CHAT_MAX_TOKENS,
            tools=(tick_action_tool(game_definition, chat_targets),),
            tool_choice="required",
            agent_id=self.player_id,
            persona=self.persona.NAME,
            trigger="agent_tick",
            phase="tick_action",
            stance=chat["stance"] if chat else "cooperate",
            metadata={
                "observation_count": len(observations),
                "game_action": game_action,
                "chat_trigger": chat["trigger"] if chat else None,
            },
        ))
        action_id = uuid.uuid4().hex
        self._trace_llm_response(
            response,
            trigger="agent_tick",
            phase="tick_action",
            action_id=action_id,
        )
        arguments: dict[str, Any] = {}
        if (
            len(response.tool_calls) == 1
            and response.tool_calls[0].name == "act_in_tick"
        ):
            arguments = dict(response.tool_calls[0].arguments)
        else:
            self.state.record_trace(
                "tick_action_invalid",
                actor_id=self.player_id,
                payload={"reason": "Expected exactly one act_in_tick call"},
            )

        raw_summary = arguments.get("decision_summary", "")
        decision_summary = (
            self._sanitize_chat_text(raw_summary)[:120]
            if isinstance(raw_summary, str)
            else ""
        )
        game_succeeded = game_definition is None
        if game_definition is not None:
            game_registry = ToolRegistry()
            game_registry.register(game_definition, game_handler)
            nested = arguments.get("game_action")
            if isinstance(nested, dict):
                call = ToolCall(
                    f"{action_id}-game",
                    str(nested.get("name", "")),
                    nested.get("arguments", {})
                    if isinstance(nested.get("arguments"), dict)
                    else {},
                )
                result = game_registry.execute(call)
                self._log_tool_execution(
                    call,
                    result,
                    attempt=1,
                    fallback=False,
                    action_id=action_id,
                )
                game_succeeded = not result.is_error
            if not game_succeeded:
                result = game_registry.execute(game_fallback)
                self._log_tool_execution(
                    game_fallback,
                    result,
                    attempt=2,
                    fallback=True,
                    action_id=action_id,
                )
                game_succeeded = not result.is_error

        nested_chat = arguments.get("chat_action")
        if chat and isinstance(nested_chat, dict):
            chat_registry = ToolRegistry()
            chat_registry.register(
                send_message_tool(chat_targets),
                lambda target, content, decision_summary="": (
                    self._send_tick_chat(
                        target=target,
                        content=content,
                        allowed_targets=chat_targets,
                        decision_summary=decision_summary,
                    )
                ),
            )
            call = ToolCall(
                f"{action_id}-chat",
                "send_message",
                {
                    "target": nested_chat.get("target"),
                    "content": nested_chat.get("content"),
                    "decision_summary": decision_summary,
                },
            )
            result = chat_registry.execute(call)
            self._log_tool_execution(
                call,
                result,
                attempt=1,
                fallback=False,
                action_id=action_id,
            )
        elif chat and chat["trigger"] == "chat_reply" and not arguments:
            try:
                self._send_tick_chat(
                    target=chat_targets[0],
                    content=self.persona.DEFAULT_REPLY,
                    allowed_targets=chat_targets,
                    decision_summary="",
                )
            except (RuntimeError, ValueError):
                pass

        self._append_context("user", prompt)
        self._append_context("assistant", decision_summary)
        return game_succeeded

    # ──────────────────────────────────────────────────────────────────
    # Outbound DM trigger (proactive + public-info-driven)
    # ──────────────────────────────────────────────────────────────────

    def _maybe_outbound_for_event(self, trigger: str, summary: str) -> None:
        """Try a two-step outbound after meaningful public/private evidence."""
        if self._collecting_tick:
            priority = {
                "round_evidence_revealed": 80,
                "private_poison_result": 70,
                "poison_picker_change": 60,
                "public_info_phase_playing": 50,
            }.get(trigger, 40)
            self._tick_event_opportunities.append({
                "trigger": trigger,
                "summary": summary,
                "priority": priority,
            })
            return
        if not self._has_budget():
            return
        stance = self._pick_stance(trigger=trigger)
        if stance == "silent":
            self._log_silent_skip(trigger=trigger)
            return
        self._outbound_two_step(trigger=trigger, stance=stance, summary=summary)

    def _proactive_chat(self) -> None:
        """Legacy tick-based dice-roll proactive DM (lowered probability)."""
        stance = self._pick_stance(trigger="proactive_chat")
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
            scores=self._visible_scores(),
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

        from game.game_tools import (
            MAX_CHAT_CHARACTERS,
            STAY_SILENT,
            send_message_tool,
        )
        from game.llm_client import ToolCall
        from game.tool_runtime import ToolRegistry

        registry = ToolRegistry()

        def send_outbound(
            target: str,
            content: str,
            decision_summary: str = "",
        ) -> dict:
            if target == self.player_id or target not in others:
                raise ValueError("Target is not allowed")
            from game.security import sanitize_model_output

            message = sanitize_model_output(
                self._sanitize_chat_text(content),
                max_length=MAX_CHAT_CHARACTERS,
            )
            if not message:
                raise ValueError("Message cannot be empty")
            self.state.send_message(
                self.player_id,
                target,
                message,
                thinking=decision_summary or None,
            )
            return {"sent": True, "target": target, "content": message}

        registry.register(send_message_tool(others), send_outbound)
        registry.register(
            STAY_SILENT,
            lambda reason="": {"sent": False, "reason": reason},
        )
        target_mapping = ", ".join(
            f"{pid}={self._player_label(pid)}" for pid in others
        )

        def action_prompt(decision_text: str) -> str:
            return (
                build_action(decision_text)
                + "\nFor send_message.target use one of these IDs exactly: "
                + target_mapping
            )

        decision, call, result = self._tool_two_step(
            trigger=trigger,
            stance=stance,
            decision_prompt=decision_prompt,
            action_prompt=action_prompt,
            registry=registry,
            fallback_call=ToolCall(
                "fallback-silent", "stay_silent", {"reason": "offline fallback"}
            ),
        )
        if result.is_error or call.name == "stay_silent":
            return
        message_text = str(result.output.get("content", ""))

        self._append_context("user", "(action turn — see decision above)")
        self._append_context("assistant", message_text)

    # ──────────────────────────────────────────────────────────────────
    # Game decisions — write/guess/bottle (Phase A + B with stance)
    # ──────────────────────────────────────────────────────────────────

    def _decide_write_word(self, payload: dict) -> None:
        spec = self._game_tool_spec("write_word", payload)
        return self._make_game_tool_decision(payload, *spec)

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

    def _game_tool_spec(self, action: str, payload: dict):
        """Return instruction, tool definition, handler, and safe fallback."""
        from game.game_tools import (
            GUESS_WORD,
            SUBMIT_WORD,
            attribute_words_tool,
            choose_bottle_tool,
        )
        from game.llm_client import ToolCall

        if action == "write_word":
            game_type = payload.get("game_type", "")
            avoid_pool = self._past_words_pool()
            avoid_clause = (
                " Avoid repeating these played words: "
                f"{avoid_pool}." if avoid_pool else ""
            )
            if game_type == "guess_the_word":
                instruction = (
                    "As writer, choose a single English word at the sweet "
                    "spot: guessable for some but not all. Avoid clichés and "
                    "ultra-obscure words; good examples include mirror, "
                    f"thunder, whisper, shadow, and compass.{avoid_clause}"
                )
            else:
                instruction = (
                    "Choose one English word that reveals something about "
                    "your persona but could plausibly come from anyone. Avoid "
                    f"obvious defaults such as apple, dog, sky, and love."
                    f"{avoid_clause}"
                )

            def submit_word(word: str, decision_summary: str = "") -> dict:
                self._require_decision_window(payload)
                normalized = word.strip().lower()
                self.state.submit_choice(self.player_id, normalized)
                self._round_state["my_submission"] = normalized
                return {"submitted": True, "word": normalized}

            fallback = random.choice(
                ["mirror", "thunder", "whisper", "shadow", "compass", "echo"]
            )
            return (
                instruction,
                SUBMIT_WORD,
                submit_word,
                ToolCall("fallback-submit-word", "submit_word", {"word": fallback}),
            )

        if action == "guess_word":
            instruction = (
                "Guess the writer's single English word from current evidence."
            )

            def guess_word(word: str, decision_summary: str = "") -> dict:
                self._require_decision_window(payload)
                normalized = word.strip().lower()
                self.state.submit_choice(self.player_id, normalized)
                self._round_state["my_submission"] = normalized
                return {"submitted": True, "word": normalized}

            fallback = random.choice(["apple", "sky", "moon", "star"])
            return (
                instruction,
                GUESS_WORD,
                guess_word,
                ToolCall("fallback-guess-word", "guess_word", {"word": fallback}),
            )

        if action == "guess_authors":
            words = payload.get("words", [])
            candidates = payload.get("candidate_authors", [])
            instruction = (
                f"Attribute words {words} to candidate author IDs {candidates} "
                "in the displayed order, using each author once."
            )

            def attribute_words(
                authors: list[str],
                decision_summary: str = "",
            ) -> dict:
                self._require_decision_window(payload)
                if len(authors) != len(words):
                    raise ValueError("One author is required for every word")
                if len(set(authors)) != len(authors):
                    raise ValueError("Each candidate author must be used once")
                raw = ",".join(authors)
                self.state.submit_choice(self.player_id, raw)
                self._round_state["my_submission"] = raw
                return {"submitted": True, "authors": authors}

            return (
                instruction,
                attribute_words_tool(candidates, len(words)),
                attribute_words,
                ToolCall(
                    "fallback-attribute-words",
                    "attribute_words",
                    {"authors": list(candidates)},
                ),
            )

        if action == "choose_bottle":
            available = payload.get(
                "available_bottles",
                ["Red", "Blue", "Green", "Yellow"],
            )
            instruction = f"Choose one available bottle from {available}."

            def choose_bottle(
                bottle: str,
                decision_summary: str = "",
            ) -> dict:
                self._require_decision_window(payload)
                self.state.submit_choice(self.player_id, bottle)
                self._round_state["your_pick"] = bottle
                return {"submitted": True, "bottle": bottle}

            return (
                instruction,
                choose_bottle_tool(available),
                choose_bottle,
                ToolCall(
                    "fallback-choose-bottle",
                    "choose_bottle",
                    {"bottle": random.choice(available)},
                ),
            )
        raise ValueError(f"Unsupported game action: {action!r}")

    def _decide_guess_word(self, payload: dict) -> None:
        spec = self._game_tool_spec("guess_word", payload)
        return self._make_game_tool_decision(payload, *spec)

    def _decide_guess_authors(self, payload: dict) -> None:
        spec = self._game_tool_spec("guess_authors", payload)
        return self._make_game_tool_decision(payload, *spec)

    def _decide_choose_bottle(self, payload: dict) -> None:
        spec = self._game_tool_spec("choose_bottle", payload)
        return self._make_game_tool_decision(payload, *spec)

    def _make_game_tool_decision(
        self,
        payload: dict,
        instruction: str,
        definition,
        handler,
        fallback_call,
    ):
        """Run a strategic decision followed by a validated game tool."""
        from game.tool_runtime import ToolRegistry
        from game.llm_client import ToolResult
        from prompts.templates import (
            build_game_action_prompt,
            build_game_decision_prompt,
        )

        try:
            self._require_decision_window(payload)
        except ValueError as exc:
            round_info = self.state.get_round_info()
            self.state.record_trace(
                "stale_decision_discarded",
                actor_id=self.player_id,
                payload={
                    "action": fallback_call.name,
                    "decision_round": payload.get("round"),
                    "current_round": round_info.get("current_round"),
                    "reason": str(exc),
                },
            )
            return fallback_call, ToolResult(
                fallback_call.id,
                {"error": str(exc)},
                True,
            )

        situation = {**payload, "instruction": instruction}
        stance = self._pick_stance(trigger="game_decision")

        round_info = self.state.get_round_info()
        game_type = situation.get("game_type", "unknown_game")
        decision_prompt = build_game_decision_prompt(
            game_type=game_type,
            situation=situation,
            score=self.state.get_score(self.player_id),
            send_budget=self.state.get_send_budget(self.player_id),
            scores=self._visible_scores(),
            current_round=round_info.get("current_round"),
            total_rounds=round_info.get("total_rounds"),
            round_state_text=self._round_state_text(),
            round_history=self._round_history,
            recent_chat_summary=self._recent_chat_summary(),
            stance=stance,
        )

        def action_prompt(decision_text: str) -> str:
            self._require_decision_window(payload)
            return build_game_action_prompt(
                game_type=game_type,
                situation=situation,
                decision_text=decision_text,
                score=self.state.get_score(self.player_id),
                send_budget=self.state.get_send_budget(self.player_id),
                round_state_text=self._round_state_text(),
                stance=stance,
            )

        registry = ToolRegistry()
        registry.register(definition, handler)
        decision, call, result = self._tool_two_step(
            trigger="game_decision",
            stance=stance,
            decision_prompt=decision_prompt,
            action_prompt=action_prompt,
            registry=registry,
            fallback_call=fallback_call,
            action_max_tokens=_ACTION_GAME_MAX_TOKENS,
        )
        self._append_context("user", "(game tool action)")
        self._append_context("assistant", decision[:300])
        return call, result

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

    def _visible_scores(self) -> dict[str, int | str]:
        """Return only score information this agent is allowed to know."""
        out: dict[str, int | str] = {}
        for pid in self.state.player_ids:
            try:
                name = self.state.get_display_name(pid)
            except Exception:
                name = pid
            out[name] = (
                self.state.get_score(pid)
                if pid == self.player_id
                else "hidden"
            )
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
            return (
                "(round just starting — no role info yet)\n\n"
                "RELEVANT MEMORY:\n"
                + self.memory.prompt_summary()
            )

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

        lines.extend([
            "",
            "RELEVANT MEMORY:",
            self.memory.prompt_summary(gt),
        ])
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
