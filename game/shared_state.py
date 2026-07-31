"""Thread-safe shared state for the multi-agent game."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from game.trace import TraceRecorder


@dataclass
class ChatMessage:
    sender: str
    recipient: str
    text: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class GameEvent:
    event_type: str
    payload: Any
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SharedState:
    """
    Central state shared across all Streamlit and background AI threads.

    All mutations are protected by a single RLock so concurrent score
    updates, message sends, and broadcasts never race.

    Parameters
    ----------
    player_ids:
        List of player identifiers to register at construction time,
        e.g. ["ai_0", "ai_1", "ai_2"].
    initial_score:
        Starting score for every player (default 0).
    initial_send_budget:
        Number of messages each player may send before the budget hits 0
        (default 10).  Pass None for unlimited.
    """

    def __init__(
        self,
        player_ids: List[str],
        initial_score: int = 0,
        initial_send_budget: int | None = 10,
        *,
        human_player_id: Optional[str] = None,
        trace_recorder: Optional[TraceRecorder] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._scores: Dict[str, int] = {pid: initial_score for pid in player_ids}
        self._initial_send_budget: int | None = initial_send_budget
        self._send_budgets: Dict[str, int | None] = {
            pid: initial_send_budget for pid in player_ids
        }
        # Each player gets their own inbox queue.
        self._inboxes: Dict[str, queue.Queue] = {
            pid: queue.Queue() for pid in player_ids
        }
        # Game phase and round metadata (readable by UI and AI agents).
        self._phase: str = "setup"
        self._current_round: int = 0
        self._total_rounds: int = 0
        self._current_game_type: str = ""
        # Per-player choice submission for mini-games.
        self._choices: Dict[str, Any] = {}
        self._choice_events: Dict[str, threading.Event] = {}
        # Human-visible chat history — accumulates entries of two kinds:
        #   {"type": "chat",    "sender", "recipient", "text", "timestamp"}
        #   {"type": "whisper", "sender", "recipient",          "timestamp"}
        # Append-only; never drained.  UI reads a snapshot via get_chat_history().
        self._human_player_id: Optional[str] = human_player_id
        self._chat_history: List[Dict[str, Any]] = []
        # Display names used by UI/prompt text (defaults to IDs).
        self._display_names: Dict[str, str] = {pid: pid for pid in player_ids}
        # Event signalled by the human clicking "Next Round" on the reveal
        # screen.  The engine waits on this between rounds instead of a
        # fixed-duration sleep.
        self._continue_event: threading.Event = threading.Event()
        self._trace_recorder = trace_recorder

    @property
    def game_id(self) -> str | None:
        return (
            self._trace_recorder.game_id
            if self._trace_recorder is not None
            else None
        )

    def record_trace(
        self,
        event_type: str,
        *,
        actor_id: str | None = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._trace_recorder is None:
            return
        with self._lock:
            current_round = self._current_round
            game_type = self._current_game_type
        self._trace_recorder.record(
            event_type,
            actor_id=actor_id,
            round=current_round or None,
            game_type=game_type or None,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_player(self, player_id: str) -> None:
        if player_id not in self._inboxes:
            raise KeyError(f"Unknown player: {player_id!r}")

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send_message(
        self,
        sender: str,
        recipient: str,
        text: str,
        thinking: Optional[str] = None,
    ) -> None:
        """
        Deliver a ChatMessage to *recipient*'s inbox and deduct one unit
        from *sender*'s send_budget.

        Parameters
        ----------
        thinking:
            Optional private reasoning attached by the sender (AI agents).
            Stored in chat history but NEVER delivered as part of the
            ChatMessage payload — the recipient sees only *text*.  Used by
            the end-of-game timeline to reveal what each AI was thinking.

        Raises
        ------
        KeyError
            If sender or recipient is not a registered player.
        RuntimeError
            If sender has exhausted their send_budget.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message text must be non-empty")
        if len(text) > 1000:
            raise ValueError("Message text cannot exceed 1000 characters")
        with self._lock:
            self._require_player(sender)
            self._require_player(recipient)

            budget = self._send_budgets[sender]
            if budget is not None:
                if budget <= 0:
                    raise RuntimeError(
                        f"Player {sender!r} has no remaining send_budget."
                    )
                self._send_budgets[sender] = budget - 1

            msg = ChatMessage(sender=sender, recipient=recipient, text=text)
            self._inboxes[recipient].put(msg)
            self._record_human_visible_message(msg, thinking=thinking)
            remaining_budget = self._send_budgets[sender]
        self.record_trace(
            "chat_message",
            actor_id=sender,
            payload={
                "recipient": recipient,
                "text": text,
                "thinking": thinking,
                "remaining_budget": remaining_budget,
            },
        )

    def send_system_message(self, sender: str, recipient: str, text: str) -> None:
        """Deliver a message without consuming sender budget.

        Use for non-gameplay notifications (e.g. "AI has exhausted this round's
        send budget"), where we still want the human to receive feedback.
        """
        with self._lock:
            self._require_player(sender)
            self._require_player(recipient)
            msg = ChatMessage(sender=sender, recipient=recipient, text=text)
            self._inboxes[recipient].put(msg)
            self._record_human_visible_message(msg)

    def _record_human_visible_message(
        self,
        msg: ChatMessage,
        thinking: Optional[str] = None,
    ) -> None:
        """Append chat/whisper entries for the human-facing chat panel.

        ``thinking`` (if provided) is stored alongside the entry so the
        end-of-game timeline can reveal it.  The live chat fragment ignores
        this field.
        """
        if not self._human_player_id:
            return
        hid = self._human_player_id

        def _entry(kind: str) -> Dict[str, Any]:
            e: Dict[str, Any] = {
                "type": kind,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "text": msg.text,
                "timestamp": msg.timestamp,
                "round": self._current_round,
            }
            if thinking:
                e["thinking"] = thinking
            return e

        if msg.sender == hid or msg.recipient == hid:
            self._chat_history.append(_entry("chat"))
        elif msg.sender != hid and msg.recipient != hid:
            # AI-to-AI: text is recorded but hidden behind "whisper" type so
            # the live chat panel only shows a notification.  The end-of-game
            # timeline renders the stored text in full.
            self._chat_history.append(_entry("whisper"))

    def get_my_messages(self, player_id: str) -> List[Union[ChatMessage, GameEvent]]:
        """
        Drain and return all items currently queued in *player_id*'s inbox.
        Does NOT deduct send_budget.
        """
        with self._lock:
            self._require_player(player_id)
            inbox = self._inboxes[player_id]

        messages: List[Union[ChatMessage, GameEvent]] = []
        # Drain outside the main lock — queue.Queue is thread-safe internally.
        while True:
            try:
                messages.append(inbox.get_nowait())
            except queue.Empty:
                break
        return messages

    def broadcast_event(self, event: GameEvent) -> None:
        """Put *event* into every registered player's inbox."""
        with self._lock:
            inboxes = list(self._inboxes.values())
        for inbox in inboxes:
            inbox.put(event)
        self.record_trace(
            "game_event",
            payload={
                "delivery": "broadcast",
                "event_type": event.event_type,
                "event_payload": event.payload,
            },
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_score(self, player_id: str) -> int:
        with self._lock:
            self._require_player(player_id)
            return self._scores[player_id]

    def update_score(self, player_id: str, delta: int) -> int:
        """
        Thread-safely add *delta* to *player_id*'s score.

        Returns the new score.
        """
        with self._lock:
            self._require_player(player_id)
            self._scores[player_id] += delta
            score = self._scores[player_id]
        self.record_trace(
            "score_updated",
            actor_id=player_id,
            payload={"delta": delta, "score": score},
        )
        return score

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def get_send_budget(self, player_id: str) -> int | None:
        with self._lock:
            self._require_player(player_id)
            return self._send_budgets[player_id]

    # ------------------------------------------------------------------
    # Game event routing (targeted, not broadcast)
    # ------------------------------------------------------------------

    def send_event_to_player(self, player_id: str, event: "GameEvent") -> None:
        """Place *event* into *player_id*'s inbox without touching send budgets."""
        with self._lock:
            self._require_player(player_id)
            inbox = self._inboxes[player_id]
        inbox.put(event)
        self.record_trace(
            "game_event",
            actor_id=player_id,
            payload={
                "delivery": "targeted",
                "event_type": event.event_type,
                "event_payload": event.payload,
            },
        )

    # ------------------------------------------------------------------
    # Phase and round metadata
    # ------------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase
        self.record_trace("phase_changed", payload={"phase": phase})

    def get_phase(self) -> str:
        with self._lock:
            return self._phase

    def set_round_info(self, current: int, total: int, game_type: str) -> None:
        with self._lock:
            self._current_round = current
            self._total_rounds = total
            self._current_game_type = game_type
        self.record_trace(
            "round_configured",
            payload={"current": current, "total": total, "game_type": game_type},
        )

    def get_round_info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current_round": self._current_round,
                "total_rounds": self._total_rounds,
                "game_type": self._current_game_type,
            }

    # ------------------------------------------------------------------
    # Per-round send budget reset
    # ------------------------------------------------------------------

    def reset_send_budgets(self) -> None:
        """Restore every player's send budget to the value set at construction."""
        with self._lock:
            for pid in self._send_budgets:
                self._send_budgets[pid] = self._initial_send_budget
        self.record_trace(
            "send_budgets_reset",
            payload={"budget": self._initial_send_budget},
        )

    # ------------------------------------------------------------------
    # Mini-game choice submission
    # ------------------------------------------------------------------

    def reset_choices(self, player_ids: Optional[List[str]] = None) -> None:
        """Clear pending choices and create fresh Events for *player_ids*.

        Call this before soliciting a new set of decisions so stale events
        cannot spuriously unblock a waiting game-engine thread.

        Parameters
        ----------
        player_ids:
            Subset of players to reset.  Defaults to all registered players.
        """
        with self._lock:
            ids = player_ids if player_ids is not None else list(self._inboxes.keys())
            for pid in ids:
                self._require_player(pid)
                self._choices[pid] = None
                self._choice_events[pid] = threading.Event()

    def submit_choice(self, player_id: str, choice: Any) -> None:
        """Record *player_id*'s decision and signal the waiting game-engine.

        Thread-safe: AI agents and the human-IO thread call this; the game
        engine thread is unblocked via the corresponding threading.Event.
        """
        with self._lock:
            self._require_player(player_id)
            self._choices[player_id] = choice
            event = self._choice_events.get(player_id)
        # Set the event outside the lock to avoid holding it while notifying
        # waiting threads (which may themselves try to acquire the lock).
        if event is not None:
            event.set()
        self.record_trace(
            "choice_submitted",
            actor_id=player_id,
            payload={"choice": choice},
        )

    def get_choice(self, player_id: str) -> Any:
        with self._lock:
            self._require_player(player_id)
            return self._choices.get(player_id)

    def get_choice_event(self, player_id: str) -> threading.Event:
        """Return the current threading.Event for *player_id*'s pending choice.

        Creates a new (unset) event if none exists yet.  Always call
        :meth:`reset_choices` before a new decision phase so a fresh event
        is returned rather than a previously-set one.
        """
        with self._lock:
            self._require_player(player_id)
            if player_id not in self._choice_events:
                self._choice_events[player_id] = threading.Event()
            return self._choice_events[player_id]

    # ------------------------------------------------------------------
    # Continue-to-next-round signalling
    # ------------------------------------------------------------------

    def reset_continue(self) -> None:
        """Clear the continue event before the engine starts waiting."""
        with self._lock:
            self._continue_event = threading.Event()

    def signal_continue(self) -> None:
        """Called by the UI when the player clicks Next Round."""
        with self._lock:
            event = self._continue_event
        event.set()

    def wait_for_continue(self, timeout: float | None = None) -> bool:
        """Block until :meth:`signal_continue` is called or *timeout* expires.

        Returns ``True`` if the event was set, ``False`` on timeout.
        """
        with self._lock:
            event = self._continue_event
        return event.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Human-visible chat history
    # ------------------------------------------------------------------

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Return a snapshot of the human-visible chat history.

        Entries are dicts with ``type`` equal to ``"chat"`` or ``"whisper"``.
        The list is append-only so callers can track how many entries they
        have already rendered by remembering the previous length.
        """
        with self._lock:
            return list(self._chat_history)

    # ------------------------------------------------------------------
    # Player display names
    # ------------------------------------------------------------------

    def set_display_name(self, player_id: str, display_name: str) -> None:
        with self._lock:
            self._require_player(player_id)
            self._display_names[player_id] = display_name

    def get_display_name(self, player_id: str) -> str:
        with self._lock:
            self._require_player(player_id)
            return self._display_names.get(player_id, player_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def player_ids(self) -> List[str]:
        with self._lock:
            return list(self._inboxes.keys())
