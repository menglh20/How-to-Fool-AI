"""How to Fool AI — main Streamlit entry point.

Screens
-------
setup     → player enters name + round count, then clicks Start Game
game      → auto-refreshing game area (left) + chat panel (right)
game_over → final scores revealed, Play Again button

Thread model
------------
* GameEngine daemon thread — drives rounds and mini-game phases.
* Three AIAgent daemon threads — autonomous AI players.
* Streamlit main thread — this file.

All threads share a single SharedState instance stored in st.session_state.
Fragments refresh independently:
  _game_area_fragment  run_every=3 s — phase, round info, pending decisions
  _chat_fragment       run_every=2 s — chat history, send form, whispers
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

import streamlit as st

from game.ai_agent import AIAgent
from game.game_engine import GameEngine
from game.llm_client import MODEL_CHOICES, set_active_model
from game.shared_state import ChatMessage, GameEvent, SharedState
import prompts.bai as bai
import prompts.fox as fox
import prompts.ironface as ironface

# ── constants ─────────────────────────────────────────────────────────────────

HUMAN_ID = "human"
ALL_PLAYERS = [HUMAN_ID, "ai_0", "ai_1", "ai_2"]

# Display metadata for each AI opponent.
AI_META: Dict[str, Dict[str, str]] = {
    "ai_0": {
        "name": "Bunny",
        "emoji": "🐰",
        "trait": "Naive & Kind",
        "desc": "Trusting and chatty. Believes almost everything you tell them — easy to manipulate, but their enthusiasm makes them a wildcard.",
        "persona": bai,
    },
    "ai_1": {
        "name": "Fox",
        "emoji": "🦊",
        "trait": "Cunning & Strategic",
        "desc": "Suspicious by default. Sets traps, cross-validates your claims, and strikes when you least expect it.",
        "persona": fox,
    },
    "ai_2": {
        "name": "Stoneface",
        "emoji": "🗿",
        "trait": "Cold & Rational",
        "desc": "Speaks only when necessary. Trusts almost no one. Their rare messages carry decisive weight.",
        "persona": ironface,
    },
}

GAME_TYPE_LABELS: Dict[str, str] = {
    "guess_the_word": "🔮 Guess the Word",
    "who_wrote_it": "🎭 Who Wrote It",
    "poison_bottle": "☠️ Poison Bottle",
}

PHASE_LABELS: Dict[str, str] = {
    "setup": "Setting up…",
    "playing": "▶ Playing",
    "reveal": "🔍 Reveal",
    "game_over": "🏁 Game Over",
}

ROUND_OPTIONS = [3, 5, 7, 10]


# ── session-state initialisation ─────────────────────────────────────────────

def _init_session() -> None:
    ss = st.session_state
    ss.setdefault("page", "setup")
    ss.setdefault("player_name", "Eric")
    ss.setdefault("pending_decision", None)   # dict payload from make_decision event
    ss.setdefault("last_round_results", None) # dict payload from round_end event
    ss.setdefault("round_history", [])        # per-round deltas + cumulative scores
    ss.setdefault("chat_rendered_up_to", 0)   # index into chat history already shown
    ss.setdefault("pending_poison_result", None)   # immediate result for poison_bottle
    ss.setdefault("current_poison_picker", None)   # who is currently choosing a bottle
    ss.setdefault("active_chat_ai", "ai_0")        # which AI's window is open
    ss.setdefault("chat_last_seen", {})            # {ai_id: count of msgs seen}


# ── thread lifecycle ──────────────────────────────────────────────────────────

def _start_threads(state: SharedState, total_rounds: int) -> None:
    """Start GameEngine + three AIAgent daemon threads (idempotent)."""
    ss = st.session_state

    # Agents — one per AI player.
    if "agents" not in ss:
        agents: List[AIAgent] = []
        for pid, meta in AI_META.items():
            agent = AIAgent(
                player_id=pid,
                persona=meta["persona"],
                state=state,
            )
            agent.start()
            agents.append(agent)
        ss.agents = agents

    # Engine — starts the round loop.
    if "engine" not in ss:
        engine = GameEngine(state, total_rounds=total_rounds)
        engine.start()
        ss.engine = engine


def _stop_threads() -> None:
    """Signal all daemon threads to stop cleanly."""
    ss = st.session_state
    if engine := ss.pop("engine", None):
        engine.stop()
    for agent in ss.pop("agents", []):
        agent.stop()


# ── page: Setup ───────────────────────────────────────────────────────────────

def render_setup() -> None:
    st.set_page_config(
        page_title="How to Fool AI", page_icon="🎭", layout="centered"
    )

    st.title("🎭 How to Fool AI")
    st.markdown(
        "A social-deduction game where you face three AI opponents across "
        "a series of mini-games. Scores are hidden. Private messages may be lies. "
        "Every alliance is temporary."
    )

    st.divider()

    with st.form("setup_form"):
        player_name = st.text_input(
            "Your display name",
            value=st.session_state.player_name,
            max_chars=20,
            placeholder="Enter your name…",
        )
        total_rounds = st.select_slider(
            "Number of rounds",
            options=ROUND_OPTIONS,
            value=5,
        )
        model_label = st.selectbox(
            "AI model",
            options=list(MODEL_CHOICES.keys()),
            index=0,
        )
        st.markdown("&nbsp;")
        start = st.form_submit_button("🚀 Start Game", use_container_width=True)

    if start:
        if not player_name.strip():
            st.error("Please enter a display name.")
            return
        _launch_game(
            player_name.strip(),
            int(total_rounds),
            MODEL_CHOICES[model_label],
        )

    st.divider()
    st.subheader("Your opponents")
    cols = st.columns(3)
    for col, (pid, meta) in zip(cols, AI_META.items()):
        with col:
            st.markdown(f"### {meta['emoji']} {meta['name']}")
            st.caption(f"*{meta['trait']}*")
            st.write(meta["desc"])


def _launch_game(player_name: str, total_rounds: int, model_id: str) -> None:
    """Create SharedState, start threads, transition to game screen."""
    ss = st.session_state
    # Apply the chosen model to all subsequent LLM calls (single-session app).
    set_active_model(model_id)
    ss.active_model = model_id

    # Build a fresh SharedState for this game session.
    state = SharedState(
        ALL_PLAYERS,
        initial_score=0,
        initial_send_budget=10,
        human_player_id=HUMAN_ID,
    )
    ss.state = state
    ss.player_name = player_name
    ss.total_rounds = total_rounds
    ss.pending_decision = None
    ss.last_round_results = None
    ss.round_history = []
    ss.chat_rendered_up_to = 0
    ss.pending_poison_result = None
    ss.current_poison_picker = None
    ss.active_chat_ai = "ai_0"
    ss.chat_last_seen = {}
    state.set_display_name(HUMAN_ID, player_name)
    for pid, meta in AI_META.items():
        state.set_display_name(pid, meta["name"])

    # Discard any leftover threads from a previous game.
    _stop_threads()
    _start_threads(state, total_rounds)

    ss.page = "game"
    st.rerun()


# ── page: Game ────────────────────────────────────────────────────────────────

def render_game() -> None:
    st.set_page_config(
        page_title="How to Fool AI — Game", page_icon="🎭", layout="wide"
    )

    # Strip Streamlit's chrome so our own header sits at the top of the
    # viewport, and suppress the floating "Press Enter to apply" hint that
    # overlaps text inputs.
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] { display: none; }
        [data-testid="stToolbar"]      { display: none; }
        [data-testid="stDecoration"]   { display: none; }
        .block-container               { padding-top: 0.6rem; padding-bottom: 0.6rem; }
        [data-testid="InputInstructions"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_game, col_chat = st.columns([2, 1], gap="medium")
    with col_game:
        _header_fragment()
        _game_area_fragment()
    with col_chat:
        _chat_fragment()


@st.fragment(run_every=1)
def _header_fragment() -> None:
    """Compact single-row header: title · round/phase/game · scoreboard."""
    state: SharedState = st.session_state.state
    info = state.get_round_info()
    current = info.get("current_round", 0)
    total = info.get("total_rounds", 0)
    game_type = info.get("game_type", "")
    phase = state.get_phase()

    game_label = GAME_TYPE_LABELS.get(game_type, game_type or "—")
    phase_label = PHASE_LABELS.get(phase, phase)
    round_label = f"R{current}/{total}" if current else "—"

    score_parts: List[str] = []
    for pid in ALL_PLAYERS:
        if pid == HUMAN_ID:
            label = f"{st.session_state.player_name} (You)"
        else:
            label = f"{AI_META[pid]['emoji']} {AI_META[pid]['name']}"
        score_parts.append(f"{label} **{state.get_score(pid)}**")

    st.markdown(
        f"#### 🎭 How to Fool AI &nbsp;·&nbsp; {round_label} &nbsp;·&nbsp; {phase_label} &nbsp;·&nbsp; {game_label}"
    )
    score_caption = " &nbsp;·&nbsp; ".join(score_parts)
    active_model = st.session_state.get("active_model")
    if active_model:
        score_caption += f" &nbsp;·&nbsp; <span style='opacity:0.6'>model: {active_model}</span>"
    st.caption(score_caption, unsafe_allow_html=True)


@st.fragment(run_every=1)
def _game_area_fragment() -> None:
    """Auto-refreshes every 1 s: drains human inbox, runs the countdown."""
    state: SharedState = st.session_state.state

    # ── drain human inbox ────────────────────────────────────────────────────
    for item in state.get_my_messages(HUMAN_ID):
        if not isinstance(item, GameEvent):
            continue
        etype = item.event_type
        if etype == "make_decision":
            st.session_state.pending_decision = item.payload
        elif etype == "poison_result":
            st.session_state.pending_poison_result = item.payload
        elif etype == "poison_picker_change":
            st.session_state.current_poison_picker = item.payload
        elif etype == "round_end":
            payload = item.payload or {}
            flat_results = {**payload, **payload.get("results", {})}
            st.session_state.last_round_results = flat_results
            st.session_state.round_history.append(
                {
                    "round": payload.get("round"),
                    "game_type": payload.get("game_type"),
                    "score_deltas": payload.get("score_deltas", {}),
                    "scores_after_round": payload.get("scores_after_round", {}),
                }
            )
            # Clear pending decision + per-round transient state.
            st.session_state.pending_decision = None
            st.session_state.pending_poison_result = None
            st.session_state.current_poison_picker = None
        elif etype == "phase_change":
            # Clear pending decision when entering reveal phase.
            if item.payload.get("phase") == "reveal":
                st.session_state.pending_decision = None
                st.session_state.current_poison_picker = None
        elif etype == "game_over":
            st.session_state.page = "game_over"
            st.rerun()
            return

    # ── phase / round header ─────────────────────────────────────────────────
    phase = state.get_phase()
    if phase == "game_over":
        st.session_state.page = "game_over"
        st.rerun()
        return

    info = state.get_round_info()
    current = info["current_round"]
    total = info["total_rounds"]
    game_type = info["game_type"]

    if current == 0:
        st.info("⏳ Game is starting…")
        return

    # ── reveal: show last round results + Next Round button ─────────────────
    if phase == "reveal" and st.session_state.last_round_results:
        _render_round_results(st.session_state.last_round_results)
        st.divider()
        is_last = current >= total
        label = "🏁 See Final Results" if is_last else "➡️ Next Round"
        if st.button(label, key=f"next_round_btn_{current}", use_container_width=True):
            state.signal_continue()
            st.session_state.last_round_results = None
        return

    # ── playing: show pending decision or waiting message ────────────────────
    if phase == "playing":
        # Poison-bottle: always show "currently picking" banner above
        # everything else for this mini-game.
        if game_type == "poison_bottle":
            _render_poison_picker_banner()

        decision = st.session_state.pending_decision
        if decision:
            _render_decision_ui(decision, state)
        else:
            # Show immediate poison-bottle result if available.
            poison_res = st.session_state.pending_poison_result
            if poison_res is not None:
                _render_poison_result(poison_res)
                st.divider()
            _render_waiting_guidance(game_type, current)


def _render_round_results(results: dict) -> None:
    """Show results after a round ends (reveal phase)."""
    game_type = results.get("game_type", "")
    st.subheader("🔍 Round Results")

    if game_type == "guess_the_word":
        writer = results.get("writer", "?")
        word = results.get("word", "?")
        writer_name = _player_display(writer)
        st.write(f"**Writer:** {writer_name}  |  **Word:** `{word}`")
        st.write("**Guesses:**")
        for pid, guess in results.get("guesses", {}).items():
            correct = guess.strip().lower() == word.strip().lower()
            icon = "✅" if correct else "❌"
            st.write(f"  {icon} {_player_display(pid)}: `{guess}`")

    elif game_type == "who_wrote_it":
        st.write("**Words written:**")
        for pid, word in results.get("words", {}).items():
            st.write(f"  {_player_display(pid)}: `{word}`")

    elif game_type == "poison_bottle":
        poisoned = results.get("poisoned_bottle", "?")
        st.write(f"**Poisoned bottle:** {poisoned}")
        if order := results.get("selection_order"):
            st.write(
                "**Selection order (highest score first):** "
                + " → ".join(_player_display(pid) for pid in order)
            )
        for pid, bottle in results.get("choices", {}).items():
            poisoned_flag = " ☠️" if bottle.lower() == poisoned.lower() else ""
            st.write(f"  {_player_display(pid)}: **{bottle}**{poisoned_flag}")

    # Score deltas
    deltas = results.get("score_deltas", {})
    if any(d != 0 for d in deltas.values()):
        st.write("**Score changes this round:**")
        for pid, delta in deltas.items():
            if delta != 0:
                sign = "+" if delta > 0 else ""
                st.write(f"  {_player_display(pid)}: {sign}{delta}")


def _render_poison_picker_banner() -> None:
    """Show 'X is choosing a bottle now' above the bottle area during poison."""
    info = st.session_state.get("current_poison_picker")
    if not info:
        return
    picker_id = info.get("picker")
    pos = info.get("position", "?")
    total = info.get("total", "?")
    picker_name = _player_display(picker_id) if picker_id else "?"
    is_you = picker_id == HUMAN_ID
    label = "🎯 **Your turn**" if is_you else f"🎯 **{picker_name}** is choosing"
    st.warning(f"{label}  ·  position {pos}/{total}")


def _render_poison_result(result: dict) -> None:
    """Show the immediate poison/safe feedback after the player picks."""
    choice = result.get("choice", "?")
    if result.get("is_poisoned"):
        st.error(f"☠️ You drank the **{choice}** bottle — it was poisoned! −1 point.")
    else:
        st.success(f"✅ You drank the **{choice}** bottle — it was safe!")


def _render_countdown(decision: dict) -> None:
    """Show remaining seconds in the current decision window."""
    deadline = decision.get("deadline_ts")
    if deadline is None:
        return
    import time as _time
    remaining = max(0, int(deadline - _time.time()))
    total = int(decision.get("phase_duration") or remaining or 1)
    progress = max(0.0, min(1.0, remaining / total)) if total else 0.0
    if remaining <= 10:
        st.error(f"⏰ Time left: **{remaining}s** — hurry!")
    elif remaining <= 30:
        st.warning(f"⏳ Time left: **{remaining}s**")
    else:
        mins, secs = divmod(remaining, 60)
        st.info(f"⏳ Time left: **{mins}:{secs:02d}**")
    st.progress(progress)


def _render_decision_ui(decision: dict, state: SharedState) -> None:
    """Show the appropriate choice widget based on the pending decision action."""
    action = decision.get("action", "")
    game_type = decision.get("game_type", "")

    st.subheader(f"Your turn — {GAME_TYPE_LABELS.get(game_type, game_type)}")
    _render_countdown(decision)

    current_choice = state.get_choice(HUMAN_ID)

    if action == "write_word":
        role = decision.get("your_role", "writer")
        if role == "writer":
            st.info("You are the writer this round. Mislead or be honest in private chat — anything goes.")
        st.write("✍️ **Write a word** (one English word) — locked once submitted:")
        word = st.text_input("Your word", key="decision_write_word", max_chars=20)
        if st.button("Submit word", key="btn_write_word") and word.strip():
            state.submit_choice(HUMAN_ID, word.strip())
            st.session_state.pending_decision = None
            st.success(f"Submitted: `{word.strip()}` — locked.")

    elif action == "guess_word":
        writer_name = _player_display(decision.get("writer", "?"))
        st.info("Goal: guess the word for +1. Use private chat to probe for truths or lies.")
        st.write(f"🔮 **Guess {writer_name}'s secret word:**")
        guess = st.text_input("Your guess", key="decision_guess_word", max_chars=20)
        label = "Update guess" if current_choice else "Submit guess"
        if st.button(label, key="btn_guess_word") and guess.strip():
            state.submit_choice(HUMAN_ID, guess.strip())
            st.success(f"Submitted: `{guess.strip()}`")
        if current_choice:
            st.caption(f"✅ Currently submitted: `{current_choice}` — you can update before time runs out.")

    elif action == "guess_authors":
        words = decision.get("words", [])
        candidates = decision.get("candidate_authors", [])
        st.info("Goal: identify as many correct authors as possible. In chat, you may claim authorship of any word.")
        st.write("🎭 **Guess who wrote each word:**")
        guesses = []
        for i, word in enumerate(words):
            options = [_player_display(c) for c in candidates]
            choice = st.selectbox(
                f"`{word}` was written by…",
                options,
                key=f"decision_author_{i}",
            )
            # Map display name back to player_id
            guesses.append(candidates[options.index(choice)])
        label = "Update attributions" if current_choice else "Submit attributions"
        if st.button(label, key="btn_guess_authors"):
            state.submit_choice(HUMAN_ID, ",".join(guesses))
            st.success("Submitted!")
        if current_choice:
            st.caption("✅ Submitted — you can update before time runs out.")

    elif action == "choose_bottle":
        available = decision.get("available_bottles", [])
        position = decision.get("your_position", "?")
        st.write(f"☠️ **Choose a bottle** (you pick #{position} in selection order):")
        st.caption("One bottle is poisoned. Drinking it costs 1 point. You will be told immediately whether your pick was poisoned.")
        st.info("It's your turn. Other players' scores are hidden, but the selection order leaks the relative ranking.")
        bottle_cols = st.columns(len(available))
        bottle_emojis = {"Red": "🔴", "Blue": "🔵", "Green": "🟢", "Yellow": "🟡"}
        for col, bottle in zip(bottle_cols, available):
            emoji = bottle_emojis.get(bottle, "🍾")
            if col.button(f"{emoji} {bottle}", key=f"bottle_{bottle}"):
                state.submit_choice(HUMAN_ID, bottle)
                # Drinking is irreversible — clear the widget immediately
                # so the panel doesn't look frozen while later players pick.
                st.session_state.pending_decision = None

    else:
        st.warning(f"Unknown action: `{action}`")


# ── page: Chat fragment ───────────────────────────────────────────────────────

@st.fragment(run_every=2)
def _chat_fragment() -> None:
    """WeChat-style chat: avatar row with unread badges + a single active panel."""
    state: SharedState = st.session_state.state
    ss = st.session_state
    budget = state.get_send_budget(HUMAN_ID)
    history = state.get_chat_history()

    # ── Whisper strip — last 2 AI-to-AI notifications (content hidden) ──
    whispers = [e for e in history if e.get("type") == "whisper"]
    with st.container(border=True):
        if not whispers:
            st.caption("🤫 *No AI-to-AI whispers yet.*")
        else:
            for e in whispers[-2:]:
                s = _ai_display_name(e["sender"])
                r = _ai_display_name(e["recipient"])
                ts = e.get("timestamp")
                ts_str = ts.strftime("%H:%M:%S") if ts is not None else ""
                st.caption(f"🤫 `{ts_str}`  ·  {s} → {r}")

    # ── Send-budget indicator ─────────────────────────────────────────
    budget_display = str(budget) if budget is not None else "∞"
    if budget is not None and budget <= 0:
        st.warning("📭 No sends remaining this round.", icon="⚠️")
    else:
        st.caption(f"Sends remaining: **{budget_display} / 10**")

    # ── Avatar row with unread badges ─────────────────────────────────
    ai_ids = [pid for pid in ALL_PLAYERS if pid != HUMAN_ID]
    # Per-AI count of INCOMING messages (AI → you).  Sent messages don't
    # produce unread badges.
    incoming_counts: Dict[str, int] = {
        ai_id: sum(
            1 for e in history
            if e.get("type") == "chat"
            and e["sender"] == ai_id
            and e["recipient"] == HUMAN_ID
        )
        for ai_id in ai_ids
    }

    cols = st.columns(len(ai_ids))
    for col, ai_id in zip(cols, ai_ids):
        meta = AI_META[ai_id]
        last_seen = ss.chat_last_seen.get(ai_id, 0)
        is_active = ss.active_chat_ai == ai_id
        unread = incoming_counts[ai_id] - last_seen
        # Build label: emoji name + 🔴N when there's an unread for a non-active AI.
        label_parts = [meta["emoji"], meta["name"]]
        if unread > 0 and not is_active:
            label_parts.append(f"🔴{unread}")
        label = " ".join(label_parts)
        if col.button(
            label,
            key=f"avatar_btn_{ai_id}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            ss.active_chat_ai = ai_id
            ss.chat_last_seen[ai_id] = incoming_counts[ai_id]
            st.rerun(scope="fragment")

    # The currently-active AI is always considered "read up to now".
    ss.chat_last_seen[ss.active_chat_ai] = incoming_counts.get(ss.active_chat_ai, 0)

    # ── Single active conversation panel ──────────────────────────────
    send_disabled = budget is not None and budget <= 0
    _render_ai_chat_panel(state, history, ss.active_chat_ai, send_disabled)


def _render_ai_chat_panel(
    state: SharedState,
    history: List[Dict[str, Any]],
    ai_id: str,
    send_disabled: bool,
) -> None:
    """WeChat-style single-conversation panel: AI bubbles left, human bubbles right."""
    meta = AI_META[ai_id]
    emoji = meta["emoji"]
    name = meta["name"]

    entries = [
        e for e in history
        if e.get("type") == "chat"
        and (
            (e["sender"] == HUMAN_ID and e["recipient"] == ai_id)
            or (e["sender"] == ai_id and e["recipient"] == HUMAN_ID)
        )
    ]

    with st.container(border=True):
        st.markdown(f"##### {emoji} {name}")
        with st.container(height=360, border=False):
            if not entries:
                st.caption(f"*No messages with {name} yet.*")
            else:
                for entry in entries:
                    is_human = entry["sender"] == HUMAN_ID
                    avatar = "🧑" if is_human else emoji
                    text = entry["text"].replace("<", "&lt;").replace(">", "&gt;")
                    st.markdown(
                        _bubble_html(text, avatar, right_aligned=is_human),
                        unsafe_allow_html=True,
                    )

        with st.form(f"chat_form_{ai_id}", clear_on_submit=True):
            st.text_input(
                f"Message to {name}",
                label_visibility="collapsed",
                placeholder=f"Type a message to {name}…",
                disabled=send_disabled,
                key=f"chat_input_{ai_id}",
            )
            st.caption("↩ Press Enter or click Send to deliver the message.")
            submitted = st.form_submit_button(
                f"Send to {name}",
                disabled=send_disabled,
                use_container_width=True,
            )
            if submitted:
                msg = st.session_state.get(f"chat_input_{ai_id}", "")
                if msg and msg.strip():
                    try:
                        state.send_message(HUMAN_ID, ai_id, msg.strip())
                    except RuntimeError as exc:
                        st.error(str(exc))


def _bubble_html(text: str, avatar: str, *, right_aligned: bool) -> str:
    """Render a single WeChat-style message bubble as HTML.

    Right-aligned (human) bubbles get a green-ish background and the avatar
    on the right; left-aligned (AI) bubbles get a neutral grey with avatar
    on the left.
    """
    bubble_bg = "#dcf8c6" if right_aligned else "#f1f0f0"
    text_color = "#111"
    justify = "flex-end" if right_aligned else "flex-start"
    avatar_html = (
        f'<div style="font-size:1.3em; margin-left:6px;">{avatar}</div>'
        if right_aligned
        else f'<div style="font-size:1.3em; margin-right:6px;">{avatar}</div>'
    )
    bubble = (
        f'<div style="background:{bubble_bg}; color:{text_color}; '
        f'padding:6px 10px; border-radius:10px; max-width:75%; '
        f'word-wrap:break-word; line-height:1.35;">{text}</div>'
    )
    inner = (bubble + avatar_html) if right_aligned else (avatar_html + bubble)
    return (
        f'<div style="display:flex; justify-content:{justify}; '
        f'align-items:flex-end; margin:4px 0;">{inner}</div>'
    )


# ── page: Game Over ───────────────────────────────────────────────────────────

def render_game_over() -> None:
    st.set_page_config(
        page_title="How to Fool AI — Results", page_icon="🏁", layout="centered"
    )

    state: SharedState = st.session_state.state

    st.title("🏁 Game Over")
    st.markdown("All scores are now revealed.")
    st.divider()

    # Build rankings.
    scores = {pid: state.get_score(pid) for pid in ALL_PLAYERS}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    medal = ["🥇", "🥈", "🥉"] + ["  "] * 10
    rows = []
    for rank, (pid, score) in enumerate(ranked):
        name = (
            f"{st.session_state.player_name} (You)"
            if pid == HUMAN_ID
            else f"{AI_META[pid]['emoji']} {AI_META[pid]['name']}"
        )
        rows.append((medal[rank], name, score))

    st.subheader("Final Rankings")
    for m, name, score in rows:
        cols = st.columns([1, 4, 1])
        cols[0].markdown(f"### {m}")
        cols[1].markdown(f"**{name}**")
        cols[2].markdown(f"**{score} pts**")

    # Winner announcement.
    winner_pid, winner_score = ranked[0]
    human_score = scores[HUMAN_ID]
    st.divider()
    if winner_pid == HUMAN_ID:
        st.success(
            f"🎉 You win with **{human_score} point{'s' if human_score != 1 else ''}**!"
            " You successfully outfoxed the AIs."
        )
    else:
        winner_name = f"{AI_META[winner_pid]['emoji']} {AI_META[winner_pid]['name']}"
        st.info(
            f"The AIs win this round — **{winner_name}** takes first place "
            f"with **{winner_score} pts**.  Better luck next time!"
        )

    st.divider()
    if st.session_state.round_history:
        st.subheader("Round-by-Round Score History")
        table_rows = []
        for rec in st.session_state.round_history:
            row = {
                "Round": rec.get("round"),
                "Game": GAME_TYPE_LABELS.get(rec.get("game_type"), rec.get("game_type")),
            }
            deltas = rec.get("score_deltas", {})
            scores_after = rec.get("scores_after_round", {})
            for pid in ALL_PLAYERS:
                delta = deltas.get(pid, 0)
                sign = "+" if delta > 0 else ""
                row[_player_display(pid)] = f"{sign}{delta} (total {scores_after.get(pid, 0)})"
            table_rows.append(row)
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.divider()
    _render_chat_timeline(state)

    st.divider()
    if st.button("🔄 Play Again", use_container_width=True):
        _stop_threads()
        # Reset relevant session state without clearing player_name.
        for key in ("state", "pending_decision", "last_round_results",
                    "chat_rendered_up_to", "round_history"):
            st.session_state.pop(key, None)
        st.session_state.page = "setup"
        st.rerun()


def _render_chat_timeline(state: SharedState) -> None:
    """Show every chat record (incl. previously hidden AI-to-AI text) in order."""
    history = state.get_chat_history()
    st.subheader("💬 Full Chat Timeline")
    if not history:
        st.caption("*No messages were exchanged this game.*")
        return
    st.caption("All private messages, in chronological order. AI-to-AI conversations and AI private reasoning that were hidden during play are now revealed.")

    # Round filter — collect rounds actually present in history.
    rounds_present = sorted({(e.get("round") or 0) for e in history})
    options = ["All rounds"] + [
        ("Pre-game" if r == 0 else f"Round {r}") for r in rounds_present
    ]
    label_to_round: Dict[str, int | None] = {"All rounds": None}
    for r in rounds_present:
        label_to_round["Pre-game" if r == 0 else f"Round {r}"] = r

    selected = st.radio(
        "Show:",
        options,
        horizontal=True,
        key="timeline_round_filter",
    )
    target_round = label_to_round.get(selected)

    filtered = [
        e for e in history
        if target_round is None or (e.get("round") or 0) == target_round
    ]
    if not filtered:
        st.caption(f"*No messages in {selected}.*")
        return

    with st.container(height=560, border=True):
        last_round_shown: int | None = None
        for entry in filtered:
            entry_round = entry.get("round") or 0
            # When viewing All, insert a small divider between rounds.
            if target_round is None and entry_round != last_round_shown:
                hdr = "Pre-game" if entry_round == 0 else f"Round {entry_round}"
                st.markdown(f"##### — {hdr} —")
                last_round_shown = entry_round
            ts = entry.get("timestamp")
            ts_str = ts.strftime("%H:%M:%S") if ts is not None else ""
            sender = _player_display(entry["sender"])
            recipient = _player_display(entry["recipient"])
            text = entry.get("text", "")
            thinking = entry.get("thinking")
            tag = "🤫" if entry["type"] == "whisper" else "💬"
            st.markdown(
                f"`{ts_str}` {tag} **{sender} → {recipient}:** {text}"
            )
            if thinking:
                with st.expander("🧠 reasoning", expanded=False):
                    st.caption(thinking)


# ── helpers ───────────────────────────────────────────────────────────────────

def _player_display(player_id: str) -> str:
    """Human-readable label for any player ID."""
    if player_id == HUMAN_ID:
        return f"**{st.session_state.get('player_name', 'You')}** (You)"
    meta = AI_META.get(player_id)
    if meta:
        return f"{meta['emoji']} {meta['name']}"
    return player_id


def _ai_display_name(player_id: str) -> str:
    """Short display name for an AI (used in whisper notifications)."""
    meta = AI_META.get(player_id)
    if meta:
        return f"{meta['emoji']} {meta['name']}"
    return player_id


def _render_waiting_guidance(game_type: str, current_round: int) -> None:
    """Show contextual guidance when the player is waiting for their turn."""
    st.info("⏳ Waiting for other players. Use chat to probe, bluff, or coordinate.")
    if game_type == "guess_the_word":
        st.caption(
            f"Round {current_round}: Writer sets a hidden word, others guess. "
            "During this window, private messages can leak truth or lies."
        )
    elif game_type == "who_wrote_it":
        st.caption(
            f"Round {current_round}: Everyone writes one word and attributes others. "
            "Claiming authorship (true or false) is a valid strategy."
        )
    elif game_type == "poison_bottle":
        st.caption(
            f"Round {current_round}: Selection order follows hidden score ranking "
            "(highest picks first). Watch order and chat for tells."
        )


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    page = st.session_state.page
    if page == "setup":
        render_setup()
    elif page == "game":
        render_game()
    elif page == "game_over":
        render_game_over()
    else:
        st.error(f"Unknown page: {page!r}")


main()
