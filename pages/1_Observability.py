"""Streamlit development dashboard for traces and replay comparison."""

from pathlib import Path

import streamlit as st

from game.dashboard_data import build_dashboard_snapshot, compare_replays
from game.replay import ReplayRepository


st.set_page_config(page_title="Agent Observability", page_icon="🔍", layout="wide")
st.title("🔍 Agent Observability & Replay")
st.caption(
    "Development-only trace inspection. Private messages are hidden by default."
)

repository = ReplayRepository(
    Path(__file__).resolve().parents[1] / "logs" / "traces"
)
rows = repository.list_replays(limit=200)
if not rows:
    st.info("No replay traces yet. Finish a game first.")
    st.stop()

labels = {
    row["game_id"]: (
        f"{row['game_id'][:8]} · {row['event_count']} events · "
        f"{len(row['rounds'])} rounds"
    )
    for row in rows
}
game_id = st.selectbox(
    "Replay",
    options=list(labels),
    format_func=labels.get,
)
include_sensitive = st.toggle(
    "Reveal private messages and model text",
    value=False,
)
if include_sensitive:
    st.warning("This view may contain private AI-to-AI messages.")

replay = repository.get(game_id)
snapshot = build_dashboard_snapshot(
    replay,
    include_sensitive=include_sensitive,
)
summary = snapshot["summary"]
metrics = snapshot["metrics"]

metric_columns = st.columns(6)
metric_columns[0].metric("Events", summary["event_count"])
metric_columns[1].metric("Actions", metrics["actions"])
metric_columns[2].metric(
    "First-call legal",
    metrics["first_call_legal_rate"],
)
metric_columns[3].metric("Fallback rate", metrics["fallback_rate"])
metric_columns[4].metric("P95 latency", metrics["latency_p95_ms"])
metric_columns[5].metric(
    "Tokens",
    metrics["input_tokens"] + metrics["output_tokens"],
)

tab_timeline, tab_llm, tab_tools, tab_state, tab_compare = st.tabs([
    "Timeline",
    "LLM",
    "Tools",
    "State",
    "Compare",
])

with tab_timeline:
    event_types = sorted({
        event["event_type"] for event in snapshot["timeline"]
    })
    actors = sorted({
        event["actor_id"]
        for event in snapshot["timeline"]
        if event["actor_id"]
    })
    rounds = sorted({
        event["round"]
        for event in snapshot["timeline"]
        if event["round"] is not None
    })
    selected_types = st.multiselect("Event types", event_types)
    selected_actor = st.selectbox("Actor", ["All"] + actors)
    selected_round = st.selectbox("Round", ["All"] + rounds)
    filtered = [
        event
        for event in snapshot["timeline"]
        if (not selected_types or event["event_type"] in selected_types)
        and (selected_actor == "All" or event["actor_id"] == selected_actor)
        and (selected_round == "All" or event["round"] == selected_round)
    ]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    with st.expander("Raw selected events"):
        st.json(filtered)

with tab_llm:
    st.dataframe(
        snapshot["latency_rows"],
        use_container_width=True,
        hide_index=True,
    )
    if snapshot["latency_rows"]:
        st.line_chart(
            snapshot["latency_rows"],
            x="sequence",
            y=["latency_ms", "input_tokens", "output_tokens"],
        )

with tab_tools:
    st.bar_chart(snapshot["tool_counts"])
    tool_events = [
        event
        for event in snapshot["timeline"]
        if event["event_type"] == "tool_execution"
    ]
    st.dataframe(tool_events, use_container_width=True, hide_index=True)

with tab_state:
    st.dataframe(
        snapshot["state_changes"],
        use_container_width=True,
        hide_index=True,
    )

with tab_compare:
    other_ids = [item for item in labels if item != game_id]
    if not other_ids:
        st.caption("A second replay is required for comparison.")
    else:
        candidate_id = st.selectbox(
            "Candidate replay",
            other_ids,
            format_func=labels.get,
        )
        st.dataframe(
            compare_replays(replay, repository.get(candidate_id)),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Deltas are descriptive only; statistical significance is not claimed."
        )
