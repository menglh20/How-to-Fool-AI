# Architecture

## Directory Structure

```
developer/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── ARCHITECTURE.md
├── .streamlit/
│   └── secrets.toml.example      # API key template
├── game/
│   ├── __init__.py
│   └── shared_state.py           # Thread-safe shared state
└── tests/
    └── test_shared_state.py
```

## Core Concepts

### Players
Each AI player is identified by a string ID (e.g. `"ai_0"`, `"ai_1"`, `"ai_2"`).
Players are registered at startup. Each player has:
- An independent `queue.Queue` inbox — messages sent *to* them land here.
- A score (int).
- A `send_budget` (int) — decremented each time the player sends a message.

### SharedState (`game/shared_state.py`)
Single object shared across all Streamlit threads and background AI threads.
Protected by `threading.RLock` so concurrent score updates and message sends are safe.

Key methods:
| Method | Description |
|---|---|
| `send_message(sender, recipient, text)` | Puts a `ChatMessage` in recipient's inbox, decrements sender's `send_budget` |
| `get_my_messages(player_id)` | Drains and returns all queued `ChatMessage` objects for `player_id` |
| `get_score(player_id)` | Returns current score |
| `update_score(player_id, delta)` | Thread-safely adds `delta` to score |
| `broadcast_event(event)` | Puts a `GameEvent` into every player's inbox |

### Data Classes
- `ChatMessage(sender, recipient, text, timestamp)`
- `GameEvent(event_type, payload, timestamp)`
