"""Player state persistence.

Privacy decision: progress lives in a single JSON file per player under
`data/players/`. There is no database, no account, and nothing that leaves the
machine unless the player submits evidence for model verification. Precise
location is never stored or requested -- `Evidence.location_label` is a
free-text human label.

An in-memory index is kept for speed and written back atomically so a crash
mid-write cannot corrupt the file.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

from app.models.schemas import PlayerState, XpEvent

_LOCK = threading.RLock()
_CACHE: dict[str, PlayerState] = {}

#: XP needed for level 1, growing by a constant factor per level.
_BASE_XP = 100
_XP_STEP = 50


def data_dir() -> Path:
    root = os.environ.get("SIDEQUEST_DATA_DIR", "data")
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def players_dir() -> Path:
    d = data_dir() / "players"
    d.mkdir(parents=True, exist_ok=True)
    return d


def xp_for_level(level: int) -> int:
    """Cumulative XP required to *reach* `level`."""
    if level <= 1:
        return 0
    return _BASE_XP * (level - 1) + _XP_STEP * ((level - 1) * (level - 2) // 2)


def level_for_xp(xp: int) -> int:
    level = 1
    while xp_for_level(level + 1) <= xp and level < 99:
        level += 1
    return level


def _path(player_id: str) -> Path:
    # player_id is generated hex, but sanitize defensively: it reaches the FS.
    safe = "".join(c for c in player_id if c.isalnum() or c in "-_")[:64] or "unknown"
    return players_dir() / f"{safe}.json"


def _persist(state: PlayerState) -> None:
    """Atomic write: temp file in the same dir, then replace."""
    state.updated_at = time.time()
    target = _path(state.player_id)
    fd, tmp = tempfile.mkstemp(dir=str(players_dir()), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state.model_dump(), fh, indent=2)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_or_create(player_id: str | None = None) -> PlayerState:
    """Load a player, creating a fresh one if unknown.

    This never raises for an unknown id -- that is the "new player" path.
    """
    with _LOCK:
        if player_id and player_id in _CACHE:
            return _CACHE[player_id]

        if player_id:
            p = _path(player_id)
            if p.exists():
                try:
                    state = PlayerState.model_validate_json(p.read_text(encoding="utf-8"))
                    _CACHE[state.player_id] = state
                    return state
                except Exception:
                    # Corrupt file: start fresh rather than 500 forever.
                    pass

        state = PlayerState(player_id=player_id or PlayerState().player_id)
        _CACHE[state.player_id] = state
        _persist(state)
        return state


def save(state: PlayerState) -> PlayerState:
    with _LOCK:
        state.level = level_for_xp(state.xp)
        _persist(state)
        _CACHE[state.player_id] = state
        return state


def award(state: PlayerState, amount: int, reason: str, quest_id: str | None = None) -> PlayerState:
    """Add XP and record the event. Negative amounts are allowed for corrections."""
    state.xp = max(0, state.xp + amount)
    state.history.append(XpEvent(amount=amount, reason=reason, quest_id=quest_id))
    # Keep the log bounded; the eval suite and README document the cap.
    if len(state.history) > 200:
        del state.history[:-200]
    return save(state)


def clear_cache() -> None:
    """Test hook."""
    with _LOCK:
        _CACHE.clear()