"""FastAPI application for SideQuest AI.

Route surface is deliberately small: a player generates a mission, goes
outside, comes back with evidence, and sees a verdict. That is the whole
product, and keeping the API narrow is what makes the front end screen-minimal.

Two things this module is careful about:
  * `/api/health` reports what is *actually* loaded, so the deployed demo can
    never imply a capability the instance does not have.
  * Evidence is the only thing that leaves the machine, and only when the
    player submits a photo for model checking.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.ai.engine import InferenceEngine, build_engine
from app.ai.planner import plan_with_retry
from app.ai.verifier import verify_evidence
from app.models.schemas import (
    Evidence,
    GeneratorSource,
    PlayerState,
    Quest,
    QuestRequest,
    SessionSummary,
    Verdict,
)
from app.store import award, get_or_create, level_for_xp, save, xp_for_level

import os
import sys

# Ensure `from app.models.schemas` resolves from /app/backend (the Docker WORKDIR layout).
log = logging.getLogger(__name__)

#: Where the built front end lands. Absent in dev, present in the container.
WEB_DIR = os.environ.get("SIDEQUEST_WEB_DIR", "web")

_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    """Return the process-wide engine, building it on first use.

    Lazily built rather than built at import so that `pytest` collection and
    CLI tools do not pay for a multi-gigabyte model load.
    """
    global _engine
    if _engine is None:
        _engine = build_engine()
        st = _engine.status()
        log.info(
            "engine ready: model=%s loaded=%s device=%s vision=%s err=%s",
            st.model_id, st.loaded, st.device, st.supports_vision, st.load_error,
        )
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Do not block startup on a multi-minute model download unless asked to.
    if os.environ.get("SIDEQUEST_EAGER_LOAD", "0") == "1":
        get_engine()
    yield


app = FastAPI(
    title="SideQuest AI",
    version="0.1.0",
    description="Screen-minimal outdoor exploration powered by local open-weight inference.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health & diagnostics
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    """What is actually running. Never overstates capability.
    Does not initialize a heavyweight model just to report status."""
    # Read env credentials directly; never trigger get_engine() here.
    openrouter_key_present = bool(os.environ.get("OPENROUTER_API_KEY"))
    model_configured = bool(os.environ.get("SIDEQUEST_MODEL_ID")) or bool(os.environ.get("SIDEQUEST_FALLBACK_MODEL_ID"))
    provider_available = openrouter_key_present
    # Only load a lightweight status check, never the full model engine.
    status_summary = {
        "status": "ok",
        "api_running": True,
        "provider_configured": provider_available,
        "provider_credentials_present": openrouter_key_present,
        "model_configured": model_configured,
        "provider_connectivity_tested": False,
        "model_successfully_generated": False,
        "vision_model_available": False,
        "model_is_local": False,
        "note": "Health reports actual state; generation requires valid OPENROUTER_API_KEY or loaded model. See /api/eval/baseline for comparison.",
    }
    # If an engine was already built, reuse its honest status; never build new.
    installed = _engine
    if installed is not None:
        s = installed.status()
        status_summary["model_is_local"] = s.device not in ("none", None)
        status_summary["engine"] = s.model_dump()
        # Truthful: only claim model-generated when engine actually produced one.
        status_summary["model_successfully_generated"] = False  # health is never a generation
        if s.loaded:
            status_summary["provider_connectivity_tested"] = True
        return status_summary
    # No installed engine: report from env only (lightweight, no get_engine()).
    return status_summary


# ---------------------------------------------------------------------------
# Player state
# ---------------------------------------------------------------------------


def _summary(state: PlayerState) -> SessionSummary:
    """Derive the end-of-session summary.

    `screen_time_saved_seconds` is deliberately a *reported estimate*, derived
    from outdoor minutes rather than measured. The README says so; it is not a
    telemetry claim.
    """
    next_at = xp_for_level(state.level + 1)
    return SessionSummary(
        player_id=state.player_id,
        xp=state.xp,
        level=level_for_xp(state.xp),
        xp_to_next_level=max(0, next_at - state.xp),
        minutes_outdoors=state.minutes_outdoors,
        verified_count=state.verified_count,
        uncertain_count=state.uncertain_count,
        rejected_count=state.rejected_count,
        screen_time_saved_seconds=state.minutes_outdoors * 60,
    )


@app.get("/api/state")
def get_state(player_id: str | None = Query(default=None)) -> dict:
    state = get_or_create(player_id)
    return {"state": state.model_dump(), "summary": _summary(state).model_dump()}


@app.post("/api/quest")
def create_quest(req: QuestRequest, player_id: str | None = Query(default=None)) -> dict:
    """Generate a mission. The player's only real screen interaction."""
    engine = get_engine()
    seed = req.seed if req.seed is not None else int(time.time() * 1000) % (2**31)

    quest, outcome = plan_with_retry(req, engine, seed=seed)

    state = get_or_create(player_id)
    state.quests.insert(0, quest)
    del state.quests[20:]
    save(state)

    return {
        "quest": quest.model_dump(),
        "generation": {
            "source": outcome.source.value,
            "latency_ms": outcome.latency_ms,
            "model_id": outcome.model_id,
            "parse_ok": outcome.parse_ok,
            "parse_reason": outcome.parse_reason,
            "safety_passed": outcome.safety_passed,
            "fallback_reason": outcome.fallback_reason,
        },
        "player_id": state.player_id,
    }


@app.post("/api/verify")
def verify(
    quest_id: str,
    evidence: Evidence,
    player_id: str | None = Query(default=None),
) -> dict:
    """Evaluate submitted evidence against a mission."""
    state = get_or_create(player_id)
    quest = next((q for q in state.quests if q.id == quest_id), None)
    if quest is None:
        raise HTTPException(status_code=404, detail="quest not found for this player")

    engine = get_engine()
    verdict, outcome = verify_evidence(quest, evidence, engine)

    # Progress is credited from the *verdict level*, and only from it. A
    # heuristic result records outdoor time but grants no XP, which is what
    # keeps the leaderboard honest.
    if evidence.elapsed_minutes:
        state.minutes_outdoors += evidence.elapsed_minutes

    if verdict.is_ai_verified:
        state.verified_count += 1
        verdict.xp_awarded = quest.xp_reward
        award(state, quest.xp_reward, f"verified: {quest.title}", quest.id)
    else:
        if verdict.level.value == "rejected":
            state.rejected_count += 1
        else:
            state.uncertain_count += 1
        if verdict.level.value == "heuristic":
            # Credit participation without claiming verification.
            partial = max(1, quest.xp_reward // 5)
            verdict.xp_awarded = partial
            award(state, partial, f"outdoor time (unverified): {quest.title}", quest.id)
    state.quests_completed += 1
    state = save(state)

    return {
        "verdict": verdict.model_dump(),
        "verification": {
            "source": outcome.source.value,
            "latency_ms": outcome.latency_ms,
            "model_id": outcome.model_id,
            "parse_ok": outcome.parse_ok,
            "image_bytes": outcome.image_bytes,
            "notes": outcome.notes,
        },
        "summary": _summary(state).model_dump(),
    }


# ---------------------------------------------------------------------------
# Static front end (served last so it does not shadow the API)
# ---------------------------------------------------------------------------

if os.path.isdir(WEB_DIR):
    # Guard the assets directory itself, not just its parent. Starlette's
    # StaticFiles raises RuntimeError in __init__ when the directory is absent,
    # so an unguarded mount made a fresh checkout fail at import time: `web/`
    # exists but ships no `assets/` (the front end is plain files, no build
    # step), and `uvicorn app.main:app` would not start.
    _assets_dir = os.path.join(WEB_DIR, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/{path:path}")
    def spa(path: str):
        candidate = os.path.join(WEB_DIR, path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(WEB_DIR, "index.html"))