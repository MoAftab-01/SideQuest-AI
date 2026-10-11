"""HTTP surface tests for the SideQuest AI API.

Two promises this file exists to hold the app to:

1. `/api/health` reports what is *actually* loaded. With no weights on disk it
   must say ``loaded: false``, not imply a model is waiting behind the next
   request.
2. ``Verdict.is_ai_verified`` is never set by a heuristic or button-press path.
   The only route to True is a model verdict, and the XP rules follow from that:
   a heuristic run credits outdoor minutes plus a fraction of XP; an
   AI-verified verdict credits the full reward.

Everything runs against `UnavailableEngine` or a tiny scripted
`InferenceEngine`. No weights are loaded, no network is touched, and the whole
module finishes in seconds on CPU.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# `app.main` mounts a static front end at import time when `SIDEQUEST_WEB_DIR`
# exists. Point it at a path that cannot exist so this module imports the same
# way whether or not a built front end is present, and never mounts `web/`.
os.environ["SIDEQUEST_WEB_DIR"] = str(Path(__file__).parent / "__no_web_here__")

from fastapi.testclient import TestClient  # noqa: E402

from app.ai.engine import EngineStatus, GenerateResult, GeneratorSource, InferenceEngine  # noqa: E402
from app.ai.engine import UnavailableEngine  # noqa: E402
from app import main as api  # noqa: E402
from app import store  # noqa: E402


# ---------------------------------------------------------------------------
# Scripted engine
# ---------------------------------------------------------------------------


class ScriptedEngine(InferenceEngine):
    """Replays canned text instead of running a model.

    Every knob the API is allowed to branch on -- `status().loaded` and
    `supports_vision` -- is settable, so the "model present" and "model absent"
    configurations are exercised through the same routes.
    """

    def __init__(
        self,
        *,
        text: str = "",
        vision_text: str | None = None,
        loaded: bool = True,
        vision: bool = True,
        model_id: str = "fake/scripted-model",
    ) -> None:
        self.model_id = model_id
        self.supports_vision = vision
        self._loaded = loaded
        self._text = text
        self._vision_text = text if vision_text is None else vision_text
        #: Prompts this engine was asked to complete, in order.
        self.prompts: list[str] = []
        self.images: list[bytes] = []

    def load(self) -> bool:
        return self._loaded

    def generate(self, prompt: str, **kw) -> GenerateResult | None:
        self.prompts.append(prompt)
        if not self._loaded:
            return None
        return GenerateResult(
            text=self._text,
            latency_ms=1,
            model_id=self.model_id,
            source=GeneratorSource.MODEL,
        )

    def generate_with_image(self, prompt: str, image_bytes: bytes, **kw) -> GenerateResult | None:
        self.prompts.append(prompt)
        self.images.append(image_bytes)
        if not (self._loaded and self.supports_vision):
            return None
        return GenerateResult(
            text=self._vision_text,
            latency_ms=2,
            model_id=self.model_id,
            source=GeneratorSource.MODEL,
        )

    def status(self) -> EngineStatus:
        return EngineStatus(
            model_id=self.model_id if self._loaded else None,
            loaded=self._loaded,
            device="cpu" if self._loaded else "none",
            dtype="float32" if self._loaded else "none",
            supports_vision=self.supports_vision if self._loaded else False,
            load_error=None if self._loaded else "scripted engine offline",
        )


# A small 1x1 PNG, base64-encoded, used as a "photo" that decodes cleanly.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PHOTO = f"data:image/png;base64,{TINY_PNG_B64}"

GOOD_QUEST = {
    "title": "Count the Doorbells",
    "hook": "Twelve front doors, one walk.",
    "premise": "A slow look at the public face of a street.",
    "steps": [
        {"instruction": "Pick one stretch of pavement.", "minutes": 4},
        {"instruction": "Photograph four doorbells.", "minutes": 6},
    ],
    "tags": ["details", "street"],
    "xp_reward": 120,
    "verification": {
        "subject": "A close photograph of a doorbell on a residential front door.",
        "must_include": ["a doorbell is visible"],
    },
}

GOOD_VERDICT = {
    "verdict": "verified",
    "confidence": 0.91,
    "summary": "A doorbell fills the frame, lit and legible.",
    "matched": ["doorbell", "front door"],
    "missing": [],
    "needs_better_photo": False,
}

UNSAFE_QUEST = {
    "title": "Swim Across the River",
    "premise": "A fast mission in open water.",
    "steps": [{"instruction": "Swim to the far bank before lunch.", "minutes": 20}],
    "xp_reward": 300,
}

PLAYER = "apitestplayer"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the store at a per-test tmp dir and drop the in-memory cache."""
    monkeypatch.setenv("SIDEQUEST_DATA_DIR", str(tmp_path / "data"))
    store.clear_cache()
    yield
    store.clear_cache()


@pytest.fixture
def client(monkeypatch):
    """A TestClient whose engine is whatever the test installed."""

    def _make(engine: InferenceEngine | None) -> TestClient:
        installed = engine if engine is not None else UnavailableEngine("no engine registered")
        monkeypatch.setattr(api, "_engine", installed)
        # Any route that reaches build_engine() could contact OpenRouter.
        # Failing loudly proves it is never called in this test.
        monkeypatch.setattr(
            api,
            "build_engine",
            lambda: pytest.fail("build_engine() must not be called in tests"),
        )
        return TestClient(api.app)

    return _make


@pytest.fixture
def no_model_client(client):
    return client(UnavailableEngine("no engine registered"))


def make_quest(client: TestClient, **overrides) -> dict:
    """POST /api/quest with a valid body and return the parsed quest."""
    body = {"time_budget_minutes": 60, "seed": 1234}
    body.update(overrides)
    resp = client.post("/api/quest", json=body, params={"player_id": PLAYER})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# /api/health -- never overstate what is loaded
# ---------------------------------------------------------------------------


def test_health_reports_no_model_when_none_is_loaded(no_model_client):
    body = no_model_client.get("/api/health").json()

    assert body["status"] == "ok"
    engine = body["engine"]
    assert engine["loaded"] is False
    assert engine["model_id"] is None
    assert engine["supports_vision"] is False
    assert engine["device"] == "none"
    assert engine["dtype"] == "none"
    # The reason is surfaced rather than hidden behind a cheerful status.
    assert engine["load_error"]


def test_health_reports_loaded_model_honestly(client):
    loaded = ScriptedEngine(loaded=True, vision=True, model_id="fake/4b")
    body = client(loaded).get("/api/health").json()

    assert body["engine"]["loaded"] is True
    assert body["engine"]["model_id"] == "fake/4b"
    assert body["engine"]["supports_vision"] is True
    assert body["engine"]["load_error"] is None


def test_health_does_not_claim_a_model_when_engine_load_failed(client):
    """A failed load is reported as a failure, not as a working fallback model."""
    engine = ScriptedEngine(loaded=False, vision=True, model_id="fake/4b")
    engine_report = client(engine).get("/api/health").json()["engine"]

    assert engine_report["loaded"] is False
    # Even though the object knows a model id, nothing loaded, so nothing claims one.
    assert engine_report["model_id"] is None
    assert engine_report["load_error"] == "scripted engine offline"


def test_health_reuses_the_installed_engine_and_never_builds_one(monkeypatch):
    """With an engine installed, nothing may reach for weights."""
    called = []
    installed = UnavailableEngine("no engine registered")
    monkeypatch.setattr(api, "_engine", installed)
    monkeypatch.setattr(api, "build_engine", lambda: called.append(1) or UnavailableEngine())

    body = TestClient(api.app).get("/api/health").json()

    assert not called, "get_engine() rebuilt instead of reusing the installed engine"
    assert body["engine"]["loaded"] is False


# ---------------------------------------------------------------------------
# /api/quest -- a quest, and a truthful generation.source
# ---------------------------------------------------------------------------


def test_post_quest_returns_a_complete_quest(no_model_client):
    payload = make_quest(no_model_client)

    quest = payload["quest"]
    assert quest["title"]
    assert quest["steps"], "a quest with no steps is not playable"
    assert [s["order"] for s in quest["steps"]] == list(range(1, len(quest["steps"]) + 1))
    assert quest["verification"]["subject"]
    assert quest["estimated_minutes"] >= 1
    assert quest["xp_reward"] >= 5
    assert payload["player_id"] == PLAYER


def test_post_quest_without_a_model_says_so_in_generation(no_model_client):
    payload = make_quest(no_model_client)

    generation = payload["generation"]
    assert generation["source"] == "fallback"
    assert generation["model_id"] is None
    assert generation["parse_ok"] is False
    assert generation["fallback_reason"], "a fallback must record why it fell back"
    # The quest object agrees with the generation record.
    assert payload["quest"]["source"] == "fallback"
    assert payload["quest"]["model_id"] is None
    assert payload["quest"]["raw_model_output"] is None


def test_post_quest_from_a_model_attributes_the_source(client):
    engine = ScriptedEngine(text=json.dumps(GOOD_QUEST))
    payload = make_quest(client(engine))

    generation = payload["generation"]
    assert generation["source"] == "model"
    assert generation["model_id"] == "fake/scripted-model"
    assert generation["parse_ok"] is True
    assert generation["safety_passed"] is True
    assert generation["fallback_reason"] == ""
    assert payload["quest"]["title"] == GOOD_QUEST["title"]
    assert payload["quest"]["source"] == "model"
    assert payload["quest"]["model_id"] == "fake/scripted-model"
    # Full untruncated output is kept for the eval suite.
    assert GOOD_QUEST["title"] in payload["quest"]["raw_model_output"]


def test_post_quest_falls_back_when_model_output_is_unparseable(client):
    engine = ScriptedEngine(text="Sure! Here is a lovely idea: go outside and look at a bird.")
    payload = make_quest(client(engine))

    assert payload["generation"]["source"] == "fallback"
    assert payload["generation"]["parse_ok"] is False
    assert payload["generation"]["parse_reason"] == "unparseable_json"
    assert payload["quest"]["source"] == "fallback"
    assert payload["quest"]["steps"], "the player still gets a playable mission"


def test_post_quest_retries_once_on_a_parse_failure(client):
    """One blunter retry is worth it; the second failure gives up."""
    engine = ScriptedEngine(text="no json here, sorry")
    make_quest(client(engine))

    assert len(engine.prompts) == 2, "expected exactly one retry prompt"
    assert engine.prompts[0] != engine.prompts[1]


def test_post_quest_safety_gate_overrides_the_model(client):
    """The model does not get the last word: an unsafe mission is replaced."""
    engine = ScriptedEngine(text=json.dumps(UNSAFE_QUEST))
    payload = make_quest(client(engine))

    assert payload["generation"]["source"] == "fallback"
    assert payload["generation"]["safety_passed"] is False
    assert "safety gate blocked" in payload["generation"]["fallback_reason"]

    delivered = payload["quest"]
    assert delivered["source"] == "fallback"
    assert "swim" not in delivered["title"].lower()
    # The replacement itself cleared the policy.
    assert delivered["safety"]["passed"] is True
    assert delivered["steps"]


def test_safety_block_is_not_retried(client):
    """A safety block is a real signal, not a formatting problem to re-prompt."""
    engine = ScriptedEngine(text=json.dumps(UNSAFE_QUEST))
    make_quest(client(engine))

    assert len(engine.prompts) == 1, "a blocked mission must not trigger the retry"


def test_post_quest_same_seed_is_reproducible(no_model_client):
    first = make_quest(no_model_client)["quest"]
    second = make_quest(no_model_client)["quest"]

    for field in ("title", "hook", "premise", "estimated_minutes", "xp_reward", "steps"):
        assert first[field] == second[field], f"{field} was not reproducible for one seed"


def test_post_quest_keeps_only_the_twenty_most_recent(no_model_client):
    for _ in range(23):
        make_quest(no_model_client)

    state = no_model_client.get("/api/state", params={"player_id": PLAYER}).json()["state"]
    assert len(state["quests"]) == 20


def test_post_quest_honours_the_time_budget(no_model_client):
    for budget in (5, 20, 90):
        quest = make_quest(no_model_client, time_budget_minutes=budget)["quest"]
        total = sum(s["approx_minutes"] for s in quest["steps"])
        assert total <= budget, f"steps overran a {budget} minute budget"


@pytest.mark.parametrize(
    "payload",
    [
        {"time_budget_minutes": 4},
        {"time_budget_minutes": 500},
        {"interests": ["a", "b", "c", "d", "e", "f", "g", "h", "i"]},
        {"context_note": "x" * 401},
        {"max_radius_m": 10},
        {"weather": "monsoon"},
    ],
)
def test_post_quest_rejects_impossible_requests(no_model_client, payload):
    resp = no_model_client.post("/api/quest", json=payload, params={"player_id": PLAYER})
    assert resp.status_code == 422


def test_interests_are_normalised_before_they_reach_the_generator(client):
    """Tags are trimmed, lower-cased and stripped of blanks on the way in."""
    engine = ScriptedEngine(text=json.dumps(GOOD_QUEST))
    make_quest(client(engine), interests=["  Birds ", "NATURE", "", "   "])

    prompt = engine.prompts[0]
    assert "interests: birds, nature" in prompt
    assert "  Birds " not in prompt


def test_quests_are_isolated_per_player(no_model_client):
    mine = make_quest(no_model_client)["quest"]
    other = no_model_client.post(
        "/api/quest",
        json={"time_budget_minutes": 60, "seed": 7},
        params={"player_id": "someoneelse"},
    ).json()["quest"]

    assert mine["id"] != other["id"]
    resp = no_model_client.post(
        "/api/verify",
        json={"note": "", "elapsed_minutes": 5},
        params={"quest_id": other["id"], "player_id": PLAYER},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/verify -- unknown quests are 404
# ---------------------------------------------------------------------------


def test_verify_unknown_quest_id_is_404(no_model_client):
    resp = no_model_client.post(
        "/api/verify",
        json={"note": "did it", "elapsed_minutes": 12},
        params={"quest_id": "does-not-exist", "player_id": PLAYER},
    )

    assert resp.status_code == 404
    assert resp.json()["detail"] == "quest not found for this player"


def test_verify_unknown_quest_id_credits_nothing(no_model_client):
    no_model_client.post(
        "/api/verify",
        json={"note": "did it", "elapsed_minutes": 12},
        params={"quest_id": "does-not-exist", "player_id": PLAYER},
    )

    summary = no_model_client.get("/api/state", params={"player_id": PLAYER}).json()["summary"]
    assert summary["xp"] == 0
    assert summary["minutes_outdoors"] == 0


def test_verify_rejects_an_unknown_player_id(no_model_client):
    quest = make_quest(no_model_client)["quest"]
    resp = no_model_client.post(
        "/api/verify",
        json={"note": "", "elapsed_minutes": 1},
        params={"quest_id": quest["id"], "player_id": "stranger"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/verify -- the XP rules
# ---------------------------------------------------------------------------


def verify(client: TestClient, quest: dict, evidence: dict, player: str = PLAYER):
    resp = client.post(
        "/api/verify",
        json=evidence,
        params={"quest_id": quest["id"], "player_id": player},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_heuristic_verdict_credits_minutes_but_only_a_fraction_of_xp(no_model_client):
    """No model, no photo: outdoor time is real, verification is not."""
    quest = make_quest(no_model_client)["quest"]
    payload = verify(no_model_client, quest, {"note": "went outside", "elapsed_minutes": 17})

    verdict = payload["verdict"]
    expected_partial = max(1, quest["xp_reward"] // 5)

    assert verdict["is_ai_verified"] is False
    assert verdict["level"] == "heuristic"
    assert verdict["source"] == "fallback"
    assert verdict["xp_awarded"] == expected_partial
    assert expected_partial < quest["xp_reward"]

    summary = payload["summary"]
    assert summary["minutes_outdoors"] == 17
    assert summary["xp"] == expected_partial
    assert summary["verified_count"] == 0


def test_heuristic_verdict_never_claims_verification(no_model_client):
    """The core promise, on the button-press path: nothing sets the flag."""
    quest = make_quest(no_model_client)["quest"]
    # A confident, detailed note and a photo -- still no model ran.
    payload = verify(
        no_model_client,
        quest,
        {"note": "I did the whole thing and photographed everything.", "elapsed_minutes": 5,
         "photo_data_url": PHOTO},
    )

    verdict = payload["verdict"]
    # A photo without a vision model is UNCERTAIN, not verified.
    assert verdict["is_ai_verified"] is False
    assert verdict["level"] in ("uncertain", "heuristic")
    assert verdict["model_id"] is None
    assert payload["verification"]["source"] == "fallback"
    assert payload["summary"]["verified_count"] == 0
    assert payload["summary"]["minutes_outdoors"] == 5


def test_ai_verified_verdict_credits_the_full_reward(client):
    engine = ScriptedEngine(vision_text=json.dumps(GOOD_VERDICT))
    c = client(engine)
    quest = make_quest(c)["quest"]

    payload = verify(c, quest, {"note": "photo attached", "elapsed_minutes": 25, "photo_data_url": PHOTO})

    verdict = payload["verdict"]
    assert verdict["is_ai_verified"] is True
    assert verdict["level"] == "verified"
    assert verdict["source"] == "model"
    assert verdict["model_id"] == "fake/scripted-model"
    assert verdict["xp_awarded"] == quest["xp_reward"]

    summary = payload["summary"]
    assert summary["xp"] == quest["xp_reward"]
    assert summary["verified_count"] == 1
    assert summary["minutes_outdoors"] == 25


def test_ai_verified_passes_only_after_the_model_actually_saw_an_image(client):
    """A model verdict on evidence we never gave it an image for is not earned."""
    engine = ScriptedEngine(vision_text=json.dumps(GOOD_VERDICT))
    c = client(engine)
    quest = make_quest(c)["quest"]

    payload = verify(c, quest, {"note": "no photo this time", "elapsed_minutes": 4})

    verdict = payload["verdict"]
    assert verdict["is_ai_verified"] is False
    assert verdict["level"] == "heuristic"
    assert not engine.images, "the model was called without any image bytes"


def test_model_asking_for_a_better_photo_is_not_verified(client):
    engine = ScriptedEngine(
        vision_text=json.dumps(
            {
                "verdict": "verified",
                "confidence": 0.95,
                "summary": "Looks fine I guess",
                "needs_better_photo": True,
            }
        )
    )
    c = client(engine)
    quest = make_quest(c)["quest"]

    payload = verify(c, quest, {"note": "", "elapsed_minutes": 9, "photo_data_url": PHOTO})

    verdict = payload["verdict"]
    assert verdict["needs_better_photo"] is True
    assert verdict["level"] == "uncertain"
    assert verdict["is_ai_verified"] is False
    assert payload["summary"]["xp"] == 0
    assert payload["summary"]["uncertain_count"] == 1


def test_model_refusal_is_uncertain_not_a_pass(client):
    engine = ScriptedEngine(
        vision_text=json.dumps({"verdict": "I cannot determine what is in this photo"})
    )
    c = client(engine)
    quest = make_quest(c)["quest"]

    verdict = verify(c, quest, {"note": "", "elapsed_minutes": 3, "photo_data_url": PHOTO})["verdict"]

    assert verdict["level"] == "uncertain"
    assert verdict["is_ai_verified"] is False
    assert verdict["needs_better_photo"] is True


def test_rejected_verdict_records_no_progress_but_still_credits_the_walk(client):
    engine = ScriptedEngine(
        vision_text=json.dumps(
            {"verdict": "rejected", "confidence": 0.8, "summary": "No doorbell in frame."}
        )
    )
    c = client(engine)
    quest = make_quest(c)["quest"]

    payload = verify(c, quest, {"note": "", "elapsed_minutes": 14, "photo_data_url": PHOTO})

    assert payload["verdict"]["is_ai_verified"] is False
    assert payload["verdict"]["level"] == "rejected"
    assert payload["verdict"]["xp_awarded"] == 0
    assert payload["summary"]["rejected_count"] == 1
    assert payload["summary"]["minutes_outdoors"] == 14


def test_unparseable_verdict_falls_back_rather_than_claiming_a_pass(client):
    """A confident-looking but unreadable reply grants nothing."""
    engine = ScriptedEngine(vision_text="Yes, that is definitely a doorbell!")
    c = client(engine)
    quest = make_quest(c)["quest"]

    payload = verify(c, quest, {"note": "", "elapsed_minutes": 11, "photo_data_url": PHOTO})

    verdict = payload["verdict"]
    assert verdict["is_ai_verified"] is False
    assert verdict["level"] in ("uncertain", "heuristic")
    assert payload["verification"]["source"] == "fallback"
    assert "unparseable_verdict" in payload["verification"]["notes"]
    assert payload["summary"]["xp"] == 0
    assert payload["summary"]["minutes_outdoors"] == 11


def test_loaded_but_text_only_model_cannot_verify_a_photo(client):
    engine = ScriptedEngine(vision_text=json.dumps(GOOD_VERDICT), vision=False)
    c = client(engine)
    quest = make_quest(c)["quest"]

    payload = verify(c, quest, {"note": "", "elapsed_minutes": 7, "photo_data_url": PHOTO})

    assert payload["verdict"]["is_ai_verified"] is False
    assert payload["verdict"]["level"] == "uncertain"
    assert "engine_text_only" in payload["verification"]["notes"]
    assert not engine.images


def test_undecodable_photo_is_not_a_verification(client):
    engine = ScriptedEngine(vision_text=json.dumps(GOOD_VERDICT))
    c = client(engine)
    quest = make_quest(c)["quest"]

    payload = verify(
        c,
        quest,
        {"note": "", "elapsed_minutes": 6, "photo_data_url": "data:image/png;base64,%%%"},
    )

    assert payload["verdict"]["is_ai_verified"] is False
    assert payload["verdict"]["needs_better_photo"] is True
    assert "undecodable_photo" in payload["verification"]["notes"]
    assert not engine.images


def test_zero_elapsed_minutes_does_not_inflate_outdoor_time(no_model_client):
    quest = make_quest(no_model_client)["quest"]
    payload = verify(no_model_client, quest, {"note": "forgot to time it", "elapsed_minutes": 0})

    assert payload["summary"]["minutes_outdoors"] == 0


def test_repeated_verifications_accumulate_outdoor_minutes(no_model_client):
    quest = make_quest(no_model_client)["quest"]
    first = verify(no_model_client, quest, {"note": "", "elapsed_minutes": 10})["summary"]
    second = verify(no_model_client, quest, {"note": "", "elapsed_minutes": 8})["summary"]

    assert first["minutes_outdoors"] == 10
    assert second["minutes_outdoors"] == 18

    state = no_model_client.get("/api/state", params={"player_id": PLAYER}).json()["state"]
    assert state["quests_completed"] == 2
    # Each run credited its own share of XP, exactly once.
    assert state["xp"] == 2 * max(1, quest["xp_reward"] // 5)


@pytest.mark.parametrize("elapsed", [-1, 2000])
def test_verify_rejects_impossible_elapsed_minutes(no_model_client, elapsed):
    quest = make_quest(no_model_client)["quest"]
    resp = no_model_client.post(
        "/api/verify",
        json={"note": "", "elapsed_minutes": elapsed},
        params={"quest_id": quest["id"], "player_id": PLAYER},
    )
    assert resp.status_code == 422


def test_verify_rejects_an_oversized_note(no_model_client):
    quest = make_quest(no_model_client)["quest"]
    resp = no_model_client.post(
        "/api/verify",
        json={"note": "x" * 601, "elapsed_minutes": 1},
        params={"quest_id": quest["id"], "player_id": PLAYER},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/state and the session summary
# ---------------------------------------------------------------------------


def test_state_starts_empty_for_an_unknown_player(no_model_client):
    payload = no_model_client.get("/api/state", params={"player_id": "brandnew"}).json()

    state = payload["state"]
    assert state["player_id"] == "brandnew"
    assert state["xp"] == 0
    assert state["level"] == 1
    assert state["quests"] == []
    assert payload["summary"]["xp_to_next_level"] == store.xp_for_level(2)


def test_screen_time_saved_is_derived_from_outdoor_minutes(no_model_client):
    quest = make_quest(no_model_client)["quest"]
    summary = verify(no_model_client, quest, {"note": "", "elapsed_minutes": 30})["summary"]

    assert summary["screen_time_saved_seconds"] == 30 * 60


def test_state_persists_across_clients(client):
    """A restart must not lose the player's progress or the XP ledger."""
    c1 = client(ScriptedEngine(text=json.dumps(GOOD_QUEST), vision_text=json.dumps(GOOD_VERDICT)))
    quest = make_quest(c1)["quest"]
    verify(c1, quest, {"note": "", "elapsed_minutes": 22, "photo_data_url": PHOTO})

    # A brand-new process: empty cache, fresh client, same data directory.
    store.clear_cache()
    c2 = client(ScriptedEngine(text=json.dumps(GOOD_QUEST), vision_text=json.dumps(GOOD_VERDICT)))
    payload = c2.get("/api/state", params={"player_id": PLAYER}).json()

    assert payload["summary"]["xp"] == quest["xp_reward"]
    assert payload["summary"]["minutes_outdoors"] == 22
    assert payload["summary"]["verified_count"] == 1
    assert payload["state"]["quests"][0]["id"] == quest["id"]
    # The XP ledger records the award with a reason a human can read.
    assert payload["state"]["history"][-1]["amount"] == quest["xp_reward"]
    assert "verified" in payload["state"]["history"][-1]["reason"]


def test_state_survives_a_corrupt_player_file(client):
    """A half-written file must not wedge the player on a permanent 500."""
    c = client(ScriptedEngine(text=json.dumps(GOOD_QUEST)))
    quest = make_quest(c)["quest"]

    # Truncate the on-disk save to something no longer valid JSON.
    path = store.players_dir() / f"{PLAYER}.json"
    path.write_text('{"player_id": "apitestplayer", "xp": 4', encoding="utf-8")

    store.clear_cache()
    fresh = client(ScriptedEngine(text=json.dumps(GOOD_QUEST)))
    payload = fresh.get("/api/state", params={"player_id": PLAYER}).json()

    # Recovery is a fresh player, not a crash and not resurrected garbage.
    assert payload["state"]["xp"] == 0
    assert payload["state"]["quests"] == []
    assert payload["summary"]["xp"] == 0


def test_level_rises_with_earned_xp(no_model_client):
    quest = make_quest(no_model_client)["quest"]
    summary = verify(no_model_client, quest, {"note": "", "elapsed_minutes": 10})["summary"]

    assert summary["xp"] == max(1, quest["xp_reward"] // 5)
    assert summary["level"] == store.level_for_xp(summary["xp"])
    assert summary["xp_to_next_level"] == max(
        0, store.xp_for_level(summary["level"] + 1) - summary["xp"]
    )

# ---------------------------------------------------------------------------
# Front-end bootstrap
# ---------------------------------------------------------------------------


def test_app_imports_when_web_dir_has_no_assets_subdir(tmp_path):
    """A web dir without ``assets/`` must not stop the module from importing.

    This test caught a real startup crash, and it is kept because the failure it
    guards against is easy to reintroduce. The static mount in ``main.py`` was
    guarded only on ``os.path.isdir(WEB_DIR)`` and then unconditionally mounted
    ``os.path.join(WEB_DIR, "assets")``. This repo's own ``web/`` exists but has
    no ``assets/`` subdirectory -- the front end is plain files, no build step --
    so importing ``app.main`` with the default WEB_DIR raised at import time:
    ``RuntimeError: Directory 'web\assets' does not exist``.

    Every other test in this file hid the bug by pointing SIDEQUEST_WEB_DIR at a
    path that does not exist at all, which made the isdir() guard False and
    skipped the mount entirely. So ``uvicorn app.main:app`` failed out of the box
    on a fresh checkout while the whole suite stayed green.
    """
    import subprocess
    import sys

    import app

    backend_dir = Path(app.__file__).resolve().parent.parent

    # A front end directory that genuinely exists, with index.html, but no
    # assets/ subdirectory -- exactly the shape of this repo's web/.
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<html></html>", encoding="utf-8")

    env = dict(os.environ)
    env["SIDEQUEST_WEB_DIR"] = str(web_dir)
    env["PYTHONPATH"] = str(backend_dir)

    # Subprocess: app.main is already imported in this process and its module
    # cache would ignore the env var, and re-importing in-process would leak the
    # broken module state into every later test.
    proc = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, (
        "importing app.main failed with an existing web dir that has no assets/:\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_app_serves_a_full_mission_with_the_network_forcibly_disabled(tmp_path):
    """The deterministic mode must survive egress being switched off entirely.

    The README claims the deterministic mode runs with the network off. This is
    that claim turned into a test, because the project specifically does not
    want prose asserting a capability it has not exercised.

    The distinction this pins is narrower than it might look, and worth stating
    precisely, because it is easy to overclaim here:

      * CLAIMED AND TESTED (this test): with no network and no provider key, the app
        still generates a complete, safety-checked mission and says out loud
        that it fell back.
      * NOT CLAIMED: that OpenRouter inference works offline.

    With no OpenRouter API key, the app must not make a provider request and
    must keep serving its deterministic fallback. The environment is set in a
    subprocess because ``app.main`` is already imported in this process.
    """
    import subprocess
    import sys

    import app

    backend_dir = Path(app.__file__).resolve().parent.parent
    probe = f"""
import json
from fastapi.testclient import TestClient
from app.main import app as fastapi_app

client = TestClient(fastapi_app)
out = {{}}

health = client.get("/api/health").json()
out["provider_configured"] = health["provider_configured"]

resp = client.post(
    "/api/quest?player_id=offline-probe",
    json={{
        "time_budget_minutes": 25,
        "interests": ["birds"],
        "energy": "low",
        "location_type": "park",
    }},
)
out["status"] = resp.status_code
body = resp.json()
quest = body["quest"]
out["source"] = body["generation"]["source"]
out["title"] = quest["title"]
out["steps"] = len(quest["steps"])
out["safety_passed"] = quest["safety"]["passed"]
out["subject"] = quest["verification"]["subject"]
out["minutes"] = sum(s["approx_minutes"] for s in quest["steps"])
print("RESULT:" + json.dumps(out))
"""
    env = dict(os.environ)
    env["SIDEQUEST_WEB_DIR"] = str(tmp_path / "no_web")
    env["SIDEQUEST_DATA_DIR"] = str(tmp_path / "data")
    env.pop("OPENROUTER_API_KEY", None)
    env["PYTHONPATH"] = str(backend_dir)

    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            payload = json.loads(line[len("RESULT:"):])

    context = (
        "The app must generate a complete mission with the network disabled and "
        f"say it fell back.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert payload is not None, context

    assert payload["status"] == 200, context
    assert payload["provider_configured"] is False, context
    assert payload["source"] == "fallback", context
    assert payload["title"].strip(), context
    assert payload["steps"] > 0, context
    assert payload["safety_passed"] is True, context
    assert payload["subject"].strip(), context
    assert payload["minutes"] <= 25, context
