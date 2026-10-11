"""Mission planning: the reliability ladder in ``backend/app/ai/planner.py``.

This module is where the project claims open-weight inference is load-bearing.
The claim only holds if the ladder is honest, so these tests pin the parts a
player would notice if they broke:

  1. a clean, parseable, safe model response is used *as-is*;
  2. unparseable output falls back to the deterministic generator, honestly
     labelled ``GeneratorSource.FALLBACK`` with a machine-readable reason;
  3. a response the safety gate blocks falls back -- and is **never** retried,
     because a blocked mission is a real signal, not a formatting glitch;
  4. a *parse* failure **is** retried, exactly once, with a stricter prompt;
  5. the quest handed back always fits the request's time budget, even when the
     model's step durations are hostile;
  6. ``PlanOutcome`` -- the object the eval harness and ``/api/quest`` serialise
     -- is populated consistently with the quest actually returned;
  7. with no model at all (``UnavailableEngine``) the result is still a valid,
     safe, in-budget quest.

The headline rule, per the docstring on ``plan_with_retry``:

    Only retries on *parse* failures -- a safety-blocked mission is a real
    signal that we should fall back, not a formatting problem to re-prompt.

Engines here are hand-built fakes returning canned ``GenerateResult`` objects.
No external provider is called, no weights are loaded, and no test
touches the network, so the whole file runs in well under a second.
"""

from __future__ import annotations

import json

import pytest

from app.ai import planner
from app.ai.engine import EngineStatus, InferenceEngine, UnavailableEngine
from app.ai.fallback import compose_prompt
from app.models.schemas import GenerateResult, GeneratorSource, QuestRequest

# ---------------------------------------------------------------------------
# Canned model payloads. Small models are messy in predictable ways, so each
# one below reproduces a failure mode the eval harness has already seen.
# ---------------------------------------------------------------------------

#: A well-formed, safe mission. This is the "model did its job" path.
CLEAN = json.dumps(
    {
        "title": "Count the Birds",
        "hook": "Twenty minutes, one flock.",
        "premise": "Find a flock and count them properly.",
        "steps": [{"instruction": "Count them", "minutes": 5, "detail": "Use binoculars."}],
        "estimated_minutes": 20,
        "subject": "One clear photo of the flock",
        "must_include": ["the whole flock in frame"],
    }
)

#: Trips ``haz-water-unprotected``: open water, no lifeguard, no PFD.
UNSAFE_WATER = json.dumps(
    {
        "title": "Swim Across The Reservoir",
        "premise": "Jump in the deep end and dive for the bottom.",
        "steps": [{"instruction": "Swim to the far bank", "minutes": 5}],
    }
)

#: Steps that reach for the tower -- blocked when the player asked to avoid stairs.
UNSAFE_STAIRS = json.dumps(
    {
        "title": "Tower Climb",
        "premise": "Head up the tower.",
        "steps": [{"instruction": "Climb the tower stairs", "minutes": 5}],
    }
)

#: The classic small-model failure: fluent prose, no JSON at all.
PROSE = "I think a nice walk would be fun! Here are some ideas:"

#: Advisory-only trip: "after dark" raises a warning but must not block.
WARN_ONLY = json.dumps(
    {
        "title": "Evening Pigeon Count",
        "premise": "Count pigeons after dark on the station concourse.",
        "steps": [{"instruction": "Count them", "minutes": 5}],
    }
)

RETRY_SUFFIX = planner._RETRY_SUFFIX


def make_request(**overrides) -> QuestRequest:
    """A cheap, always-valid request. Overrides make budgets/parties explicit."""
    payload = {
        "time_budget_minutes": 30,
        "interests": ["birds"],
        "seed": 7,
    }
    payload.update(overrides)
    return QuestRequest(**payload)


class ScriptedEngine(InferenceEngine):
    """An engine that replays canned texts and records how it was called.

    ``texts[i]`` is returned on call ``i``; once the script runs out the last
    entry repeats, so a test that cares *how many* times we called the model
    cannot accidentally pass because the script fell off the end.
    """

    def __init__(self, texts, *, model_id: str = "fake-gemma", latency_ms: int = 7):
        self._texts = list(texts)
        self.model_id = model_id
        self.latency_ms = latency_ms
        self.prompts: list[str] = []
        self.calls = 0

    def load(self) -> None:  # pragma: no cover - never reached in these tests
        return None

    def generate(self, prompt: str, **_kwargs) -> GenerateResult | None:
        self.prompts.append(prompt)
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        if text is None:
            return None  # a dead engine: no result at all
        return GenerateResult(
            text=text,
            latency_ms=self.latency_ms,
            model_id=self.model_id,
            source=GeneratorSource.MODEL,
        )

    def generate_with_image(self, prompt, image, **_kwargs):  # pragma: no cover
        return None

    def status(self) -> EngineStatus:
        return EngineStatus(loaded=True, device="cpu", dtype="float32", model_id=self.model_id)


# ---------------------------------------------------------------------------
# The ladder, rung by rung: what happens at each step of
# model -> parse -> validate -> safety gate.
# ---------------------------------------------------------------------------


def test_clean_model_response_is_used_as_is():
    """A good mission survives the whole ladder untouched and keeps its provenance."""
    engine = ScriptedEngine([CLEAN])
    quest, outcome = planner.plan_quest(make_request(), engine)

    assert outcome.source is GeneratorSource.MODEL
    assert outcome.parse_ok is True
    assert outcome.safety_passed is True
    assert outcome.fallback_reason == "", "using the model must not report a fallback reason"

    assert quest.source is GeneratorSource.MODEL
    assert quest.title == "Count the Birds"
    assert quest.model_id == "fake-gemma"
    assert [s.order for s in quest.steps] == [1]
    assert quest.steps[0].approx_minutes == 5
    assert quest.verification.subject == "One clear photo of the flock"
    assert "the whole flock in frame" in quest.verification.must_include
    assert quest.raw_model_output == CLEAN, "raw output is kept as evidence"


@pytest.mark.parametrize(
    "text, expected_reason",
    [
        pytest.param(CLEAN, "ok", id="bare-json"),
        pytest.param("Here you go:\n```json\n" + CLEAN + "\n```\nHope that helps!", "ok_fenced", id="fenced"),
        pytest.param("Sure! " + CLEAN + " Let me know if you want more.", "ok_extracted", id="prose-wrapped"),
    ],
)
def test_json_is_recovered_from_the_messy_shapes_models_actually_emit(text, expected_reason):
    """Fences and trailing chatter are formatting, not failure -- we should still use the mission.

    Small open-weight models rarely emit a bare object. Recovering the object is
    the difference between a real model mission and a template, so this pins all
    three extraction strategies from ``parsing.extract_json``.
    """
    quest, outcome = planner.plan_quest(make_request(), ScriptedEngine([text]))

    assert outcome.source is GeneratorSource.MODEL
    assert outcome.parse_reason == expected_reason
    assert quest.title == "Count the Birds"


def test_unparseable_output_falls_back_to_the_deterministic_generator():
    """Fluent prose with no JSON in it must not reach the player as a quest."""
    engine = ScriptedEngine([PROSE])
    quest, outcome = planner.plan_quest(make_request(), engine, seed=7)

    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.parse_ok is False
    assert outcome.parse_reason == "unparseable_json"
    assert outcome.fallback_reason == "model output unparseable (unparseable_json)"
    assert outcome.model_id == "fake-gemma", "we record which model failed, even on fallback"

    assert quest.source is GeneratorSource.FALLBACK
    assert quest.title != "", "the player still gets a mission"
    assert quest.model_id is None, "a fallback quest is not attributed to the model"
    assert quest.steps, "the fallback quest has steps"


@pytest.mark.parametrize(
    "payload, expected_reason",
    [
        pytest.param({"steps": [{"instruction": "x", "minutes": 5}]}, "model produced no title", id="no-title"),
        pytest.param({"title": "T"}, "model produced no steps", id="no-steps"),
        pytest.param({"title": "T", "steps": []}, "model produced no steps", id="empty-steps"),
        pytest.param({"title": "T", "steps": [{"detail": "no instruction"}]}, "model produced no usable steps", id="no-usable-steps"),
    ],
)
def test_json_that_parses_but_fails_validation_falls_back(payload, expected_reason):
    """A parse failure is not a pass. Validation runs after parsing, and its failures are reported as parse failures.

    This is the rung people get wrong: returning a dict is not the same as
    producing a usable mission. ``parse_ok`` must be ``False`` for these, which
    is also what makes them eligible for a retry.
    """
    quest, outcome = planner.plan_quest(make_request(), ScriptedEngine([json.dumps(payload)]))

    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.parse_ok is False
    assert outcome.parse_reason == expected_reason
    assert outcome.fallback_reason == f"model output failed validation: {expected_reason}"
    assert quest.source is GeneratorSource.FALLBACK
    assert quest.steps


# ---------------------------------------------------------------------------
# The safety gate. Model output is gated exactly like fallback output, and a
# block is a decision, not a hiccup.
# ---------------------------------------------------------------------------


def test_safety_blocked_model_mission_falls_back_to_a_safe_quest():
    """A mission the gate blocks never reaches the player -- not even in edited form."""
    quest, outcome = planner.plan_quest(make_request(), ScriptedEngine([UNSAFE_WATER]))

    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.safety_passed is False
    assert outcome.parse_ok is True, "the JSON was fine; it was the content that was blocked"
    assert outcome.fallback_reason.startswith("safety gate blocked model mission: haz-water-unprotected")

    assert quest.source is GeneratorSource.FALLBACK
    assert quest.safety.passed is True, "what we return must itself pass the gate"
    assert quest.title != "Swim Across The Reservoir"
    assert "dive" not in quest.premise.lower()
    assert quest.raw_model_output == UNSAFE_WATER, "the blocked text is kept for the eval harness"


def test_safety_block_is_not_retried_because_blocking_is_not_a_formatting_problem():
    """The headline rule: ``plan_with_retry`` must call the engine exactly once on a safety block.

    Retrying a blocked mission would be asking the model to re-roll a safety
    decision, and would spend a second generation's latency to be told "no"
    again. The engine is handed a second (clean) payload so that a regression
    which *does* retry would visibly succeed and flip ``source`` to MODEL --
    the assertion cannot pass vacuously.
    """
    engine = ScriptedEngine([UNSAFE_WATER, CLEAN])
    quest, outcome = planner.plan_with_retry(make_request(), engine)

    assert engine.calls == 1, f"a safety block must not be re-prompted (prompts={engine.prompts!r})"
    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.safety_passed is False
    assert quest.safety.passed is True


def test_advisory_warnings_do_not_block_the_model_mission():
    """"After dark" is advice, not a veto: warn, keep the model mission, and pass it on.

    A gate that blocked on warnings would fall back on almost every evening
    quest and quietly turn the product into a template generator.
    """
    quest, outcome = planner.plan_quest(make_request(), ScriptedEngine([WARN_ONLY]))

    assert outcome.source is GeneratorSource.MODEL
    assert outcome.safety_passed is True
    assert quest.safety.passed is True
    assert "warn-after-dark" in quest.safety.triggered
    assert quest.safety_notes, "the warning is surfaced to the player"
    assert "after dark" in " ".join(quest.safety_notes).lower()


@pytest.mark.parametrize("mobility", ["avoid_stairs", "step_free"])
def test_mobility_mismatch_blocks_and_is_not_retried(mobility):
    """A request constraint the mission ignores is blocking, and blocking is never re-prompted.

    The player told us they cannot do stairs. Getting a stair mission back
    because the model "meant well" is the failure this prevents.
    """
    engine = ScriptedEngine([UNSAFE_STAIRS, CLEAN])
    quest, outcome = planner.plan_with_retry(make_request(mobility=mobility), engine)

    assert engine.calls == 1
    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.safety_passed is False
    assert "mobility-mismatch" in outcome.fallback_reason
    assert quest.safety.passed is True


# ---------------------------------------------------------------------------
# Retry: parse failures get exactly one stricter nudge, and nothing else does.
# ---------------------------------------------------------------------------


def test_parse_failure_is_retried_exactly_once_and_can_recover():
    """Prose on the first attempt gets one blunter prompt; the model gets a chance to fix it.

    The engine is scripted with a clean payload second, so a regression that
    stopped retrying would return FALLBACK and fail the ``source`` assertion.
    """
    engine = ScriptedEngine([PROSE, CLEAN])
    quest, outcome = planner.plan_with_retry(make_request(), engine)

    assert engine.calls > 1, "a parse failure must be retried"
    assert engine.calls == 2, "one retry, not a loop"
    assert outcome.source is GeneratorSource.MODEL
    assert outcome.parse_ok is True
    assert outcome.parse_reason == "retry:ok"
    assert outcome.warnings == ["recovered on retry"]
    assert quest.title == "Count the Birds"


def test_the_retry_prompt_is_the_original_plus_a_stricter_suffix():
    """The retry reuses the request and appends the nudge -- it must not re-ask differently.

    If the retry prompt were not a superset of the first, the model would be
    answering a different question on the second attempt and the comparison
    would be meaningless.
    """
    engine = ScriptedEngine([PROSE, PROSE])
    planner.plan_with_retry(make_request(), engine)

    assert len(engine.prompts) == 2
    first, second = engine.prompts
    assert second.startswith(first), "the retry prompt is the original prompt plus a suffix"
    assert second.endswith(RETRY_SUFFIX)
    assert RETRY_SUFFIX not in first
    assert first == compose_prompt(make_request())


@pytest.mark.parametrize(
    "retry_text",
    [
        pytest.param(PROSE, id="still-unparseable"),
        pytest.param('{"steps": [{"instruction": "x", "minutes": 5}]}', id="parses-but-no-title"),
        pytest.param('{"title": "T"}', id="parses-but-no-steps"),
        pytest.param(UNSAFE_WATER, id="parses-but-unsafe"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
    ],
)
def test_a_retry_that_still_fails_keeps_the_first_safe_fallback(retry_text):
    """Whatever the retry says, we keep the already-built safe fallback and say why.

    Crucially the unsafe retry case must not be used: the fallback was built
    *before* the retry, so returning it is what guarantees we never hand back a
    blocked mission just because a second sample looked better.
    """
    engine = ScriptedEngine([PROSE, retry_text])
    first_pass, _ = planner.plan_quest(make_request(), ScriptedEngine([PROSE]), seed=7)

    quest, outcome = planner.plan_with_retry(make_request(), engine, seed=7)

    assert engine.calls == 2, "one retry, then stop"
    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.safety_passed is True, "we are returning the safe fallback, not the retry"
    assert quest.safety.passed is True
    assert quest.title == first_pass.title, "the quest is the first attempt's fallback, not the retry"


def test_a_retry_that_produces_no_result_at_all_keeps_the_fallback():
    """An engine that dies mid-retry must not turn into a crash or an unsafe quest."""
    engine = ScriptedEngine([PROSE, None])
    quest, outcome = planner.plan_with_retry(make_request(), engine, seed=7)

    assert engine.calls == 2
    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.fallback_reason == "model output unparseable (unparseable_json)"
    assert quest.safety.passed is True
    assert quest.steps


# ---------------------------------------------------------------------------
# The time budget. This is the constraint a player actually feels.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240])
@pytest.mark.parametrize("steps", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("minutes", [5, 20, 45, 120, 240])
def test_the_quest_always_fits_the_requested_time_budget(budget, steps, minutes):
    """No combination of hostile step durations may produce a mission over budget.

    A model that says "40 minutes" but writes steps totalling 95 is simply
    wrong, and the player is the one who finds out. The planner reconciles the
    two; this sweep is the regression net for that reconciliation.
    """
    payload = {
        "title": "Count the Birds",
        "steps": [{"instruction": f"Step {i}", "minutes": minutes} for i in range(steps)],
    }
    request = make_request(time_budget_minutes=budget)
    quest, outcome = planner.plan_quest(request, ScriptedEngine([json.dumps(payload)]))

    assert outcome.source is GeneratorSource.MODEL, "a valid in-budget mission should not fall back"
    total = sum(s.approx_minutes for s in quest.steps)
    assert total <= budget, f"{steps} steps of {minutes}m overran a {budget}m budget: {total}m"
    assert quest.estimated_minutes <= budget
    assert all(s.approx_minutes >= 1 for s in quest.steps), "no zero-minute steps"
    assert quest.steps, "budget reconciliation must not empty the quest"


@pytest.mark.parametrize("minutes", [5, 20, 45, 120, 240])
def test_the_fallback_quest_also_fits_the_time_budget(minutes):
    """The deterministic generator honours the budget too -- it is the safety net, not an escape hatch.

    ``minutes`` is the engine's (hostile) payload size; the assertion is about
    the returned quest, whatever rung of the ladder produced it.
    """
    payload = {
        "title": "Count the Birds",
        "steps": [{"instruction": f"Step {i}", "minutes": minutes} for i in range(8)],
    }
    request = make_request(time_budget_minutes=15)
    quest, outcome = planner.plan_quest(request, ScriptedEngine([json.dumps(payload)]))
    assert sum(s.approx_minutes for s in quest.steps) <= 15

    fallback_quest, fallback_outcome = planner.plan_quest(
        request, ScriptedEngine([PROSE]), seed=7
    )
    assert fallback_outcome.source is GeneratorSource.FALLBACK
    assert sum(s.approx_minutes for s in fallback_quest.steps) <= 15


@pytest.mark.parametrize("budget", [5, 15, 30, 60])
def test_no_model_at_all_still_yields_a_valid_in_budget_safe_quest(budget):
    """With ``UnavailableEngine`` the player must still get a real mission.

    This is the state the product ships in by default (no weights installed),
    so "still a valid, safe, in-budget quest" is the floor, not a nice-to-have.
    """
    request = make_request(time_budget_minutes=budget)
    quest, outcome = planner.plan_quest(request, UnavailableEngine(), seed=7)

    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.parse_ok is False
    assert outcome.safety_passed is True
    assert outcome.model_id is None
    assert outcome.fallback_reason, "an unavailable engine must be reported honestly"

    assert quest.title and quest.hook
    assert quest.steps and all(s.instruction for s in quest.steps)
    assert sum(s.approx_minutes for s in quest.steps) <= budget
    assert quest.estimated_minutes <= budget
    assert quest.safety.passed is True
    assert quest.verification.subject, "verification is never left blank"
    assert quest.verification.must_include


def test_plan_with_retry_with_an_unavailable_engine_is_still_safe_and_in_budget():
    """The retry wrapper degrades to the same guarantee when there is no model.

    A dead engine is not a parse failure, so there is nothing to re-prompt, but
    the caller must still receive a usable mission and an honest outcome.
    """
    engine = UnavailableEngine()
    request = make_request()
    quest, outcome = planner.plan_with_retry(request, engine, seed=7)

    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.safety_passed is True
    assert outcome.fallback_reason == "no engine registered"
    assert quest.safety.passed is True
    assert sum(s.approx_minutes for s in quest.steps) <= request.time_budget_minutes


# ---------------------------------------------------------------------------
# PlanOutcome consistency: the object the eval harness and the API serialise.
# ---------------------------------------------------------------------------


def test_outcome_fields_are_consistent_on_the_model_path():
    """On the model path every provenance field agrees, and nothing is left blank."""
    engine = ScriptedEngine([CLEAN])
    quest, outcome = planner.plan_quest(make_request(), engine)

    assert outcome.source is GeneratorSource.MODEL
    assert outcome.model_id == "fake-gemma"
    assert outcome.parse_reason
    assert outcome.parse_reason != "retry:ok"
    assert outcome.safety_passed is True
    assert outcome.fallback_reason == ""
    assert isinstance(outcome.latency_ms, int) and outcome.latency_ms >= 0
    assert outcome.latency_ms == quest.generation_ms
    assert outcome.warnings == []
    assert quest.model_id == outcome.model_id


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(PROSE, "parse", id="parse-failure"),
        pytest.param(UNSAFE_WATER, "safety", id="safety-block"),
        pytest.param(None, "unavailable", id="engine-returned-nothing"),
    ],
)
def test_outcome_fields_are_consistent_on_every_fallback_path(text, expected):
    """Each fallback rung populates its own fields -- and only its own.

    The eval suite publishes a fallback-trigger rate from these fields, so a
    path that quietly inherited another path's values would publish a number
    that does not describe what happened.
    """
    engine = ScriptedEngine([text])
    quest, outcome = planner.plan_quest(make_request(), engine, seed=7)

    assert outcome.source is GeneratorSource.FALLBACK
    assert isinstance(outcome.latency_ms, int) and outcome.latency_ms >= 0
    assert outcome.latency_ms == quest.generation_ms
    assert outcome.fallback_reason, "a fallback must always say why"
    assert quest.source is GeneratorSource.FALLBACK
    assert quest.safety.passed is True

    if expected == "parse":
        assert outcome.parse_ok is False
        assert outcome.parse_reason == "unparseable_json"
        assert outcome.safety_passed is True, "the gate never ran; do not claim it passed"
        assert outcome.model_id == "fake-gemma"
    elif expected == "safety":
        assert outcome.parse_ok is True
        assert outcome.parse_reason == "ok"
        assert outcome.safety_passed is False
        assert outcome.model_id == "fake-gemma"
    else:
        assert outcome.parse_ok is False
        assert outcome.parse_reason == "", "nothing was generated, so nothing was parsed"
        assert outcome.safety_passed is True
        assert outcome.model_id is None


def test_outcome_recovery_fields_distinguish_a_retry_from_a_first_pass():
    """A recovered quest is traceable: it is tagged so the eval suite can count retries."""
    outcome = planner.plan_with_retry(make_request(), ScriptedEngine([PROSE, CLEAN]))[1]

    assert outcome.source is GeneratorSource.MODEL
    assert outcome.parse_reason.startswith("retry:")
    assert outcome.warnings == ["recovered on retry"]
    assert outcome.fallback_reason == ""
    assert outcome.safety_passed is True


def test_latency_is_never_negative_or_non_integer_on_every_path():
    """``latency_ms`` is typed ``int`` in the dataclass and feeds the API's ``generation`` block."""
    outcomes = [
        planner.plan_quest(make_request(), ScriptedEngine([CLEAN]))[1],
        planner.plan_quest(make_request(), ScriptedEngine([PROSE]))[1],
        planner.plan_quest(make_request(), ScriptedEngine([UNSAFE_WATER]))[1],
        planner.plan_quest(make_request(), UnavailableEngine())[1],
        planner.plan_with_retry(make_request(), ScriptedEngine([PROSE, CLEAN]))[1],
        planner.plan_with_retry(make_request(), UnavailableEngine())[1],
    ]
    assert len(outcomes) == 6
    for outcome in outcomes:
        assert isinstance(outcome.latency_ms, int), f"{outcome.source} reported a non-int latency"
        assert outcome.latency_ms >= 0


def test_the_seed_reaches_the_deterministic_generator():
    """The same seed must produce the same mission, or eval runs are not comparable."""
    request = make_request()
    first, _ = planner.plan_quest(request, UnavailableEngine(), seed=99)
    second, _ = planner.plan_quest(request, UnavailableEngine(), seed=99)
    third, _ = planner.plan_quest(request, UnavailableEngine(), seed=100)

    stable_fields = ("title", "hook", "premise", "steps", "tags")
    assert [getattr(first, f) for f in stable_fields] == [getattr(second, f) for f in stable_fields]
    assert first.title != third.title or first.steps != third.steps, "a different seed should explore"


# ---------------------------------------------------------------------------
# Small-model containment: coerce and clamp, never trust.
# ---------------------------------------------------------------------------


def test_hostile_field_types_are_coerced_rather_than_trusted():
    """Numbers where strings belong, absurd values, and nested junk must not crash or leak through."""
    payload = {
        "title": "Count the Birds",
        "steps": [
            {"instruction": "Count them", "minutes": "20 minutes"},
            "Watch the sky for an hour",
            {"detail": "step with no instruction, dropped"},
            {"instruction": "Sketch the view", "minutes": "not a number"},
        ],
        "estimated_minutes": "about half an hour",
        "xp_reward": 10_000_000,
        "tags": {"not": "a list"},
        "verification": {"subject": "ignored nested object", "must_include": ["also ignored"]},
    }
    request = make_request()
    quest, outcome = planner.plan_quest(request, ScriptedEngine([json.dumps(payload)]))

    assert outcome.source is GeneratorSource.MODEL
    assert quest.steps[0].approx_minutes == 20, "'20 minutes' is coerced to an int"
    assert quest.steps[1].approx_minutes == 5, "a bare string step gets the default duration"
    assert all("no instruction" not in s.instruction for s in quest.steps)
    assert 5 <= quest.xp_reward <= 500, f"xp is clamped, got {quest.xp_reward}"
    assert isinstance(quest.tags, list)
    assert isinstance(quest.verification.subject, str) and quest.verification.subject
    assert sum(s.approx_minutes for s in quest.steps) <= request.time_budget_minutes


def test_model_output_is_capped_so_one_field_cannot_bloat_the_payload():
    """A model that ignores ``max_len`` gets truncated, and raw output is bounded for storage."""
    payload = {
        "title": "Count the Birds",
        "hook": "x" * 9000,
        "premise": "y" * 9000,
        "steps": [{"instruction": "Count them", "minutes": 5, "detail": "z" * 9000}],
    }
    quest, _ = planner.plan_quest(make_request(), ScriptedEngine([json.dumps(payload)]))

    assert len(quest.hook) == 200
    assert len(quest.premise) == 500
    assert len(quest.steps[0].detail) == 300
    assert len(quest.raw_model_output) == 4000


@pytest.mark.parametrize("count", [9, 12, 20])
def test_a_runaway_step_list_is_truncated_to_eight(count):
    """Eight steps is the cap; a 20-step mission would not fit any real outing."""
    payload = {"title": "Count the Birds", "steps": [{"instruction": f"S{i}", "minutes": 1} for i in range(count)]}
    quest, outcome = planner.plan_quest(make_request(), ScriptedEngine([json.dumps(payload)]))

    assert outcome.source is GeneratorSource.MODEL
    assert len(quest.steps) == 8
    assert [s.order for s in quest.steps] == list(range(1, 9))