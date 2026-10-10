"""Data-contract tests for ``app.models.schemas``.

This module is the integration seam of the project: every other module imports
its types from here, so a silent change to it is a silent change to the claims
the README makes. These tests pin the properties that back the public safety
story rather than incidental getters.

The three promises this file exists to hold the schema to:

1. **The honesty invariant has a data-layer half.** ``ConfidenceLevel`` has six
   members, but only two of them -- ``VERIFIED`` and ``LIKELY`` -- report
   ``counts_as_verified``. A verdict built at any other level is, by definition,
   not something the UI may badge as "AI-verified". ``HEURISTIC`` in particular
   is the level the button-press path produces, so widening
   ``counts_as_verified`` by even one member would silently promote a
   heuristic result to a verified one in every downstream consumer.

2. **``Verdict.is_ai_verified`` defaults to False.** The flag is the single
   field the UI branches on for the verified badge, so a fresh verdict must
   claim nothing until a model has actually earned it.

3. **No precise location, ever.** ``Evidence.location_label`` is a coarse
   human-typed string ("the park on 5th"). It must stay a string: not a float,
   not a lat/lon pair, and no model in this module may grow a coordinate field.
   The front end already truncates it to 120 chars (``web/app.js``), matching
   the schema.

Also pinned here: enum round-tripping and rejection, which fields are required
versus defaulted, step ordering, and the lengths the fallback generator relies
on.

Everything runs in-process against pure pydantic models. No weights are
loaded, no network is touched, and the module finishes in milliseconds.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from app.models.schemas import (
    ConfidenceLevel,
    EnergyLevel,
    Evidence,
    GeneratorSource,
    LocationType,
    Mobility,
    PartyType,
    PlayerState,
    Quest,
    QuestRequest,
    QuestStatus,
    QuestStep,
    SafetyFinding,
    SafetyReport,
    SessionSummary,
    VerificationSpec,
    Verdict,
    Weather,
    XpEvent,
)


ALL_ENUMS = (
    PartyType,
    Mobility,
    EnergyLevel,
    LocationType,
    Weather,
    QuestStatus,
    ConfidenceLevel,
    GeneratorSource,
)


def make_step(**overrides) -> QuestStep:
    """One playable step. `order` is filled in if the caller omits it."""
    data = {"order": 1, "instruction": "Walk one block and look at a wall.", "approx_minutes": 5}
    data.update(overrides)
    return QuestStep(**data)


def make_spec(**overrides) -> VerificationSpec:
    data = {"subject": "A close photograph of a doorbell."}
    data.update(overrides)
    return VerificationSpec(**data)


def make_quest(**overrides) -> Quest:
    """A minimal but complete quest: every required field, nothing else."""
    data = {
        "title": "Count the Doorbells",
        "steps": [make_step()],
        "estimated_minutes": 5,
        "xp_reward": 100,
        "verification": make_spec(),
    }
    data.update(overrides)
    return Quest(**data)


def required_fields(model: type[BaseModel]) -> set[str]:
    return {name for name, field in model.model_fields.items() if field.is_required()}


# ---------------------------------------------------------------------------
# Enums -- round-trip by value, reject anything else
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enum_cls", ALL_ENUMS, ids=lambda e: e.__name__)
def test_every_member_round_trips_through_its_own_value(enum_cls):
    """`Enum(value)` must return the identical member, or a persisted record
    written by one release will not read back under the next."""
    for member in enum_cls:
        assert enum_cls(member.value) is member


@pytest.mark.parametrize("enum_cls", ALL_ENUMS, ids=lambda e: e.__name__)
def test_member_values_are_unique(enum_cls):
    """A duplicate value would make two distinct states indistinguishable."""
    values = [m.value for m in enum_cls]
    assert len(values) == len(set(values))


@pytest.mark.parametrize("enum_cls", ALL_ENUMS, ids=lambda e: e.__name__)
def test_member_names_are_screaming_snake_case_matches_of_their_values(enum_cls):
    """Values are what travel over the wire; the name is the Python-facing
    spelling. `VERIFIED = "verified"` keeps the two obvious to each other."""
    for member in enum_cls:
        assert member.name == member.value.upper(), member


@pytest.mark.parametrize(
    "enum_cls",
    ALL_ENUMS,
    ids=lambda e: e.__name__,
)
def test_unknown_values_are_rejected_not_silently_defaulted(enum_cls):
    """An unrecognised label must raise. Defaulting it would be a quiet lie:
    a weather of "monsoon" becoming UNKNOWN, or worse, a verdict becoming
    VERIFIED."""
    with pytest.raises(ValueError):
        enum_cls("definitely-not-a-real-value")


@pytest.mark.parametrize("enum_cls", ALL_ENUMS, ids=lambda e: e.__name__)
def test_member_names_are_not_accepted_as_values(enum_cls):
    """`"VERIFIED"` is not `"verified"`. Case-insensitive acceptance here
    would make the wire format looser than the contract claims."""
    for member in enum_cls:
        if member.name == member.value:  # e.g. a member literally named "solo"
            continue
        with pytest.raises(ValueError):
            enum_cls(member.name)


def test_confidence_levels_are_the_exact_six_the_project_documents():
    assert [m.value for m in ConfidenceLevel] == [
        "verified",
        "likely",
        "uncertain",
        "rejected",
        "heuristic",
        "unavailable",
    ]


def test_generator_source_distinguishes_a_model_from_everything_else():
    """`MODEL` exists so a fallback artifact can be told from a generated one.
    Losing it would let a heuristic mission be attributed to the LLM."""
    assert GeneratorSource.MODEL.value == "model"
    assert GeneratorSource.FALLBACK.value == "fallback"
    assert GeneratorSource.CACHED.value == "cached"


# ---------------------------------------------------------------------------
# ConfidenceLevel.counts_as_verified -- the honesty invariant, data layer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [ConfidenceLevel.VERIFIED, ConfidenceLevel.LIKELY])
def test_only_verified_and_likely_count_as_verified(level):
    """The whole point of the flag. These two are the only levels a model can
    earn, and every downstream XP rule keys off this property."""
    assert level.counts_as_verified is True


@pytest.mark.parametrize(
    "level",
    [
        ConfidenceLevel.UNCERTAIN,
        ConfidenceLevel.REJECTED,
        ConfidenceLevel.HEURISTIC,
        ConfidenceLevel.UNAVAILABLE,
    ],
)
def test_nothing_else_counts_as_verified(level):
    """Not UNCERTAIN, not REJECTED, not HEURISTIC, not UNAVAILABLE.

    HEURISTIC is the button-press path and UNAVAILABLE is the no-model path;
    both are rules-only results, and both must stay out of this set.
    """
    assert level.counts_as_verified is False


def test_counts_as_verified_is_a_property_not_a_method_or_a_field():
    """Callers write `level.counts_as_verified`, so it must not be a method --
    `level.counts_as_verified()` would return a bound method, which is truthy,
    and would quietly verify everything."""
    assert isinstance(ConfidenceLevel.VERIFIED.counts_as_verified, bool)
    with pytest.raises(TypeError):
        ConfidenceLevel.VERIFIED.counts_as_verified()  # type: ignore[operator]


def test_counts_as_verified_is_not_serialised_onto_the_enum():
    """It is a derived property. It must not leak into a payload as a stored
    field that could later disagree with `level`."""
    assert not hasattr(ConfidenceLevel.VERIFIED, "model_fields")
    dumped = json.dumps({"level": ConfidenceLevel.VERIFIED})
    assert dumped == '{"level": "verified"}'


# ---------------------------------------------------------------------------
# Verdict.is_ai_verified -- defaults to False
# ---------------------------------------------------------------------------


def test_is_ai_verified_defaults_to_false_when_nothing_ran():
    """A verdict nobody earned must not claim verification. The badge in the
    UI is driven by this field alone."""
    verdict = Verdict(quest_id="q1", level=ConfidenceLevel.VERIFIED)

    assert verdict.is_ai_verified is False
    assert verdict.confidence == 0.0
    assert verdict.xp_awarded == 0
    assert verdict.source is GeneratorSource.FALLBACK
    assert verdict.model_id is None
    assert verdict.needs_better_photo is False
    assert verdict.summary == ""


def test_is_ai_verified_is_a_real_bool_not_a_truthy_string():
    """`main.py` branches on this with `if verdict.is_ai_verified:` and stores
    it as a count. A string "false" would be truthy and inflate verified_count.
    """
    verdict = Verdict(quest_id="q1", level=ConfidenceLevel.LIKELY, is_ai_verified=False)

    assert verdict.is_ai_verified is False


def test_verdict_requires_a_quest_id_and_a_level():
    """Both are required: a verdict that cannot be attributed to a mission, or
    that names no confidence level, has no meaning."""
    assert required_fields(Verdict) == {"quest_id", "level"}

    with pytest.raises(ValidationError) as missing_quest:
        Verdict(level=ConfidenceLevel.VERIFIED)
    assert [e["loc"] for e in missing_quest.value.errors()] == [("quest_id",)]

    with pytest.raises(ValidationError) as missing_level:
        Verdict(quest_id="q1")
    assert [e["loc"] for e in missing_level.value.errors()] == [("level",)]


def test_verdict_rejects_an_unknown_level():
    """`"probably"` is not a level. It must not become UNCERTAIN by accident
    and it must not become VERIFIED either."""
    with pytest.raises(ValidationError):
        Verdict(quest_id="q1", level="probably")


@pytest.mark.parametrize("level", [m.value for m in ConfidenceLevel])
def test_verdict_accepts_every_level_by_value(level):
    verdict = Verdict(quest_id="q1", level=level)
    assert verdict.level is ConfidenceLevel(level)


# ---------------------------------------------------------------------------
# Evidence.location_label -- a human label, never coordinates
# ---------------------------------------------------------------------------


def test_location_label_is_typed_as_a_string():
    """This is the field's contract in one assertion. Anything numeric would
    mean the app started collecting precision it promises not to."""
    field = Evidence.model_fields["location_label"]

    assert field.annotation is str
    assert not field.is_required()
    assert Evidence().location_label == ""


def test_location_label_json_schema_declares_a_bounded_string():
    """The generated OpenAPI schema is what a client sees; a number here would
    be an API contract change, not just an internal one."""
    prop = Evidence.model_json_schema()["properties"]["location_label"]

    assert prop["type"] == "string"
    assert prop["maxLength"] == 120
    assert prop["default"] == ""


def test_evidence_has_no_coordinate_field_at_all():
    """The strongest form of the promise: not "the label is a string" but
    "there is nowhere to put a latitude". Checked against the schema, so it
    holds for the wire format and not just the Python object."""
    props = Evidence.model_json_schema()["properties"]

    assert set(props) == {
        "photo_data_url",
        "note",
        "location_label",
        "elapsed_minutes",
    }
    blob = json.dumps(Evidence.model_json_schema()).lower()
    for word in ("lat", "lon", "lng", "gps", "geohash", "coordinate", "accuracy"):
        assert word not in blob, f"Evidence schema grew a {word!r} field"


def test_no_model_in_the_schema_module_exposes_a_coordinate_field():
    """Guards the whole seam, not just `Evidence`: a lat/lon added anywhere in
    `schemas.py` would be persisted by `store.py` on the way through."""
    import app.models.schemas as schemas_module

    checked = 0
    for name in dir(schemas_module):
        obj = getattr(schemas_module, name)
        if not (isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel):
            continue
        checked += 1
        blob = json.dumps(obj.model_json_schema()).lower()
        for word in ("gps", "geohash", "coordinate", "latitude", "longitude"):
            assert word not in blob, f"{name} grew a {word!r} field"
    assert checked >= 12, f"only inspected {checked} models; the sweep is too narrow"


@pytest.mark.parametrize(
    "value",
    [41.8, -87.6, 41, {"lat": 41.8, "lon": -87.6}, [41.8, -87.6], True],
)
def test_location_label_refuses_numeric_or_pairwise_coordinates(value):
    """A float or a [lat, lon] pair is a coordinate by any reading. Pydantic
    rejects them for a `str` field in non-strict mode only because none of them
    is a string, and that is the property being pinned."""
    with pytest.raises(ValidationError):
        Evidence(location_label=value)


def test_a_coordinate_shaped_string_is_stored_verbatim_not_parsed():
    """If someone types coordinates, the app stores the text they typed. It
    does not silently normalise them into a machine-readable position."""
    evidence = Evidence(location_label="41.8781, -87.6298")

    assert evidence.location_label == "41.8781, -87.6298"
    assert isinstance(evidence.location_label, str)
    # And it round-trips through JSON unchanged, with no numeric shadow field.
    dumped = evidence.model_dump()
    assert dumped["location_label"] == "41.8781, -87.6298"
    assert set(dumped) == {
        "photo_data_url",
        "note",
        "location_label",
        "elapsed_minutes",
    }


def test_an_oversized_location_label_is_rejected():
    """Matches the front end's own `.slice(0, 120)` on `#inp-place`, so the
    two ends cannot drift apart silently."""
    assert len(Evidence(location_label="x" * 120).location_label) == 120
    with pytest.raises(ValidationError):
        Evidence(location_label="x" * 121)


def test_elapsed_minutes_is_bounded_and_never_negative():
    """Outdoor minutes are the headline "Touch Grass" number; a negative value
    could subtract from it and an absurd one could inflate it."""
    assert Evidence(elapsed_minutes=0).elapsed_minutes == 0
    assert Evidence(elapsed_minutes=1440).elapsed_minutes == 1440

    with pytest.raises(ValidationError):
        Evidence(elapsed_minutes=-1)
    with pytest.raises(ValidationError):
        Evidence(elapsed_minutes=1441)


def test_evidence_has_no_required_fields():
    """Every field of evidence is optional, so a player who submits nothing but
    a button press still produces a parsable `Evidence` and gets a real
    (unverified) verdict rather than a 422."""
    assert required_fields(Evidence) == set()
    bare = Evidence()
    assert bare.photo_data_url is None
    assert bare.note == ""
    assert bare.elapsed_minutes == 0


# ---------------------------------------------------------------------------
# QuestRequest -- the feasibility envelope
# ---------------------------------------------------------------------------


def test_quest_request_requires_only_a_time_budget():
    """Everything else has a sane default, so the smallest legal request is a
    single number."""
    assert required_fields(QuestRequest) == {"time_budget_minutes"}

    request = QuestRequest(time_budget_minutes=45)
    assert request.interests == []
    assert request.avoid == []
    assert request.context_note == ""
    assert request.seed is None
    assert request.party is PartyType.SOLO
    assert request.mobility is Mobility.UNRESTRICTED
    assert request.energy is EnergyLevel.MEDIUM
    assert request.location_type is LocationType.UNKNOWN
    assert request.weather is Weather.UNKNOWN
    assert request.max_radius_m == 800


@pytest.mark.parametrize("budget", [4, 241, 0, -30])
def test_time_budget_outside_five_to_two_forty_is_rejected(budget):
    """The generator promises a mission that fits the budget; a budget under 5
    or over 240 minutes cannot produce one."""
    with pytest.raises(ValidationError):
        QuestRequest(time_budget_minutes=budget)


@pytest.mark.parametrize("budget", [5, 240])
def test_the_budget_bounds_themselves_are_accepted(budget):
    assert QuestRequest(time_budget_minutes=budget).time_budget_minutes == budget


def test_interests_are_trimmed_lowercased_and_blanks_dropped():
    """Normalisation happens on the way in, so the prompt builder cannot be
    handed a tag with stray whitespace. `tests/test_api.py` pins the other
    half -- that the generator sees the cleaned form."""
    request = QuestRequest(time_budget_minutes=60, interests=["  Birds ", "", "NATURE", "   "])

    assert request.interests == ["birds", "nature"]


def test_interests_are_capped_at_eight():
    with pytest.raises(ValidationError):
        QuestRequest(time_budget_minutes=60, interests=[f"tag{i}" for i in range(9)])


def test_more_than_eight_interests_are_rejected_even_if_most_are_blank():
    """The cap is checked before the trim, so a client cannot smuggle an
    unbounded list past it with empty strings."""
    with pytest.raises(ValidationError):
        QuestRequest(time_budget_minutes=60, interests=["  "] * 9)


def test_avoid_is_capped_and_normalised_too():
    request = QuestRequest(time_budget_minutes=60, avoid=["  Crowds ", "", "TRAFFIC"])

    assert request.avoid == ["crowds", "traffic"]
    with pytest.raises(ValidationError):
        QuestRequest(time_budget_minutes=60, avoid=[f"x{i}" for i in range(9)])


@pytest.mark.parametrize("radius", [49, 5001])
def test_radius_bounds_are_enforced(radius):
    with pytest.raises(ValidationError):
        QuestRequest(time_budget_minutes=60, max_radius_m=radius)


def test_an_oversized_context_note_is_rejected():
    assert len(QuestRequest(time_budget_minutes=60, context_note="x" * 400).context_note) == 400
    with pytest.raises(ValidationError):
        QuestRequest(time_budget_minutes=60, context_note="x" * 401)


def test_an_unknown_enum_value_in_a_request_is_rejected():
    with pytest.raises(ValidationError):
        QuestRequest(time_budget_minutes=60, weather="monsoon")


def test_mobility_is_carried_through_verbatim():
    """Step-free routing is an accessibility setting; coercing it to
    UNRESTRICTED would be the worst possible default."""
    request = QuestRequest(time_budget_minutes=60, mobility=Mobility.STEP_FREE)

    assert request.mobility is Mobility.STEP_FREE
    assert QuestRequest(time_budget_minutes=60, mobility="step_free").mobility is Mobility.STEP_FREE


# ---------------------------------------------------------------------------
# Quest / QuestStep -- a playable mission or a ValidationError
# ---------------------------------------------------------------------------


def test_a_quest_with_no_steps_is_rejected():
    """`steps` has `min_length=1`. A mission the player cannot start must not
    survive construction; `test_api.py` pins that the HTTP layer never returns
    one either."""
    with pytest.raises(ValidationError) as err:
        make_quest(steps=[])

    assert err.value.errors()[0]["type"] == "too_short"


def test_a_quest_with_more_than_eight_steps_is_rejected():
    with pytest.raises(ValidationError) as err:
        make_quest(steps=[make_step(order=i + 1) for i in range(9)])

    assert err.value.errors()[0]["type"] == "too_long"


def test_a_step_missing_its_minutes_is_rejected_not_defaulted():
    """`approx_minutes` is required. Defaulting it would let the time-budget
    check in `planner.py` sum a step that silently costs nothing."""
    with pytest.raises(ValidationError) as err:
        QuestStep(order=1, instruction="Walk to the corner.")

    assert [e["loc"] for e in err.value.errors()] == [("approx_minutes",)]


def test_a_step_does_not_accept_a_minutes_alias():
    """The fallback generator and the prompt speak `approx_minutes`. Accepting
    a second spelling would let two durations exist for one step depending on
    which writer produced it."""
    with pytest.raises(ValidationError):
        QuestStep(order=1, instruction="Walk to the corner.", minutes=5)


def test_a_step_with_no_instruction_or_zero_minutes_is_rejected():
    with pytest.raises(ValidationError):
        make_step(instruction="")
    with pytest.raises(ValidationError):
        make_step(approx_minutes=0)
    with pytest.raises(ValidationError):
        make_step(approx_minutes=241)


def test_step_order_is_renumbered_from_one_on_construction():
    """A model that emits `order: [9, 4, 7]` must not produce a quest the UI
    renders out of sequence. The validator overwrites whatever arrived."""
    quest = make_quest(
        steps=[
            make_step(order=9, instruction="First."),
            make_step(order=4, instruction="Second."),
            make_step(order=7, instruction="Third."),
        ]
    )

    assert [s.order for s in quest.steps] == [1, 2, 3]
    assert [s.instruction for s in quest.steps] == ["First.", "Second.", "Third."]


def test_step_detail_defaults_to_empty():
    """The optional screen-free prompt. Absent is normal, not an error."""
    assert make_step().detail == ""


def test_a_quest_requires_the_fields_that_make_it_playable():
    """title, steps, estimated_minutes, xp_reward and verification have no
    defaults; everything else does."""
    assert required_fields(Quest) == {
        "title",
        "steps",
        "estimated_minutes",
        "xp_reward",
        "verification",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", ""),
        ("title", "x" * 81),
        ("estimated_minutes", 0),
        ("estimated_minutes", 241),
        ("xp_reward", -1),
        ("xp_reward", 501),
    ],
)
def test_quest_field_bounds_are_enforced(field, value):
    with pytest.raises(ValidationError):
        make_quest(**{field: value})


def test_a_quest_needs_a_verification_spec_with_a_subject():
    """The spec is what makes model verification mission-specific; an empty
    subject would leave nothing to judge the photo against."""
    with pytest.raises(ValidationError):
        make_quest(verification={})

    spec = make_spec(subject="A red door.", must_include=["a door", "red paint"])
    assert spec.subject == "A red door."
    assert spec.must_include == ["a door", "red paint"]


def test_verification_spec_lists_default_to_empty_and_are_capped():
    spec = make_spec()
    assert spec.must_include == []
    assert spec.accept_hints == []
    assert spec.reject_if == []
    assert spec.ambiguous_cases == []

    with pytest.raises(ValidationError):
        make_spec(must_include=[f"c{i}" for i in range(7)])


def test_quest_defaults_are_conservative():
    """With nothing said about provenance, the quest must look like a
    fallback, not like something a model produced."""
    quest = make_quest()

    assert quest.source is GeneratorSource.FALLBACK
    assert quest.model_id is None
    assert quest.raw_model_output is None
    assert quest.generation_ms == 0
    assert quest.hook == ""
    assert quest.premise == ""
    assert quest.tags == []
    assert quest.safety_notes == []
    assert quest.safety.passed is True
    assert quest.safety.findings == []
    assert quest.safety.triggered == []


def test_a_fresh_quest_gets_a_short_hex_id():
    """Ids are 12 hex chars and are the key the API scopes verification by, so
    they must be unique and url-safe."""
    ids = {make_quest().id for _ in range(20)}

    assert len(ids) == 20, "quest ids collided"
    for quest_id in ids:
        assert len(quest_id) == 12
        assert all(c in "0123456789abcdef" for c in quest_id)


def test_quest_tags_are_capped_at_eight():
    assert len(make_quest(tags=[f"t{i}" for i in range(8)]).tags) == 8
    with pytest.raises(ValidationError):
        make_quest(tags=[f"t{i}" for i in range(9)])


def test_a_safety_block_survives_on_the_quest():
    """`planner.py` swaps the mission and records why. The report on the
    quest must keep `passed=False` and the rule that fired."""
    quest = make_quest(
        safety=SafetyReport(
            passed=False,
            findings=[
                SafetyFinding(rule="open_water", severity="blocking", message="Do not enter water.")
            ],
            triggered=["open_water"],
        )
    )

    assert quest.safety.passed is False
    assert quest.safety.triggered == ["open_water"]
    assert quest.safety.findings[0].severity == "blocking"


def test_safety_finding_rejects_a_severity_outside_the_literal():
    """The three severities are a closed set the safety gate branches on."""
    for severity in ("info", "warning", "blocking"):
        assert SafetyFinding(rule="r", severity=severity, message="m").severity == severity

    with pytest.raises(ValidationError):
        SafetyFinding(rule="r", severity="catastrophic", message="m")


def test_safety_report_requires_only_passed():
    assert required_fields(SafetyReport) == {"passed"}
    assert SafetyReport(passed=True).findings == []


# ---------------------------------------------------------------------------
# Verdict bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_outside_zero_to_one_is_rejected(confidence):
    with pytest.raises(ValidationError):
        Verdict(quest_id="q1", level=ConfidenceLevel.VERIFIED, confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_confidence_bounds_themselves_are_accepted(confidence):
    verdict = Verdict(quest_id="q1", level=ConfidenceLevel.LIKELY, confidence=confidence)
    assert verdict.confidence == confidence


def test_a_nan_confidence_is_rejected():
    """`coerce_float` already guards NaN in `parsing.py`; the schema must not
    let one slip in from anywhere else."""
    with pytest.raises(ValidationError):
        Verdict(quest_id="q1", level=ConfidenceLevel.VERIFIED, confidence=float("nan"))


def test_verdict_lists_are_capped_and_summary_is_bounded():
    assert len(Verdict(quest_id="q", level="likely", matched=list("abcdef")).matched) == 6
    with pytest.raises(ValidationError):
        Verdict(quest_id="q", level="likely", matched=list("abcdefg"))
    with pytest.raises(ValidationError):
        Verdict(quest_id="q", level="likely", missing=list("abcdefg"))
    with pytest.raises(ValidationError):
        Verdict(quest_id="q", level="likely", summary="x" * 301)


# ---------------------------------------------------------------------------
# Serialisation -- what actually crosses the wire
# ---------------------------------------------------------------------------


def test_verdict_round_trips_through_json_without_losing_its_enum():
    """`store.py` persists players as JSON. A verdict written by one process
    must read back as the same `ConfidenceLevel` member, not a bare string."""
    verdict = Verdict(
        quest_id="q1",
        level=ConfidenceLevel.LIKELY,
        confidence=0.72,
        matched=["doorbell"],
        source=GeneratorSource.MODEL,
        model_id="fake/4b",
    )

    payload = json.loads(verdict.model_dump_json())
    assert payload["level"] == "likely"
    assert payload["source"] == "model"

    restored = Verdict.model_validate_json(verdict.model_dump_json())
    assert restored.level is ConfidenceLevel.LIKELY
    assert restored.source is GeneratorSource.MODEL
    assert restored.matched == ["doorbell"]
    assert restored.confidence == pytest.approx(0.72)


def test_a_persisted_verdict_with_a_bad_level_is_rejected_on_load():
    """A corrupt or hand-edited save must fail loudly rather than resurrect an
    unknown level that some caller would treat as a pass."""
    with pytest.raises(ValidationError):
        Verdict.model_validate_json(json.dumps({"quest_id": "q1", "level": "LIKELY"}))
    with pytest.raises(ValidationError):
        Verdict.model_validate_json(json.dumps({"quest_id": "q1", "level": "probably"}))


def test_player_state_round_trips_a_whole_quest():
    """`state` is saved and reloaded per player, so nesting must survive."""
    state = PlayerState(
        player_id="p1",
        xp=340,
        quests=[make_quest()],
        history=[XpEvent(amount=100, reason="verified: Count the Doorbells", quest_id="q")],
    )

    restored = PlayerState.model_validate_json(state.model_dump_json())

    assert restored.player_id == "p1"
    assert restored.xp == 340
    assert isinstance(restored.quests[0], Quest)
    assert restored.quests[0].steps[0].order == 1
    assert restored.quests[0].verification.subject
    assert restored.history[0].amount == 100


# ---------------------------------------------------------------------------
# PlayerState and SessionSummary
# ---------------------------------------------------------------------------


def test_a_new_player_starts_at_level_one_with_nothing_banked():
    """No XP, no verified quests, and separate counters per confidence level
    so the UI cannot present a heuristic run as a verified one."""
    state = PlayerState()

    assert state.level == 1
    assert state.xp == 0
    assert state.quests_completed == 0
    assert state.minutes_outdoors == 0
    assert state.verified_count == 0
    assert state.uncertain_count == 0
    assert state.rejected_count == 0
    assert state.streak_days == 0
    assert state.quests == []
    assert state.history == []
    assert isinstance(state.created_at, float)
    assert isinstance(state.updated_at, float)


def test_each_player_gets_a_distinct_id():
    ids = {PlayerState().player_id for _ in range(20)}
    assert len(ids) == 20


def test_the_xp_ledger_is_capped_at_two_hundred_entries():
    """An unbounded ledger would grow a player's JSON file forever."""
    with pytest.raises(ValidationError):
        PlayerState(history=[XpEvent(amount=1, reason="r") for _ in range(201)])


def test_session_summary_requires_every_number_it_displays():
    """All nine fields are required. A summary with a missing counter would
    render as an honest-looking blank rather than a visible gap."""
    assert required_fields(SessionSummary) == {
        "player_id",
        "xp",
        "level",
        "xp_to_next_level",
        "minutes_outdoors",
        "verified_count",
        "uncertain_count",
        "rejected_count",
        "screen_time_saved_seconds",
    }

    with pytest.raises(ValidationError) as err:
        SessionSummary()
    assert len(err.value.errors()) == 9


def test_session_summary_counts_stay_independent():
    """`main.py` increments exactly one of verified/uncertain/rejected per
    verdict. Separate fields are what keep the three totals from drifting."""
    summary = SessionSummary(
        player_id="p1",
        xp=500,
        level=4,
        xp_to_next_level=250,
        minutes_outdoors=90,
        verified_count=3,
        uncertain_count=2,
        rejected_count=1,
        screen_time_saved_seconds=5400,
    )

    assert (summary.verified_count, summary.uncertain_count, summary.rejected_count) == (3, 2, 1)
    assert summary.minutes_outdoors == 90
    assert summary.screen_time_saved_seconds == 90 * 60


# ---------------------------------------------------------------------------
# QuestStatus -- the one enum nothing in the request path uses yet
# ---------------------------------------------------------------------------


def test_quest_status_values_are_stable():
    """These strings are what the front end switches on; renaming one silently
    changes client behaviour."""
    assert {s.value for s in QuestStatus} == {"active", "completed", "abandoned"}


def test_energy_levels_are_ordered_by_name_as_the_prompt_expects():
    assert [e.value for e in EnergyLevel] == ["low", "medium", "high"]


def test_location_types_cover_the_rain_contingency():
    """`INDOOR_MALL` is documented as the rain plan, so its absence would
    remove the project's only indoor fallback."""
    assert LocationType.INDOOR_MALL.value == "indoor_mall"
    assert LocationType.UNKNOWN.value == "unknown"


def test_mobility_distinguishes_stairs_from_step_free():
    """These are different accessibility needs; collapsing them would route a
    wheelchair user up a staircase."""
    assert Mobility.AVOID_STAIRS.value == "avoid_stairs"
    assert Mobility.STEP_FREE.value == "step_free"
    assert Mobility.STEP_FREE is not Mobility.AVOID_STAIRS


def test_party_types_cover_the_dog_and_the_children():
    assert {p.value for p in PartyType} == {"solo", "with_kids", "with_dog", "group"}


def test_weather_has_an_unknown_for_when_it_cannot_be_fetched():
    """Weather lookup may fail; UNKNOWN must be a real state, not an
    exception."""
    assert Weather.UNKNOWN.value == "unknown"
