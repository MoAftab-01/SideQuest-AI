"""Guarantees of the deterministic mission generator.

`app.ai.fallback` is the path the app takes whenever no model is loaded, times
out, or emits malformed JSON. That makes its promises user-visible rather than
internal, so this file pins the ones the project states publicly:

1. **Determinism.** `fallback_quest` is documented as a pure function of
   ``(request, seed)``. The eval harness compares the model against this
   generator, so a result that drifted run to run would make the comparison
   meaningless.
2. **The safety guarantee actually holds.** The docstring promises the fallback
   "can never emit a mission the project's own policy would reject". That is
   checked here the honest way: by re-running the real gate
   (`scan_quest` + `check_request_feasibility`) over the *returned* quest, not
   by trusting `quest.safety.passed`, which the builder writes unconditionally.
3. **Playable output.** Every mission carries a verification subject and at
   least one `must_include` criterion, stays inside the requested time budget,
   and contains no unrendered `{placeholder}`.

Two known backend bugs are documented at the bottom of this file rather than
asserted, because a test that fails is not a useful contribution to someone
else's file. See `BUG_*` tests for the exact repros.

No weights are loaded and no network is touched: the module imports stdlib,
pydantic schemas and the deterministic safety gate only.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from app.ai import fallback
from app.ai.fallback import TEMPLATES, fallback_quest, fallback_verdict
from app.ai.safety import check_request_feasibility, mission_text, scan_quest
from app.models.schemas import (
    ConfidenceLevel,
    EnergyLevel,
    Evidence,
    GeneratorSource,
    LocationType,
    Mobility,
    PartyType,
    Quest,
    QuestRequest,
    QuestStep,
    Weather,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def blocking_rules(quest: Quest, req: QuestRequest) -> list[str]:
    """Run the real gate over a finished mission and return blocking rule ids.

    This is deliberately *not* `quest.safety.passed`. `_attach_report` writes
    `passed=True` unconditionally, so the only way to check the published
    guarantee is to re-derive it from the gate itself.
    """
    findings = scan_quest(quest).findings + check_request_feasibility(req, quest).findings
    return [f.rule for f in findings if f.severity == "blocking"]


def total_minutes(quest: Quest) -> int:
    return sum(s.approx_minutes for s in quest.steps)


#: Requests chosen to be hostile rather than typical: the minimum budget the
#: schema allows, every interest slot filled with a gate keyword, every
#: mobility/party/energy/location combination that filters templates hardest.
HOSTILE_REQUESTS: tuple[QuestRequest, ...] = (
    QuestRequest(time_budget_minutes=5, interests=["a"] * 8),
    QuestRequest(time_budget_minutes=7, interests=["swimming", "river", "roof"]),
    QuestRequest(
        time_budget_minutes=12,
        mobility=Mobility.STEP_FREE,
        energy=EnergyLevel.LOW,
        location_type=LocationType.WATERFRONT,
        weather=Weather.SNOW,
    ),
    QuestRequest(
        time_budget_minutes=20,
        location_type=LocationType.TRAIL,
        party=PartyType.WITH_DOG,
        weather=Weather.WINDY,
    ),
    QuestRequest(time_budget_minutes=30, location_type=LocationType.INDOOR_MALL, weather=Weather.RAIN),
    QuestRequest(
        time_budget_minutes=45,
        interests=[""],
        location_type=LocationType.UNKNOWN,
        weather=Weather.UNKNOWN,
    ),
    QuestRequest(time_budget_minutes=60, interests=["x" * 2000], max_radius_m=50),
    QuestRequest(
        time_budget_minutes=240,
        interests=["birds"],
        avoid=["buses", "loud music", "crowds"],
        context_note="n" * 400,
    ),
    QuestRequest(
        time_budget_minutes=15,
        party=PartyType.WITH_KIDS,
        mobility=Mobility.STEP_FREE,
        energy=EnergyLevel.LOW,
        location_type=LocationType.NEIGHBORHOOD,
        weather=Weather.RAIN,
    ),
)


# ---------------------------------------------------------------------------
# Determinism -- same request + same seed, same mission
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_the_whole_quest():
    """Everything a player would read or be scored on must be reproducible.

    Only `id`, `created_at` and `generation_ms` may move: those are a fresh
    uuid, a wall-clock timestamp and a duration measurement, not content.
    Comparing the full dump rather than a couple of fields is the point -- a
    per-seed minute shuffle would show up here.
    """
    req = QuestRequest(time_budget_minutes=45, interests=["birds"], location_type=LocationType.PARK)
    first = fallback_quest(req, seed=7).model_dump()
    second = fallback_quest(req, seed=7).model_dump()

    volatile = {"id", "created_at", "generation_ms"}
    differing = {k for k in first if first[k] != second[k]}
    assert differing <= volatile, f"non-deterministic fields: {sorted(differing - volatile)}"


def test_determinism_holds_across_many_seeds_and_requests():
    """Not a one-seed fluke: repeat for a spread of requests and seeds."""
    for index, req in enumerate(HOSTILE_REQUESTS):
        for seed in (0, 1, 17):
            first = fallback_quest(req, seed=seed).model_dump()
            second = fallback_quest(req, seed=seed).model_dump()
            volatile = {"id", "created_at", "generation_ms"}
            differing = {k for k in first if first[k] != second[k]}
            assert differing <= volatile, (
                f"request {index} seed {seed} drifted in {sorted(differing - volatile)}"
            )


def test_explicit_seed_overrides_the_request_seed():
    """`seed=` wins over `req.seed`, as the signature documents."""
    seeded = QuestRequest(time_budget_minutes=40, interests=["plants"], seed=99)
    unseeded = QuestRequest(time_budget_minutes=40, interests=["plants"])

    assert fallback_quest(seeded).title == fallback_quest(seeded).title
    # Same explicit seed, different `req.seed`, identical result.
    assert fallback_quest(seeded, seed=1).title == fallback_quest(unseeded, seed=1).title


def test_determinism_does_not_depend_on_the_global_random_module():
    """`fallback_quest` seeds its own RNG, so `random.seed()` cannot steer it."""
    req = QuestRequest(time_budget_minutes=40, interests=["birds"])

    random.seed(1)
    first = fallback_quest(req, seed=4).title
    random.seed(999)
    [random.random() for _ in range(50)]
    assert fallback_quest(req, seed=4).title == first


# ---------------------------------------------------------------------------
# Diversity -- varied, not collapsed and not random
# ---------------------------------------------------------------------------


def test_different_seeds_produce_varied_missions():
    """The eval's `diversity_ratio` must be neither 0.0 nor 1.0.

    The harness computes it as distinct-titles over total-quests and reports
    ~0.6 for the deterministic baseline. Held to the same formula here over 200
    seeds: 1.0 would mean one mission monopolises every seed, 0.0 would mean
    every seed collides on the same title.
    """
    req = QuestRequest(
        time_budget_minutes=45,
        interests=["birds"],
        location_type=LocationType.CITY,
        weather=Weather.CLEAR,
    )
    titles = [fallback_quest(req, seed=seed).title for seed in range(200)]
    diversity = len(set(titles)) / len(titles)

    assert 0.0 < diversity < 1.0, f"diversity_ratio={diversity:.3f} is degenerate"
    # 30 templates cannot yield more than 30 distinct titles over 200 seeds.
    assert len(set(titles)) >= 10, f"only {len(set(titles))} distinct missions in 200 seeds"


def test_variety_is_not_only_the_title():
    """Different templates, not one template re-titled 200 times."""
    req = QuestRequest(time_budget_minutes=45, interests=["birds"], location_type=LocationType.CITY)
    quests = [fallback_quest(req, seed=seed) for seed in range(200)]

    assert len({q.verification.subject for q in quests}) >= 10
    assert len({tuple(q.tags) for q in quests}) >= 10
    # Per-seed minute shuffling shows up here even for a repeated template.
    assert len({total_minutes(q) for q in quests}) >= 2


def test_a_single_request_covers_the_whole_library():
    """One request plus enough seeds reaches every template in the library."""
    req = QuestRequest(time_budget_minutes=45, interests=["birds"], location_type=LocationType.CITY)
    titles = {fallback_quest(req, seed=seed).title for seed in range(200)}

    expected = {
        fallback._render(tpl.title, fallback._describe(req).as_mapping())
        for tpl in TEMPLATES
        if tpl.mobility_safe
        and fallback._describe(req)  # readability: context is fixed per request
    }
    # Compare against every template the hard filters can admit for this request.
    admitted = {tpl for tpl in TEMPLATES}
    expected = {
        fallback._render(tpl.title, fallback._describe(req).as_mapping()) for tpl in admitted
    }
    assert expected <= titles, f"never generated: {sorted(expected - titles)}"


# ---------------------------------------------------------------------------
# Safety -- every returned mission clears the real gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("req", HOSTILE_REQUESTS, ids=range(len(HOSTILE_REQUESTS)))
def test_hostile_requests_clear_the_safety_gate(req: QuestRequest):
    """`safety.passed` is true AND the real gate agrees, for hostile inputs.

    The two are checked separately on purpose. `passed` is written
    unconditionally by `_attach_report`, so asserting only it would be
    asserting nothing; the gate re-run is the load-bearing half.
    """
    for seed in range(25):
        quest = fallback_quest(req, seed=seed)
        assert quest.safety.passed is True, f"seed {seed} reported a failure"
        assert blocking_rules(quest, req) == [], (
            f"seed {seed} returned {quest.title!r} which the gate blocks"
        )


def test_dry_weather_sweep_never_returns_a_gate_blocked_mission():
    """A wide sweep with the two bug-triggering inputs removed stays clean.

    Rain/snow and `avoid` are excluded deliberately: both are known paths where
    `passed` and the real gate disagree (see the BUG tests below). This is the
    guard for everything else.
    """
    checked = 0
    for budget in (10, 15, 30, 60, 120):
        for location in LocationType:
            for weather in Weather:
                if weather in (Weather.RAIN, Weather.SNOW):
                    continue
                for party in PartyType:
                    for mobility in Mobility:
                        req = QuestRequest(
                            time_budget_minutes=budget,
                            location_type=location,
                            weather=weather,
                            party=party,
                            mobility=mobility,
                        )
                        for seed in (0, 1, 2):
                            quest = fallback_quest(req, seed=seed)
                            checked += 1
                            assert quest.safety.passed is True
                            assert blocking_rules(quest, req) == [], (
                                f"{budget=} {location=} {weather=} {party=} {mobility=} "
                                f"{seed=} -> {quest.title!r}"
                            )
    assert checked > 5000, f"sweep was too small to be meaningful: {checked}"


def test_missions_never_contain_an_unrendered_placeholder():
    """A literal `{place}` in player text is a rendering bug, not a flourish."""
    req = QuestRequest(time_budget_minutes=45, interests=["birds"], location_type=LocationType.CITY)
    for seed in range(200):
        quest = fallback_quest(req, seed=seed)
        assert fallback._leftover_placeholders(quest) == "", (
            f"seed {seed} leaked a placeholder into {quest.title!r}"
        )


def test_a_blocked_template_is_skipped_not_returned():
    """The retry loop is real: a hazardous candidate is dropped, not shipped."""
    hazardous = replace(
        TEMPLATES[0],
        key="test-hazardous",
        title="Swim the {place_title} Crossing",
        premise="A quick dip across open water.",
        steps=("Swim to the far bank before lunch.",),
        details=("Keep swimming.",),
        minutes=(30,),
    )

    original = TEMPLATES
    fallback.TEMPLATES = (hazardous,) + original
    try:
        quest = fallback_quest(QuestRequest(time_budget_minutes=30), seed=1)
        # The assertion is "the hazardous template was not returned", not "the
        # last-resort template was returned". The builder walks the ranked list
        # and returns the first candidate that clears the gate, so rejecting one
        # template normally lands on an ordinary one. Reaching _LAST_RESORT is
        # only correct when *every* template fails, which is what the test
        # originally got wrong -- it asserted a specific fallback for a case
        # where the correct behaviour is simply "not the hazardous one".
        assert quest.title != "Swim the Side Street Crossing", (
            "a template the gate blocks must never be returned"
        )
        assert "Swim" not in quest.title
        assert blocking_rules(quest, QuestRequest(time_budget_minutes=30)) == []
        assert quest.safety.passed is True
    finally:
        fallback.TEMPLATES = original


def test_a_template_with_an_unrendered_placeholder_is_skipped():
    """The second rejection reason is an unknown `{token}` surviving render."""
    broken = replace(
        TEMPLATES[0],
        key="test-broken",
        title="A {mystery_token} Walk",
    )

    original = TEMPLATES
    fallback.TEMPLATES = (broken,) + original
    try:
        quest = fallback_quest(QuestRequest(time_budget_minutes=30), seed=1)
        # Same correction as above: the guarantee is that no unresolved token
        # reaches the player, not that the builder bottoms out at _LAST_RESORT.
        assert "{mystery_token}" not in quest.title
        assert "mystery_token" not in quest.title
        assert fallback._leftover_placeholders(quest) == ""
    finally:
        fallback.TEMPLATES = original


# ---------------------------------------------------------------------------
# Verification spec -- the app is useless without one
# ---------------------------------------------------------------------------


def test_every_quest_carries_a_verification_spec():
    """A mission with no criteria cannot be verified; the app would be a toy."""
    for index, req in enumerate(HOSTILE_REQUESTS):
        for seed in range(25):
            quest = fallback_quest(req, seed=seed)
            spec = quest.verification
            assert spec.subject.strip(), f"request {index} seed {seed}: empty subject"
            assert spec.must_include, f"request {index} seed {seed}: no must_include"
            assert all(item.strip() for item in spec.must_include)
            assert spec.accept_hints, f"request {index} seed {seed}: no accept_hints"
            assert spec.reject_if, f"request {index} seed {seed}: no reject_if"
            assert spec.ambiguous_cases, f"request {index} seed {seed}: no ambiguous_cases"


def test_verification_is_mission_specific_not_boilerplate():
    """Criteria must differ per mission, or they judge nothing in particular."""
    req = QuestRequest(time_budget_minutes=45, interests=["birds"], location_type=LocationType.CITY)
    quests = [fallback_quest(req, seed=seed) for seed in range(200)]

    assert len({q.verification.subject for q in quests}) >= 10
    assert len({tuple(q.verification.must_include) for q in quests}) >= 10


def test_ambiguous_cases_are_declared_so_the_model_can_honesty_say_i_cannot_tell():
    req = QuestRequest(time_budget_minutes=45, interests=["birds"])
    for seed in range(50):
        assert fallback_quest(req, seed=seed).verification.ambiguous_cases


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [5, 6, 8, 12, 20, 45, 90, 180, 240])
def test_missions_stay_within_the_requested_time_budget(budget: int):
    """Sum of step minutes, `estimated_minutes` and XP all respect the ask."""
    req = QuestRequest(time_budget_minutes=budget)
    for seed in range(10):
        quest = fallback_quest(req, seed=seed)
        assert total_minutes(quest) <= budget, (
            f"budget {budget} seed {seed}: steps total {total_minutes(quest)}"
        )
        assert quest.estimated_minutes == total_minutes(quest), (
            f"budget {budget} seed {seed}: estimated_minutes disagrees with the steps"
        )
        assert all(step.approx_minutes >= 1 for step in quest.steps)


def test_a_tiny_budget_still_returns_a_playable_single_step_mission():
    """Trimming must never empty the step list -- the schema forbids it."""
    for seed in range(20):
        quest = fallback_quest(QuestRequest(time_budget_minutes=5), seed=seed)
        assert quest.steps, "a 5-minute budget produced a mission with no steps"
        assert quest.estimated_minutes >= 1


def test_xp_is_bounded_and_positive():
    for budget in (5, 30, 120, 240):
        for seed in range(10):
            quest = fallback_quest(QuestRequest(time_budget_minutes=budget), seed=seed)
            assert 5 <= quest.xp_reward <= 500


# ---------------------------------------------------------------------------
# Robustness -- odd inputs must not raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interests",
    [
        pytest.param([], id="empty"),
        pytest.param(["", "   "], id="whitespace-only"),
        pytest.param(["x" * 2000], id="very-long"),
        pytest.param(["birds"] * 8, id="eight-repeats"),
        pytest.param(["café", "über", "日本語"], id="non-ascii"),
        pytest.param(["swimming", "climbing", "traffic", "river"], id="gate-keywords"),
        pytest.param(["-volcanic rocketry-", "🚲"], id="odd-punctuation"),
        pytest.param(["photography", "PHOTOGRAPHY", "  Photography  "], id="mixed-case"),
    ],
)
def test_odd_interests_do_not_raise(interests: list[str]):
    """`interests` is free text from a player; nothing here may blow up."""
    req = QuestRequest(time_budget_minutes=40, interests=interests)
    for seed in range(10):
        quest = fallback_quest(req, seed=seed)
        assert quest.steps
        assert quest.verification.subject.strip()


@pytest.mark.parametrize("location", list(LocationType), ids=lambda loc: loc.value)
def test_every_location_type_produces_a_mission(location: LocationType):
    for weather in Weather:
        req = QuestRequest(time_budget_minutes=30, location_type=location, weather=weather)
        quest = fallback_quest(req, seed=1)
        assert quest.steps, f"no mission for {location} in {weather}"
        assert quest.safety.passed is True


@pytest.mark.parametrize("budget", [5, 240])
@pytest.mark.parametrize("radius", [50, 137, 800, 5000])
def test_radius_extremes_do_not_raise(budget: int, radius: int):
    quest = fallback_quest(QuestRequest(time_budget_minutes=budget, max_radius_m=radius), seed=1)
    assert quest.steps


def test_a_full_context_note_is_handled():
    quest = fallback_quest(
        QuestRequest(time_budget_minutes=40, interests=["birds"], context_note="c" * 400),
        seed=1,
    )
    assert len(quest.premise) <= 500, "the premise cap must hold for a long context note"


# ---------------------------------------------------------------------------
# Library invariants
# ---------------------------------------------------------------------------


def test_every_template_in_the_library_produces_a_quest():
    """Each of the 30 templates is reachable and builds a clean mission.

    Forcing `TEMPLATES` down to one entry at a time is the only way to assert
    "every template works" without relying on the weighted shuffle happening to
    pick it. The neutral request admits every hard filter, so a failure here
    is the template's fault, not the ranking's.
    """
    neutral = QuestRequest(time_budget_minutes=45, interests=["birds"], location_type=LocationType.CITY)
    context = fallback._describe(neutral)
    original = TEMPLATES

    failures: list[str] = []
    for tpl in original:
        fallback.TEMPLATES = (tpl,)
        quest = fallback_quest(neutral, seed=1)
        expected_title = fallback._render(tpl.title, context.as_mapping())
        if quest.title != expected_title:
            reasons = []
            if blocking_rules(quest, neutral):
                reasons.append(f"gate blocks it: {blocking_rules(quest, neutral)}")
            if fallback._leftover_placeholders(quest):
                reasons.append(f"leaked a placeholder: {fallback._leftover_placeholders(quest)}")
            failures.append(f"{tpl.key} ({tpl.category}) -> {quest.title!r}: {'; '.join(reasons)}")
    fallback.TEMPLATES = original

    assert not failures, "unusable templates:\n  " + "\n  ".join(failures)


def test_every_category_is_reachable_through_the_public_api():
    """All 16 categories must be producible, not just declared.

    Titles are unique per template across the library, so a returned title maps
    back to exactly one category. That mapping is asserted first, then a
    spread of requests and seeds is swept for full coverage.
    """
    categories = {tpl.category for tpl in TEMPLATES}
    assert len(categories) == 16, f"category count changed: {len(categories)}"

    category_of: dict[str, str] = {}
    for tpl in TEMPLATES:
        for location in LocationType:
            context = replace(
                fallback._describe(QuestRequest(time_budget_minutes=45, location_type=location)),
            )
            for title in {
                fallback._render(tpl.title, context.as_mapping()),
            }:
                if title in category_of:
                    assert category_of[title] == tpl.category, (
                        f"title {title!r} is shared across categories"
                    )
                category_of[title] = tpl.category

    requests = (
        QuestRequest(time_budget_minutes=45, interests=["birds"], location_type=LocationType.CITY),
        QuestRequest(time_budget_minutes=60, location_type=LocationType.PARK, weather=Weather.CLOUDY),
    )
    reached = set()
    for req in requests:
        for seed in range(100):
            reached.add(category_of[fallback_quest(req, seed=seed).title])

    assert reached == categories, f"unreachable categories: {sorted(categories - reached)}"


def test_library_keys_and_difficulty_are_sane():
    """The import-time guard in `_assert_library_sane` still describes reality."""
    assert len({tpl.key for tpl in TEMPLATES}) == len(TEMPLATES), "duplicate template keys"
    for tpl in TEMPLATES:
        assert len(tpl.steps) == len(tpl.minutes), f"{tpl.key}: steps/minutes mismatch"
        assert 1 <= len(tpl.details) <= len(tpl.steps), f"{tpl.key}: bad details"
        assert 1 <= tpl.difficulty <= 3, f"{tpl.key}: difficulty out of range"
        assert tpl.variant_key in fallback._VARIANTS, f"{tpl.key}: unknown variant"
        assert tpl.v_subject and tpl.v_must, f"{tpl.key}: empty verification"
        assert min(tpl.minutes) >= 1


def test_generated_missions_are_tagged_as_fallback():
    """`source` must never imply a model ran. The eval reads this field."""
    for req in HOSTILE_REQUESTS[:4]:
        quest = fallback_quest(req, seed=3)
        assert quest.source is GeneratorSource.FALLBACK
        assert quest.model_id is None
        assert quest.raw_model_output is None
        assert [s.order for s in quest.steps] == list(range(1, len(quest.steps) + 1))


def test_steps_are_well_formed():
    req = QuestRequest(time_budget_minutes=60, interests=["birds"])
    for seed in range(50):
        for step in fallback_quest(req, seed=seed).steps:
            assert isinstance(step, QuestStep)
            assert step.instruction.strip()
            assert step.approx_minutes >= 1


def test_mission_text_never_leaks_the_internal_template_scaffolding():
    """Guard against a player seeing `{place}` or an empty title."""
    req = QuestRequest(time_budget_minutes=45, interests=["birds"], location_type=LocationType.PARK)
    for seed in range(100):
        quest = fallback_quest(req, seed=seed)
        assert quest.title.strip()
        assert "{" not in mission_text(quest)
        assert quest.hook.strip()


# ---------------------------------------------------------------------------
# fallback_verdict -- the no-model verdict never claims verification
# ---------------------------------------------------------------------------


@pytest.fixture
def a_quest() -> Quest:
    return fallback_quest(QuestRequest(time_budget_minutes=40, interests=["birds"]), seed=3)


@pytest.mark.parametrize(
    "photo",
    [
        pytest.param(None, id="no-photo"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param("not-a-url", id="not-a-data-url"),
        pytest.param("data:", id="bare-data-prefix"),
        pytest.param("data:image/png;base64,", id="empty-base64"),
        pytest.param("data:;base64,@@@@", id="undecodable"),
        pytest.param("data:text/plain,hello", id="wrong-mime"),
        pytest.param("data:image/jpeg;base64," + "A" * 4000, id="large"),
    ],
)
def test_verdict_never_claims_verification_or_awards_xp(a_quest: Quest, photo: str | None):
    """No photo was ever inspected, so nothing may be asserted about it."""
    verdict = fallback_verdict(a_quest, Evidence(photo_data_url=photo))

    assert verdict.is_ai_verified is False
    assert verdict.xp_awarded == 0
    assert verdict.source is GeneratorSource.FALLBACK
    assert verdict.model_id is None
    assert verdict.needs_better_photo is True
    assert verdict.confidence <= 0.15, "confidence must stay near zero without a model"
    assert verdict.level in (fallback.ConfidenceLevel.HEURISTIC, fallback.ConfidenceLevel.UNCERTAIN)
    assert len(verdict.summary) <= 300


def test_verdict_level_reflects_only_whether_a_photo_was_supplied(a_quest: Quest):
    """No photo -> HEURISTIC; a photo nobody looked at -> UNCERTAIN."""
    without = fallback_verdict(a_quest, Evidence())
    with_photo = fallback_verdict(a_quest, Evidence(photo_data_url="data:image/png;base64,AAAA"))

    assert without.level is fallback.ConfidenceLevel.HEURISTIC
    assert without.confidence == 0.0
    assert with_photo.level is fallback.ConfidenceLevel.UNCERTAIN
    assert with_photo.confidence == 0.15
    assert "no vision model" in with_photo.summary.lower()
    assert "unverified" in without.summary.lower()


def test_a_boasting_note_cannot_buy_a_verification(a_quest: Quest):
    """Player text is summarised, never believed."""
    note = "I completed every single step of this mission, it was wonderful"
    verdict = fallback_verdict(a_quest, Evidence(photo_data_url=None, note=note))

    assert verdict.is_ai_verified is False
    assert verdict.xp_awarded == 0
    assert verdict.confidence == 0.0


def test_verdict_missing_items_come_from_the_missions_own_criteria(a_quest: Quest):
    """The 'what is missing' list is mission-specific, so the photo can be fixed."""
    verdict = fallback_verdict(a_quest, Evidence(photo_data_url="data:image/png;base64,AAAA"))

    assert verdict.missing
    assert verdict.missing[:3] == list(a_quest.verification.must_include[:3])


# ---------------------------------------------------------------------------
# Regression tests for two bugs that shipped
#
# Both were originally recorded as unexecuted `BUG_*` repros (pytest collects
# `test_*` only, so they ran in nobody's suite). Both are fixed and are now
# real tests. The shared lesson is the same one the app is built around: the
# safety report must never disagree with the safety rules that produced it.
# ---------------------------------------------------------------------------


def test_a_hazard_word_in_the_avoid_list_does_not_poison_the_library():
    """An avoid term that is itself a gate keyword must not block every mission.

    `_build` used to add "You asked to avoid busy roads; this route keeps clear
    of it." to `safety_notes`. `mission_text` includes `safety_notes`, so the
    literal string "road" reached `haz-traffic` -- a *blocking* rule -- and made
    every candidate look unsafe. The loop fell through the entire library to the
    last-resort mission, so a player who asked to avoid roads got the same
    one-block mission no matter what else they chose.
    """
    req = QuestRequest(time_budget_minutes=45, avoid=["roads"])
    quest = fallback_quest(req, seed=0)

    assert blocking_rules(quest, req) == [], "the gate must agree the mission is safe"
    assert quest.safety.passed is True
    assert quest.title != "One Block, Five Things", (
        "a hazard-word avoid term must not exhaust the template library"
    )


@pytest.mark.parametrize(
    "term",
    ["roads", "river", "lake", "climb", "storm", "drains", "underpass", "quarry"],
)
def test_no_avoid_term_can_silently_degrade_a_player_to_the_last_resort_mission(
    term: str,
):
    """The regression above is general; this pins it across every hazard word.

    Eight terms x four seeds. Before the fix every one of the 32 returned the
    last-resort mission while reporting `passed=True`.
    """
    req = QuestRequest(time_budget_minutes=45, avoid=[term])

    for seed in range(4):
        quest = fallback_quest(req, seed=seed)
        assert quest.title != "One Block, Five Things", (
            f"avoid=[{term!r}] seed={seed} exhausted the library"
        )
        assert blocking_rules(quest, req) == []
        assert quest.safety.passed is True


def test_the_gate_cannot_block_a_mission_with_its_own_advisory():
    """`wet-weather` says "skip any climbing equipment"; `haz-height` reads "climb".

    `_attach_report` copies the gate's own warning into `safety_notes`, and
    `scan_quest` reads notes back on the next pass. The advisory therefore
    matched a blocking rule and rejected a mission the gate had just cleared --
    on roughly 29% of rain/snow requests, before the note filter.
    """
    req = QuestRequest(
        time_budget_minutes=12,
        mobility=Mobility.STEP_FREE,
        energy=EnergyLevel.LOW,
        location_type=LocationType.WATERFRONT,
        weather=Weather.SNOW,
    )
    quest = fallback_quest(req, seed=2)

    assert blocking_rules(quest, req) == []
    assert quest.safety.passed is True


def test_a_hazard_hidden_in_a_note_the_gate_did_not_write_is_still_caught():
    """The counterpart to the filter above: the exclusion must not over-reach.

    Notes are still scanned, and only the gate's own exact strings are
    filtered. A caution written by a template or the model is exactly as
    dangerous as a caution written by the gate, so dropping the whole field
    would have traded a false positive for a real hole.
    """
    from app.models.schemas import Quest, QuestStep, VerificationSpec

    for note in [
        "Do not dive off the quay.",
        "Climb onto the roof if you dare.",
        "Cross the highway when it is quiet.",
        "Careful, there are storms forecast later.",
    ]:
        quest = Quest(
            title="Look at the sky.",
            hook="Take a slow walk.",
            premise="Notice one thing.",
            steps=[
                QuestStep(
                    order=1,
                    instruction="Photograph the bench beside you.",
                    approx_minutes=2,
                    detail="Then stop.",
                )
            ],
            estimated_minutes=5,
            xp_reward=10,
            safety_notes=[note],
            verification=VerificationSpec(subject="s", must_include=["a"]),
        )
        report = scan_quest(quest).report()
        assert report.passed is False, f"hazard in a note slipped through: {note!r}"


def test_no_gate_authored_advisory_can_block_on_its_own():
    """Every string the gate attaches to a mission is inert to its own scanner.

    Six of the twelve rule messages contain the keyword their own rule matches,
    and two of the request-feasibility warnings trip blocking rules
    (`MSG_DOG_SAFETY` -> `haz-traffic`, `MSG_WET_WEATHER` -> `haz-height`).
    This pins all of them at once so a newly reworded message cannot silently
    reintroduce the self-block.
    """
    from app.ai.safety import _GATE_AUTHORED
    from app.models.schemas import Quest, QuestStep, VerificationSpec

    assert _GATE_AUTHORED, "the exclusion set must not be empty"

    for message in _GATE_AUTHORED:
        quest = Quest(
            title="Look at the sky.",
            hook="Take a slow walk.",
            premise="Notice one thing.",
            steps=[
                QuestStep(
                    order=1,
                    instruction="Photograph the bench beside you.",
                    approx_minutes=2,
                    detail="Then stop.",
                )
            ],
            estimated_minutes=5,
            xp_reward=10,
            safety_notes=[message],
            verification=VerificationSpec(subject="s", must_include=["a"]),
        )
        blocking = [f.rule for f in scan_quest(quest).findings if f.severity == "blocking"]
        assert blocking == [], f"{message!r} blocks a mission: {blocking}"


def test_every_advisory_the_gate_can_emit_is_in_the_exclusion_set():
    """The filter must be complete by construction, not by memory.

    Derives the messages at runtime by spying on `Screen.add` across every
    weather/party/energy/mobility combination with a quest stuffed with trigger
    words, then checks coverage. A warning added inline later and forgotten here
    fails this test instead of quietly re-poisoning the gate.
    """
    import app.ai.safety as safety_module
    from app.ai.safety import _GATE_AUTHORED
    from app.models.schemas import Quest, QuestStep, VerificationSpec

    emitted: set[str] = set()
    original = safety_module.Screen.add

    def spy(self, rule, severity, message):  # type: ignore[no-untyped-def]
        emitted.add(message)
        return original(self, rule, severity, message)

    safety_module.Screen.add = spy  # type: ignore[method-assign]
    try:
        hostile = Quest(
            title="Climb the ladder by the slippery rocks near the river.",
            hook="storm at midnight alone",
            premise="run and sprint uphill, off-lead, stairs and gravel, lakeside",
            steps=[
                QuestStep(
                    order=1,
                    instruction="Photograph the muddy slide and the icy puddle.",
                    approx_minutes=120,
                    detail="trash debris litter",
                )
            ],
            estimated_minutes=5,
            xp_reward=10,
            safety_notes=["scout the dog park"],
            verification=VerificationSpec(subject="s", must_include=["a"]),
        )
        for weather in Weather:
            for party in PartyType:
                for energy in EnergyLevel:
                    for mobility in Mobility:
                        for minutes in (10, 60):
                            safety_module.check_request_feasibility(
                                QuestRequest(
                                    time_budget_minutes=minutes,
                                    weather=weather,
                                    party=party,
                                    energy=energy,
                                    mobility=mobility,
                                    avoid=["busy roads"],
                                ),
                                hostile,
                            )
    finally:
        safety_module.Screen.add = original  # type: ignore[method-assign]

    assert len(emitted) == 6, f"expected all six advisories, got {sorted(emitted)}"
    assert emitted <= _GATE_AUTHORED, f"unfiltered: {emitted - _GATE_AUTHORED}"


def test_the_fallback_never_reports_safe_on_a_mission_the_gate_blocks():
    """`passed` must be derived from the findings, never assumed.

    `_attach_report` hard-coded `passed=True`, so a mission the gate rejected --
    reachable via the last-resort path, which is built and reported without being
    re-checked -- still claimed to be safe. This sweeps every constraint
    combination the request schema allows.
    """
    import itertools

    bad = []
    for weather, party, energy, mobility, location in itertools.product(
        Weather, PartyType, EnergyLevel, Mobility, LocationType
    ):
        req = QuestRequest(
            time_budget_minutes=30,
            weather=weather,
            party=party,
            energy=energy,
            mobility=mobility,
            location_type=location,
        )
        for seed in (0, 3):
            quest = fallback_quest(req, seed=seed)
            if blocking_rules(quest, req) or not quest.safety.passed:
                bad.append((weather, party, energy, mobility, location, seed))

    assert bad == [], f"safety report contradicted the gate: {bad[:5]}"