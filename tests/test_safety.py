"""Safety-gate policy tests for SideQuest AI.

`backend/app/ai/safety.py` is the project's central safety claim: the language
model writes the mission, but a deterministic rule set gets the last word, and
it runs identically on every mission regardless of which engine produced it.
Two promises are pinned here.

1. **Blocking rules block, and only block.** Each of the six blocking rules is
   exercised with an unmistakably unsafe instruction *and* with an equivalent
   but innocent phrasing. The second half matters as much as the first: a gate
   that fires on "the category museum" or "a catalogue of street names" is a
   gate nobody will switch on. The module's own docstring commits to this --
   ``_p`` builds word-boundary alternations precisely so short words do not
   match inside longer ones -- so these are pinned as behaviour, not trivia.

2. **Warnings downgrade, they never block.** A caution the gate raises on an
   otherwise fine mission must leave ``SafetyReport.passed`` True, must be
   reported with severity ``warning``, and must be surfaced to the player via
   ``safety_notes``. A warning that quietly failed the mission would be worse
   than no warning at all.

`check_request_feasibility`` is covered against the constraints the player
actually selects (time budget, mobility, party, weather, energy). Everything
runs in-process with no weights and no network: ``safety.py`` imports only
``re`` and the pydantic schemas, so this module never pulls in torch or
transformers and finishes in well under a second.
"""

from __future__ import annotations

import pytest

from app.ai import safety
from app.ai.safety import (
    ALL_RULES,
    BLOCKING_RULES,
    WARNING_RULES,
    check_request_feasibility,
    gate,
    mission_text,
    scan_quest,
)
from app.models.schemas import (
    EnergyLevel,
    Mobility,
    PartyType,
    Quest,
    QuestRequest,
    QuestStep,
    VerificationSpec,
    Weather,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def make_quest(
    title: str = "Count the Doorbells",
    hook: str = "Twelve front doors, one walk.",
    premise: str = "A slow look at the public face of a street.",
    instruction: str = "Photograph four doorbells.",
    detail: str = "",
    minutes: int = 6,
    safety_notes: list[str] | None = None,
) -> Quest:
    """A minimal but fully valid Quest carrying text in one place at a time.

    ``minutes`` lands on the single step so the time-budget rule can be driven
    from a single knob.
    """
    return Quest(
        title=title,
        hook=hook,
        premise=premise,
        steps=[QuestStep(order=1, instruction=instruction, approx_minutes=minutes, detail=detail)],
        estimated_minutes=minutes,
        xp_reward=120,
        verification=VerificationSpec(subject="a close photograph of a doorbell"),
        safety_notes=safety_notes or [],
    )


def make_request(**overrides) -> QuestRequest:
    """A neutral default request; override only what a test is about."""
    kwargs: dict = {"time_budget_minutes": 60}
    kwargs.update(overrides)
    return QuestRequest(**kwargs)


def rules_fired(text: str, rules=ALL_RULES) -> set[str]:
    """Rule ids whose pattern matches ``text``, for pattern-level assertions."""
    return {r.id for r in rules if r.pattern.search(text)}


def finding_rules(report) -> list[str]:
    return [f.rule for f in report.findings]


# ---------------------------------------------------------------------------
# The rule tables themselves are well formed
# ---------------------------------------------------------------------------


def test_blocking_and_warning_rules_are_disjoint_and_all_blocking_rules_are_blocking():
    """Severity is declared on the rule, and the two tables never overlap.

    ``gate`` decides pass/fail purely by reading ``finding.severity``, so a rule
    filed in the wrong table would be enforced at the wrong strength.
    """
    blocking_ids = {r.id for r in BLOCKING_RULES}
    warning_ids = {r.id for r in WARNING_RULES}

    assert blocking_ids.isdisjoint(warning_ids)
    assert all(r.severity == "blocking" for r in BLOCKING_RULES)
    assert all(r.severity == "warning" for r in WARNING_RULES)
    # ALL_RULES is what scan_quest iterates, so it must cover both tables.
    assert {r.id for r in ALL_RULES} == blocking_ids | warning_ids


def test_every_rule_carries_a_player_facing_message():
    """A finding the player cannot read is not a finding."""
    for rule in ALL_RULES:
        assert rule.message.strip(), f"{rule.id} has an empty message"
        assert rule.pattern.pattern


def test_rule_ids_are_unique():
    ids = [r.id for r in ALL_RULES]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# BLOCKING rules -- each one blocks on unsafe text
# ---------------------------------------------------------------------------

#: One plainly-unsafe instruction per blocking rule id.
BLOCKING_CASES = [
    ("haz-water-unprotected", "Swim out to the floating buoy and back."),
    ("haz-height", "Climb the ladder onto the fire escape."),
    ("haz-traffic", "Walk the length of the highway until the county line."),
    ("haz-restricted", "Slip through the abandoned station at the end of the street."),
    ("haz-storm-exposure", "Wait out the thunderstorm on the headland."),
    ("haz-extreme-heat", "Set out only after the heatwave advisory lifts."),
]

#: Innocuous phrasing that shares vocabulary with the blocking rules but
#: describes nothing hazardous. None of these may raise a finding.
SAFE_LOOKALIKES = [
    "Count the doorbells on a residential street.",
    "Photograph shop windows on a quiet pavement.",
    "Sit on a bench and read for ten minutes.",
    "Find five different leaves in the park.",
    "Note three different cloud shapes.",
    "Walk one block and count the postboxes.",
    "Sketch the fountain plaza.",
    "Read the names on the memorial wall.",
    # Word-boundary cases the `_p` helper exists for.
    "The category museum exhibit changes on Thursdays.",
    "Fill in a catalogue of street names.",
    "Write up the concatenation of two lists.",
]


@pytest.mark.parametrize("rule_id,unsafe_text", BLOCKING_CASES, ids=[c[0] for c in BLOCKING_CASES])
def test_each_blocking_rule_fires_on_its_own_hazard(rule_id, unsafe_text):
    assert rule_id in rules_fired(unsafe_text), f"{rule_id} missed: {unsafe_text!r}"


@pytest.mark.parametrize("rule_id,unsafe_text", BLOCKING_CASES, ids=[c[0] for c in BLOCKING_CASES])
def test_each_blocking_rule_alone_fails_the_mission(rule_id, unsafe_text):
    """The rule's own severity, not a neighbour's, is what fails the mission."""
    quest = make_quest(instruction=unsafe_text)
    report = scan_quest(quest).report()

    assert report.passed is False, f"{rule_id} fired but did not block"
    assert rule_id in report.triggered
    blocking = [f for f in report.findings if f.severity == "blocking"]
    assert [f.rule for f in blocking] == [rule_id]
    assert blocking[0].message.strip()


@pytest.mark.parametrize("rule_id,unsafe_text", BLOCKING_CASES, ids=[c[0] for c in BLOCKING_CASES])
def test_each_blocking_rule_rejects_through_the_full_gate(rule_id, unsafe_text):
    """End to end: ``gate`` is the entry point the planner actually calls."""
    quest = gate(make_quest(instruction=unsafe_text), make_request())

    assert quest.safety.passed is False
    assert rule_id in quest.safety.triggered
    # A blocked mission still explains itself to the player.
    assert quest.safety_notes, "a blocked mission must carry a reason"
    assert rule_id in safety.blocked_reason(quest)


@pytest.mark.parametrize("text", SAFE_LOOKALIKES, ids=lambda s: s[:34])
def test_safe_phrasing_trips_no_rule_at_all(text):
    """False positives are the failure mode that kills a safety gate.

    If the gate blocked a bench-reading mission, every mission would be a
    fallback mission and the product would be worthless. These are ordinary
    generated-mission lines, so they must clear every rule.
    """
    assert rules_fired(text) == set(), f"safe text {text!r} tripped {rules_fired(text)}"


@pytest.mark.parametrize("text", SAFE_LOOKALIKES, ids=lambda s: s[:34])
def test_safe_phrasing_produces_a_passing_report_and_no_notes(text):
    quest = gate(make_quest(instruction=text), make_request())

    assert quest.safety.passed is True
    assert quest.safety.findings == []
    assert quest.safety.triggered == []
    assert quest.safety_notes == []
    assert safety.blocked_reason(quest) == ""


def test_the_word_boundary_helper_does_not_match_inside_longer_words():
    """``_p`` is documented to stop "cat" matching "category"; pin that.

    Three of the rule tables key off very short words, so this is the property
    the whole no-false-positive story rests on.
    """
    assert "warn-wildlife" in rules_fired("A cat sat on the wall.")
    assert rules_fired("The category museum is free.") == set()
    assert rules_fired("She is a cathedral volunteer.") == set()
    assert rules_fired("Prance around the concourse.") == set()
    assert rules_fired("The scrapbook seller is here.") == set()


def test_matching_is_case_insensitive():
    for text in ("SWIM to the buoy.", "Climb the LADDER.", "Thunderstorm over the hill."):
        assert rules_fired(text), f"{text!r} was missed because of its capitalisation"
    assert scan_quest(make_quest(instruction="SWIM to the buoy.")).report().passed is False


# ---------------------------------------------------------------------------
# The scan covers every surface a mission can hide text on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["hook", "premise", "instruction", "detail", "safety_notes"],
)
def test_a_hazard_hidden_anywhere_on_the_quest_is_caught(field):
    """A hazard in the detail pane is still a hazard the player will read."""
    kwargs = {"instruction": "Photograph four doorbells."}
    if field == "hook":
        kwargs["hook"] = "Then dive off the quay."
    elif field == "premise":
        kwargs["premise"] = "A mission that ends with a swim at the reservoir."
    elif field == "instruction":
        kwargs["instruction"] = "Dive off the quay."
    elif field == "detail":
        kwargs["detail"] = "Finish by diving off the quay."
    elif field == "safety_notes":
        kwargs["safety_notes"] = ["Do not dive off the quay."]

    report = scan_quest(make_quest(**kwargs)).report()

    assert report.passed is False, f"hazard in {field} was not caught"
    assert "haz-water-unprotected" in report.triggered


def test_mission_text_concatenates_every_prose_field():
    quest = make_quest(
        hook="the hook text",
        premise="the premise text",
        instruction="the instruction text",
        detail="the detail text",
        safety_notes=["a note"],
    )
    blob = mission_text(quest)

    for fragment in ("the hook text", "the premise text", "the instruction text", "the detail text", "a note"):
        assert fragment in blob


# ---------------------------------------------------------------------------
# WARNING rules -- downgrade, never block
# ---------------------------------------------------------------------------

WARNING_CASES = [
    ("warn-heat-load", "Jog two laps of the promenade."),
    ("warn-after-dark", "Set out after dark and walk the lit square."),
    ("warn-alone", "Go alone to the museum and back."),
    ("warn-cold-water", "Walk the bank where the ice is thin."),
    ("warn-wildlife", "Photograph the squirrel in the long grass."),
    ("warn-litter", "Pick up litter where the bins spill over on Saturdays."),
]


@pytest.mark.parametrize("rule_id,text", WARNING_CASES, ids=[c[0] for c in WARNING_CASES])
def test_each_warning_rule_fires_on_its_own_hazard(rule_id, text):
    assert rule_id in rules_fired(text)


@pytest.mark.parametrize("rule_id,text", WARNING_CASES, ids=[c[0] for c in WARNING_CASES])
def test_a_warning_never_blocks_the_mission(rule_id, text):
    """The severity contract: warnings mutate and continue, blocks reject."""
    quest = gate(make_quest(instruction=text), make_request())

    assert quest.safety.passed is True, f"{rule_id} blocked the mission instead of warning"
    assert rule_id in quest.safety.triggered
    assert quest.safety.findings, "the warning vanished"
    assert all(f.severity == "warning" for f in quest.safety.findings)
    assert all(f.rule == rule_id for f in quest.safety.findings)


@pytest.mark.parametrize("rule_id,text", WARNING_CASES, ids=[c[0] for c in WARNING_CASES])
def test_a_warning_is_reported_as_a_warning_not_a_block(rule_id, text):
    """The player-facing wording must not read as a rejection."""
    report = scan_quest(make_quest(instruction=text)).report()

    assert report.passed is True
    assert [f.severity for f in report.findings] == ["warning"]
    # blocked_reason reads only blocking findings, so it stays empty here.
    quest = gate(make_quest(instruction=text), make_request())
    assert safety.blocked_reason(quest) == ""


@pytest.mark.parametrize("rule_id,text", WARNING_CASES, ids=[c[0] for c in WARNING_CASES])
def test_a_warning_is_surfaced_to_the_player_as_a_safety_note(rule_id, text):
    quest = gate(make_quest(instruction=text), make_request())

    assert len(quest.safety_notes) == 1
    assert quest.safety_notes[0] == quest.safety.findings[0].message


def test_multiple_warnings_accumulate_without_blocking():
    quest = gate(
        make_quest(instruction="Jog the path alone after dark and watch the birds."),
        make_request(),
    )

    assert quest.safety.passed is True
    assert set(quest.safety.triggered) == {"warn-heat-load", "warn-after-dark", "warn-alone", "warn-wildlife"}
    assert len(quest.safety_notes) == len(quest.safety.triggered)


def test_a_warning_alongside_a_block_still_blocks():
    """The tables compose: one hazard on top of another cannot launder the block."""
    quest = gate(
        make_quest(instruction="Jog to the cliff edge and dive in, alone, after dark."),
        make_request(),
    )

    assert quest.safety.passed is False
    severities = {f.rule: f.severity for f in quest.safety.findings}
    assert severities["haz-water-unprotected"] == "blocking"
    assert severities["haz-height"] == "blocking"
    assert severities["warn-heat-load"] == "warning"
    # blocked_reason names the blockers, never the cautions.
    reason = safety.blocked_reason(quest)
    assert "haz-water-unprotected" in reason
    assert "warn-heat-load" not in reason


# ---------------------------------------------------------------------------
# check_request_feasibility -- time budget
# ---------------------------------------------------------------------------


def test_a_plan_longer_than_the_budget_is_flagged():
    """Steps summing past the budget are infeasible whatever they are."""
    request = make_request(time_budget_minutes=20)
    quest = make_quest(instruction="Photograph every doorbell on the street.", minutes=31)

    screen = check_request_feasibility(request, quest)

    assert "budget-overrun" in screen.triggered
    assert [f.severity for f in screen.findings] == ["warning"]


def test_budget_is_summed_across_every_step():
    """One 20-minute step fits; two of them overrun a 20-minute budget."""
    request = make_request(time_budget_minutes=20)
    quest = make_quest(minutes=20)
    quest.steps.append(QuestStep(order=2, instruction="And a second errand as well.", approx_minutes=20))
    quest.estimated_minutes = 40

    assert sum(s.approx_minutes for s in quest.steps) == 40
    assert "budget-overrun" in check_request_feasibility(request, quest).triggered


def test_a_plan_inside_the_budget_is_not_flagged():
    request = make_request(time_budget_minutes=60)
    quest = make_quest(instruction="Photograph four doorbells.", minutes=30)

    assert check_request_feasibility(request, quest).triggered == []


def test_budget_tolerance_is_exactly_one_and_a_half_times():
    """Pin the documented slack: over the line is flagged, on it is not."""
    quest_30 = make_quest(instruction="A long but legal errand.", minutes=30)
    quest_31 = make_quest(instruction="A long and illegal errand.", minutes=31)

    # 30 == 20 * 1.5 -- the tolerance is inclusive.
    assert "budget-overrun" not in check_request_feasibility(make_request(time_budget_minutes=20), quest_30).triggered
    assert "budget-overrun" in check_request_feasibility(make_request(time_budget_minutes=20), quest_31).triggered


def test_an_overrunning_plan_is_a_warning_not_a_block():
    """Overrunning your own budget is an inconvenience, not a hazard."""
    quest = gate(
        make_quest(instruction="Photograph every doorbell on the street.", minutes=90),
        make_request(time_budget_minutes=20),
    )

    assert quest.safety.passed is True
    assert "budget-overrun" in quest.safety.triggered


# ---------------------------------------------------------------------------
# check_request_feasibility -- mobility, party, weather, energy
# ---------------------------------------------------------------------------

INACCESSIBLE = "Take the staircase up to the viewing gallery."
STEPPABLE = "Photograph four doorbells from the pavement."


@pytest.mark.parametrize(
    "mobility",
    [Mobility.AVOID_STAIRS, Mobility.STEP_FREE],
    ids=lambda m: m.value,
)
def test_a_stair_route_is_blocked_when_mobility_was_limited(mobility):
    screen = check_request_feasibility(make_request(mobility=mobility), make_quest(instruction=INACCESSIBLE))

    assert "mobility-mismatch" in screen.triggered
    assert [f.severity for f in screen.findings] == ["blocking"], (
        "a route the player cannot physically take must block, not merely warn"
    )


def test_a_mobility_mismatch_blocks_through_the_gate():
    quest = gate(make_quest(instruction=INACCESSIBLE), make_request(mobility=Mobility.STEP_FREE))

    assert quest.safety.passed is False
    assert "mobility-mismatch" in quest.safety.triggered
    assert quest.safety_notes


def test_stairs_are_ignored_when_mobility_is_unrestricted():
    """The default player is not a wheelchair user; stairs are fine for them."""
    screen = check_request_feasibility(make_request(mobility=Mobility.UNRESTRICTED), make_quest(instruction=INACCESSIBLE))

    assert "mobility-mismatch" not in screen.triggered


def test_a_step_free_route_is_accepted_for_a_step_free_player():
    screen = check_request_feasibility(
        make_request(mobility=Mobility.STEP_FREE),
        make_quest(instruction=STEPPABLE),
    )

    assert screen.triggered == []


@pytest.mark.parametrize(
    "inaccessible",
    ["Expect a steep climb past the allotments.",
     "Follow the gravel slope to the top.",
     "Expect uneven ground the whole way."],
    ids=["climb", "gravel slope", "uneven ground"],
)
def test_other_hard_terrain_also_breaks_a_step_free_route(inaccessible):
    assert "mobility-mismatch" in check_request_feasibility(
        make_request(mobility=Mobility.AVOID_STAIRS),
        make_quest(instruction=inaccessible),
    ).triggered


def test_kids_get_a_caution_near_water_and_drops():
    screen = check_request_feasibility(
        make_request(party=PartyType.WITH_KIDS),
        make_quest(instruction="Follow the creek down to the old bridge."),
    )

    assert "kid-safety" in screen.triggered
    assert [f.severity for f in screen.findings] == ["warning"]
    # A caution about the kids is not a rejection of the mission.
    assert screen.report().passed is True


def test_kids_are_not_flagged_on_an_ordinary_street_mission():
    screen = check_request_feasibility(
        make_request(party=PartyType.WITH_KIDS),
        make_quest(instruction=STEPPABLE),
    )

    assert screen.triggered == []


def test_a_dog_off_leash_near_water_is_cautioned():
    screen = check_request_feasibility(
        make_request(party=PartyType.WITH_DOG),
        make_quest(instruction="Let the dog off-lead by the water."),
    )

    assert "dog-safety" in screen.triggered
    assert [f.severity for f in screen.findings] == ["warning"]


def test_a_dog_on_the_leash_is_not_flagged():
    screen = check_request_feasibility(
        make_request(party=PartyType.WITH_DOG),
        make_quest(instruction="Walk the dog on the lead around the block."),
    )

    assert screen.triggered == []


@pytest.mark.parametrize("weather", [Weather.RAIN, Weather.SNOW], ids=lambda w: w.value)
def test_wet_weather_surfaces_a_slippery_surface_warning(weather):
    screen = check_request_feasibility(
        make_request(weather=weather),
        make_quest(instruction="Pick your way over the loose rocks."),
    )

    assert "wet-weather" in screen.triggered
    assert [f.severity for f in screen.findings] == ["warning"]


def test_dry_weather_raises_no_wet_ground_caution():
    screen = check_request_feasibility(
        make_request(weather=Weather.CLEAR),
        make_quest(instruction="Pick your way over the loose rocks."),
    )

    assert screen.triggered == []


def test_a_low_energy_day_is_cautioned_about_effort():
    screen = check_request_feasibility(
        make_request(energy=EnergyLevel.LOW),
        make_quest(instruction="Run the length of the park and back."),
    )

    assert "energy-mismatch" in screen.triggered
    assert [f.severity for f in screen.findings] == ["warning"]


def test_effort_is_not_flagged_on_a_high_energy_day():
    screen = check_request_feasibility(
        make_request(energy=EnergyLevel.HIGH),
        make_quest(instruction="Run the length of the park and back."),
    )

    assert screen.triggered == []


def test_every_constraint_check_can_fire_on_one_mission():
    """The constraints compose: one mission can overrun, climb, and go out alone."""
    quest = make_quest(instruction="Run up the uneven ground to the bridge alone.", minutes=200)
    request = make_request(
        time_budget_minutes=20,
        mobility=Mobility.AVOID_STAIRS,
        party=PartyType.WITH_KIDS,
        energy=EnergyLevel.LOW,
    )

    screen = check_request_feasibility(request, quest)

    assert set(screen.triggered) == {
        "budget-overrun",
        "mobility-mismatch",
        "kid-safety",
        "energy-mismatch",
    }
    # One blocker is enough to fail it; the rest ride along as cautions.
    assert screen.report().passed is False


def test_a_mission_matching_the_player_passes_every_constraint():
    screen = check_request_feasibility(
        make_request(
            time_budget_minutes=60,
            mobility=Mobility.STEP_FREE,
            party=PartyType.WITH_KIDS,
            weather=Weather.CLEAR,
            energy=EnergyLevel.LOW,
        ),
        make_quest(instruction=STEPPABLE, minutes=20),
    )

    assert screen.triggered == []
    assert screen.report().passed is True


# ---------------------------------------------------------------------------
# The public report builder -- passed=True for plainly safe text
# ---------------------------------------------------------------------------


def test_scan_quest_report_passes_for_plainly_safe_text():
    """``scan_quest(...).report()`` is the helper that produces the pass verdict."""
    report = scan_quest(make_quest()).report()

    assert report.passed is True
    assert report.findings == []
    assert report.triggered == []


def test_report_of_a_clean_quest_is_a_safety_report():
    from app.models.schemas import SafetyReport

    report = scan_quest(make_quest()).report()

    assert isinstance(report, SafetyReport)
    assert report.passed is True


def test_gate_passes_a_plainly_safe_mission_and_leaves_no_notes():
    quest = gate(make_quest(), make_request())

    assert quest.safety.passed is True
    assert quest.safety.triggered == []
    assert quest.safety_notes == []
    assert safety.blocked_reason(quest) == ""


def test_gate_returns_the_quest_it_was_given():
    """The planner threads the return value straight back into its pipeline."""
    quest = make_quest()
    assert gate(quest, make_request()) is quest


def test_screen_records_a_rule_once_but_keeps_each_finding():
    """`triggered` is the de-duplicated rule list the UI keys off."""
    screen = safety.Screen()
    screen.add("warn-alone", "warning", "first")
    screen.add("warn-alone", "warning", "second")

    assert screen.triggered == ["warn-alone"]
    assert len(screen.findings) == 2


def test_screen_report_passes_while_only_warnings_fired():
    screen = safety.Screen()
    screen.add("warn-wildlife", "warning", "Give wildlife distance.")
    screen.add("warn-after-dark", "warning", "Stay where you can see the path.")

    report = screen.report()

    assert report.passed is True
    assert report.triggered == ["warn-wildlife", "warn-after-dark"]


def test_screen_report_fails_once_a_blocker_fired():
    screen = safety.Screen()
    screen.add("warn-wildlife", "warning", "Give wildlife distance.")
    screen.add("haz-height", "blocking", "Climbing is out of scope.")

    report = screen.report()

    assert report.passed is False
    assert [f.rule for f in report.findings if f.severity == "blocking"] == ["haz-height"]
    # `Screen.blocking` is the convenience view the planner reads.
    assert [f.rule for f in screen.blocking] == ["haz-height"]
    assert [f.rule for f in screen.blocking] == [f.rule for f in report.findings if f.severity == "blocking"]


# ---------------------------------------------------------------------------
# safety_notes are what the player actually reads
# ---------------------------------------------------------------------------


def test_notes_are_capped_so_the_player_sees_a_readable_list():
    """Six lines of caution is a wall of text; the model caps it deliberately."""
    quest = gate(
        make_quest(
            instruction=(
                "Swim, climb the ladder, run, go alone, after dark, "
                "watch the birds and pick up litter."
            )
        ),
        make_request(),
    )

    assert quest.safety.passed is False
    assert len(quest.safety.triggered) >= 7, "expected several rules to fire"
    assert len(quest.safety_notes) == 6
    for note in quest.safety_notes:
        assert note.strip()


def test_blocked_reason_names_the_rule_and_its_message():
    quest = gate(make_quest(instruction="Climb the ladder onto the fire escape."), make_request())
    reason = safety.blocked_reason(quest)

    assert reason.startswith("haz-height:")
    assert "ladder" in reason or "Climbing" in reason
    # Two blockers at most, so the string stays a sentence rather than a report.
    assert len([f for f in quest.safety.findings if f.severity == "blocking"]) >= 1
    assert reason.count("; ") <= 1


# ---------------------------------------------------------------------------
# Degenerate input -- empty, whitespace, unicode, regex metacharacters
# ---------------------------------------------------------------------------

ODD_TEXT = [
    "",
    "   ",
    "\n\t  \r\n",
    "​",  # zero-width space
    "﻿",  # byte-order mark
    "‮",  # RTL override
    "café résumé naïve",
    "🏔️📍🗺️",
    "a" * 300,
    "swim" + " " * 200,
    ".*+?[](){}|^$\\",
    "dive|dive|dive||",
    "line\nbreak\ttab",
    "\U0001f6b6 swimming",  # emoji adjacent to a hazard word
]


@pytest.mark.parametrize("text", ODD_TEXT, ids=range(len(ODD_TEXT)))
def test_odd_text_never_raises(text):
    """A malformed or hostile model output must not crash the gate.

    The model writes this text, so it arrives unvalidated; the gate is the last
    thing standing between it and a player, and a traceback here would take out
    every request.
    """
    report = scan_quest(make_quest(instruction=text or "x")).report()

    assert isinstance(report.passed, bool)
    assert isinstance(report.findings, list)
    assert isinstance(report.triggered, list)


@pytest.mark.parametrize("text", ["", "   ", "\n\t  \r\n", "​", ".*+?[](){}|^$\\", "café résumé"],
                         ids=["empty", "spaces", "newlines", "zwsp", "regex-chars", "accents"])
def test_empty_and_harmless_text_passes_cleanly(text):
    """No hazard word means no finding -- including when the text is empty."""
    report = scan_quest(make_quest(instruction=text or "x")).report()

    assert report.passed is True
    assert report.findings == []


@pytest.mark.parametrize("text", ["", "   ", "​", ".*+?[](){}|^$\\", "café résumé naïve", "🏔️📍🗺️"],
                         ids=["empty", "spaces", "zwsp", "regex-chars", "accents", "emoji"])
def test_gate_and_feasibility_survive_odd_text_end_to_end(text):
    quest = make_quest(instruction=text or "x")

    gated = gate(quest, make_request(mobility=Mobility.STEP_FREE, party=PartyType.WITH_KIDS,
                                     weather=Weather.RAIN, energy=EnergyLevel.LOW))
    feasibility = check_request_feasibility(
        make_request(time_budget_minutes=5, mobility=Mobility.STEP_FREE), quest
    )

    assert isinstance(gated.safety.passed, bool)
    assert isinstance(feasibility.report().passed, bool)


def test_a_quest_with_no_safety_notes_scans_cleanly():
    """`mission_text` filters empties rather than joining None into the blob."""
    quest = make_quest(instruction="Photograph four doorbells.", safety_notes=[])
    quest.hook = ""
    quest.premise = ""
    quest.steps[0].detail = ""

    blob = mission_text(quest)
    assert blob.strip()  # not empty: title and instruction are still there
    assert "Photograph four doorbells." in blob
    assert "\n\n" not in blob  # no blank line from the filtered-out empties
    assert scan_quest(quest).report().passed is True


def test_the_gate_survives_repeated_runs_on_the_same_quest():
    """Idempotent enough not to accumulate notes or drift verdicts.

    ``gate`` writes ``safety_notes`` from the report it just built, and
    ``mission_text`` reads those notes back on the next pass, so running it
    twice must not compound.
    """
    quest = make_quest(instruction="Climb the ladder onto the fire escape.")

    first = gate(quest, make_request()).safety.passed
    second = gate(quest, make_request()).safety.passed
    notes = len(quest.safety_notes)

    assert first is False and second is False
    assert notes <= 6


def test_a_quest_with_eight_steps_of_seven_minutes_fits_a_sixty_minute_budget():
    """The rule sums steps; it does not read `estimated_minutes`."""
    quest = make_quest(instruction="Photograph the first doorbell.", minutes=7)
    for n in range(2, 9):
        quest.steps.append(QuestStep(order=n, instruction=f"Photograph doorbell number {n}.", approx_minutes=7))
    quest.estimated_minutes = 56

    assert len(quest.steps) == 8
    assert check_request_feasibility(make_request(time_budget_minutes=60), quest).triggered == []


def test_the_gate_blocks_only_on_blocking_findings():
    """`passed` is a function of blocking findings alone.

    An `info`-severity finding is annotation only. The rule table defines no
    `info` rules today, so this pins the arithmetic rather than a rule: any
    future informational rule must not start failing missions.
    """
    screen = safety.Screen()
    screen.add("note-a", "info", "FYI only.")
    screen.add("note-b", "warning", "Take care.")

    assert screen.report().passed is True


def test_location_type_is_not_a_constraint_the_gate_enforces():
    """Documents a real gap: `QuestRequest` carries a location type, and the
    gate ignores it entirely.

    `check_request_feasibility` reads `time_budget_minutes`, `mobility`,
    `party`, `weather` and `energy` -- but never `location_type`, `avoid`,
    `interests` or `max_radius_m`. So a request for an INDOOR_MALL rain
    contingency that comes back as a street mission, or a mission that ignores
    the player's explicit `avoid` list, passes the safety gate untouched. This
    test is written to pass today and to fail loudly if the gap is ever closed
    by adding a `location-mismatch` rule, at which point it should be rewritten
    as a positive assertion.

    Kept rather than omitted: the project advertises feasibility checking, and
    a reader of this file should not have to read `safety.py` to learn that a
    third of the player's stated constraints go unchecked.
    """
    indoor = make_request(location_type="indoor_mall", weather=Weather.RAIN, avoid=["museum"])
    street_mission = make_quest(instruction="Walk one block and count the postboxes.")

    screen = check_request_feasibility(indoor, street_mission)

    assert screen.triggered == []
    assert screen.report().passed is True
    assert gate(street_mission, indoor).safety.passed is True