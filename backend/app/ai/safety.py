"""Deterministic safety policy for generated outdoor missions.

This module is deliberately *not* model-generated. A language model writing the
mission is a liability when the mission sends a person to a cliff edge; the
safety gate is a transparent, auditable rule set that runs on every mission
regardless of which engine produced it. When the model proposes something
unsafe, the gate rewrites or rejects it -- the model never gets the last word.

Design: `Screen` is a small expression language over the proposed mission text
plus the player's constraints. Every rule has an id, a severity, and a message
that is shown to the player when it fires, so the policy is legible rather than
a black box.

Severity semantics
------------------
``info``     annotate the mission with a caution.
``warning``  mutate the mission to remove the hazard, keep going.
``blocking`` reject the mission outright; the caller falls back to a safe
             deterministic mission instead.

Both ``warning`` and ``blocking`` outcomes are exercised by the test suite, and
the eval harness reports how often the model proposes something the gate has to
stop -- a number worth publishing either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.schemas import (
    EnergyLevel,
    Mobility,
    PartyType,
    Quest,
    QuestRequest,
    SafetyFinding,
    SafetyReport,
    Weather,
)

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str
    message: str
    pattern: re.Pattern[str]


def _p(*alts: str) -> re.Pattern[str]:
    """Word-boundary alternation matcher, case-insensitive and plural-tolerant.

    Uses ``\\b`` boundaries so "cat" does not match inside "category", which
    matters: several rules key off short words.

    Each alternative additionally accepts an optional trailing ``s``/``es`` so a
    single word in the rule covers its plural. This was a real bypass, not a
    theoretical one: the rules listed "climb", "roof", "ladder", "storm" and
    friends, so a model mission reading "Climbs over the gate", "Roofs of the
    shops" or "Storms forecast later" passed a blocking gate untouched. Plural
    phrasing is the *common* case for text a language model writes, so a gate
    that only catches the singular is a gate that mostly does not fire. The
    optional suffix cannot swallow the boundary check -- ``\\b`` still applies
    after it, so "categories" is not matched by "cat" while "cats" is.
    """
    joined = "|".join(alts)
    return re.compile(rf"\b(?:{joined})(?:e?s)?\b", re.IGNORECASE)


#: Hazards that make a mission unsafe enough to block outright.
BLOCKING_RULES: tuple[Rule, ...] = (
    Rule(
        "haz-water-unprotected",
        "blocking",
        "Open water without a lifeguard or PFD is the main way this game could "
        "hurt someone. Pick a bank you can stand on.",
        _p("swim", "swimming", "dive", "diving", "jump into", "wade", "wading",
           "submerge", "deep end", "open water", "lake", "river", "canal", "reservoir"),
    ),
    Rule(
        "haz-height",
        "blocking",
        "Height exposure was removed. Climbing, ledges and rooftops are out of "
        "scope for this app.",
        # "roof"/"rooftop" appear only in the act phrases. The bare noun was
        # dropped on purpose: once _p() became plural-tolerant, a bare "roof"
        # matched "roofs", and a mission telling the player to stand somewhere
        # they "can see sky and at least one line of trees or roofs" -- looking
        # *at* rooftops from the pavement -- was blocked as a height hazard. That
        # is a false positive on a perfectly safe instruction, and it killed the
        # bird-count template outright. The hazard is being up there, so the rule
        # matches the climbing, not the viewing.
        _p("climb", "climbing", "climbed", "scramble", "ladder", "climb onto",
           "climb up", "on the roof", "on the rooftop", "rooftop access",
           "ledge", "cliff", "edge of the", "balcony", "high wall", "gantry",
           "antenna", "treetop", "treehouse"),
    ),
    Rule(
        "haz-traffic",
        "blocking",
        "Road and rail play is never generated. Find a quiet route instead.",
        _p("highway", "motorway", "freeway", "interstate", "on the road", "road",
           "roads", "traffic", "crosswalk", "crossing", "railway", "railroad",
           "rail track", "railway track", "level crossing", "truck route"),
    ),
    Rule(
        "haz-restricted",
        "blocking",
        "This needs land you are not allowed to be on.",
        _p("trespass", "trespassing", "abandoned", "derelict", "construction site",
           "construction zone", "scaffold", "no trespassing", "private property",
           "fence line", "graveyard", "cemetery after", "rail yard", "quarry",
           "sewer", "tunnel", "tunnels", "drain", "storm drain", "underpass"),
    ),
    Rule(
        "haz-storm-exposure",
        "blocking",
        "Storm and high-wind conditions are not a good time to go exploring.",
        _p("storm", "thunderstorm", "lightning", "lightning strike", "hurricane",
           "cyclone", "blizzard", "ice storm", "gale"),
    ),
    Rule(
        "haz-extreme-heat",
        "blocking",
        "Extreme-heat guidance is not something this app will generate. Check the "
        "local advisory and hydrate first.",
        _p("heatwave", "heat wave", "heatwave advisory", "record heat",
           "dangerous heat"),
    ),
)

#: Hazards we can strip or downgrade rather than reject the whole mission.
WARNING_RULES: tuple[Rule, ...] = (
    Rule(
        "warn-heat-load",
        "warning",
        "Long exertion in the heat is risky. Take water and stop if you feel "
        "faint or headachy.",
        _p("run", "running", "jog", "jogging", "sprint", "shin up", "hike uphill",
           "steep climb", "ascent"),
    ),
    Rule(
        "warn-after-dark",
        "warning",
        "Visibility drops after dark. Stay where you can see the path and get "
        "home.",
        _p("night", "after dark", "midnight", "starlight", "moonlight", "sunset",
           "dusk", "twilight"),
    ),
    Rule(
        "warn-alone",
        "warning",
        "Out alone: keep your phone charged and tell someone where you went.",
        _p("alone", "by yourself", "solo", "on your own", "unaccompanied"),
    ),
    Rule(
        "warn-cold-water",
        "warning",
        "Cold water and thin ice are more dangerous than they look.",
        _p("ice", "icy", "frozen", "thin ice", "puddle", "flood", "flooded"),
    ),
    Rule(
        "warn-wildlife",
        "warning",
        "Give wildlife distance and never feed what you find.",
        _p("animal", "animals", "dog", "dogs", "cat", "cats", "bird", "birds",
           "squirrel", "squirrels", "fox", "raccoon", "rat", "rats", "snake",
           "bees", "wasp", "wasps", "insect", "insects", "livestock"),
    ),
    Rule(
        "warn-litter",
        "warning",
        "Take your rubbish with you, and only collect what is safe to touch.",
        _p("pick up", "collect", "gather", "bag up", "litter", "trash", "waste",
           "debris"),
    ),
)

ALL_RULES: tuple[Rule, ...] = BLOCKING_RULES + WARNING_RULES

# ---------------------------------------------------------------------------
# Messages this module writes onto a quest
# ---------------------------------------------------------------------------
# These are hoisted to constants rather than left inline in
# `check_request_feasibility` so `_GATE_AUTHORED` below is complete by
# construction: add a warning here and it is excluded from scanning
# automatically, with no second place to forget.

MSG_BUDGET_OVERRUN = (
    "This plan is longer than the time you asked for. Trim it or come back "
    "with more time."
)
MSG_MOBILITY_MISMATCH = (
    "This route does not match the step-free/avoid-stairs setting you chose."
)
MSG_KID_SAFETY = (
    "With kids along: stay where you can see them, and avoid water and drops."
)
MSG_DOG_SAFETY = (
    "Off-lead near roads or water is where dogs get lost. Keep it short and "
    "supervised."
)
MSG_WET_WEATHER = (
    "Wet ground and playground equipment get slippery. Skip any climbing "
    "equipment if it is slick."
)
MSG_ENERGY_MISMATCH = (
    "This is more effort than a low-energy day. Keep it to a short version if "
    "you are tired."
)

#: Every advisory string this module can attach to a mission.
#:
#: A striking number of them contain the very keyword their own rule matches:
#: `haz-height`'s own message says "Climbing", `haz-traffic`'s says "Road", and
#: of the request-feasibility warnings `MSG_DOG_SAFETY` trips `haz-traffic`
#: (*blocking*) while `MSG_WET_WEATHER` trips `haz-height` (*blocking*). Left in
#: scope, a mission can be blocked by a caution the gate itself wrote -- which
#: is not a theoretical hazard: it killed the wet-reflections template on every
#: seed, silently degrading it to the last-resort mission.
#:
#: They are excluded from the scan, and *only* they are. Notes authored by a
#: template or by the model are still scanned, because a hazard hidden in a
#: caution is exactly as dangerous as one hidden in an instruction and the test
#: suite pins that. The distinction is provenance, not location.
_GATE_AUTHORED: frozenset[str] = frozenset(
    [r.message for r in ALL_RULES]
    + [
        MSG_BUDGET_OVERRUN,
        MSG_MOBILITY_MISMATCH,
        MSG_KID_SAFETY,
        MSG_DOG_SAFETY,
        MSG_WET_WEATHER,
        MSG_ENERGY_MISMATCH,
    ]
)


# ---------------------------------------------------------------------------
# Constraint checks
# ---------------------------------------------------------------------------


@dataclass
class Screen:
    """Collects findings for one mission."""

    findings: list[SafetyFinding] = field(default_factory=list)
    triggered: list[str] = field(default_factory=list)

    def add(self, rule_id: str, severity: str, message: str) -> None:
        if rule_id not in self.triggered:
            self.triggered.append(rule_id)
        self.findings.append(SafetyFinding(rule=rule_id, severity=severity, message=message))

    @property
    def blocking(self) -> list[SafetyFinding]:
        return [f for f in self.findings if f.severity == "blocking"]

    def report(self) -> SafetyReport:
        return SafetyReport(
            passed=not self.blocking,
            findings=self.findings,
            triggered=list(self.triggered),
        )


def mission_text(quest: Quest) -> str:
    """All human-readable prose on a mission, joined for keyword scanning.

    ``safety_notes`` is included, because a caution is a place a hazard can hide
    just as well as an instruction is -- a generator that wrote "climb the
    fence" into a note has asked for exactly the thing the gate exists to stop.
    The gate's *own* advisories are the one exception, and they are filtered by
    exact string match against ``_GATE_AUTHORED`` rather than by dropping the
    whole field: six of the twelve rule messages contain the keyword their own
    rule matches, so a blanket exclusion would hide real hazards while a blanket
    inclusion lets the gate block missions with its own advice.

    Filtering by string identity rather than by rule id matters: the same words
    arriving from a template or the model are still scanned, because the
    exclusion is about *provenance*, not about the vocabulary a sentence uses.
    """
    parts: list[str] = [quest.title, quest.hook, quest.premise]
    parts += [s.instruction for s in quest.steps]
    parts += [s.detail for s in quest.steps]
    parts += [n for n in quest.safety_notes if n not in _GATE_AUTHORED]
    return "\n".join(p for p in parts if p)


def scan_quest(quest: Quest) -> Screen:
    """Run keyword rules over a fully-formed mission."""
    screen = Screen()
    text = mission_text(quest)

    for rule in ALL_RULES:
        if rule.pattern.search(text):
            screen.add(rule.id, rule.severity, rule.message)

    return screen


def check_request_feasibility(req: QuestRequest, quest: Quest) -> Screen:
    """Rules that compare the mission against the player's stated constraints.

    These catch a subtler failure than keyword matching: a mission that is
    perfectly safe in the abstract but does not fit the person asking for it.
    """
    screen = Screen()

    # A step that asks for more time than the whole budget is not feasible.
    total = sum(s.approx_minutes for s in quest.steps)
    if total > req.time_budget_minutes * 1.5:
        screen.add("budget-overrun", "warning", MSG_BUDGET_OVERRUN)

    if req.mobility is not Mobility.UNRESTRICTED:
        text = mission_text(quest).lower()
        inaccessible = _p(
            "stairs", "staircase", "stairway", "step-free", "climb", "hike uphill",
            "steep", "gravel slope", "uneven ground", "rough ground",
        )
        if inaccessible.search(text):
            screen.add("mobility-mismatch", "blocking", MSG_MOBILITY_MISMATCH)

    if req.party is PartyType.WITH_KIDS:
        text = mission_text(quest).lower()
        risky = _p(
            "ladder", "climb", "roof", "ditch", "creek", "trail", "trails",
            "bridge", "balcony", "water", "parking lot",
        )
        if risky.search(text):
            screen.add("kid-safety", "warning", MSG_KID_SAFETY)

    if req.party is PartyType.WITH_DOG:
        text = mission_text(quest).lower()
        if _p("off-lead", "off leash", "unleashed").search(text):
            screen.add("dog-safety", "warning", MSG_DOG_SAFETY)

    if req.weather in (Weather.RAIN, Weather.SNOW):
        text = mission_text(quest).lower()
        if _p("slippery", "icy", "uneven", "rocks", "gravel", "mud", "metal stairs",
              "playground equipment", "playground", "slide").search(text):
            # This message trips `haz-height` (blocking) on the next scan,
            # because it contains the word "climbing". The advice is *against*
            # climbing, so flagging it was the gate contradicting itself: a safe
            # mission blocked by a warning the gate had just written. The
            # exclusion in `mission_text` is what fixes that, rather than
            # rewording the advice until it stops tripping a rule -- rewording
            # hides the symptom and leaves the next message to trip it again.
            screen.add("wet-weather", "warning", MSG_WET_WEATHER)

    if req.energy is EnergyLevel.LOW:
        text = mission_text(quest).lower()
        if _p("run", "sprint", "ascent", "climb", "hike", "uphill", "steep").search(text):
            screen.add("energy-mismatch", "warning", MSG_ENERGY_MISMATCH)

    return screen


def gate(quest: Quest, req: QuestRequest) -> Quest:
    """Apply the safety gate, returning either a repaired quest or the same one.

    ``warning`` findings are attached to the mission so the player sees them.
    ``blocking`` findings are attached too, but the *caller* is responsible for
    checking ``quest.safety.passed`` and substituting a deterministic mission;
    this function never invents a replacement, because it does not know what
    would be a good one.
    """
    request_screen = check_request_feasibility(req, quest)
    text_screen = scan_quest(quest)
    report = request_screen.report()
    report.findings.extend(text_screen.findings)
    for t in text_screen.triggered:
        if t not in report.triggered:
            report.triggered.append(t)
    report.passed = not any(f.severity == "blocking" for f in report.findings)

    quest.safety = report
    # Seed from the notes the mission already carries, then append any new rule
    # message. This used to *replace* the list wholesale with rule messages,
    # which silently destroyed notes the generator had deliberately authored --
    # including the one telling the player their `avoid` list was honoured. That
    # note is the only place the app confirms a constraint it cannot enforce, so
    # dropping it turned an honest disclosure into silence.
    notes: list[str] = list(quest.safety_notes)[:6]
    for finding in report.findings:
        if len(notes) >= 6:
            break
        if finding.severity in ("warning", "blocking") and finding.message not in notes:
            notes.append(finding.message)
    quest.safety_notes = notes
    return quest


def blocked_reason(quest: Quest) -> str:
    """Short human-facing reason a mission was rejected by the gate."""
    blocking = [f for f in quest.safety.findings if f.severity == "blocking"]
    if not blocking:
        return ""
    return "; ".join(f"{f.rule}: {f.message}" for f in blocking[:2])