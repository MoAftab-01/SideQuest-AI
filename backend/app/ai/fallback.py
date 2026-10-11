"""Deterministic mission generator: the documented baseline and the safety net.

Why this module exists
----------------------
Two jobs, and both matter for honesty:

1. **Baseline.** The configured model is measured against this library. A weak
   deterministic generator would make the model look good by comparison, which
   is exactly the kind of flattering baseline a results table should not have.
   So the library below is curated, varied and genuinely playable.
2. **Safety net.** When the model is unavailable, times out, or emits malformed
   JSON, the app must still hand the player a real mission. The fallback is a
   normal code path, not an error handler, so it is built to the same standard
   as the model path.

Determinism
-----------
`fallback_quest` is a pure function of ``(request fields, seed)``. Given the same
request and the same seed it returns byte-identical prose, the same step count,
the same XP and the same verification spec. Tests assert this, and the eval
harness relies on it to compare the model against a stable reference.

Safety posture
--------------
The library is written hazard-free against `app.ai.safety` (no swimming, height
exposure, road play, trespass or storms), but hand-written prose drifts. So the
builder runs every candidate through the *real* gate (`scan_quest` +
`check_request_feasibility`) and retries with another template if anything
trips. The safety net can therefore never emit a mission the project's own
policy would reject.

Verification posture
--------------------
`fallback_verdict` never claims verification. `is_ai_verified` is False, XP is
always 0, and the level is HEURISTIC (no photo) or UNCERTAIN (photo present but
no vision model ran). That is the whole point: a hardcoded rule must never be
dressed up as a model judgement.

No heavy imports. This module must import in milliseconds on a machine with no
external inference installed, so it depends only on the stdlib, pydantic-backed schemas and
the deterministic safety gate.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass

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
    SafetyFinding,
    SafetyReport,
    VerificationSpec,
    Verdict,
    Weather,
)

__all__ = ["compose_prompt", "fallback_quest", "fallback_verdict"]


# ---------------------------------------------------------------------------
# Request -> prose context
# ---------------------------------------------------------------------------

#: Lower-case noun phrase for the setting, used inside step instructions.
_PLACE = {
    LocationType.CITY: "side street",
    LocationType.NEIGHBORHOOD: "residential block",
    LocationType.PARK: "park path",
    LocationType.TRAIL: "wooded trailhead",
    LocationType.WATERFRONT: "promenade",
    LocationType.INDOOR_MALL: "atrium",
    LocationType.UNKNOWN: "block around you",
}

#: Title-case variant for mission titles.
_PLACE_TITLE = {
    LocationType.CITY: "Side Street",
    LocationType.NEIGHBORHOOD: "Residential Block",
    LocationType.PARK: "Park Path",
    LocationType.TRAIL: "Wooded Trailhead",
    LocationType.WATERFRONT: "Promenade",
    LocationType.INDOOR_MALL: "Atrium",
    LocationType.UNKNOWN: "Familiar Block",
}

#: Single adjective describing the light and air.
_WEATHER_WORD = {
    Weather.CLEAR: "bright",
    Weather.CLOUDY: "overcast",
    Weather.RAIN: "damp",
    Weather.HEAT: "warm",
    Weather.COLD: "crisp",
    Weather.SNOW: "snowy",
    Weather.WINDY: "breezy",
    Weather.UNKNOWN: "flat",
}

#: Season-ish cue. There is no date on the request, so this is derived from the
#: reported weather rather than the calendar -- and stays deterministic.
_SEASON = {
    Weather.CLEAR: "clear-season light",
    Weather.CLOUDY: "flat-season light",
    Weather.RAIN: "wet-season light",
    Weather.HEAT: "high-summer light",
    Weather.COLD: "cold-season light",
    Weather.SNOW: "snow-season light",
    Weather.WINDY: "windy-season light",
    Weather.UNKNOWN: "midday light",
}

_PARTY = {
    PartyType.SOLO: "just you",
    PartyType.WITH_KIDS: "the grown-up and the kids",
    PartyType.WITH_DOG: "you and your dog",
    PartyType.GROUP: "your group",
}

_ENERGY_LABEL = {
    EnergyLevel.LOW: "an easy, unhurried pace",
    EnergyLevel.MEDIUM: "a steady walking pace",
    EnergyLevel.HIGH: "a pace with some ground to cover",
}


@dataclass(frozen=True)
class _Ctx:
    """Every placeholder a template may use, resolved once per request."""

    place: str
    place_title: str
    weather_word: str
    season: str
    party: str
    pace: str
    time_phrase: str
    radius_phrase: str
    interest: str
    interest2: str
    interest_lead: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "place": self.place,
            "place_title": self.place_title,
            "weather_word": self.weather_word,
            "season": self.season,
            "party": self.party,
            "pace": self.pace,
            "time_phrase": self.time_phrase,
            "radius_phrase": self.radius_phrase,
            "interest": self.interest,
            "interest2": self.interest2,
            "interest_lead": self.interest_lead,
        }


def _describe(req: QuestRequest) -> _Ctx:
    """Turn a request into the placeholder values templates substitute in."""
    interests = [t for t in req.interests if t]
    first = interests[0] if interests else "small details you walk past"
    second = interests[1] if len(interests) > 1 else first
    if interests:
        lead = f"You mentioned {first}, so this mission leans that way."
    else:
        lead = "You did not name an interest, so start with whatever catches your eye."
    radius = max(50, int(req.max_radius_m) // 50 * 50)
    return _Ctx(
        place=_PLACE.get(req.location_type, _PLACE[LocationType.UNKNOWN]),
        place_title=_PLACE_TITLE.get(
            req.location_type, _PLACE_TITLE[LocationType.UNKNOWN]
        ),
        weather_word=_WEATHER_WORD.get(req.weather, _WEATHER_WORD[Weather.UNKNOWN]),
        season=_SEASON.get(req.weather, _SEASON[Weather.UNKNOWN]),
        party=_PARTY.get(req.party, _PARTY[PartyType.SOLO]),
        pace=_ENERGY_LABEL.get(req.energy, _ENERGY_LABEL[EnergyLevel.MEDIUM]),
        time_phrase=f"{req.time_budget_minutes} minutes",
        radius_phrase=f"about {radius} m",
        interest=first,
        interest2=second,
        interest_lead=lead,
    )


_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")


def _render(text: str, values: dict[str, str]) -> str:
    """Substitute known placeholders; leave unknown `{x}` tokens untouched."""
    return _PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)


# ---------------------------------------------------------------------------
# Mission library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """One hand-written mission, before request-specific substitution.

    The boolean flags are not decoration: they are how the library stays
    feasible for a player who asked for step-free access, a low-energy day or
    a walk with kids. `app.ai.safety` promotes those mismatches to findings, so
    a template that ignores them would produce missions the gate rejects.

    `details` holds mission-level hints rather than one line per step: they are
    spread over the first steps, and later steps simply have no detail.
    """

    key: str
    category: str
    title: str
    hook: str
    premise: str
    steps: tuple[str, ...]
    #: Mission-level screen-free hints. Fewer than `steps`, in general: they
    #: are distributed over the first steps and the rest carry an empty detail.
    details: tuple[str, ...]
    minutes: tuple[int, ...]
    tags: tuple[str, ...]
    difficulty: int
    v_subject: str
    v_must: tuple[str, ...]
    v_accept: tuple[str, ...]
    v_reject: tuple[str, ...]
    v_ambiguous: tuple[str, ...]
    variant_key: str
    mobility_safe: bool = True
    low_energy: bool = True
    kid_safe: bool = True
    dog_friendly: bool = True
    indoors_ok: bool = False
    weather_fit: tuple[Weather, ...] = ()
    keywords: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


#: Optional premise add-ons, chosen per seed so repeat visits feel different.
_VARIANTS: dict[str, tuple[str, ...]] = {
    "bird": (
        "A pair of binoculars is nice to have, not required.",
        "Counting is allowed to be approximate; say so in your note.",
        "Staying still for a minute gets you more birds than moving about.",
    ),
    "tree": (
        "Bring a ruler or use your thumb-widths as the scale.",
        "The same species twice in a row is a useful comparison.",
        "Bark is readable all year; leaves are not.",
    ),
    "plant": (
        "Handle nothing wild -- look, photograph, leave it standing.",
        "A leaf pressed in a book is a fine keepsake if you took a fallen one.",
        "Smell is a legitimate ID clue and costs nothing.",
    ),
    "pollinator": (
        "Flowering planters near doorways are surprisingly productive.",
        "Stay on the paving; insects do not need you standing in the bed.",
        "One patch of nectar plants can host a surprising number of visitors.",
    ),
    "sky": (
        "Cumulus change shape fastest in the twenty minutes after they appear.",
        "High thin cloud often arrives before a change in the weather.",
        "Look up in several directions; one patch of sky tells you little.",
    ),
    "shadow": (
        "A stick held flat on the ground gives a clean shadow edge.",
        "Compare a shadow taken now with the same object later in your walk.",
        "Overcast light still casts a soft, usable shadow.",
    ),
    "wind": (
        "Hold the strip at arm's length and note how it moves.",
        "Big trees move before small ones; that gap is the interesting part.",
        "Wind you can hear but not feel is usually coming from elsewhere.",
    ),
    "wet": (
        "A dry lens matters more than any other piece of kit here.",
        "Reflections are sharpest in the ten minutes after the rain eases.",
        "Shoes matter; pick the pavements you are happy to stand on.",
    ),
    "sketch": (
        "Three minutes per sketch; the third one is always the best.",
        "Draw the shape, not the detail. Detail comes later if you want it.",
        "If your hands are cold, work smaller -- it is faster anyway.",
    ),
    "photo": (
        "Turn the screen off once you have the framing you want.",
        "One subject, many frames beats many subjects in one frame.",
        "Stand still after the frame; the good one is often the next one.",
    ),
    "sound": (
        "Close your eyes for half of it and listen rather than look.",
        "Continuous sound counts; a single car passing does not.",
        "Write times down as you go, not at the end.",
    ),
    "surface": (
        "Count only surfaces you actually put a foot on.",
        "Wet stone feels very different from dry stone -- note it.",
        "The same material in two places is worth comparing.",
    ),
    "history": (
        "Brick, render and stone tell you roughly how old a street is.",
        "House numbers often predate the doors they now sit on.",
        "You are reading the public face of a building, nothing more.",
    ),
    "notice": (
        "Read only what is already public and already there.",
        "Headings tell you what a neighbourhood is worried about.",
        "Photograph from an angle that keeps faces out of frame.",
    ),
    "detail": (
        "Doorbells, knockers, house numbers and lamps are the usual finds.",
        "Repetition is interesting: four houses, four versions of one idea.",
        "Look at eye level, then at knee level, then above head height.",
    ),
    "letter": (
        "Letterforms differ between decades far more than you expect.",
        "Old signage is often the most interesting kind on a street.",
        "Read whole words where you can; partial words mislead.",
    ),
    "grid": (
        "Count panes row by row and it takes under a minute per window.",
        "Shutters and blinds count as part of the design.",
        "Repeat designs are usually the work of one builder.",
    ),
    "mural": (
        "Get the whole piece in frame first, then move closer.",
        "Look for the repeated motif; that is usually the artist's signature.",
        "Faded paint is still worth photographing.",
    ),
    "tally": (
        "Counting only. Nothing here is for picking up or touching.",
        "Windblown items concentrate near edges and gaps.",
        "One category of packaging gives you a cleaner number than five.",
    ),
    "season": (
        "Bud, flower, seed and bare twig are four different seasons.",
        "The same plant can be at different stages two streets apart.",
        "Dead material still counts -- last year's cones and husks.",
    ),
    "ripple": (
        "Wind and passing craft both write on the surface; note which.",
        "Stand back a step or two and reflections get much cleaner.",
        "Reflections are always moving, so work quickly.",
    ),
    "sun": (
        "A compass app or a watch face both work well enough for this.",
        "Shadow direction beats shadow length for orientation.",
        "Check your shadow twice, ten minutes apart, and note the change.",
    ),
    "sniff": (
        "Short stops beat long ones; a dog's nose works on a static moment.",
        "Water break first, then the loop.",
        "Keep the lead slack and the route boring.",
    ),
    "colour": (
        "Kids pick the colours, grown-ups keep the tally.",
        "Counting out loud keeps everyone honest.",
        "The last colour of the day is the one they remember.",
    ),
    "atrium": (
        "Indoor light behaves predictably enough to measure.",
        "Look up: the structure overhead is usually the best part.",
        "Pick a spot where you are not blocking a walkway.",
    ),
    "museum": (
        "A shelf, a windowsill and a bollard make a fine temporary gallery.",
        "Look at the objects in pairs; a pair is much easier to describe.",
        "Everything you look at stays where it is.",
    ),
    "seat": (
        "Sit properly -- a bench is a piece of public design.",
        "Five minutes of stillness is the whole trick.",
        "Notice who else chose this spot and what they were looking at.",
    ),
    "moss": (
        "Green is not a colour, it is a whole community.",
        "Close-ups reward you more than wide shots here.",
        "Look on the shaded side of structures, not the sunny side.",
    ),
}


TEMPLATES: tuple[Template, ...] = (
    # -- nature observation ------------------------------------------------
    Template(
        key="bird-count",
        category="nature-observation",
        title="The {place_title} Bird Count",
        hook="Ten minutes, one pair of eyes, and the birds you have walked past all week.",
        premise="{interest_lead} This mission asks for numbers rather than species names, because counting is honest and naming is guesswork.",
        steps=(
            "Stand somewhere open on the {place} where you can see sky and at least one line of trees or roofs.",
            "Count every bird you can see for sixty full seconds. Do not try to name anything yet.",
            "Pick the three busiest and note in three words what each one was doing.",
            "Take one photo that shows your busiest bird against sky, leaves or open ground.",
        ),
        details=(
            "Binoculars help but are not needed; phone zoom is fine.",
            "Stay on the paving and keep other pedestrians in view.",
        ),
        minutes=(3, 5, 4, 3),
        tags=("birds", "nature", "outdoors", "counting"),
        difficulty=1,
        v_subject="One photograph of a live bird seen outdoors, framed against sky, foliage or open ground.",
        v_must=(
            "one clearly visible living bird",
            "the bird is outside and in its normal setting",
            "enough surrounding scene to place the bird",
        ),
        v_accept=(
            "a silhouette against bright sky",
            "a bird perched on a wire, post, branch or railing",
            "a group with one individual clearly separable",
        ),
        v_reject=(
            "an illustration, model or printed picture of a bird",
            "a dead bird, an egg, or a nest with the bird absent",
            "a photo with no bird in it",
        ),
        v_ambiguous=(
            "the bird is a blurred speck in the distance",
            "only a wing or tail is inside the frame",
        ),
        variant_key="bird",
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.WINDY),
        keywords=("bird", "birds", "birding", "nature", "wildlife", "outdoors"),
    ),
    Template(
        key="tree-id",
        category="plant-id",
        title="Bark and Branch Notes on the {place_title}",
        hook="Two trees, one close look each, and a much better eye for both.",
        premise="Tree identification is mostly texture and outline. {interest_lead}",
        steps=(
            "Pick a tree on the {place} you can stand beside without getting in anyone's way.",
            "Stand back and photograph the whole shape of it, trunk to crown.",
            "Move close and photograph the bark, filling the frame, without touching it.",
            "Find a second tree nearby, note how its bark differs, and take the same two photos of it.",
            "Write one line comparing the two: shape, bark texture, leaf arrangement.",
        ),
        details=(
            "Bark is readable all year; leaves are not.",
            "Nothing here needs picking up.",
        ),
        minutes=(2, 4, 5, 7, 4),
        tags=("trees", "botany", "nature", "field-notes"),
        difficulty=2,
        v_subject="A pair of photographs of two different mature trees, one showing full shape and one showing bark texture close up.",
        v_must=(
            "at least two separate trees are visible across the set",
            "one image shows a whole tree from base to crown",
            "one image shows bark texture filling much of the frame",
        ),
        v_accept=(
            "an evergreen and a deciduous tree side by side",
            "a tree beside a building wall",
            "winter branches with no leaves",
        ),
        v_reject=(
            "cut timber, logs or stacked wood",
            "a seedling too small to judge",
            "an indoor potted plant or a printed picture of a tree",
        ),
        v_ambiguous=(
            "both photos show the same tree and no comparison is possible",
            "the close-up is too dark to read any texture",
        ),
        variant_key="tree",
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.COLD, Weather.SNOW),
        keywords=("tree", "trees", "botany", "plants", "nature", "park", "forest"),
    ),
    Template(
        key="leaf-quest",
        category="plant-id",
        title="Three Unknown Leaves",
        hook="A botany scavenger hunt that does not need a botany degree.",
        premise="You do not need the species. You need three leaves that are honestly different from each other.",
        steps=(
            "Find three fallen leaves on the {place} -- fallen only, never picked.",
            "Photograph each one flat against a plain background, one leaf per photo.",
            "For each, note its edge (smooth, toothed, lobed) and its vein pattern.",
            "Rank the three from most unusual to least and say why in one sentence.",
        ),
        details=(
            "A sheet of paper makes the plain background.",
            "A pressed fallen leaf is a fine keepsake.",
        ),
        minutes=(5, 8, 6, 4),
        tags=("plants", "leaves", "botany", "nature"),
        difficulty=2,
        v_subject="Three photographs, each showing a single fallen leaf laid flat on a plain surface.",
        v_must=(
            "each image contains exactly one leaf",
            "the leaf fills most of the frame",
            "a plain, uncluttered background behind the leaf",
        ),
        v_accept=(
            "leaves of clearly different shapes",
            "a leaf with visible edge serrations",
            "a leaf still attached to a small twig",
        ),
        v_reject=(
            "leaves still on a growing plant",
            "an image of several leaves together",
            "a plastic or fabric leaf",
        ),
        v_ambiguous=(
            "leaf shape is not readable at the resolution provided",
            "the background is too busy to judge the edge",
        ),
        variant_key="plant",
        keywords=("plants", "leaf", "leaves", "botany", "nature", "flowers"),
    ),
    Template(
        key="pollinator",
        category="nature-observation",
        title="Pollinator Count on the {place_title}",
        hook="Count the insects, never the flowers. The insects are harder and better.",
        premise="Pollinator activity is a fair proxy for how healthy a planted strip is. {interest_lead}",
        steps=(
            "Find a patch of flowering plants on the {place} you can view from the paving.",
            "Stand still for sixty seconds and count every visiting insect.",
            "Repeat for a second patch at least fifty metres away.",
            "Note which patch had more visits and which flower colour was most popular.",
            "Take one photo of an insect on a flower, in focus, filling the frame.",
        ),
        details=(
            "Give every insect a wide berth; no swatting, no feeding.",
            "If you are allergic, watch from further away -- the count still works.",
        ),
        minutes=(4, 4, 6, 5, 4),
        tags=("insects", "pollinators", "nature", "garden"),
        difficulty=2,
        v_subject="One photograph of a pollinating insect resting on a flower, in focus.",
        v_must=(
            "a single insect is clearly in focus",
            "the insect is on a flower or flowering plant",
            "the setting is outdoors",
        ),
        v_accept=(
            "a bee, hoverfly, butterfly, moth or beetle on a flower",
            "a wasp on foliage rather than a flower",
            "an insect feeding with its head inside the flower",
        ),
        v_reject=(
            "a wasp nest or an insect colony",
            "a dead or trapped insect",
            "a photo of a flowering plant with no insect visible",
        ),
        v_ambiguous=(
            "the insect is present but too small to identify",
            "a blurred streak where an insect passed",
        ),
        variant_key="pollinator",
        weather_fit=(Weather.CLEAR, Weather.HEAT, Weather.CLOUDY),
        keywords=("insects", "bees", "pollinator", "garden", "nature", "flowers"),
    ),
    Template(
        key="cloud-notebook",
        category="weather-watching",
        title="Cloud Shape Notebook",
        hook="Sky-watching from the pavement, which is a hobby nobody charges you for.",
        premise="Cloud shape is the cheapest weather forecast available. This mission takes {time_phrase} and no equipment.",
        steps=(
            "Find the most open patch of sky visible from the {place}.",
            "Lie back or tilt your head up and describe the first cloud you see in five words.",
            "Pick three clouds and sketch their outlines as they actually are, not as you remember them.",
            "Look at the same patch again ten minutes later and note what changed.",
            "Take one photo of the sky including a building or treeline for scale.",
        ),
        details=(
            "Keep your neck comfortable; look in more than one direction.",
            "A phone photo of a bright sky needs the exposure nudged down.",
        ),
        minutes=(3, 4, 8, 3, 4),
        tags=("sky", "clouds", "weather", "nature"),
        difficulty=1,
        v_subject="One photograph of the sky that includes a horizon reference such as a treeline, rooftopline or building.",
        v_must=(
            "the sky is the main subject",
            "a treeline, building or similar horizon reference is visible",
            "cloud detail is visible rather than a blank white field",
        ),
        v_accept=(
            "any cloud type over a horizon",
            "a partly clouded sky at sunset-like light",
            "a narrow strip of sky between buildings",
        ),
        v_reject=(
            "an image that is only ground",
            "a screenshot of a weather app",
            "a fully overcast white frame with no reference object",
        ),
        v_ambiguous=(
            "the sky is blown out to white by exposure",
            "no horizon reference can be found in the frame",
        ),
        variant_key="sky",
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.COLD),
        keywords=("sky", "clouds", "weather", "nature", "photography"),
    ),
    Template(
        key="shadow-study",
        category="weather-watching",
        title="Shadow Length Study",
        hook="A ruler, a stick and ten minutes of measuring that tells you the hour.",
        premise="Shadows are the oldest sundial and they still work. {interest_lead}",
        steps=(
            "Choose a straight stick or pole you are allowed to touch on the {place}.",
            "Hold it flat against the ground and photograph the shadow it casts.",
            "Estimate the shadow length in shoe-widths before measuring it.",
            "Repeat the same photograph twenty minutes later, in the same spot.",
            "Compare the two and note whether the shadow grew or shrank.",
        ),
        details=(
            "Never place yourself in the path of anyone else's shadow study; you are not the only one.",
            "Overcast light gives a soft edge, which is still usable.",
        ),
        minutes=(3, 4, 3, 5, 3),
        tags=("shadows", "sun", "weather", "measuring"),
        difficulty=2,
        v_subject="A photograph of a distinct cast shadow on the ground, with the object casting it at least partly in frame.",
        v_must=(
            "a recognisable cast shadow is visible",
            "the shadow falls on a flat surface",
            "either the object or part of the scene gives scale",
        ),
        v_accept=(
            "a human shadow with legs apart",
            "a railing or lamp-post shadow across paving",
            "a tree shadow on a wall",
        ),
        v_reject=(
            "an indoor shadow from a lamp",
            "a graphic or filter pretending to be a shadow",
            "an overcast frame with no distinct shadow",
        ),
        v_ambiguous=(
            "the shadow is present but its edges are unreadable",
            "the frame only shows a shadow with no casting object at all",
        ),
        variant_key="shadow",
        weather_fit=(Weather.CLEAR, Weather.COLD, Weather.CLOUDY),
        keywords=("shadow", "sun", "light", "weather", "science"),
    ),
    Template(
        key="wind-report",
        category="weather-watching",
 title="The Ten-Minute Wind Report",
        hook="Cheap strips of paper turn an invisible thing into something you can measure.",
        premise="Wind is a real force with a direction and a rough speed. This mission gives you both in {time_phrase}.",
        steps=(
            "Tear one strip of paper roughly the length of your hand.",
            "Hold it at arm's length on the {place} and watch which way it points for sixty seconds.",
            "Compare with a second open spot: does it agree, or does it disagree?",
            "Estimate the strength on a scale of one to five from how hard the strip pulls.",
            "Note what tall things are doing -- trees usually answer before you notice.",
        ),
        details=(
            "Nothing loose goes in the air, ever.",
            "If the strip is torn from a sign or notice, leave it alone and use a receipt instead.",
        ),
        minutes=(2, 4, 5, 3, 4),
        tags=("wind", "weather", "measuring"),
        difficulty=1,
        v_subject="One photograph of a small strip of paper being held up outdoors and visibly moved by wind.",
        v_must=(
            "a strip or scrap of paper is visible",
            "it is outdoors",
            "the strip is lifted off the ground, not lying down",
        ),
        v_accept=(
            "the strip angled or curved by wind",
            "a hand holding the strip against a sky or treeline",
            "a receipt or ticket rather than plain paper",
        ),
        v_reject=(
            "a strip lying on a table or the ground",
            "a paper aeroplane or paper boat",
            "an indoor shot",
        ),
        v_ambiguous=(
            "the strip is visible but its direction cannot be read",
            "the hand holding it fills the frame",
        ),
        variant_key="wind",
        weather_fit=(Weather.WINDY, Weather.CLEAR, Weather.CLOUDY),
        keywords=("wind", "weather", "science", "measuring"),
    ),
    Template(
        key="wet-reflections",
        category="photography",
        title="Wet Window Reflections",
        hook="Rain is a gift to anyone who owns a phone camera.",
        premise="After rain, every dark shopfront becomes a mirror. {interest_lead}",
        steps=(
            "Walk a short stretch of the {place} looking for a dark, flat surface: a window or a polished wall.",
            "Stand so a sign, a lamp or a building across the way falls inside the reflection.",
            "Frame it so the real world is at the edge and the reflection fills most of the picture.",
            "Try three different surfaces and keep the one with the clearest reflection.",
            "Take the final photograph, then wipe the lens and check it.",
        ),
        details=(
            "A dry cloth matters more than any other piece of kit here.",
            "Stand on dry paving if you can; wet stone is slippery.",
            "Reflections disappear fast, so have the phone ready before you stop.",
        ),
        minutes=(4, 5, 6, 5, 4),
        tags=("photography", "rain", "reflections", "city"),
        difficulty=2,
        v_subject="One photograph dominated by a reflection in a wet window, polished wall or similar flat surface.",
        v_must=(
            "a reflection is the main subject",
            "the reflecting surface is wet or glossy",
            "the reflected subject is readable",
        ),
        v_accept=(
            "a sign or shop name mirrored in a window",
            "a street scene reflected in a parked car's wet flank",
            "a reflection broken by a few drops",
        ),
        v_reject=(
            "a normal photograph with no reflection",
            "a reflection created with an editing filter",
            "an indoor photo of a screen",
        ),
        v_ambiguous=(
            "the surface is wet but nothing readable is reflected",
            "the reflection is too distorted to identify",
        ),
        variant_key="wet",
        weather_fit=(Weather.RAIN,),
        keywords=("photography", "camera", "photo", "rain", "city", "art"),
    ),
    # -- drawing ----------------------------------------------------------
    Template(
        key="ink-sketches",
        category="sketching",
        title="Three-Minute Ink Sketches",
        hook="Three sketches, three minutes each, and no erasing.",
        premise="Speed kills the inner critic. Three minutes each forces a decision about what actually matters. {interest_lead}",
        steps=(
            "Pick a subject on the {place}: a doorway, a bollard, a bin, a bench end.",
            "Sketch it for three minutes without lifting more than you have to. No erasing.",
            "Stand thirty steps further away and sketch the same subject again for three minutes.",
            "Do a third, choosing deliberately the most awkward angle available.",
            "Photograph all three sketches laid out side by side.",
        ),
        details=(
            "A biro, a pencil and a flattened receipt all work.",
            "Cold hands mean small drawings; that is fine.",
        ),
        minutes=(3, 5, 5, 5, 4),
        tags=("sketching", "drawing", "art", "creative"),
        difficulty=2,
        v_subject="One photograph of hand-drawn sketches on paper, ideally all three visible together.",
        v_must=(
            "at least one hand-drawn sketch is visible",
            "the sketch is clearly made by hand, on paper",
            "the drawn subject can be identified from the lines",
        ),
        v_accept=(
            "a single detailed sketch",
            "several sketches photographed together",
            "a sketch on a notebook page with writing around it",
        ),
        v_reject=(
            "a printed or digital drawing",
            "an abstract scribble with no readable subject",
            "a photograph of an artwork someone else made",
        ),
        v_ambiguous=(
            "the sketch is visible but too small to identify the subject",
            "pencil lines are too faint to read",
        ),
        variant_key="sketch",
        keywords=("sketch", "sketching", "drawing", "art", "creative", "design"),
    ),
    Template(
        key="pocket-sketch-map",
        category="sketching",
        title="Map What You Can See in Ten Minutes",
        hook="A rough map beats a perfect one, and a rough map still works.",
        premise="You do not need cartography skills to draw a usable map. {interest_lead}",
        steps=(
            "Stand at a spot on the {place} with a clear view in at least three directions.",
            "Draw anything within a five-minute walk, marking directions from the sun and your entry point.",
            "Label the things a stranger would need: the water fountain, the corner shop, the noticeboard.",
            "Walk the route you have just drawn and correct anything you got wrong.",
            "Photograph the finished map flat, with the page filling the frame.",
        ),
        details=(
            "A folded receipt works if you have no paper.",
            "Never cross where you would have to step off the pavement to do it safely.",
        ),
        minutes=(4, 12, 6, 8, 3),
        tags=("sketching", "map", "drawing", "orientation"),
        difficulty=3,
        v_subject="One photograph of a hand-drawn map on paper showing at least three labelled landmarks.",
        v_must=(
            "a hand-drawn map is visible",
            "at least three labelled landmarks appear",
            "the page is legible and fills most of the frame",
        ),
        v_accept=(
            "a schematic map rather than a true-to-scale one",
            "a map with a north arrow drawn on it",
            "a route drawn as a path rather than a plan view",
        ),
        v_reject=(
            "a printed map",
            "a screenshot of a maps app",
            "a blank or near-blank page",
        ),
        v_ambiguous=(
            "labels are illegible at the given resolution",
            "the frame shows paper with no distinguishable map",
        ),
        variant_key="sketch",
        mobility_safe=False,
        keywords=("map", "sketch", "drawing", "orientation", "navigation"),
    ),
    # -- photography ------------------------------------------------------
    Template(
        key="one-colour",
        category="photography",
        title="Six Frames of One Colour",
        hook="Restrict the palette and the composition has to do the work.",
        premise="A single-colour constraint is the cheapest creative brief there is. {interest_lead}",
        steps=(
            "Pick one colour from the {place} and commit to it: red, green, blue, yellow or white.",
            "Photograph six separate subjects in that colour, one per frame.",
            "Vary your distance -- one close, one across the space, one from above or below.",
            "Delete any frame where the colour only appears by accident, and shoot a replacement.",
            "Put your five or six best in order and photograph the sequence laid out.",
        ),
        details=(
            "Getting the whole set is the point; the good frame is often number five.",
            "This works indoors too if the weather turns.",
        ),
        minutes=(4, 10, 6, 6, 4),
        tags=("photography", "camera", "photo", "colour", "art"),
        difficulty=2,
        v_subject="A set of photographs in which the same single colour is the dominant subject across several separate frames.",
        v_must=(
            "at least three separate frames are shown",
            "one colour dominates across the frames",
            "the subjects differ from each other",
        ),
        v_accept=(
            "the frames are laid out together on a surface",
            "the colour appears as an accent rather than a fill",
            "some frames are close-ups and others are wide",
        ),
        v_reject=(
            "a single photograph only",
            "a filter applied to an unrelated scene",
            "frames where the colours have nothing in common",
        ),
        v_ambiguous=(
            "only one frame is visible",
            "the shared colour cannot be identified from the images",
        ),
        variant_key="photo",
        indoors_ok=True,
        keywords=("photography", "camera", "photo", "art", "colour", "creative"),
    ),
    Template(
        key="texture-handshake",
        category="sensory",
        title="The Texture Handshake",
        hook="Four public surfaces, four completely different feelings.",
        premise="Touch is the sense people use least when they are out. This mission is a controlled experiment in noticing it. {interest_lead}",
        steps=(
            "Find four surfaces you can touch on the {place}: brick, metal, wood, stone, glass.",
            "Press your palm flat on each for five seconds and rate it one to five for cold and one to five for rough.",
            "Notice whether the temperature changes across the four.",
            "Photograph your hand resting on the two that rated most different.",
            "Write one line: which surface would you most want to lean against?",
        ),
        details=(
            "Public surfaces only -- touch nothing private, nothing sharp, nothing that might be hot.",
            "If something is painted wet or a work site, skip it.",
        ),
        minutes=(4, 8, 4, 5, 3),
        tags=("sensory", "touch", "observation", "creative"),
        difficulty=1,
        v_subject="One photograph of a hand resting flat on an ordinary outdoor public surface such as brick, stone, metal or wood.",
        v_must=(
            "a human hand is in contact with a surface",
            "the surface material is identifiable",
            "the setting is outdoors",
        ),
        v_accept=(
            "two hands on two contrasting surfaces in one frame",
            "a hand beside a surface for scale rather than touching",
            "a fingertip close-up of a textured surface",
        ),
        v_reject=(
            "a hand holding a phone or a screenshot",
            "a surface indoors only",
            "a bare surface with no hand at all",
        ),
        v_ambiguous=(
            "a hand is present but not clearly in contact",
            "the surface material cannot be identified",
        ),
        variant_key="surface",
        keywords=("sensory", "touch", "senses", "observation", "creative"),
    ),
    Template(
        key="five-senses",
        category="sensory",
        title="Five Senses, One Street Corner",
        hook="One corner, one minute per sense, and a much better memory afterwards.",
        premise="Attention is the whole trick. This mission turns a corner you have used a thousand times into an event. {interest_lead}",
        steps=(
            "Stand at a corner of the {place} and close your eyes for sixty seconds. Note everything you hear.",
            "Open your eyes and name three things you can see that you have never noticed there.",
            "Ask your nose one question: what is the nearest distinct smell?",
            "Feel the temperature on your face and on your feet -- do they agree?",
            "Take one photo of the corner as you found it, then write your five-sentence record.",
        ),
        details=(
            "Stay out of other people's way while you stand still.",
            "Do not sample, touch or pick anything; looking and smelling is enough.",
        ),
        minutes=(4, 6, 3, 3, 5),
        tags=("sensory", "observation", "mindfulness", "noticing"),
        difficulty=1,
        v_subject="One photograph of the corner or spot where the five-senses record was made.",
        v_must=(
            "an outdoor street, path or plaza corner is visible",
            "the frame shows more than a blank wall",
            "the location is recognisable as somewhere you stand and pause",
        ),
        v_accept=(
            "a wide view of the corner",
            "a detail of a doorway, post or bin at the corner",
            "a view looking down the street from the corner",
        ),
        v_reject=(
            "a selfie",
            "an indoor room or corridor",
            "a blurred image with no identifiable location",
        ),
        v_ambiguous=(
            "the corner is visible but too dark to read",
            "the frame shows only a slice of wall",
        ),
        variant_key="museum",
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.RAIN),
        keywords=("sensory", "senses", "mindfulness", "observation", "noticing"),
    ),
    # -- sound ------------------------------------------------------------
    Template(
        key="sound-map",
        category="sound-mapping",
        title="Sixty Seconds of Sound",
        hook="Close your eyes and your street turns into a map.",
        premise="Sound maps show you a place's shape far more honestly than a photograph does. {interest_lead}",
        steps=(
            "Stand at one spot on the {place}, close your eyes and hold still for sixty seconds.",
            "Every distinct sound gets a timestamp and one word: birds, voices, machinery, leaves, bells.",
            "Walk to a second spot about fifty metres away and repeat.",
            "Repeat once more at a third spot if you have the time.",
            "Write which spot was loudest, quietest, and which sound appeared at all three.",
        ),
        details=(
            "Keep to the side of the path so people can pass.",
            "Never stop in a doorway or a narrow passage -- stand aside and let people by.",
        ),
        minutes=(4, 5, 6, 6, 4),
        tags=("sound", "observation", "mindfulness", "map"),
        difficulty=2,
        v_subject="One photograph of the place where a sound recording was taken, wide enough that the surroundings are recognisable.",
        v_must=(
            "an outdoor location is visible",
            "the scene is recognisable as a specific place",
            "the frame is steady rather than a blurred movement shot",
        ),
        v_accept=(
            "a street, path or plaza view",
            "a view toward a building frontage",
            "a park or open area with trees",
        ),
        v_reject=(
            "a selfie",
            "a screen showing an audio app",
            "an indoor space",
        ),
        v_ambiguous=(
            "the location is visible but too generic to place",
            "the frame is heavily blurred",
        ),
        variant_key="sound",
        keywords=("sound", "audio", "observation", "mindfulness", "music"),
    ),
    Template(
        key="sound-compare",
        category="sound-mapping",
        title="One Sound, Three Places",
        hook="Find a sound that exists on your street and time how long it lasts.",
        premise="The most interesting sound on any street is the one that stops. {interest_lead}",
        steps=(
            "Stay on the {place} and listen for a repeating sound -- a chime, a bird, a gate, a machine cycle.",
            "Time five repetitions with your phone's stopwatch.",
            "Note how long the gaps are between them.",
            "Walk one street-length over and time five more there.",
            "Compare the two: same rhythm, or completely different?",
        ),
        details=(
            "Chiming and machinery are usually near buildings; stay on public ground.",
            "If you cannot tell the pattern after five tries, choose a different sound.",
        ),
        minutes=(5, 6, 4, 8, 4),
        tags=("sound", "observation", "pattern", "science"),
        difficulty=2,
        v_subject="One photograph of the location where a repeating sound was timed.",
        v_must=(
            "an outdoor place is visible",
            "something that could plausibly make a sound is in frame",
            "the image is steady and legible",
        ),
        v_accept=(
            "a building frontage",
            "a lamp post, gate or sign",
            "an open street view",
        ),
        v_reject=(
            "a close-up portrait",
            "an indoor scene",
            "a photo of headphones or a speaker",
        ),
        v_ambiguous=(
            "the location is not identifiable",
            "nothing that could produce sound is visible",
        ),
        variant_key="sound",
        keywords=("sound", "audio", "observation", "pattern", "science"),
    ),
    # -- urban noticing / history ----------------------------------------
    Template(
        key="doorstep-census",
        category="neighborhood-history",
        title="The Doorstep Time Capsule",
        hook="Doorbells, knockers and house numbers: a hundred years of design in one street.",
        premise="You are reading the public face of buildings -- the part built to be seen from outside. {interest_lead}",
        steps=(
            "Pick one stretch of the {place} and walk it slowly, looking only at the lower two metres of the buildings.",
            "Photograph four doorways: doorknocker, doorbell, house number, letterbox.",
            "For each, note the material and whether it looks original or replaced.",
            "Guess the decade for the oldest-looking one and write the guess down.",
            "Photograph the two oldest side by side.",
        ),
        details=(
            "Stay on the pavement and look, do not enter.",
            "Keep faces and house numbers out of any wider group shot.",
        ),
        minutes=(6, 8, 6, 4, 4),
        tags=("architecture", "history", "noticing", "detail"),
        difficulty=2,
        v_subject="One photograph of a doorway detail on a residential building: a doorknocker, doorbell, letterbox or house number.",
        v_must=(
            "a doorway or its immediate surround is visible",
            "at least one metal or timber fitting is identifiable",
            "the building is a real structure, not a model",
        ),
        v_accept=(
            "a close-up of a house number",
            "a pair of knockers compared side by side",
            "a letterbox with a worn paint finish",
        ),
        v_reject=(
            "a flat interior door",
            "a shop front with the shutters down",
            "an image with no doorway detail readable",
        ),
        v_ambiguous=(
            "a doorway is present but the fitting cannot be identified",
            "the photo is too dark to read any detail",
        ),
        variant_key="detail",
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.RAIN),
        keywords=("history", "architecture", "buildings", "noticing", "detail", "heritage"),
    ),
    Template(
        key="noticeboard-survey",
        category="community-noticing",
        title="What the Noticeboard Is Worried About",
        hook="A public noticeboard tells you what a neighbourhood is actually thinking.",
        premise="Community boards are public information and they are usually more honest than a survey. {interest_lead}",
        steps=(
            "Find a public noticeboard or community display on the {place}.",
            "Stand back and photograph the whole board before reading anything closely.",
            "Read the headings only and tally which subject appears most: lost pets, classes, offers, warnings.",
            "Note the date of the oldest notice still pinned up.",
            "Photograph the single most interesting notice, with no personal details legible.",
        ),
        details=(
            "Read only what is already public. Do not remove or add anything.",
            "Step aside for anyone reading it -- this is their board too.",
            "If someone's details are clearly readable, do not photograph that notice.",
        ),
        minutes=(4, 4, 6, 4, 5),
        tags=("community", "noticing", "history", "observation"),
        difficulty=1,
        v_subject="One photograph of a public noticeboard or community display board showing notices pinned to it.",
        v_must=(
            "a noticeboard or display panel is visible",
            "at least two separate notices are visible",
            "the scene is outdoors or in a public indoor area",
        ),
        v_accept=(
            "a single large notice on a board",
            "a school or library display board",
            "a cafe or shop community board",
        ),
        v_reject=(
            "a phone screen showing a photo of a board",
            "an office whiteboard inside a private space",
            "a board where no notice is readable at all",
        ),
        v_ambiguous=(
            "notices are visible but all text is blurred",
            "what looks like a board cannot be confirmed",
        ),
        variant_key="notice",
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.RAIN, Weather.HEAT),
        keywords=("community", "noticing", "history", "observation", "people"),
    ),
    Template(
        key="letter-hunt",
        category="typography",
        title="Letter and Sign Hunt",
        hook="Typeface tells you the decade faster than architecture does.",
        premise="Signs are the most honest record of what a street needed to say. {interest_lead}",
        steps=(
            "Walk the {place} looking only at lettering: shop names, street plates, plaques, hand-painted signs.",
            "Photograph three signs that are clearly different in style.",
            "Identify the oldest lettering style you can and say why it looks older.",
            "Note the one sign that is a deliberate mismatch with its building.",
            "Photograph your favourite sign square-on so the lettering is level.",
        ),
        details=(
            "Do not touch signs or anything attached to a building.",
            "Photograph from the pavement only.",
        ),
        minutes=(5, 8, 6, 5, 4),
        tags=("typography", "signs", "art", "history", "noticing"),
        difficulty=2,
        v_subject="One photograph of lettering on a sign, taken square-on so the text is readable.",
        v_must=(
            "lettering is clearly visible",
            "the sign is photographed close enough to read the style",
            "the sign is attached to a real building or structure outdoors",
        ),
        v_accept=(
            "a hand-painted shop name",
            "a cast metal street plate",
            "a modern sans-serif sign",
        ),
        v_reject=(
            "a printed page or a screen",
            "a sign where only part of the lettering is in frame",
            "a blank fascia",
        ),
        v_ambiguous=(
            "lettering is visible but the style cannot be judged",
            "the sign is too small in the frame",
        ),
        variant_key="letter",
        keywords=("typography", "signs", "design", "art", "history", "letters"),
    ),
    Template(
        key="window-grid",
        category="architecture",
        title="Window Grid Inventory",
        hook="Count the panes. The pattern is the building's fingerprint.",
        premise="Window grids are built from a limited vocabulary, so repetition is information. {interest_lead}",
        steps=(
            "Choose one building on the {place} and find four windows on the same floor.",
            "Count the panes in each window and write the four numbers down.",
            "Note the material: wood, metal or stone frames.",
            "Work out whether the ground floor matches the floor above it.",
            "Photograph two adjacent windows so the comparison is in one frame.",
        ),
        details=(
            "Both feet stay on the pavement; a phone held at arm's length is plenty.",
            "Never lean out over anything to get the angle.",
        ),
        minutes=(5, 8, 4, 5, 4),
        tags=("architecture", "counting", "pattern", "buildings"),
        difficulty=2,
        v_subject="One photograph containing at least two windows of the same building, side by side.",
        v_must=(
            "at least two windows are visible",
            "the windows share the same building facade",
            "the pane divisions are readable",
        ),
        v_accept=(
            "two identical grid patterns",
            "two clearly different patterns",
            "windows with shutters or blinds",
        ),
        v_reject=(
            "a shop front interior",
            "a single window only",
            "a window in a photograph or poster",
        ),
        v_ambiguous=(
            "two windows are visible but their grids cannot be counted",
            "the windows belong to different buildings",
        ),
        variant_key="grid",
        mobility_safe=False,
        keywords=("architecture", "buildings", "pattern", "counting", "city"),
    ),
    Template(
        key="mural-survey",
        category="urban-sketching",
        title="Mural Detail Survey",
        hook="Get the whole piece, then get one square centimetre of it.",
        premise="Public art rewards the wide shot and the close shot in equal measure. {interest_lead}",
        steps=(
            "Find a mural, painted panel or decorated wall on the {place} that is on public ground.",
            "Stand back until the whole piece is in frame and take that photograph first.",
            "Find the repeated motif -- there is almost always one -- and photograph it close.",
            "Note the two or three colours used most.",
            "Write down what the piece appears to be about, in one sentence.",
        ),
        details=(
            "Only photograph art that is meant to be seen from public space.",
            "Never touch or lean on painted walls.",
        ),
        minutes=(5, 6, 6, 5, 4),
        tags=("art", "photography", "city", "noticing"),
        difficulty=2,
        v_subject="One photograph of a piece of public art -- a mural, painted panel or decorated wall -- seen from public space.",
        v_must=(
            "painted or decorated public art is the main subject",
            "the piece is seen from ground level",
            "the artwork is complete or substantially visible",
        ),
        v_accept=(
            "a mural on a retaining wall",
            "a painted utility box",
            "a tiled or mosaic panel",
        ),
        v_reject=(
            "art inside a gallery that required entry",
            "a gallery wall label",
            "graffiti on private property behind a barrier",
        ),
        v_ambiguous=(
            "painted surface is visible but the artwork cannot be identified",
            "the frame shows only a fragment with no context",
        ),
        variant_key="mural",
        weather_fit=(Weather.CLEAR, Weather.CLOUDY),
        keywords=("art", "street", "photography", "city", "creative", "sketch"),
    ),
    Template(
        key="discard-tally",
        category="community-noticing",
        title="Discarded Packaging Tally",
        hook="Count what gets left behind. Nothing gets picked up.",
        premise="Where rubbish collects tells you where the wind and the crowd push things. {interest_lead}",
        steps=(
            "Choose a stretch of the {place} about twenty paces long.",
            "Stand still and count every discarded item you can see, by type: cups, bottles, wrappers, newspapers.",
            "Mark the two spots where the count is highest without picking anything up.",
            "Note whether the items are concentrated at edges, against walls or in open areas.",
            "Take one photograph showing the densest spot from a respectful distance.",
        ),
        details=(
            "Counting only. Do not pick anything up, and do not handle anything.",
            "Nothing sharp, broken or with a sharp edge goes anywhere near your hands.",
        ),
        minutes=(4, 8, 5, 5, 4),
        tags=("environment", "counting", "community", "noticing"),
        difficulty=1,
        v_subject="One photograph of an outdoor area showing several discarded items on the ground.",
        v_must=(
            "the scene is outdoors at ground level",
            "at least two discarded items are identifiable",
            "the surrounding context of the spot is visible",
        ),
        v_accept=(
            "a bin overflowing with items",
            "items gathered against a wall or kerb",
            "a windblown line of items along an edge",
        ),
        v_reject=(
            "a recycling bin in an empty street",
            "an indoor scene",
            "a photo where no discarded item is visible",
        ),
        v_ambiguous=(
            "items are visible but it is unclear whether they are discarded",
            "the frame is too blurred to identify anything",
        ),
        variant_key="tally",
        notes=("Counting only -- do not pick anything up.",),
        keywords=("environment", "litter", "community", "counting", "clean"),
    ),
    Template(
        key="time-capsule",
        category="neighborhood-history",
        title="Front Step Time Capsule",
        hook="One doorstep, one layer of history, and three clues that give it away.",
        premise="Building materials change slowly. Reading them is a quiet way to date a street without any research. {interest_lead}",
        steps=(
            "Stand on the pavement on the {place} and choose one house front you can see fully from the street.",
            "Photograph the front from across the way, keeping the whole facade in frame.",
            "Note the brick or stone colour, the tiles if visible, and the window proportions.",
            "Write your best guess at the build decade and the single clue it rests on.",
            "Zoom in on the most interesting detail as your second photo.",
        ),
        details=(
            "Public view only. Do not approach a door, try a bell or enter any garden.",
            "Keep front doors and house numbers readable only if that feels appropriate; there is no need to photograph them.",
        ),
        minutes=(4, 5, 6, 5, 4),
        tags=("history", "architecture", "observation", "detail"),
        difficulty=2,
        v_subject="One photograph of the front facade of a residential building, seen from across the street.",
        v_must=(
            "a building front is the main subject",
            "brick, stone, render or siding is visible",
            "the facade is recognisable as a home rather than a shop",
        ),
        v_accept=(
            "a brick terrace house",
            "a rendered or stucco front",
            "a stone frontage with an arched opening",
        ),
        v_reject=(
            "a shop frontage",
            "an office block or industrial unit",
            "a photograph of a property listing",
        ),
        v_ambiguous=(
            "a facade is visible but the materials cannot be read",
            "the building is too distant to judge",
        ),
        variant_key="history",
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.COLD),
        keywords=("history", "heritage", "architecture", "buildings", "observation"),
    ),
    # -- outdoor skills ---------------------------------------------------
    Template(
        key="sun-path",
        category="navigation",
        title="Sun Path Check",
        hook="Which way is north, without asking a phone.",
        premise="The sun is a compass that happens to be free and always on schedule. {interest_lead}",
        steps=(
            "Find an open spot on the {place} with an unobstructed patch of sky.",
            "Stand still and note where the sun is relative to you right now.",
            "Hold your arm out at shoulder height, pointing your right hand along the sun's direction, and note the time.",
            "Twenty minutes later, repeat and note whether the sun has moved left or right.",
            "Use that movement to estimate north, then check it against a compass app.",
        ),
        details=(
            "Never look directly at the sun.",
            "This is an estimate, not navigation -- tell someone where you are setting off from.",
        ),
        minutes=(3, 4, 5, 5, 4),
        tags=("navigation", "sun", "orientation", "science"),
        difficulty=3,
        v_subject="One photograph of an open sky view taken from the spot where the sun-path check was made.",
        v_must=(
            "open sky is visible",
            "the horizon or a landmark gives orientation",
            "the frame is level rather than tilted to the sky",
        ),
        v_accept=(
            "a horizon with the sun just out of frame",
            "a view across open ground",
            "a street view used for the landmark reference",
        ),
        v_reject=(
            "a photo taken straight up into empty sky",
            "a sunset over water",
            "a screen showing a compass app",
        ),
        v_ambiguous=(
            "the sky is blown out and nothing is readable",
            "no horizon or landmark is visible",
        ),
        variant_key="sun",
        weather_fit=(Weather.CLEAR,),
        keywords=("navigation", "compass", "sun", "orientation", "science", "outdoors"),
    ),
    Template(
        key="ripple-watch",
        category="waterfront",
        title="Ripple Watch on the {place_title}",
        hook="A surface that is never still is worth ten minutes of attention.",
        premise="Movement on a surface is a free physics lesson. {interest_lead}",
        steps=(
            "Find a vantage point along the {place} where you can see a wide area of the surface without getting close to the edge.",
            "Stand still for sixty seconds and count the distinct ripple sources: wind, craft, birds, thrown stones.",
            "Note the interval between gusts of ripple, in seconds.",
            "Compare one sheltered spot with one more exposed spot.",
            "Photograph the wider view with the ripples clearly visible across it.",
        ),
        details=(
            "Stay behind any railing or barrier and well back from the edge.",
            "Nothing goes in, and nothing is thrown in.",
            "Watch footing on wet stone.",
        ),
        minutes=(4, 6, 5, 7, 4),
        tags=("water", "observation", "science", "photography"),
        difficulty=2,
        v_subject="One photograph of a wide expanse of moving water seen from a safe, set-back vantage point.",
        v_must=(
            "an expanse of water fills a meaningful part of the frame",
            "ripples or surface movement are visible",
            "the camera position appears to be on solid public ground",
        ),
        v_accept=(
            "a harbour or basin with craft in it",
            "a canal or river surface with current",
            "a lake surface with reflected light on it",
        ),
        v_reject=(
            "a person in the water",
            "a swimming pool",
            "a photograph of water on a screen",
        ),
        v_ambiguous=(
            "water is visible but no ripple detail is readable",
            "it is not possible to tell the water is outdoors",
        ),
        variant_key="ripple",
        mobility_safe=False,
        kid_safe=False,
        dog_friendly=False,
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.WINDY),
        keywords=("water", "river", "sea", "ocean", "nature", "photography"),
    ),
    Template(
        key="moss-survey",
        category="plant-id",
        title="Moss and Lichen Survey",
        hook="Green fuzz is a whole community, and it grows where nothing else can.",
        premise="Moss and lichen are the slow indicators of a place. {interest_lead}",
        steps=(
            "Look along the shaded side of structures on the {place} -- walls, kerbs, the bases of trees.",
            "Find three patches of moss or lichen at least two metres apart.",
            "Photograph each patch close enough that its texture is readable.",
            "Note the surface each one is growing on: stone, brick, metal or bark.",
            "Rank the three by how much green they cover and write one line about the top one.",
        ),
        details=(
            "Look only. Do not scrape, brush or take samples.",
            "If a surface is freshly painted or has recent works, skip it.",
        ),
        minutes=(5, 7, 8, 5, 4),
        tags=("moss", "lichen", "plants", "nature", "observation"),
        difficulty=2,
        v_subject="One close photograph of a patch of moss or lichen growing on a hard surface such as stone, brick, metal or bark.",
        v_must=(
            "green moss or pale lichen is the main subject",
            "the growing surface is identifiable",
            "the patch fills a substantial part of the frame",
        ),
        v_accept=(
            "crustose lichen on stone",
            "moss in a mortar joint",
            "foliose lichen on a wall",
        ),
        v_reject=(
            "ivy or a small herbaceous plant",
            "moss photographed through a window",
            "an artificial plant or a model",
        ),
        v_ambiguous=(
            "green growth is visible but cannot be identified as moss or lichen",
            "the surface texture is too soft to read",
        ),
        variant_key="moss",
        mobility_safe=False,
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.RAIN, Weather.COLD),
        keywords=("moss", "lichen", "plants", "botany", "nature", "micro"),
    ),
    Template(
        key="season-signpost",
        category="nature-observation",
        title="Season Signpost",
        hook="The season is not a date. It is four things growing at once.",
        premise="You can read the local season better from bud, flower, fruit and bare twig than from any calendar. {interest_lead}",
        steps=(
            "Find one well-established plant or tree on the {place} that you can see from the pavement.",
            "Check for four signs: new buds, open flowers, fruit or seed, and bare twigs.",
            "Photograph whatever is present, close up.",
            "Find a second plant of a different kind and check the same four signs.",
            "Write which sign appears on both and which appears on only one.",
        ),
        details=(
            "Use fallen material or a hand lens; do not break anything to look.",
            "Some plants are protected locally -- if in doubt, photograph only.",
        ),
        minutes=(5, 6, 5, 7, 5),
        tags=("plants", "season", "nature", "observation"),
        difficulty=2,
        v_subject="One close photograph of a seasonal growth stage: a bud, an open flower, a fruit or a seed head, or bare twigs.",
        v_must=(
            "a plant is the main subject",
            "the growth stage is visible and identifiable",
            "the setting is outdoors",
        ),
        v_accept=(
            "a bud opening",
            "a single flower with its petals visible",
            "a seed head or cone still attached",
            "bare winter twigs against the sky",
        ),
        v_reject=(
            "houseplants",
            "cut flowers in a vase",
            "a tree with no seasonal detail identifiable",
        ),
        v_ambiguous=(
            "a plant is present but the growth stage cannot be read",
            "the image is too blurred to identify the plant",
        ),
        variant_key="season",
        weather_fit=(Weather.CLEAR, Weather.COLD, Weather.SNOW, Weather.HEAT),
        keywords=("season", "plants", "nature", "botany", "spring", "autumn"),
    ),
    # -- resting and party variants ---------------------------------------
    Template(
        key="best-seat",
        category="observation",
        title="Best Seat in Sight",
        hook="Five minutes sitting still, and the place finally explains itself.",
        premise="You see a place differently from a seat than from a walking pace. {interest_lead}",
        steps=(
            "Find a public bench or low wall on the {place} that faces something worth looking at.",
            "Sit down for a full five minutes. No phone in your hands.",
            "Note three things that change while you are sitting still.",
            "Notice who else sits here and what they are looking at.",
            "Take one photograph of the view from the seat, framed as they would see it.",
        ),
        details=(
            "Public seating is for sitting on; you do not need to buy anything.",
            "Keep a jacket and water with you if it is warm out.",
        ),
        minutes=(4, 6, 5, 4, 4),
        tags=("rest", "observation", "noticing", "photo"),
        difficulty=1,
        v_subject="One photograph of the view from a seated position, showing what a person sitting on public seating would see.",
        v_must=(
            "the view is photographed from a low, seated height",
            "the scene contains something worth looking at",
            "the setting is outdoors",
        ),
        v_accept=(
            "a view across a square or green",
            "a view down a path or street",
            "a view of a fountain or planting bed",
        ),
        v_reject=(
            "a standing-height overview shot",
            "a selfie",
            "an indoor scene",
        ),
        v_ambiguous=(
            "the view is visible but the height cannot be confirmed",
            "the frame is mostly sky or mostly wall",
        ),
        variant_key="seat",
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.RAIN, Weather.HEAT),
        keywords=("rest", "relax", "observation", "noticing", "mindfulness"),
    ),
    Template(
        key="colour-hunt",
        category="observation",
        title="Colour Hunt Round One Block",
        hook="Kids choose the colours. You keep the tally.",
        premise="A colour hunt works because it turns a walk into a game with a score. {interest_lead}",
        steps=(
            "Agree on one colour before you set off: {party} picks it together.",
            "Walk a single block of the {place} at {pace}.",
            "Every time someone spots the colour, they say it out loud and you count it.",
            "Stop after five sightings and see whether the last one was in the same direction.",
            "Photograph the three best finds laid out in one frame if you can gather them safely -- otherwise photograph the best one where it is.",
        ),
        details=(
            "Stay on the pavement and hold hands near corners.",
            "Nothing is picked up or moved; looking and counting is the whole game.",
            "Keep it to the block; nobody needs to be tired.",
        ),
        minutes=(3, 8, 6, 4, 4),
        tags=("family", "kids", "colour", "game", "observation"),
        difficulty=1,
        v_subject="One photograph showing the colour that was hunted, found outdoors during the walk.",
        v_must=(
            "the hunted colour is present in the image",
            "the image was taken outdoors during the walk",
            "the subject is a real object in the world, not a drawing",
        ),
        v_accept=(
            "a shop sign in the colour",
            "a door in the colour",
            "a vehicle or object in the colour",
        ),
        v_reject=(
            "a printed or drawn picture of the colour",
            "a colour swatch or paint chart",
            "an image where the colour is not present",
        ),
        v_ambiguous=(
            "the colour is present but incidental to the subject",
            "the image is too blurred to judge the colour",
        ),
        variant_key="colour",
        dog_friendly=False,
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.RAIN, Weather.SNOW),
        keywords=("kids", "family", "children", "game", "colour", "walk"),
    ),
    Template(
        key="sniff-route",
        category="observation",
        title="Sniff Route Checkpoints",
        hook="For {party}: five short stops beat one long walk.",
        premise="A dog does not experience a walk as distance, it experiences as smells. This mission is built around that. {interest_lead}",
        steps=(
            "Pick a short loop on the {place}, about the length of three slow streets.",
            "Stop at four spots the dog wants to stop at, rather than steering past them.",
            "At each stop, note what the dog is focused on: ground, plants, a wall, another animal at distance.",
            "Carry water and take a break halfway, even if the dog does not ask.",
            "Photograph one stop where the dog is engaged but not about to eat anything.",
        ),
        details=(
            "Keep the lead short near anything you cannot see, and let the dog off the lead only where it is permitted.",
            "Stop the dog from eating anything; note the spot instead.",
            "Give wildlife a wide berth and never let the dog chase anything.",
        ),
        minutes=(4, 14, 6, 4, 5),
        tags=("dog", "walk", "observation", "smell"),
        difficulty=1,
        v_subject="One photograph of a dog engaged with a scent on an outdoor walk, standing or sniffing.",
        v_must=(
            "a dog is visible in the frame",
            "the dog is outdoors on a walk",
            "the dog is standing or sniffing, not running at something",
        ),
        v_accept=(
            "the dog's nose down at the ground",
            "the dog standing and looking, alert",
            "the dog beside a wall or hedge line",
        ),
        v_reject=(
            "a dog running off-lead in an open field",
            "a dog eating something",
            "a photo of a person with no dog visible",
        ),
        v_ambiguous=(
            "a dog is present but its focus is not readable",
            "the setting cannot be confirmed as outdoors",
        ),
        variant_key="sniff",
        dog_friendly=False,
        weather_fit=(Weather.CLEAR, Weather.CLOUDY, Weather.COLD),
        keywords=("dog", "dogs", "walk", "pet", "smell"),
    ),
    Template(
        key="atrium-light",
        category="indoor",
        title="Atrium Light Study",
        hook="A rainy-day mission that still needs you to leave the house.",
        premise="Indoor light behaves predictably enough to study: where it comes from, what it hits, what it reflects off. {interest_lead}",
        steps=(
            "Find a tall public indoor space near you: an atrium, a station concourse, a large lobby.",
            "Stand in one place and note where the light is coming from: above you, through the side glazing, or from shop fronts.",
            "Photograph the space wide enough to show the structure overhead.",
            "Find one thing the light does to the space that you would not expect -- a shadow shape, a bright patch, a reflection.",
            "Take a closer photograph of that effect.",
        ),
        details=(
            "Public indoor space only; do not enter shops or restricted areas.",
            "Keep out of walkways and away from doorways.",
            "Stay warm and dry -- that is the point of this mission.",
        ),
        minutes=(4, 5, 5, 6, 4),
        tags=("indoor", "photography", "light", "rain", "city"),
        difficulty=2,
        v_subject="One photograph of a tall public indoor space showing where its light comes from and at least one visible lighting effect.",
        v_must=(
            "a tall indoor public space is visible",
            "the ceiling, glazing or an overhead light fitting is in frame",
            "there is a visible effect: a shadow, a bright patch or a reflection",
        ),
        v_accept=(
            "a ceiling structure or skylight in view",
            "a skylight with light falling through it",
            "a large window wall lighting an interior",
        ),
        v_reject=(
            "a private home interior",
            "a shop floor",
            "an outdoor scene",
        ),
        v_ambiguous=(
            "an interior is visible but the light source cannot be identified",
            "the space is too dark to read",
        ),
        variant_key="atrium",
        indoors_ok=True,
        weather_fit=(Weather.RAIN, Weather.SNOW, Weather.WINDY),
        keywords=("indoor", "rain", "photography", "city", "light"),
    ),
)


def _assert_library_sane() -> None:
    """Fail loudly at import if a hand-edited template is internally inconsistent.

    Cheap insurance for a hand-maintained library: every parallel tuple must
    line up, or a bad edit would silently produce a mission with the wrong
    timing rather than an error.
    """
    seen: set[str] = set()
    for tpl in TEMPLATES:
        if tpl.key in seen:
            raise RuntimeError(f"duplicate template key: {tpl.key}")
        seen.add(tpl.key)
        n = len(tpl.steps)
        if not (n == len(tpl.minutes)):
            raise RuntimeError(f"template {tpl.key}: steps/minutes length mismatch")
        if not (1 <= len(tpl.details) <= n):
            raise RuntimeError(f"template {tpl.key}: details must be 1..len(steps) entries")
        if len(tpl.v_must) > 6 or len(tpl.v_reject) > 6:
            raise RuntimeError(f"template {tpl.key}: verification list too long")
        if len(tpl.v_accept) > 4 or len(tpl.v_ambiguous) > 4:
            raise RuntimeError(f"template {tpl.key}: verification list too long")
        if tpl.variant_key not in _VARIANTS:
            raise RuntimeError(f"template {tpl.key}: unknown variant key {tpl.variant_key}")
        if not 1 <= tpl.difficulty <= 3:
            raise RuntimeError(f"template {tpl.key}: difficulty out of range")


_assert_library_sane()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

#: Hard generation ceiling. Verified against the gate's blocking rule set
#: (swimming/open water, height, road play, trespass, storms) so the model is
#: given the same boundaries the deterministic path is held to.
_FORBIDDEN = (
    "NEVER propose: swimming or any open water; climbing, ledges, rooftops or "
    "height exposure; roads, crossings or traffic; trespassing, derelict or "
    "restricted land; or storms and severe weather."
)

_JSON_SKELETON = (
    '{"title":"...","hook":"...","premise":"...","steps":[{"minutes":8,'
    '"instruction":"...","detail":""}],"tags":["..."],"safety_notes":["..."],'
    '"verification":{"subject":"...","must_include":["..."],"accept_hints":["..."],'
    '"reject_if":["..."],"ambiguous_cases":["..."]}}'
)


def compose_prompt(req: QuestRequest) -> str:
    """Build the generation prompt for the open-weight model.

    Kept deliberately short (well under 450 tokens of instruction) because the
    target is a small instruction-tuned model: every token spent on policy is a
    token not spent on staying on format. The JSON skeleton is inlined rather
    than described, since small models copy shapes far more reliably than they
    follow prose descriptions of shapes.
    """
    ctx = _describe(req)
    interests = ", ".join(req.interests) if req.interests else "none given"
    avoid = ", ".join(req.avoid) if req.avoid else "none given"
    note = req.context_note.strip()
    lines = [
        "You write short outdoor side quests for a screen-minimal exploration game.",
        "",
        "OUTPUT RULES",
        "- Reply with ONE JSON object. No prose, no markdown, no code fences, no comments.",
        "- Use exactly these keys and no others:",
        _JSON_SKELETON,
        "- English only. Keep title<=60 chars, hook<=140, premise<=300, each step<=140.",
        "- Write 3-5 steps. sum of steps[].minutes MUST be <= time_budget_minutes.",
        "- The mission must be doable on foot in one outing, ending at a photo point.",
        f"- {_FORBIDDEN}",
        "- If weather is bad, choose an indoor or sheltered mission instead.",
        "",
        "PLAYER",
        f"time_budget_minutes: {req.time_budget_minutes}",
        f"interests: {interests}",
        f"party: {req.party.value} | mobility: {req.mobility.value} | energy: {req.energy.value}",
        f"location: {req.location_type.value} | weather: {req.weather.value}",
        f"stay within {ctx.radius_phrase} | avoid: {avoid}",
        f"context: {note}" if note else "context: none given",
        "",
        "VERIFICATION (this is what the player's photo will be judged against)",
        "- subject: the single thing the photo must show.",
        "- must_include: 2-4 concrete visual facts. accept_hints: 2-4 likely subjects.",
        "- reject_if: 2-3 failure cases. ambiguous_cases: 1-2 honest I-cannot-tell cases.",
        "",
        "JSON only.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic generation
# ---------------------------------------------------------------------------

_XP_PER_MINUTE = 1.0
_XP_DIFFICULTY = {1: 1.0, 2: 1.25, 3: 1.55}


def _interest_score(tpl: Template, req: QuestRequest) -> int:
    """How well a template matches what the player said they care about."""
    if not req.interests:
        return 0
    keys = set(tpl.keywords) | {tpl.category, tpl.key}
    hits = 0
    for tag in req.interests:
        if not tag:
            continue
        if tag in keys:
            hits += 2
        elif any(tag in key or key in tag for key in keys if len(key) > 3):
            hits += 1
    return hits


def _avoided(tpl: Template, req: QuestRequest) -> bool:
    """True when the template talks about something the player asked to avoid."""
    if not req.avoid:
        return False
    blob = f"{tpl.title} {tpl.premise} {' '.join(tpl.steps)}".lower()
    return any(term and term.lower() in blob for term in req.avoid)


def _rank_templates(req: QuestRequest, rng: random.Random) -> list[Template]:
    """Order templates best-fit-first, deterministically for a given seed.

    Ordering is a weighted shuffle, not a plain sort by relevance. Sorting
    outright would let a single perfect interest match ("birds" -> bird-count)
    monopolise every request and every seed, which would make the fallback feel
    like it only ever had one idea. Instead each template is drawn from a
    weight that favours relevant missions without pinning the winner, so the
    same interests still steer the result while variation survives.

    Hard filters (`mobility_safe`, `low_energy`, `indoors_ok`) are applied
    first and are not negotiable: those express player constraints the gate
    treats as blocking. Weather and party filters are soft and only reorder,
    so a rain request degrades to a wet-weather mission without ever emptying
    the pool.
    """
    pool = [t for t in TEMPLATES if not _avoided(t, req)]

    hard: list[Template] = []
    for tpl in pool:
        if req.mobility is not Mobility.UNRESTRICTED and not tpl.mobility_safe:
            continue
        if req.energy is EnergyLevel.LOW and not tpl.low_energy:
            continue
        if req.location_type is LocationType.INDOOR_MALL and not tpl.indoors_ok:
            continue
        hard.append(tpl)

    if not hard:  # keep a mission available for any combination
        hard = pool or list(TEMPLATES)

    def soft_penalty(tpl: Template) -> int:
        """0 = ideal fit; higher = usable but second choice."""
        penalty = 0
        if req.party is PartyType.WITH_KIDS and not tpl.kid_safe:
            penalty += 1
        if req.party is PartyType.WITH_DOG and not tpl.dog_friendly:
            penalty += 1
        if req.weather in (Weather.RAIN, Weather.SNOW) and tpl.weather_fit and (
            req.weather not in tpl.weather_fit
        ):
            penalty += 1
        return penalty

    # Weighted selection: higher weight is drawn earlier. `rng.random()` is
    # consumed in template order so the sequence stays stable for a seed.
    ranked: list[Template] = []
    remaining = sorted(hard, key=lambda t: t.key)
    while remaining:
        weights = [
            (4.0 + 2.0 * _interest_score(t, req)) / (1.0 + soft_penalty(t))
            for t in remaining
        ]
        pick = rng.choices(range(len(remaining)), weights=weights, k=1)[0]
        ranked.append(remaining.pop(pick))
    return ranked


def _fit_steps(
    tpl: Template, budget: int, rng: random.Random
) -> tuple[list[QuestStep], int]:
    """Trim and re-time a template so it always fits `budget` minutes.

    Three stages, cheapest first:

    1. Drop trailing steps. Missions are written so the last step is the photo
       or the finale, so trimming from the end keeps the remaining narrative
       coherent -- dropping a middle step would leave "repeat that at the
       second spot" pointing at a step that no longer exists.
    2. Scale the survivors proportionally, then hand the rounding remainder out
       one minute at a time in a seeded order. That is where per-seed variety
       comes from: the same template with the same budget still differs between
       seeds in how its minutes land.
    3. Clamp. A final hard pass guarantees the sum fits, whatever rounding did.

    Args:
        tpl: The template being rendered.
        budget: `req.time_budget_minutes`; the returned steps sum to <= this.
        rng: Seeded RNG, consumed only for the remainder shuffle.

    Returns:
        `(steps, total_minutes)` where `total_minutes <= budget` always.
    """
    rows = [
        (ins, tpl.details[i] if i < len(tpl.details) else "", minutes)
        for i, (ins, minutes) in enumerate(zip(tpl.steps, tpl.minutes))
    ]

    while sum(row[2] for row in rows) > budget and len(rows) > 1:
        rows.pop()

    total = sum(row[2] for row in rows)
    if total > budget:
        factor = budget / total
        scaled = [max(1, int(round(row[2] * factor))) for row in rows]
        total = sum(scaled)
        # Hand out the rounding surplus one minute at a time, seeded.
        order = list(range(len(scaled)))
        rng.shuffle(order)
        while total > budget:
            progressed = False
            for i in order:
                if total <= budget:
                    break
                if scaled[i] > 1:
                    scaled[i] -= 1
                    total -= 1
                    progressed = True
            if not progressed:
                break
        rows = [(ins, det, scaled[i]) for i, (ins, det, _) in enumerate(rows)]

    # Final clamp: shed whole steps, then shave minutes, until the sum fits.
    while len(rows) > 1 and sum(row[2] for row in rows) > budget:
        rows.pop()
    i = 0
    while sum(row[2] for row in rows) > budget and i < len(rows):
        if rows[i][2] > 1:
            ins, det, minutes = rows[i]
            rows[i] = (ins, det, minutes - 1)
        i += 1
    while len(rows) > 1 and sum(row[2] for row in rows) > budget:
        rows.pop()
    if rows and sum(row[2] for row in rows) > budget:
        ins, det, _ = rows[0]
        rows[0] = (ins, det, max(1, budget))

    steps = [
        QuestStep(order=i, instruction=ins, approx_minutes=minutes, detail=det)
        for i, (ins, det, minutes) in enumerate(rows, start=1)
    ]
    return steps, sum(row[2] for row in rows)


def _xp_for(minutes: int, difficulty: int, req: QuestRequest) -> int:
    """Reward scales with time invested and with how demanding the mission is."""
    if req.energy is EnergyLevel.LOW:
        xp = minutes * _XP_PER_MINUTE * 0.8
    else:
        xp = minutes * _XP_PER_MINUTE * _XP_DIFFICULTY.get(difficulty, 1.0)
    return max(5, min(500, int(round(xp))))


def _build(req: QuestRequest, ctx: _Ctx, tpl: Template, rng: random.Random) -> Quest:
    """Render one template into a fully-formed `Quest`.

    Every player-visible string passes through `_render`, including step
    instructions, step details and the verification spec -- a template is
    written with `{place}` and friends inline, so any field skipped here would
    ship a literal `{placeholder}` to the player.
    """
    values = ctx.as_mapping()
    steps, minutes = _fit_steps(tpl, req.time_budget_minutes, rng)
    for step in steps:
        step.instruction = _render(step.instruction, values)
        step.detail = _render(step.detail, values)

    variant = rng.choice(_VARIANTS[tpl.variant_key])
    premise = _render(tpl.premise, values)
    if len(premise) + len(variant) + 2 <= 500:
        premise = f"{premise} {variant}"

    notes: list[str] = []
    for note in tpl.notes:
        rendered = _render(note, values)
        if rendered not in notes:
            notes.append(rendered)
    if req.avoid:
        # Do NOT quote the player's avoid terms back into a note. `mission_text`
        # includes `safety_notes`, so "You asked to avoid busy roads" put the
        # literal string "road" in front of `haz-traffic` and made *every*
        # candidate look blocked. Any avoid term that is also a hazard keyword
        # ("roads", "river", "climb", "storm", "drains") exhausted the entire
        # template library and silently degraded every such player to the
        # last-resort mission -- while `_attach_report` still reported
        # `passed=True`, so the app claimed the mission was safe as the gate
        # rejected it.
        #
        # Naming the terms back adds nothing: the player just typed them. The
        # disclosure stays, and stays truthful, because `_rank_templates` really
        # does drop every template mentioning them (`_avoided`). It simply
        # refers to the constraint instead of restating its vocabulary, because
        # a statement *about* a hazard is not a hazard.
        note = "This route stays clear of everything you asked to avoid."
        if note not in notes:
            notes.insert(0, note)
    if req.party is PartyType.WITH_KIDS and len(notes) < 6:
        notes.append("Stay where you can see the kids the whole time.")
    notes = notes[:6]

    return Quest(
        title=_render(tpl.title, values),
        hook=_render(tpl.hook, values)[:200],
        premise=premise[:500],
        steps=steps,
        estimated_minutes=max(1, min(240, minutes)),
        xp_reward=_xp_for(minutes, tpl.difficulty, req),
        tags=list(tpl.tags)[:8],
        safety_notes=notes,
        verification=VerificationSpec(
            subject=_render(tpl.v_subject, values),
            must_include=[_render(s, values) for s in tpl.v_must][:6],
            accept_hints=[_render(s, values) for s in tpl.v_accept][:4],
            reject_if=[_render(s, values) for s in tpl.v_reject][:6],
            ambiguous_cases=[_render(s, values) for s in tpl.v_ambiguous][:4],
        ),
        source=GeneratorSource.FALLBACK,
        model_id=None,
        raw_model_output=None,
        generation_ms=0,
    )


def _leftover_placeholders(quest: Quest) -> str:
    """Return the first unrendered `{placeholder}` in a mission, or "".

    Covers the verification spec as well as the prose, since `mission_text`
    only joins the prose. A leftover token means the template was authored with
    a placeholder the renderer does not know, which would otherwise ship to the
    player verbatim.
    """
    spec = quest.verification
    parts: list[str] = [mission_text(quest), spec.subject]
    parts += list(spec.must_include) + list(spec.accept_hints)
    parts += list(spec.reject_if) + list(spec.ambiguous_cases)
    match = _LEFTOVER_RE.search("\n".join(parts))
    return match.group(0) if match else ""


def _blocking(quest: Quest, req: QuestRequest) -> list[str]:
    """Run the real gate over a candidate mission; return blocking rule ids.

    The fallback is the safety net, so it must not be the thing that produces a
    mission the project's own policy rejects. Checking here -- rather than
    trusting hand-written prose to stay clean -- is what makes that guarantee
    hold as the library is edited over time.
    """
    text_screen = scan_quest(quest)
    req_screen = check_request_feasibility(req, quest)
    return [f.rule for f in req_screen.findings + text_screen.findings if f.severity == "blocking"]


#: A `{placeholder}` surviving into player-facing text. Templates are authored
#: with placeholders inline, so any that reaches output is a rendering bug.
_LEFTOVER_RE = re.compile(r"\{[a-z_]+\}")


def _attach_report(quest: Quest, req: QuestRequest) -> Quest:
    """Record the gate's findings on an accepted mission.

    ``passed`` is *derived* from the findings rather than hard-coded. It used to
    be ``True`` on every call, on the reasoning that only accepted missions
    reach here -- but the last-resort mission at the bottom of ``fallback_quest``
    is built and reported without being re-checked, so a mission the gate
    rejected could still ship carrying ``passed=True``. That is the worst
    possible failure for this field: the app asserting a mission is safe at the
    exact moment its own policy says it is not. Computing it from the findings
    means the report cannot disagree with the rules that produced it.
    """
    text_screen = scan_quest(quest)
    req_screen = check_request_feasibility(req, quest)
    all_findings = req_screen.findings + text_screen.findings
    blocking = [f for f in all_findings if f.severity == "blocking"]
    findings: list[SafetyFinding] = [f for f in all_findings if f.severity != "blocking"]
    triggered: list[str] = []
    for finding in all_findings:
        if finding.rule not in triggered:
            triggered.append(finding.rule)
    quest.safety = SafetyReport(
        passed=not blocking,
        findings=(findings + blocking)[:6],
        triggered=triggered,
    )
    for finding in findings:
        if len(quest.safety_notes) >= 6:
            break
        if finding.message not in quest.safety_notes:
            quest.safety_notes.append(finding.message)
    return quest


#: Last-resort mission. Written to be hazard-free under every rule in the gate,
#: so the builder always has somewhere valid to land.
_LAST_RESORT = Template(
    key="last-resort",
    category="observation",
    title="One Block, Five Things",
    hook="A slow loop around one familiar stretch of pavement.",
    premise="You already know this way. This mission is about noticing it properly instead of walking through it.",
    steps=(
        "Stand at the nearest corner and pick one ordinary thing you have passed more than ten times.",
        "Look at it for sixty seconds without touching it and say a description out loud.",
        "Walk a few doors along and find a second thing that shares a colour with the first.",
        "Photograph both together in one frame from a low angle, then stop.",
    ),
    details=(
        "The phone stays in your hand until the last step.",
        "Everything you look at stays where it is.",
    ),
    minutes=(3, 4, 5, 4),
    tags=("noticing", "observation", "walk"),
    difficulty=1,
    v_subject="One photograph showing two everyday street objects side by side outdoors.",
    v_must=(
        "two separate everyday objects are visible",
        "the setting is outdoors",
        "both objects are in the same frame",
    ),
    v_accept=(
        "a bin and a postbox",
        "a window and a door",
        "two signs in the same colour",
    ),
    v_reject=(
        "a selfie or a photo of a screen",
        "a blurred movement shot",
        "an indoor room",
    ),
    v_ambiguous=(
        "the two objects overlap so only one is visible",
        "the frame is almost entirely sky or pavement",
    ),
    variant_key="museum",
    notes=("Stay on public pavements and keep other people in view.",),
)


def fallback_quest(req: QuestRequest, seed: int | None = None) -> Quest:
    """Build a safe, feasible, deterministic mission from the curated library.

    `seed` (or `req.seed`) fully determines the result: same request, same seed,
    same mission. Every candidate template is checked against the real safety
    gate before it is returned, so the fallback can never hand back a mission
    the project's own policy would reject.

    Args:
        req: The player's constraints -- time budget, interests, party,
            mobility, energy, location and weather.
        seed: Overrides `req.seed` when given.

    Returns:
        A `Quest` with `source=FALLBACK`, steps summing to at most
        `req.time_budget_minutes`, and a mission-specific verification spec.
    """
    started = time.perf_counter()
    rng = random.Random(seed if seed is not None else req.seed)
    ctx = _describe(req)

    for tpl in _rank_templates(req, rng):
        candidate = _build(req, ctx, tpl, rng)
        if _blocking(candidate, req):
            continue  # library drifted into the policy: try the next template
        candidate = _attach_report(candidate, req)
        if _leftover_placeholders(candidate):
            continue  # unrendered placeholder: treat as an unusable template
        candidate.generation_ms = int((time.perf_counter() - started) * 1000)
        return candidate

    # Nothing in the library cleared the gate (only possible if the library has
    # been edited badly). Rather than return nothing, use the one template that
    # is written to be hazard-free under every rule.
    rng2 = random.Random(seed if seed is not None else req.seed)
    quest = _attach_report(_build(req, ctx, _LAST_RESORT, rng2), req)
    quest.generation_ms = int((time.perf_counter() - started) * 1000)
    return quest


# ---------------------------------------------------------------------------
# Verification (never claims a verification)
# ---------------------------------------------------------------------------

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.DOTALL)


def _describe_photo(data_url: str | None) -> tuple[bool, str]:
    """Report whether a usable photo was supplied, without inspecting content.

    Returns `(present, detail)` where `detail` is a short, honest description of
    what was received. Never decodes or interprets the image -- that is the
    model's job, and no model ran here. Any malformed input yields a
    description rather than an exception, because the verdict path must not be
    the thing that crashes on a bad upload.
    """
    if not data_url or not data_url.strip():
        return False, ""
    text = data_url.strip()
    if not text.lower().startswith("data:"):
        return True, "something that is not a data: URL"
    match = _DATA_URL_RE.match(text)
    if match is None:
        return True, "a malformed data: URL"
    mime = match.group("mime") or "an unknown file type"
    encoded = match.group("data") or ""
    size = len(encoded)
    if size == 0:
        return True, f"an empty data: URL declaring {mime}"
    if match.group("b64"):
        # base64 length is 4/3 of the byte count; arithmetic only, no decoding.
        approx_bytes = (size // 4) * 3
    else:
        approx_bytes = size
    size_text = f"about {approx_bytes / 1024:.1f} KB" if approx_bytes >= 1024 else "under 1 KB"
    return True, f"{mime}, {size_text}, supplied but never opened"


def _first_words(text: str, limit: int = 3) -> list[str]:
    """First `limit` meaningful words from the evidence note, for `missing`."""
    cleaned = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    words = [w for w in cleaned.split() if len(w) > 2]
    return words[:limit]


def fallback_verdict(quest: Quest, evidence: Evidence) -> Verdict:
    """Produce an honest, non-verifying verdict with no model available.

    This path exists so the app degrades gracefully rather than failing, which
    means it must be conservative. It does not look at the photo, cannot judge
    whether the mission was completed, and therefore:

    * sets `is_ai_verified=False` unconditionally;
    * returns level HEURISTIC when no photo was supplied, and UNCERTAIN when a
      photo was supplied but no vision model ran to interpret it;
    * awards 0 XP, always.

    Args:
        quest: The mission the evidence is claimed against.
        evidence: What the player submitted. `photo_data_url` is noted, never
            parsed or analysed.

    Returns:
        A `Verdict` tagged `source=FALLBACK` with `model_id=None`.
    """
    has_photo, photo_detail = _describe_photo(evidence.photo_data_url)

    if has_photo:
        level = ConfidenceLevel.UNCERTAIN
        summary = (
            f"No vision model was available, so the evidence ({photo_detail}) could "
            f"not be checked against \"{quest.verification.subject}\". Not a pass and "
            f"not a fail: this is waiting on a model run."
        )
        missing = list(quest.verification.must_include[:3])
        needs_better_photo = True
    else:
        level = ConfidenceLevel.HEURISTIC
        summary = (
            "Heuristic result only: no photo was submitted and no model was available, "
            "so this mission is unverified. Nothing has been confirmed either way."
        )
        missing = ["a photo showing " + quest.verification.subject.lower()]
        needs_better_photo = True

    note_words = _first_words(evidence.note)
    if note_words:
        summary += f" Note text read: {', '.join(note_words)}."

    return Verdict(
        quest_id=quest.id,
        level=level,
        is_ai_verified=False,
        # Deliberately low: this is a "we looked at nothing" number.
        confidence=0.15 if has_photo else 0.0,
        summary=summary[:300],
        matched=[],
        missing=missing[:6],
        needs_better_photo=needs_better_photo,
        source=GeneratorSource.FALLBACK,
        model_id=None,
        xp_awarded=0,
        latency_ms=0,
        raw_model_output=None,
    )