"""Frozen data contract for SideQuest AI.

Every module in the project imports types from here. This file is the
integration seam: it is written first and changed only deliberately.

Design note
-----------
The three verification levels (`ConfidenceLevel`) exist to satisfy a hard
project rule: never label a mission "AI-verified" when the only evidence is a
button press or a hardcoded rule. `Verdict.is_ai_verified` is derived from the
model actually running *and* the evidence being model-confirmed; it is never
set by a heuristic path.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class PartyType(str, Enum):
    SOLO = "solo"
    WITH_KIDS = "with_kids"
    WITH_DOG = "with_dog"
    GROUP = "group"


class Mobility(str, Enum):
    UNRESTRICTED = "unrestricted"
    AVOID_STAIRS = "avoid_stairs"
    STEP_FREE = "step_free"  # wheelchair / stroller friendly


class EnergyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LocationType(str, Enum):
    CITY = "city"
    NEIGHBORHOOD = "neighborhood"
    PARK = "park"
    TRAIL = "trail"
    WATERFRONT = "waterfront"
    INDOOR_MALL = "indoor_mall"  # rain contingency
    UNKNOWN = "unknown"


class Weather(str, Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    HEAT = "heat"
    COLD = "cold"
    SNOW = "snow"
    WINDY = "windy"
    UNKNOWN = "unknown"


class QuestStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ConfidenceLevel(str, Enum):
    """How much the system should trust the verification.

    These are deliberately ordered. The UI maps them to distinct visual
    treatments so a user can never confuse a model prediction with a
    confirmed fact.
    """

    VERIFIED = "verified"          # model ran, evidence matched, high confidence
    LIKELY = "likely"              # model ran, evidence plausible, moderate confidence
    UNCERTAIN = "uncertain"        # model ran but evidence was ambiguous
    REJECTED = "rejected"          # model ran and evidence did not match
    HEURISTIC = "heuristic"        # no model available; rules-only result
    UNAVAILABLE = "unavailable"    # no model and no fallback possible

    @property
    def counts_as_verified(self) -> bool:
        return self in (ConfidenceLevel.VERIFIED, ConfidenceLevel.LIKELY)


class GeneratorSource(str, Enum):
    """Which engine produced a given artifact. Recorded on every generation."""

    MODEL = "model"
    FALLBACK = "fallback"
    CACHED = "cached"


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


class QuestRequest(BaseModel):
    """Everything the generator needs to produce a *safe, feasible* mission."""

    time_budget_minutes: int = Field(ge=5, le=240)
    interests: list[str] = Field(default_factory=list, max_length=8)
    party: PartyType = PartyType.SOLO
    mobility: Mobility = Mobility.UNRESTRICTED
    energy: EnergyLevel = EnergyLevel.MEDIUM
    location_type: LocationType = LocationType.UNKNOWN
    weather: Weather = Weather.UNKNOWN
    max_radius_m: int = Field(default=800, ge=50, le=5000)
    avoid: list[str] = Field(default_factory=list, max_length=8)
    #: Free-text, e.g. "near a school playground, lots of pavement".
    context_note: str = Field(default="", max_length=400)
    #: Stable per-player seed keeps regeneration varied but reproducible in tests.
    seed: int | None = None

    @field_validator("interests", "avoid")
    @classmethod
    def _clean_tags(cls, v: list[str]) -> list[str]:
        return [t.strip().lower() for t in v if t and t.strip()][:8]


class Evidence(BaseModel):
    """What the player brings back. No precise location, ever."""

    photo_data_url: str | None = Field(
        default=None,
        description="data:image/... URL. Transmitted only for model verification.",
    )
    note: str = Field(default="", max_length=600)
    #: Coarse, human-entered label ("the park on 5th") — never GPS coordinates.
    location_label: str = Field(default="", max_length=120)
    elapsed_minutes: int = Field(default=0, ge=0, le=1440)


# --------------------------------------------------------------------------
# Core domain objects
# --------------------------------------------------------------------------


class VerificationSpec(BaseModel):
    """What evidence the model should look for, generated with the mission.

    This is the part that makes model verification non-trivial: the criteria
    are *mission-specific* and authored by the same model pass that wrote the
    mission, so verification is not a generic image classifier.
    """

    subject: str = Field(description="The single thing the photo must show.")
    must_include: list[str] = Field(default_factory=list, max_length=6)
    accept_hints: list[str] = Field(default_factory=list, max_length=4)
    reject_if: list[str] = Field(default_factory=list, max_length=6)
    #: Cases where the model should legitimately say "I cannot tell".
    ambiguous_cases: list[str] = Field(default_factory=list, max_length=4)


class QuestStep(BaseModel):
    order: int = Field(ge=1)
    instruction: str = Field(min_length=1, max_length=300)
    approx_minutes: int = Field(ge=1, le=240)
    #: Optional screen-free prompt shown once, before the player leaves.
    detail: str = Field(default="", max_length=300)


class SafetyFinding(BaseModel):
    rule: str
    severity: Literal["info", "warning", "blocking"]
    message: str


class SafetyReport(BaseModel):
    passed: bool
    findings: list[SafetyFinding] = Field(default_factory=list)
    #: Rule names that fired. Empty means the mission cleared the policy.
    triggered: list[str] = Field(default_factory=list)


class Quest(BaseModel):
    id: str = Field(default_factory=_new_id)
    title: str = Field(min_length=1, max_length=80)
    hook: str = Field(default="", max_length=200)
    premise: str = Field(default="", max_length=500)
    steps: list[QuestStep] = Field(min_length=1, max_length=8)
    estimated_minutes: int = Field(ge=1, le=240)
    xp_reward: int = Field(ge=0, le=500)
    tags: list[str] = Field(default_factory=list, max_length=8)
    safety_notes: list[str] = Field(default_factory=list, max_length=6)
    verification: VerificationSpec
    safety: SafetyReport = Field(default_factory=lambda: SafetyReport(passed=True))
    source: GeneratorSource = GeneratorSource.FALLBACK
    model_id: str | None = None
    #: Full untruncated model output, kept for the eval suite and debugging.
    raw_model_output: str | None = None
    generation_ms: int = 0
    created_at: float = Field(default_factory=time.time)

    @field_validator("steps")
    @classmethod
    def _number_steps(cls, v: list[QuestStep]) -> list[QuestStep]:
        for i, s in enumerate(v, start=1):
            s.order = i
        return v


class Verdict(BaseModel):
    """The outcome of evaluating one piece of evidence.

    `is_ai_verified` is the field the UI must branch on for the "verified"
    label. It is True only when a model actually ran and produced
    VERIFIED/LIKELY. A heuristic result can never set it.
    """

    quest_id: str
    level: ConfidenceLevel
    is_ai_verified: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = Field(default="", max_length=300)
    matched: list[str] = Field(default_factory=list, max_length=6)
    missing: list[str] = Field(default_factory=list, max_length=6)
    #: Populated when the model refused to judge; drives the retry affordance.
    needs_better_photo: bool = False
    source: GeneratorSource = GeneratorSource.FALLBACK
    model_id: str | None = None
    xp_awarded: int = 0
    latency_ms: int = 0
    raw_model_output: str | None = None


class XpEvent(BaseModel):
    at: float = Field(default_factory=time.time)
    amount: int
    reason: str
    quest_id: str | None = None


class PlayerState(BaseModel):
    player_id: str = Field(default_factory=_new_id)
    xp: int = 0
    level: int = 1
    quests_completed: int = 0
    #: Minutes actually spent outside, as reported by the player. This is the
    #: headline number for the "Touch Grass" theme.
    minutes_outdoors: int = 0
    verified_count: int = 0
    uncertain_count: int = 0
    rejected_count: int = 0
    streak_days: int = 0
    quests: list[Quest] = Field(default_factory=list)
    history: list[XpEvent] = Field(default_factory=list, max_length=200)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class SessionSummary(BaseModel):
    """What the player sees when they put the phone away and come back."""

    player_id: str
    xp: int
    level: int
    xp_to_next_level: int
    minutes_outdoors: int
    verified_count: int
    uncertain_count: int
    rejected_count: int
    screen_time_saved_seconds: int


# --------------------------------------------------------------------------
# Engine / diagnostics
# --------------------------------------------------------------------------


class EngineStatus(BaseModel):
    """Reported at /api/health so the demo never overstates what it can do."""

    model_id: str | None = None
    loaded: bool = False
    device: str = "none"
    dtype: str = "none"
    supports_vision: bool = False
    load_error: str | None = None
    last_latency_ms: int | None = None
    torch_version: str | None = None
    transformers_version: str | None = None


class GenerateResult(BaseModel):
    text: str
    latency_ms: int
    model_id: str | None
    source: GeneratorSource
    #: Set when generation failed or timed out and the fallback was used.
    failure_reason: str | None = None
    raw_prompt_tokens: int | None = None
    completion_tokens: int | None = None