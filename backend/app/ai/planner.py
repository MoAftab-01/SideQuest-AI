"""Mission planning: turn a player's constraints into a safe, feasible quest.

This is the module that makes open-weight inference central to the product
rather than bolted on. The model does the thing templates cannot do well: read a
specific combination of time, interests, party, mobility and weather and write
a mission that fits *that* person, including the verification criteria for the
mission.

The reliability ladder, in order:
  1. model generates -> parses -> validates -> passes safety gate  => use it
  2. model output unusable (bad JSON, empty, or blocked by safety)      => fallback
  3. no model loaded at all                                            => fallback

Every failure at step 1 is counted in `PlanOutcome`, so the eval harness can
publish a real fallback-trigger rate instead of asserting the model "works".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.ai import safety
from app.ai.engine import GENERATION_TIMEOUT_S, InferenceEngine
from app.ai.fallback import compose_prompt, fallback_quest
from app.ai.parsing import coerce_int, coerce_list, coerce_str, extract_json
from app.models.schemas import (
    GeneratorSource,
    Quest,
    QuestRequest,
    QuestStep,
    VerificationSpec,
)

log = logging.getLogger(__name__)

#: Retry once with a stricter nudge before giving up. Small models often need a
#: second, blunter instruction -- one retry is worth it, a loop is not.
_RETRY_SUFFIX = (
    "\n\nIMPORTANT: reply with ONE JSON object and nothing else. "
    "No explanation before or after. No markdown fences."
)


@dataclass
class PlanOutcome:
    """Everything the eval suite needs to know about one planning call."""

    source: GeneratorSource
    latency_ms: int
    model_id: str | None = None
    parse_ok: bool = False
    parse_reason: str = ""
    safety_passed: bool = True
    #: Why we fell back, if we did. Empty string means we used the model.
    fallback_reason: str = ""
    truncated_to_budget: bool = False
    warnings: list[str] = field(default_factory=list)


def _build_verification(data: dict) -> VerificationSpec:
    """Build a mission-specific verification spec, never leaving it empty.

    A blank spec would make verification meaningless, so if the model returned
    nothing usable we fall back to a generic-but-honest criterion derived from
    the mission title. Better a weak spec, clearly weak, than a fabricated one.
    """
    subject = coerce_str(data.get("subject") or data.get("verification_subject"), max_len=160)
    must = coerce_list(data.get("must_include") or data.get("must_show"), limit=6)
    accept = coerce_list(data.get("accept_hints") or data.get("hints"), limit=4)
    reject = coerce_list(data.get("reject_if") or data.get("reject"), limit=6)
    amb = coerce_list(data.get("ambiguous_cases") or data.get("ambiguous"), limit=4)

    if not subject:
        subject = "A single photo that clearly shows what you were asked to look for."
    if not must:
        must = [subject]

    return VerificationSpec(
        subject=subject,
        must_include=must,
        accept_hints=accept,
        reject_if=reject,
        ambiguous_cases=amb,
    )


def quest_from_model_data(data: dict, req: QuestRequest) -> Quest:
    """Convert validated JSON into a `Quest`, enforcing the hard constraints.

    This is where a small model's mistakes get contained: every field is
    coerced and clamped rather than trusted, and step minutes are reconciled
    against the player's actual time budget.
    """
    title = coerce_str(data.get("title"), max_len=80)
    if not title:
        raise ValueError("model produced no title")

    raw_steps = data.get("steps") or data.get("itinerary") or []
    if not isinstance(raw_steps, (list, tuple)) or not raw_steps:
        raise ValueError("model produced no steps")

    steps: list[QuestStep] = []
    for i, raw in enumerate(raw_steps[:8], start=1):
        if isinstance(raw, str):
            instruction, minutes, detail = raw, 5, ""
        elif isinstance(raw, dict):
            instruction = coerce_str(raw.get("instruction") or raw.get("text"), max_len=300)
            minutes = coerce_int(raw.get("minutes") or raw.get("approx_minutes"), lo=1, hi=240, default=5)
            detail = coerce_str(raw.get("detail"), max_len=300)
        else:
            continue
        if instruction:
            steps.append(QuestStep(order=i, instruction=instruction, approx_minutes=minutes, detail=detail))

    if not steps:
        raise ValueError("model produced no usable steps")

    estimated = coerce_int(data.get("estimated_minutes"), lo=1, hi=req.time_budget_minutes, default=req.time_budget_minutes)
    xp = coerce_int(data.get("xp_reward"), lo=5, hi=500, default=min(120, 10 + estimated))

    quest = Quest(
        title=title,
        hook=coerce_str(data.get("hook"), max_len=200),
        premise=coerce_str(data.get("premise"), max_len=500),
        steps=steps,
        estimated_minutes=estimated,
        xp_reward=xp,
        tags=coerce_list(data.get("tags") or req.interests, limit=8),
        verification=_build_verification(data),
    )

    # Reconcile step durations with the stated budget. A model that says a
    # 40-minute mission has steps totalling 95 minutes is wrong, and the player
    # is the one who finds out if we pass it through.
    total = sum(s.approx_minutes for s in quest.steps)
    if total > req.time_budget_minutes:
        if total > req.time_budget_minutes * 1.5:
            quest.steps = quest.steps[: max(1, len(quest.steps) // 2)]
        scale = req.time_budget_minutes / max(1, total)
        for s in quest.steps:
            s.approx_minutes = max(1, int(s.approx_minutes * scale))
        total = sum(s.approx_minutes for s in quest.steps)
        quest.estimated_minutes = total

    return quest


def plan_quest(
    req: QuestRequest,
    engine: InferenceEngine,
    *,
    seed: int | None = None,
    timeout_s: float = GENERATION_TIMEOUT_S,
) -> tuple[Quest, PlanOutcome]:
    """Generate a mission. Always returns a usable quest plus an honest outcome."""
    started = time.perf_counter()

    result = engine.generate(compose_prompt(req), timeout_s=timeout_s)

    if result is None:
        quest = fallback_quest(req, seed=seed)
        quest.source = GeneratorSource.FALLBACK
        quest.generation_ms = int((time.perf_counter() - started) * 1000)
        return quest, PlanOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=quest.generation_ms,
            fallback_reason=engine.status().load_error or "engine unavailable",
        )

    data, reason = extract_json(result.text)

    if data is None:
        quest = fallback_quest(req, seed=seed)
        quest.source = GeneratorSource.FALLBACK
        quest.generation_ms = int((time.perf_counter() - started) * 1000)
        return quest, PlanOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=quest.generation_ms,
            model_id=result.model_id,
            parse_ok=False,
            parse_reason=reason,
            fallback_reason=f"model output unparseable ({reason})",
        )

    try:
        quest = quest_from_model_data(data, req)
    except ValueError as exc:
        quest = fallback_quest(req, seed=seed)
        quest.source = GeneratorSource.FALLBACK
        quest.generation_ms = int((time.perf_counter() - started) * 1000)
        return quest, PlanOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=quest.generation_ms,
            model_id=result.model_id,
            parse_ok=False,
            parse_reason=str(exc)[:120],
            fallback_reason=f"model output failed validation: {exc}",
        )

    # Safety gate runs on model output exactly as it runs on fallback output.
    quest = safety.gate(quest, req)

    if not quest.safety.passed:
        blocked = safety.blocked_reason(quest)
        log.info("safety gate blocked model mission (%s); falling back", quest.title)
        safe = fallback_quest(req, seed=seed)
        safe.source = GeneratorSource.FALLBACK
        safe.generation_ms = int((time.perf_counter() - started) * 1000)
        safe.raw_model_output = result.text[:4000]
        return safe, PlanOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=safe.generation_ms,
            model_id=result.model_id,
            parse_ok=True,
            parse_reason=reason,
            safety_passed=False,
            fallback_reason=f"safety gate blocked model mission: {blocked}",
        )

    quest.source = GeneratorSource.MODEL
    quest.model_id = result.model_id
    quest.raw_model_output = result.text[:4000]
    quest.generation_ms = int((time.perf_counter() - started) * 1000)

    return quest, PlanOutcome(
        source=GeneratorSource.MODEL,
        latency_ms=quest.generation_ms,
        model_id=result.model_id,
        parse_ok=True,
        parse_reason=reason,
        safety_passed=True,
    )


def plan_with_retry(
    req: QuestRequest,
    engine: InferenceEngine,
    *,
    seed: int | None = None,
    timeout_s: float = GENERATION_TIMEOUT_S,
) -> tuple[Quest, PlanOutcome]:
    """`plan_quest` with one stricter retry when parsing fails.

    Only retries on *parse* failures -- a safety-blocked mission is a real
    signal that we should fall back, not a formatting problem to re-prompt.
    """
    quest, outcome = plan_quest(req, engine, seed=seed, timeout_s=timeout_s)

    if outcome.source is GeneratorSource.MODEL or outcome.parse_ok:
        return quest, outcome

    # One nudge, only if the engine actually produced text we could not parse.
    retry = engine.generate(compose_prompt(req) + _RETRY_SUFFIX, timeout_s=timeout_s)
    if retry is None or not retry.text.strip():
        return quest, outcome

    data, reason = extract_json(retry.text)
    if data is None:
        return quest, outcome

    try:
        candidate = quest_from_model_data(data, req)
    except ValueError:
        return quest, outcome

    candidate = safety.gate(candidate, req)
    if not candidate.safety.passed:
        return quest, outcome

    candidate.source = GeneratorSource.MODEL
    candidate.model_id = retry.model_id
    candidate.raw_model_output = retry.text[:4000]
    candidate.generation_ms = quest.generation_ms + retry.latency_ms

    return candidate, PlanOutcome(
        source=GeneratorSource.MODEL,
        latency_ms=candidate.generation_ms,
        model_id=retry.model_id,
        parse_ok=True,
        parse_reason=f"retry:{reason}",
        safety_passed=True,
        warnings=["recovered on retry"],
    )