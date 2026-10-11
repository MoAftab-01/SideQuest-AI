"""Evidence verification: decide whether a photo supports the mission.

The project's hardest honesty rule lives here. Three things must all be true
before `Verdict.is_ai_verified` is set:

  1. a model actually ran and returned a parseable verdict,
  2. the model did not ask for a better photo,
  3. the resulting level is VERIFIED or LIKELY.

If the model is absent, the verdict is HEURISTIC and `is_ai_verified` stays
False. No combination of heuristic signals is allowed to set it. The UI keys
its "verified" badge off this field alone, so a bug here would be a bug in a
claim the project makes to a judge.

The other half of honesty is *uncertainty*: a photo that is genuinely
ambiguous must come back UNCERTAIN with `needs_better_photo`, not a coin flip
dressed up as a score. Ambiguity is a first-class result here, not a failure.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import time
from dataclasses import dataclass, field

from app.ai.engine import VERIFY_TIMEOUT_S, InferenceEngine
from app.ai.fallback import fallback_verdict
from app.ai.parsing import coerce_float, coerce_list, coerce_str, extract_json
from app.models.schemas import (
    ConfidenceLevel,
    Evidence,
    GeneratorSource,
    Quest,
    Verdict,
)

log = logging.getLogger(__name__)

#: Refuse absurd uploads early rather than at model-tokenisation time.
MAX_IMAGE_BYTES = 8 * 1024 * 1024
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<params>(?:;[\w.-]+=[^;,]*)*)?;base64,(?P<payload>.*)$", re.DOTALL)


@dataclass
class VerifyOutcome:
    source: GeneratorSource
    latency_ms: int
    model_id: str | None = None
    parse_ok: bool = False
    parse_reason: str = ""
    image_bytes: int = 0
    refused: bool = False
    notes: list[str] = field(default_factory=list)


def decode_data_url(data_url: str) -> bytes | None:
    """Decode a base64 image data URL. Returns None for anything malformed.

    Deliberately total: the input is user-supplied and may be truncated by a
    flaky mobile upload, so this never raises.
    """
    if not data_url or len(data_url) > (MAX_IMAGE_BYTES * 4) // 3 + 1024:
        return None

    match = _DATA_URL_RE.match(data_url.strip())
    if match is None:
        return None

    payload = match.group("payload")
    if not payload:
        return None

    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None

    return raw or None


def compose_verify_prompt(quest: Quest) -> str:
    """Build the vision-verification prompt.

    The prompt is deliberately short and explicit about the failure mode we
    care about: a 4B model will otherwise answer "yes, this is a leaf" for any
    photograph. It is told to reject, and to name a better photo instead.
    """
    v = quest.verification
    lines = [
        "You are judging one photograph against a single mission requirement.",
        "",
        "MISSION:",
        quest.title,
        quest.premise or "",
        "",
        "THE PHOTO MUST SHOW:",
        v.subject,
    ]
    if v.must_include:
        lines.append("Must also include: " + ", ".join(v.must_include))
    if v.accept_hints:
        lines.append("Acceptable variations: " + ", ".join(v.accept_hints))
    if v.reject_if:
        lines.append("Reject if you see: " + ", ".join(v.reject_if))
    if v.ambiguous_cases:
        lines.append(
            "Answer 'ambiguous' if: " + ", ".join(v.ambiguous_cases)
        )

    lines += [
        "",
        "Answer with ONE JSON object, nothing else:",
        '{"verdict": "verified|likely|uncertain|rejected",',
        ' "confidence": 0.0-1.0,',
        ' "summary": "one short sentence",',
        ' "matched": ["things clearly present"],',
        ' "missing": ["things clearly absent"],',
        ' "needs_better_photo": true or false}',
        "",
        "Be strict. If the photo does not clearly show the requirement, say "
        "rejected or uncertain. Never guess that an unclear photo is correct.",
    ]
    return "\n".join(l for l in lines if l is not None)


_LEVEL_MAP: dict[str, ConfidenceLevel] = {
    "verified": ConfidenceLevel.VERIFIED,
    "likely": ConfidenceLevel.LIKELY,
    "uncertain": ConfidenceLevel.UNCERTAIN,
    "rejected": ConfidenceLevel.REJECTED,
}

#: Model output that means "I decline to judge", handled explicitly so it can
#: never be laundered into a pass.
_REFUSALS = ("cannot determine", "can't determine", "not possible to", "i cannot tell", "unable to determine")


def verdict_from_data(data: dict, quest: Quest) -> Verdict:
    """Build a `Verdict` from parsed model JSON. The only place `is_ai_verified` is set True."""
    raw_level = coerce_str(data.get("verdict") or data.get("level") or data.get("result"), max_len=40).lower()

    if any(r in raw_level for r in _REFUSALS):
        return Verdict(
            quest_id=quest.id,
            level=ConfidenceLevel.UNCERTAIN,
            is_ai_verified=False,
            confidence=0.0,
            summary="The model would not judge this photo. Try a clearer one.",
            needs_better_photo=True,
            source=GeneratorSource.MODEL,
        )

    level = _LEVEL_MAP.get(raw_level)
    if level is None:
        # An unrecognised label is uncertainty, not a pass. Defaulting unknown
        # to "verified" would be the exact overclaiming this project forbids.
        level = ConfidenceLevel.UNCERTAIN

    confidence = coerce_float(data.get("confidence"), lo=0.0, hi=1.0, default=0.5)
    needs_better = bool(data.get("needs_better_photo"))

    if needs_better:
        level = ConfidenceLevel.UNCERTAIN

    summary = coerce_str(data.get("summary") or data.get("reason"), max_len=300)
    if not summary:
        summary = f"Model judged this {level.value} against: {quest.verification.subject}"

    verdict = Verdict(
        quest_id=quest.id,
        level=level,
        is_ai_verified=level.counts_as_verified and not needs_better,
        confidence=confidence,
        summary=summary,
        matched=coerce_list(data.get("matched"), limit=6),
        missing=coerce_list(data.get("missing"), limit=6),
        needs_better_photo=needs_better,
        source=GeneratorSource.MODEL,
    )
    return verdict


def verify_evidence(
    quest: Quest,
    evidence: Evidence,
    engine: InferenceEngine,
    *,
    timeout_s: float = VERIFY_TIMEOUT_S,
) -> tuple[Verdict, VerifyOutcome]:
    """Verify evidence against a mission. Always returns a verdict, never raises."""
    started = time.perf_counter()

    # No photo: we cannot claim anything, and saying so plainly is the correct
    # product behaviour rather than a gap.
    if not evidence.photo_data_url:
        verdict = fallback_verdict(quest, evidence)
        verdict.summary = "No photo attached, so there is nothing for a model to check."
        return verdict, VerifyOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=int((time.perf_counter() - started) * 1000),
            notes=["no_photo"],
        )

    image_bytes = decode_data_url(evidence.photo_data_url)
    if image_bytes is None:
        verdict = fallback_verdict(quest, evidence)
        verdict.summary = "That photo could not be decoded. Try uploading it again."
        verdict.needs_better_photo = True
        return verdict, VerifyOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=int((time.perf_counter() - started) * 1000),
            notes=["undecodable_photo"],
        )

    status = engine.status()
    if not status.loaded:
        verdict = fallback_verdict(quest, evidence)
        verdict.summary = (
            "OpenRouter is unavailable, so this photo was not checked. "
            "Configure a valid OPENROUTER_API_KEY to enable verification."
        )
        return verdict, VerifyOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=int((time.perf_counter() - started) * 1000),
            image_bytes=len(image_bytes),
            notes=["engine_unloaded"],
        )

    if not engine.supports_vision:
        verdict = fallback_verdict(quest, evidence)
        verdict.summary = (
            f"The loaded model ({status.model_id}) cannot see images, so this "
            "photo was not checked."
        )
        return verdict, VerifyOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=int((time.perf_counter() - started) * 1000),
            image_bytes=len(image_bytes),
            notes=["engine_text_only"],
        )

    result = engine.generate_with_image(
        compose_verify_prompt(quest), image_bytes, timeout_s=timeout_s
    )

    if result is None or not result.text.strip():
        verdict = fallback_verdict(quest, evidence)
        verdict.summary = "The model could not be reached, so this photo is unverified."
        return verdict, VerifyOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_id=status.model_id,
            image_bytes=len(image_bytes),
            notes=["generation_failed"],
        )

    data, reason = extract_json(result.text)

    if data is None:
        verdict = fallback_verdict(quest, evidence)
        verdict.summary = "The model's reply could not be read, so this photo is unverified."
        return verdict, VerifyOutcome(
            source=GeneratorSource.FALLBACK,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_id=result.model_id,
            parse_ok=False,
            parse_reason=reason,
            image_bytes=len(image_bytes),
            notes=["unparseable_verdict"],
        )

    verdict = verdict_from_data(data, quest)
    verdict.model_id = result.model_id
    verdict.raw_model_output = result.text[:4000]
    verdict.latency_ms = int((time.perf_counter() - started) * 1000)

    return verdict, VerifyOutcome(
        source=GeneratorSource.MODEL,
        latency_ms=verdict.latency_ms,
        model_id=result.model_id,
        parse_ok=True,
        parse_reason=reason,
        image_bytes=len(image_bytes),
    )