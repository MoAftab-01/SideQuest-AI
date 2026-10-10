"""Evidence verification: pinning the project's honesty invariant.

`backend/app/ai/verifier.py` decides whether a player's photo supports the
mission, and `Verdict.is_ai_verified` is the single field the UI keys its
"verified" badge off. The module docstring states the rule as three conditions
that must *all* hold before that flag may be set:

    1. a model actually ran and returned a parseable verdict,
    2. the model did not ask for a better photo,
    3. the resulting level is VERIFIED or LIKELY.

This file exists so that a well-meaning edit cannot quietly break any of those
three, and cannot break the two adjacent guarantees the module leans on:

* **Uncertainty is a real result.** An ambiguous photo must come back
  UNCERTAIN with `needs_better_photo`, never a coin flip dressed as a score.
  In particular an *unrecognised* verdict label maps to UNCERTAIN -- the code
  calls this out as the overclaiming guard, and it is the easiest single line
  in the project to "fix" in the wrong direction.
* **`decode_data_url` is total.** The input is user-supplied and may be
  truncated by a flaky mobile upload, so every malformed shape must return
  `None` rather than raising. An exception here is a 500 on the verify route.

Everything runs against a tiny scripted engine that replays canned text. No
weights are loaded, no network is touched, and the whole module finishes in a
couple of seconds on CPU.
"""

from __future__ import annotations

import base64
import json

import pytest

from app.ai.engine import (
    EngineStatus,
    GenerateResult,
    GeneratorSource,
    InferenceEngine,
    UnavailableEngine,
)
from app.ai.verifier import (
    MAX_IMAGE_BYTES,
    _LEVEL_MAP,
    _REFUSALS,
    VerifyOutcome,
    compose_verify_prompt,
    decode_data_url,
    verdict_from_data,
    verify_evidence,
)
from app.models.schemas import (
    ConfidenceLevel,
    Evidence,
    Quest,
    QuestStep,
    VerificationSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A 1x1 PNG. 70 bytes decoded, so it exercises the decode path without
# pulling in Pillow or any image library.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PHOTO = f"data:image/png;base64,{TINY_PNG_B64}"

#: The JSON the verify prompt asks for, as a well-behaved model would answer it.
GOOD_VERDICT = {
    "verdict": "verified",
    "confidence": 0.91,
    "summary": "A live bird sits on the railing against open sky.",
    "matched": ["bird", "railing"],
    "missing": [],
    "needs_better_photo": False,
}


def make_quest(**verification_overrides) -> Quest:
    """A hand-built quest. Nothing here needs a generator or a store."""
    verification = dict(subject="One photograph of a live bird seen outdoors.")
    verification.update(verification_overrides)
    return Quest(
        id="verifyquest01",
        title="Count the Doorbells",
        hook="Twelve front doors, one walk.",
        premise="A slow look at the public face of a street.",
        steps=[QuestStep(order=1, instruction="Pick one stretch of pavement.", approx_minutes=4)],
        estimated_minutes=4,
        xp_reward=120,
        verification=VerificationSpec(**verification),
    )


QUEST = make_quest(
    must_include=["a live bird", "an outdoor setting"],
    accept_hints=["a silhouette against sky"],
    reject_if=["an illustration of a bird"],
    ambiguous_cases=["the bird is a blurred speck"],
)


class ScriptedEngine(InferenceEngine):
    """Replays canned text instead of running a model.

    Every knob `verify_evidence` branches on is settable, so "model present"
    and "model absent" run through the same code path. `images` records what
    the model was actually shown, which is how the tests prove a verdict was
    earned from a real image rather than from a button press.
    """

    def __init__(
        self,
        *,
        text: str | None = None,
        loaded: bool = True,
        vision: bool = True,
        model_id: str = "fake/scripted-vision",
        return_none: bool = False,
    ) -> None:
        self.model_id = model_id
        self.supports_vision = vision
        self._loaded = loaded
        self._text = text
        self._return_none = return_none
        #: Prompts this engine was asked to complete, in order.
        self.prompts: list[str] = []
        self.images: list[bytes] = []
        self.kwargs: list[dict] = []

    def load(self) -> bool:
        return self._loaded

    def generate(self, prompt: str, **kw) -> GenerateResult | None:
        return None

    def generate_with_image(self, prompt: str, image_bytes: bytes, **kw) -> GenerateResult | None:
        self.prompts.append(prompt)
        self.images.append(image_bytes)
        self.kwargs.append(kw)
        if self._return_none or not (self._loaded and self.supports_vision):
            return None
        return GenerateResult(
            text=self._text,
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


def verify(quest: Quest, evidence: Evidence, engine: InferenceEngine, **kw):
    """`verify_evidence` with the defaults a route would pass."""
    return verify_evidence(quest, evidence, engine, **kw)


# ---------------------------------------------------------------------------
# decode_data_url -- total by contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_url",
    [
        pytest.param("", id="empty-string"),
        pytest.param(None, id="none"),
        pytest.param("   ", id="whitespace-only"),
        pytest.param("not a url at all", id="free-text-garbage"),
        pytest.param("https://example.com/a.png", id="http-url-not-a-data-url"),
        pytest.param("data:", id="scheme-only"),
        pytest.param("data:image/png;base64", id="base64-marker-without-comma"),
        pytest.param("data:image/png;base64,", id="empty-payload"),
        pytest.param("data:image/png;base64,!!!!", id="no-base64-alphabet-at-all"),
        pytest.param("data:text/plain,hello", id="percent-encoded-not-base64"),
        pytest.param("data:image/png;base64,%%%", id="truncated-upload-percent-chars"),
        pytest.param("DATA:image/png;base64,aGVsbG8=", id="uppercase-scheme"),
        pytest.param("x" * 500, id="long-non-data-string"),
    ],
)
def test_decode_data_url_returns_none_for_anything_malformed(data_url):
    """The contract is `None`, never an exception.

    A raised exception here becomes a 500 on the verify route for a player
    whose upload was mangled in transit, which is exactly the moment the app
    must be at its calmest.
    """
    assert decode_data_url(data_url) is None


@pytest.mark.parametrize(
    "data_url",
    [
        pytest.param("data:image/png;base64,!!!not base64 at all!!!", id="punctuation-only"),
        pytest.param("data:image/png;base64,aGVsbG8-", id="url-safe-dash-not-base64"),
        pytest.param("data:image/png;base64,aGVs bG8=", id="interior-space"),
    ],
)
def test_decode_data_url_survives_inputs_that_cannot_really_be_an_image(data_url):
    """Total means total: whatever comes back is bytes or `None`, never a raise.

    `b64decode(validate=False)` discards characters outside the alphabet, so
    some junk decodes to *something*. These assertions do not care which; they
    pin the absence of an exception, which is the property the module promises.
    """
    result = decode_data_url(data_url)
    assert result is None or isinstance(result, bytes)


def test_decode_data_url_returns_the_exact_decoded_bytes():
    """The happy path must actually produce the image bytes."""
    raw = base64.b64decode(TINY_PNG_B64)

    assert decode_data_url(PHOTO) == raw
    assert decode_data_url(PHOTO).startswith(b"\x89PNG")


def test_decode_data_url_tolerates_surrounding_whitespace():
    """Mobile uploads arrive padded. Stripping is done before the regex match."""
    assert decode_data_url(f"  {PHOTO}\n") == decode_data_url(PHOTO)


def test_decode_data_url_accepts_media_type_parameters():
    """A declared parameter such as `;charset=utf-8` is a valid shape."""
    assert decode_data_url("data:text/plain;charset=utf-8;base64,aGVsbG8=") == b"hello"


def test_decode_data_url_accepts_a_data_url_with_no_media_type():
    assert decode_data_url("data:;base64,aGVsbG8=") == b"hello"


def test_decode_data_url_refuses_an_oversized_upload_without_decoding_it():
    """The size guard fires before any base64 work, so a huge upload is cheap."""
    limit = (MAX_IMAGE_BYTES * 4) // 3 + 1024

    assert decode_data_url("data:image/png;base64," + "A" * (limit + 1)) is None
    assert decode_data_url("data:image/png;base64," + "A" * (limit * 2)) is None


def test_the_size_guard_is_keyed_to_max_image_bytes():
    """Pin the guard's threshold so the constant and the check stay in step."""
    assert MAX_IMAGE_BYTES == 8 * 1024 * 1024
    # One char over the limit is refused; the constant is what sets the limit.
    just_over = "data:image/png;base64," + "A" * ((MAX_IMAGE_BYTES * 4) // 3 + 1024)
    assert len(just_over) > (MAX_IMAGE_BYTES * 4) // 3 + 1024
    assert decode_data_url(just_over) is None


# ---------------------------------------------------------------------------
# compose_verify_prompt -- what the 4B model is actually told
# ---------------------------------------------------------------------------


def test_verify_prompt_ends_with_the_be_strict_instruction():
    """The closing instruction is the anti-overclaiming lever on a small model.

    A 4B model will otherwise answer "yes, this is a leaf" for any photograph,
    so the last thing it reads must be told to reject rather than guess. This
    is asserted as the final non-empty line, not merely as a substring, so
    appending new closing guidance that dilutes it would fail.
    """
    prompt = compose_verify_prompt(QUEST)

    non_empty = [line for line in prompt.split("\n") if line.strip()]
    assert non_empty[-1] == (
        "Be strict. If the photo does not clearly show the requirement, say "
        "rejected or uncertain. Never guess that an unclear photo is correct."
    )


def test_verify_prompt_names_the_mission_and_the_requirement():
    prompt = compose_verify_prompt(QUEST)

    assert QUEST.title in prompt
    assert QUEST.premise in prompt
    assert QUEST.verification.subject in prompt


def test_verify_prompt_carries_every_verification_clause_when_present():
    """The mission-specific criteria are the whole point of this verification."""
    prompt = compose_verify_prompt(QUEST)
    v = QUEST.verification

    assert "Must also include: " + ", ".join(v.must_include) in prompt
    assert "Acceptable variations: " + ", ".join(v.accept_hints) in prompt
    assert "Reject if you see: " + ", ".join(v.reject_if) in prompt
    assert "Answer 'ambiguous' if: " + ", ".join(v.ambiguous_cases) in prompt


def test_verify_prompt_omits_clauses_the_mission_does_not_have():
    """A spec with only a subject must not prompt the model about absent rules.

    Asking a small model to weigh criteria the mission never defined invites
    it to invent some, which is the overclaiming this prompt is written against.
    """
    bare = make_quest()
    prompt = compose_verify_prompt(bare)

    assert bare.verification.subject in prompt
    for label in ("Must also include:", "Acceptable variations:", "Reject if you see:", "Answer 'ambiguous' if:"):
        assert label not in prompt


def test_verify_prompt_requests_exactly_the_fields_verdict_from_data_reads():
    """The prompt and the parser must agree, or a good photo is silently lost.

    `needs_better_photo` is the field the honesty invariant leans on hardest,
    so an edit to either side of this contract needs to break a test.
    """
    prompt = compose_verify_prompt(QUEST)

    for key in ("verdict", "confidence", "summary", "matched", "missing", "needs_better_photo"):
        assert f'"{key}"' in prompt
    for label in ("verified", "likely", "uncertain", "rejected"):
        assert label in prompt


def test_verify_prompt_never_leaks_a_python_none():
    """`compose_verify_prompt` filters None out of its line list; prove it works.

    A literal "None" in the prompt is a model asked to weigh a requirement that
    does not exist, which it will happily do.
    """
    quest_without_premise = make_quest()
    quest_without_premise.premise = ""

    for quest in (QUEST, quest_without_premise):
        assert "None" not in compose_verify_prompt(quest)


# ---------------------------------------------------------------------------
# verdict_from_data -- the level map
# ---------------------------------------------------------------------------


def test_the_level_map_holds_exactly_the_four_labels_the_prompt_requests():
    """A closed vocabulary: a label either means a documented thing or nothing."""
    assert set(_LEVEL_MAP) == {"verified", "likely", "uncertain", "rejected"}
    assert _LEVEL_MAP["verified"] is ConfidenceLevel.VERIFIED
    assert _LEVEL_MAP["likely"] is ConfidenceLevel.LIKELY
    assert _LEVEL_MAP["uncertain"] is ConfidenceLevel.UNCERTAIN
    assert _LEVEL_MAP["rejected"] is ConfidenceLevel.REJECTED


@pytest.mark.parametrize(
    ("label", "level", "verified"),
    [
        pytest.param("verified", ConfidenceLevel.VERIFIED, True, id="verified"),
        pytest.param("likely", ConfidenceLevel.LIKELY, True, id="likely"),
        pytest.param("uncertain", ConfidenceLevel.UNCERTAIN, False, id="uncertain"),
        pytest.param("rejected", ConfidenceLevel.REJECTED, False, id="rejected"),
        # Small models are inconsistent about case and padding.
        pytest.param("VERIFIED", ConfidenceLevel.VERIFIED, True, id="uppercase"),
        pytest.param("Likely", ConfidenceLevel.LIKELY, True, id="mixed-case"),
        pytest.param("  verified  ", ConfidenceLevel.VERIFIED, True, id="padded"),
    ],
)
def test_a_recognised_label_maps_to_its_level_and_sets_the_badge(label, level, verified):
    verdict = verdict_from_data({"verdict": label}, QUEST)

    assert verdict.level is level
    assert verdict.is_ai_verified is verified
    assert verdict.source is GeneratorSource.MODEL


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"verdict": "probably fine"}, id="unrecognised-word"),
        pytest.param({"verdict": "verified!"}, id="label-with-punctuation"),
        pytest.param({"verdict": "verified!"}, id="label-not-exactly-matched"),
        pytest.param({"verdict": "yes"}, id="bare-affirmative"),
        pytest.param({"verdict": "pass"}, id="bare-pass"),
        pytest.param({"verdict": "true"}, id="bare-true"),
        pytest.param({"verdict": ""}, id="empty-label"),
        pytest.param({}, id="label-key-absent"),
        pytest.param({"verdict": None}, id="label-null"),
        pytest.param({"verdict": {"level": "verified"}}, id="label-is-an-object"),
        pytest.param({"verdict": ["verified"]}, id="label-is-a-list"),
        pytest.param({"verdict": "v" * 60}, id="label-longer-than-the-parser-truncates"),
    ],
)
def test_an_unrecognised_label_is_uncertainty_never_a_pass(data):
    """The overclaiming guard.

    This is the single easiest line in the project to "fix" in the wrong
    direction -- defaulting an unknown label to VERIFIED would award the full
    reward for a reply the project does not understand.
    """
    verdict = verdict_from_data(data, QUEST)

    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert verdict.is_ai_verified is False


@pytest.mark.parametrize(
    ("key", "label"),
    [
        pytest.param("verdict", "verified", id="verdict"),
        pytest.param("level", "verified", id="level-alias"),
        pytest.param("result", "verified", id="result-alias"),
    ],
)
def test_a_bare_label_under_any_known_alias_still_goes_through_the_same_map(key, label):
    """Aliases are accepted for tolerance, but they are not a bypass.

    An alias must land on exactly the same verdict the canonical key would,
    including the unknown-label fallback.
    """
    assert verdict_from_data({key: label}, QUEST).level is ConfidenceLevel.VERIFIED

    unknown = verdict_from_data({key: "who knows"}, QUEST)
    assert unknown.level is ConfidenceLevel.UNCERTAIN
    assert unknown.is_ai_verified is False


def test_the_canonical_key_wins_when_two_aliases_disagree():
    verdict = verdict_from_data({"verdict": "rejected", "level": "verified"}, QUEST)

    assert verdict.level is ConfidenceLevel.REJECTED
    assert verdict.is_ai_verified is False


# ---------------------------------------------------------------------------
# verdict_from_data -- refusals are handled explicitly
# ---------------------------------------------------------------------------


def test_the_refusal_vocabulary_is_the_documented_one():
    """Pinned so widening or narrowing the list is a deliberate edit."""
    assert _REFUSALS == (
        "cannot determine",
        "can't determine",
        "not possible to",
        "i cannot tell",
        "unable to determine",
    )


@pytest.mark.parametrize("phrase", list(_REFUSALS))
def test_a_refusal_phrase_becomes_uncertain_with_needs_better_photo(phrase):
    """Model output meaning "I decline to judge" can never be laundered into a pass."""
    verdict = verdict_from_data({"verdict": phrase}, QUEST)

    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert verdict.is_ai_verified is False
    assert verdict.needs_better_photo is True
    # Confidence is zeroed: refusing is not partial evidence.
    assert verdict.confidence == 0.0


@pytest.mark.parametrize(
    "label",
    [
        pytest.param("I CANNOT TELL", id="uppercase-refusal"),
        pytest.param("Unable To Determine", id="mixed-case-refusal"),
        pytest.param("verified, i cannot tell", id="refusal-appended-after-a-pass"),
        pytest.param("likely -- it is not possible to tell", id="pass-then-hedge"),
    ],
)
def test_a_refusal_is_found_even_when_the_model_starts_optimistic(label):
    """A model that says "verified" and then hedges has refused.

    Matching on the whole lower-cased label, rather than on equality with a
    refusal token, is what stops `"verified ... i cannot tell"` from being
    read as a pass.
    """
    verdict = verdict_from_data({"verdict": label}, QUEST)

    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert verdict.is_ai_verified is False
    assert verdict.needs_better_photo is True


def test_a_refusal_in_the_summary_does_not_become_one():
    """The refusal check is on the label; the summary is prose, not a verdict.

    A model can legitimately write "I cannot determine the species" in a
    summary while still having judged the photo against the mission. Reading
    the summary as a refusal would discard a real judgement.
    """
    verdict = verdict_from_data(
        {"verdict": "likely", "summary": "I cannot determine the species, but a bird is clearly present."},
        QUEST,
    )

    assert verdict.level is ConfidenceLevel.LIKELY
    assert verdict.is_ai_verified is True


# ---------------------------------------------------------------------------
# verdict_from_data -- needs_better_photo overrides everything
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["verified", "likely", "uncertain", "rejected"])
def test_needs_better_photo_downgrades_even_a_verified_label(label):
    """Condition 2 of the invariant: asking for a better photo is not a pass.

    This is the case the API test calls out as the retry affordance. The model
    may have said "verified"; if it also said the photo was too unclear to
    rely on, the honest level is UNCERTAIN and the badge stays off.
    """
    verdict = verdict_from_data({"verdict": label, "needs_better_photo": True}, QUEST)

    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert verdict.is_ai_verified is False
    assert verdict.needs_better_photo is True


@pytest.mark.parametrize(
    "flag",
    [
        pytest.param(True, id="true"),
        pytest.param(1, id="int-one"),
        pytest.param("yes", id="string-yes"),
        pytest.param("false", id="string-false-is-truthy"),
        pytest.param("no", id="string-no-is-truthy"),
        pytest.param(["x"], id="non-empty-list"),
    ],
)
def test_any_truthy_needs_better_photo_flag_suppresses_the_badge(flag):
    """A sloppy flag fails closed.

    Small models emit the *string* "false" as often as the boolean, and
    `bool("false")` is True. The model asking for a better photo is the safe
    way to be wrong here: the player gets a retry, not a false badge.
    """
    verdict = verdict_from_data({"verdict": "verified", "needs_better_photo": flag}, QUEST)

    assert verdict.is_ai_verified is False
    assert verdict.level is ConfidenceLevel.UNCERTAIN


@pytest.mark.parametrize("flag", [False, 0, None, "", [], {}])
def test_a_falsy_needs_better_photo_flag_leaves_the_badge_alone(flag):
    verdict = verdict_from_data({"verdict": "verified", "needs_better_photo": flag}, QUEST)

    assert verdict.level is ConfidenceLevel.VERIFIED
    assert verdict.is_ai_verified is True
    assert verdict.needs_better_photo is False


def test_confidence_is_recorded_but_does_not_gate_the_badge():
    """The stated invariant is about the *level*, not the number.

    `verdict_from_data` has no confidence threshold, so a model that says
    "verified" with the lowest possible confidence still gets the badge. That
    matches the module docstring's three conditions, and this test exists so
    that adding a threshold later is a deliberate, visible change.
    """
    zero_confidence = verdict_from_data({"verdict": "verified", "confidence": 0.0}, QUEST)
    full_confidence = verdict_from_data({"verdict": "verified", "confidence": 1.0}, QUEST)

    assert zero_confidence.confidence == 0.0
    assert full_confidence.confidence == 1.0
    assert zero_confidence.is_ai_verified is full_confidence.is_ai_verified is True


# ---------------------------------------------------------------------------
# verdict_from_data -- the invariant as a single algebraic statement
# ---------------------------------------------------------------------------


def test_is_ai_verified_holds_if_and_only_if_the_three_conditions_hold():
    """Sweep the label space and assert the invariant directly.

    Rather than asserting case after case, this states the rule once and
    checks every combination: the badge is True exactly when the level counts
    as verified, the model did not ask for a better photo, and the label was
    not a refusal. Any future edit that breaks one leg of the conjunction fails
    here regardless of which input exposed it.
    """
    labels = [
        "verified", "likely", "uncertain", "rejected",
        "VERIFIED", "  likely  ",
        "verified!", "probably", "", "who knows",
        *(_REFUSALS),
        "i cannot tell", "I CANNOT DETERMINE",
    ]
    flags = [False, True]
    confidences = [0.0, 0.5, 1.0]

    checked = 0
    for label in labels:
        for flag in flags:
            for confidence in confidences:
                verdict = verdict_from_data(
                    {"verdict": label, "needs_better_photo": flag, "confidence": confidence},
                    QUEST,
                )

                refused = any(r in label.strip().lower() for r in _REFUSALS)
                expected = verdict.level.counts_as_verified and not flag and not refused

                assert verdict.is_ai_verified is expected, (
                    f"label={label!r} needs_better_photo={flag} confidence={confidence} "
                    f"gave level={verdict.level.value} is_ai_verified={verdict.is_ai_verified}"
                )
                checked += 1

    # Guard against a vacuous loop.
    assert checked == len(labels) * len(flags) * len(confidences)


# ---------------------------------------------------------------------------
# verdict_from_data -- shape of a good verdict
# ---------------------------------------------------------------------------


def test_a_full_verdict_carries_its_evidence_through():
    verdict = verdict_from_data(dict(GOOD_VERDICT), QUEST)

    assert verdict.quest_id == QUEST.id
    assert verdict.matched == ["bird", "railing"]
    assert verdict.missing == []
    assert verdict.summary == GOOD_VERDICT["summary"]
    assert verdict.confidence == pytest.approx(0.91)
    assert verdict.source is GeneratorSource.MODEL
    # The badge is claimed by no path except a model run.
    assert verdict.model_id is None
    assert verdict.xp_awarded == 0


def test_a_verdict_without_a_summary_says_what_was_judged():
    """An empty summary is replaced, not shipped, so the UI is never mute."""
    verdict = verdict_from_data({"verdict": "rejected"}, QUEST)

    assert verdict.summary == (
        f"Model judged this rejected against: {QUEST.verification.subject}"
    )


def test_a_verdict_never_awards_xp_itself():
    """XP is the API's decision, keyed on `is_ai_verified` -- not this module's.

    A module-level XP award would be a second, unguarded route to a reward.
    """
    for label in ("verified", "likely", "rejected"):
        assert verdict_from_data({"verdict": label}, QUEST).xp_awarded == 0


def test_evidence_lists_are_capped_so_the_schema_cannot_be_overrun():
    """A small model that lists everything must not produce an invalid verdict."""
    verdict = verdict_from_data(
        {"verdict": "likely", "matched": list("abcdefgh"), "missing": "one, two, three"},
        QUEST,
    )

    assert len(verdict.matched) == 6
    assert verdict.missing == ["one", "two", "three"]


# ---------------------------------------------------------------------------
# verify_evidence -- every path where no model ran
# ---------------------------------------------------------------------------


def test_no_photo_gives_a_heuristic_verdict_and_never_calls_the_model():
    """With no photo there is nothing to judge, and saying so is the answer."""
    engine = ScriptedEngine(text=json.dumps(GOOD_VERDICT))

    verdict, outcome = verify(QUEST, Evidence(note="went outside"), engine)

    assert verdict.level is ConfidenceLevel.HEURISTIC
    assert verdict.is_ai_verified is False
    assert verdict.source is GeneratorSource.FALLBACK
    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.notes == ["no_photo"]
    assert outcome.parse_ok is False
    assert outcome.image_bytes == 0
    # A confident note plus a fully working model still grants nothing.
    assert not engine.prompts, "the model was called with no photo to look at"


def test_an_empty_photo_string_is_treated_as_no_photo():
    verdict, outcome = verify(QUEST, Evidence(photo_data_url=""), ScriptedEngine(text=json.dumps(GOOD_VERDICT)))

    assert verdict.level is ConfidenceLevel.HEURISTIC
    assert verdict.is_ai_verified is False
    assert outcome.notes == ["no_photo"]


def test_an_undecodable_photo_never_reaches_the_model():
    """The image must be decoded *before* the engine is asked to look at it."""
    engine = ScriptedEngine(text=json.dumps(GOOD_VERDICT))

    verdict, outcome = verify(QUEST, Evidence(photo_data_url="data:image/png;base64,%%%"), engine)

    assert verdict.is_ai_verified is False
    assert verdict.needs_better_photo is True
    assert verdict.source is GeneratorSource.FALLBACK
    assert outcome.notes == ["undecodable_photo"]
    assert outcome.image_bytes == 0
    assert not engine.prompts
    assert not engine.images


def test_an_unloaded_engine_gives_an_uncertain_verdict_and_says_why():
    """The honest answer is "this was not checked", in those words.

    The UI renders `summary` next to the badge, so this string is load-bearing:
    it is what stops an unverified submission from looking merely unlucky.
    """
    verdict, outcome = verify(
        QUEST,
        Evidence(photo_data_url=PHOTO),
        ScriptedEngine(text=json.dumps(GOOD_VERDICT), loaded=False),
    )

    assert verdict.is_ai_verified is False
    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert verdict.model_id is None
    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.notes == ["engine_unloaded"]
    assert "not checked" in verdict.summary
    assert "model weights" in verdict.summary


def test_a_text_only_model_cannot_verify_a_photo():
    """A model that cannot see images has verified nothing, however well it writes."""
    engine = ScriptedEngine(text=json.dumps(GOOD_VERDICT), vision=False)

    verdict, outcome = verify(QUEST, Evidence(photo_data_url=PHOTO), engine)

    assert verdict.is_ai_verified is False
    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert outcome.notes == ["engine_text_only"]
    assert not engine.images, "the model was called without any image bytes"


def test_a_failed_generation_is_unverified_and_names_the_model_that_failed():
    """`generate_with_image` returning None is a normal outcome, not an error."""
    verdict, outcome = verify(
        QUEST,
        Evidence(photo_data_url=PHOTO),
        ScriptedEngine(return_none=True),
    )

    assert verdict.is_ai_verified is False
    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert outcome.notes == ["generation_failed"]
    assert outcome.model_id == "fake/scripted-vision"
    assert outcome.image_bytes > 0
    assert "could not be reached" in verdict.summary


def test_empty_model_output_is_treated_as_a_failed_generation():
    """A whitespace-only reply is not a verdict, however confident it sounds."""
    for text in ("", "   ", "\n\t "):
        verdict, outcome = verify(QUEST, Evidence(photo_data_url=PHOTO), ScriptedEngine(text=text))

        assert verdict.is_ai_verified is False
        assert outcome.notes == ["generation_failed"]


def test_an_unparseable_reply_is_unverified_and_records_why():
    """A confident-sounding sentence is not a verdict.

    The parse reason is kept because the eval suite reports a real
    malformed-output rate, and a fabricated one would flatter the model.
    """
    verdict, outcome = verify(
        QUEST,
        Evidence(photo_data_url=PHOTO),
        ScriptedEngine(text="Yes, that is definitely a bird!"),
    )

    assert verdict.is_ai_verified is False
    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert verdict.source is GeneratorSource.FALLBACK
    assert outcome.notes == ["unparseable_verdict"]
    assert outcome.parse_ok is False
    assert outcome.parse_reason == "unparseable_json"
    assert "could not be read" in verdict.summary


def test_a_real_unavailable_engine_still_yields_an_unverified_verdict():
    """The shipped zero-weights configuration, end to end through this module."""
    verdict, outcome = verify(QUEST, Evidence(photo_data_url=PHOTO), UnavailableEngine("no engine registered"))

    assert verdict.is_ai_verified is False
    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert verdict.source is GeneratorSource.FALLBACK
    assert outcome.notes == ["engine_unloaded"]


# ---------------------------------------------------------------------------
# verify_evidence -- the paths where a model really did run
# ---------------------------------------------------------------------------


def test_a_parseable_verdict_from_a_vision_model_is_the_only_earned_pass():
    """All three conditions met: model ran, did not ask for a retry, said verified."""
    engine = ScriptedEngine(text=json.dumps(GOOD_VERDICT))

    verdict, outcome = verify(QUEST, Evidence(photo_data_url=PHOTO), engine)

    assert verdict.is_ai_verified is True
    assert verdict.level is ConfidenceLevel.VERIFIED
    assert verdict.source is GeneratorSource.MODEL
    assert verdict.model_id == "fake/scripted-vision"
    assert outcome.source is GeneratorSource.MODEL
    assert outcome.parse_ok is True
    assert outcome.parse_reason == "ok"
    assert outcome.image_bytes == len(engine.images[0]) > 0


def test_the_model_is_handed_the_composed_prompt_and_the_real_image_bytes():
    """The verdict is only earned if the right bytes and the right question arrived."""
    engine = ScriptedEngine(text=json.dumps(GOOD_VERDICT))

    verify(QUEST, Evidence(photo_data_url=PHOTO), engine)

    assert len(engine.prompts) == 1
    assert engine.prompts[0] == compose_verify_prompt(QUEST)
    assert engine.images == [decode_data_url(PHOTO)]


def test_the_verify_timeout_is_forwarded_to_the_engine():
    engine = ScriptedEngine(text=json.dumps(GOOD_VERDICT))

    verify(QUEST, Evidence(photo_data_url=PHOTO), engine, timeout_s=1.5)

    assert engine.kwargs == [{"timeout_s": 1.5}]


@pytest.mark.parametrize(
    ("wrapper", "reason"),
    [
        pytest.param(lambda j: j, "ok", id="bare-json"),
        pytest.param(lambda j: f"Here is my answer:\n```json\n{j}\n```", "ok_fenced", id="fenced"),
        pytest.param(lambda j: f"I looked carefully. {j}", "ok_extracted", id="prose-wrapped"),
    ],
)
def test_a_verdict_wrapped_in_prose_still_counts_as_a_model_verdict(wrapper, reason):
    """Small models fence and chat. The badge needs a parse, not a clean reply."""
    engine = ScriptedEngine(text=wrapper(json.dumps(GOOD_VERDICT)))

    verdict, outcome = verify(QUEST, Evidence(photo_data_url=PHOTO), engine)

    assert verdict.is_ai_verified is True
    assert outcome.parse_reason == reason


def test_the_raw_model_output_is_kept_for_the_eval_suite_and_truncated():
    """`raw_model_output` is the evidence behind every number the eval reports."""
    raw = json.dumps({"verdict": "verified", "summary": "x" * 5000})

    verdict, _ = verify(QUEST, Evidence(photo_data_url=PHOTO), ScriptedEngine(text=raw))

    assert verdict.raw_model_output == raw[:4000]
    assert len(verdict.raw_model_output) == 4000


def test_a_needs_better_photo_reply_never_reaches_the_badge_through_this_entry_point():
    """Condition 2, exercised through `verify_evidence` rather than the parser."""
    engine = ScriptedEngine(
        text=json.dumps({"verdict": "verified", "confidence": 0.95, "needs_better_photo": True})
    )

    verdict, outcome = verify(QUEST, Evidence(photo_data_url=PHOTO), engine)

    assert verdict.needs_better_photo is True
    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert verdict.is_ai_verified is False
    # A model ran here, so the outcome must not claim otherwise.
    assert outcome.source is GeneratorSource.MODEL
    assert outcome.parse_ok is True


def test_a_refusal_through_this_entry_point_is_uncertain_with_a_retry_affordance():
    engine = ScriptedEngine(text=json.dumps({"verdict": "I cannot determine what is in this photo"}))

    verdict, outcome = verify(QUEST, Evidence(photo_data_url=PHOTO), engine)

    assert verdict.level is ConfidenceLevel.UNCERTAIN
    assert verdict.is_ai_verified is False
    assert verdict.needs_better_photo is True
    assert verdict.confidence == 0.0
    assert outcome.source is GeneratorSource.MODEL


def test_the_model_verdict_carries_the_latency_of_the_whole_check():
    verdict, outcome = verify(QUEST, Evidence(photo_data_url=PHOTO), ScriptedEngine(text=json.dumps(GOOD_VERDICT)))

    assert verdict.latency_ms == outcome.latency_ms
    assert outcome.latency_ms >= 0


# ---------------------------------------------------------------------------
# verify_evidence -- the invariant across the whole entry point
# ---------------------------------------------------------------------------


#: Every configuration in which no model produced a parseable verdict. Every
#: one of these must land on `is_ai_verified is False`; the whole point of the
#: project is that this list cannot be shortened by a careless edit.
NO_MODEL_RAN = [
    pytest.param(
        {"evidence": Evidence(note="done"), "engine": ScriptedEngine(text=json.dumps(GOOD_VERDICT))},
        "heuristic",
        id="no-photo-with-a-model-available",
    ),
    pytest.param(
        {"evidence": Evidence(photo_data_url="data:image/png;base64,%%%"),
         "engine": ScriptedEngine(text=json.dumps(GOOD_VERDICT))},
        "uncertain",
        id="undecodable-photo-with-a-model-available",
    ),
    pytest.param(
        {"evidence": Evidence(photo_data_url=PHOTO),
         "engine": ScriptedEngine(text=json.dumps(GOOD_VERDICT), loaded=False)},
        "uncertain",
        id="engine-unloaded",
    ),
    pytest.param(
        {"evidence": Evidence(photo_data_url=PHOTO),
         "engine": ScriptedEngine(text=json.dumps(GOOD_VERDICT), vision=False)},
        "uncertain",
        id="engine-cannot-see",
    ),
    pytest.param(
        {"evidence": Evidence(photo_data_url=PHOTO), "engine": ScriptedEngine(return_none=True)},
        "uncertain",
        id="generation-returned-none",
    ),
    pytest.param(
        {"evidence": Evidence(photo_data_url=PHOTO), "engine": ScriptedEngine(text="   ")},
        "uncertain",
        id="generation-returned-blank",
    ),
    pytest.param(
        {"evidence": Evidence(photo_data_url=PHOTO),
         "engine": ScriptedEngine(text="That is a bird, definitely!")},
        "uncertain",
        id="reply-was-not-json",
    ),
    pytest.param(
        {"evidence": Evidence(photo_data_url=PHOTO), "engine": UnavailableEngine("no engine registered")},
        "uncertain",
        id="shipped-zero-weights-configuration",
    ),
]


@pytest.mark.parametrize(("config", "level"), NO_MODEL_RAN)
def test_no_configuration_without_a_model_verdict_sets_the_badge(config, level):
    """The badge is unreachable from every fallback path, in one sweep.

    Each entry pairs a real configuration with the level a player is entitled
    to see. If a fallback ever starts producing a level that counts as
    verified, this fails with the configuration named rather than in a
    one-off test nobody connects to the claim.
    """
    verdict, outcome = verify(QUEST, config["evidence"], config["engine"])

    assert verdict.is_ai_verified is False
    assert verdict.level.value == level
    assert not verdict.level.counts_as_verified
    assert verdict.source is GeneratorSource.FALLBACK
    assert outcome.source is GeneratorSource.FALLBACK
    assert outcome.parse_ok is False
    assert verdict.xp_awarded == 0


@pytest.mark.parametrize(("config", "level"), NO_MODEL_RAN)
def test_every_fallback_explains_itself_in_a_note(config, level):
    """`notes` is the machine-readable reason the fallback fired.

    An empty `notes` list on a fallback path would mean a judge reading the
    verification record has no way to tell which guard produced the result.
    """
    _, outcome = verify(QUEST, config["evidence"], config["engine"])

    assert outcome.notes, "a fallback recorded no reason"
    assert all(isinstance(note, str) and note for note in outcome.notes)


# ---------------------------------------------------------------------------
# VerifyOutcome -- the bookkeeping the API and eval suite read
# ---------------------------------------------------------------------------


def test_verify_outcome_defaults_are_conservative():
    """A freshly built outcome claims nothing about a model run."""
    outcome = VerifyOutcome(source=GeneratorSource.MODEL, latency_ms=3)

    assert outcome.parse_ok is False
    assert outcome.parse_reason == ""
    assert outcome.image_bytes == 0
    assert outcome.model_id is None
    assert outcome.notes == []
    assert outcome.refused is False


def test_outcome_records_the_image_size_only_once_it_has_been_decoded():
    """`image_bytes` is 0 on the paths where decoding never happened.

    Reporting a size for a photo that was never decoded would imply the app
    inspected an upload it did not look at.
    """
    decoded = ScriptedEngine(text=json.dumps(GOOD_VERDICT), loaded=False)
    _, ok = verify(QUEST, Evidence(photo_data_url=PHOTO), decoded)
    _, bad = verify(QUEST, Evidence(photo_data_url="data:image/png;base64,%%%"), decoded)

    assert ok.image_bytes == len(decode_data_url(PHOTO)) > 0
    assert bad.image_bytes == 0