"""Inference engine abstraction.

Everything above this layer (planner, verifier, API) talks to `InferenceEngine`
and never imports torch or transformers directly. That separation is what makes
the "does it still work with the model unavailable?" question answerable: the
deterministic fallback is just another engine.

Two implementations ship:
  * `GemmaEngine`     -- local open-weight Gemma via transformers (see gemma_engine.py)
  * `UnavailableEngine` -- a null engine used when no backend can load

`engine_status()` reports what is actually loaded, so the API never claims more
capability than it has.
"""

from __future__ import annotations

import abc
import logging
import os
import time

from app.models.schemas import EngineStatus, GenerateResult, GeneratorSource

log = logging.getLogger(__name__)

#: Default open-weight model. Gemma 3 4B is the primary target; the 1B variant
#: is the low-resource profile documented in the README.
DEFAULT_MODEL_ID = os.environ.get("SIDEQUEST_MODEL_ID", "google/gemma-3-4b-it")
FALLBACK_MODEL_ID = os.environ.get("SIDEQUEST_FALLBACK_MODEL_ID", "google/gemma-3-1b-it")

#: Hard ceiling on generation. Mission text is short; a runaway decode is a bug
#: or a misbehaving model, not a feature.
MAX_NEW_TOKENS = 512
#: Generation deadline. Exceeding it returns None and the caller falls back.
GENERATION_TIMEOUT_S = float(os.environ.get("SIDEQUEST_GEN_TIMEOUT", "45"))
#: Inference deadline for verification.
VERIFY_TIMEOUT_S = float(os.environ.get("SIDEQUEST_VERIFY_TIMEOUT", "60"))


class InferenceEngine(abc.ABC):
    """A text (and optionally image) generator the rest of the app can call."""

    model_id: str | None = None
    supports_vision: bool = False

    @abc.abstractmethod
    def load(self) -> bool:
        """Best-effort load. Returns True if the backend is usable."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.7,
        timeout_s: float = GENERATION_TIMEOUT_S,
    ) -> GenerateResult | None:
        """Return a generation, or None if the backend could not serve one.

        Returning None (rather than raising) is deliberate: the caller's
        fallback path is a normal code path, not an error handler.
        """

    @abc.abstractmethod
    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.2,
        timeout_s: float = VERIFY_TIMEOUT_S,
    ) -> GenerateResult | None:
        """Vision path. Return None when the backend cannot see images."""

    @abc.abstractmethod
    def status(self) -> EngineStatus:
        """Honest report of what is loaded, for /api/health."""

    def unload(self) -> None:  # pragma: no cover - trivial
        pass


class UnavailableEngine(InferenceEngine):
    """Null engine: always reports itself unusable, always returns None.

    This is what the app runs with zero model weights installed. Having it be a
    real class rather than a code path means the no-model behaviour is covered
    by the same tests as every other configuration.
    """

    model_id = None

    def __init__(self, reason: str = "no engine registered") -> None:
        self.reason = reason

    def load(self) -> bool:
        return False

    def generate(self, prompt: str, **kw) -> GenerateResult | None:
        return None

    def generate_with_image(self, prompt: str, image_bytes: bytes, **kw) -> GenerateResult | None:
        return None

    def status(self) -> EngineStatus:
        return EngineStatus(
            model_id=None,
            loaded=False,
            device="none",
            dtype="none",
            supports_vision=False,
            load_error=self.reason,
        )


def build_engine() -> InferenceEngine:
    """Construct the best available engine for this machine.

    Order (verified from design): provider (external inference) preferred when
    configured; then Gemma (local, preserved); then unavailable (deterministic
    fallback, never claims model result).
    """
    # Provider first — keeps Gemma available but not loaded when provider configured
    from app.ai.provider import OPENROUTER_KEY, ProviderEngine, UnavailableProvider
    if OPENROUTER_KEY:
        provider = ProviderEngine()
        try:
            if provider.load():
                return provider
        except Exception as exc:
            log.warning("provider load failed: %s", exc)
        return UnavailableProvider(f"provider failed: {provider.last_error or 'unknown'}")

    # Gemma preserved (self-hosting available when practical)
    try:
        from app.ai.gemma_engine import GemmaEngine
    except Exception as exc:  # pragma: no cover - import guard
        log.warning("transformers stack unavailable: %s", exc)
        return UnavailableEngine(f"transformers import failed: {exc}")

    reasons: list[str] = []
    for candidate in (DEFAULT_MODEL_ID, FALLBACK_MODEL_ID):
        if candidate in reasons:
            continue
        eng = GemmaEngine(candidate)
        try:
            if eng.load():
                return eng
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("engine %s failed to load: %s", candidate, exc)
        reasons.append(f"{candidate}: {eng.last_error or 'load failed'}")
        eng.unload()

    return UnavailableEngine("; ".join(reasons) or "no model candidates configured")


class TimedMixin:
    """Small helper so latency is measured the same way everywhere."""

    @staticmethod
    def elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)


def fallback_result(
    text: str, *, reason: str, latency_ms: int = 0
) -> GenerateResult:
    """Build a GenerateResult tagged as a fallback, for uniform logging."""
    return GenerateResult(
        text=text,
        latency_ms=latency_ms,
        model_id=None,
        source=GeneratorSource.FALLBACK,
        failure_reason=reason,
    )