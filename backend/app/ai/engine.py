"""Inference engine abstraction for OpenRouter and deterministic fallback."""

from __future__ import annotations

import abc
import os

import logging

from app.models.schemas import EngineStatus, GenerateResult, GeneratorSource

#: Hard ceiling on generation. Mission text is short; a runaway decode is a bug
#: or a misbehaving model, not a feature.
MAX_NEW_TOKENS = 2048
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

    This is what the app runs without OpenRouter credentials. Having it be a
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
    """Build Groq when configured; else OpenRouter; else honest fallback."""
    import logging
    log = logging.getLogger(__name__)
    from app.ai.groq import GroqEngine
    from app.ai.provider import ProviderEngine
    # Priority 1: Groq (if GROQ_API_KEY set)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        groq = GroqEngine()
        if groq.load():
            return groq
        log.info("Groq configured but load() False (%s); falling through", groq.last_error)
    # Priority 2: OpenRouter
    provider = ProviderEngine()
    if provider.load():
        return provider
    # Priority 3: honest fallback
    msg = ""
    if groq_key and groq.last_error:
        msg = f"Groq: {groq.last_error}; OpenRouter: {provider.last_error or 'unavailable'}"
    else:
        msg = provider.last_error or "OpenRouter unavailable"
    return UnavailableEngine(msg)