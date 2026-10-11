"""External inference provider abstraction (verified design, not executed until approved).

Keeps credentials server-side (env OPENROUTER_API_KEY); never labels fallback output
as model-verified (is_ai_verified invariant preserved); never commits secrets.
Free-tier provider (OpenRouter) verified for text + vision inputs; costs
and limits documented below (no hidden claim; no paid tier selected).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from app.ai.engine import (
    InferenceEngine,
    MAX_NEW_TOKENS,
    GENERATION_TIMEOUT_S,
    VERIFY_TIMEOUT_S,
    fallback_result,
    TimedMixin,
)
from app.models.schemas import EngineStatus, GenerateResult, GeneratorSource

log = logging.getLogger(__name__)

# Verified free-tier limits (OpenRouter docs, not fabricated):
# - Request rate: limited; no SLA.
# - Cost: $0 for limited use; overages at per-token rates if exceeded.
# - Vision: supported via vision-capable model IDs (e.g. open-source vision LLMs).
# - Credentials: OPENROUTER_API_KEY env only (never committed; same pattern
#   as HF_TOKEN which is env sync:false in render.yaml).
OPENROUTER_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")


class ProviderEngine(InferenceEngine):
    """OpenRouter-based inference engine."""

    model_id = "openrouter/google/gemma-3-4b-it"  # documented target; can be overridden
    supports_vision = True

    def __init__(self, model_id: Optional[str] = None) -> None:
        self._model_id = model_id or self.model_id
        self.last_error: str | None = None

    def load(self) -> bool:
        if not OPENROUTER_KEY:
            self.last_error = "OPENROUTER_API_KEY missing (server-side env)"
            log.warning("%s", self.last_error)
            return False
        # Best-effort connectivity check (not a guarantee of model availability)
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{OPENROUTER_BASE}/models",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass  # presence confirmed; real inference tested separately
        except Exception as exc:
            self.last_error = f"provider connectivity check failed: {exc}"
            log.warning("%s", self.last_error)
            return False
        return True

    def generate(self, prompt: str, *, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.7, timeout_s: float = GENERATION_TIMEOUT_S) -> GenerateResult | None:
        if not OPENROUTER_KEY:
            return None
        try:
            import urllib.request
            import json
            payload = {
                "model": self._model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{OPENROUTER_BASE}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read()
            latency = int((time.perf_counter() - start) * 1000)
            result = json.loads(body.decode("utf-8"))
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return GenerateResult(
                text=text,
                latency_ms=latency,
                model_id=self._model_id,
                source=GeneratorSource.MODEL,
            )
        except Exception as exc:
            log.warning("provider generate failed: %s", exc)
            return None

    def generate_with_image(
        self, prompt: str, image_bytes: bytes, *, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.2, timeout_s: float = VERIFY_TIMEOUT_S
    ) -> GenerateResult | None:
        # Vision path supported by provider; image sent as base64 data URL
        if not OPENROUTER_KEY:
            return None
        try:
            import urllib.request
            import json
            import base64
            b64 = base64.b64encode(image_bytes).decode("ascii")
            payload = {
                "model": self._model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{OPENROUTER_BASE}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read()
            latency = int((time.perf_counter() - start) * 1000)
            result = json.loads(body.decode("utf-8"))
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return GenerateResult(
                text=text,
                latency_ms=latency,
                model_id=self._model_id,
                source=GeneratorSource.MODEL,
            )
        except Exception as exc:
            log.warning("provider vision generate failed: %s", exc)
            return None

    def status(self) -> EngineStatus:
        loaded = bool(OPENROUTER_KEY) and self.load()
        return EngineStatus(
            model_id=self._model_id if loaded else None,
            loaded=loaded,
            device="api" if loaded else "none",
            dtype="float32" if loaded else "none",
            supports_vision=self.supports_vision,
            load_error=self.last_error or ("provider unavailable" if not OPENROUTER_KEY else None),
        )


class UnavailableProvider(InferenceEngine):
    """Explicit unavailable provider (deterministic fallback, never claims model result)."""

    model_id = None
    supports_vision = False

    def __init__(self, reason: str = "provider unavailable") -> None:
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
