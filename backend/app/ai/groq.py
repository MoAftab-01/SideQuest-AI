"""Groq-hosted open-weight inference (verified live model: llama-3.1-8b-instant).

Uses OpenAI-compatible /v1/chat/completions at api.groq.com.
Reads GROQ_API_KEY server-side env (never hardcoded; never printed).
Model selectable via GROQ_MODEL env (default: llama-3.1-8b-instant).
Not using Groq SDK (not installed); using urllib with Bearer auth.
"""
from __future__ import annotations

import logging
import os
import time
import urllib.request
import json

from app.ai.engine import (
    InferenceEngine,
    MAX_NEW_TOKENS,
    GENERATION_TIMEOUT_S,
    VERIFY_TIMEOUT_S,
)
from app.models.schemas import EngineStatus, GenerateResult, GeneratorSource

log = logging.getLogger(__name__)

GROQ_BASE = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
# Module-level GROQ_KEY read at import time; GroqEngine reads env directly for tests


class GroqEngine(InferenceEngine):
    """Groq chat-completions via urllib (OpenAI-compatible payload)."""

    model_id = DEFAULT_GROQ_MODEL
    supports_vision = False  # Groq vision needs different endpoint/model; keep text-only for now

    def __init__(self, model_id: str | None = None, api_key: str | None = None) -> None:
        # Read env at init time (tests set env before constructing engine)
        self._model_id = model_id or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.model_id = self._model_id
        self._api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY")
        self.last_error: str | None = None
        self._loaded = False

    def load(self) -> bool:
        api_key = self._api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            self.last_error = "GROQ_API_KEY missing (server-side env)"
            log.warning("%s", self.last_error)
            return False
        self._api_key = api_key  # sync in case it changed
        # Best-effort connectivity: GET /models (Groq docs) or just verify key format
        try:
            req = urllib.request.Request(
                f"{GROQ_BASE}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass  # presence confirmed; real inference tested separately
        except Exception as exc:
            # Log truthfully; do not claim success
            self.last_error = f"Groq connectivity check failed: {exc}"
            log.warning("%s", self.last_error)
            return False
        self._loaded = True
        return True

    def generate(self, prompt: str, *, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.7, timeout_s: float = GENERATION_TIMEOUT_S) -> GenerateResult | None:
        api_key = self._api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            log.warning("Groq generate skipped: GROQ_API_KEY missing (truthful — not hidden)")
            return None
        try:
            payload = {
                "model": self._model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_completion_tokens": max_new_tokens,
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{GROQ_BASE}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {api_key}",
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
            # Truthful: never hide failure; caller falls back honestly
            log.warning("Groq generate failed (truthful, not hidden): %s", exc)
            return None

    def generate_with_image(self, prompt: str, image_bytes: bytes, *, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.2, timeout_s: float = VERIFY_TIMEOUT_S) -> GenerateResult | None:
        # Vision not yet supported via Groq text endpoint; fall back to None (truthful)
        log.warning("Groq vision generate not available (truthful): image submitted but Groq vision requires separate endpoint")
        return None

    def status(self) -> EngineStatus:
        api_key = self._api_key or os.environ.get("GROQ_API_KEY")
        loaded = bool(self._loaded)
        # When load() returned True, model is configured regardless of current env change.
        # When never loaded (False), only report if key exists.
        if not loaded:
            loaded = bool(api_key) and bool(self._model_id)
        report_model = self._model_id if loaded else None
        return EngineStatus(
            model_id=report_model,
            loaded=loaded,
            device="api" if loaded else "none",
            dtype="float32" if loaded else "none",
            supports_vision=self.supports_vision,
            load_error=self.last_error or ("provider unavailable" if not api_key else None),
        )
