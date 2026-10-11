"""OpenRouter chat-completions engine with server-side credentials."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request

from app.ai.engine import (
    GENERATION_TIMEOUT_S,
    MAX_NEW_TOKENS,
    VERIFY_TIMEOUT_S,
    InferenceEngine,
)
from app.models.schemas import EngineStatus, GenerateResult, GeneratorSource

log = logging.getLogger(__name__)

OPENROUTER_BASE = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
DEFAULT_MODEL_ID = "google/gemma-3-4b-it"


class ProviderEngine(InferenceEngine):
    """Inference through OpenRouter's OpenAI-compatible chat API."""

    supports_vision = True

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model_id = model_id or os.environ.get(
            "OPENROUTER_MODEL_ID", DEFAULT_MODEL_ID
        )
        self.model_id = self._model_id
        self._api_key = api_key if api_key is not None else os.environ.get(
            "OPENROUTER_API_KEY"
        )
        self.last_error: str | None = None
        self._loaded = False

    def load(self) -> bool:
        if not self._api_key:
            self.last_error = "OPENROUTER_API_KEY is not configured"
            return False

        try:
            request = urllib.request.Request(
                f"{OPENROUTER_BASE}/models",
                headers={"Authorization": "Bearer " + self._api_key},
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=10):
                self._loaded = True
                self.last_error = None
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
            self._loaded = False
            self.last_error = f"OpenRouter connectivity check failed: {exc}"
            log.warning("%s", self.last_error)
        return self._loaded

    def _chat(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> GenerateResult | None:
        if not self._api_key:
            log.warning(
                "OpenRouter request skipped: OPENROUTER_API_KEY is not configured"
            )
            return None

        payload = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                f"{OPENROUTER_BASE}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": "Bearer " + self._api_key,
                    "Content-Type": "application/json",
                    "X-Title": "SideQuest AI",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
            text = result["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("OpenRouter returned an empty or non-text completion")
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            self.last_error = f"OpenRouter request failed: {exc}"
            log.warning("%s", self.last_error)
            return None

        self.last_error = None
        return GenerateResult(
            text=text,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_id=self._model_id,
            source=GeneratorSource.MODEL,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.7,
        timeout_s: float = GENERATION_TIMEOUT_S,
    ) -> GenerateResult | None:
        return self._chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.2,
        timeout_s: float = VERIFY_TIMEOUT_S,
    ) -> GenerateResult | None:
        mime_type = "image/jpeg"
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            mime_type = "image/png"
        elif image_bytes.startswith((b"GIF87a", b"GIF89a")):
            mime_type = "image/gif"
        elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            mime_type = "image/webp"
        image = base64.b64encode(image_bytes).decode("ascii")
        return self._chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image}"
                            },
                        },
                    ],
                }
            ],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    def status(self) -> EngineStatus:
        return EngineStatus(
            model_id=self._model_id if self._loaded else None,
            loaded=self._loaded,
            device="api" if self._loaded else "none",
            dtype="n/a" if self._loaded else "none",
            supports_vision=self.supports_vision and self._loaded,
            load_error=self.last_error
            or (
                "OpenRouter connection has not been checked"
                if self._api_key and not self._loaded
                else "OPENROUTER_API_KEY is not configured"
                if not self._api_key
                else None
            ),
        )
