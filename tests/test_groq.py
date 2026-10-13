"""Tests for Groq provider — mocked responses; never hits live API unless explicitly configured."""
from __future__ import annotations

import os
import json

# Mock urllib before importing provider modules
from unittest.mock import patch, MagicMock

import pytest
from app.ai.groq import GroqEngine, DEFAULT_GROQ_MODEL


class MockResponse:
    def __init__(self, body_bytes, status=200):
        self.body = body_bytes
        self.status = status

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


MOCK_GROQ_MODEL = "llama-3.1-8b-instant"


def mock_urlopen_success(req, *args, **kw):
    method = req.get_method() if hasattr(req, "get_method") else (req.method if hasattr(req, "method") else "GET")
    if method == "GET" or (req.full_url if hasattr(req, "full_url") else "").endswith("/models") or not req.data:
        # /models connectivity check (GET, no body) — just return empty OK
        return MockResponse(b'{"data":[{"id":"test"}]}')
    # POST chat/completions — verify payload and return mission
    payload = json.loads(req.data) if req.data else {}
    assert payload.get("model") == MOCK_GROQ_MODEL, f"expected model {MOCK_GROQ_MODEL}, got {payload.get('model')}"
    assert "messages" in payload, "missing messages in payload"
    body = json.dumps({"choices": [{"message": {"content": "Walk to a tree, take a photo, come back. 15 minutes."}}]}).encode()
    return MockResponse(body)


def test_groq_init_reads_env():
    """Engine reads GROQ_API_KEY and GROQ_MODEL from env."""
    os.environ["GROQ_API_KEY"] = "test-key-redacted"
    os.environ["GROQ_MODEL"] = MOCK_GROQ_MODEL
    engine = GroqEngine()
    assert engine._api_key == "test-key-redacted"
    assert engine.model_id == MOCK_GROQ_MODEL
    # Never expose key value
    assert engine._api_key is not None


def test_groq_load_success():
    os.environ["GROQ_API_KEY"] = "test-key-redacted"
    os.environ["GROQ_MODEL"] = MOCK_GROQ_MODEL
    engine = GroqEngine()
    with patch("urllib.request.urlopen", mock_urlopen_success):
        result = engine.load()
    assert result is True
    assert engine._loaded is True


def test_groq_generate_success():
    os.environ["GROQ_API_KEY"] = "test-key-redacted"
    os.environ["GROQ_MODEL"] = MOCK_GROQ_MODEL
    engine = GroqEngine()
    with patch("urllib.request.urlopen", mock_urlopen_success):
        with patch.object(engine, "load", return_value=True):
            res = engine.generate("find a flower near water", max_new_tokens=256)
    assert res is not None
    assert res.source.value == "model"
    assert "Walk to a tree" in res.text
    assert res.model_id == MOCK_GROQ_MODEL


def test_groq_generate_no_key():
    # Ensure no key -> truthful failure (None returned), not fabricated success
    if "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
    engine = GroqEngine()
    res = engine.generate("any prompt")
    assert res is None  # truthful: no key means no generation


def test_groq_missing_key_load_false():
    if "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
    engine = GroqEngine()
    assert engine.load() is False
    assert "missing" in (engine.last_error or "").lower()


def test_groq_vision_not_available_truthful():
    """Vision returns None (truthful) rather than claiming vision works."""
    os.environ["GROQ_API_KEY"] = "test-key-redacted"
    os.environ["GROQ_MODEL"] = MOCK_GROQ_MODEL
    engine = GroqEngine()
    with patch.object(engine, "load", return_value=True):
        res = engine.generate_with_image("describe this", b"fake-image-bytes")
    assert res is None  # truthful — vision path not supported


def test_groq_status_truthful():
    os.environ["GROQ_API_KEY"] = "test-key-redacted"
    os.environ["GROQ_MODEL"] = MOCK_GROQ_MODEL
    engine = GroqEngine()
    with patch.object(engine, "load", return_value=True):
        s = engine.status()
    assert s.model_id == MOCK_GROQ_MODEL
    assert s.loaded is True
    assert s.device == "api"
    # Health endpoint never claims vision unless supported; vision is False for text-only Groq
    assert s.supports_vision is False
