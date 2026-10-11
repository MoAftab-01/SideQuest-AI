from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from app.ai.engine import InferenceEngine, UnavailableEngine, build_engine
from app.ai.provider import DEFAULT_MODEL_ID, ProviderEngine


class ProviderTests(unittest.TestCase):
    def _response(self, payload: dict) -> MagicMock:
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode("utf-8")
        return response

    def test_text_generation_sends_bearer_key_and_model(self) -> None:
        provider = ProviderEngine(api_key="test-key")
        response = self._response(
            {"choices": [{"message": {"content": '{"title":"A mission"}'}}]}
        )
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = response
            result = provider.generate("make a mission")

        self.assertIsNotNone(result)
        self.assertEqual(result.text, '{"title":"A mission"}')
        self.assertEqual(result.model_id, DEFAULT_MODEL_ID)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(
            json.loads(request.data)["model"],
            DEFAULT_MODEL_ID,
        )
        self.assertEqual(json.loads(request.data)["max_tokens"], 2048)
        self.assertEqual(
            json.loads(request.data)["response_format"],
            {"type": "json_object"},
        )

    def test_vision_generation_sends_image_as_data_url(self) -> None:
        provider = ProviderEngine(api_key="test-key")
        response = self._response(
            {"choices": [{"message": {"content": '{"verdict":"uncertain"}'}}]}
        )
        image = b"\x89PNG\r\n\x1a\nimage bytes"
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = response
            result = provider.generate_with_image("check this", image)

        self.assertIsNotNone(result)
        request = urlopen.call_args.args[0]
        content = json.loads(request.data)["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "check this"})
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,iVBORw0KGgppbWFnZSBieXRlcw==")

    def test_request_failure_returns_none_and_is_logged(self) -> None:
        provider = ProviderEngine(api_key="test-key")
        with patch("urllib.request.urlopen", side_effect=URLError("offline")):
            with self.assertLogs("app.ai.provider", level="WARNING"):
                result = provider.generate("make a mission")

        self.assertIsNone(result)
        self.assertIn("offline", provider.last_error)

    def test_missing_key_uses_deterministic_fallback_without_network(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            with patch("urllib.request.urlopen") as urlopen:
                engine = build_engine()

        self.assertIsInstance(engine, UnavailableEngine)
        self.assertIsInstance(engine, InferenceEngine)
        self.assertFalse(engine.status().loaded)
        urlopen.assert_not_called()

    def test_load_checks_openrouter_with_bearer_key(self) -> None:
        provider = ProviderEngine(api_key="test-key")
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = MagicMock()
            self.assertTrue(provider.load())

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(provider.status().device, "api")

    def test_build_engine_selects_openrouter_when_available(self) -> None:
        response = MagicMock()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("urllib.request.urlopen") as urlopen:
                urlopen.return_value.__enter__.return_value = response
                engine = build_engine()

        self.assertIsInstance(engine, ProviderEngine)
        self.assertTrue(engine.status().loaded)


if __name__ == "__main__":
    unittest.main()
