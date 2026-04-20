"""
Tests for the Ollama client layer.

Uses mocking so tests pass without a running Ollama instance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock
import pytest
import httpx

from aiviz.ai.ollama_client import OllamaClient, OllamaResponse


class TestOllamaClient:
    def setup_method(self):
        self.client = OllamaClient(
            base_url="http://localhost:11434",
            default_model="test-model",
            timeout=5,
        )

    def test_generate_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "Hello world!", "done": True}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            result = self.client.generate("Hello?")

        assert result.ok
        assert result.text == "Hello world!"
        assert result.model == "test-model"

    def test_generate_connection_error(self):
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            result = self.client.generate("Hello?")

        assert not result.ok
        assert "Cannot connect" in result.error or "not" in result.error.lower()

    def test_generate_timeout(self):
        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            result = self.client.generate("Hello?")

        assert not result.ok
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    def test_is_healthy_true(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            assert self.client.is_healthy() is True

    def test_is_healthy_false_on_connection_error(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert self.client.is_healthy() is False

    def test_list_models_returns_names(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.2"}, {"name": "llava"}]
        }
        with patch("httpx.get", return_value=mock_resp):
            models = self.client.list_models()
        assert "llama3.2" in models
        assert "llava" in models

    def test_list_models_empty_on_error(self):
        with patch("httpx.get", side_effect=Exception("network error")):
            models = self.client.list_models()
        assert models == []

    def test_model_override(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": "ok", "done": True}

        captured = {}
        def fake_post(url, json=None, timeout=None):
            captured["model"] = json.get("model")
            return mock_resp

        with patch("httpx.post", side_effect=fake_post):
            self.client.generate("Hi", model="custom-model")

        assert captured["model"] == "custom-model"

    def test_ollama_response_ok_property(self):
        good = OllamaResponse(text="hi", model="m", done=True)
        bad = OllamaResponse(text="", model="m", done=False, error="oops")
        assert good.ok is True
        assert bad.ok is False
