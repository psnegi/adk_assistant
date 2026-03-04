"""Tests for the shared model configuration helper."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


class TestBuildModelGemini:
    """build_model() returns a plain string when Ollama is disabled."""

    def test_returns_default_gemini_model(self, monkeypatch):
        monkeypatch.delenv("USE_OLLAMA", raising=False)
        monkeypatch.delenv("AGENT_MODEL", raising=False)

        from personal_assistant.model_config import build_model

        result = build_model()
        assert result == "gemini-2.0-flash"

    def test_honours_agent_model_env_var(self, monkeypatch):
        monkeypatch.setenv("AGENT_MODEL", "gemini-1.5-pro")
        monkeypatch.setenv("USE_OLLAMA", "false")

        from personal_assistant.model_config import build_model

        result = build_model()
        assert result == "gemini-1.5-pro"

    def test_use_ollama_false_returns_string(self, monkeypatch):
        monkeypatch.setenv("USE_OLLAMA", "false")
        monkeypatch.setenv("AGENT_MODEL", "gemini-2.0-flash-001")

        from personal_assistant.model_config import build_model

        result = build_model()
        assert isinstance(result, str)
        assert result == "gemini-2.0-flash-001"


class TestBuildModelOllama:
    """build_model() returns a LiteLlm instance when USE_OLLAMA=true."""

    def _mock_litellm(self, monkeypatch):
        """Patch LiteLlm so we don't need the heavy dependency to be active."""
        mock_cls = MagicMock(name="LiteLlm")
        mock_instance = MagicMock(name="LiteLlmInstance")
        mock_cls.return_value = mock_instance

        monkeypatch.setitem(
            __import__("sys").modules,
            "google.adk.models.lite_llm",
            MagicMock(LiteLlm=mock_cls),
        )
        return mock_cls, mock_instance

    def test_returns_litellm_instance_when_ollama_enabled(self, monkeypatch):
        monkeypatch.setenv("USE_OLLAMA", "true")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

        mock_cls, mock_instance = self._mock_litellm(monkeypatch)

        # Reload module to pick up patched sys.modules
        import importlib
        import personal_assistant.model_config as mc
        importlib.reload(mc)

        result = mc.build_model()
        mock_cls.assert_called_once_with(
            model="ollama/llama3.2",
            api_base="http://localhost:11434",
        )
        assert result is mock_instance

    def test_default_ollama_values(self, monkeypatch):
        monkeypatch.setenv("USE_OLLAMA", "true")
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        mock_cls, _ = self._mock_litellm(monkeypatch)

        import importlib
        import personal_assistant.model_config as mc
        importlib.reload(mc)

        mc.build_model()
        mock_cls.assert_called_once_with(
            model="ollama/llama3.2",
            api_base="http://localhost:11434",
        )

    def test_custom_ollama_model_and_url(self, monkeypatch):
        monkeypatch.setenv("USE_OLLAMA", "true")
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.10:11434")

        mock_cls, _ = self._mock_litellm(monkeypatch)

        import importlib
        import personal_assistant.model_config as mc
        importlib.reload(mc)

        mc.build_model()
        mock_cls.assert_called_once_with(
            model="ollama/mistral",
            api_base="http://192.168.1.10:11434",
        )


class TestIsVoiceModel:
    """is_voice_model() returns True only for Gemini Live model names."""

    def test_returns_false_for_default_gemini_model(self, monkeypatch):
        monkeypatch.setenv("USE_OLLAMA", "false")
        monkeypatch.setenv("AGENT_MODEL", "gemini-2.0-flash")

        from personal_assistant.model_config import is_voice_model
        assert is_voice_model() is False

    def test_returns_true_for_live_gemini_model(self, monkeypatch):
        monkeypatch.setenv("USE_OLLAMA", "false")
        monkeypatch.setenv("AGENT_MODEL", "gemini-2.0-flash-live-001")

        from personal_assistant.model_config import is_voice_model
        assert is_voice_model() is True

    def test_returns_false_for_ollama_even_with_live_in_name(self, monkeypatch):
        monkeypatch.setenv("USE_OLLAMA", "true")
        monkeypatch.setenv("AGENT_MODEL", "gemini-2.0-flash-live-001")

        from personal_assistant.model_config import is_voice_model
        assert is_voice_model() is False

    def test_returns_false_when_agent_model_unset(self, monkeypatch):
        monkeypatch.setenv("USE_OLLAMA", "false")
        monkeypatch.delenv("AGENT_MODEL", raising=False)

        from personal_assistant.model_config import is_voice_model
        assert is_voice_model() is False
