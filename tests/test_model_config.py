"""Tests for the shared model configuration helper."""

from __future__ import annotations

import os


class TestBuildModelGemini:
    """build_model() returns a plain string."""

    def test_returns_default_gemini_model(self, monkeypatch):
        monkeypatch.delenv("AGENT_MODEL", raising=False)

        from personal_assistant.model_config import build_model

        result = build_model()
        assert result == "gemini-2.0-flash"

    def test_honours_agent_model_env_var(self, monkeypatch):
        monkeypatch.setenv("AGENT_MODEL", "gemini-1.5-pro")

        from personal_assistant.model_config import build_model

        result = build_model()
        assert result == "gemini-1.5-pro"

    def test_returns_string(self, monkeypatch):
        monkeypatch.setenv("AGENT_MODEL", "gemini-2.0-flash-001")

        from personal_assistant.model_config import build_model

        result = build_model()
        assert isinstance(result, str)
        assert result == "gemini-2.0-flash-001"
