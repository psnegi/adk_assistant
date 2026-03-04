"""Tests for Ollama configuration helpers in agent.py."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

# Pre-load google.adk so its native extensions are in sys.modules before any
# test-time reimport dance.  When tests run in isolation (i.e. without
# test_research_agent.py being collected first), google.adk is not yet in
# sys.modules.  If _load_agent_module() then patches sys.modules with
# MagicMock entries and those entries are removed when the context manager
# exits, a second import re-initialises the PyO3-compiled cryptography
# extension which raises "only initialized once per interpreter process".
# Importing google.adk here ensures it stays in sys.modules throughout.
import google.adk.agents.llm_agent  # noqa: F401
import google.adk.tools  # noqa: F401


# ---------------------------------------------------------------------------
# Helper: import the agent module without triggering heavy side-effects
# ---------------------------------------------------------------------------

def _load_agent_module(extra_env: dict | None = None):
    """Import personal_assistant.agent with heavy dependencies mocked out.

    Only ``personal_assistant.agent`` is evicted from the module cache on each
    call so that its module-level env-var checks re-execute, while heavy ADK
    and native-extension modules stay cached to avoid PyO3 reinitialization
    errors.

    Returns the freshly imported module so that tests can access its helpers
    directly (e.g. ``mod._ensure_ollama_model``).
    """
    # Evict only the agent and research_agent modules so their module-level
    # env-var checks and sub-agent instantiation re-execute cleanly.
    # Keeping heavy ADK / native-extension modules in sys.modules avoids
    # PyO3 "only initialized once" errors on repeated imports.
    sys.modules.pop("personal_assistant.agent", None)
    sys.modules.pop("personal_assistant.research_agent", None)

    # Ensure tool mocks are present (only set if not already there).
    _TOOL_MOCKS = {
        "personal_assistant.tools.gmail_summary",
        "personal_assistant.tools.token_cost_calculator",
        "personal_assistant.tools.youtube_summary",
    }
    for mod_name in _TOOL_MOCKS:
        sys.modules.setdefault(mod_name, MagicMock())

    env = {
        "USE_OLLAMA": "false",
        "GOOGLE_API_KEY": "test-key",
        "YOUTUBE_API_KEY": "test-key",
        **(extra_env or {}),
    }

    with (
        patch("dotenv.load_dotenv"),
        patch.dict(os.environ, env, clear=True),
    ):
        import personal_assistant.agent as ag

    return ag


# ---------------------------------------------------------------------------
# Tests for _ollama_list_models
# ---------------------------------------------------------------------------

class TestOllamaListModels:
    def test_returns_model_names_on_success(self):
        mod = _load_agent_module()
        response_data = json.dumps({
            "models": [{"name": "llama3.2:latest"}, {"name": "mistral:latest"}]
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = mod._ollama_list_models("http://localhost:11434")

        assert result == ["llama3.2:latest", "mistral:latest"]

    def test_returns_empty_list_on_connection_error(self):
        mod = _load_agent_module()

        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            result = mod._ollama_list_models("http://localhost:11434")

        assert result == []

    def test_returns_empty_list_on_invalid_json(self):
        mod = _load_agent_module()

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not-json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = mod._ollama_list_models("http://localhost:11434")

        assert result == []


# ---------------------------------------------------------------------------
# Tests for _ollama_pull_model
# ---------------------------------------------------------------------------

class TestOllamaPullModel:
    def _make_streaming_response(self, lines: list[dict]):
        """Build a mock HTTP response that iterates newline-delimited JSON."""
        encoded = [json.dumps(l).encode() + b"\n" for l in lines]
        mock_resp = MagicMock()
        mock_resp.__iter__ = MagicMock(return_value=iter(encoded))
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_sends_pull_request_with_correct_payload(self):
        mod = _load_agent_module()
        mock_resp = self._make_streaming_response([{"status": "pulling manifest"}])

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            mod._ollama_pull_model("http://localhost:11434", "llama3.2")

        mock_open.assert_called_once()
        req = mock_open.call_args[0][0]
        assert req.full_url == "http://localhost:11434/api/pull"
        assert json.loads(req.data) == {"name": "llama3.2", "stream": True}

    def test_logs_progress_with_percentage(self):
        mod = _load_agent_module()
        events = [
            {"status": "downloading", "total": 100, "completed": 50},
            {"status": "done"},
        ]
        mock_resp = self._make_streaming_response(events)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            # Should not raise
            mod._ollama_pull_model("http://localhost:11434", "llama3.2")

    def test_handles_connection_error_gracefully(self):
        mod = _load_agent_module()

        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            # Should not raise
            mod._ollama_pull_model("http://localhost:11434", "llama3.2")


# ---------------------------------------------------------------------------
# Tests for _ensure_ollama_model
# ---------------------------------------------------------------------------

class TestEnsureOllamaModel:
    def test_does_not_pull_when_model_already_available(self):
        mod = _load_agent_module()

        with (
            patch.object(mod, "_ollama_list_models", return_value=["llama3.2:latest"]),
            patch.object(mod, "_ollama_pull_model") as mock_pull,
        ):
            mod._ensure_ollama_model("http://localhost:11434", "llama3.2")
            mock_pull.assert_not_called()

    def test_pulls_when_model_not_available(self):
        mod = _load_agent_module()

        with (
            patch.object(mod, "_ollama_list_models", return_value=["mistral:latest"]),
            patch.object(mod, "_ollama_pull_model") as mock_pull,
        ):
            mod._ensure_ollama_model("http://localhost:11434", "llama3.2")
            mock_pull.assert_called_once_with("http://localhost:11434", "llama3.2")

    def test_pulls_when_no_models_available(self):
        mod = _load_agent_module()

        with (
            patch.object(mod, "_ollama_list_models", return_value=[]),
            patch.object(mod, "_ollama_pull_model") as mock_pull,
        ):
            mod._ensure_ollama_model("http://localhost:11434", "llama3.2")
            mock_pull.assert_called_once_with("http://localhost:11434", "llama3.2")


# ---------------------------------------------------------------------------
# Tests for voice / TTS configuration
# ---------------------------------------------------------------------------

class TestVoiceConfig:
    """Verify TTS generate_content_config is set for live models only."""

    def test_no_voice_config_for_default_model(self):
        mod = _load_agent_module({"AGENT_MODEL": "gemini-2.0-flash"})
        assert mod._GENERATE_CONTENT_CONFIG is None

    def test_voice_config_set_for_live_model(self):
        from google.genai import types as genai_types

        mod = _load_agent_module({"AGENT_MODEL": "gemini-2.0-flash-live-001"})
        cfg = mod._GENERATE_CONTENT_CONFIG

        assert cfg is not None
        assert isinstance(cfg, genai_types.GenerateContentConfig)
        assert "AUDIO" in cfg.response_modalities
        assert cfg.speech_config is not None

    def test_no_voice_config_for_ollama(self):
        mod = _load_agent_module({"USE_OLLAMA": "true", "OLLAMA_MODEL": "llama3.2"})
        assert mod._GENERATE_CONTENT_CONFIG is None
