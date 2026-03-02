"""Tests for Ollama configuration helpers in agent.py."""

import json
import os
import sys
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: import the agent module without triggering heavy side-effects
# ---------------------------------------------------------------------------

def _load_agent_module():
    """Import personal_assistant.agent with heavy dependencies mocked out.

    Returns the freshly imported module so that tests can access its helpers
    directly (e.g. ``mod._ensure_ollama_model``).
    """
    # Drop any cached copy so each test-class gets an isolated import
    for key in list(sys.modules):
        if key.startswith("personal_assistant"):
            del sys.modules[key]

    with (
        patch.dict("sys.modules", {
            "google.adk.agents.llm_agent": MagicMock(),
            "google.adk.tools": MagicMock(),
            "personal_assistant.tools.gmail_summary": MagicMock(),
            "personal_assistant.tools.token_cost_calculator": MagicMock(),
            "personal_assistant.tools.youtube_summary": MagicMock(),
        }),
        patch("dotenv.load_dotenv"),
        patch.dict(os.environ, {
            "USE_OLLAMA": "false",
            "GOOGLE_API_KEY": "test-key",
            "YOUTUBE_API_KEY": "test-key",
        }, clear=True),
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
