"""Shared model configuration for all agents.

Supports two backends, controlled by environment variables:

* **Gemini** (default) — Google AI Studio or Vertex AI.
  Set ``AGENT_MODEL`` to the desired Gemini model name.

* **Ollama** (local) — a locally hosted Ollama server.
  Set ``USE_OLLAMA=true`` and optionally ``OLLAMA_BASE_URL`` /
  ``OLLAMA_MODEL`` to point at your server.

Usage::

    from personal_assistant.model_config import build_model
    model = build_model()   # returns str or LiteLlm instance
"""

from __future__ import annotations

import os


def build_model():
    """Return a model for use in ADK agents.

    Returns:
        A ``LiteLlm`` instance when ``USE_OLLAMA=true``, otherwise the
        ``AGENT_MODEL`` string (defaults to ``"gemini-2.0-flash"``).
    """
    if os.getenv("USE_OLLAMA", "false").lower() == "true":
        # Import lazily so that the heavy LiteLLM dependency is only loaded
        # when Ollama support is actually requested.
        from google.adk.models.lite_llm import LiteLlm  # noqa: PLC0415

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        return LiteLlm(
            model=f"ollama/{ollama_model}",
            api_base=ollama_base_url,
        )

    return os.getenv("AGENT_MODEL", "gemini-2.0-flash")
