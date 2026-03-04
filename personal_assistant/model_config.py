"""Shared model configuration for all agents.

Supports two backends, controlled by environment variables:

* **Gemini** (default) — Google AI Studio or Vertex AI.
  Set ``AGENT_MODEL`` to the desired Gemini model name.

* **Ollama** (local) — a locally hosted Ollama server.
  Set ``USE_OLLAMA=true`` and optionally ``OLLAMA_BASE_URL`` /
  ``OLLAMA_MODEL`` to point at your server.
  Note: Ollama models are text-only and do not support live TTS/STT.

Usage::

    from personal_assistant.model_config import build_model, is_voice_model
    model = build_model()           # returns str or LiteLlm instance
    voice = is_voice_model()        # True only for Gemini Live models
"""

from __future__ import annotations

import os

# Model name prefixes that support the Gemini Live API (native TTS/STT).
# Keep this list in sync with Google's Live API model releases:
# https://cloud.google.com/vertex-ai/generative-ai/docs/live-api
_LIVE_MODEL_PREFIXES = ("gemini-2.0-flash-live", "gemini-live")


def build_model():
    """Return a model for use in ADK agents.

    Returns:
        A ``LiteLlm`` instance when ``USE_OLLAMA=true``, otherwise the
        ``AGENT_MODEL`` string (defaults to ``"gemini-2.0-flash"``).
    """
    if os.getenv("USE_OLLAMA", "false").lower() == "true":
        # Import lazily so that the heavy LiteLLM dependency is only loaded
        # when Ollama support is actually requested.
        # ADK requires a model object that implements its interface; using
        # LiteLLM as the adapter is the correct approach — calling the Ollama
        # REST API directly would bypass ADK's agent framework.
        from google.adk.models.lite_llm import LiteLlm  # noqa: PLC0415

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        return LiteLlm(
            model=f"ollama/{ollama_model}",
            api_base=ollama_base_url,
        )

    return os.getenv("AGENT_MODEL", "gemini-2.0-flash")


def is_voice_model() -> bool:
    """Return ``True`` if the configured model supports live voice interaction.

    Only Gemini Live models (e.g. ``gemini-2.0-flash-live-001``) natively
    support TTS/STT through ADK's ``generate_content_config``.  Ollama models
    and standard Gemini chat models are text-only and return ``False``.

    Returns:
        ``True`` when ``USE_OLLAMA=false`` and ``AGENT_MODEL`` starts with a
        known Live API prefix; ``False`` otherwise.
    """
    if os.getenv("USE_OLLAMA", "false").lower() == "true":
        return False
    model = os.getenv("AGENT_MODEL", "gemini-2.0-flash")
    return any(model.startswith(prefix) for prefix in _LIVE_MODEL_PREFIXES)
