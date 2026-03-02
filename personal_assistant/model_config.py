"""Shared model configuration for all agents.

Supports Gemini models via Google AI Studio or Vertex AI.
Set ``AGENT_MODEL`` to the desired Gemini model name.

Usage::

    from personal_assistant.model_config import build_model
    model = build_model()   # returns the AGENT_MODEL string
"""

from __future__ import annotations

import os


def build_model():
    """Return a model string for use in ADK agents.

    Returns:
        The ``AGENT_MODEL`` environment variable value, defaulting to
        ``"gemini-2.0-flash"``.
    """
    return os.getenv("AGENT_MODEL", "gemini-2.0-flash")
