import json
import logging
import os
import urllib.request

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search
from personal_assistant.tools.gmail_summary import gmail_summary_tool, get_email_content_tool
from personal_assistant.tools.token_cost_calculator import (
    token_cost_calculator_tool,
    batch_cost_estimator_tool,
)
from personal_assistant.tools.youtube_summary import (
    search_youtube_tool, 
    youtube_summary_tool,
    check_transcripts_tool
)

# Load .env from the package directory so the assistant works when run from
# any working directory.
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_PATH, override=False)

# Configure logger
logger = logging.getLogger(__name__)


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _ollama_list_models(base_url: str) -> list[str]:
    """Return names of models already available in the local Ollama instance."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not reach Ollama at %s: %s. "
            "Ensure Ollama is running (`ollama serve`) and OLLAMA_BASE_URL is correct.",
            base_url,
            exc,
        )
        return []


def _ollama_pull_model(base_url: str, model: str) -> None:
    """Stream-pull *model* from the Ollama registry, logging progress."""
    logger.info("Pulling Ollama model '%s' — this may take a few minutes …", model)
    payload = json.dumps({"name": model, "stream": True}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                status = event.get("status", "")
                if "total" in event and "completed" in event:
                    pct = int(event["completed"] / event["total"] * 100)
                    logger.info("[ollama pull] %s — %d%%", status, pct)
                else:
                    logger.info("[ollama pull] %s", status)
        logger.info("Ollama model '%s' is ready.", model)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to pull Ollama model '%s': %s. "
            "Check your internet connection and verify the model name at https://ollama.com/library. "
            "You can also pull it manually with: ollama pull %s",
            model,
            exc,
            model,
        )
        raise


def _ensure_ollama_model(base_url: str, model: str) -> None:
    """Pull *model* from Ollama if it is not already available locally."""
    available = _ollama_list_models(base_url)
    # Ollama tag names may include ":latest" suffix; match on base name too
    if not any(m == model or m.startswith(f"{model}:") for m in available):
        logger.info(
            "Ollama model '%s' not found locally (available: %s). Pulling now …",
            model,
            available or "none",
        )
        _ollama_pull_model(base_url, model)
    else:
        logger.info("Ollama model '%s' is already available locally.", model)


# ── Environment validation ────────────────────────────────────────────────────
_use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
_use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

if _use_ollama:
    _ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    _ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    # LiteLLM (used by ADK for non-Gemini models) reads OLLAMA_API_BASE to
    # locate a non-default Ollama server.
    os.environ.setdefault("OLLAMA_API_BASE", _ollama_base_url)
    _ensure_ollama_model(_ollama_base_url, _ollama_model)
    MODEL = f"ollama/{_ollama_model}"
    logger.info("Using local Ollama model: %s (server: %s)", _ollama_model, _ollama_base_url)
else:
    if _use_vertex:
        _missing = [v for v in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION") if not os.getenv(v)]
        if _missing:
            logger.warning(
                "Vertex AI mode enabled but the following env vars are not set: %s. "
                "Set them in personal_assistant/.env or export them before running.",
                ", ".join(_missing),
            )
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning(
                "GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey and add it to "
                "personal_assistant/.env as GOOGLE_API_KEY=<your-key>."
            )

    # ── Model selection (Google / Vertex AI) ─────────────────────────────────
    # Override via AGENT_MODEL env var.
    # Use "gemini-2.0-flash" for free Google AI Studio keys (chat mode).
    # Use "gemini-2.0-flash-live-001" for Vertex AI live/streaming mode.
    MODEL = os.getenv("AGENT_MODEL", "gemini-2.0-flash")

if not os.getenv("YOUTUBE_API_KEY"):
    logger.warning(
        "YOUTUBE_API_KEY is not set — YouTube search/summary tools will not work. "
        "Get a key at https://console.cloud.google.com/apis/credentials."
    )

# ── Agent instructions (plan → execute → verify loop) ────────────────────────
_INSTRUCTION = """
You are a helpful personal assistant with access to Gmail, YouTube, Google Search,
and token cost tools. Follow this plan-execute-verify approach for every request:

1. **Plan** — briefly state which tool(s) you will use and why.
2. **Execute** — call the appropriate tool(s) to gather information.
3. **Verify** — check the tool output for completeness and accuracy.
   - If the result is empty, an error, or clearly incomplete, try an alternative
     approach (e.g., adjust the query, use a different tool) up to two more times.
   - If all attempts fail, clearly explain the error and suggest next steps to the user.
4. **Respond** — present a concise, well-formatted answer based on the verified output.

Always be transparent about what information you found and where it came from.
If you are unable to complete a request (e.g., missing API keys, no transcript
available), explain the issue and suggest how the user can resolve it.

**Voice / streaming interaction guidelines:**
- When responding to spoken input, keep each sentence short and self-contained so
  that the text-to-speech engine can begin speaking before the full response is ready.
- Avoid long preambles; deliver the most important information first.
- Use natural spoken language rather than markdown formatting in voice responses.
""".strip()

root_agent = Agent(
    model=MODEL,
    name='root_agent',
    description='A helpful personal assistant for Gmail, YouTube, Search, and cost queries.',
    instruction=_INSTRUCTION,
    tools=[
        google_search, 
        gmail_summary_tool, 
        get_email_content_tool,
        token_cost_calculator_tool, 
        batch_cost_estimator_tool,
        search_youtube_tool,
        youtube_summary_tool,
        check_transcripts_tool
    ],
)
