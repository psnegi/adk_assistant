import logging
import os

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
from personal_assistant.research_agent import research_pipeline

# Load .env from the package directory so the assistant works when run from
# any working directory.
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_PATH, override=False)

# Configure logger
logger = logging.getLogger(__name__)

# ── Environment validation ────────────────────────────────────────────────────
_use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"

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

if not os.getenv("YOUTUBE_API_KEY"):
    logger.warning(
        "YOUTUBE_API_KEY is not set — YouTube search/summary tools will not work. "
        "Get a key at https://console.cloud.google.com/apis/credentials."
    )

# ── Model selection ───────────────────────────────────────────────────────────
# Override via AGENT_MODEL env var.
# Use "gemini-2.0-flash" for free Google AI Studio keys (chat mode).
# Use "gemini-2.0-flash-live-001" for Vertex AI live/streaming mode.
MODEL = os.getenv("AGENT_MODEL", "gemini-2.0-flash")

# ── Agent instructions (plan → execute → verify loop) ────────────────────────
_INSTRUCTION = """
You are a helpful personal assistant with access to Gmail, YouTube, Google Search,
token cost tools, and a dedicated research pipeline. Follow this plan-execute-verify
approach for every request:

1. **Plan** — briefly state which tool(s) or agent you will use and why.
2. **Execute** — call the appropriate tool(s) or delegate to an agent to gather information.
3. **Verify** — check the output for completeness and accuracy.
   - If the result is empty, an error, or clearly incomplete, try an alternative
     approach (e.g., adjust the query, use a different tool) up to two more times.
   - If all attempts fail, clearly explain the error and suggest next steps to the user.
4. **Respond** — present a concise, well-formatted answer based on the verified output.

For in-depth research requests (e.g. "research X", "write a report on Y",
"investigate Z"), delegate to the **research_pipeline** sub-agent, which will
automatically search the web, collect sources, draft a report, and critique it.

Always be transparent about what information you found and where it came from.
If you are unable to complete a request (e.g., missing API keys, no transcript
available), explain the issue and suggest how the user can resolve it.
""".strip()

root_agent = Agent(
    model=MODEL,
    name='root_agent',
    description='A helpful personal assistant for Gmail, YouTube, Search, cost queries, and in-depth research.',
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
    sub_agents=[research_pipeline],
)
