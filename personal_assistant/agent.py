import logging
import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search
from personal_assistant.model_config import build_model
from personal_assistant.research_agent import research_pipeline
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
from personal_assistant.tools.memory_manager import (
    update_memory_tool,
    read_memory_tool,
    list_memory_sections_tool,
    clear_memory_section_tool,
)
from personal_assistant.tools.file_search import (
    search_files_tool,
    search_file_content_tool,
)

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
# Override via AGENT_MODEL env var to select a different Gemini model.
MODEL = build_model()

# ── Agent instructions (plan → execute → verify loop) ────────────────────────
_INSTRUCTION = """
You are a helpful, friendly personal assistant with a consistent personality:
you are curious, concise, and honest. You remember details about the user
across conversations using your local memory tools, and you can search the
user's local file system when asked.

Follow this plan-execute-verify approach for every request:

1. **Plan** — briefly state which tool(s) you will use and why.
2. **Execute** — call the appropriate tool(s) to gather information.
3. **Verify** — check the tool output for completeness and accuracy.
   - If the result is empty, an error, or clearly incomplete, try an
     alternative approach up to two more times.
   - If all attempts fail, clearly explain the error and suggest next steps.
4. **Respond** — present a concise, well-formatted answer based on the
   verified output.

## Memory guidelines
- When the user shares preferences, facts, ongoing tasks, or anything worth
  remembering, proactively call **update_memory** to persist it.
- At the start of a conversation, call **read_memory** (no arguments) to
  recall relevant context about the user.
- Organize memory into meaningful sections such as "Preferences", "Ongoing
  Tasks", "Skills", "People", "Notes", etc.
- When asked "what do you remember?", call **list_memory_sections** then
  **read_memory** for the relevant section(s).

## File search guidelines
- Use **search_files** to locate files by partial name or glob pattern,
  optionally filtered by modification time.
- Use **search_file_content** to search text inside files under a directory,
  optionally filtered by file name and modification time.
- Times must be in ISO-8601 format, e.g. ``"2024-01-15"`` or
  ``"2024-01-15T09:00:00"``.

Always be transparent about what information you found and where it came from.
If you are unable to complete a request (e.g., missing API keys, no transcript
available), explain the issue and suggest how the user can resolve it.
""".strip()

root_agent = Agent(
    model=MODEL,
    name='root_agent',
    description='A helpful personal assistant for Gmail, YouTube, Search, cost queries, local memory, and file search.',
    instruction=_INSTRUCTION,
    tools=[
        google_search, 
        gmail_summary_tool, 
        get_email_content_tool,
        token_cost_calculator_tool, 
        batch_cost_estimator_tool,
        search_youtube_tool,
        youtube_summary_tool,
        check_transcripts_tool,
        update_memory_tool,
        read_memory_tool,
        list_memory_sections_tool,
        clear_memory_section_tool,
        search_files_tool,
        search_file_content_tool,
    ],
    sub_agents=[research_pipeline],
)
