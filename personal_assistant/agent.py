import logging
import os

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

# Configure logger
logger = logging.getLogger(__name__)

# Model selection — override via AGENT_MODEL env var
MODEL = os.getenv("AGENT_MODEL", "gemini-2.0-flash-001")

root_agent = Agent(
    model=MODEL,
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge.',
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
