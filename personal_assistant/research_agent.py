"""Research agent implemented as a full state machine pipeline.

Stages:
  1. search_agent   – queries Google Search and records source URLs.
  2. collect_agent  – fetches full page content for each discovered URL.
  3. write_critique_loop (LoopAgent):
       a. write_agent    – drafts (or revises) the research report.
       b. critique_agent – evaluates quality; escalates to stop the loop
                          when the draft is satisfactory.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import google_search

from personal_assistant.model_config import build_model
from personal_assistant.tools.web_research import fetch_web_page_tool

# ---------------------------------------------------------------------------
# Model selection – honours the same env var as the root agent
# ---------------------------------------------------------------------------
_MODEL = build_model()

# ---------------------------------------------------------------------------
# Stage 1 – Search
# ---------------------------------------------------------------------------
_SEARCH_INSTRUCTION = """
You are the Search Agent in a research pipeline.

Your ONLY task: use the google_search tool to find high-quality, relevant
sources for the research topic provided by the user.

Steps:
1. Run 2-3 targeted google_search queries to cover the topic broadly.
2. Collect the URLs returned by each search.
3. Output a numbered list of the most relevant URLs (up to 10) together with
   a one-sentence description of each source.  Format:

   SEARCH_RESULTS:
   1. <url> — <description>
   2. <url> — <description>
   ...

Do NOT write any research yet.  Only find sources.
""".strip()

search_agent = LlmAgent(
    model=_MODEL,
    name="search_agent",
    description="Searches the web for relevant sources on the research topic.",
    instruction=_SEARCH_INSTRUCTION,
    tools=[google_search],
    output_key="search_results",
)

# ---------------------------------------------------------------------------
# Stage 2 – Collect
# ---------------------------------------------------------------------------
_COLLECT_INSTRUCTION = """
You are the Collect Agent in a research pipeline.

The previous Search Agent has stored source URLs in the session state under
the key "search_results".  Those results are provided to you as context.

Your ONLY task: use the fetch_web_page tool to retrieve the full text of each
URL listed in "search_results".  Fetch up to 5 of the most relevant URLs.

After fetching, output a consolidated block of the extracted content labelled
by source URL so the Write Agent can use it.  Format:

   COLLECTED_CONTENT:
   --- [<url>] ---
   <page text>
   ...

Do NOT write the research report yet.  Only collect content.
""".strip()

collect_agent = LlmAgent(
    model=_MODEL,
    name="collect_agent",
    description="Fetches full content from the URLs found by the Search Agent.",
    instruction=_COLLECT_INSTRUCTION,
    tools=[fetch_web_page_tool],
    output_key="collected_content",
)

# ---------------------------------------------------------------------------
# Stage 3a – Write / Revise
# ---------------------------------------------------------------------------
_WRITE_INSTRUCTION = """
You are the Write Agent in a research pipeline.

You have access to:
- "search_results"     – URLs and descriptions found by the Search Agent.
- "collected_content"  – Full text fetched from those URLs.
- "critique_feedback"  – (optional) Critique from a previous iteration.
                         If present, revise the draft to address the feedback.

Your task:
1. If "critique_feedback" is absent or empty, write an initial research report.
2. If "critique_feedback" is present, revise the existing draft to address
   every point raised.

The report must:
- Have a clear title and introduction.
- Contain well-organised sections with headings.
- Cite sources inline (e.g. [Source: <url>]).
- End with a "References" section listing all sources used.
- Be comprehensive but concise (aim for 500-800 words).

Store your output as the research draft.
""".strip()

write_agent = LlmAgent(
    model=_MODEL,
    name="write_agent",
    description="Writes or revises the research report draft.",
    instruction=_WRITE_INSTRUCTION,
    output_key="research_draft",
)

# ---------------------------------------------------------------------------
# Stage 3b – Critique
# ---------------------------------------------------------------------------
_CRITIQUE_INSTRUCTION = """
You are the Critique Agent in a research pipeline.

Review the research draft stored in "research_draft".

Evaluate it against these criteria:
1. Accuracy – Does it correctly reflect the collected sources?
2. Completeness – Does it cover the main aspects of the topic?
3. Structure – Is it well-organised with clear headings and citations?
4. Clarity – Is it easy to read and understand?
5. References – Are all sources cited and listed?

Decision rules:
- If the draft satisfies ALL criteria, respond with exactly:
    APPROVED
  followed by a short summary of why it is good.
  Then use the `escalate` action to signal the loop to stop.

- If the draft needs improvement, respond with:
    NEEDS_REVISION
  followed by a numbered list of specific, actionable improvements.

Store your decision in "critique_feedback".
""".strip()

critique_agent = LlmAgent(
    model=_MODEL,
    name="critique_agent",
    description="Critiques the research draft and either approves it or requests revision.",
    instruction=_CRITIQUE_INSTRUCTION,
    output_key="critique_feedback",
)

# ---------------------------------------------------------------------------
# Write-Critique loop (max 3 iterations to prevent infinite loops)
# ---------------------------------------------------------------------------
write_critique_loop = LoopAgent(
    name="write_critique_loop",
    sub_agents=[write_agent, critique_agent],
    max_iterations=3,
)

# ---------------------------------------------------------------------------
# Full research pipeline
# ---------------------------------------------------------------------------
research_pipeline = SequentialAgent(
    name="research_pipeline",
    description=(
        "Full state machine research pipeline: searches the web, collects "
        "source content, writes a report, and iteratively critiques and "
        "revises it until it meets quality standards."
    ),
    sub_agents=[search_agent, collect_agent, write_critique_loop],
)
