"""Tests for the web research tool and research agent pipeline structure."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from personal_assistant.tools.web_research import fetch_web_page


class TestFetchWebPage:
    def test_invalid_scheme_returns_error(self):
        result = fetch_web_page("ftp://example.com/file.txt")
        assert "Invalid URL scheme" in result

    def test_timeout_returns_error(self):
        import requests

        with patch("personal_assistant.tools.web_research.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout()
            result = fetch_web_page("https://example.com")
        assert "timed out" in result.lower()

    def test_http_error_returns_error(self):
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch("personal_assistant.tools.web_research.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.HTTPError(
                response=mock_response
            )
            result = fetch_web_page("https://example.com/notfound")
        assert "HTTP error" in result

    def test_request_exception_returns_error(self):
        import requests

        with patch("personal_assistant.tools.web_research.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("refused")
            result = fetch_web_page("https://example.com")
        assert "Error fetching" in result

    def test_empty_page_returns_no_content_message(self):
        mock_response = MagicMock()
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status = MagicMock()
        with patch("personal_assistant.tools.web_research.requests.get", return_value=mock_response):
            result = fetch_web_page("https://example.com")
        assert "No readable text content found" in result

    def test_successful_fetch_returns_content(self):
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello world</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        with patch("personal_assistant.tools.web_research.requests.get", return_value=mock_response):
            result = fetch_web_page("https://example.com")
        assert "Hello world" in result
        assert "[Source: https://example.com]" in result

    def test_long_content_is_truncated(self):
        long_text = "word " * 10000
        mock_response = MagicMock()
        mock_response.text = f"<html><body><p>{long_text}</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        with patch("personal_assistant.tools.web_research.requests.get", return_value=mock_response):
            result = fetch_web_page("https://example.com")
        assert "truncated" in result


class TestResearchAgentStructure:
    """Verify the research pipeline is wired correctly without making LLM calls."""

    def test_research_pipeline_is_sequential_agent(self):
        from google.adk.agents import SequentialAgent
        from personal_assistant.research_agent import research_pipeline

        assert isinstance(research_pipeline, SequentialAgent)

    def test_research_pipeline_has_three_sub_agents(self):
        from personal_assistant.research_agent import research_pipeline

        assert len(research_pipeline.sub_agents) == 3

    def test_search_agent_has_google_search_tool(self):
        from personal_assistant.research_agent import search_agent

        tool_names = [getattr(t, "name", None) for t in search_agent.tools]
        assert "google_search" in tool_names

    def test_collect_agent_has_fetch_web_page_tool(self):
        from personal_assistant.research_agent import collect_agent

        tool_names = [getattr(t, "name", None) for t in collect_agent.tools]
        assert "fetch_web_page" in tool_names

    def test_write_critique_loop_is_loop_agent(self):
        from google.adk.agents import LoopAgent
        from personal_assistant.research_agent import write_critique_loop

        assert isinstance(write_critique_loop, LoopAgent)

    def test_write_critique_loop_has_max_iterations(self):
        from personal_assistant.research_agent import write_critique_loop

        assert write_critique_loop.max_iterations is not None
        assert write_critique_loop.max_iterations > 0

    def test_write_critique_loop_sub_agents(self):
        from personal_assistant.research_agent import (
            critique_agent,
            write_agent,
            write_critique_loop,
        )

        assert write_critique_loop.sub_agents[0] is write_agent
        assert write_critique_loop.sub_agents[1] is critique_agent

    def test_agents_have_output_keys(self):
        from personal_assistant.research_agent import (
            collect_agent,
            critique_agent,
            search_agent,
            write_agent,
        )

        assert search_agent.output_key == "search_results"
        assert collect_agent.output_key == "collected_content"
        assert write_agent.output_key == "research_draft"
        assert critique_agent.output_key == "critique_feedback"


class TestRootAgentHasResearchPipeline:
    def test_root_agent_has_research_pipeline_sub_agent(self):
        from personal_assistant.research_agent import research_pipeline

        # Import agent.py; it may warn about missing env vars but should not crash
        import personal_assistant.agent as agent_mod

        agent = agent_mod.root_agent
        sub_agent_names = [a.name for a in agent.sub_agents]
        assert research_pipeline.name in sub_agent_names
