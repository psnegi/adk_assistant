"""Tests for the hierarchical memory manager."""

from __future__ import annotations

import os
import pytest
from personal_assistant.tools.memory_manager import (
    update_memory,
    read_memory,
    list_memory_sections,
    clear_memory_section,
)


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Redirect all memory operations to a temporary file for each test."""
    monkeypatch.setenv("MEMORY_FILE", str(tmp_path / "test_memory.md"))


class TestUpdateAndReadMemory:
    def test_update_creates_section(self):
        update_memory("Preferences", "I like dark mode.")
        result = read_memory("Preferences")
        assert "Preferences" in result
        assert "dark mode" in result

    def test_read_all_memory(self):
        update_memory("Preferences", "Dark mode preferred.")
        update_memory("Notes", "Remember to call Alice.")
        result = read_memory()
        assert "Preferences" in result
        assert "Notes" in result

    def test_read_missing_section_returns_message(self):
        update_memory("Notes", "Some content.")
        result = read_memory("NonExistent")
        assert "No memory section" in result or "not found" in result.lower() or "empty" in result.lower()

    def test_read_empty_memory_returns_message(self):
        result = read_memory()
        assert "empty" in result.lower()

    def test_update_replaces_existing_section(self):
        update_memory("Preferences", "Light mode.")
        update_memory("Preferences", "Dark mode.")
        result = read_memory("Preferences")
        assert "Dark mode" in result
        assert "Light mode" not in result

    def test_update_with_subsection(self):
        update_memory("Skills", "Can do web search.", subsection="Search")
        result = read_memory("Skills")
        assert "Search" in result
        assert "web search" in result

    def test_update_subsection_preserves_sibling(self):
        update_memory("Skills", "Python programming.", subsection="Python")
        update_memory("Skills", "Web search.", subsection="Search")
        result = read_memory("Skills")
        assert "Python" in result
        assert "Search" in result

    def test_update_replaces_subsection(self):
        update_memory("Skills", "Basic Python.", subsection="Python")
        update_memory("Skills", "Advanced Python.", subsection="Python")
        result = read_memory("Skills")
        assert "Advanced Python" in result
        assert "Basic Python" not in result

    def test_confirmation_message_returned(self):
        msg = update_memory("Notes", "Test note.")
        assert "Notes" in msg


class TestListMemorySections:
    def test_lists_sections(self):
        update_memory("Preferences", "Dark mode.")
        update_memory("Skills", "Python.")
        result = list_memory_sections()
        assert "Preferences" in result
        assert "Skills" in result

    def test_empty_memory_returns_message(self):
        result = list_memory_sections()
        assert "empty" in result.lower() or "no sections" in result.lower()


class TestClearMemorySection:
    def test_clear_existing_section(self):
        update_memory("Notes", "Temporary note.")
        result = clear_memory_section("Notes")
        assert "Notes" in result
        after = read_memory("Notes")
        assert "not found" in after.lower() or "No memory section" in after or "empty" in after.lower()

    def test_clear_missing_section_returns_message(self):
        update_memory("Notes", "Some content.")
        result = clear_memory_section("Ghost")
        assert "not found" in result.lower() or "Ghost" in result

    def test_clear_empty_memory_returns_message(self):
        result = clear_memory_section("Anything")
        assert "empty" in result.lower() or "not found" in result.lower()

    def test_clear_subsection_only(self):
        update_memory("Skills", "Python.", subsection="Python")
        update_memory("Skills", "Web search.", subsection="Search")
        clear_memory_section("Skills", subsection="Python")
        result = read_memory("Skills")
        assert "Web search" in result
        assert "### Python" not in result
