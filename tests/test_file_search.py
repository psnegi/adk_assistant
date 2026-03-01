"""Tests for the local file search tools."""

from __future__ import annotations

import os
import time
import pytest
from personal_assistant.tools.file_search import search_files, search_file_content


@pytest.fixture()
def sample_dir(tmp_path):
    """Create a small directory tree for testing."""
    (tmp_path / "report_2024.txt").write_text("Annual report 2024 data.\nRevenue: 100k")
    # Write a file with null bytes so it is detected as binary
    (tmp_path / "invoice_jan.pdf").write_bytes(b"%PDF-1.4\x00binary\x00content")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "notes.txt").write_text("Meeting notes: discuss budget.\nAction items: review invoice.")
    (sub / "readme.md").write_text("# Project\n\nThis is the readme file.")
    return tmp_path


class TestSearchFiles:
    def test_finds_file_by_partial_name(self, sample_dir):
        result = search_files(str(sample_dir), name_pattern="report")
        assert "report_2024.txt" in result

    def test_finds_file_by_glob_pattern(self, sample_dir):
        result = search_files(str(sample_dir), name_pattern="*.txt")
        assert "report_2024.txt" in result
        assert "notes.txt" in result

    def test_no_pattern_returns_all_files(self, sample_dir):
        result = search_files(str(sample_dir))
        assert "report_2024.txt" in result
        assert "notes.txt" in result

    def test_nonexistent_directory_returns_error(self):
        result = search_files("/nonexistent/path/xyz")
        assert "not found" in result.lower() or "Directory not found" in result

    def test_no_match_returns_informative_message(self, sample_dir):
        result = search_files(str(sample_dir), name_pattern="zzz_no_such_file")
        assert "No files found" in result

    def test_after_time_filters_old_files(self, sample_dir):
        # Use a far-future time so nothing passes
        result = search_files(str(sample_dir), after_time="2099-01-01")
        assert "No files found" in result

    def test_before_time_filters_new_files(self, sample_dir):
        # Use a far-past time so nothing passes
        result = search_files(str(sample_dir), before_time="2000-01-01")
        assert "No files found" in result

    def test_after_time_returns_recent_files(self, sample_dir):
        result = search_files(str(sample_dir), after_time="2000-01-01")
        assert "report_2024.txt" in result

    def test_invalid_after_time_returns_error(self, sample_dir):
        result = search_files(str(sample_dir), after_time="not-a-date")
        assert "Could not parse" in result or "after_time" in result

    def test_invalid_before_time_returns_error(self, sample_dir):
        result = search_files(str(sample_dir), before_time="not-a-date")
        assert "Could not parse" in result or "before_time" in result

    def test_max_results_respected(self, sample_dir):
        result = search_files(str(sample_dir), max_results=1)
        # Only 1 result should appear; count lines starting with two spaces
        file_lines = [l for l in result.splitlines() if l.startswith("  ")]
        assert len(file_lines) == 1

    def test_result_includes_modification_time(self, sample_dir):
        result = search_files(str(sample_dir), name_pattern="report")
        assert "modified:" in result


class TestSearchFileContent:
    def test_finds_content_in_text_file(self, sample_dir):
        result = search_file_content(str(sample_dir), query="Revenue")
        assert "report_2024.txt" in result
        assert "Revenue" in result

    def test_case_insensitive_search(self, sample_dir):
        result = search_file_content(str(sample_dir), query="revenue")
        assert "report_2024.txt" in result

    def test_finds_content_in_subdirectory(self, sample_dir):
        result = search_file_content(str(sample_dir), query="budget")
        assert "notes.txt" in result

    def test_no_match_returns_informative_message(self, sample_dir):
        result = search_file_content(str(sample_dir), query="zzz_not_present")
        assert "No matches" in result

    def test_empty_query_returns_error(self, sample_dir):
        result = search_file_content(str(sample_dir), query="")
        assert "provide a search query" in result.lower() or "query" in result.lower()

    def test_nonexistent_directory_returns_error(self):
        result = search_file_content("/nonexistent/path", query="test")
        assert "not found" in result.lower() or "Directory not found" in result

    def test_name_pattern_filter(self, sample_dir):
        result = search_file_content(str(sample_dir), query="invoice", name_pattern="*.txt")
        # notes.txt contains "invoice"; invoice_jan.pdf is binary so skipped anyway
        assert "notes.txt" in result

    def test_after_time_filters_files(self, sample_dir):
        result = search_file_content(str(sample_dir), query="report", after_time="2099-01-01")
        assert "No matches" in result

    def test_binary_file_skipped(self, sample_dir):
        # invoice_jan.pdf is binary; should not appear in content search
        result = search_file_content(str(sample_dir), query="PDF")
        assert "invoice_jan.pdf" not in result

    def test_result_includes_line_numbers(self, sample_dir):
        result = search_file_content(str(sample_dir), query="Revenue")
        # Expect "filename:lineno:" format
        assert ":1:" in result or ":2:" in result
