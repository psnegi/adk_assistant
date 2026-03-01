"""Tests for the YouTube summary tools."""

import pytest
from unittest.mock import patch, MagicMock
from personal_assistant.tools.youtube_summary import (
    search_youtube_videos,
    get_video_summary,
    check_video_transcripts,
)


class TestSearchYoutubeVideos:
    def test_missing_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        result = search_youtube_videos("AI news")
        assert "YouTube API key not found" in result

    def test_http_error_returns_error_message(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key")
        from googleapiclient.errors import HttpError
        mock_resp = MagicMock()
        mock_resp.status = 403
        mock_resp.reason = "Forbidden"
        with patch("personal_assistant.tools.youtube_summary.build") as mock_build:
            mock_youtube = MagicMock()
            mock_build.return_value = mock_youtube
            mock_youtube.search().list().execute.side_effect = HttpError(mock_resp, b"Forbidden")
            with patch("personal_assistant.tools.retry_utils.time.sleep"):
                result = search_youtube_videos("test query")
        assert "error" in result.lower()

    def test_no_results_returns_no_videos_message(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key")
        with patch("personal_assistant.tools.youtube_summary.build") as mock_build:
            mock_youtube = MagicMock()
            mock_build.return_value = mock_youtube
            mock_youtube.search().list().execute.return_value = {"items": []}
            result = search_youtube_videos("unlikely query xyz123")
        assert "No videos found" in result

    def test_successful_search_returns_formatted_output(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key")
        search_items = [
            {
                "id": {"videoId": "abc123"},
                "snippet": {
                    "title": "Test Video",
                    "channelTitle": "Test Channel",
                    "publishedAt": "2025-01-01T00:00:00Z",
                    "description": "A test description",
                },
            }
        ]
        stats_items = [
            {
                "statistics": {"viewCount": "1000", "likeCount": "100"},
                "contentDetails": {"duration": "PT10M"},
            }
        ]
        with patch("personal_assistant.tools.youtube_summary.build") as mock_build:
            mock_youtube = MagicMock()
            mock_build.return_value = mock_youtube
            mock_youtube.search().list().execute.return_value = {"items": search_items}
            mock_youtube.videos().list().execute.return_value = {"items": stats_items}
            result = search_youtube_videos("test")
        assert "Test Video" in result
        assert "Test Channel" in result
        assert "1,000" in result


class TestGetVideoSummary:
    def test_invalid_url_returns_error(self):
        result = get_video_summary("https://not-a-youtube-url.com")
        assert "Invalid YouTube URL" in result

    def test_no_transcript_available_returns_error(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        with patch("personal_assistant.tools.youtube_summary.TRANSCRIPT_AVAILABLE", True):
            with patch("personal_assistant.tools.youtube_summary.YouTubeTranscriptApi") as mock_api_cls:
                instance = MagicMock()
                mock_api_cls.return_value = instance
                instance.fetch.side_effect = Exception("Could not retrieve a transcript")
                result = get_video_summary("https://www.youtube.com/watch?v=abc123")
        assert "No transcripts available" in result or "transcript" in result.lower()

    def test_transcript_unavailable_library_returns_install_message(self):
        with patch("personal_assistant.tools.youtube_summary.TRANSCRIPT_AVAILABLE", False):
            result = get_video_summary("https://www.youtube.com/watch?v=abc123")
        assert "not installed" in result.lower()


class TestCheckVideoTranscripts:
    def test_invalid_url_returns_error(self):
        result = check_video_transcripts("not-a-url")
        assert "Invalid YouTube URL" in result

    def test_transcript_library_missing_returns_message(self):
        with patch("personal_assistant.tools.youtube_summary.TRANSCRIPT_AVAILABLE", False):
            result = check_video_transcripts("https://www.youtube.com/watch?v=abc123")
        assert "not installed" in result.lower()
