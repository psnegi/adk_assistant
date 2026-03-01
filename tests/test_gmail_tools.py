"""Tests for the Gmail summary tools."""

import pytest
from unittest.mock import patch, MagicMock
from personal_assistant.tools.gmail_summary import gmail_summary, get_email_content


class TestGmailSummary:
    def test_missing_token_file_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GMAIL_TOKEN_FILE", str(tmp_path / "nonexistent.json"))
        result = gmail_summary()
        assert "not found" in result.lower() or "credentials" in result.lower()

    def test_returns_summary_when_messages_present(self, tmp_path, monkeypatch):
        token_file = tmp_path / "token.json"
        token_file.write_text("{}")
        monkeypatch.setenv("GMAIL_TOKEN_FILE", str(token_file))

        fake_creds = MagicMock()
        fake_service = MagicMock()
        fake_msg_list = {"messages": [{"id": "msg1"}]}
        fake_msg_detail = {
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Hello"},
                    {"name": "From", "value": "test@example.com"},
                ]
            }
        }
        fake_service.users().messages().list().execute.return_value = fake_msg_list
        fake_service.users().messages().get().execute.return_value = fake_msg_detail

        with patch("personal_assistant.tools.gmail_summary.Credentials.from_authorized_user_file", return_value=fake_creds):
            with patch("personal_assistant.tools.gmail_summary.build", return_value=fake_service):
                result = gmail_summary(lookback_minutes=60, max_results=5)

        assert "Hello" in result
        assert "test@example.com" in result

    def test_no_messages_returns_no_emails_message(self, tmp_path, monkeypatch):
        token_file = tmp_path / "token.json"
        token_file.write_text("{}")
        monkeypatch.setenv("GMAIL_TOKEN_FILE", str(token_file))

        fake_creds = MagicMock()
        fake_service = MagicMock()
        fake_service.users().messages().list().execute.return_value = {"messages": []}

        with patch("personal_assistant.tools.gmail_summary.Credentials.from_authorized_user_file", return_value=fake_creds):
            with patch("personal_assistant.tools.gmail_summary.build", return_value=fake_service):
                result = gmail_summary()

        assert "No emails found" in result


class TestGetEmailContent:
    def test_no_query_returns_error(self, tmp_path, monkeypatch):
        token_file = tmp_path / "token.json"
        token_file.write_text("{}")
        monkeypatch.setenv("GMAIL_TOKEN_FILE", str(token_file))

        fake_creds = MagicMock()
        with patch("personal_assistant.tools.gmail_summary.Credentials.from_authorized_user_file", return_value=fake_creds):
            result = get_email_content(sender="", subject="")

        assert "provide at least" in result.lower()

    def test_missing_token_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GMAIL_TOKEN_FILE", str(tmp_path / "missing.json"))
        result = get_email_content(sender="someone@example.com")
        assert "not found" in result.lower() or "credentials" in result.lower()
