from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List
from collections import Counter

from google.adk.tools import FunctionTool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

_SCOPES: List[str] = ["https://www.googleapis.com/auth/gmail.readonly"]


def gmail_summary(lookback_minutes: int = 60, max_results: int = 20) -> str:
    """Return a bullet summary of recent Gmail messages from Primary inbox only.
    
    Args:
        lookback_minutes: How far back to search (default 60 minutes).
        max_results: Maximum number of messages to fetch (default 20).
    """

    token_path = os.getenv("GMAIL_TOKEN_FILE", "token.json")
    if not os.path.exists(token_path):
        return (
            "Gmail credentials not found. Run the OAuth flow and set GMAIL_TOKEN_FILE to the"
            f" saved token.json (looked in {token_path})."
        )

    try:
        credentials = Credentials.from_authorized_user_file(token_path, scopes=_SCOPES)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Failed to load Gmail credentials: {exc}"

    try:
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        # Use category:primary to get only Primary category messages (excludes Social, Promotions, etc.)
        query = f"after:{int(cutoff.timestamp())} category:primary"

        response = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max(1, max_results))
            .execute()
        )
        messages = response.get("messages", [])
        if not messages:
            return f"No emails found in Primary from the last {lookback_minutes} minutes."

        summaries = []
        senders = []
        for msg_meta in messages:
            message = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=msg_meta["id"],
                    format="metadata",
                    metadataHeaders=["Subject", "From"],
                )
                .execute()
            )
            headers = {header["name"]: header["value"] for header in message["payload"].get("headers", [])}
            subject = headers.get("Subject", "(no subject)")
            sender = headers.get("From", "unknown sender")
            senders.append(sender)
            summaries.append(f"- {subject} — {sender}")

        result = f"Recent emails from Primary inbox:\n" + "\n".join(summaries)
        
        if senders:
            sender_counts = Counter(senders)
            total_count = len(summaries)
            sender_summary = "\n".join(
                f"  • {sender}: {count} message(s)"
                for sender, count in sender_counts.most_common()
            )
            result += f"\n\nSummary Statistics:\n- Total messages: {total_count}\n- Messages by sender:\n{sender_summary}"
        
        return result
    except HttpError as http_err:
        return f"Gmail API request failed: {http_err}"


def get_email_content(sender: str = "", subject: str = "", max_results: int = 5) -> str:
    """Get the full content of specific email(s) by searching for sender and/or subject.
    
    Args:
        sender: Email address OR display name of the sender to search for (e.g., "John Doe" or "john@example.com")
        subject: Subject line keywords to search for (optional)
        max_results: Maximum number of matching emails to return (default 5)
    
    Returns:
        Full content of matching email(s)
    
    Note: Gmail search for 'from:' matches both email addresses and display names.
    """
    token_path = os.getenv("GMAIL_TOKEN_FILE", "token.json")
    if not os.path.exists(token_path):
        return (
            "Gmail credentials not found. Run the OAuth flow and set GMAIL_TOKEN_FILE to the"
            f" saved token.json (looked in {token_path})."
        )

    try:
        credentials = Credentials.from_authorized_user_file(token_path, scopes=_SCOPES)
    except Exception as exc:
        return f"Failed to load Gmail credentials: {exc}"

    try:
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)

        # Build search query
        # Gmail's 'from:' operator matches both display names and email addresses
        query_parts = []
        if sender:
            # If sender contains spaces, it's likely a display name - wrap in quotes
            if " " in sender and not sender.startswith('"'):
                query_parts.append(f'from:"{sender}"')
            else:
                query_parts.append(f"from:{sender}")
        if subject:
            # Wrap subject in quotes if it contains spaces
            if " " in subject and not subject.startswith('"'):
                query_parts.append(f'subject:"{subject}"')
            else:
                query_parts.append(f"subject:{subject}")
        
        if not query_parts:
            return "Please provide at least a sender or subject to search for."
        
        query = " ".join(query_parts)

        response = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max(1, max_results))
            .execute()
        )
        messages = response.get("messages", [])
        
        if not messages:
            return f"No emails found matching: {query}"

        results = []
        for msg_meta in messages:
            message = (
                service.users()
                .messages()
                .get(userId="me", id=msg_meta["id"], format="full")
                .execute()
            )
            
            # Extract headers
            headers = {h["name"]: h["value"] for h in message["payload"].get("headers", [])}
            email_subject = headers.get("Subject", "(no subject)")
            email_from = headers.get("From", "unknown")
            email_date = headers.get("Date", "unknown date")
            
            # Extract body
            body = ""
            if "parts" in message["payload"]:
                for part in message["payload"]["parts"]:
                    if part["mimeType"] == "text/plain" and "data" in part["body"]:
                        import base64
                        body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                        break
            elif "body" in message["payload"] and "data" in message["payload"]["body"]:
                import base64
                body = base64.urlsafe_b64decode(message["payload"]["body"]["data"]).decode("utf-8")
            
            if not body:
                body = "(No text content found)"
            
            # Truncate long bodies
            if len(body) > 2000:
                body = body[:2000] + "\n... (truncated)"
            
            results.append(
                f"📧 Email {len(results)+1}\n"
                f"From: {email_from}\n"
                f"Subject: {email_subject}\n"
                f"Date: {email_date}\n"
                f"{'─'*60}\n"
                f"{body}\n"
                f"{'═'*60}\n"
            )
        
        return "\n".join(results)
        
    except HttpError as http_err:
        return f"Gmail API request failed: {http_err}"


gmail_summary_tool = FunctionTool(gmail_summary)
get_email_content_tool = FunctionTool(get_email_content)