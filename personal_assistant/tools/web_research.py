"""Web page fetching tool for the research pipeline."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 10  # seconds
_MAX_CONTENT_CHARS = 8000


def fetch_web_page(url: str) -> str:
    """Fetch and extract the main text content from a web page.

    Args:
        url: The full URL of the web page to fetch.

    Returns:
        Extracted plain-text content of the page, or an error message.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Invalid URL scheme '{parsed.scheme}'. Only http and https are supported."

    try:
        response = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ADK-Research-Bot/1.0)"},
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return f"Request timed out while fetching: {url}"
    except requests.exceptions.HTTPError as exc:
        return f"HTTP error {exc.response.status_code} while fetching: {url}"
    except requests.exceptions.RequestException as exc:
        return f"Error fetching {url}: {exc}"

    # Parse HTML and extract readable text
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = [line for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    if not text:
        return f"No readable text content found at: {url}"

    if len(text) > _MAX_CONTENT_CHARS:
        text = text[:_MAX_CONTENT_CHARS] + "\n\n[... content truncated ...]"

    return f"[Source: {url}]\n\n{text}"


fetch_web_page_tool = FunctionTool(fetch_web_page)
