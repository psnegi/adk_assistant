"""Local file search tool.

Provides two capabilities:
1. **search_files** — find files whose names match a partial string (or glob
   pattern), optionally bounded by modification/creation time.
2. **search_file_content** — grep-style search for a text query inside files
   under a directory, with the same optional time bounds.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import List, Optional

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

_MAX_RESULTS = 200  # hard cap to avoid flooding the context window
_MAX_CONTENT_MATCHES = 50
_SNIPPET_CONTEXT = 80  # characters on each side of a content match


def _parse_time(time_str: str) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string (with or without timezone info).

    Returns a timezone-aware :class:`datetime` or ``None`` on failure.
    """
    if not time_str:
        return None
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _mtime(path: str) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)


def _is_text_file(path: str, sample_bytes: int = 8192) -> bool:
    """Heuristic: return True if the file looks like UTF-8 / ASCII text.

    A file is considered binary if it contains null bytes or cannot be decoded
    as UTF-8.
    """
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(sample_bytes)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def search_files(
    directory: str,
    name_pattern: str = "",
    after_time: str = "",
    before_time: str = "",
    max_results: int = 50,
) -> str:
    """Search for files under *directory* by name pattern and/or modification time.

    Walks the directory tree recursively and returns matching file paths.

    Args:
        directory: Root directory to search from (absolute or relative path).
        name_pattern: Partial filename or glob pattern to match against the
            file name (not the full path).  Case-insensitive.  If empty, all
            files are considered.  Examples: ``"report"``, ``"*.pdf"``,
            ``"invoice_2024*"``.
        after_time: Only return files modified **after** this time.
            ISO-8601 format: ``"2024-01-15"`` or ``"2024-01-15T09:00:00"``.
        before_time: Only return files modified **before** this time.
            Same format as *after_time*.
        max_results: Maximum number of results to return (default 50, max 200).

    Returns:
        A formatted list of matching file paths with modification times, or a
        message explaining why no results were found.
    """
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"

    after_dt = _parse_time(after_time)
    before_dt = _parse_time(before_time)

    if after_time and after_dt is None:
        return f"Could not parse after_time: {after_time!r}. Use ISO-8601 format, e.g. '2024-01-15' or '2024-01-15T09:00:00'."
    if before_time and before_dt is None:
        return f"Could not parse before_time: {before_time!r}. Use ISO-8601 format, e.g. '2024-01-15' or '2024-01-15T09:00:00'."

    cap = min(max(1, max_results), _MAX_RESULTS)
    matches: List[str] = []

    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if len(matches) >= cap:
                break
            # Name filter
            if name_pattern:
                # If the pattern contains glob chars, use fnmatch; otherwise
                # do a case-insensitive substring match.
                if any(c in name_pattern for c in ("*", "?", "[")):
                    if not fnmatch(fname.lower(), name_pattern.lower()):
                        continue
                else:
                    if name_pattern.lower() not in fname.lower():
                        continue

            fpath = os.path.join(root, fname)
            try:
                mt = _mtime(fpath)
            except OSError:
                continue

            if after_dt and mt <= after_dt:
                continue
            if before_dt and mt >= before_dt:
                continue

            matches.append(fpath)

        if len(matches) >= cap:
            break

    if not matches:
        parts = []
        if name_pattern:
            parts.append(f"name matching '{name_pattern}'")
        if after_time:
            parts.append(f"modified after {after_time}")
        if before_time:
            parts.append(f"modified before {before_time}")
        criteria = ", ".join(parts) if parts else "any criteria"
        return f"No files found in '{directory}' matching {criteria}."

    lines = [f"📁 Found {len(matches)} file(s) in '{directory}':\n"]
    for fpath in matches:
        try:
            mt = _mtime(fpath)
            mt_str = mt.strftime("%Y-%m-%d %H:%M UTC")
        except OSError:
            mt_str = "unknown"
        lines.append(f"  {fpath}  (modified: {mt_str})")

    return "\n".join(lines)


def search_file_content(
    directory: str,
    query: str,
    name_pattern: str = "",
    after_time: str = "",
    before_time: str = "",
    max_matches: int = 20,
) -> str:
    """Search for *query* text inside files under *directory*.

    Walks the directory tree and returns line-level matches with context
    snippets.  Only plain-text (UTF-8 / ASCII) files are inspected.

    Args:
        directory: Root directory to search from.
        query: Text to search for (case-insensitive substring match).
        name_pattern: Optional file name filter (partial name or glob), same
            as in ``search_files``.
        after_time: Only search files modified **after** this time (ISO-8601).
        before_time: Only search files modified **before** this time (ISO-8601).
        max_matches: Maximum number of matching lines to return (default 20,
            max 50).

    Returns:
        Formatted list of matches with file path, line number, and a short
        context snippet.
    """
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"

    if not query:
        return "Please provide a search query."

    after_dt = _parse_time(after_time)
    before_dt = _parse_time(before_time)

    if after_time and after_dt is None:
        return f"Could not parse after_time: {after_time!r}."
    if before_time and before_dt is None:
        return f"Could not parse before_time: {before_time!r}."

    cap = min(max(1, max_matches), _MAX_CONTENT_MATCHES)
    results: List[str] = []
    files_searched = 0
    query_lower = query.lower()

    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if len(results) >= cap:
                break

            # Name filter
            if name_pattern:
                if any(c in name_pattern for c in ("*", "?", "[")):
                    if not fnmatch(fname.lower(), name_pattern.lower()):
                        continue
                else:
                    if name_pattern.lower() not in fname.lower():
                        continue

            fpath = os.path.join(root, fname)

            # Time filter
            try:
                mt = _mtime(fpath)
            except OSError:
                continue
            if after_dt and mt <= after_dt:
                continue
            if before_dt and mt >= before_dt:
                continue

            if not _is_text_file(fpath):
                continue

            files_searched += 1

            try:
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if len(results) >= cap:
                            break
                        if query_lower in line.lower():
                            # Build a short snippet around the match
                            idx = line.lower().find(query_lower)
                            start = max(0, idx - _SNIPPET_CONTEXT)
                            end = min(len(line), idx + len(query) + _SNIPPET_CONTEXT)
                            snippet = line[start:end].strip()
                            results.append(f"  {fpath}:{lineno}: {snippet}")
            except OSError:
                continue

        if len(results) >= cap:
            break

    if not results:
        return (
            f"No matches for '{query}' found in '{directory}'"
            + (f" (searched {files_searched} text file(s))" if files_searched else "")
            + "."
        )

    header = f"🔍 Found {len(results)} match(es) for '{query}' in '{directory}' ({files_searched} file(s) searched):\n"
    return header + "\n".join(results)


# ── ADK tool instances ────────────────────────────────────────────────────────

search_files_tool = FunctionTool(search_files)
search_file_content_tool = FunctionTool(search_file_content)
