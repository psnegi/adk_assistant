"""Hierarchical local memory manager backed by a markdown file.

The memory file is structured as markdown with ``##`` top-level section
headings and optional ``###`` sub-section headings so it remains human-
readable and easy to edit outside the agent.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# Default memory file path; override with MEMORY_FILE env var.
_DEFAULT_MEMORY_FILE = os.path.join(os.path.expanduser("~"), ".adk_assistant_memory.md")


def _memory_file_path() -> str:
    return os.getenv("MEMORY_FILE", _DEFAULT_MEMORY_FILE)


def _load_memory() -> str:
    """Return the raw markdown content of the memory file, or an empty string."""
    path = _memory_file_path()
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _save_memory(content: str) -> None:
    """Persist *content* to the memory file, creating it if needed."""
    path = _memory_file_path()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# ── Section parsing helpers ───────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _split_into_sections(content: str) -> list[dict]:
    """Split markdown *content* into a list of section dicts.

    Each dict has keys ``level`` (1-3), ``title``, and ``body`` (text that
    follows the heading up to the next same-or-higher-level heading).
    """
    matches = list(_SECTION_RE.finditer(content))
    if not matches:
        return []

    sections = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()
        sections.append({"level": level, "title": title, "body": body, "start": m.start()})
    return sections


def _section_heading(level: int, title: str) -> str:
    return f"{'#' * level} {title}"


# ── Public tool functions ─────────────────────────────────────────────────────


def update_memory(
    section: str,
    content: str,
    subsection: str = "",
) -> str:
    """Update (or create) a section in the agent's local memory file.

    The memory is stored as a markdown file so it stays human-readable.
    Use this tool whenever the user shares information worth remembering
    (preferences, facts, ongoing tasks, skill notes, etc.).

    Args:
        section: Top-level section name (e.g. "Preferences", "Skills",
            "Ongoing Tasks").  Created if it does not exist.
        content: Markdown-formatted content to write into the section (or
            sub-section).  The existing content of the section is replaced.
        subsection: Optional sub-section name within *section*.  If provided,
            only that sub-section is replaced; other sub-sections are kept.

    Returns:
        Confirmation message.
    """
    raw = _load_memory()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not subsection:
        # Replace entire top-level section
        new_block = f"## {section}\n\n{content}\n\n_Last updated: {timestamp}_"
        # Remove existing section (and its children) if present
        raw = _remove_section(raw, section, level=2)
        raw = raw.rstrip() + "\n\n" + new_block + "\n"
    else:
        # Ensure top-level section exists
        if not _find_section(raw, section, level=2):
            raw = raw.rstrip() + f"\n\n## {section}\n"
        new_block = f"### {subsection}\n\n{content}\n\n_Last updated: {timestamp}_"
        raw = _remove_section(raw, subsection, level=3, parent=section)
        raw = _append_under_section(raw, section, new_block)

    _save_memory(raw)
    loc = f"{section} > {subsection}" if subsection else section
    return f"✅ Memory updated: **{loc}**"


def read_memory(section: str = "") -> str:
    """Read the agent's local memory.

    Args:
        section: If provided, return only that top-level section (and its
            sub-sections).  If empty, return the entire memory file.

    Returns:
        Markdown-formatted memory content, or a message indicating what is
        missing.
    """
    raw = _load_memory()
    if not raw.strip():
        return "Memory is empty. Nothing has been saved yet."

    if not section:
        return raw

    block = _extract_section(raw, section, level=2)
    if block is None:
        return f"No memory section named **{section}** found."
    return block


def list_memory_sections() -> str:
    """List all top-level sections currently stored in memory.

    Returns:
        Bullet list of section names, or a message if memory is empty.
    """
    raw = _load_memory()
    if not raw.strip():
        return "Memory is empty. No sections found."

    sections = _split_into_sections(raw)
    top_level = [s["title"] for s in sections if s["level"] == 2]
    if not top_level:
        return "Memory file exists but contains no sections."
    return "**Memory sections:**\n" + "\n".join(f"- {t}" for t in top_level)


def clear_memory_section(section: str, subsection: str = "") -> str:
    """Remove a section (or sub-section) from memory.

    Args:
        section: Top-level section name to remove (or the parent when
            *subsection* is provided).
        subsection: If provided, only this sub-section is removed; the parent
            section and other sub-sections are kept.

    Returns:
        Confirmation message.
    """
    raw = _load_memory()
    if not raw.strip():
        return "Memory is already empty."

    if subsection:
        if not _find_section(raw, subsection, level=3):
            return f"Sub-section **{subsection}** not found in memory."
        raw = _remove_section(raw, subsection, level=3, parent=section)
        _save_memory(raw)
        return f"🗑️ Removed sub-section **{section} > {subsection}** from memory."
    else:
        if not _find_section(raw, section, level=2):
            return f"Section **{section}** not found in memory."
        raw = _remove_section(raw, section, level=2)
        _save_memory(raw)
        return f"🗑️ Removed section **{section}** from memory."


# ── Internal helpers ──────────────────────────────────────────────────────────


def _find_section(content: str, title: str, level: int) -> bool:
    return any(
        s["level"] == level and s["title"].lower() == title.lower()
        for s in _split_into_sections(content)
    )


def _extract_section(content: str, title: str, level: int) -> Optional[str]:
    """Return the markdown block for *title* at *level* (including children), or None."""
    sections = _split_into_sections(content)
    parts: list[str] = []
    capturing = False
    for s in sections:
        if not capturing:
            if s["level"] == level and s["title"].lower() == title.lower():
                capturing = True
                heading = "#" * s["level"] + " " + s["title"]
                parts.append(heading + ("\n\n" + s["body"] if s["body"] else ""))
        else:
            # Keep children (deeper levels); stop at same or higher level.
            if s["level"] > level:
                heading = "#" * s["level"] + " " + s["title"]
                parts.append(heading + ("\n\n" + s["body"] if s["body"] else ""))
            else:
                break
    return "\n\n".join(parts).strip() if parts else None


def _remove_section(
    content: str,
    title: str,
    level: int,
    parent: str = "",
) -> str:
    """Return *content* with the named section (and its children) removed."""
    sections = _split_into_sections(content)
    skip = False
    keep_parts: list[str] = []
    for s in sections:
        if not skip:
            if s["level"] == level and s["title"].lower() == title.lower():
                skip = True
                continue
        else:
            # Stop skipping when we reach a section at the same or higher level.
            if s["level"] <= level:
                skip = False
        if not skip:
            heading = "#" * s["level"] + " " + s["title"]
            keep_parts.append(heading + ("\n\n" + s["body"] if s["body"] else ""))

    # Preserve any leading content before the first heading
    first_match = _SECTION_RE.search(content)
    prefix = content[: first_match.start()].rstrip() if first_match else ""

    rebuilt = ("\n\n".join(keep_parts)).strip()
    return (prefix + "\n\n" + rebuilt + "\n").lstrip("\n") if rebuilt else (prefix + "\n")


def _append_under_section(content: str, section_title: str, block: str) -> str:
    """Append *block* at the end of the top-level *section_title* block."""
    sections = _split_into_sections(content)
    result_sections: list[str] = []
    appended = False

    i = 0
    while i < len(sections):
        s = sections[i]
        heading = "#" * s["level"] + " " + s["title"]
        part = heading + ("\n\n" + s["body"] if s["body"] else "")

        if s["level"] == 2 and s["title"].lower() == section_title.lower():
            # Collect this section and all its children
            child_parts: list[str] = [part]
            i += 1
            while i < len(sections) and sections[i]["level"] > 2:
                cs = sections[i]
                ch = "#" * cs["level"] + " " + cs["title"]
                child_parts.append(ch + ("\n\n" + cs["body"] if cs["body"] else ""))
                i += 1
            child_parts.append(block)
            result_sections.append("\n\n".join(child_parts))
            appended = True
            continue
        else:
            result_sections.append(part)

        i += 1

    if not appended:
        result_sections.append(f"## {section_title}\n\n{block}")

    first_match = _SECTION_RE.search(content)
    prefix = content[: first_match.start()].rstrip() if first_match else ""
    rebuilt = "\n\n".join(result_sections).strip()
    return (prefix + "\n\n" + rebuilt + "\n").lstrip("\n") if rebuilt else (prefix + "\n")


# ── ADK tool instances ────────────────────────────────────────────────────────

update_memory_tool = FunctionTool(update_memory)
read_memory_tool = FunctionTool(read_memory)
list_memory_sections_tool = FunctionTool(list_memory_sections)
clear_memory_section_tool = FunctionTool(clear_memory_section)
