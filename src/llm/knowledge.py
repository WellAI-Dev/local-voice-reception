"""
Knowledge Manager for LLM context injection.
Loads, manages, and injects markdown knowledge files into the LLM context.
"""

import logging
import re
import threading
import unicodedata
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _is_safe_filename(filename: str) -> bool:
    """
    Check that a filename is safe and does not contain path traversal.

    Args:
        filename: The filename to validate.

    Returns:
        True if the filename is safe, False otherwise.
    """
    if not filename:
        return False
    if "/" in filename or "\\" in filename:
        return False
    if filename.startswith("."):
        return False
    if ".." in filename:
        return False
    return True


def _sanitize_filename(title: str) -> str:
    """
    Sanitize a title string into a safe filename.

    Normalizes unicode, replaces spaces and special characters with underscores,
    removes consecutive underscores, and strips leading/trailing underscores.

    Args:
        title: Raw title string to convert to a filename.

    Returns:
        Sanitized filename string (without extension).
    """
    normalized = unicodedata.normalize("NFKC", title)
    sanitized = re.sub(r"[^\w\-]", "_", normalized)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    if not sanitized:
        sanitized = "untitled"
    return sanitized


def _derive_title(filename: str) -> str:
    """
    Derive a display title from a filename.

    Strips the .md extension if present.

    Args:
        filename: Filename string (e.g., "company_info.md").

    Returns:
        Title string (e.g., "company_info").
    """
    return Path(filename).stem


class KnowledgeManager:
    """Manages markdown knowledge files for LLM context injection."""

    def __init__(self, knowledge_dir: str = "data/knowledge"):
        """
        Initialize the knowledge manager.

        Args:
            knowledge_dir: Path to the directory containing knowledge markdown files.
        """
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: dict[str, str] = {}
        self.load_all()

    def load_all(self) -> None:
        """Load all .md files from the knowledge directory."""
        entries: dict[str, str] = {}
        md_files = sorted(self.knowledge_dir.glob("*.md"))

        for filepath in md_files:
            try:
                content = filepath.read_text(encoding="utf-8")
                entries[filepath.name] = content
                logger.info(f"Loaded knowledge file: {filepath.name}")
            except Exception as e:
                logger.error(f"Failed to load knowledge file {filepath.name}: {e}")

        with self._lock:
            self._entries = entries

        logger.info(f"Loaded {len(entries)} knowledge entries")

    def get_context(self, max_chars: int = 2000) -> str:
        """
        Build a context string from all loaded knowledge entries.

        Concatenates all entries with section headers. If the total exceeds
        max_chars, the output is truncated at a section boundary or hard-cut.

        Args:
            max_chars: Maximum character count for the context string.

        Returns:
            Formatted context string, or empty string if no entries are loaded.
        """
        with self._lock:
            snapshot = dict(self._entries)

        if not snapshot:
            return ""

        sections = []
        for filename, content in snapshot.items():
            title = _derive_title(filename)
            section = f"=== {title} ===\n{content.strip()}"
            sections.append(section)

        combined = "\n\n".join(sections)

        if len(combined) <= max_chars:
            return combined

        truncated = combined[:max_chars]
        last_section_break = truncated.rfind("\n\n===")
        if last_section_break > 0:
            return truncated[:last_section_break]

        return truncated

    def add_entry(self, title: str, content: str) -> str:
        """
        Save a new knowledge entry as a markdown file.

        Args:
            title: Title for the knowledge entry (used to generate filename).
            content: Markdown content of the entry.

        Returns:
            The filename that was created.

        Raises:
            ValueError: If title or content is empty.
        """
        if not title or not title.strip():
            raise ValueError("Title cannot be empty")
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        sanitized = _sanitize_filename(title)
        filename = f"{sanitized}.md"
        filepath = self.knowledge_dir / filename

        filepath.write_text(content, encoding="utf-8")
        with self._lock:
            self._entries[filename] = content
        logger.info(f"Added knowledge entry: {filename}")

        return filename

    def update_entry(self, filename: str, content: str) -> bool:
        """
        Update an existing knowledge entry.

        Args:
            filename: The filename of the entry to update.
            content: New markdown content.

        Returns:
            True if the entry was updated, False if the file does not exist.

        Raises:
            ValueError: If content is empty.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        if not _is_safe_filename(filename):
            logger.warning(f"Unsafe filename rejected: {filename}")
            return False

        filepath = self.knowledge_dir / filename
        if filepath.resolve().parent != self.knowledge_dir.resolve():
            logger.warning(f"Path traversal rejected: {filename}")
            return False

        if not filepath.exists():
            logger.warning(f"Knowledge entry not found: {filename}")
            return False

        filepath.write_text(content, encoding="utf-8")
        with self._lock:
            self._entries[filename] = content
        logger.info(f"Updated knowledge entry: {filename}")

        return True

    def remove_entry(self, filename: str) -> bool:
        """
        Remove a knowledge entry.

        Args:
            filename: The filename of the entry to remove.

        Returns:
            True if the entry was removed, False if it did not exist.
        """
        if not _is_safe_filename(filename):
            logger.warning(f"Unsafe filename rejected: {filename}")
            return False

        filepath = self.knowledge_dir / filename
        if filepath.resolve().parent != self.knowledge_dir.resolve():
            logger.warning(f"Path traversal rejected: {filename}")
            return False

        if not filepath.exists():
            logger.warning(f"Knowledge entry not found: {filename}")
            return False

        filepath.unlink()
        with self._lock:
            self._entries.pop(filename, None)
        logger.info(f"Removed knowledge entry: {filename}")

        return True

    def list_entries(self) -> list[dict]:
        """
        List all entries with metadata.

        Returns:
            A list of dicts containing filename, title, size, and preview
            for each loaded knowledge entry.
        """
        with self._lock:
            snapshot = dict(self._entries)

        entries = []
        for filename, content in snapshot.items():
            entries.append({
                "filename": filename,
                "title": _derive_title(filename),
                "size": len(content),
                "preview": content[:100].replace("\n", " "),
            })
        return entries

    def get_entry(self, filename: str) -> Optional[str]:
        """
        Get the content of a specific entry.

        Args:
            filename: The filename of the entry to retrieve.

        Returns:
            The content string, or None if the entry is not found.
        """
        if not _is_safe_filename(filename):
            logger.warning(f"Unsafe filename rejected: {filename}")
            return None
        with self._lock:
            return self._entries.get(filename)
