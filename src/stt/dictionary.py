"""
STT Dictionary Module.
Provides post-processing correction for Vosk STT output.

Loads a YAML-based correction dictionary and applies:
- Exact string replacements (e.g., "こあ" -> "コア")
- Regex pattern-based replacements (e.g., "おでんわ" -> "お電話")
"""

import logging
import re
import threading
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Maximum allowed length for regex patterns to prevent DoS
MAX_PATTERN_LENGTH = 200


class STTDictionary:
    """
    STT correction dictionary for fixing Vosk misrecognitions.

    Supports two types of corrections:
    - corrections: exact string replacements with optional notes
    - patterns: regex-based replacements for flexible matching
    """

    def __init__(self, dict_path: str = "config/stt_dictionary.yaml"):
        """
        Initialize STT Dictionary.

        Args:
            dict_path: Path to the YAML dictionary file
        """
        self._dict_path = Path(dict_path)
        self._lock = threading.Lock()
        self._corrections: list[dict] = []
        self._patterns: list[dict] = []

        self.load()

    def load(self) -> None:
        """Load dictionary from YAML file."""
        if not self._dict_path.exists():
            logger.warning(f"STT dictionary not found: {self._dict_path}")
            with self._lock:
                self._corrections = []
                self._patterns = []
            return

        try:
            with open(self._dict_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.warning(
                    f"STT dictionary has invalid format: {self._dict_path}"
                )
                with self._lock:
                    self._corrections = []
                    self._patterns = []
                return

            corrections = list(data.get("corrections", []) or [])
            patterns = list(data.get("patterns", []) or [])

            with self._lock:
                self._corrections = corrections
                self._patterns = patterns

            logger.info(
                f"STT dictionary loaded: {len(corrections)} corrections, "
                f"{len(patterns)} patterns"
            )

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse STT dictionary: {e}")
            with self._lock:
                self._corrections = []
                self._patterns = []

    def save(self) -> None:
        """Save dictionary to YAML file."""
        data = {
            "corrections": self._corrections,
            "patterns": self._patterns,
        }

        self._dict_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self._dict_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
            logger.info(f"STT dictionary saved: {self._dict_path}")

        except OSError as e:
            logger.error(f"Failed to save STT dictionary: {e}")
            raise

    def correct(self, text: str) -> str:
        """
        Apply all corrections to recognized text.

        Applies exact corrections first (sorted by length, longest first
        to avoid partial matches), then regex patterns.

        Args:
            text: Recognized text from STT

        Returns:
            Corrected text
        """
        if not text:
            return text

        # Take a snapshot for thread safety
        with self._lock:
            corrections = list(self._corrections)
            patterns = list(self._patterns)

        result = text

        # Apply exact corrections (longest match first)
        sorted_corrections = sorted(
            corrections,
            key=lambda c: len(c.get("wrong", "")),
            reverse=True,
        )
        for entry in sorted_corrections:
            wrong = entry.get("wrong", "")
            correct_text = entry.get("correct", "")
            if wrong and correct_text:
                result = result.replace(wrong, correct_text)

        # Apply regex patterns
        for entry in patterns:
            pattern = entry.get("pattern", "")
            replacement = entry.get("replacement", "")
            if pattern:
                try:
                    result = re.sub(pattern, replacement, result)
                except re.error as e:
                    logger.warning(
                        f"Invalid regex pattern '{pattern}': {e}"
                    )

        return result

    def add_correction(
        self, wrong: str, correct: str, note: str = ""
    ) -> None:
        """
        Add a new correction entry.

        If an entry with the same 'wrong' value exists, it is updated.

        Args:
            wrong: The misrecognized text
            correct: The correct replacement text
            note: Optional description for this entry
        """
        with self._lock:
            # Update existing entry if present
            for entry in self._corrections:
                if entry.get("wrong") == wrong:
                    entry["correct"] = correct
                    entry["note"] = note
                    logger.info(f"Updated STT correction: '{wrong}' -> '{correct}'")
                    return

            new_entry = {"wrong": wrong, "correct": correct, "note": note}
            self._corrections.append(new_entry)
        logger.info(f"Added STT correction: '{wrong}' -> '{correct}'")

    def remove_correction(self, wrong: str) -> bool:
        """
        Remove a correction entry.

        Args:
            wrong: The misrecognized text to remove

        Returns:
            True if the entry was found and removed, False otherwise
        """
        with self._lock:
            original_len = len(self._corrections)
            self._corrections = [
                entry
                for entry in self._corrections
                if entry.get("wrong") != wrong
            ]
            removed = len(self._corrections) < original_len

        if removed:
            logger.info(f"Removed STT correction: '{wrong}'")
        else:
            logger.debug(f"STT correction not found: '{wrong}'")

        return removed

    def list_corrections(self) -> list[dict]:
        """
        List all correction entries.

        Returns:
            Copy of the corrections list
        """
        with self._lock:
            return list(self._corrections)

    def add_pattern(self, pattern: str, replacement: str) -> None:
        """
        Add a regex pattern correction.

        Validates the pattern before adding. If a pattern with the same
        value exists, it is updated.

        Args:
            pattern: Regex pattern to match
            replacement: Replacement string (supports backreferences)

        Raises:
            ValueError: If the regex pattern is invalid
        """
        # Validate pattern length to prevent ReDoS
        if len(pattern) > MAX_PATTERN_LENGTH:
            raise ValueError(
                f"Pattern too long ({len(pattern)} chars, max {MAX_PATTERN_LENGTH})"
            )

        # Validate regex
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

        with self._lock:
            # Update existing pattern if present
            for entry in self._patterns:
                if entry.get("pattern") == pattern:
                    entry["replacement"] = replacement
                    logger.info(f"Updated STT pattern: '{pattern}'")
                    return

            new_entry = {"pattern": pattern, "replacement": replacement}
            self._patterns.append(new_entry)
        logger.info(f"Added STT pattern: '{pattern}'")

    def remove_pattern(self, pattern: str) -> bool:
        """
        Remove a pattern correction.

        Args:
            pattern: The regex pattern to remove

        Returns:
            True if the pattern was found and removed, False otherwise
        """
        with self._lock:
            original_len = len(self._patterns)
            self._patterns = [
                entry
                for entry in self._patterns
                if entry.get("pattern") != pattern
            ]
            removed = len(self._patterns) < original_len

        if removed:
            logger.info(f"Removed STT pattern: '{pattern}'")
        else:
            logger.debug(f"STT pattern not found: '{pattern}'")

        return removed

    def list_patterns(self) -> list[dict]:
        """
        List all pattern entries.

        Returns:
            Copy of the patterns list
        """
        with self._lock:
            return list(self._patterns)
