"""Text normalization utilities for JKLU Compass."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Configuration for conservative document cleaning."""

    normalize_unicode: bool = True
    normalize_line_endings: bool = True
    remove_null_bytes: bool = True
    collapse_horizontal_whitespace: bool = True
    max_consecutive_blank_lines: int = 1
    strip_outer_whitespace: bool = True

    def __post_init__(self) -> None:
        """Validate cleaning configuration."""
        if self.max_consecutive_blank_lines < 0:
            raise ValueError("max_consecutive_blank_lines cannot be negative.")


class DocumentCleaner:
    """Normalize scraped text without intentionally removing its meaning."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        """Initialize the cleaner with optional configuration."""
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        """Clean a document while preserving headings and paragraph boundaries.

        Args:
            text: Raw document text.

        Returns:
            Normalized text.

        Raises:
            TypeError: If the input is not a string.
        """
        if not isinstance(text, str):
            raise TypeError("Document text must be a string.")

        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        if self.config.remove_null_bytes:
            text = text.replace("\x00", "")

        if self.config.normalize_line_endings:
            text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize horizontal whitespace without joining separate lines.
        if self.config.collapse_horizontal_whitespace:
            text = re.sub(r"[^\S\n]+", " ", text)

        # Remove spaces and tabs at the beginning and end of each line.
        lines = [line.strip(" \t") for line in text.split("\n")]

        # Limit consecutive blank lines while retaining paragraph separation.
        cleaned_lines: list[str] = []
        blank_lines = 0
        maximum_blank_lines = self.config.max_consecutive_blank_lines

        for line in lines:
            if line:
                cleaned_lines.append(line)
                blank_lines = 0
                continue

            blank_lines += 1

            if blank_lines <= maximum_blank_lines:
                cleaned_lines.append("")

        text = "\n".join(cleaned_lines)

        if self.config.strip_outer_whitespace:
            text = text.strip()

        return text
