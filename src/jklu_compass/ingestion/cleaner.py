"""Text normalization and conservative boilerplate removal."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    """Configuration for document cleaning."""

    normalize_unicode: bool = True
    normalize_line_endings: bool = True
    remove_null_bytes: bool = True
    collapse_horizontal_whitespace: bool = True
    max_consecutive_blank_lines: int = 1
    strip_outer_whitespace: bool = True
    remove_jklu_navigation: bool = True

    def __post_init__(self) -> None:
        """Validate cleaning configuration."""
        if self.max_consecutive_blank_lines < 0:
            raise ValueError(
                "max_consecutive_blank_lines cannot be negative."
            )


class DocumentCleaner:
    """Normalize scraped text and remove identifiable JKLU boilerplate."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        """Initialize the cleaner with optional configuration."""
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        """Normalize text and optionally remove JKLU navigation.

        Args:
            text: Raw document text.

        Returns:
            Normalized document text.

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

        if self.config.remove_jklu_navigation:
            text = self._remove_jklu_navigation(text)

        if self.config.collapse_horizontal_whitespace:
            text = re.sub(r"[^\S\n]+", " ", text)

        lines = [line.strip(" \t") for line in text.split("\n")]

        cleaned_lines: list[str] = []
        blank_lines = 0

        for line in lines:
            if line:
                cleaned_lines.append(line)
                blank_lines = 0
                continue

            blank_lines += 1
            if blank_lines <= self.config.max_consecutive_blank_lines:
                cleaned_lines.append("")

        text = "\n".join(cleaned_lines)

        if self.config.strip_outer_whitespace:
            text = text.strip()

        return text

    @staticmethod
    def _remove_jklu_navigation(text: str) -> str:
        """Remove a recognizable repeated JKLU navigation/footer section.

        A footer marker is required before truncation. This avoids cutting
        documents merely because their main content contains navigation-like
        links or headings.
        """
        lines = text.split("\n")

        navigation_markers = (
            re.compile(
                r"^\s*-\s+###\s+\[(?:About Us|ACADEMICS|Quick Links)\]\(",
                re.IGNORECASE,
            ),
        )

        footer_markers = (
            re.compile(r"^\s*####\s+Other Links\s*$", re.IGNORECASE),
            re.compile(
                r"^\s*-\s+###\s+\[Other Links\]\(",
                re.IGNORECASE,
            ),
        )

        footer_positions = [
            i
            for i, line in enumerate(lines)
            if any(marker.search(line) for marker in footer_markers)
        ]

        if not footer_positions:
            return text

        for footer_position in footer_positions:
            navigation_positions = [
                i
                for i, line in enumerate(lines[:footer_position])
                if any(marker.search(line) for marker in navigation_markers)
            ]

            if not navigation_positions:
                continue

            navigation_start = navigation_positions[0]

            # Only truncate when the expected footer follows the navigation.
            # This removes the entire repeated template block, including
            # shared contact details, social links, and application links.
            return "\n".join(lines[:navigation_start]).rstrip()

        return text
