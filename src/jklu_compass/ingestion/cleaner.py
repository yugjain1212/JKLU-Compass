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
        """Initialize the cleaner with an optional configuration."""
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

        # Normalize Unicode characters.
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # Remove null bytes.
        if self.config.remove_null_bytes:
            text = text.replace("\x00", "")

        # Normalize Windows and legacy Mac line endings.
        if self.config.normalize_line_endings:
            text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove recognizable JKLU navigation and shared boilerplate.
        if self.config.remove_jklu_navigation:
            text = self._remove_jklu_navigation(text)

        # Collapse horizontal whitespace without merging separate lines.
        if self.config.collapse_horizontal_whitespace:
            text = re.sub(r"[^\S\n]+", " ", text)

        # Normalize individual lines and control consecutive blank lines.
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

        # Remove whitespace from the beginning and end of the document.
        if self.config.strip_outer_whitespace:
            text = text.strip()

        return text

    @staticmethod
    def _remove_jklu_navigation(text: str) -> str:
        """Remove recognizable JKLU navigation and shared page-template content.

        The shared admissions block is removed only when the document also
        contains the recognizable repeated JKLU navigation/footer structure.
        Page-specific content before the shared block is preserved.
        """
        lines = text.split("\n")

        # Recognizable repeated navigation headings.
        navigation_pattern = re.compile(
            r"^\s*-\s+###\s+\[(?:About Us|ACADEMICS|Quick Links)\]\(",
            re.IGNORECASE,
        )

        # Marker near the end of the repeated navigation section.
        footer_pattern = re.compile(
            r"^\s*####\s+Other Links\s*$",
            re.IGNORECASE,
        )

        # Beginning of the shared admissions navigation block observed
        # in the inspected JKLU scraped documents.
        admissions_pattern = re.compile(
            r"^\s*-\s+###\s+Doctor of Philosophy\s+_PhD_\s*$",
            re.IGNORECASE,
        )

        navigation_positions = [
            index
            for index, line in enumerate(lines)
            if navigation_pattern.search(line)
        ]

        footer_positions = [
            index
            for index, line in enumerate(lines)
            if footer_pattern.search(line)
        ]

        # Confirm that the recognizable navigation marker is followed
        # by the expected footer marker.
        has_template_navigation = (
            bool(navigation_positions)
            and any(
                footer_position > navigation_positions[0]
                for footer_position in footer_positions
            )
        )

        if not has_template_navigation:
            return text

        navigation_start = navigation_positions[0]

        admissions_positions = [
            index
            for index, line in enumerate(lines)
            if admissions_pattern.search(line)
        ]

        # Prefer the start of the shared admissions block because it
        # preserves page-specific content, including genuine contact details.
        if (
            admissions_positions
            and admissions_positions[0] < navigation_start
        ):
            cutoff = admissions_positions[0]
        else:
            # If the admissions marker is absent, remove the repeated
            # navigation and everything after it.
            cutoff = navigation_start

        return "\n".join(lines[:cutoff]).rstrip()
