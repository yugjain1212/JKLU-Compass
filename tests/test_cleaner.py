
"""Tests for JKLU Compass document cleaning."""

import pytest

from jklu_compass.ingestion.cleaner import CleaningConfig, DocumentCleaner


def test_cleaner_normalizes_horizontal_whitespace() -> None:
    """Repeated spaces should be collapsed without joining lines."""
    cleaner = DocumentCleaner()

    result = cleaner.clean("JKLU    offers   programs.\nSecond   line.")

    assert result == "JKLU offers programs.\nSecond line."


def test_cleaner_preserves_headings_and_paragraphs() -> None:
    """Headings and meaningful paragraph boundaries should remain."""
    cleaner = DocumentCleaner()
    text = "#   Academics\n\n\nUniversity programs.\n\nAdmission details."

    result = cleaner.clean(text)

    assert result == "# Academics\n\nUniversity programs.\n\nAdmission details."


def test_cleaner_normalizes_line_endings() -> None:
    """Windows and legacy Mac line endings should become LF."""
    cleaner = DocumentCleaner()

    result = cleaner.clean("First line\r\nSecond line\rThird line")

    assert result == "First line\nSecond line\nThird line"


def test_cleaner_removes_null_bytes() -> None:
    """Null bytes should be removed."""
    cleaner = DocumentCleaner()

    result = cleaner.clean("JKLU\x00 Compass")

    assert result == "JKLU Compass"


def test_cleaner_normalizes_unicode() -> None:
    """Equivalent Unicode representations should be normalized."""
    cleaner = DocumentCleaner()

    result = cleaner.clean("Cafe\u0301")

    assert result == "Café"


def test_cleaner_rejects_non_string_input() -> None:
    """Invalid input types should raise a clear exception."""
    cleaner = DocumentCleaner()

    with pytest.raises(TypeError, match="must be a string"):
        cleaner.clean(None)  # type: ignore[arg-type]


def test_cleaner_can_remove_all_blank_lines() -> None:
    """Configuration should allow all blank lines to be removed."""
    cleaner = DocumentCleaner(
        CleaningConfig(max_consecutive_blank_lines=0)
    )

    result = cleaner.clean("First paragraph.\n\nSecond paragraph.")

    assert result == "First paragraph.\nSecond paragraph."


def test_cleaner_rejects_negative_blank_line_limit() -> None:
    """Negative blank-line limits should be rejected."""
    with pytest.raises(ValueError, match="cannot be negative"):
        CleaningConfig(max_consecutive_blank_lines=-1)
