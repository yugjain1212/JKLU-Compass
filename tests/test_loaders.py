"""Tests for JKLU Compass document loading."""

import json

import pytest

from jklu_compass.ingestion.loaders import DocumentLoader, SourceDocument


def test_source_document_rejects_empty_content() -> None:
    """Empty document content should be rejected."""
    with pytest.raises(ValueError, match="content cannot be empty"):
        SourceDocument(content="  ", source="test.md")


def test_load_jsonl_preserves_source_and_metadata(tmp_path) -> None:
    """JSONL content, source URL, and metadata should be preserved."""
    jsonl_path = tmp_path / "documents.jsonl"
    record = {
        "content": "JKLU provides academic programs.",
        "source_url": "https://example.edu/academics",
        "category": "academics",
    }
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = DocumentLoader().load_jsonl(jsonl_path)

    assert report.documents_loaded == 1
    assert report.records_seen == 1
    assert report.records_skipped == 0
    assert report.documents[0].content == "JKLU provides academic programs."
    assert report.documents[0].source == "https://example.edu/academics"
    assert report.documents[0].metadata["category"] == "academics"


def test_load_jsonl_skips_malformed_json(tmp_path) -> None:
    """Malformed JSON lines should be reported without crashing the loader."""
    jsonl_path = tmp_path / "documents.jsonl"
    jsonl_path.write_text(
        '{"content": "Valid record"}\n{"content": invalid json}\n',
        encoding="utf-8",
    )

    report = DocumentLoader().load_jsonl(jsonl_path)

    assert report.documents_loaded == 1
    assert report.records_seen == 2
    assert len(report.errors) == 1
    assert "Invalid JSON" in report.errors[0]


def test_load_markdown_directory_recursively(tmp_path) -> None:
    """Markdown files in nested directories should be loaded."""
    nested_directory = tmp_path / "academics"
    nested_directory.mkdir()

    (tmp_path / "overview.md").write_text(
        "# JKLU\nUniversity overview.",
        encoding="utf-8",
    )
    (nested_directory / "programs.md").write_text(
        "# Programs\nAcademic programs.",
        encoding="utf-8",
    )

    report = DocumentLoader().load_markdown_directory(tmp_path)

    assert report.documents_loaded == 2
    assert report.files_seen == 2
    assert any(document.metadata["file_name"] == "programs.md" for document in report.documents)


def test_load_path_rejects_unsupported_extension(tmp_path) -> None:
    """Unsupported input file types should raise a clear error."""
    input_file = tmp_path / "data.csv"
    input_file.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentLoader().load_path(input_file)
