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



def test_markdown_front_matter_preserves_source_url_and_title(tmp_path) -> None:
    """Markdown front matter should become metadata, not document content."""
    markdown_path = tmp_path / "fee-structure.md"
    markdown_path.write_text(
        '---\n'
        'url: "https://jklu.edu.in/admissions/fee-structure"\n'
        'title: "  JKLU Fee Structure | UG, PG & PhD Programmes  "\n'
        '---\n'
        '# Fee Structure\n'
        'Information about academic fees.\n',
        encoding="utf-8",
    )

    report = DocumentLoader().load_markdown_file(markdown_path)

    assert report.documents_loaded == 1

    document = report.documents[0]
    assert document.metadata["source_url"] == (
        "https://jklu.edu.in/admissions/fee-structure"
    )
    assert document.metadata["title"] == (
        "JKLU Fee Structure | UG, PG & PhD Programmes"
    )
    assert document.content.startswith("# Fee Structure")
    assert "url:" not in document.content
    assert "title:" not in document.content
    assert document.source == str(markdown_path.resolve())
    assert document.metadata["file_name"] == "fee-structure.md"
    assert document.metadata["source_type"] == "markdown"

def test_load_path_rejects_unsupported_extension(tmp_path) -> None:
    """Unsupported input file types should raise a clear error."""
    input_file = tmp_path / "data.csv"
    input_file.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentLoader().load_path(input_file)
