"""Tests for the JKLU Compass ingestion pipeline."""

import json
from pathlib import Path

import pytest

from jklu_compass.ingestion.chunker import ChunkingConfig, DocumentChunker
from jklu_compass.ingestion.pipeline import IngestionPipeline


def test_pipeline_processes_markdown_directory(tmp_path: Path) -> None:
    """The pipeline should load, clean, and chunk Markdown documents."""
    (tmp_path / "academics.md").write_text(
        "# Academics\n\nJKLU offers undergraduate programs.",
        encoding="utf-8",
    )
    (tmp_path / "admissions.md").write_text(
        "# Admissions\n\nStudents can apply for admission.",
        encoding="utf-8",
    )

    report = IngestionPipeline().run(tmp_path)

    assert report.files_seen == 2
    assert report.documents_loaded == 2
    assert report.documents_cleaned == 2
    assert report.documents_skipped == 0
    assert report.chunks_created == 2
    assert len(report.chunks) == 2
    assert report.errors == []


def test_pipeline_cleans_document_content(tmp_path: Path) -> None:
    """The pipeline should normalize text before chunking."""
    (tmp_path / "sample.md").write_text(
        "JKLU    offers   programs.\r\nSecond   line.",
        encoding="utf-8",
    )

    report = IngestionPipeline().run(tmp_path)

    assert report.documents_cleaned == 1
    assert report.chunks_created == 1
    assert report.chunks[0].content == ("JKLU offers programs.\nSecond line.")


def test_pipeline_preserves_source_metadata(tmp_path: Path) -> None:
    """Chunks should retain source information from the loader."""
    (tmp_path / "academics.md").write_text(
        "# Academics\n\nJKLU academic programs.",
        encoding="utf-8",
    )

    report = IngestionPipeline().run(tmp_path)

    assert len(report.chunks) == 1
    assert report.chunks[0].source.endswith("academics.md")
    assert report.chunks[0].metadata["file_name"] == "academics.md"
    assert report.chunks[0].metadata["source_type"] == "markdown"


def test_pipeline_accepts_jsonl_input(tmp_path: Path) -> None:
    """The pipeline should process a JSONL file."""
    jsonl_path = tmp_path / "jklu.jsonl"
    record = {
        "content": "# Campus\n\nJKLU has a university campus.",
        "source_url": "https://example.edu/campus",
        "category": "campus",
    }
    jsonl_path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    report = IngestionPipeline().run(jsonl_path)

    assert report.files_seen == 1
    assert report.documents_loaded == 1
    assert report.documents_cleaned == 1
    assert report.chunks_created == 1
    assert report.chunks[0].source == "https://example.edu/campus"
    assert report.chunks[0].metadata["category"] == "campus"


def test_pipeline_reports_empty_directory(tmp_path: Path) -> None:
    """An empty directory should produce an empty report."""
    report = IngestionPipeline().run(tmp_path)

    assert report.files_seen == 0
    assert report.documents_loaded == 0
    assert report.documents_cleaned == 0
    assert report.chunks_created == 0
    assert report.errors == []


def test_pipeline_summary_contains_expected_statistics(
    tmp_path: Path,
) -> None:
    """The summary should expose ingestion statistics."""
    (tmp_path / "sample.md").write_text(
        "JKLU university information.",
        encoding="utf-8",
    )

    report = IngestionPipeline().run(tmp_path)
    summary = IngestionPipeline.summarize(report)

    assert summary["files_seen"] == 1
    assert summary["documents_loaded"] == 1
    assert summary["documents_cleaned"] == 1
    assert summary["documents_skipped"] == 0
    assert summary["chunks_created"] == 1
    assert summary["errors"] == 0


def test_pipeline_raises_for_missing_input(tmp_path: Path) -> None:
    """A missing input path should raise FileNotFoundError."""
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        IngestionPipeline().run(missing_path)


def test_pipeline_uses_custom_chunking_configuration(
    tmp_path: Path,
) -> None:
    """Custom chunking configuration should be passed to the pipeline."""
    text = " ".join(f"word{i}" for i in range(100))
    (tmp_path / "long.md").write_text(text, encoding="utf-8")

    chunker = DocumentChunker(
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10,
        )
    )
    pipeline = IngestionPipeline(chunker=chunker)

    report = pipeline.run(tmp_path)

    assert report.documents_loaded == 1
    assert report.chunks_created > 1
    assert all(len(chunk.content) <= 100 for chunk in report.chunks)
