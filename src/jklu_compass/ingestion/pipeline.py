"""End-to-end document ingestion pipeline for JKLU Compass."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jklu_compass.ingestion.chunker import DocumentChunk, DocumentChunker
from jklu_compass.ingestion.cleaner import DocumentCleaner
from jklu_compass.ingestion.loaders import (
    DocumentLoader,
    SourceDocument,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionReport:
    """Results and statistics from an ingestion run."""

    source_path: str
    documents_loaded: int = 0
    documents_cleaned: int = 0
    documents_skipped: int = 0
    files_seen: int = 0
    records_seen: int = 0
    chunks_created: int = 0
    chunks: list[DocumentChunk] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class IngestionPipeline:
    """Load, clean, and chunk source documents."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        cleaner: DocumentCleaner | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        """Initialize the pipeline with optional custom components."""
        self.loader = loader or DocumentLoader()
        self.cleaner = cleaner or DocumentCleaner()
        self.chunker = chunker or DocumentChunker()

    def run(self, path: str | Path) -> IngestionReport:
        """Process a file or directory into retrieval-ready chunks.

        Args:
            path: A JSONL file, Markdown file, or directory of Markdown files.

        Returns:
            A report containing processed chunks and ingestion statistics.
        """
        source_path = Path(path).expanduser()

        # Stage 1: Load source documents.
        load_report = self.loader.load_path(source_path)

        report = IngestionReport(
            source_path=str(source_path.resolve()),
            documents_loaded=load_report.documents_loaded,
            files_seen=load_report.files_seen,
            records_seen=load_report.records_seen,
            errors=list(load_report.errors),
        )

        cleaned_documents: list[SourceDocument] = []

        # Stage 2: Clean documents and preserve their metadata.
        for document in load_report.documents:
            try:
                cleaned_content = self.cleaner.clean(document.content)

                if not cleaned_content.strip():
                    report.documents_skipped += 1
                    continue

                cleaned_documents.append(
                    SourceDocument(
                        content=cleaned_content,
                        source=document.source,
                        metadata=dict(document.metadata),
                    )
                )

            except (TypeError, ValueError) as exc:
                report.documents_skipped += 1
                report.errors.append(f"Could not clean document from {document.source}: {exc}")
                logger.warning(
                    "Skipping document from %s: %s",
                    document.source,
                    exc,
                )

        report.documents_cleaned = len(cleaned_documents)

        # Stage 3: Split cleaned documents into chunks.
        report.chunks = self.chunker.chunk_documents(cleaned_documents)
        report.chunks_created = len(report.chunks)

        logger.info(
            "Ingestion complete: %d loaded, %d cleaned, %d skipped, %d chunks created.",
            report.documents_loaded,
            report.documents_cleaned,
            report.documents_skipped,
            report.chunks_created,
        )

        return report

    @staticmethod
    def summarize(report: IngestionReport) -> dict[str, Any]:
        """Return a concise, serializable summary of an ingestion run."""
        return {
            "source_path": report.source_path,
            "files_seen": report.files_seen,
            "records_seen": report.records_seen,
            "documents_loaded": report.documents_loaded,
            "documents_cleaned": report.documents_cleaned,
            "documents_skipped": report.documents_skipped,
            "chunks_created": report.chunks_created,
            "errors": len(report.errors),
        }
