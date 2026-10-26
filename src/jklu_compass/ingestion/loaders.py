"""Load source documents for the JKLU Compass knowledge base."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEXT_FIELDS = (
    "content",
    "text",
    "markdown",
    "body",
    "markdown_content",
)

SOURCE_FIELDS = (
    "source_url",
    "url",
    "source",
    "link",
    "file_path",
)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A normalized source document with its origin and metadata."""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the document after initialization."""
        if not self.content.strip():
            raise ValueError("Document content cannot be empty.")

        if not self.source.strip():
            raise ValueError("Document source cannot be empty.")


@dataclass(slots=True)
class LoadReport:
    """Summary of a document-loading operation."""

    documents: list[SourceDocument] = field(default_factory=list)
    files_seen: int = 0
    records_seen: int = 0
    records_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def documents_loaded(self) -> int:
        """Return the number of successfully loaded documents."""
        return len(self.documents)


class DocumentLoader:
    """Load JSONL and Markdown documents into a common representation."""

    def load_path(self, path: str | Path) -> LoadReport:
        """Load a JSONL file, Markdown file, or directory of Markdown files."""
        source_path = Path(path).expanduser()

        if not source_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {source_path}")

        if source_path.is_file():
            suffix = source_path.suffix.lower()

            if suffix == ".jsonl":
                return self.load_jsonl(source_path)

            if suffix in {".md", ".markdown"}:
                return self.load_markdown_file(source_path)

            raise ValueError(
                f"Unsupported file type '{suffix}' for {source_path}. "
                "Expected .jsonl, .md, or .markdown."
            )

        if not source_path.is_dir():
            raise ValueError(f"Input path is not a regular file or directory: {source_path}")

        return self.load_markdown_directory(source_path)

    def load_jsonl(self, path: str | Path) -> LoadReport:
        """Load a JSONL file, reporting malformed records individually."""
        jsonl_path = Path(path).expanduser()

        if not jsonl_path.is_file():
            raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")

        report = LoadReport(files_seen=1)

        with jsonl_path.open("r", encoding="utf-8-sig") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue

                report.records_seen += 1

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    self._record_error(
                        report,
                        f"{jsonl_path}:{line_number}: Invalid JSON: {exc.msg}",
                    )
                    continue

                if not isinstance(record, dict):
                    self._record_error(
                        report,
                        f"{jsonl_path}:{line_number}: Expected a JSON object.",
                    )
                    continue

                document = self._normalize_record(
                    record,
                    default_source=f"{jsonl_path}:{line_number}",
                )

                if document is None:
                    report.records_skipped += 1
                    continue

                report.documents.append(document)

        logger.info(
            "Loaded %d documents from %s; skipped %d records.",
            report.documents_loaded,
            jsonl_path,
            report.records_skipped,
        )

        return report

    def load_markdown_file(self, path: str | Path) -> LoadReport:
        """Load one Markdown file."""
        markdown_path = Path(path).expanduser()

        if not markdown_path.is_file():
            raise FileNotFoundError(f"Markdown file does not exist: {markdown_path}")

        if markdown_path.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError(f"Not a Markdown file: {markdown_path}")

        report = LoadReport(files_seen=1, records_seen=1)

        try:
            content = markdown_path.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            self._record_error(report, f"{markdown_path}: Could not decode UTF-8 text: {exc}")
            report.records_skipped += 1
            return report

        if not content:
            report.records_skipped += 1
            return report

        document = SourceDocument(
            content=content,
            source=str(markdown_path.resolve()),
            metadata={
                "source_type": "markdown",
                "file_name": markdown_path.name,
                "file_path": str(markdown_path.resolve()),
            },
        )

        report.documents.append(document)
        return report

    def load_markdown_directory(self, path: str | Path) -> LoadReport:
        """Recursively load Markdown files from a directory."""
        directory = Path(path).expanduser()

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        report = LoadReport()
        markdown_files = sorted(
            file_path
            for file_path in directory.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in {".md", ".markdown"}
        )

        for markdown_path in markdown_files:
            report.files_seen += 1
            report.records_seen += 1

            try:
                content = markdown_path.read_text(encoding="utf-8-sig").strip()
            except UnicodeDecodeError as exc:
                report.records_skipped += 1
                self._record_error(
                    report,
                    f"{markdown_path}: Could not decode UTF-8 text: {exc}",
                )
                continue

            if not content:
                report.records_skipped += 1
                continue

            report.documents.append(
                SourceDocument(
                    content=content,
                    source=str(markdown_path.resolve()),
                    metadata={
                        "source_type": "markdown",
                        "file_name": markdown_path.name,
                        "file_path": str(markdown_path.resolve()),
                    },
                )
            )

        logger.info(
            "Loaded %d documents from %s; skipped %d files.",
            report.documents_loaded,
            directory,
            report.records_skipped,
        )

        return report

    @staticmethod
    def _normalize_record(
        record: dict[str, Any],
        default_source: str,
    ) -> SourceDocument | None:
        """Normalize a JSON object into a source document."""
        content: str | None = None

        for field_name in TEXT_FIELDS:
            value = record.get(field_name)

            if isinstance(value, str) and value.strip():
                content = value.strip()
                break

        if content is None:
            logger.warning("Skipping record without a recognized text field.")
            return None

        source = default_source

        for field_name in SOURCE_FIELDS:
            value = record.get(field_name)

            if isinstance(value, str) and value.strip():
                source = value.strip()
                break

        metadata = {
            key: value
            for key, value in record.items()
            if key not in TEXT_FIELDS and isinstance(value, (str, int, float, bool))
        }
        metadata["source_type"] = "jsonl"

        return SourceDocument(
            content=content,
            source=source,
            metadata=metadata,
        )

    @staticmethod
    def _record_error(report: LoadReport, message: str) -> None:
        """Record an error and log it."""
        report.errors.append(message)
        logger.warning(message)