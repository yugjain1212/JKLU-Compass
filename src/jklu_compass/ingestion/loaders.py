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

MARKDOWN_EXTENSIONS = {".md", ".markdown"}


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


def _parse_markdown_front_matter(
    content: str,
) -> tuple[str, dict[str, str]]:
    """Extract source URL and title from simple YAML front matter.

    Returns the Markdown body and any supported metadata fields.
    If front matter is absent or has no closing delimiter, the content
    is returned unchanged apart from surrounding whitespace.
    """
    lines = content.splitlines()

    if not lines or lines[0].strip() != "---":
        return content.strip(), {}

    closing_index: int | None = None

    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            closing_index = index
            break

    if closing_index is None:
        return content.strip(), {}

    metadata: dict[str, str] = {}

    for line in lines[1:closing_index]:
        key, separator, value = line.partition(":")

        if not separator:
            continue

        key = key.strip()
        value = value.strip()

        if not key or not value:
            continue

        # Remove matching quotes around simple YAML scalar values.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1].strip()

        if not value:
            continue

        if key in {"url", "source_url"}:
            metadata["source_url"] = value
        elif key == "title":
            metadata["title"] = value

    body = "\n".join(lines[closing_index + 1 :]).strip()

    return body, metadata


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

            if suffix in MARKDOWN_EXTENSIONS:
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
        """Load one Markdown file and extract its front-matter metadata."""
        markdown_path = Path(path).expanduser()

        if not markdown_path.is_file():
            raise FileNotFoundError(f"Markdown file does not exist: {markdown_path}")

        if markdown_path.suffix.lower() not in MARKDOWN_EXTENSIONS:
            raise ValueError(f"Not a Markdown file: {markdown_path}")

        report = LoadReport(files_seen=1, records_seen=1)

        try:
            content = markdown_path.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            self._record_error(
                report,
                f"{markdown_path}: Could not decode UTF-8 text: {exc}",
            )
            report.records_skipped += 1
            return report

        if not content:
            report.records_skipped += 1
            return report

        body, front_matter = _parse_markdown_front_matter(content)

        if not body:
            report.records_skipped += 1
            return report

        metadata: dict[str, Any] = {
            "source_type": "markdown",
            "file_name": markdown_path.name,
            "file_path": str(markdown_path.resolve()),
            **front_matter,
        }

        document = SourceDocument(
            content=body,
            source=str(markdown_path.resolve()),
            metadata=metadata,
        )

        report.documents.append(document)

        return report

    def load_markdown_directory(self, path: str | Path) -> LoadReport:
        """Recursively load Markdown files and extract front-matter metadata."""
        directory = Path(path).expanduser()

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        report = LoadReport()

        markdown_files = sorted(
            file_path
            for file_path in directory.rglob("*")
            if (file_path.is_file() and file_path.suffix.lower() in MARKDOWN_EXTENSIONS)
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

            body, front_matter = _parse_markdown_front_matter(content)

            if not body:
                report.records_skipped += 1
                continue

            metadata: dict[str, Any] = {
                "source_type": "markdown",
                "file_name": markdown_path.name,
                "file_path": str(markdown_path.resolve()),
                **front_matter,
            }

            report.documents.append(
                SourceDocument(
                    content=body,
                    source=str(markdown_path.resolve()),
                    metadata=metadata,
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
    def _record_error(
        report: LoadReport,
        message: str,
    ) -> None:
        """Record an error and log it."""
        report.errors.append(message)
        logger.warning(message)
