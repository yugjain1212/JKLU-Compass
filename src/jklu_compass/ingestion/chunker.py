
"""Split source documents into Markdown-aware overlapping chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from jklu_compass.ingestion.loaders import SourceDocument


_HEADING_PATTERN = re.compile(
    r"^\s{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$"
)
_FENCE_PATTERN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration for text chunking."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    min_chunk_size: int = 50

    def __post_init__(self) -> None:
        """Validate chunking parameters."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if self.min_chunk_size < 1:
            raise ValueError("min_chunk_size must be at least 1.")

        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size cannot exceed chunk_size.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A text chunk with source metadata and a stable identifier."""

    chunk_id: str
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, object] = field(default_factory=dict)


class DocumentChunker:
    """Split documents while preserving Markdown section context."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize the chunker."""
        self.config = config or ChunkingConfig()

    def chunk_document(self, document: SourceDocument) -> list[DocumentChunk]:
        """Split one document into ordered, metadata-preserving chunks."""
        text = document.content.strip()

        if not text:
            return []

        sections = self._split_markdown_sections(text)
        chunks: list[DocumentChunk] = []

        for heading_path, body in sections:
            section_chunks = self._chunk_section(heading_path, body)

            for content in section_chunks:
                if not content:
                    continue

                chunk_index = len(chunks)
                chunk_id = self._make_chunk_id(
                    source=document.source,
                    chunk_index=chunk_index,
                    content=content,
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        content=content,
                        source=document.source,
                        chunk_index=chunk_index,
                        metadata=dict(document.metadata),
                    )
                )

        return chunks

    def chunk_documents(
        self,
        documents: list[SourceDocument],
    ) -> list[DocumentChunk]:
        """Split multiple documents into one ordered chunk list."""
        chunks: list[DocumentChunk] = []

        for document in documents:
            chunks.extend(self.chunk_document(document))

        return chunks

    @staticmethod
    def _split_markdown_sections(
        text: str,
    ) -> list[tuple[tuple[str, ...], str]]:
        """Split text at Markdown headings while retaining parent headings."""
        sections: list[tuple[tuple[str, ...], str]] = []
        heading_stack: list[str] = []
        current_heading_path: tuple[str, ...] = ()
        current_lines: list[str] = []

        fence_marker: str | None = None

        for line in text.splitlines():
            fence_match = _FENCE_PATTERN.match(line)

            if fence_match:
                marker = fence_match.group(1)

                if fence_marker is None:
                    fence_marker = marker
                elif (
                    marker[0] == fence_marker[0]
                    and len(marker) >= len(fence_marker)
                ):
                    fence_marker = None

                current_lines.append(line)
                continue

            if fence_marker is not None:
                current_lines.append(line)
                continue

            heading_match = _HEADING_PATTERN.match(line)

            if not heading_match:
                current_lines.append(line)
                continue

            body = "\n".join(current_lines).strip()

            if body:
                sections.append((current_heading_path, body))

            level = len(heading_match.group(1))
            heading_title = heading_match.group(2).strip().rstrip("#").strip()

            heading_stack = heading_stack[: level - 1]
            heading_stack.append(f"{'#' * level} {heading_title}")

            current_heading_path = tuple(heading_stack)
            current_lines = []

        final_body = "\n".join(current_lines).strip()

        if final_body or current_heading_path:
            sections.append((current_heading_path, final_body))

        if not sections:
            sections.append(((), text))

        return sections

    def _chunk_section(
        self,
        heading_path: tuple[str, ...],
        body: str,
    ) -> list[str]:
        """Chunk one section and repeat its heading context where possible."""
        context = "\n".join(heading_path)

        if context and len(context) > self.config.chunk_size // 2:
            # Keep the closest heading when the full hierarchy is too long.
            context = heading_path[-1][
                : self.config.chunk_size // 2
            ]

        if not body.strip():
            return [context[: self.config.chunk_size]] if context else []

        prefix = f"{context}\n\n" if context else ""
        body_budget = self.config.chunk_size - len(prefix)

        if body_budget < 1:
            prefix = ""
            body_budget = self.config.chunk_size

        body_parts = self._split_text(body.strip(), body_budget)

        return [
            f"{prefix}{part}".strip() if prefix else part.strip()
            for part in body_parts
            if part.strip()
        ]

    def _split_text(self, text: str, max_chars: int) -> list[str]:
        """Split text near natural boundaries without excessive overlap."""
        text = text.strip()

        if not text:
            return []

        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        effective_overlap = min(
            self.config.chunk_overlap,
            max_chars - 1,
        )

        while start < text_length:
            hard_end = min(start + max_chars, text_length)
            end = hard_end

            if hard_end < text_length:
                minimum_boundary = start + max(
                    effective_overlap + 1,
                    min(self.config.min_chunk_size, max_chars),
                )

                for separator in ("\n\n", "\n", " "):
                    boundary = text.rfind(
                        separator,
                        minimum_boundary,
                        hard_end,
                    )

                    if boundary > start:
                        end = boundary
                        break

            if end <= start:
                end = hard_end

            piece = text[start:end].strip()

            if piece:
                chunks.append(piece)

            if end >= text_length:
                break

            next_start = end - effective_overlap

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    @staticmethod
    def _make_chunk_id(
        source: str,
        chunk_index: int,
        content: str,
    ) -> str:
        """Generate a deterministic ID for a chunk."""
        fingerprint = hashlib.sha256(
            f"{source}\0{chunk_index}\0{content}".encode()
        ).hexdigest()[:16]

        return f"chunk_{fingerprint}"
