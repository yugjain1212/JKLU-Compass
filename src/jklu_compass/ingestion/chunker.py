"""Split source documents into overlapping chunks for retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from jklu_compass.ingestion.loaders import SourceDocument


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

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        if self.min_chunk_size < 1:
            raise ValueError("min_chunk_size must be at least 1.")

        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size cannot exceed chunk_size.")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A text chunk with source metadata and a stable identifier."""

    chunk_id: str
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, object] = field(default_factory=dict)


class DocumentChunker:
    """Split documents into overlapping, metadata-preserving chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialize the chunker."""
        self.config = config or ChunkingConfig()

    def chunk_document(self, document: SourceDocument) -> list[DocumentChunk]:
        """Split one document into chunks.

        Args:
            document: Normalized source document.

        Returns:
            Ordered chunks containing the source text and metadata.
        """
        text = document.content.strip()

        if not text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.config.chunk_size, text_length)

            # Prefer a nearby word boundary instead of cutting a word in half.
            if end < text_length:
                boundary = text.rfind(" ", start, end)

                if boundary > start:
                    end = boundary

            content = text[start:end].strip()

            if content and (
                len(content) >= self.config.min_chunk_size or not chunks
            ):
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
                chunk_index += 1

            if end >= text_length:
                break

            # Move forward while retaining the configured overlap.
            next_start = end - self.config.chunk_overlap

            # Ensure progress even if unusual whitespace or boundaries occur.
            if next_start <= start:
                next_start = start + self.config.chunk_size

            start = next_start

        return chunks

    def chunk_documents(
        self,
        documents: list[SourceDocument],
    ) -> list[DocumentChunk]:
        """Split multiple documents into a single ordered chunk list."""
        chunks: list[DocumentChunk] = []

        for document in documents:
            chunks.extend(self.chunk_document(document))

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