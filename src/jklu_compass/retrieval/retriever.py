"""Source-aware semantic retrieval for JKLU Compass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from jklu_compass.embeddings.encoder import EmbeddingEncoder
from jklu_compass.vectorstore.chroma_store import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A retrieved document chunk with source attribution."""

    id: str
    content: str
    title: str
    source_url: str | None
    source: str
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentRetriever:
    """Retrieve relevant document chunks with their original source metadata."""

    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        """Initialize retrieval dependencies."""
        self.encoder = encoder or EmbeddingEncoder()
        self.vector_store = vector_store or ChromaVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a non-empty question."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.encoder.encode(query.strip())

        records = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            top_k=top_k,
        )

        results: list[RetrievalResult] = []

        for record in records:
            metadata = dict(record.get("metadata") or {})
            source = str(metadata.get("source") or "")
            candidate_url = metadata.get("source_url")

            if not candidate_url and source:
                candidate_url = source

            source_url = self._valid_http_url(candidate_url)

            raw_title = metadata.get("title")
            raw_file_name = metadata.get("file_name")

            invalid_values = {"", "undefined", "none", "null"}

            if raw_title and str(raw_title).strip().lower() not in invalid_values:
                title = str(raw_title).strip()
            elif raw_file_name and str(raw_file_name).strip().lower() not in invalid_values:
                title = str(raw_file_name).strip()
            elif source:
                title = source
            else:
                title = "Unknown source"

            results.append(
                RetrievalResult(
                    id=str(record["id"]),
                    content=str(record.get("document") or ""),
                    title=title,
                    source_url=source_url,
                    source=source,
                    distance=float(record["distance"]),
                    metadata=metadata,
                )
            )

        return results

    @staticmethod
    def _valid_http_url(value: Any) -> str | None:
        """Return an HTTP(S) URL, excluding local paths and invalid values."""
        if not isinstance(value, str) or not value.strip():
            return None

        candidate = value.strip()
        parsed = urlparse(candidate)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None

        return candidate
