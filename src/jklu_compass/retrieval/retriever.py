"""Source-aware semantic retrieval with controlled duplicate filtering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

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
    """Retrieve relevant chunks while controlling duplicate-source results."""

    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        vector_store: ChromaVectorStore | None = None,
        candidate_multiplier: int = 3,
        max_chunks_per_source: int = 2,
    ) -> None:
        """Initialize retrieval dependencies."""
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1.")

        if max_chunks_per_source < 1:
            raise ValueError("max_chunks_per_source must be at least 1.")

        self.encoder = encoder or EmbeddingEncoder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.candidate_multiplier = candidate_multiplier
        self.max_chunks_per_source = max_chunks_per_source

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks with source diversity and controlled repetition."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.encoder.encode(query.strip())
        candidate_k = top_k * self.candidate_multiplier

        records = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            top_k=candidate_k,
        )

        candidates: list[tuple[RetrievalResult, str]] = []

        for record in records:
            metadata = dict(record.get("metadata") or {})
            source = str(metadata.get("source") or "").strip()

            source_url = self._valid_http_url(metadata.get("source_url"))
            if source_url is None:
                source_url = self._valid_http_url(source)

            if source_url:
                source_key = self._canonical_url(source_url)
            elif source:
                source_key = f"path:{source}"
            else:
                source_key = f"id:{record['id']}"

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

            result = RetrievalResult(
                id=str(record["id"]),
                content=str(record.get("document") or ""),
                title=title,
                source_url=source_url,
                source=source,
                distance=float(record["distance"]),
                metadata=metadata,
            )

            candidates.append((result, source_key))

        results: list[RetrievalResult] = []
        source_counts: dict[str, int] = defaultdict(int)
        selected_ids: set[str] = set()

        # Pass 1: choose the best chunk from each distinct source.
        # Candidates are already ordered by similarity.
        for result, source_key in candidates:
            if source_counts[source_key] != 0:
                continue

            results.append(result)
            source_counts[source_key] += 1
            selected_ids.add(result.id)

            if len(results) >= top_k:
                return results

        # Pass 2: allow a second chunk from local-file sources only.
        # URL-based sources remain unique in the final result list.
        for result, source_key in candidates:
            if result.id in selected_ids:
                continue

            if source_key.startswith(("http://", "https://")):
                continue

            if source_counts[source_key] >= self.max_chunks_per_source:
                continue

            results.append(result)
            source_counts[source_key] += 1
            selected_ids.add(result.id)

            if len(results) >= top_k:
                break

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

    @staticmethod
    def _canonical_url(value: str) -> str:
        """Normalize URLs for duplicate-source detection."""
        parsed = urlparse(value)
        normalized_path = parsed.path.rstrip("/") or "/"

        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                normalized_path,
                "",
                "",
                "",
            )
        )
