"""Source-aware semantic retrieval with controlled duplicate filtering."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

from jklu_compass.embeddings.encoder import EmbeddingEncoder
from jklu_compass.retrieval.bm25_retriever import BM25Retriever
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
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = 60,
        max_chunks_per_pdf: int = 6,
    ) -> None:
        """Initialize dense and lexical retrieval dependencies."""
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1.")

        if max_chunks_per_source < 1:
            raise ValueError("max_chunks_per_source must be at least 1.")

        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        if max_chunks_per_pdf < 1:
            raise ValueError("max_chunks_per_pdf must be at least 1.")

        self.encoder = encoder or EmbeddingEncoder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.candidate_multiplier = candidate_multiplier
        self.max_chunks_per_source = max_chunks_per_source
        self.rrf_k = rrf_k
        self.max_chunks_per_pdf = max_chunks_per_pdf

        # Initialize BM25 once, only when the store supports corpus retrieval.
        supports_corpus_loading = callable(
            getattr(type(self.vector_store), "get_all_records", None)
        )

        self.bm25_retriever = bm25_retriever

        if self.bm25_retriever is None and supports_corpus_loading:
            self.bm25_retriever = BM25Retriever(self.vector_store)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve and rank relevant chunks using dense and lexical search."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = query.strip()
        candidate_k = top_k * self.candidate_multiplier

        # 1. Dense retrieval
        query_embedding = self.encoder.encode(query)
        dense_records = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            top_k=candidate_k,
        )

        results_by_id: dict[str, RetrievalResult] = {}
        source_keys: dict[str, str] = {}

        for record in dense_records:
            result, source_key = self._build_result(record)
            results_by_id[result.id] = result
            source_keys[result.id] = source_key

        # 2. Hybrid retrieval using BM25 and Reciprocal Rank Fusion
        if self.bm25_retriever is not None:
            lexical_results = self.bm25_retriever.retrieve(
                query,
                top_k=candidate_k,
            )

            lexical_records = {str(record["id"]): record for record in self.bm25_retriever.records}

            rrf_scores: dict[str, float] = defaultdict(float)

            for rank, record in enumerate(dense_records, start=1):
                record_id = str(record["id"])
                rrf_scores[record_id] += 1.0 / (self.rrf_k + rank)

            for rank, lexical_result in enumerate(
                lexical_results,
                start=1,
            ):
                record_id = lexical_result.id
                rrf_scores[record_id] += 1.0 / (self.rrf_k + rank)

                if record_id not in results_by_id:
                    record = lexical_records.get(record_id)

                    if record is None:
                        continue

                    result, source_key = self._build_result(
                        record,
                        distance=1.0,
                    )
                    results_by_id[record_id] = result
                    source_keys[record_id] = source_key

            ranked_ids = sorted(
                rrf_scores,
                key=lambda record_id: (
                    -rrf_scores[record_id],
                    results_by_id[record_id].distance,
                ),
            )

            candidates = [
                (results_by_id[record_id], source_keys[record_id])
                for record_id in ranked_ids
                if record_id in results_by_id
            ]

        else:
            # Dense-only fallback
            query_terms = self._query_terms(query)

            candidates = [(result, source_keys[result.id]) for result in results_by_id.values()]

            candidates.sort(
                key=lambda item: (
                    self._relevance_score(query_terms, item[0]),
                    -item[0].distance,
                ),
                reverse=True,
            )

        # 3. Structured programme retrieval
        # Programme sections from the same brochure must not be discarded
        # by the generic source-diversity filter.
        if self.bm25_retriever is not None and self._is_btech_programme_query(query):
            get_programmes = getattr(
                self.bm25_retriever,
                "get_btech_programme_records",
                None,
            )

            if callable(get_programmes):
                programme_records = get_programmes()

                programme_results = [
                    self._build_result(record, distance=1.0)[0] for record in programme_records
                ]

                if programme_results:
                    return programme_results[:top_k]

        # 4. Generic source-diversity selection
        results: list[RetrievalResult] = []
        source_counts: dict[str, int] = defaultdict(int)
        selected_ids: set[str] = set()

        # First pass: prioritize different sources.
        # Reserve some slots for additional relevant chunks.
        reserved_slots = min(2, top_k // 3)
        diversity_target = max(1, top_k - reserved_slots)

        for result, source_key in candidates:
            if source_counts[source_key] != 0:
                continue

            results.append(result)
            source_counts[source_key] += 1
            selected_ids.add(result.id)

            if len(results) >= diversity_target:
                break

        # Second pass: allow additional chunks from PDFs and local files.
        # Keep HTML webpages limited to one chunk per canonical source.
        for result, source_key in candidates:
            if len(results) >= top_k:
                break

            if result.id in selected_ids:
                continue

            is_pdf = self._is_pdf_source(result)

            if source_key.startswith(("http://", "https://")) and not is_pdf:
                continue

            source_limit = self.max_chunks_per_pdf if is_pdf else self.max_chunks_per_source

            if source_counts[source_key] >= source_limit:
                continue

            results.append(result)
            source_counts[source_key] += 1
            selected_ids.add(result.id)

        return results

    @staticmethod
    def _is_btech_programme_query(query: str) -> bool:
        """Identify queries asking for B.Tech programmes or specialisations."""
        normalized = query.lower()

        mentions_btech = bool(
            re.search(
                r"\bb\.?\s*tech\b|\bbachelor\s+of\s+technology\b",
                normalized,
            )
        )

        asks_for_list = any(
            term in normalized
            for term in (
                "specialisation",
                "specialization",
                "programme",
                "program",
                "available",
                "list",
                "offer",
            )
        )

        return mentions_btech and asks_for_list

    @staticmethod
    def _is_pdf_source(result: RetrievalResult) -> bool:
        """Identify PDF sources using their URL or source path."""
        for value in (result.source_url, result.source):
            if not isinstance(value, str) or not value.strip():
                continue

            if urlparse(value.strip()).path.lower().endswith(".pdf"):
                return True

        return False

    def _build_result(
        self,
        record: dict[str, Any],
        distance: float | None = None,
    ) -> tuple[RetrievalResult, str]:
        """Convert a stored record into a result with source identity."""
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
            distance=(float(record["distance"]) if distance is None else distance),
            metadata=metadata,
        )

        return result, source_key

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

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        """Extract meaningful terms from a query."""
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "does",
            "for",
            "how",
            "in",
            "is",
            "of",
            "on",
            "or",
            "the",
            "to",
            "what",
            "which",
            "who",
            "with",
        }

        return {
            term
            for term in re.findall(r"[a-z0-9]+", query.lower())
            if len(term) > 1 and term not in stop_words
        }

    @classmethod
    def _relevance_score(
        cls,
        query_terms: set[str],
        result: RetrievalResult,
    ) -> float:
        """Calculate a lightweight metadata-aware relevance score."""
        if not query_terms:
            return 0.0

        section_title = str(result.metadata.get("section_title") or "").lower()
        heading_path = str(result.metadata.get("heading_path") or "").lower()
        content = result.content.lower()

        section_terms = set(re.findall(r"[a-z0-9]+", section_title))
        heading_terms = set(re.findall(r"[a-z0-9]+", heading_path))
        content_terms = set(re.findall(r"[a-z0-9]+", content))

        section_match = len(query_terms & section_terms) / len(query_terms)
        heading_match = len(query_terms & heading_terms) / len(query_terms)
        content_match = len(query_terms & content_terms) / len(query_terms)

        return 0.60 * section_match + 0.30 * heading_match + 0.10 * content_match
