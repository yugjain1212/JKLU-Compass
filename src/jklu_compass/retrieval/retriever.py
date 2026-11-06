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
        trace: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve and rank relevant chunks using dense and lexical search."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query = query.strip()
        candidate_k = top_k * self.candidate_multiplier

        if trace is not None:
            trace.clear()
            trace["candidate_k"] = candidate_k
            trace["selection_mode"] = "source_diversity"

        query_terms = self._query_terms(query)

        # 1. Dense retrieval
        query_embedding = self.encoder.encode(query)
        dense_records = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            top_k=candidate_k,
        )

        results_by_id: dict[str, RetrievalResult] = {}
        source_keys: dict[str, str] = {}
        dense_rank_by_id: dict[str, int] = {}

        for rank, record in enumerate(dense_records, start=1):
            result, source_key = self._build_result(record)
            if result.id in dense_rank_by_id:
                continue

            dense_rank_by_id[result.id] = rank
            results_by_id[result.id] = result
            source_keys[result.id] = source_key

        ranked_candidate_details: list[dict[str, Any]] = []
        bm25_rank_by_id: dict[str, int] = {}
        missing_record_ids: list[str] = []

        # 2. Hybrid retrieval using BM25 and Reciprocal Rank Fusion
        if self.bm25_retriever is not None:
            lexical_results = self.bm25_retriever.retrieve(
                query,
                top_k=candidate_k,
            )

            lexical_records = {str(record["id"]): record for record in self.bm25_retriever.records}

            rrf_scores: dict[str, float] = defaultdict(float)

            for record_id, rank in dense_rank_by_id.items():
                rrf_scores[record_id] += 1.0 / (self.rrf_k + rank)

            for rank, lexical_result in enumerate(
                lexical_results,
                start=1,
            ):
                record_id = lexical_result.id
                if record_id in bm25_rank_by_id:
                    continue

                bm25_rank_by_id[record_id] = rank
                rrf_scores[record_id] += 1.0 / (self.rrf_k + rank)

                if record_id not in results_by_id:
                    record = lexical_records.get(record_id)

                    if record is None:
                        missing_record_ids.append(record_id)
                        continue

                    result, source_key = self._build_result(
                        record,
                        distance=1.0,
                    )
                    results_by_id[record_id] = result
                    source_keys[record_id] = source_key

            ranked_ids = sorted(
                (record_id for record_id in rrf_scores if record_id in results_by_id),
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

            ranked_candidate_details = [
                {
                    "rank": rank,
                    "id": record_id,
                    "dense_rank": dense_rank_by_id.get(record_id),
                    "bm25_rank": bm25_rank_by_id.get(record_id),
                    "rrf_score": rrf_scores[record_id],
                    "source_key": source_keys[record_id],
                    "result": results_by_id[record_id],
                }
                for rank, record_id in enumerate(ranked_ids, start=1)
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

            ranked_candidate_details = [
                {
                    "rank": rank,
                    "id": result.id,
                    "dense_rank": dense_rank_by_id.get(result.id),
                    "bm25_rank": None,
                    "rrf_score": None,
                    "source_key": source_key,
                    "result": result,
                }
                for rank, (result, source_key) in enumerate(candidates, start=1)
            ]

        if trace is not None:
            trace["dense_candidates"] = [
                {"rank": rank, "id": record_id} for record_id, rank in dense_rank_by_id.items()
            ]
            trace["bm25_candidates"] = [
                {"rank": rank, "id": record_id} for record_id, rank in bm25_rank_by_id.items()
            ]
            trace["missing_record_ids"] = missing_record_ids
            trace["ranked_candidates"] = ranked_candidate_details
            trace["selection_reasons"] = {}

        # 3. Structured programme retrieval
        # Programme sections from the same brochure must not be discarded
        # by the generic source-diversity filter.
        if self.bm25_retriever is not None and self.is_btech_programme_list_query(query):
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
                    selected_programme_results = programme_results[:top_k]
                    if trace is not None:
                        trace["selection_mode"] = "structured_programme"
                        trace["selected_ids"] = [result.id for result in selected_programme_results]
                        trace["selection_reasons"] = {
                            candidate["id"]: (
                                "selected_structured_programme"
                                if candidate["id"] in trace["selected_ids"]
                                else "bypassed_by_structured_programme"
                            )
                            for candidate in ranked_candidate_details
                        }
                        for candidate in ranked_candidate_details:
                            candidate["selection_reason"] = trace["selection_reasons"][
                                candidate["id"]
                            ]
                    return selected_programme_results

        # 4. Generic source-diversity selection
        results: list[RetrievalResult] = []
        source_counts: dict[str, int] = defaultdict(int)
        source_query_terms: dict[str, set[str]] = defaultdict(set)
        selected_ids: set[str] = set()
        selection_reasons: dict[str, str] = {}

        priority_evidence = next(
            (
                (result, source_key, reason)
                for result, source_key in candidates
                if (reason := self._priority_evidence_reason(query, result)) is not None
            ),
            None,
        )
        if priority_evidence is not None:
            result, source_key, reason = priority_evidence
            results.append(result)
            source_counts[source_key] += 1
            source_query_terms[source_key].update(self._matching_query_terms(query_terms, result))
            selected_ids.add(result.id)
            selection_reasons[result.id] = reason
            if trace is not None:
                trace["selection_mode"] = "answer_specific_evidence_priority"

        # First pass: prioritize different sources.
        # Reserve some slots for additional relevant chunks.
        reserved_slots = min(2, top_k // 3)
        diversity_target = max(1, top_k - reserved_slots)

        for result, source_key in candidates:
            if source_counts[source_key] != 0:
                selection_reasons.setdefault(result.id, "deferred_same_source")
                continue

            results.append(result)
            source_counts[source_key] += 1
            source_query_terms[source_key].update(self._matching_query_terms(query_terms, result))
            selected_ids.add(result.id)
            selection_reasons[result.id] = "selected_for_diversity"

            if len(results) >= diversity_target:
                break

        # Fill reserved slots with candidates that add query terms to their source first.
        while len(results) < top_k:
            eligible: list[tuple[int, int, RetrievalResult, str, bool, set[str]]] = []

            for rank, (result, source_key) in enumerate(candidates):
                if result.id in selected_ids:
                    continue

                is_pdf = self._is_pdf_source(result)
                source_limit = self.max_chunks_per_pdf if is_pdf else self.max_chunks_per_source
                if source_counts[source_key] >= source_limit:
                    selection_reasons[result.id] = "pdf_source_limit" if is_pdf else "source_limit"
                    continue

                matching_terms = self._matching_query_terms(query_terms, result)
                new_terms = matching_terms - source_query_terms[source_key]
                is_html = source_key.startswith(("http://", "https://")) and not is_pdf

                if is_html and source_counts[source_key] and not new_terms:
                    selection_reasons[result.id] = "html_no_new_query_terms"
                    continue

                eligible.append((len(new_terms), rank, result, source_key, is_pdf, matching_terms))

            if not eligible:
                break

            _, _, result, source_key, _, matching_terms = min(
                eligible,
                key=lambda item: (item[1] > 2, -item[0], item[1]),
            )
            results.append(result)
            source_counts[source_key] += 1
            source_query_terms[source_key].update(matching_terms)
            selected_ids.add(result.id)
            selection_reasons[result.id] = "selected_for_fill"

        if trace is not None:
            for candidate in ranked_candidate_details:
                candidate_id = candidate["id"]
                if (
                    candidate_id not in selected_ids
                    and selection_reasons.get(candidate_id) == "deferred_same_source"
                ):
                    selection_reasons[candidate_id] = (
                        "not_selected_top_k"
                        if len(results) >= top_k
                        else "not_selected_by_source_selection"
                    )
                candidate["selection_reason"] = selection_reasons.setdefault(
                    candidate_id,
                    "not_selected_top_k",
                )

            trace["selection_reasons"] = selection_reasons
            trace["selected_ids"] = [result.id for result in results]

        return results

    @classmethod
    def _priority_evidence_reason(cls, query: str, result: RetrievalResult) -> str | None:
        """Return a narrow evidence-priority reason for known list and fee intents."""
        if cls._is_degree_fee_evidence(query, result):
            return "selected_degree_fee_evidence"

        if cls._is_comprehensive_lab_list(query, result):
            return "selected_comprehensive_lab_list"

        if cls._is_hostel_room_type_evidence(query, result):
            return "selected_hostel_room_type_evidence"

        return None

    @classmethod
    def _is_degree_fee_evidence(cls, query: str, result: RetrievalResult) -> bool:
        """Prioritize a fee chunk matching the requested degree and academic year."""
        if not re.search(r"\b(?:fees?|tuition|costs?|charges?)\b", query, re.I):
            return False
        if re.search(
            r"\b(?:hostel|accommodation|transport|mess|registration|application)\s+fees?\b",
            query,
            re.I,
        ):
            return False

        degree_patterns = {
            "btech": r"\b(?:b\.?\s*tech|bachelor\s+of\s+technology)\b",
            "bdes": r"\b(?:b\.?\s*des|bachelor\s+of\s+design)\b",
            "mtech": r"\b(?:m\.?\s*tech|master\s+of\s+technology)\b",
            "mdes": r"\b(?:m\.?\s*des|master\s+of\s+design)\b",
            "bba": r"\b(?:bba|bachelor\s+of\s+business(?:\s+administration)?)\b",
            "mba": r"\b(?:mba|master\s+of\s+business(?:\s+administration)?)\b",
        }
        requested_degrees = {
            degree for degree, pattern in degree_patterns.items() if re.search(pattern, query, re.I)
        }
        if len(requested_degrees) != 1:
            return False

        searchable_text = " ".join(
            str(value)
            for value in (
                result.content,
                result.metadata.get("section_title", ""),
                result.metadata.get("heading_path", ""),
            )
            if value
        )
        target_degree = next(iter(requested_degrees))
        if not re.search(degree_patterns[target_degree], searchable_text, re.I):
            return False
        if not re.search(r"\b(?:academic|technology|tuition)\s+fees?\b", searchable_text, re.I):
            return False
        if not re.search(r"\d[\d,]{3,}", result.content):
            return False

        requested_years = cls._academic_years(query)
        candidate_years = cls._academic_years(searchable_text)
        return not requested_years or bool(requested_years & candidate_years)

    @staticmethod
    def _is_comprehensive_lab_list(query: str, result: RetrievalResult) -> bool:
        """Prefer a chunk that enumerates several labs for a lab-list question."""
        if not re.search(r"\b(?:which|what|list|name|all)\b", query, re.I):
            return False
        if not re.search(r"\blabs\b|\blaborator(?:ies)\b", query, re.I):
            return False
        lab_mentions = re.findall(r"\b(?:lab|laborator(?:y|ies))\b", result.content, re.I)
        return len(lab_mentions) >= 5

    @staticmethod
    def _is_hostel_room_type_evidence(query: str, result: RetrievalResult) -> bool:
        """Prefer explicit occupancy and AC options for hostel room-type questions."""
        if not re.search(r"\bhostel\b", query, re.I):
            return False
        if not re.search(r"\broom(?:s)?\b", query, re.I) or not re.search(
            r"\btypes?\b", query, re.I
        ):
            return False
        content = result.content.casefold()
        has_occupancy = bool(
            re.search(r"\b(?:single|double|triple)\b.{0,40}\boccupancy\b", content)
        )
        has_ac_options = bool(re.search(r"\b(?:non[-\s]*ac|ac\s+and\s+non[-\s]*ac)\b", content))
        return has_occupancy and has_ac_options

    @staticmethod
    def _academic_years(text: str) -> set[str]:
        """Normalize year ranges such as 2026–27 and 2026/2027."""
        return {
            f"{start}-{end[-2:]}"
            for start, end in re.findall(r"\b(20\d{2})\s*[-–—/]\s*(\d{2,4})\b", text)
        }

    @staticmethod
    def is_btech_programme_list_query(query: str) -> bool:
        """Identify explicit requests for B.Tech course or specialisation lists."""
        normalized = query.casefold().replace("–", "-").replace("—", "-")
        mentions_btech = bool(
            re.search(
                r"\bb\.?\s*tech\b|\bbachelor\s+of\s+technology\b",
                normalized,
            )
        )
        asks_about_mobility = bool(
            re.search(
                r"\b(?:semester|exchange|mobility|opportunit(?:y|ies)|study abroad)\b",
                normalized,
            )
        )
        names_a_programme_list = bool(
            re.search(
                r"\b(?:speciali[sz]ations?|programmes?|programs?|courses?|degrees?|majors?)\b",
                normalized,
            )
        )

        return mentions_btech and names_a_programme_list and not asks_about_mobility

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

    @staticmethod
    def _matching_query_terms(
        query_terms: set[str],
        result: RetrievalResult,
    ) -> set[str]:
        """Return query terms found in a result's content or structural metadata."""
        metadata = result.metadata
        searchable_text = " ".join(
            str(value)
            for value in (
                result.title,
                metadata.get("section_title", ""),
                metadata.get("heading_path", ""),
                result.content,
            )
            if value
        ).lower()
        result_terms = set(re.findall(r"[a-z0-9]+", searchable_text))
        return query_terms & result_terms

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
