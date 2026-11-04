"""Lexical retrieval using BM25 over the existing ChromaDB corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from jklu_compass.vectorstore.chroma_store import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class BM25Result:
    """A document ranked by BM25 relevance."""

    id: str
    score: float


class BM25Retriever:
    """Build an in-memory BM25 index from existing stored documents."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        batch_size: int = 500,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.vector_store = vector_store
        self.records = vector_store.get_all_records(batch_size=batch_size)

        self.document_ids: list[str] = []
        tokenized_corpus: list[list[str]] = []

        for record in self.records:
            metadata: dict[str, Any] = record.get("metadata") or {}

            searchable_text = " ".join(
                str(value)
                for value in (
                    metadata.get("title", ""),
                    metadata.get("section_title", ""),
                    metadata.get("heading_path", ""),
                    record.get("document", ""),
                )
                if value
            )

            tokens = self._tokenize(searchable_text)

            if not tokens:
                continue

            self.document_ids.append(str(record["id"]))
            tokenized_corpus.append(tokens)

        self.index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[BM25Result]:
        """Return the highest-scoring lexical matches."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = self._tokenize(query)

        if self.index is None or not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            results.append(
                BM25Result(
                    id=self.document_ids[index],
                    score=score,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def get_btech_programme_records(self) -> list[dict[str, Any]]:
        """Return distinct programme sections under the B.Tech heading."""
        parent_heading = "B.TECH BACHELOR OF TECHNOLOGY"

        programme_records: dict[str, dict[str, Any]] = {}

        for record in self.records:
            metadata: dict[str, Any] = record.get("metadata") or {}
            heading_path = metadata.get("heading_path") or []

            if isinstance(heading_path, str):
                try:
                    import ast

                    parsed = ast.literal_eval(heading_path)
                    heading_path = parsed if isinstance(parsed, list) else [heading_path]
                except (ValueError, SyntaxError):
                    heading_path = [heading_path]

            if not isinstance(heading_path, list):
                continue

            normalized_path = [str(heading).lstrip("#").strip().upper() for heading in heading_path]

            if parent_heading not in normalized_path:
                continue

            if int(metadata.get("heading_level") or 0) < 5:
                continue

            programme_heading = normalized_path[-1]

            if not programme_heading or programme_heading == parent_heading:
                continue

            programme_records.setdefault(programme_heading, record)

        return sorted(
            programme_records.values(),
            key=lambda record: int((record.get("metadata") or {}).get("chunk_index", 0)),
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into searchable alphanumeric tokens."""
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "does",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "what",
            "which",
            "who",
            "with",
        }

        return [
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 1 and token not in stop_words
        ]
