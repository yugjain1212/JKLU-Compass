"""Persistent ChromaDB vector store for JKLU Compass."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

DEFAULT_COLLECTION_NAME = "jklu_compass"
DEFAULT_PERSIST_DIRECTORY = "data/chroma"


class ChromaVectorStore:
    """Manage persistent document embeddings using ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize a persistent ChromaDB collection."""
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> int:
        """Insert or update document chunks and their embeddings."""
        if isinstance(ids, str) or isinstance(documents, str):
            raise TypeError("ids and documents must be sequences, not strings.")

        if len(ids) == 0:
            raise ValueError("At least one record is required.")

        if len(ids) != len(documents) or len(ids) != len(embeddings):
            raise ValueError("ids, documents, and embeddings must have equal lengths.")

        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique within a batch.")

        if any(not item.strip() for item in ids):
            raise ValueError("IDs must not be empty.")

        if any(not item.strip() for item in documents):
            raise ValueError("Documents must not be empty.")

        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas must have the same length as ids.")

        for vector in embeddings:
            if len(vector) == 0:
                raise ValueError("Embeddings must not be empty.")

        if metadatas is not None:
            for metadata in metadatas:
                for key, value in metadata.items():
                    if value is not None and not isinstance(value, (str, int, float, bool)):
                        raise ValueError(
                            f"Metadata value for {key!r} must be a "
                            "string, number, boolean, or None."
                        )

        self.collection.upsert(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(vector) for vector in embeddings],
            metadatas=(
                [dict(metadata) for metadata in metadatas] if metadatas is not None else None
            ),
        )

        return len(ids)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most similar stored documents for a query vector."""
        if len(query_embedding) == 0:
            raise ValueError("query_embedding must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "id": record_id,
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
            }
            for record_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
                strict=True,
            )
        ]

    def get_all_records(
        self,
        batch_size: int = 500,
    ) -> list[dict[str, Any]]:
        """Read all stored documents and metadata in paginated batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        records: list[dict[str, Any]] = []
        offset = 0

        while True:
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )

            ids = batch.get("ids", [])
            documents = batch.get("documents") or []
            metadatas = batch.get("metadatas") or []

            if not ids:
                break

            records.extend(
                {
                    "id": record_id,
                    "document": document or "",
                    "metadata": metadata or {},
                }
                for record_id, document, metadata in zip(
                    ids,
                    documents,
                    metadatas,
                    strict=True,
                )
            )

            offset += len(ids)

            if len(ids) < batch_size:
                break

        return records

    def count(self) -> int:
        """Return the number of stored records."""
        return self.collection.count()

    def get_ids_by_source(self, source: str) -> list[str]:
        """Return all stored chunk IDs belonging to one source."""
        if not source.strip():
            raise ValueError("source must not be empty.")

        records = self.collection.get(
            where={"source": source},
            include=[],
        )

        return records.get("ids", [])

    def delete_ids(self, ids: Sequence[str]) -> int:
        """Delete specific chunk IDs from the collection."""
        if isinstance(ids, str):
            raise TypeError("ids must be a sequence, not a string.")

        unique_ids = list(dict.fromkeys(ids))

        if not unique_ids:
            return 0

        self.collection.delete(ids=unique_ids)
        return len(unique_ids)

    def reset(self) -> None:
        """Delete all records from the collection."""
        self.collection.delete(ids=self.collection.get()["ids"])
