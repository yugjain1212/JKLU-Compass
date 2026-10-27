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

    def count(self) -> int:
        """Return the number of stored records."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all records from the collection."""
        self.collection.delete(ids=self.collection.get()["ids"])
