"""Indexing pipeline for JKLU Compass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jklu_compass.embeddings.encoder import EmbeddingEncoder
from jklu_compass.ingestion.pipeline import IngestionPipeline
from jklu_compass.vectorstore.chroma_store import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class IndexingReport:
    """Statistics describing an indexing run."""

    documents_loaded: int
    chunks_created: int
    chunks_indexed: int
    vector_store_count: int


class DocumentIndexer:
    """Coordinate document ingestion, embedding, and vector storage."""

    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        vector_store: ChromaVectorStore | None = None,
        ingestion_pipeline: IngestionPipeline | None = None,
        batch_size: int = 64,
    ) -> None:
        """Initialize the indexer and its dependencies."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.encoder = encoder or EmbeddingEncoder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.ingestion_pipeline = ingestion_pipeline or IngestionPipeline()
        self.batch_size = batch_size

    def index_directory(self, source_path: str | Path) -> IndexingReport:
        """Process a source directory and persist its document chunks."""
        ingestion_report = self.ingestion_pipeline.run(source_path)
        chunks = ingestion_report.chunks

        indexed_count = 0

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [chunk.content for chunk in batch]
            embeddings = self.encoder.encode_batch(texts)

            self.vector_store.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[
                   self._prepare_metadata(chunk.metadata, chunk.source, chunk.chunk_index)
                   for chunk in batch
                ],
            )

            indexed_count += len(batch)

        return IndexingReport(
            documents_loaded=ingestion_report.documents_loaded,
            chunks_created=len(chunks),
            chunks_indexed=indexed_count,
            vector_store_count=self.vector_store.count(),
        )

    @staticmethod
    def _prepare_metadata(
        metadata: dict[str, Any],
        source: str,
        chunk_index: int,
    ) -> dict[str, str | int | float | bool]:
        """Convert chunk metadata into ChromaDB-compatible scalar values."""
        prepared: dict[str, str | int | float | bool] = {
            "source": source,
            "chunk_index": chunk_index,
        }

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value

            elif isinstance(value, (list, dict)):
                prepared[key] = str(value)

            else:
                prepared[key] = str(value)

        return prepared
