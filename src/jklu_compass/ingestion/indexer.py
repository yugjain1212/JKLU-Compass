"""Indexing pipeline for JKLU Compass."""

from __future__ import annotations

from collections import defaultdict
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
    chunks_deleted: int = 0
    sources_synced: int = 0


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

    def index_directory(
        self,
        source_path: str | Path,
        *,
        sync: bool = False,
    ) -> IndexingReport:
        """Process documents and persist their chunks.

        When sync=True, remove obsolete chunks belonging to successfully
        processed sources. Synchronization fails closed if ingestion reports
        skipped documents or errors.
        """
        ingestion_report = self.ingestion_pipeline.run(source_path)
        chunks = ingestion_report.chunks

        if sync and (ingestion_report.errors or ingestion_report.documents_skipped):
            raise RuntimeError(
                "Synchronization aborted: ingestion reported errors or "
                "skipped documents. Existing vectors were not deleted."
            )

        if sync and not chunks:
            raise RuntimeError(
                "Synchronization aborted: no chunks were created. "
                "Existing vectors were not deleted."
            )

        indexed_count = 0
        deleted_count = 0
        synced_sources = 0

        if not sync:
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                self._index_batch(batch)
                indexed_count += len(batch)

        else:
            chunks_by_source = defaultdict(list)

            for chunk in chunks:
                chunks_by_source[chunk.source].append(chunk)

            for source, source_chunks in chunks_by_source.items():
                # Embed all chunks for this source before modifying its records.
                embedded_batches = []

                for start in range(0, len(source_chunks), self.batch_size):
                    batch = source_chunks[start : start + self.batch_size]
                    texts = [chunk.content for chunk in batch]
                    embeddings = self.encoder.encode_batch(texts)

                    embedded_batches.append((batch, embeddings.tolist()))

                # Upsert every batch before deleting any old records.
                for batch, embeddings in embedded_batches:
                    self._store_batch(batch, embeddings)
                    indexed_count += len(batch)

                # Only now is it safe to remove obsolete IDs for this source.
                current_ids = {chunk.chunk_id for chunk in source_chunks}

                existing_ids = set(self.vector_store.get_ids_by_source(source))

                stale_ids = existing_ids - current_ids

                deleted_count += self.vector_store.delete_ids(list(stale_ids))

                synced_sources += 1

        return IndexingReport(
            documents_loaded=ingestion_report.documents_loaded,
            chunks_created=len(chunks),
            chunks_indexed=indexed_count,
            vector_store_count=self.vector_store.count(),
            chunks_deleted=deleted_count,
            sources_synced=synced_sources,
        )

    def _index_batch(self, batch: list[Any]) -> None:
        """Embed and store one batch of chunks."""
        texts = [chunk.content for chunk in batch]
        embeddings = self.encoder.encode_batch(texts)

        self._store_batch(batch, embeddings.tolist())

    def _store_batch(
        self,
        batch: list[Any],
        embeddings: list[list[float]],
    ) -> None:
        """Persist an already embedded batch."""
        texts = [chunk.content for chunk in batch]

        self.vector_store.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                self._prepare_metadata(
                    chunk.metadata,
                    chunk.source,
                    chunk.chunk_index,
                )
                for chunk in batch
            ],
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
