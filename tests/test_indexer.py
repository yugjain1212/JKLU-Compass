"""Tests for the JKLU Compass document indexer."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from jklu_compass.ingestion.indexer import DocumentIndexer


@pytest.fixture
def mock_components() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Create mocked dependencies for indexer tests."""
    encoder = MagicMock()
    encoder.encode_batch.side_effect = lambda texts: np.ones(
        (len(texts), 384),
        dtype=np.float32,
    )

    vector_store = MagicMock()
    vector_store.count.return_value = 3

    pipeline = MagicMock()

    return encoder, vector_store, pipeline


def make_chunk(
    chunk_id: str,
    content: str,
    index: int,
) -> SimpleNamespace:
    """Create a lightweight test chunk."""
    return SimpleNamespace(
        chunk_id=chunk_id,
        content=content,
        source="https://example.edu/academics",
        chunk_index=index,
        metadata={"category": "academics"},
    )


def test_indexer_indexes_chunks_in_batches(
    tmp_path: Path,
    mock_components: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """The indexer should embed and store all chunks in batches."""
    encoder, vector_store, pipeline = mock_components

    chunks = [make_chunk(f"chunk-{i}", f"JKLU document content {i}", i) for i in range(5)]

    pipeline.run.return_value = SimpleNamespace(
        documents_loaded=2,
        chunks=chunks,
    )

    indexer = DocumentIndexer(
        encoder=encoder,
        vector_store=vector_store,
        ingestion_pipeline=pipeline,
        batch_size=2,
    )

    report = indexer.index_directory(tmp_path)

    assert report.documents_loaded == 2
    assert report.chunks_created == 5
    assert report.chunks_indexed == 5
    assert report.vector_store_count == 3

    assert encoder.encode_batch.call_count == 3
    assert vector_store.add.call_count == 3


def test_indexer_preserves_chunk_metadata(
    tmp_path: Path,
    mock_components: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Stored records should include source and chunk metadata."""
    encoder, vector_store, pipeline = mock_components

    chunk = make_chunk("chunk-1", "JKLU academic programs", 0)

    pipeline.run.return_value = SimpleNamespace(
        documents_loaded=1,
        chunks=[chunk],
    )

    indexer = DocumentIndexer(
        encoder=encoder,
        vector_store=vector_store,
        ingestion_pipeline=pipeline,
    )

    indexer.index_directory(tmp_path)

    kwargs = vector_store.add.call_args.kwargs

    assert kwargs["ids"] == ["chunk-1"]
    assert kwargs["documents"] == ["JKLU academic programs"]
    assert kwargs["metadatas"][0]["source"] == ("https://example.edu/academics")
    assert kwargs["metadatas"][0]["chunk_index"] == 0
    assert kwargs["metadatas"][0]["category"] == "academics"


def test_indexer_handles_empty_ingestion(
    tmp_path: Path,
    mock_components: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """An empty ingestion result should not invoke the encoder."""
    encoder, vector_store, pipeline = mock_components

    pipeline.run.return_value = SimpleNamespace(
        documents_loaded=0,
        chunks=[],
    )

    indexer = DocumentIndexer(
        encoder=encoder,
        vector_store=vector_store,
        ingestion_pipeline=pipeline,
    )

    report = indexer.index_directory(tmp_path)

    assert report.chunks_created == 0
    assert report.chunks_indexed == 0
    encoder.encode_batch.assert_not_called()
    vector_store.add.assert_not_called()


def test_indexer_rejects_invalid_batch_size() -> None:
    """Batch size must be positive."""
    with pytest.raises(ValueError, match="batch_size"):
        DocumentIndexer(batch_size=0)


def test_indexer_converts_complex_metadata_to_strings(
    tmp_path: Path,
    mock_components: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Nested metadata should be converted to Chroma-compatible values."""
    encoder, vector_store, pipeline = mock_components

    chunk = make_chunk("chunk-1", "JKLU campus details", 0)
    chunk.metadata = {
        "category": "campus",
        "tags": ["hostel", "library"],
        "details": {"building": "main"},
    }

    pipeline.run.return_value = SimpleNamespace(
        documents_loaded=1,
        chunks=[chunk],
    )

    indexer = DocumentIndexer(
        encoder=encoder,
        vector_store=vector_store,
        ingestion_pipeline=pipeline,
    )

    indexer.index_directory(tmp_path)

    metadata = vector_store.add.call_args.kwargs["metadatas"][0]

    assert metadata["category"] == "campus"
    assert isinstance(metadata["tags"], str)
    assert isinstance(metadata["details"], str)


def test_indexer_sync_deletes_obsolete_chunks(
    tmp_path: Path,
    mock_components: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Sync should remove obsolete chunks from the processed source."""
    encoder, vector_store, pipeline = mock_components

    chunks = [
        make_chunk("chunk-1", "Updated content one", 0),
        make_chunk("chunk-2", "Updated content two", 1),
    ]

    pipeline.run.return_value = SimpleNamespace(
        documents_loaded=1,
        chunks=chunks,
        errors=[],
        documents_skipped=0,
    )

    vector_store.get_ids_by_source.return_value = [
        "chunk-1",
        "chunk-2",
        "obsolete-chunk",
    ]
    vector_store.delete_ids.return_value = 1

    indexer = DocumentIndexer(
        encoder=encoder,
        vector_store=vector_store,
        ingestion_pipeline=pipeline,
        batch_size=2,
    )

    report = indexer.index_directory(tmp_path, sync=True)

    vector_store.add.assert_called_once()
    vector_store.delete_ids.assert_called_once_with(["obsolete-chunk"])
    vector_store.get_ids_by_source.assert_called_once_with("https://example.edu/academics")

    assert report.chunks_indexed == 2
    assert report.chunks_deleted == 1
    assert report.sources_synced == 1


def test_indexer_sync_preserves_unrelated_sources(
    tmp_path: Path,
    mock_components: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Sync should inspect only sources present in the new ingestion."""
    encoder, vector_store, pipeline = mock_components

    chunk = make_chunk("chunk-1", "Updated academic content", 0)

    pipeline.run.return_value = SimpleNamespace(
        documents_loaded=1,
        chunks=[chunk],
        errors=[],
        documents_skipped=0,
    )

    vector_store.get_ids_by_source.return_value = [
        "chunk-1",
        "obsolete-chunk",
    ]

    indexer = DocumentIndexer(
        encoder=encoder,
        vector_store=vector_store,
        ingestion_pipeline=pipeline,
    )

    indexer.index_directory(tmp_path, sync=True)

    vector_store.get_ids_by_source.assert_called_once_with("https://example.edu/academics")
    vector_store.delete_ids.assert_called_once_with(["obsolete-chunk"])


@pytest.mark.parametrize(
    ("errors", "documents_skipped"),
    [
        (["Failed to load a document"], 0),
        ([], 1),
    ],
)
def test_indexer_sync_aborts_on_ingestion_problems(
    tmp_path: Path,
    mock_components: tuple[MagicMock, MagicMock, MagicMock],
    errors: list[str],
    documents_skipped: int,
) -> None:
    """Sync must not delete existing records after incomplete ingestion."""
    encoder, vector_store, pipeline = mock_components

    pipeline.run.return_value = SimpleNamespace(
        documents_loaded=1,
        chunks=[make_chunk("chunk-1", "Content", 0)],
        errors=errors,
        documents_skipped=documents_skipped,
    )

    indexer = DocumentIndexer(
        encoder=encoder,
        vector_store=vector_store,
        ingestion_pipeline=pipeline,
    )

    with pytest.raises(RuntimeError, match="Synchronization aborted"):
        indexer.index_directory(tmp_path, sync=True)

    encoder.encode_batch.assert_not_called()
    vector_store.add.assert_not_called()
    vector_store.delete_ids.assert_not_called()


def test_indexer_sync_does_not_delete_after_embedding_failure(
    tmp_path: Path,
    mock_components: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Existing IDs must remain if embedding fails."""
    encoder, vector_store, pipeline = mock_components

    pipeline.run.return_value = SimpleNamespace(
        documents_loaded=1,
        chunks=[
            make_chunk("chunk-1", "Updated content", 0),
        ],
        errors=[],
        documents_skipped=0,
    )

    encoder.encode_batch.side_effect = RuntimeError("Embedding failed")

    indexer = DocumentIndexer(
        encoder=encoder,
        vector_store=vector_store,
        ingestion_pipeline=pipeline,
    )

    with pytest.raises(RuntimeError, match="Embedding failed"):
        indexer.index_directory(tmp_path, sync=True)

    vector_store.add.assert_not_called()
    vector_store.delete_ids.assert_not_called()
