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
