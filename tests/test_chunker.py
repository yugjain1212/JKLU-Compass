"""Tests for JKLU Compass document chunking."""

import pytest

from jklu_compass.ingestion.chunker import (
    ChunkingConfig,
    DocumentChunker,
)
from jklu_compass.ingestion.loaders import SourceDocument


def make_document(
    content: str,
    source: str = "https://jklu.edu.in/academics",
) -> SourceDocument:
    """Create a source document for testing."""
    return SourceDocument(
        content=content,
        source=source,
        metadata={"category": "academics", "source_type": "test"},
    )


def test_chunker_rejects_whitespace_only_document() -> None:
    """SourceDocument should reject whitespace-only content."""
    with pytest.raises(ValueError, match="Document content cannot be empty"):
        make_document("   \n\t  ")


def test_chunker_keeps_short_document_in_one_chunk() -> None:
    """Short documents should not be split unnecessarily."""
    chunker = DocumentChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))
    document = make_document("JKLU offers undergraduate programs.")

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content
    assert chunks[0].chunk_index == 0


def test_chunker_splits_long_document() -> None:
    """Long documents should be divided into multiple chunks."""
    config = ChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=10,
    )
    chunker = DocumentChunker(config)
    document = make_document("Academic information " * 30)

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= config.chunk_size for chunk in chunks)


def test_chunker_preserves_source_and_metadata() -> None:
    """Each chunk should retain its source and document metadata."""
    chunker = DocumentChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))
    document = make_document("University academic information.")

    chunks = chunker.chunk_document(document)

    assert chunks[0].source == document.source
    assert chunks[0].metadata == document.metadata
    assert chunks[0].metadata["category"] == "academics"


def test_chunker_assigns_sequential_indexes() -> None:
    """Chunk indexes should start at zero and increase sequentially."""
    chunker = DocumentChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=10))
    document = make_document("Academic information " * 30)

    chunks = chunker.chunk_document(document)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_deterministic() -> None:
    """Identical inputs should produce identical chunk IDs."""
    chunker = DocumentChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))
    document = make_document("JKLU academic information.")

    first = chunker.chunk_document(document)
    second = chunker.chunk_document(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_different_sources_produce_different_chunk_ids() -> None:
    """Different source identifiers should produce different chunk IDs."""
    chunker = DocumentChunker()

    first = chunker.chunk_document(make_document("JKLU academic information.", source="source-a"))
    second = chunker.chunk_document(make_document("JKLU academic information.", source="source-b"))

    assert first[0].chunk_id != second[0].chunk_id


def test_chunking_multiple_documents() -> None:
    """Multiple documents should be processed in input order."""
    chunker = DocumentChunker()
    documents = [
        make_document("Admissions information.", source="admissions"),
        make_document("Campus information.", source="campus"),
    ]

    chunks = chunker.chunk_documents(documents)

    assert len(chunks) == 2
    assert [chunk.source for chunk in chunks] == ["admissions", "campus"]


def test_chunk_overlap_retains_shared_text() -> None:
    """Adjacent chunks should share text when overlap is configured."""
    config = ChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=10,
    )
    chunker = DocumentChunker(config)
    document = make_document(" ".join(f"word{i}" for i in range(60)))

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.content.split()[-1] in current.content.split()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"chunk_size": 0}, "chunk_size must be greater than zero"),
        ({"chunk_overlap": -1}, "chunk_overlap cannot be negative"),
        (
            {"chunk_size": 100, "chunk_overlap": 100},
            "chunk_overlap must be smaller than chunk_size",
        ),
        ({"min_chunk_size": 0}, "min_chunk_size must be at least 1"),
        (
            {"chunk_size": 100, "min_chunk_size": 101},
            "min_chunk_size cannot exceed chunk_size",
        ),
    ],
)
def test_chunking_rejects_invalid_configuration(
    config: dict[str, int],
    message: str,
) -> None:
    """Invalid chunking configurations should raise clear errors."""
    with pytest.raises(ValueError, match=message):
        ChunkingConfig(**config)
