"""Tests for the persistent ChromaDB vector store."""

from pathlib import Path

import pytest

from jklu_compass.vectorstore.chroma_store import ChromaVectorStore


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    """Create an isolated vector store for each test."""
    return ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_collection",
    )


def test_store_starts_empty(store: ChromaVectorStore) -> None:
    assert store.count() == 0


def test_add_and_count_records(store: ChromaVectorStore) -> None:
    added = store.add(
        ids=["chunk-1", "chunk-2"],
        documents=["JKLU admissions", "JKLU academic programs"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"category": "admissions"}, {"category": "academics"}],
    )

    assert added == 2
    assert store.count() == 2


def test_search_returns_relevant_record(store: ChromaVectorStore) -> None:
    store.add(
        ids=["admissions", "academics"],
        documents=["JKLU admission process", "JKLU academic programs"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    results = store.search([0.99, 0.01], top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "admissions"
    assert results[0]["document"] == "JKLU admission process"


def test_upsert_does_not_duplicate_ids(store: ChromaVectorStore) -> None:
    store.add(
        ids=["chunk-1"],
        documents=["Original text"],
        embeddings=[[1.0, 0.0]],
    )
    store.add(
        ids=["chunk-1"],
        documents=["Updated text"],
        embeddings=[[0.0, 1.0]],
    )

    assert store.count() == 1
    result = store.search([0.0, 1.0], top_k=1)
    assert result[0]["document"] == "Updated text"


def test_store_persists_between_instances(tmp_path: Path) -> None:
    directory = tmp_path / "persistent_chroma"

    first_store = ChromaVectorStore(
        persist_directory=directory,
        collection_name="persistence_test",
    )
    first_store.add(
        ids=["chunk-1"],
        documents=["JKLU campus"],
        embeddings=[[1.0, 0.0]],
    )

    second_store = ChromaVectorStore(
        persist_directory=directory,
        collection_name="persistence_test",
    )

    assert second_store.count() == 1
    assert second_store.search([1.0, 0.0])[0]["document"] == "JKLU campus"


def test_add_rejects_mismatched_lengths(store: ChromaVectorStore) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        store.add(
            ids=["chunk-1"],
            documents=["Text one", "Text two"],
            embeddings=[[1.0, 0.0]],
        )


def test_add_rejects_duplicate_ids_in_batch(
    store: ChromaVectorStore,
) -> None:
    with pytest.raises(ValueError, match="unique"):
        store.add(
            ids=["chunk-1", "chunk-1"],
            documents=["Text one", "Text two"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_search_rejects_invalid_top_k(store: ChromaVectorStore) -> None:
    with pytest.raises(ValueError, match="top_k"):
        store.search([1.0, 0.0], top_k=0)


def test_get_ids_by_source(store: ChromaVectorStore) -> None:
    """Return only IDs belonging to the requested source."""
    store.add(
        ids=["chunk-1", "chunk-2", "chunk-3"],
        documents=["First", "Second", "Third"],
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        metadatas=[
            {"source": "source-a"},
            {"source": "source-a"},
            {"source": "source-b"},
        ],
    )

    assert set(store.get_ids_by_source("source-a")) == {
        "chunk-1",
        "chunk-2",
    }


def test_delete_ids_preserves_other_records(
    store: ChromaVectorStore,
) -> None:
    """Delete selected IDs without affecting unrelated records."""
    store.add(
        ids=["chunk-1", "chunk-2"],
        documents=["First", "Second"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[
            {"source": "source-a"},
            {"source": "source-b"},
        ],
    )

    deleted = store.delete_ids(["chunk-1"])

    assert deleted == 1
    assert store.count() == 1
    assert store.get_ids_by_source("source-b") == ["chunk-2"]


def test_reset_clears_records(store: ChromaVectorStore) -> None:
    store.add(
        ids=["chunk-1"],
        documents=["JKLU campus"],
        embeddings=[[1.0, 0.0]],
    )

    store.reset()

    assert store.count() == 0
