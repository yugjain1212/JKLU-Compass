"""Tests for BM25 lexical retrieval."""

from unittest.mock import MagicMock

import pytest

from jklu_compass.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def vector_store() -> MagicMock:
    store = MagicMock()
    store.get_all_records.return_value = [
        {
            "id": "btech-1",
            "document": "Bachelor of Technology offers engineering specialisations.",
            "metadata": {
                "title": "Bachelor of Technology",
                "section_title": "Specialisations",
            },
        },
        {
            "id": "bdes-1",
            "document": "Bachelor of Design focuses on design education.",
            "metadata": {
                "title": "Bachelor of Design",
                "section_title": "Programme Overview",
            },
        },
        {
            "id": "bba-1",
            "document": "BBA Fintech curriculum and semester structure.",
            "metadata": {
                "title": "BBA Fintech",
                "section_title": "Curriculum",
            },
        },
    ]
    return store


def test_initializes_index_from_existing_records(
    vector_store: MagicMock,
) -> None:
    retriever = BM25Retriever(vector_store)

    assert len(retriever.document_ids) == 3
    vector_store.get_all_records.assert_called_once_with(batch_size=500)


def test_retrieves_relevant_document(
    vector_store: MagicMock,
) -> None:
    retriever = BM25Retriever(vector_store)

    results = retriever.retrieve("BBA Fintech curriculum", top_k=3)

    assert results
    assert results[0].id == "bba-1"
    assert results[0].score > 0


def test_retrieves_programme_specialisations(
    vector_store: MagicMock,
) -> None:
    retriever = BM25Retriever(vector_store)

    results = retriever.retrieve("Bachelor of Technology specialisations")

    assert results[0].id == "btech-1"


def test_returns_empty_for_empty_corpus() -> None:
    store = MagicMock()
    store.get_all_records.return_value = []

    retriever = BM25Retriever(store)

    assert retriever.retrieve("JKLU admissions") == []


def test_rejects_empty_query(vector_store: MagicMock) -> None:
    retriever = BM25Retriever(vector_store)

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("  ")


def test_rejects_invalid_top_k(vector_store: MagicMock) -> None:
    retriever = BM25Retriever(vector_store)

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("JKLU programmes", top_k=0)


def test_rejects_invalid_batch_size(vector_store: MagicMock) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BM25Retriever(vector_store, batch_size=0)
