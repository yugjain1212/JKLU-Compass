"""Tests for source-aware document retrieval."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from jklu_compass.retrieval import DocumentRetriever


@pytest.fixture
def encoder() -> MagicMock:
    """Provide a fake encoder without loading a real model."""
    mock = MagicMock()
    mock.encode.return_value = np.array([0.1, 0.2, 0.3])
    return mock


@pytest.fixture
def vector_store() -> MagicMock:
    """Provide a fake vector store with source-aware results."""
    store = MagicMock()
    store.search.return_value = [
        {
            "id": "fee-chunk-1",
            "document": "JKLU publishes fee information for its programmes.",
            "metadata": {
                "source": "/data/fee-structure.md",
                "source_url": "https://jklu.edu.in/admissions/fee-structure",
                "title": "JKLU Fee Structure",
                "file_name": "fee-structure.md",
            },
            "distance": 0.12,
        }
    ]
    return store


@pytest.fixture
def retriever(
    encoder: MagicMock,
    vector_store: MagicMock,
) -> DocumentRetriever:
    """Create a retriever using mocked dependencies."""
    return DocumentRetriever(encoder=encoder, vector_store=vector_store)


def test_retrieve_returns_source_url_and_title(
    retriever: DocumentRetriever,
) -> None:
    results = retriever.retrieve("What are JKLU fees?")

    assert len(results) == 1
    assert results[0].title == "JKLU Fee Structure"
    assert results[0].source_url == ("https://jklu.edu.in/admissions/fee-structure")
    assert results[0].content == ("JKLU publishes fee information for its programmes.")
    assert results[0].distance == 0.12


def test_retrieve_encodes_query_and_searches(
    retriever: DocumentRetriever,
    encoder: MagicMock,
    vector_store: MagicMock,
) -> None:
    retriever.retrieve("  What are JKLU fees?  ", top_k=3)

    encoder.encode.assert_called_once_with("What are JKLU fees?")
    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=3,
    )


def test_retrieve_rejects_empty_query(
    retriever: DocumentRetriever,
) -> None:
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("   ")


def test_retrieve_rejects_invalid_top_k(
    retriever: DocumentRetriever,
) -> None:
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("JKLU fees", top_k=0)


def test_local_source_is_not_returned_as_clickable_url(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
) -> None:
    vector_store.search.return_value[0]["metadata"].pop("source_url")

    results = retriever.retrieve("JKLU fees")

    assert results[0].source == "/data/fee-structure.md"
    assert results[0].source_url is None


@pytest.mark.parametrize(
    "candidate",
    [
        "javascript:alert(1)",
        "file:///tmp/fee-structure.md",
        "not-a-url",
        "",
    ],
)
def test_invalid_source_url_is_rejected(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
    candidate: str,
) -> None:
    vector_store.search.return_value[0]["metadata"]["source_url"] = candidate

    results = retriever.retrieve("JKLU fees")

    assert results[0].source_url is None


def test_http_source_is_used_when_source_url_is_missing(
    retriever: DocumentRetriever,
    vector_store: MagicMock,
) -> None:
    metadata: dict[str, Any] = vector_store.search.return_value[0]["metadata"]
    metadata.pop("source_url")
    metadata["source"] = "https://jklu.edu.in/academics"

    results = retriever.retrieve("JKLU academics")

    assert results[0].source_url == "https://jklu.edu.in/academics"
