"""Tests for retrieval evaluation metrics."""

import pytest

from jklu_compass.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


@pytest.fixture
def retrieved_ids() -> list[str]:
    return ["chunk-001", "chunk-002", "chunk-003", "chunk-004", "chunk-005"]


@pytest.fixture
def relevant_ids() -> set[str]:
    return {"chunk-002", "chunk-004"}


def test_precision_at_k(retrieved_ids, relevant_ids):
    assert precision_at_k(retrieved_ids, relevant_ids, k=5) == pytest.approx(0.4)


def test_recall_at_k(retrieved_ids, relevant_ids):
    assert recall_at_k(retrieved_ids, relevant_ids, k=5) == pytest.approx(1.0)


def test_hit_rate_at_k(retrieved_ids, relevant_ids):
    assert hit_rate_at_k(retrieved_ids, relevant_ids, k=5) == 1.0


def test_hit_rate_returns_zero_when_no_relevant_results():
    assert hit_rate_at_k(["chunk-001", "chunk-002"], {"chunk-003"}, k=2) == 0.0


def test_mean_reciprocal_rank(retrieved_ids, relevant_ids):
    assert mean_reciprocal_rank(retrieved_ids, relevant_ids, k=5) == pytest.approx(0.5)


def test_metrics_reject_invalid_k():
    with pytest.raises(ValueError, match="k"):
        precision_at_k(["chunk-001"], {"chunk-001"}, k=0)


def test_metrics_reject_empty_relevant_ids():
    with pytest.raises(ValueError, match="relevant_ids"):
        recall_at_k(["chunk-001"], set(), k=5)


def test_metrics_remove_duplicate_retrieved_ids():
    result = precision_at_k(
        ["chunk-001", "chunk-001", "chunk-002"],
        {"chunk-001", "chunk-002"},
        k=2,
    )

    assert result == pytest.approx(1.0)
