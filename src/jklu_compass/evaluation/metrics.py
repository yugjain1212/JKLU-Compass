"""Deterministic evaluation metrics for information retrieval."""

from collections.abc import Sequence


def _validate_inputs(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> list[str]:
    """Validate and normalize retrieval metric inputs."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty.")

    if any(not isinstance(item, str) or not item for item in retrieved_ids):
        raise ValueError("retrieved_ids must contain non-empty strings.")

    if any(not isinstance(item, str) or not item for item in relevant_ids):
        raise ValueError("relevant_ids must contain non-empty strings.")

    # Remove duplicate retrieved IDs while preserving ranking order.
    return list(dict.fromkeys(retrieved_ids))


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Calculate Precision@K."""
    retrieved = _validate_inputs(retrieved_ids, relevant_ids, k)
    top_k = retrieved[:k]

    return sum(item in relevant_ids for item in top_k) / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Calculate Recall@K."""
    retrieved = _validate_inputs(retrieved_ids, relevant_ids, k)
    top_k = retrieved[:k]

    relevant_retrieved = sum(item in relevant_ids for item in top_k)

    return relevant_retrieved / len(relevant_ids)


def hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Return 1.0 if at least one relevant result appears in the top K."""
    retrieved = _validate_inputs(retrieved_ids, relevant_ids, k)

    return float(any(item in relevant_ids for item in retrieved[:k]))


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Calculate reciprocal rank of the first relevant result."""
    retrieved = _validate_inputs(retrieved_ids, relevant_ids, k)

    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant_ids:
            return 1.0 / rank

    return 0.0
