"""Metrics for document retrieval evaluation."""

from __future__ import annotations

from collections.abc import Sequence


def _validate_inputs(
    relevant_document_ids: set[str],
    k: int | None = None,
) -> None:
    """Validate common metric inputs."""
    if not relevant_document_ids:
        raise ValueError(
            "relevant_document_ids must not be empty."
        )

    if k is not None and k <= 0:
        raise ValueError("k must be greater than zero.")


def recall_at_k(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    """Return the proportion of relevant documents found in top-k."""
    _validate_inputs(relevant_document_ids, k)

    retrieved = set(ranked_document_ids[:k])
    relevant_retrieved = (
        retrieved & relevant_document_ids
    )

    return (
        len(relevant_retrieved)
        / len(relevant_document_ids)
    )


def hit_at_k(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    """Return 1 when top-k contains at least one relevant document."""
    _validate_inputs(relevant_document_ids, k)

    retrieved = set(ranked_document_ids[:k])

    return float(
        bool(retrieved & relevant_document_ids)
    )


def complete_recall_at_k(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    """Return 1 when top-k contains every relevant document."""
    _validate_inputs(relevant_document_ids, k)

    retrieved = set(ranked_document_ids[:k])

    return float(
        relevant_document_ids.issubset(retrieved)
    )


def reciprocal_rank(
    ranked_document_ids: Sequence[str],
    relevant_document_ids: set[str],
) -> float:
    """Return reciprocal rank of the first relevant result."""
    _validate_inputs(relevant_document_ids)

    for rank, document_id in enumerate(
        ranked_document_ids,
        start=1,
    ):
        if document_id in relevant_document_ids:
            return 1.0 / rank

    return 0.0
