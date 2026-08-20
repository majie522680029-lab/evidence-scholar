"""Tests for retrieval evaluation metrics."""

import pytest

from evidence_scholar.evaluation.metrics import (
    complete_recall_at_k,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k() -> None:
    ranked = [
        "doc-a",
        "doc-c",
        "doc-b",
    ]
    relevant = {
        "doc-a",
        "doc-b",
    }

    assert recall_at_k(
        ranked,
        relevant,
        1,
    ) == 0.5

    assert recall_at_k(
        ranked,
        relevant,
        2,
    ) == 0.5

    assert recall_at_k(
        ranked,
        relevant,
        3,
    ) == 1.0


def test_hit_at_k() -> None:
    ranked = [
        "doc-c",
        "doc-a",
        "doc-b",
    ]
    relevant = {
        "doc-a",
        "doc-b",
    }

    assert hit_at_k(
        ranked,
        relevant,
        1,
    ) == 0.0

    assert hit_at_k(
        ranked,
        relevant,
        2,
    ) == 1.0


def test_complete_recall_at_k() -> None:
    ranked = [
        "doc-a",
        "doc-c",
        "doc-b",
    ]
    relevant = {
        "doc-a",
        "doc-b",
    }

    assert complete_recall_at_k(
        ranked,
        relevant,
        2,
    ) == 0.0

    assert complete_recall_at_k(
        ranked,
        relevant,
        3,
    ) == 1.0


def test_reciprocal_rank() -> None:
    ranked = [
        "doc-c",
        "doc-a",
        "doc-b",
    ]
    relevant = {
        "doc-a",
        "doc-b",
    }

    assert reciprocal_rank(
        ranked,
        relevant,
    ) == 0.5


def test_empty_relevance_set_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        recall_at_k(
            ["doc-a"],
            set(),
            1,
        )
