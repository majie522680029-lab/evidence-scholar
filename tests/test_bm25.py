"""Tests for the BM25 retrieval implementation."""

import pytest

from evidence_scholar.retrieval.bm25 import (
    BM25Index,
    build_document_tokens,
    tokenize,
)


def test_tokenize_normalizes_english_text() -> None:
    """Tokenizer should lowercase and preserve useful compounds."""
    tokens = tokenize(
        "Scott Derrickson's 2016 horror-film."
    )

    assert tokens == [
        "scott",
        "derrickson's",
        "2016",
        "horror-film",
    ]


def test_build_document_tokens_weights_title() -> None:
    """Title tokens should be repeated according to title_weight."""
    tokens = build_document_tokens(
        title="Scott Derrickson",
        text="American filmmaker",
        title_weight=2,
    )

    assert tokens == [
        "scott",
        "derrickson",
        "scott",
        "derrickson",
        "american",
        "filmmaker",
    ]


def test_bm25_ranks_matching_document_first() -> None:
    """A strongly matching document should rank first."""
    index = BM25Index(
        document_ids=[
            "doc-a",
            "doc-b",
            "doc-c",
        ],
        tokenized_documents=[
            tokenize(
                "Scott Derrickson American filmmaker"
            ),
            tokenize(
                "Ed Wood American filmmaker"
            ),
            tokenize(
                "Woodson Arkansas city"
            ),
        ],
    )

    results = index.search(
        "Scott Derrickson",
        top_k=3,
    )

    assert results[0].document_id == "doc-a"
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_search_limits_results_to_collection_size() -> None:
    """top_k may be larger than the document collection."""
    index = BM25Index(
        document_ids=[
            "doc-a",
            "doc-b",
        ],
        tokenized_documents=[
            tokenize("apple"),
            tokenize("banana"),
        ],
    )

    results = index.search(
        "apple",
        top_k=10,
    )

    assert len(results) == 2


def test_invalid_bm25_parameters_raise_error() -> None:
    """Invalid index parameters should be rejected."""
    with pytest.raises(
        ValueError,
        match="k1",
    ):
        BM25Index(
            document_ids=["doc-a"],
            tokenized_documents=[
                tokenize("example")
            ],
            k1=0,
        )

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        BM25Index(
            document_ids=["doc-a"],
            tokenized_documents=[
                tokenize("example")
            ],
            b=1.5,
        )
