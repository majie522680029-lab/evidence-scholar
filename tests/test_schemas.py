"""Tests for the common retrieval schemas."""

import pytest
from pydantic import ValidationError

from evidence_scholar.retrieval.schemas import (
    Document,
    Query,
    RetrievalResult,
    SupportingFact,
)


def test_document_json_round_trip() -> None:
    document = Document(
        document_id="doc-001",
        title="Example Paper",
        text="This is an example document.",
        sentences=(
            "This is an example document.",
            "It is used for testing.",
        ),
    )

    serialized = document.model_dump_json()
    restored = Document.model_validate_json(serialized)

    assert restored == document


def test_query_contains_gold_evidence() -> None:
    query = Query(
        query_id="query-001",
        text="Who proposed the example method?",
        answer="Alice",
        gold_document_ids=("doc-001",),
        supporting_facts=(
            SupportingFact(
                title="Example Paper",
                sentence_index=1,
            ),
        ),
    )

    assert query.gold_document_ids == ("doc-001",)
    assert query.supporting_facts[0].sentence_index == 1


def test_retrieval_rank_must_start_at_one() -> None:
    with pytest.raises(ValidationError):
        RetrievalResult(
            document_id="doc-001",
            score=0.9,
            rank=0,
            title="Example Paper",
            text="Example text.",
        )


def test_document_id_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        Document(
            document_id="",
            title="Example Paper",
            text="Example text.",
        )
