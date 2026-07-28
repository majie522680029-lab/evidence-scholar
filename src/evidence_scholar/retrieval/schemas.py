"""Shared data schemas for datasets, retrieval and evaluation."""

from pydantic import BaseModel, ConfigDict, Field


class SupportingFact(BaseModel):
    """A sentence-level gold evidence annotation."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    sentence_index: int = Field(ge=0)


class Document(BaseModel):
    """A searchable document in the retrieval corpus."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    sentences: tuple[str, ...] = ()


class Query(BaseModel):
    """A question and its gold answer and evidence annotations."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    answer: str
    gold_document_ids: tuple[str, ...] = ()
    supporting_facts: tuple[SupportingFact, ...] = ()


class RetrievalResult(BaseModel):
    """One ranked document returned by a retriever."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=1)
    score: float
    rank: int = Field(ge=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
