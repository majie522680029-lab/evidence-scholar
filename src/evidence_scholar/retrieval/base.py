"""Abstract interface implemented by every retriever."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from evidence_scholar.retrieval.schemas import Document, RetrievalResult


class BaseRetriever(ABC):
    """Common contract for sparse, dense and hybrid retrievers."""

    @abstractmethod
    def build_index(self, documents: Sequence[Document]) -> None:
        """Build an in-memory index from documents."""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Return documents ranked by relevance to the query."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the index and required metadata."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a previously persisted index."""
