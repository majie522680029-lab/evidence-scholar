"""Tests for the Hybrid (RRF) retrieval implementation.

设计原则：不加载真模型、不依赖 GPU。
用两个 FakeRetriever（实现 BaseRetriever 契约、返回预设 RetrievalResult）
精确控制每篇文档在两边的 rank，手算 RRF 融合分做断言。
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from evidence_scholar.retrieval.base import BaseRetriever
from evidence_scholar.retrieval.hybrid import HybridRetriever
from evidence_scholar.retrieval.schemas import Document, RetrievalResult


class _FakeRetriever(BaseRetriever):
    """A controllable BaseRetriever returning preset rankings.

    预置一个 query -> list[RetrievalResult] 的映射，search 时直接返回。
    用于精确控制每篇文档的 rank，让 RRF 融合分可手算验证。
    """

    def __init__(self, rankings: dict[str, list[RetrievalResult]]) -> None:
        self.rankings = rankings
        self._documents: list[Document] = []

    def build_index(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("Cannot build index from empty document set.")
        self._documents = list(documents)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        return self.rankings.get(query, [])[:top_k]

    def save(self, path) -> None:
        raise NotImplementedError

    def load(self, path) -> None:
        raise NotImplementedError


def _result(
    document_id: str,
    rank: int,
    title: str = "T",
    text: str = "X",
    score: float = 0.0,
) -> RetrievalResult:
    """快捷构造一个 RetrievalResult。"""
    return RetrievalResult(
        document_id=document_id,
        title=title,
        text=text,
        score=score,
        rank=rank,
    )


def _make_docs(*ids: str) -> list[Document]:
    return [Document(document_id=i, title="T", text="X") for i in ids]


# --- 参数校验测试 ---------------------------------------------------------


def test_non_positive_rrf_k_raises_error() -> None:
    """rrf_k <= 0 must be rejected."""
    bm25 = _FakeRetriever({})
    dense = _FakeRetriever({})

    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(bm25, dense, rrf_k=0)


def test_search_rejects_non_positive_top_k() -> None:
    """top_k <= 0 must be rejected."""
    bm25 = _FakeRetriever({})
    dense = _FakeRetriever({})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    with pytest.raises(ValueError, match="top_k"):
        hybrid.search("q", top_k=0)


def test_search_before_build_index_does_not_error() -> None:
    """search 不依赖 build_index（FakeRetriever 自带预设排名），应能跑。

    说明：HybridRetriever.search 只调子检索器的 search，不读索引状态，
    所以未 build_index 也能 search（只要子检索器有预设排名）。
    """
    bm25 = _FakeRetriever({"q": [_result("a", 1)]})
    dense = _FakeRetriever({"q": [_result("a", 1)]})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("q", top_k=5)

    assert len(results) == 1
    assert results[0].document_id == "a"


# --- RRF 融合正确性测试 ---------------------------------------------------


def test_both_sided_document_outranks_single_sided() -> None:
    """两边都召回的文档应排在只一边召回的文档之前。

    复刻讲解中的例子：
    - A：BM25 rank 1，Dense rank 3  -> 双份贡献
    - B：BM25 rank 2，Dense 没召回  -> 单份贡献
    A 的 rrf = 1/61 + 1/63 > B 的 rrf = 1/62 + 0，A 应排第一。
    """
    bm25 = _FakeRetriever(
        {
            "q": [
                _result("A", 1),
                _result("B", 2),
            ]
        }
    )
    dense = _FakeRetriever(
        {
            "q": [
                _result("A", 3),
            ]
        }
    )
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("q", top_k=2)

    assert results[0].document_id == "A"
    assert results[1].document_id == "B"
    # A 的融合分严格大于 B
    assert results[0].score > results[1].score


def test_rrf_score_value_matches_hand_calculation() -> None:
    """RRF 融合分应与手算公式 1/(k+rank) 一致。

    文档 C：BM25 rank 3，Dense rank 1。
    手算：1/(60+3) + 1/(60+1) = 1/63 + 1/61。
    """
    bm25 = _FakeRetriever({"q": [_result("C", 3)]})
    dense = _FakeRetriever({"q": [_result("C", 1)]})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("q", top_k=1)

    expected = 1 / 63 + 1 / 61
    assert results[0].document_id == "C"
    assert abs(results[0].score - expected) < 1e-9


def test_single_sided_document_contribution_is_correct() -> None:
    """只被一边召回的文档，另一边贡献为 0。

    文档 D：Dense rank 2，BM25 没召回。
    手算：0 + 1/(60+2) = 1/62。
    """
    bm25 = _FakeRetriever({"q": []})
    dense = _FakeRetriever({"q": [_result("D", 2)]})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("q", top_k=1)

    expected = 1 / 62
    assert results[0].document_id == "D"
    assert abs(results[0].score - expected) < 1e-9


def test_results_carry_title_and_text() -> None:
    """结果应带 title/text（从子检索器结果回填）。"""
    bm25 = _FakeRetriever(
        {"q": [_result("A", 1, title="The Title", text="the text")]}
    )
    dense = _FakeRetriever({"q": [_result("A", 1, title="X", text="Y")]})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("q", top_k=1)

    # title/text 取自任一子结果（这里 bm25 先存，取 bm25 的）
    assert results[0].title in ("The Title", "X")
    assert results[0].text in ("the text", "Y")


def test_ranks_are_consecutive_from_one() -> None:
    """返回的 rank 必须从 1 连续。"""
    bm25 = _FakeRetriever(
        {
            "q": [
                _result("A", 1),
                _result("B", 2),
                _result("C", 3),
            ]
        }
    )
    dense = _FakeRetriever({"q": []})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("q", top_k=3)

    assert [r.rank for r in results] == [1, 2, 3]


def test_top_k_truncates_union() -> None:
    """top_k 应截断并集，返回不超过 top_k 个结果。"""
    bm25 = _FakeRetriever(
        {"q": [_result(f"d{i}", i + 1) for i in range(10)]}
    )
    dense = _FakeRetriever({"q": []})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("q", top_k=3)

    assert len(results) == 3


def test_union_merges_disjoint_sets() -> None:
    """两边完全不相交时，并集应全部保留并按融合分排序。"""
    bm25 = _FakeRetriever({"q": [_result("A", 1), _result("B", 2)]})
    dense = _FakeRetriever({"q": [_result("C", 1), _result("D", 2)]})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("q", top_k=10)

    ids = {r.document_id for r in results}
    assert ids == {"A", "B", "C", "D"}
    # A 和 C 都是各自 rank 1（融合分相同），应排在 B、D 之前
    top_two = {results[0].document_id, results[1].document_id}
    assert top_two == {"A", "C"}


def test_tie_break_is_stable() -> None:
    """同分时按 document_id 字典序，保证稳定可复现。

    A 和 C 融合分相同（都是 1/61 + 0），字典序 A < C，故 A 排第一。
    """
    bm25 = _FakeRetriever({"q": [_result("A", 1), _result("C", 1)]})
    dense = _FakeRetriever({"q": []})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    results = hybrid.search("q", top_k=2)

    assert [r.document_id for r in results] == ["A", "C"]
    assert results[0].score == results[1].score


# --- save / load 测试 -----------------------------------------------------


def test_save_raises_not_implemented() -> None:
    """Hybrid 不直接持久化，save 应明确报错。"""
    hybrid = HybridRetriever(_FakeRetriever({}), _FakeRetriever({}))

    with pytest.raises(NotImplementedError, match="save each sub-retriever"):
        hybrid.save("/tmp/should_not_exist")


def test_load_raises_not_implemented() -> None:
    """Hybrid 不直接加载，load 应明确报错。"""
    hybrid = HybridRetriever(_FakeRetriever({}), _FakeRetriever({}))

    with pytest.raises(NotImplementedError, match="load each sub-retriever"):
        hybrid.load("/tmp/should_not_exist")


# --- build_index 测试 -----------------------------------------------------


def test_build_index_delegates_to_both_sub_retrievers() -> None:
    """build_index 应让两个子检索器都在同一份文档上建索引。"""
    bm25 = _FakeRetriever({"q": [_result("A", 1)]})
    dense = _FakeRetriever({"q": [_result("A", 1)]})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    docs = _make_docs("A", "B")
    hybrid.build_index(docs)

    assert bm25._documents == docs
    assert dense._documents == docs


def test_build_index_rejects_empty_documents() -> None:
    """空文档集应由子检索器 build_index 拒绝。"""
    bm25 = _FakeRetriever({"q": []})
    dense = _FakeRetriever({"q": []})
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    with pytest.raises(ValueError, match="empty document set"):
        hybrid.build_index([])
