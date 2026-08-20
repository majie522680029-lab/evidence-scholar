"""Tests for the Dense retrieval implementation.

设计原则：不加载真模型、不依赖 GPU。
通过 monkeypatch 把 DenseRetriever 内部的 SentenceTransformer 替换成一个
返回固定可控向量的 FakeModel，使测试：
1. 秒级完成（无模型加载、无编码开销）；
2. 在无 GPU、无网络的机器上也能跑（CI 友好）；
3. 可精确预测排序结果，断言更严格。

真正加载模型 + GPU 编码的端到端验证已在 scripts/evaluate_dense_hotpotqa.py
的小样本试跑中覆盖，这里不重复。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from evidence_scholar.retrieval import dense as dense_module
from evidence_scholar.retrieval.dense import DenseRetriever
from evidence_scholar.retrieval.schemas import Document


class _FakeModel:
    """A controllable stand-in for SentenceTransformer.

    按文本内容返回预置的固定向量，使排序结果可预测、可断言。
    未知文本返回零向量（不相关）。
    """

    def __init__(
        self,
        vectors: dict[str, np.ndarray] | None = None,
        default_dim: int = 4,
    ) -> None:
        self.vectors = vectors or {}
        # 优先用已注入向量的维度；若没有注入，用 default_dim 兜底，
        # 使只关心参数校验、不在乎具体向量的测试也能正常建索引。
        if self.vectors:
            self.dim = next(iter(self.vectors.values())).shape[0]
        else:
            self.dim = default_dim

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Return stacked vectors for the given texts.

        已注入的文本返回其预置向量；未注入的文本返回该维零向量，
        这样依赖参数校验的测试无需为每个占位文本都准备向量。
        """
        result = []
        for text in texts:
            vec = self.vectors.get(text)
            if vec is None:
                vec = np.zeros(self.dim, dtype=np.float32)
            result.append(vec)
        return np.ascontiguousarray(
            np.stack(result), dtype=np.float32
        )


def _patch_model(
    monkeypatch: pytest.MonkeyPatch,
    vectors: dict[str, np.ndarray],
) -> None:
    """Patch SentenceTransformer in dense.py to use _FakeModel."""
    fake_cls = lambda model_name, device=None, *args: _FakeModel(vectors)  # noqa: E731
    monkeypatch.setattr(dense_module, "SentenceTransformer", fake_cls)


# --- 参数校验测试 ---------------------------------------------------------


def test_empty_model_name_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty model_name must be rejected before touching the network."""
    _patch_model(monkeypatch, {})

    with pytest.raises(ValueError, match="model_name"):
        DenseRetriever(model_name="", device="cpu")


def test_non_positive_batch_size_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """batch_size <= 0 must be rejected."""
    _patch_model(monkeypatch, {})

    with pytest.raises(ValueError, match="batch_size"):
        DenseRetriever(
            model_name="fake-model",
            device="cpu",
            batch_size=0,
        )


# --- build_index 测试 ------------------------------------------------------


def test_build_index_rejects_empty_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty document set must be rejected."""
    _patch_model(monkeypatch, {})

    retriever = DenseRetriever(model_name="fake-model", device="cpu")

    with pytest.raises(ValueError, match="empty document set"):
        retriever.build_index([])


def test_build_index_rejects_duplicate_document_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate document_ids must be rejected to keep results stable."""
    _patch_model(monkeypatch, {})

    retriever = DenseRetriever(model_name="fake-model", device="cpu")

    with pytest.raises(ValueError, match="unique"):
        retriever.build_index(
            [
                Document(
                    document_id="doc-a",
                    title="A",
                    text="alpha",
                ),
                Document(
                    document_id="doc-a",
                    title="B",
                    text="beta",
                ),
            ]
        )


# --- search 测试 ----------------------------------------------------------


def test_search_before_build_index_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Searching before building an index must raise."""
    _patch_model(monkeypatch, {})

    retriever = DenseRetriever(model_name="fake-model", device="cpu")

    with pytest.raises(ValueError, match="Index not built"):
        retriever.search("anything", top_k=5)


def test_search_rejects_non_positive_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """top_k <= 0 must be rejected."""
    _patch_model(monkeypatch, {})

    retriever = DenseRetriever(model_name="fake-model", device="cpu")
    retriever.build_index(
        [
            Document(
                document_id="doc-a",
                title="A",
                text="alpha",
            )
        ]
    )

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("alpha", top_k=0)


def test_search_ranks_most_similar_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The document most similar to the query must rank first.

    用可控向量构造：query 与 doc-b 方向一致（高相似度），
    与 doc-a 正交（零相似度），因此 doc-b 应排第一。
    """
    dim = 4
    query_vec = np.array([1, 0, 0, 0], dtype=np.float32)
    doc_a_vec = np.array([0, 1, 0, 0], dtype=np.float32)  # 正交
    doc_b_vec = np.array([1, 0, 0, 0], dtype=np.float32)  # 同向

    vectors = {
        "query text": query_vec,
        "doc a text": doc_a_vec,
        "doc b text": doc_b_vec,
    }
    _patch_model(monkeypatch, vectors)

    retriever = DenseRetriever(model_name="fake-model", device="cpu")
    retriever.build_index(
        [
            Document(
                document_id="doc-a",
                title="A",
                text="doc a text",
            ),
            Document(
                document_id="doc-b",
                title="B",
                text="doc b text",
            ),
        ]
    )

    results = retriever.search("query text", top_k=2)

    assert len(results) == 2
    assert results[0].document_id == "doc-b"
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_search_ranks_are_consecutive_from_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned ranks must be 1, 2, 3, ... consecutive."""
    dim = 3
    # 三篇文档与 query 相似度递减
    vectors = {
        "q": np.array([1, 0, 0], dtype=np.float32),
        "t1": np.array([0.9, 0, 0], dtype=np.float32),
        "t2": np.array([0.5, 0, 0], dtype=np.float32),
        "t3": np.array([0.1, 0, 0], dtype=np.float32),
    }
    _patch_model(monkeypatch, vectors)

    retriever = DenseRetriever(model_name="fake-model", device="cpu")
    retriever.build_index(
        [
            Document(document_id="d1", title="T1", text="t1"),
            Document(document_id="d2", title="T2", text="t2"),
            Document(document_id="d3", title="T3", text="t3"),
        ]
    )

    results = retriever.search("q", top_k=3)

    assert [r.rank for r in results] == [1, 2, 3]


def test_search_top_k_exceeds_collection_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """top_k larger than document count must not raise.

    FAISS 在结果不足时返回 -1 填充位，DenseRetriever 应跳过它们，
    只返回实际存在的文档。
    """
    dim = 2
    vectors = {
        "q": np.array([1, 0], dtype=np.float32),
        "t1": np.array([1, 0], dtype=np.float32),
        "t2": np.array([0, 1], dtype=np.float32),
    }
    _patch_model(monkeypatch, vectors)

    retriever = DenseRetriever(model_name="fake-model", device="cpu")
    retriever.build_index(
        [
            Document(document_id="d1", title="T1", text="t1"),
            Document(document_id="d2", title="T2", text="t2"),
        ]
    )

    results = retriever.search("q", top_k=10)

    assert len(results) == 2
    assert all(r.rank >= 1 for r in results)


def test_search_results_carry_title_and_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results must carry title and text per the BaseRetriever contract."""
    dim = 2
    vectors = {
        "q": np.array([1, 0], dtype=np.float32),
        "t1": np.array([1, 0], dtype=np.float32),
    }
    _patch_model(monkeypatch, vectors)

    retriever = DenseRetriever(model_name="fake-model", device="cpu")
    retriever.build_index(
        [
            Document(
                document_id="d1",
                title="The Title",
                text="t1",
            ),
        ]
    )

    results = retriever.search("q", top_k=1)

    assert results[0].title == "The Title"
    assert results[0].text == "t1"


# --- save / load 测试 -----------------------------------------------------


def test_save_load_preserves_search_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """save then load must reproduce identical search rankings."""
    dim = 2
    vectors = {
        "q": np.array([1, 0], dtype=np.float32),
        "t1": np.array([1, 0], dtype=np.float32),
        "t2": np.array([0, 1], dtype=np.float32),
    }
    _patch_model(monkeypatch, vectors)

    retriever = DenseRetriever(model_name="fake-model", device="cpu")
    retriever.build_index(
        [
            Document(document_id="d1", title="T1", text="t1"),
            Document(document_id="d2", title="T2", text="t2"),
        ]
    )

    index_path = tmp_path / "test_dense.faiss"
    retriever.save(index_path)

    assert index_path.exists()
    assert index_path.with_suffix(".meta.json").exists()

    # load 进一个新实例，重新检索，结果应与保存前一致
    other = DenseRetriever(model_name="fake-model", device="cpu")
    other.load(index_path)

    before = retriever.search("q", top_k=2)
    after = other.search("q", top_k=2)

    assert [r.document_id for r in before] == [r.document_id for r in after]
    assert [r.rank for r in before] == [r.rank for r in after]


def test_save_before_build_index_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving before building an index must raise."""
    _patch_model(monkeypatch, {})

    retriever = DenseRetriever(model_name="fake-model", device="cpu")

    with pytest.raises(ValueError, match="Nothing to save"):
        retriever.save("/tmp/should_not_exist.faiss")
