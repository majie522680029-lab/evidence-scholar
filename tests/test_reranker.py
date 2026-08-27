"""Tests for the CrossEncoderReranker implementation.

设计原则：不加载真模型、不依赖 GPU。
通过 monkeypatch 把 reranker 模块里的 CrossEncoder 替换成一个返回
预设分数的 _FakeCrossEncoder，使测试秒级完成、可精确预测重排结果。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from evidence_scholar.retrieval import reranker as reranker_module
from evidence_scholar.retrieval.reranker import CrossEncoderReranker
from evidence_scholar.retrieval.schemas import RetrievalResult


class _FakeCrossEncoder:
    """A controllable stand-in for sentence_transformers.CrossEncoder.

    预置一个 (query, doc_text) -> score 的函数，predict 时查表返回。
    未知配对返回 0。
    """

    def __init__(
        self,
        score_fn=None,
        scores_by_text: dict[str, float] | None = None,
    ) -> None:
        self.score_fn = score_fn
        self.scores_by_text = scores_by_text or {}

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Return preset scores for each (query, text) pair."""
        results = []
        for query, text in pairs:
            if self.score_fn is not None:
                results.append(float(self.score_fn(query, text)))
            else:
                results.append(float(self.scores_by_text.get(text, 0.0)))
        return np.array(results, dtype=np.float32)


def _patch_cross_encoder(
    monkeypatch: pytest.MonkeyPatch,
    fake_instance: _FakeCrossEncoder,
) -> None:
    """Patch CrossEncoder in reranker.py to return a fake instance."""
    def fake_cls(model_name, device=None, max_length=512, *args, **kwargs):
        return fake_instance

    monkeypatch.setattr(reranker_module, "CrossEncoder", fake_cls)


def _result(
    document_id: str,
    title: str = "T",
    text: str = "X",
    score: float = 0.0,
    rank: int = 1,
) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id,
        title=title,
        text=text,
        score=score,
        rank=rank,
    )


# --- 参数校验测试 ---------------------------------------------------------


def test_empty_model_name_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty model_name must be rejected before loading."""
    _patch_cross_encoder(monkeypatch, _FakeCrossEncoder())

    with pytest.raises(ValueError, match="model_name"):
        CrossEncoderReranker(model_name="", device="cpu")


def test_non_positive_max_length_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_length <= 0 must be rejected."""
    _patch_cross_encoder(monkeypatch, _FakeCrossEncoder())

    with pytest.raises(ValueError, match="max_length"):
        CrossEncoderReranker(
            model_name="fake",
            device="cpu",
            max_length=0,
        )


# --- rerank 测试 ----------------------------------------------------------


def test_rerank_reorders_by_cross_encoder_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rerank 应按 cross-encoder 分数重排，而非保留原顺序。

    给三个候选，分数设定让顺序倒过来：C > B > A。
    """
    fake = _FakeCrossEncoder(
        scores_by_text={"textA": 0.1, "textB": 0.5, "textC": 0.9}
    )
    _patch_cross_encoder(monkeypatch, fake)

    reranker = CrossEncoderReranker(model_name="fake", device="cpu")

    candidates = [
        _result("A", text="textA", rank=1),
        _result("B", text="textB", rank=2),
        _result("C", text="textC", rank=3),
    ]

    results = reranker.rerank("query", candidates, top_k=3)

    assert [r.document_id for r in results] == ["C", "B", "A"]
    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.5)
    assert results[2].score == pytest.approx(0.1)


def test_rerank_ranks_are_consecutive_from_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """返回的 rank 必须从 1 连续。"""
    fake = _FakeCrossEncoder(
        scores_by_text={"t1": 0.9, "t2": 0.5, "t3": 0.1}
    )
    _patch_cross_encoder(monkeypatch, fake)

    reranker = CrossEncoderReranker(model_name="fake", device="cpu")
    candidates = [
        _result("d1", text="t1", rank=1),
        _result("d2", text="t2", rank=2),
        _result("d3", text="t3", rank=3),
    ]

    results = reranker.rerank("q", candidates, top_k=3)

    assert [r.rank for r in results] == [1, 2, 3]


def test_rerank_preserves_title_and_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rerank 后 title/text 应沿用候选原值。"""
    fake = _FakeCrossEncoder(scores_by_text={"alpha": 0.9})
    _patch_cross_encoder(monkeypatch, fake)

    reranker = CrossEncoderReranker(model_name="fake", device="cpu")
    candidates = [
        _result("d1", title="The Title", text="alpha", rank=1),
    ]

    results = reranker.rerank("q", candidates, top_k=1)

    assert results[0].title == "The Title"
    assert results[0].text == "alpha"


def test_rerank_preserves_candidate_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rerank 不增删候选——输入 N 个，输出仍 N 个（top_k >= N 时）。

    这是 reranker 的关键特性：只重排、不增删，所以 recall 不变。
    """
    fake = _FakeCrossEncoder(
        scores_by_text={"t1": 0.9, "t2": 0.5, "t3": 0.1, "t4": 0.3}
    )
    _patch_cross_encoder(monkeypatch, fake)

    reranker = CrossEncoderReranker(model_name="fake", device="cpu")
    candidates = [
        _result(f"d{i}", text=f"t{i}", rank=i) for i in range(1, 5)
    ]

    results = reranker.rerank("q", candidates, top_k=10)

    assert len(results) == 4
    assert {r.document_id for r in results} == {"d1", "d2", "d3", "d4"}


def test_rerank_top_k_truncates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """top_k 小于候选数时，应只返回前 top_k 个（按新分数）。"""
    fake = _FakeCrossEncoder(
        scores_by_text={"t1": 0.9, "t2": 0.5, "t3": 0.1}
    )
    _patch_cross_encoder(monkeypatch, fake)

    reranker = CrossEncoderReranker(model_name="fake", device="cpu")
    candidates = [
        _result("d1", text="t1", rank=1),
        _result("d2", text="t2", rank=2),
        _result("d3", text="t3", rank=3),
    ]

    results = reranker.rerank("q", candidates, top_k=2)

    assert len(results) == 2
    assert [r.document_id for r in results] == ["d1", "d2"]


def test_rerank_empty_candidates_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空候选应返回空列表，不报错。"""
    _patch_cross_encoder(monkeypatch, _FakeCrossEncoder())

    reranker = CrossEncoderReranker(model_name="fake", device="cpu")

    results = reranker.rerank("q", [], top_k=5)

    assert results == []


def test_rerank_rejects_non_positive_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """top_k <= 0 必须报错。"""
    _patch_cross_encoder(monkeypatch, _FakeCrossEncoder())

    reranker = CrossEncoderReranker(model_name="fake", device="cpu")

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("q", [_result("d1", text="x")], top_k=0)


def test_rerank_tie_break_is_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同分时按原 candidates 顺序排，保证稳定可复现。

    d1 和 d2 都给 0.5 分，d1 原序在前，应排第一。
    """
    fake = _FakeCrossEncoder(
        scores_by_text={"t1": 0.5, "t2": 0.5, "t3": 0.9}
    )
    _patch_cross_encoder(monkeypatch, fake)

    reranker = CrossEncoderReranker(model_name="fake", device="cpu")
    candidates = [
        _result("d1", text="t1", rank=1),
        _result("d2", text="t2", rank=2),
        _result("d3", text="t3", rank=3),
    ]

    results = reranker.rerank("q", candidates, top_k=3)

    # d3 分最高排第一；d1/d2 同分，d1 原序在前排第二
    assert [r.document_id for r in results] == ["d3", "d1", "d2"]


def test_rerank_does_not_change_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rerank 不增删候选——金证据仍在结果集里（recall 不变的核心）。

    模拟：候选里有金证据 gold，即使 rerank 把它排到末尾，
    它仍在返回的候选集中，所以 recall@K（K=候选数）不变。
    """
    fake = _FakeCrossEncoder(
        scores_by_text={
            "gold_text": 0.1,  # 金证据但 rerank 给低分
            "noise_text": 0.9,  # 噪声但 rerank 给高分
        }
    )
    _patch_cross_encoder(monkeypatch, fake)

    reranker = CrossEncoderReranker(model_name="fake", device="cpu")
    candidates = [
        _result("noise", text="noise_text", rank=1),
        _result("gold", text="gold_text", rank=2),
    ]

    results = reranker.rerank("q", candidates, top_k=10)

    # gold 被排到第二，但仍在结果集——recall@2 不变
    ids = {r.document_id for r in results}
    assert "gold" in ids
    assert results[0].document_id == "noise"
    assert results[1].document_id == "gold"
