"""Tests for the evidence accumulation pool (B4 layer 1).

纯逻辑测，不调 LLM/不占卡。覆盖去重、累积、摘要格式、空池。
和 A 阶段/B2/B3 测试同套路，秒级跑完。
"""

from __future__ import annotations

from evidence_scholar.agent.evidence_pool import Evidence, EvidencePool
from evidence_scholar.retrieval.schemas import RetrievalResult


def _make_result(
    doc_id: str = "d1",
    *,
    rank: int = 1,
    score: float = 0.5,
    title: str | None = None,
    text: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        document_id=doc_id,
        title=title or f"Title {doc_id}",
        text=text or f"Body of {doc_id}.",
        score=score,
        rank=rank,
    )


# --- 基础 add ---

def test_add_single_hop() -> None:
    """一跳加 3 条 → 池里 3 条，各 hit_count=1。"""
    pool = EvidencePool()
    pool.add(
        [_make_result("d1"), _make_result("d2"), _make_result("d3")],
        query="q1", hop=0,
    )
    assert pool.size == 3
    assert all(ev.hit_count == 1 for ev in pool.items)


def test_add_preserves_rank_and_query() -> None:
    """add 记录 rank/source_query/hop 用于溯源。"""
    pool = EvidencePool()
    pool.add([_make_result("d1", rank=2, score=0.7)], query="my query", hop=1)
    ev = pool.items[0]
    assert ev.source_query == "my query"
    assert ev.hop == 1
    assert ev.rank == 2
    assert ev.score == 0.7


# --- 去重 ---

def test_add_dedupes_by_document_id() -> None:
    """同一 doc_id 两跳命中 → 只留一份，hit_count=2。"""
    pool = EvidencePool()
    pool.add([_make_result("d1"), _make_result("d2")], query="q1", hop=0)
    pool.add([_make_result("d1"), _make_result("d3")], query="q2", hop=1)

    assert pool.size == 3  # d1/d2/d3，d1 不重复
    d1 = next(ev for ev in pool.items if ev.document_id == "d1")
    assert d1.hit_count == 2  # 被两跳命中
    # d1 的溯源保留首次命中（hop=0, query="q1"）
    assert d1.hop == 0
    assert d1.source_query == "q1"


def test_add_same_hop_dedup() -> None:
    """同一跳内重复 doc_id（不应发生但容错）→ 去重，hit_count=2。"""
    pool = EvidencePool()
    pool.add([_make_result("d1"), _make_result("d1")], query="q", hop=0)
    assert pool.size == 1
    assert pool.items[0].hit_count == 2


# --- 累积 ---

def test_multi_hop_accumulation() -> None:
    """跳1 加 5 条、跳2 加 5 条（其中 2 条重复）→ 池里 8 条。"""
    pool = EvidencePool()
    pool.add([_make_result(f"d{i}") for i in range(5)], query="q1", hop=0)
    # d0/d1 和跳1重复，d5-d7 是新的
    pool.add(
        [_make_result("d0"), _make_result("d1"),
         _make_result("d5"), _make_result("d6"), _make_result("d7")],
        query="q2", hop=1,
    )
    assert pool.size == 8  # 5 + 3 新（2 重复）
    # 重复的 d0/d1 hit_count=2
    assert next(e for e in pool.items if e.document_id == "d0").hit_count == 2
    # 新的 d5 hit_count=1
    assert next(e for e in pool.items if e.document_id == "d5").hit_count == 1


# --- summarize 格式 ---

def test_summarize_empty_pool() -> None:
    """空池 → 明确"无证据"提示（judge 该判 insufficient）。"""
    pool = EvidencePool()
    summary = pool.summarize()
    assert "empty" in summary.lower() or "no evidence" in summary.lower()


def test_summarize_has_count_and_hops() -> None:
    """摘要含总数 + 跳数。"""
    pool = EvidencePool()
    pool.add([_make_result("d1"), _make_result("d2")], query="q1", hop=0)
    pool.add([_make_result("d3")], query="q2", hop=1)
    summary = pool.summarize()
    assert "3 items" in summary
    assert "2 hops" in summary


def test_summarize_has_query_and_titles() -> None:
    """摘要含各跳 query + 文档标题 + score。"""
    pool = EvidencePool()
    pool.add(
        [_make_result("d1", rank=1, score=0.012, title="Ed Wood (1994 film)")],
        query="Ed Wood director", hop=0,
    )
    summary = pool.summarize()
    assert "Ed Wood director" in summary  # query 出现
    assert "Ed Wood (1994 film)" in summary  # 标题出现
    assert "0.012" in summary  # score 出现


def test_summarize_groups_by_hop() -> None:
    """多跳摘要按 hop 分段（[Hop 0]...[Hop 1]...）。"""
    pool = EvidencePool()
    pool.add([_make_result("d1")], query="q1", hop=0)
    pool.add([_make_result("d2")], query="q2", hop=1)
    summary = pool.summarize()
    assert "[Hop 0]" in summary
    assert "[Hop 1]" in summary
    # hop 0 的段在 hop 1 前
    assert summary.index("[Hop 0]") < summary.index("[Hop 1]")


def test_summarize_truncates_long_text() -> None:
    """摘要里长正文被截断（防撑爆 judge 上下文）。"""
    long_text = "x" * 1000
    pool = EvidencePool()
    pool.add([_make_result("d1", text=long_text)], query="q", hop=0)
    summary = pool.summarize()
    # 截断后含省略号，且远短于原文
    assert "…" in summary
    # 摘要里的正文片段不应包含完整 1000 字符
    assert "x" * 500 not in summary


def test_summarize_marks_multi_hop_hits() -> None:
    """被多跳命中的证据标 ×N（强证据信号）。"""
    pool = EvidencePool()
    pool.add([_make_result("d1")], query="q1", hop=0)
    pool.add([_make_result("d1")], query="q2", hop=1)
    summary = pool.summarize()
    assert "×2" in summary  # d1 被两跳命中


def test_summarize_empty_within_hop_skips_gracefully() -> None:
    """边界：add 空列表 → 池不变、摘要正常。"""
    pool = EvidencePool()
    pool.add([], query="q", hop=0)
    assert pool.size == 0
    assert "empty" in pool.summarize().lower()


def test_size_after_mixed_adds() -> None:
    """size 反映去重后的不同文档数。"""
    pool = EvidencePool()
    pool.add([_make_result("d1"), _make_result("d2")], query="q", hop=0)
    pool.add([_make_result("d1")], query="q2", hop=1)  # 重复
    assert pool.size == 2


# --- Evidence 数据类 ---

def test_evidence_default_hit_count() -> None:
    """Evidence 默认 hit_count=1。"""
    ev = Evidence(
        source_query="q", hop=0, rank=1,
        document_id="d1", title="t", text="x", score=0.5,
    )
    assert ev.hit_count == 1
