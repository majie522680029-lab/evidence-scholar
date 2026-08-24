"""Hybrid retrieval: fuse BM25 (sparse) and Dense (dense) rankings via RRF.

本文件实现 HybridRetriever：把一个 BM25 检索器和一个 Dense 检索器的
Top-K 结果用 RRF（Reciprocal Rank Fusion，倒数排名融合）合并成一个
统一排序。

为什么用 RRF 而非分数加权：
- BM25 分数（基于 IDF/词频，0~十几）和 Dense 分数（cosine 相似度，0~1）
  量纲差一个数量级，加权法需把两者归一化到同一尺度，归一化方式敏感；
- RRF 只看排名、不看分数，天然绕开量纲问题，不需要归一化；
- RRF 奖励"两个检索器都认同"的文档，恰好利用 BM25（擅长实体词面匹配）
  和 Dense（擅长语义泛化）的互补性。

RRF 公式（Cormack et al. 2009，Elasticsearch/Lucene 默认方案）：

    rrf_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))

其中 rank 从 1 开始，k 是平滑常数（默认 60）。未被某检索器召回的文档
那一项贡献为 0。
"""

from __future__ import annotations

from collections.abc import Sequence

from evidence_scholar.retrieval.base import BaseRetriever
from evidence_scholar.retrieval.schemas import Document, RetrievalResult

DEFAULT_RRF_K = 60


class HybridRetriever(BaseRetriever):
    """Fuse a sparse retriever (BM25) and a dense retriever via RRF.

    采用组合而非多继承：持有一个 BM25 风格的 sparse 检索器和一个 Dense
    检索器实例，search 时各取一份 Top-K 再做 RRF 融合。

    要求两个子检索器都实现 BaseRetriever 契约（build_index / search），
    且 search 返回带 document_id / rank / title / text 的 RetrievalResult。
    """

    def __init__(
        self,
        bm25_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        *,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            bm25_retriever: 稀疏检索器实例（通常 BM25Index）。
            dense_retriever: 稠密检索器实例（通常 DenseRetriever）。
            rrf_k: RRF 平滑常数。值越大，rank 之间的贡献差异越平缓；
                论文默认 60，通常无需调整。

        Raises:
            ValueError: rrf_k <= 0 时抛出（会导致除零/负分）。
        """
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

    def build_index(self, documents: Sequence[Document]) -> None:
        """Build both sub-indexes from the same document set.

        两个子检索器用同一份 documents 建索引，保证检索时在相同语料上
        比较、排名可比。

        Args:
            documents: 待检索的文档序列，每个含 document_id/title/text。

        Raises:
            ValueError: 文档为空时抛出（会由子检索器 build_index 抛出）。
        """
        # 不重复校验空文档——子检索器的 build_index 会各自校验。
        self.bm25_retriever.build_index(documents)
        self.dense_retriever.build_index(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Fuse the two sub-retrievers' rankings via RRF.

        流程：
        1. 两个子检索器各检索 top_k 个文档；
        2. 取两份结果的文档并集；
        3. 对并集中每个文档算 rrf_score；
        4. 按 rrf_score 降序、同分按 document_id 字典序，取前 top_k；
        5. 重排 rank 从 1 开始，title/text 取自任一子结果（内容一致）。

        Args:
            query: 原始问题文本。
            top_k: 返回前 K 个结果。

        Returns:
            按 RRF 融合分降序的检索结果。

        Raises:
            ValueError: top_k <= 0 时抛出。
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        dense_results = self.dense_retriever.search(query, top_k=top_k)

        # rank_by_id[document_id][retriever] = rank，未召回则缺该键。
        rank_by_id: dict[str, dict[str, int]] = {}
        # 任意一份子结果，用于回填 title/text（两子结果内容一致）。
        sample_by_id: dict[str, RetrievalResult] = {}

        for result in bm25_results:
            rank_by_id.setdefault(result.document_id, {})[
                "bm25"
            ] = result.rank
            sample_by_id[result.document_id] = result

        for result in dense_results:
            rank_by_id.setdefault(result.document_id, {})[
                "dense"
            ] = result.rank
            # 若 bm25 已存了 sample，保留 bm25 的（内容一致，无所谓）。
            sample_by_id.setdefault(result.document_id, result)

        # 对并集每个文档算 RRF 融合分。
        scored: list[tuple[float, str]] = []
        for document_id, ranks in rank_by_id.items():
            score = 0.0
            # 未被召回的检索器不贡献（缺键即贡献 0）。
            if "bm25" in ranks:
                score += 1.0 / (self.rrf_k + ranks["bm25"])
            if "dense" in ranks:
                score += 1.0 / (self.rrf_k + ranks["dense"])
            scored.append((score, document_id))

        # 降序：融合分高优先；同分按 document_id 字典序，保证稳定可复现。
        scored.sort(key=lambda item: (-item[0], item[1]))

        fused: list[RetrievalResult] = []
        for rank, (score, document_id) in enumerate(scored[:top_k], start=1):
            sample = sample_by_id[document_id]
            fused.append(
                RetrievalResult(
                    document_id=document_id,
                    title=sample.title,
                    text=sample.text,
                    score=score,
                    rank=rank,
                )
            )

        return fused

    def save(self, path) -> None:
        """Hybrid 不直接持久化索引，save/load 交给子检索器。

        HybridRetriever 本身不持有索引数据，索引在两个子检索器内。
        上层若需持久化，应对两个子检索器分别 save。这里保留契约方法
        但抛出明确错误，避免误用。
        """
        raise NotImplementedError(
            "HybridRetriever does not persist indexes directly; "
            "save each sub-retriever instead."
        )

    def load(self, path) -> None:
        """同 save，不支持直接加载。"""
        raise NotImplementedError(
            "HybridRetriever does not load indexes directly; "
            "load each sub-retriever instead."
        )
