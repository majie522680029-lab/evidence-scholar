"""Cross-encoder reranker for precision-oriented re-ranking.

本文件实现 CrossEncoderReranker：对第一阶段检索器召回的候选文档，
用 cross-encoder（query+doc 拼接送入 Transformer 交叉注意）重新打分
排序，把真正相关的顶到前面。

与 DenseRetriever（双塔 Bi-Encoder）的本质区别：
- 双塔：query 和 doc 分别编码成向量再算相似度，doc 可预建索引、
  检索快，但 query 与 doc 互不见面，精度有限——用于全量召回。
- cross-encoder：query 和 doc 拼接送入模型，每个 token 互相 attend，
  精度远高于双塔，但每对 (query, doc) 都要过一遍模型，慢——只能用于
  少量候选（Top-K）精排，不能全量检索。

所以标准两阶段 RAG 链路是：BM25/Dense/Hybrid 召回 Top-K →
CrossEncoderReranker 精排。本类只负责第二阶段。

不继承 BaseRetriever：检索器是"从语料召回"，reranker 是"对给定候选
重排"，职责不同。本类提供 rerank(query, candidates) 接口。

关键特性：rerank 只重排候选、不增删候选，所以 recall@K 不变
（第一阶段没召回的文档，reranker 救不回来），提升的是 hit@1 / MRR。
"""

from __future__ import annotations

from collections.abc import Sequence

# 必须在 import sentence_transformers 之前设 HF_ENDPOINT（镜像），
# 否则 huggingface_hub 在 import 时缓存"用主站"配置导致模型下载超时。
# 与 dense.py 同样的处理。
from evidence_scholar.config import load_config

load_config()

from sentence_transformers import CrossEncoder  # noqa: E402

from evidence_scholar.retrieval.schemas import RetrievalResult  # noqa: E402


class CrossEncoderReranker:
    """Re-rank candidates with a cross-encoder model.

    输入第一阶段检索返回的 list[RetrievalResult]（已带 title/text），
    对每个候选用 (query, text) 配对喂 cross-encoder 打分，按新分数
    重排，返回新的 RetrievalResult（rank 从 1 开始）。
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: cross-encoder 模型名，例如
                cross-encoders/ms-marco-MiniLM-L-6-v2。
            device: 推理设备，例如 cpu / cuda / cuda:0。
            max_length: cross-encoder 输入截断长度。超过的 (query+doc)
                对会被截断。512 是 MiniLM 系列的标准上限。

        Notes:
            构造时即加载模型。调用方应在构造前调用 load_config()
            确保 HF_ENDPOINT 已设为镜像——本模块顶部已自动触发。
        """
        if not model_name:
            raise ValueError("model_name must not be empty.")

        if max_length <= 0:
            raise ValueError("max_length must be greater than zero.")

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        # 加载 cross-encoder。CrossEncoder 内部对 (query, doc) 配对
        # 拼接、做交叉注意力、输出单分数。这里不 catch 异常——
        # 模型加载失败应尽早暴露。
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Re-rank candidates by cross-encoder score.

        Args:
            query: 原始问题文本。
            candidates: 第一阶段检索返回的候选列表（带 title/text）。
            top_k: 返回前 K 个。通常等于 candidates 数量（全量重排）。

        Returns:
            按 cross-encoder 分数降序重排的 RetrievalResult。title/text
            沿用候选原值，score 替换为新分数，rank 从 1 重新编号。

        Raises:
            ValueError: top_k <= 0 时抛出。

        Notes:
            rerank 不增删候选，只重排——所以 recall@K 不变。
            若 top_k < len(candidates)，会丢弃末尾候选项，但召回率
            在 top_k 处的统计仍由上层评测决定。
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not candidates:
            return []

        # 构造 (query, text) 配对，喂 cross-encoder 打分。
        # 用 candidate.text（正文）而非 title，因为正文信息更完整。
        pairs = [(query, candidate.text) for candidate in candidates]

        # CrossEncoder.predict 接收文本对列表，返回每对的分数数组。
        scores = self.model.predict(pairs)

        # 按 cross-encoder 分数降序排；同分按原 candidates 顺序，
        # 保证稳定可复现。
        indexed = list(enumerate(candidates))
        indexed.sort(
            key=lambda item: (
                -float(scores[item[0]]),
                item[0],
            )
        )

        result_count = min(top_k, len(candidates))

        reranked: list[RetrievalResult] = []
        for rank, (original_index, candidate) in enumerate(
            indexed[:result_count],
            start=1,
        ):
            reranked.append(
                RetrievalResult(
                    document_id=candidate.document_id,
                    title=candidate.title,
                    text=candidate.text,
                    score=float(scores[original_index]),
                    rank=rank,
                )
            )

        return reranked
