"""Lightweight BM25 document retrieval implementation.

本文件实现了一个不依赖第三方检索库的 BM25 稀疏检索器。

它在 EvidenceScholar 第一周项目中的作用是：
1. 作为最基础、最容易解释的 lexical retrieval baseline；
2. 给 HotpotQA 的 question 和 corpus document 计算词面匹配分数；
3. 产出 Top-K document_id、rank 和 score，供后续评测 Recall/MRR 使用；
4. 为后面的 Dense / Hybrid / Reranker 提供可对比的基线。

当前实现是纯内存索引，适合 HotpotQA 小规模实验和单机调试。
如果后续语料扩展到完整论文库，可以再替换为 Elasticsearch、Lucene
或 Pyserini 等更成熟的倒排索引系统。
"""

from __future__ import annotations

import pickle
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import log1p
from pathlib import Path

from evidence_scholar.retrieval.base import BaseRetriever
from evidence_scholar.retrieval.schemas import Document, RetrievalResult

# 英文 token 正则：
# - 匹配连续的英文字符和数字；
# - 允许 token 内部出现 apostrophe 或 hyphen，例如 don't、state-of-the-art；
# - 不处理中文分词，因为 HotpotQA 是英文数据集。
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenize English text and normalize tokens to lowercase.

    BM25 是基于词项匹配的稀疏检索方法，因此分词策略会直接影响召回效果。
    这里采用一个简单、可复现的英文分词器：
    - 将所有 token 转成小写，避免 Apple/apple 被当作不同词；
    - 丢弃标点符号；
    - 保留数字和带连字符的英文表达。

    Args:
        text: 原始文本，可以是 question、title 或 document body。

    Returns:
        归一化后的 token 列表。

    Raises:
        TypeError: 当输入不是字符串时抛出，防止上游数据解析错误被静默吞掉。
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    # finditer 会按原文顺序返回所有匹配 token；
    # group(0) 是本次正则匹配到的完整字符串。
    return [
        match.group(0).lower()
        for match in TOKEN_PATTERN.finditer(text)
    ]


def build_document_tokens(
    title: str,
    text: str,
    *,
    title_weight: int = 2,
) -> list[str]:
    """Build document tokens with optional title repetition.

    Repeating title tokens gives document titles more influence during
    retrieval without modifying the original document text.

    在 HotpotQA 里，title 往往是实体名或页面名，和问题中的实体词强相关。
    因此这里通过重复 title token 来提高标题词的权重。这样做的好处是简单、
    可解释，并且不会改变原始 Document.text。

    Args:
        title: 文档标题，通常来自 HotpotQA context 的 Wikipedia 页面标题。
        text: 文档正文，通常是若干句子拼接后的段落文本。
        title_weight: 标题 token 重复次数。值越大，标题匹配对分数影响越强。

    Returns:
        用于 BM25 建索引的 token 序列。

    Raises:
        ValueError: title_weight 小于 1 时抛出，因为重复次数必须为正。
    """
    if title_weight < 1:
        raise ValueError("title_weight must be at least 1.")

    title_tokens = tokenize(title)
    body_tokens = tokenize(text)

    # 将标题 token 重复 title_weight 次，等价于提高标题词的 term frequency。
    # 例如 title_weight=2 时，title 中每个词会在索引文本中出现两遍。
    return title_tokens * title_weight + body_tokens


@dataclass(frozen=True, slots=True)
class BM25SearchResult:
    """One ranked BM25 search result.

    这是 BM25Index.search 的轻量返回结构，只包含评测排序所需字段。
    如果需要 title/text 等完整文档内容，可以在上层通过 document_id 回查 corpus。
    """

    document_id: str
    score: float
    rank: int


class BM25Index(BaseRetriever):
    """BM25 index over a fixed, in-memory document collection.

    BM25 的核心思想：
    - 如果 query term 在某篇文档中出现得越多，该文档越相关；
    - 但词频收益会饱和，避免一个词重复很多次就无限加分；
    - 如果 query term 在整个语料中越少见，它越有区分度，IDF 越高；
    - 长文档天然包含更多词，因此需要做长度归一化，避免长文档占便宜。

    本类继承 BaseRetriever，提供两条初始化路径：
    1. build_index(documents)：BaseRetriever 契约入口，接收 Document 对象，
       内部自动分词。推荐用于与 Dense/Hybrid 统一调用的场景。
    2. __init__(document_ids, tokenized_documents, ...)：接收已分词文档，
       适合需要自定义分词逻辑或单元测试快速构造的场景。

    两条路径都会缓存 titles/texts（若提供），使 search 能返回带 title/text
    的 RetrievalResult，与 DenseRetriever 的返回结构一致。
    """

    def __init__(
        self,
        document_ids: Sequence[str],
        tokenized_documents: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        titles: Sequence[str] | None = None,
        texts: Sequence[str] | None = None,
    ) -> None:
        """Initialize a BM25 index from pre-tokenized documents.

        Args:
            document_ids:
                Unique document identifiers.
            tokenized_documents:
                Token sequence for each document.
            k1:
                Controls term-frequency saturation.
                常见取值在 1.2 到 2.0 之间。值越大，词频增长带来的收益越不容易饱和。
            b:
                Controls document-length normalization.
                b=0 表示不做长度归一化；b=1 表示完全按文档长度归一化。
            titles:
                与 document_ids 等长的标题序列，可选。提供后 search 返回的
                RetrievalResult 会带 title，便于上层直接写 rankings 而无需回查 corpus。
            texts:
                与 document_ids 等长的正文序列，可选。作用同 titles。
        """
        # 参数校验和索引进构逻辑统一委托给 _build_internal，
        # build_index 也会复用它，避免两处重复实现。
        self.k1 = k1
        self.b = b
        self._build_internal(
            document_ids=document_ids,
            tokenized_documents=tokenized_documents,
            titles=titles,
            texts=texts,
        )

    def _build_internal(
        self,
        *,
        document_ids: Sequence[str],
        tokenized_documents: Sequence[Sequence[str]],
        titles: Sequence[str] | None,
        texts: Sequence[str] | None,
    ) -> None:
        """Shared index construction logic for __init__ and build_index."""
        # 以下校验尽量在初始化阶段暴露数据问题：
        # - 空语料无法建立 BM25；
        # - document_id 和 tokenized document 必须一一对应；
        # - document_id 必须唯一，否则评测时无法稳定映射 gold evidence。
        if not document_ids:
            raise ValueError("BM25 requires at least one document.")

        if len(document_ids) != len(tokenized_documents):
            raise ValueError(
                "document_ids and tokenized_documents must "
                "have the same length."
            )

        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document_ids must be unique.")

        if self.k1 <= 0:
            raise ValueError("k1 must be greater than zero.")

        if not 0 <= self.b <= 1:
            raise ValueError("b must be between zero and one.")

        self.document_ids = list(document_ids)
        self.document_count = len(self.document_ids)

        # titles/texts 可选：未提供时用占位值，使 RetrievalResult 仍可构造。
        # RetrievalResult.title 有 min_length=1 约束，故占位不能用空串；
        # 这里用 document_id 作为 title 占位（document_id 必非空），可追溯。
        # build_index 路径会传入真实 title/text；__init__ 路径按需传入。
        if titles is not None:
            self.titles = list(titles)
        else:
            self.titles = list(self.document_ids)

        if texts is not None:
            self.texts = list(texts)
        else:
            self.texts = list(self.document_ids)

        # 每篇文档内部的词频统计：
        # term_frequencies[i][term] 表示第 i 篇文档中 term 出现了多少次。
        # BM25 打分时需要频繁查询这个值。
        self.term_frequencies: list[Counter[str]] = [
            Counter(tokens)
            for tokens in tokenized_documents
        ]

        # 每篇文档的总 token 数。长度归一化会用到。
        self.document_lengths = [
            sum(term_frequency.values())
            for term_frequency in self.term_frequencies
        ]

        total_length = sum(self.document_lengths)

        # 允许个别文档没有 token，但不能所有文档都为空；
        # 否则平均文档长度为 0，后续 BM25 公式无法计算。
        if total_length == 0:
            raise ValueError(
                "At least one indexed document must contain tokens."
            )

        # 平均文档长度 avgdl，是 BM25 长度归一化中的基准值。
        self.average_document_length = (
            total_length / self.document_count
        )

        # document_frequencies[term] 表示包含 term 的文档数量，而不是 term 总出现次数。
        # IDF 使用的是文档频率：一个词出现在多少篇文档中。
        self.document_frequencies: Counter[str] = Counter()

        for term_frequency in self.term_frequencies:
            # 这里只遍历 Counter 的 key，确保同一篇文档里某个 term 只贡献 1 次 DF。
            for term in term_frequency:
                self.document_frequencies[term] += 1

    def inverse_document_frequency(self, term: str) -> float:
        """Calculate BM25 inverse document frequency for one term.

        IDF 衡量一个词的区分度：
        - 出现在很多文档中的常见词，IDF 较低；
        - 只出现在少数文档中的实体词、专有名词，IDF 较高。

        这里使用带 0.5 平滑项的 BM25 IDF 变体，并用 log1p 保证结果稳定。
        当 term 不在语料中出现时，document_frequency 为 0，此时 IDF 会很高；
        但因为所有文档里的 frequency 都是 0，它不会真正给任何文档加分。
        """
        document_frequency = self.document_frequencies.get(term, 0)

        # Robertson/Sparck Jones 风格的平滑：
        # numerator   = N - df + 0.5
        # denominator = df + 0.5
        # N 是文档总数，df 是包含该 term 的文档数。
        numerator = (
            self.document_count
            - document_frequency
            + 0.5
        )
        denominator = document_frequency + 0.5

        return log1p(numerator / denominator)

    def score(
        self,
        query_tokens: Sequence[str],
    ) -> list[float]:
        """Calculate BM25 scores for all indexed documents.

        Args:
            query_tokens: 查询文本分词后的 token 序列。

        Returns:
            与 self.document_ids 等长的分数列表，scores[i] 对应第 i 篇文档。

        Notes:
            标准 BM25 通常只考虑 query 中每个不同 term 一次，不额外计算 query
            term frequency。因此这里对 query_tokens 做 set 去重。
        """
        scores = [0.0] * self.document_count

        # Standard BM25 usually treats each distinct query term once.
        for term in set(query_tokens):
            inverse_document_frequency = (
                self.inverse_document_frequency(term)
            )

            # 逐篇文档累加当前 query term 的 BM25 贡献。
            # 这个实现直观但不是最高效：复杂度约为 O(|unique_query_terms| * N)。
            # 对第一周的 500 条样本和小规模 corpus 足够；大规模语料应使用倒排索引优化。
            for index, term_frequency in enumerate(
                self.term_frequencies
            ):
                frequency = term_frequency.get(term, 0)

                # 当前 term 不在该文档中出现，则该 term 对这篇文档没有贡献。
                if frequency == 0:
                    continue

                document_length = self.document_lengths[index]

                # BM25 长度归一化项：
                # - document_length / average_document_length > 1 表示长文档；
                # - 长文档 denominator 会变大，同样词频下得分会被压低；
                # - b 控制这种惩罚的强弱。
                length_normalization = (
                    1
                    - self.b
                    + self.b
                    * document_length
                    / self.average_document_length
                )

                numerator = frequency * (self.k1 + 1)
                denominator = (
                    frequency
                    + self.k1 * length_normalization
                )

                # 单个 query term 对单篇文档的 BM25 分数：
                # IDF(term) * saturated_tf(term, document)
                scores[index] += (
                    inverse_document_frequency
                    * numerator
                    / denominator
                )

        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Rank indexed documents for a raw-text query.

        Args:
            query: 原始问题文本，例如 HotpotQA 的 question。
            top_k: 返回前 K 个结果。

        Returns:
            按 BM25 分数从高到低排序的检索结果。每条结果包含 document_id、
            title、text、score 和 rank，与 DenseRetriever 的返回结构一致。

        Raises:
            ValueError: top_k 小于等于 0 时抛出。
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # search 接收原始 query，因此需要先复用同一个 tokenize 逻辑做归一化。
        # 保证 query 和 document 使用一致的 token 空间。
        query_tokens = tokenize(query)
        scores = self.score(query_tokens)

        # 排序规则：
        # 1. 分数越高越靠前；
        # 2. 分数相同则按原始文档顺序排序，保证结果稳定、可复现。
        ranked_indices = sorted(
            range(self.document_count),
            key=lambda index: (
                -scores[index],
                index,
            ),
        )

        result_count = min(top_k, self.document_count)

        # rank 从 1 开始，符合信息检索评测中 Recall@K / MRR@K 的常见约定。
        # 返回带 title/text 的 RetrievalResult，对齐 BaseRetriever 契约与 DenseRetriever。
        return [
            RetrievalResult(
                document_id=self.document_ids[index],
                title=self.titles[index],
                text=self.texts[index],
                score=scores[index],
                rank=rank,
            )
            for rank, index in enumerate(
                ranked_indices[:result_count],
                start=1,
            )
        ]

    def build_index(self, documents: Sequence[Document]) -> None:
        """Build a BM25 index from Document objects (BaseRetriever contract).

        与 DenseRetriever.build_index 对齐：接收未分词的 Document 列表，
        内部用 build_document_tokens 完成分词和标题加权，再委托给
        _build_internal 构造索引。建完后 titles/texts 已缓存，search 可直接
        返回完整 RetrievalResult，上层无需回查 corpus。

        Args:
            documents: 待检索的文档序列，每个含 document_id/title/text。

        Raises:
            ValueError: 文档为空时抛出。
        """
        if not documents:
            raise ValueError("Cannot build index from empty document set.")

        document_ids = [doc.document_id for doc in documents]
        tokenized_documents = [
            build_document_tokens(
                title=doc.title,
                text=doc.text,
            )
            for doc in documents
        ]
        titles = [doc.title for doc in documents]
        texts = [doc.text for doc in documents]

        self._build_internal(
            document_ids=document_ids,
            tokenized_documents=tokenized_documents,
            titles=titles,
            texts=texts,
        )

    def save(self, path: Path) -> None:
        """Persist the BM25 index state to a single pickle file.

        BM25 的索引是纯 Python 数据结构（Counter / list），不像 Dense 需要
        FAISS 二进制格式，因此用 pickle 整体序列化最简单。

        Args:
            path: 索引保存路径，父目录需存在或可创建。

        Raises:
            ValueError: 索引未建立（document_ids 为空）时抛出。
        """
        if not getattr(self, "document_ids", None):
            raise ValueError("Index not built. Nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "document_ids": self.document_ids,
            "titles": self.titles,
            "texts": self.texts,
            "k1": self.k1,
            "b": self.b,
            "term_frequencies": self.term_frequencies,
            "document_lengths": self.document_lengths,
            "average_document_length": self.average_document_length,
            "document_frequencies": self.document_frequencies,
        }
        with path.open("wb") as file:
            pickle.dump(state, file)

    def load(self, path: Path) -> None:
        """Load a previously persisted BM25 index.

        Args:
            path: 之前用 save 写出的 pickle 路径。
        """
        path = Path(path)
        with path.open("rb") as file:
            state = pickle.load(file)

        self.document_ids = state["document_ids"]
        self.titles = state["titles"]
        self.texts = state["texts"]
        self.k1 = state["k1"]
        self.b = state["b"]
        self.document_count = len(self.document_ids)
        self.term_frequencies = state["term_frequencies"]
        self.document_lengths = state["document_lengths"]
        self.average_document_length = state["average_document_length"]
        self.document_frequencies = state["document_frequencies"]
