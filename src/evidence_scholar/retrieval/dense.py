"""Dense (embedding-based) document retrieval using FAISS.

本文件实现 DenseRetriever：用 sentence-transformers 把文档和 query 编码成
向量，用 FAISS 建索引并做相似度检索，返回 Top-K 结果。

它在 EvidenceScholar 项目中的作用是：
1. 作为 BM25（词面匹配）之外的语义检索基线；
2. 给 HotpotQA 的 question 和 corpus document 算语义相似度；
3. 产出 Top-K document_id / title / text / score / rank，供评测使用；
4. 为后续 Hybrid（BM25 + Dense 融合）和 Reranker 提供可对比的 dense 基线。

与 BM25Index 的关键区别：
- BM25Index 没有继承 BaseRetriever，返回轻量的 BM25SearchResult；
- DenseRetriever 严格继承 BaseRetriever，返回完整的 RetrievalResult
  （带 title/text），作为检索器接口统一的标杆实现。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

# 必须在 import sentence_transformers 之前设 HF_ENDPOINT，否则
# huggingface_hub 在 import 时会缓存"用主站"的配置，后续无法覆盖。
# 国内无法直连 huggingface.co，必须走镜像（见 configs/retrieval.yaml）。
# 这里 import config 模块并调用 load_config 触发环境变量设置，
# 代价是读取默认配置文件；若配置文件缺失会抛 FileNotFoundError，
# 这是合理的——没有镜像配置 DenseRetriever 根本无法在当前网络下加载模型。
from evidence_scholar.config import load_config

load_config()

from sentence_transformers import SentenceTransformer

from evidence_scholar.retrieval.base import BaseRetriever
from evidence_scholar.retrieval.schemas import Document, RetrievalResult


class DenseRetriever(BaseRetriever):
    """Embedding-based retriever backed by sentence-transformers + FAISS.

    工作流程：
    - build_index：编码全部文档 → 向量归一化 → 建 FAISS IndexFlatIP；
    - search：编码 query → 在 FAISS 中检索最近邻 → 拼成 RetrievalResult；
    - save/load：FAISS 存向量，侧车 JSON 存 document_id/title/text。

    设计要点：
    1. 严格实现 BaseRetriever 契约，作为接口统一的标杆；
    2. 向量归一化 + IndexFlatIP（内积）等价于 cosine 相似度，
       是语义检索的标准度量；
    3. device / model_name / batch_size 都从 config 传入，不硬编码，
       换机器（无 GPU）只改 config 即可；
    4. FAISS 只存向量，文档元信息用侧车 JSON 保存。
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the dense retriever.

        Args:
            model_name: sentence-transformers 模型名或本地路径，
                例如 sentence-transformers/all-MiniLM-L6-v2。
            device: 推理设备，例如 cpu / cuda / cuda:0。
            batch_size: 编码文档/查询时的批大小，防止 GPU OOM。
            normalize_embeddings: 是否归一化向量。
                归一化后内积 == cosine 相似度，配合 IndexFlatIP 使用。

        Notes:
            构造时即加载 embedding 模型。调用方应在构造前调用
            load_config() 以确保 HF_ENDPOINT 已设为镜像，
            否则模型下载会尝试直连 huggingface.co 并超时。
        """
        if not model_name:
            raise ValueError("model_name must not be empty.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # 加载 embedding 模型。这里不 catch 异常——
        # 模型下载/加载失败应尽早暴露，而不是静默退化。
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        # 索引在 build_index 后填充；load 后也会填。
        self.index: faiss.Index | None = None
        self.document_ids: list[str] = []
        self.titles: list[str] = []
        self.texts: list[str] = []
        self.embedding_dim: int | None = None

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts to a (N, dim) float32 numpy array.

        Args:
            texts: 待编码的文本列表。

        Returns:
            形状 (N, dim) 的 float32 数组。若开启归一化，则每行为单位向量。

        Notes:
            返回 numpy 而非 torch 张量，因为 FAISS 接受 numpy。
            sentence-transformers 6.x 的 encode 默认就返回 numpy。
        """
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # FAISS 要求 float32、C 连续。
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def build_index(self, documents: Sequence[Document]) -> None:
        """Build a FAISS index from documents.

        Args:
            documents: 待检索的文档序列，每个含 document_id/title/text。

        Raises:
            ValueError: 文档为空时抛出。
        """
        if not documents:
            raise ValueError("Cannot build index from empty document set.")

        self.document_ids = [doc.document_id for doc in documents]
        self.titles = [doc.title for doc in documents]
        self.texts = [doc.text for doc in documents]

        # 校验 document_id 唯一，否则检索结果无法稳定映射回文档。
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("document_ids must be unique.")

        embeddings = self._encode(self.texts)
        self.embedding_dim = embeddings.shape[1]

        # 归一化向量 + IndexFlatIP（内积）== cosine 相似度。
        # IndexFlat 是精确索引、无近似，对本项目小规模 corpus 足够；
        # 大规模语料可换 IVF + PQ 做近似检索。
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Return documents ranked by semantic similarity to the query.

        Args:
            query: 原始问题文本。
            top_k: 返回前 K 个结果。

        Returns:
            按 FAISS 相似度从高到低排序的检索结果，每条含
            document_id/score/rank/title/text。

        Raises:
            ValueError: top_k <= 0 或索引未建立时抛出。
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # 编码单条 query，形状 (1, dim)。
        query_embedding = self._encode([query])
        # 距离是内积分数（归一化后即 cosine），无需再排序方向调整。
        scores, indices = self.index.search(query_embedding, top_k)

        results: list[RetrievalResult] = []
        for rank, position in enumerate(indices[0], start=1):
            # FAISS 可能返回 -1（当 top_k > 文档数时的填充值），跳过。
            if position < 0:
                continue

            position = int(position)
            results.append(
                RetrievalResult(
                    document_id=self.document_ids[position],
                    score=float(scores[0][rank - 1]),
                    rank=rank,
                    title=self.titles[position],
                    text=self.texts[position],
                )
            )

        return results

    def save(self, path: Path) -> None:
        """Persist the FAISS index and document metadata.

        写两个文件：
        - path：FAISS 索引（仅向量）；
        - path.meta.json：document_id/title/text 映射。

        Args:
            path: FAISS 索引保存路径，父目录需存在或可创建。

        Raises:
            ValueError: 索引未建立时抛出。
        """
        if self.index is None:
            raise ValueError("Index not built. Nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))

        meta_path = path.with_suffix(".meta.json")
        meta = {
            "document_ids": self.document_ids,
            "titles": self.titles,
            "texts": self.texts,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "normalize_embeddings": self.normalize_embeddings,
        }
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False)

    def load(self, path: Path) -> None:
        """Load a previously persisted FAISS index and metadata.

        Args:
            path: 之前用 save 写出的 FAISS 索引路径。
        """
        path = Path(path)
        self.index = faiss.read_index(str(path))

        meta_path = path.with_suffix(".meta.json")
        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)

        self.document_ids = meta["document_ids"]
        self.titles = meta["titles"]
        self.texts = meta["texts"]
        self.embedding_dim = meta["embedding_dim"]
