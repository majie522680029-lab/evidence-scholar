"""Tool-calling layer: expose retrievers as OpenAI-format tools for an LLM agent.

B2 的职责：把检索器包装成 LLM 能通过 tool-calling 协议调用的"工具"。

为什么单独一层：
- agent loop（B3）只管"调 LLM → 解析 tool_call → 执行 → 喂回 LLM"的循环逻辑，
  不该同时操心"工具签名对不对、参数校验、结果怎么序列化成 LLM 能读的文本"。
  把工具层单独抽出来，先单独测通，loop 写起来只需管控制流。

设计要点：
- 工具层不持有索引生命周期。构造时传入一个已 build_index 过的 retriever，
  索引建/换由上层（B3 的 agent runner / B6 的评测器）负责。这让本层是
  纯逻辑，可用 fake retriever 单元测试，不碰 GPU/真模型。
- 返回给 LLM 的是结构化文本（标题 + 正文片段），不是原始 pydantic 对象——
  LLM 只能读文本。正文做长度截断，防爆上下文。
- B2 只暴露 hybrid 一个工具（单工具起步），让 agent 先把 loop 跑通；
  后续可开放 bm25/dense 多工具让 agent 学选择策略。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from evidence_scholar.retrieval.base import BaseRetriever
from evidence_scholar.retrieval.schemas import RetrievalResult

# 给 LLM 看的正文片段最大字符数。HotpotQA 单篇 wiki 段落通常几百到一两千
# 字符；留 800 既够 LLM 抓实体，又不至于 top-10 把上下文撑爆。
_MAX_DOC_TEXT_CHARS = 800


def _format_result_for_llm(result: RetrievalResult) -> dict[str, Any]:
    """把一条 RetrievalResult 序列化成 LLM 可读的 dict。

    正文做字符截断，防止长文档把上下文窗撑爆。保留 rank/score 让 LLM
    知道检索器对这个文档的置信度（RRF 融合分或 cross-encoder 分）。
    """
    text = result.text
    if len(text) > _MAX_DOC_TEXT_CHARS:
        # 截断时补省略号，让 LLM 知道被截了、不是原文就这样结束。
        text = text[:_MAX_DOC_TEXT_CHARS] + "…"

    return {
        "rank": result.rank,
        "document_id": result.document_id,
        "title": result.title,
        "text": text,
        "score": round(float(result.score), 4),
    }


class RetrievalTools:
    """把一个已初始化的 retriever 暴露为 OpenAI tool-calling 工具。

    用法（上层 agent runner）：
        retriever = HybridRetriever(...)  # 已 build_index
        tools = RetrievalTools(retriever)
        # 传 tools.schema 给 LLM；LLM 回 tool_call 时调 tools.execute
    """

    # 工具名固定常量，避免散落字符串拼写错误。
    HYBRID_TOOL_NAME = "retrieve_hybrid"

    def __init__(self, retriever: BaseRetriever) -> None:
        """Initialize with an already-indexed retriever.

        Args:
            retriever: 已 build_index 的检索器。agent 调 retrieve_hybrid 时
                实际执行 retriever.search(query, top_k)。索引生命周期由上层
                负责，本类不管 build_index。
        """
        self._retriever = retriever

    @property
    def schema(self) -> list[dict[str, Any]]:
        """OpenAI tools 格式的工具签名列表。

        每个工具含 name / description / parameters(JSON Schema)。这个列表
        原样塞进 LLM 请求的 `tools` 字段即可。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": self.HYBRID_TOOL_NAME,
                    "description": (
                        "Search the document corpus for passages relevant "
                        "to a query. Returns ranked documents with title "
                        "and a text excerpt. Use this to gather evidence "
                        "for answering the user's question; call it with "
                        "a reformulated sub-query when you need more "
                        "information."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "A natural-language or keyword "
                                    "sub-query to search for. Reformulate "
                                    "the user question into a focused "
                                    "retrieval query."
                                ),
                            },
                            "top_k": {
                                "type": "integer",
                                "description": (
                                    "Number of top documents to return. "
                                    "Default 10."
                                ),
                                "default": 10,
                                "minimum": 1,
                                "maximum": 20,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute a tool call and return its result as an LLM-readable string.

        Args:
            name: 工具名（来自 LLM 的 tool_call.name）。
            arguments: 工具参数（来自 LLM 的 tool_call.arguments 解析后的 dict）。

        Returns:
            JSON 字符串，含 documents 列表（或 error 字段）。JSON 是为了让
            LLM 看到结构化字段（rank/title/text），而不是一坨自由文本。

        Raises:
            ValueError: 工具名未知，或参数不合法（query 空/top_k 越界）。
        """
        if name != self.HYBRID_TOOL_NAME:
            raise ValueError(
                f"Unknown tool name: {name!r}. "
                f"Expected {self.HYBRID_TOOL_NAME!r}."
            )

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Argument 'query' must be a non-empty string."
            )

        top_k = arguments.get("top_k", 10)
        # OpenAI tool calling 里参数可能以 int 或 str 传来，统一强转+校验。
        try:
            top_k = int(top_k)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Argument 'top_k' must be an integer, got {top_k!r}."
            ) from error

        if top_k <= 0 or top_k > 20:
            raise ValueError(
                f"Argument 'top_k' must be in [1, 20], got {top_k}."
            )

        results: Sequence[RetrievalResult] = self._retriever.search(
            query, top_k=top_k
        )

        # 序列化成 LLM 可读结构。空结果也要明确告知，避免 LLM 误以为
        # "没返回"等于"语料里确实没有"。
        payload: dict[str, Any]
        if not results:
            payload = {
                "query": query,
                "documents": [],
                "note": "No documents matched this query.",
            }
        else:
            payload = {
                "query": query,
                "documents": [_format_result_for_llm(r) for r in results],
            }

        return json.dumps(payload, ensure_ascii=False)
