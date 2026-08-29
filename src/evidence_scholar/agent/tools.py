"""Tool-calling layer: expose retrievers + judge as OpenAI-format tools.

B2 的职责：把检索器包装成 LLM 能通过 tool-calling 协议调用的"工具"。
B5 在此基础上加了第二个工具 judge_evidence——让 LLM 通过 tool_call
表达"证据充分性判断"，而不是输出 JSON 文本。

为什么 judge 也用工具表达（B5 的关键设计调整）：
- 原设计让 judge 输出 JSON 字符串（guided_json / JSON mode）。但 Qwen3
  默认开 thinking mode，思考文本会裹在 JSON 前、甚至占满 max_tokens 把
  JSON 挤没，实测 guided_json 约束不住 thinking。
- tool calling 走专用结构化通道：vLLM 的 tool-call parser（hermes）在
  解码阶段就约束 tool_call 参数满足 schema，thinking 影响 content 不影响
  tool_call 参数。联调已证明 Qwen3 的 retrieve_hybrid tool_call 参数 JSON
  完全正确，复用这条稳的通道表达 judge 判断。
- 这样 agent 只会"调工具"一种结构化动作，架构统一，不引入 JSON 模式第二套
  结构化机制。

JSON 不稳的五层防御（针对截断/漂移/思考混入）：
1. tool calling 通道治本（最稳的结构化路径）
2. judge 调用放宽 max_tokens（让参数有空间生成完）
3. 客户端解析容错（JSON 解析失败兜底成空 dict，不崩 loop）
4. judge schema 精简：reason 限 maxLength 防小作文截断，sufficient 必填防漂移
5. B3 loop 的 max_steps 兜底（judge 全坏了也不死循环）

设计要点：
- 工具层不持有索引生命周期。构造时传入一个已 build_index 过的 retriever，
  索引建/换由上层（B3 的 agent runner / B6 的评测器）负责。这让本层是
  纯逻辑，可用 fake retriever 单元测试，不碰 GPU/真模型。
- 返回给 LLM 的是结构化文本（标题 + 正文片段），不是原始 pydantic 对象——
  LLM 只能读文本。正文做长度截断，防爆上下文。
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
    JUDGE_TOOL_NAME = "judge_evidence"

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

        两个工具：retrieve_hybrid（查证据）+ judge_evidence（判够不够）。
        原样塞进 LLM 请求的 tools 字段。
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
            },
            {
                "type": "function",
                "function": {
                    "name": self.JUDGE_TOOL_NAME,
                    "description": (
                        "Judge whether the evidence gathered so far is "
                        "sufficient to answer the user's question. Call "
                        "this after every retrieval. If sufficient is "
                        "true, provide the final answer. If false, "
                        "provide next_query for the next retrieval. "
                        "Always call this before answering."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sufficient": {
                                "type": "boolean",
                                "description": (
                                    "True if gathered evidence is enough "
                                    "to answer the question."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": (
                                    "Why sufficient or not. If not, "
                                    "state what is still missing."
                                ),
                                # maxLength 防模型写小作文撑爆 max_tokens、
                                # 把整个 tool_call 参数截断（防御层 4）。
                                "maxLength": 200,
                            },
                            "next_query": {
                                "type": "string",
                                "description": (
                                    "Only when sufficient is false: the "
                                    "focused query to search next."
                                ),
                            },
                            "answer": {
                                "type": "string",
                                "description": (
                                    "Only when sufficient is true: the "
                                    "final answer grounded in evidence."
                                ),
                            },
                        },
                        # sufficient 必填，防漂移（防御层 4）。
                        "required": ["sufficient"],
                    },
                },
            },
        ]

    def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute a tool call and return its result as an LLM-readable string.

        Args:
            name: 工具名（来自 LLM 的 tool_call.name）。
            arguments: 工具参数（来自 LLM 的 tool_call.arguments 解析后的 dict）。

        Returns:
            JSON 字符串，含 documents 列表（retrieve_hybrid）或 judge 回执
            （judge_evidence）。JSON 让 LLM 看到结构化字段，而非自由文本。

        Raises:
            ValueError: 工具名未知，或参数不合法。
        """
        if name == self.JUDGE_TOOL_NAME:
            return self._execute_judge(arguments)

        if name == self.HYBRID_TOOL_NAME:
            return self._execute_hybrid(arguments)

        raise ValueError(
            f"Unknown tool name: {name!r}. "
            f"Expected one of {self.HYBRID_TOOL_NAME!r}, "
            f"{self.JUDGE_TOOL_NAME!r}."
        )

    def _execute_hybrid(self, arguments: dict[str, Any]) -> str:
        """retrieve_hybrid 的执行：真去查语料，返回文档列表。"""
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

    def _execute_judge(self, arguments: dict[str, Any]) -> str:
        """judge_evidence 的执行：不查语料，只回执确认 + 透传判断。

        judge 是"声明判断"不是"执行动作"——它告诉 loop 够不够/下一步查啥，
        本身不需要副作用。execute 回执一个确认消息给 LLM，让 LLM 知道判断
        已记录（loop 会据 sufficient 决定作答或继续）。
        """
        sufficient = arguments.get("sufficient")
        # 容错：LLM 偶尔把 boolean 传成字符串。
        if isinstance(sufficient, str):
            sufficient = sufficient.strip().lower() in ("true", "1", "yes")
        if not isinstance(sufficient, bool):
            raise ValueError(
                f"Argument 'sufficient' must be a boolean, "
                f"got {type(sufficient).__name__}."
            )

        reason = arguments.get("reason", "")
        # next_query/answer 透传，loop 用 parse_judge 取走。
        ack = {
            "judged": True,
            "sufficient": sufficient,
            "reason": str(reason)[:300],  # 防超长
            "next_query": arguments.get("next_query"),
            "answer": arguments.get("answer"),
        }
        return json.dumps(ack, ensure_ascii=False)

    def parse_judge(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """从 judge 的 tool_call 参数里取出结构化判断（供 loop 决策）。

        和 _execute_judge 分开：execute 是"执行+回执"，parse_judge 是
        "提取决策依据"。loop 在检测到 judge 调用时调它，读 sufficient/
        next_query/answer 决定走作答还是继续检索。

        容错（防御层 3/4）：sufficient 解析失败默认 False（保守，倾向继续
        检索而非过早作答）；缺 answer 时给空串。
        """
        sufficient_raw = arguments.get("sufficient")
        if isinstance(sufficient_raw, str):
            sufficient = sufficient_raw.strip().lower() in ("true", "1", "yes")
        elif isinstance(sufficient_raw, bool):
            sufficient = sufficient_raw
        else:
            sufficient = False  # 解析不出就当"不够"，保守不乱答。

        return {
            "sufficient": sufficient,
            "next_query": arguments.get("next_query"),
            "answer": arguments.get("answer") or "",
            "reason": arguments.get("reason", ""),
        }
