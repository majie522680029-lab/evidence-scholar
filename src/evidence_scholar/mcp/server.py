"""MCP server exposing the retrieval capability as a standardized tool (C1).

实习要求 #6：MCP（Model Context Protocol，Anthropic 2024 年底推的开放协议）
标准化了"LLM 应用怎么调外部工具/数据源"。本模块把 EvidenceScholar 的
检索能力包成 MCP server，任何支持 MCP 的客户端（Claude Desktop、Cursor、
其他 agent）都能直接调 retrieve，不用改 agent 代码。

设计要点：
- 只暴露 retrieve 一个工具（不给 MCP 客户端暴露 judge_evidence——judge 是
  agent 内部的退出控制，MCP 客户端是外部消费者，不该掺和 agent 的判断
  逻辑）。检索是通用能力，判断是 agent 私有逻辑，分开。
- 不走 RetrievalTools（那是给 OpenAI tool-calling 用的，带 judge schema）。
  MCP server 直接调 retriever.search()，复用 retrieval 层，不引入第二套
  工具封装。
- 工具层不持有索引生命周期。构造时传入已 build_index 的 retriever，索引
  建/换由启动入口（scripts/run_mcp_server.py）负责。和 RetrievalTools 的
  约定一致，本层纯逻辑，可用 fake retriever 单元测试。
- 输出格式和 tools.py 的 _format_result_for_llm 字段对齐
  （rank/document_id/title/text/score），保证两套入口返回一致。
- stdio 传输铁律：stdout 只能走 JSON-RPC，print 调试会破坏协议。所有
  日志走 stderr / logging，绝不 print 到 stdout。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from mcp.server.mcpserver import MCPServer

from evidence_scholar.retrieval.base import BaseRetriever
from evidence_scholar.retrieval.schemas import RetrievalResult

# 给 MCP 客户端看的正文片段最大字符数。和 tools.py 的 _MAX_DOC_TEXT_CHARS
# 对齐，保证两套入口返回的文档截断长度一致。
_MAX_DOC_TEXT_CHARS = 800


def _format_result(result: RetrievalResult) -> dict[str, Any]:
    """把一条 RetrievalResult 序列化成 MCP 客户端可读的 dict。

    和 tools.py 的 _format_result_for_llm 字段完全对齐：rank/document_id/
    title/text/score。正文做同样的字符截断，保持两套入口输出一致。
    """
    text = result.text
    if len(text) > _MAX_DOC_TEXT_CHARS:
        text = text[:_MAX_DOC_TEXT_CHARS] + "…"
    return {
        "rank": result.rank,
        "document_id": result.document_id,
        "title": result.title,
        "text": text,
        "score": round(float(result.score), 4),
    }


def build_retrieval_server(retriever: BaseRetriever) -> MCPServer:
    """Build an MCP server that exposes retrieve over a retriever.

    Args:
        retriever: 已 build_index 的检索器。MCP 客户端调 retrieve 时实际
            执行 retriever.search(query, top_k)。索引生命周期由启动入口
            负责，本函数不管 build_index。

    Returns:
        配好 retrieve 工具的 MCPServer。调用方再 .run(transport=...) 启动。
    """
    # MCPServer 2.x 构造参数：name/title/description 给客户端看的能力说明。
    server = MCPServer(
        name="evidence-scholar-retrieval",
        title="EvidenceScholar Retrieval",
        description=(
            "Retrieve relevant passages from the EvidenceScholar document "
            "corpus via hybrid (BM25+Dense+RRF) search."
        ),
    )

    @server.tool()
    def retrieve(query: str, top_k: int = 10) -> str:
        """Search the document corpus for passages relevant to a query.

        Returns ranked documents with title and a text excerpt. Use this to
        gather evidence for answering questions; call with a reformulated
        sub-query when you need different or more information.

        Args:
            query: A natural-language or keyword sub-query to search for.
                Reformulate the question into a focused retrieval query.
            top_k: Number of top documents to return. Default 10, range [1, 20].

        Returns:
            JSON string with a documents list (rank/document_id/title/text/
            score), or a note if nothing matched.
        """
        # 参数校验：top_k 可能被客户端传成别的类型，统一强转+范围检查。
        # 和 tools.py._execute_hybrid 的校验逻辑一致，保证两套入口容错一致。
        try:
            top_k = int(top_k)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"top_k must be an integer, got {top_k!r}."
            ) from error
        if top_k <= 0 or top_k > 20:
            raise ValueError(
                f"top_k must be in [1, 20], got {top_k}."
            )

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        results: Sequence[RetrievalResult] = retriever.search(
            query, top_k=top_k
        )

        # 空结果要明确告知，避免客户端误以为"没返回"等于"语料里没有"。
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
                "documents": [_format_result(r) for r in results],
            }
        return json.dumps(payload, ensure_ascii=False)

    return server
