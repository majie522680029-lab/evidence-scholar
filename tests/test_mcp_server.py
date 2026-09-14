"""Tests for the MCP retrieval server (C1).

用 fake retriever 注入（和 tools.py 测试一致），不碰 GPU/真模型。
测：工具注册、参数校验、正常检索返回、空结果、参数非法报错。
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from evidence_scholar.mcp.server import build_retrieval_server
from evidence_scholar.retrieval.base import BaseRetriever
from evidence_scholar.retrieval.schemas import Document, RetrievalResult


class FakeRetriever(BaseRetriever):
    """预设返回结果的假检索器，不碰 GPU/真模型。

    记录最后一次 search 的参数，供测试断言。返回的 RetrievalResult 的
    rank/score 字段允许测试验证序列化是否正确。
    """

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def build_index(self, documents: Sequence[Document]) -> None:
        """No-op for the fake."""

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self.last_query = query
        self.last_top_k = top_k
        # 按 top_k 截断，模拟真检索器的行为。
        return self._results[:top_k]

    def save(self, path) -> None:  # type: ignore[override]
        """No-op for the fake."""

    def load(self, path) -> None:  # type: ignore[override]
        """No-op for the fake."""


def _make_result(rank: int, title: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        document_id=f"doc-{rank}",
        score=1.0 / rank,
        rank=rank,
        title=title,
        text=text,
    )


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever(
        [
            _make_result(1, "Tim Burton", "Timothy Burton is an American filmmaker."),
            _make_result(2, "Ed Wood", "Edward Wood Jr. was an American filmmaker."),
        ]
    )


@pytest.fixture
def server(fake_retriever: FakeRetriever):
    return build_retrieval_server(fake_retriever)


# ---------------------------------------------------------------------------
# 工具注册
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_registered(server) -> None:
    """server 应注册名为 retrieve 的工具。"""
    import asyncio

    tools = await server.list_tools()
    names = [t.name for t in tools]
    assert "retrieve" in names


@pytest.mark.asyncio
async def test_tool_schema_has_query_required(server) -> None:
    """retrieve 的 input_schema 里 query 必填、top_k 可选默认 10。"""
    tools = await server.list_tools()
    retrieve = next(t for t in tools if t.name == "retrieve")
    schema = retrieve.input_schema
    assert "query" in schema["required"]
    assert schema["properties"]["top_k"]["default"] == 10
    assert schema["properties"]["top_k"]["type"] == "integer"


# ---------------------------------------------------------------------------
# 正常检索
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_returns_documents(server, fake_retriever) -> None:
    """retrieve 应返回 JSON，含 documents 列表，字段和 tools.py 对齐。"""
    import asyncio

    result = await server.call_tool(
        "retrieve", {"query": "Ed Wood director", "top_k": 10}
    )
    # call_tool 返回的内容（struct content）。取第一个 text 块的文本。
    assert result.content is not None
    text_block = result.content[0]
    payload = json.loads(text_block.text)
    assert payload["query"] == "Ed Wood director"
    assert len(payload["documents"]) == 2
    doc0 = payload["documents"][0]
    assert set(doc0.keys()) == {
        "rank", "document_id", "title", "text", "score"
    }
    assert doc0["title"] == "Tim Burton"
    assert doc0["rank"] == 1
    # 假检索器记录了调用参数。
    assert fake_retriever.last_query == "Ed Wood director"
    assert fake_retriever.last_top_k == 10


@pytest.mark.asyncio
async def test_retrieve_default_top_k(server, fake_retriever) -> None:
    """不传 top_k 时走默认 10。"""
    result = await server.call_tool("retrieve", {"query": "anything"})
    text_block = result.content[0]
    payload = json.loads(text_block.text)
    assert payload["query"] == "anything"
    assert fake_retriever.last_top_k == 10


# ---------------------------------------------------------------------------
# 空结果
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_empty_results(server) -> None:
    """空结果应返回 note，不能让客户端误以为"没返回"等于"语料没有"。"""
    empty = FakeRetriever([])
    srv = build_retrieval_server(empty)
    result = await srv.call_tool("retrieve", {"query": "nonexistent"})
    text_block = result.content[0]
    payload = json.loads(text_block.text)
    assert payload["documents"] == []
    assert "note" in payload


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_rejects_empty_query(server) -> None:
    """空 query 应报错。"""
    with pytest.raises(Exception):
        await server.call_tool("retrieve", {"query": ""})


@pytest.mark.asyncio
async def test_retrieve_rejects_bad_top_k(server) -> None:
    """top_k 越界应报错。"""
    with pytest.raises(Exception):
        await server.call_tool("retrieve", {"query": "x", "top_k": 0})
    with pytest.raises(Exception):
        await server.call_tool("retrieve", {"query": "x", "top_k": 21})


# ---------------------------------------------------------------------------
# 长文档截断
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_truncates_long_text(server) -> None:
    """超长正文应截断到 _MAX_DOC_TEXT_CHARS + 省略号。"""
    long_text = "x" * 2000
    retriever = FakeRetriever([_make_result(1, "Long Doc", long_text)])
    srv = build_retrieval_server(retriever)
    result = await srv.call_tool("retrieve", {"query": "long"})
    text_block = result.content[0]
    payload = json.loads(text_block.text)
    doc_text = payload["documents"][0]["text"]
    # 截断后含省略号，长度不超过 800 + 省略号字符。
    assert doc_text.endswith("…")
    assert len(doc_text) <= 801  # 800 + 省略号
