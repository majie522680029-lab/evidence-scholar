"""Tests for the retrieval tool-calling layer (B2).

用 FakeRetriever 注入，不加载真模型/真索引——和 A 阶段检索器测试
同套路，秒级跑完。
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from evidence_scholar.agent.tools import RetrievalTools
from evidence_scholar.retrieval.schemas import RetrievalResult


class FakeRetriever:
    """假检索器：按预设结果返回，记录每次 search 的入参便于断言。"""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        # 记录所有调用入参，便于断言 agent 传了什么。
        self.calls: list[tuple[str, int]] = []

    def search(
        self, query: str, top_k: int = 10
    ) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        # 返回前 top_k 个（FakeRetriever 不做真排序，测试不关心排序逻辑）。
        return list(self._results[:top_k])


def _make_result(rank: int, doc_id: str = "d1") -> RetrievalResult:
    """造一条测试用 RetrievalResult。"""
    return RetrievalResult(
        document_id=doc_id,
        title=f"Title {doc_id}",
        text=f"Body text of {doc_id} with some content.",
        score=1.0 / (rank + 1),
        rank=rank,
    )


# --- fixtures ---

@pytest.fixture
def three_results() -> list[RetrievalResult]:
    return [
        _make_result(1, "d1"),
        _make_result(2, "d2"),
        _make_result(3, "d3"),
    ]


@pytest.fixture
def tools(three_results: list[RetrievalResult]) -> RetrievalTools:
    return RetrievalTools(FakeRetriever(three_results))


# --- schema 格式 ---

def test_schema_has_two_tools(tools: RetrievalTools) -> None:
    """B5 后两个工具：retrieve_hybrid + judge_evidence。"""
    schema = tools.schema
    assert isinstance(schema, list)
    assert len(schema) == 2
    names = [t["function"]["name"] for t in schema]
    assert names == ["retrieve_hybrid", "judge_evidence"]


def test_schema_tool_has_openai_function_shape(tools: RetrievalTools) -> None:
    """schema 顶层是 {type: function, function: {...}}，OpenAI tools 格式。"""
    tool = tools.schema[0]
    assert tool["type"] == "function"
    fn = tool["function"]
    assert fn["name"] == "retrieve_hybrid"
    assert isinstance(fn["description"], str) and fn["description"]
    assert fn["parameters"]["type"] == "object"


def test_schema_query_is_required(tools: RetrievalTools) -> None:
    """query 是必填（top_k 有默认值，可缺省）。"""
    required = tools.schema[0]["function"]["parameters"]["required"]
    assert "query" in required


# --- execute 路由 ---

def test_execute_returns_json_string(tools: RetrievalTools) -> None:
    """execute 返回 JSON 字符串（LLM 读文本，不是 pydantic 对象）。"""
    out = tools.execute("retrieve_hybrid", {"query": "x"})
    assert isinstance(out, str)
    # 必须是合法 JSON。
    parsed = json.loads(out)
    assert isinstance(parsed, dict)


def test_execute_routes_query_and_topk_to_retriever(
    tools: RetrievalTools, three_results: list[RetrievalResult]
) -> None:
    """execute 把 query/top_k 原样传给底层 retriever.search。"""
    out = tools.execute("retrieve_hybrid", {"query": "hello", "top_k": 2})
    # 检查入参被正确传递。
    fake = tools._retriever  # type: ignore[attr-defined]
    assert fake.calls == [("hello", 2)]
    # 检查返回只含 top_k 个文档（这里 FakeRetriever 返回前 2 个）。
    parsed = json.loads(out)
    assert len(parsed["documents"]) == 2


def test_execute_default_topk_is_10(
    tools: RetrievalTools, three_results: list[RetrievalResult]
) -> None:
    """缺省 top_k 时默认 10。"""
    tools.execute("retrieve_hybrid", {"query": "q"})
    fake = tools._retriever  # type: ignore[attr-defined]
    assert fake.calls == [("q", 10)]


# --- 返回格式 ---

def test_execute_payload_has_query_and_documents(
    tools: RetrievalTools
) -> None:
    """返回结构含 query（回显）+ documents 列表。"""
    parsed = json.loads(tools.execute("retrieve_hybrid", {"query": "abc"}))
    assert parsed["query"] == "abc"
    assert "documents" in parsed
    assert isinstance(parsed["documents"], list)


def test_execute_document_has_rank_title_text_score(
    tools: RetrievalTools,
) -> None:
    """每个文档条目含 LLM 能读的 rank/title/text/score。"""
    parsed = json.loads(tools.execute("retrieve_hybrid", {"query": "q"}))
    doc = parsed["documents"][0]
    assert {"rank", "document_id", "title", "text", "score"} <= set(doc)


def test_execute_truncates_long_text() -> None:
    """正文超长被截断并补省略号，防爆上下文。"""
    long_text = "x" * 5000
    retriever = FakeRetriever(
        [RetrievalResult(
            document_id="d1", title="T", text=long_text, score=1.0, rank=1
        )]
    )
    tools = RetrievalTools(retriever)
    parsed = json.loads(tools.execute("retrieve_hybrid", {"query": "q"}))
    text = parsed["documents"][0]["text"]
    assert len(text) < 5000
    assert text.endswith("…")


def test_execute_empty_results_has_note(tools: RetrievalTools) -> None:
    """检索无结果时明确返回 note，避免 LLM 误判。"""
    empty = RetrievalTools(FakeRetriever([]))
    parsed = json.loads(empty.execute("retrieve_hybrid", {"query": "q"}))
    assert parsed["documents"] == []
    assert "note" in parsed


# --- 参数校验 ---

def test_execute_rejects_unknown_tool(tools: RetrievalTools) -> None:
    """未知工具名报 ValueError。"""
    with pytest.raises(ValueError, match="Unknown tool name"):
        tools.execute("not_a_tool", {"query": "q"})


def test_execute_rejects_empty_query(tools: RetrievalTools) -> None:
    """空 query 报错。"""
    with pytest.raises(ValueError, match="non-empty"):
        tools.execute("retrieve_hybrid", {"query": ""})


def test_execute_rejects_missing_query(tools: RetrievalTools) -> None:
    """缺 query 报错。"""
    with pytest.raises(ValueError, match="non-empty"):
        tools.execute("retrieve_hybrid", {})


def test_execute_rejects_non_string_query(tools: RetrievalTools) -> None:
    """query 非 str 报错（LLM 偶尔会传 int/None）。"""
    with pytest.raises(ValueError, match="non-empty"):
        tools.execute("retrieve_hybrid", {"query": 123})


def test_execute_rejects_topk_out_of_range(tools: RetrievalTools) -> None:
    """top_k 越界 [1,20] 报错。"""
    with pytest.raises(ValueError, match=r"\[1, 20\]"):
        tools.execute("retrieve_hybrid", {"query": "q", "top_k": 0})
    with pytest.raises(ValueError, match=r"\[1, 20\]"):
        tools.execute("retrieve_hybrid", {"query": "q", "top_k": 21})


def test_execute_rejects_non_integer_topk(tools: RetrievalTools) -> None:
    """top_k 非 int 报错。"""
    with pytest.raises(ValueError, match="integer"):
        tools.execute("retrieve_hybrid", {"query": "q", "top_k": "ten"})


# --- B5: judge_evidence 工具 ---

def test_judge_schema_shape(tools: RetrievalTools) -> None:
    """judge 工具 schema 有 sufficient 必填 + reason maxLength 约束。"""
    judge = tools.schema[1]["function"]
    assert judge["name"] == "judge_evidence"
    params = judge["parameters"]["properties"]
    assert "sufficient" in params
    assert params["sufficient"]["type"] == "boolean"
    # reason 限 maxLength（防小作文截断）。
    assert "maxLength" in params["reason"]
    # sufficient 必填。
    assert "sufficient" in judge["parameters"]["required"]


def test_judge_execute_sufficient_true(tools: RetrievalTools) -> None:
    """judge sufficient=true 时 execute 回执 + 透传 answer。"""
    out = tools.execute("judge_evidence", {
        "sufficient": True, "reason": "have both", "answer": "yes"
    })
    parsed = json.loads(out)
    assert parsed["judged"] is True
    assert parsed["sufficient"] is True
    assert parsed["answer"] == "yes"


def test_judge_execute_sufficient_false(tools: RetrievalTools) -> None:
    """judge sufficient=false 时透传 next_query。"""
    out = tools.execute("judge_evidence", {
        "sufficient": False, "reason": "need X", "next_query": "X nationality"
    })
    parsed = json.loads(out)
    assert parsed["sufficient"] is False
    assert parsed["next_query"] == "X nationality"


def test_judge_execute_accepts_string_boolean(tools: RetrievalTools) -> None:
    """LLM 偶尔把 boolean 传成字符串 'true'，execute 容错。"""
    out = tools.execute("judge_evidence", {"sufficient": "true", "reason": ""})
    assert json.loads(out)["sufficient"] is True


def test_judge_execute_rejects_missing_sufficient(tools: RetrievalTools) -> None:
    """缺 sufficient 报错（防漂移，必填）。"""
    with pytest.raises(ValueError, match="sufficient"):
        tools.execute("judge_evidence", {"reason": "no sufficient"})


def test_parse_judge_sufficient_true(tools: RetrievalTools) -> None:
    """parse_judge 从参数取 sufficient=True + answer。"""
    parsed = tools.parse_judge({
        "sufficient": True, "answer": "American", "reason": "got both"
    })
    assert parsed["sufficient"] is True
    assert parsed["answer"] == "American"


def test_parse_judge_sufficient_false(tools: RetrievalTools) -> None:
    """parse_judge 取 sufficient=False + next_query。"""
    parsed = tools.parse_judge({
        "sufficient": False, "next_query": "X age", "reason": "miss age"
    })
    assert parsed["sufficient"] is False
    assert parsed["next_query"] == "X age"


def test_parse_judge_missing_sufficient_defaults_false(
    tools: RetrievalTools,
) -> None:
    """sufficient 解析不出时默认 False（保守，倾向继续检索不乱答）。"""
    parsed = tools.parse_judge({"reason": "no sufficient field"})
    assert parsed["sufficient"] is False


def test_parse_judge_missing_answer_defaults_empty(
    tools: RetrievalTools,
) -> None:
    """缺 answer 给空串而非 None（loop 拼答案时不崩）。"""
    parsed = tools.parse_judge({"sufficient": True})
    assert parsed["answer"] == ""


def test_judge_execute_truncates_long_reason(tools: RetrievalTools) -> None:
    """超长 reason 被截断（防回执撑爆上下文）。"""
    out = tools.execute("judge_evidence", {
        "sufficient": True,
        "reason": "x" * 1000,
        "answer": "a",
    })
    parsed = json.loads(out)
    assert len(parsed["reason"]) <= 300
