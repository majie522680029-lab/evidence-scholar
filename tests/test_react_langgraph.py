"""Tests for the LangGraph ReAct rewrite (C2).

策略：复用 test_react.py 的 FakeLLM/FakeRetriever 基础设施，对每个场景
跑 run_agent_langgraph，断言和 react.py 手写版行为一致（answer/steps/
stopped_reason 相同）。这是"两版等价"的证据，也是 C2 的核心交付。

覆盖的场景（和 test_react.py 对齐）：
1. 单跳收敛（退出 A）
2. 两跳收敛（退出 A）
3. max_steps 兜底
4. judge 退出（退出 B，B5 主路径）
5. judge insufficient 不退出
6. retrieve -> judge insufficient -> retrieve -> judge sufficient 多跳
7. B4 证据池累积
8. tool 错误不崩循环
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

from evidence_scholar.agent.react import (
    DEFAULT_MAX_STEPS,
    LLMResponse,
    ToolCall,
    run_agent,
)
from evidence_scholar.agent.react_langgraph import run_agent_langgraph
from evidence_scholar.agent.tools import RetrievalTools
from evidence_scholar.retrieval.schemas import RetrievalResult


# --- Fakes（和 test_react.py 一致，保证两版同样输入） ---

class FakeLLM:
    """假 LLM：按预设序列返回 LLMResponse，记录每次入参 messages。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_idx = 0
        self.calls_messages: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        self.calls_messages.append([dict(m) for m in messages])
        if self._call_idx >= len(self._responses):
            return LLMResponse(
                tool_call=ToolCall(
                    id="exhausted", name="retrieve_hybrid",
                    arguments={"query": "more"},
                )
            )
        r = self._responses[self._call_idx]
        self._call_idx += 1
        return r


class FakeRetriever:
    """假检索器：返回预设结果，记录 search 调用。"""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 10):
        self.calls.append((query, top_k))
        return list(self._results)


def _make_result(doc_id: str = "d1") -> RetrievalResult:
    return RetrievalResult(
        document_id=doc_id, title=f"Title {doc_id}",
        text=f"Body of {doc_id}.", score=0.5, rank=1,
    )


@pytest.fixture
def tools() -> RetrievalTools:
    return RetrievalTools(FakeRetriever([_make_result("d1"), _make_result("d2")]))


def _tool_call(name: str = "retrieve_hybrid", q: str = "q") -> LLMResponse:
    return LLMResponse(
        tool_call=ToolCall(id="call_1", name=name, arguments={"query": q})
    )


def _answer(text: str = "final answer") -> LLMResponse:
    return LLMResponse(text=text)


# --- 两版等价测试 ---

def _assert_eq(a, b, label: str) -> None:
    """断言两版结果等价（answer/steps/stopped_reason）。"""
    assert a.answer == b.answer, f"{label}: answer {a.answer!r} != {b.answer!r}"
    assert a.stopped_reason == b.stopped_reason, (
        f"{label}: stopped_reason {a.stopped_reason!r} != {b.stopped_reason!r}"
    )
    assert a.steps == b.steps, f"{label}: steps {a.steps} != {b.steps}"


def test_single_step_answer(tools: RetrievalTools) -> None:
    """单跳收敛：LLM 第 1 跳直接 text 作答。两版都 1 步退出。"""
    fake = FakeLLM([_answer("Paris")])
    r1 = run_agent("capital?", llm_client=fake, tools=tools)
    fake2 = FakeLLM([_answer("Paris")])
    r2 = run_agent_langgraph("capital?", llm_client=fake2, tools=tools)
    _assert_eq(r1, r2, "single_step")
    assert r2.answer == "Paris"


def test_two_step_convergence(tools: RetrievalTools) -> None:
    """两跳收敛：retrieve -> text 作答。两版都 2 步退出。"""
    fake = FakeLLM([_tool_call(q="q1"), _answer("Paris")])
    r1 = run_agent("q?", llm_client=fake, tools=tools)
    fake2 = FakeLLM([_tool_call(q="q1"), _answer("Paris")])
    r2 = run_agent_langgraph("q?", llm_client=fake2, tools=tools)
    _assert_eq(r1, r2, "two_step")
    assert r2.steps == 2


def test_max_steps_fallback(tools: RetrievalTools) -> None:
    """max_steps 兜底：FakeLLM 永远要工具。两版都超限退出，answer=None。"""
    fake = FakeLLM([])
    r1 = run_agent("q?", llm_client=fake, tools=tools, max_steps=2)
    fake2 = FakeLLM([])
    r2 = run_agent_langgraph("q?", llm_client=fake2, tools=tools, max_steps=2)
    assert r1.answer is None and r2.answer is None
    assert r1.stopped_reason == r2.stopped_reason == "max_steps"


def test_judge_sufficient_exits(tools: RetrievalTools) -> None:
    """退出 B：judge sufficient=true -> 结构化 answer 退出。"""
    judge = LLMResponse(
        tool_call=ToolCall(
            id="j1", name="judge_evidence",
            arguments={"sufficient": True, "answer": "Paris",
                       "reason": "found", "next_query": ""},
        )
    )
    fake = FakeLLM([judge])
    r1 = run_agent("q?", llm_client=fake, tools=tools)
    fake2 = FakeLLM([judge])
    r2 = run_agent_langgraph("q?", llm_client=fake2, tools=tools)
    _assert_eq(r1, r2, "judge_sufficient")
    assert r2.answer == "Paris"
    assert r2.stopped_reason == "answered"


def test_judge_insufficient_does_not_exit(tools: RetrievalTools) -> None:
    """judge insufficient -> 不退出，继续检索。两版都继续。"""
    judge_no = LLMResponse(
        tool_call=ToolCall(
            id="j1", name="judge_evidence",
            arguments={"sufficient": False, "answer": "",
                       "reason": "need more", "next_query": "more"},
        )
    )
    # 序列：judge insufficient -> retrieve -> judge sufficient
    judge_yes = LLMResponse(
        tool_call=ToolCall(
            id="j2", name="judge_evidence",
            arguments={"sufficient": True, "answer": "Paris",
                       "reason": "done", "next_query": ""},
        )
    )
    seq = [judge_no, _tool_call(q="more"), judge_yes]
    r1 = run_agent("q?", llm_client=FakeLLM(seq), tools=tools)
    r2 = run_agent_langgraph("q?", llm_client=FakeLLM(list(seq)), tools=tools)
    _assert_eq(r1, r2, "judge_insufficient_then_yes")
    assert r2.answer == "Paris"
    assert r2.steps == 3


def test_judge_answer_empty_falls_back_to_text(tools: RetrievalTools) -> None:
    """judge sufficient 但 answer 空 -> 回退 LLM text。两版都回退。"""
    judge = LLMResponse(
        text="The answer is Paris.",
        tool_call=ToolCall(
            id="j2", name="judge_evidence",
            arguments={"sufficient": True, "answer": "",
                       "reason": "done", "next_query": ""},
        )
    )
    r1 = run_agent("q?", llm_client=FakeLLM([judge]), tools=tools)
    r2 = run_agent_langgraph("q?", llm_client=FakeLLM([judge]), tools=tools)
    _assert_eq(r1, r2, "judge_answer_empty")
    assert r2.answer == "The answer is Paris."


def test_evidence_pool_accumulates(tools: RetrievalTools) -> None:
    """B4：retrieve 后证据入池。两版 pool.size 一致。"""
    seq = [_tool_call(q="q1"), _tool_call(q="q2"), _answer("done")]
    r1 = run_agent("q?", llm_client=FakeLLM(seq), tools=tools)
    r2 = run_agent_langgraph("q?", llm_client=FakeLLM(list(seq)), tools=tools)
    assert r1.evidence_pool.size == r2.evidence_pool.size
    assert r2.evidence_pool.size == 2  # 两次 retrieve 命中 2 篇（去重后）


def test_unknown_tool_does_not_crash(tools: RetrievalTools) -> None:
    """未知工具名不崩循环，错误喂回 LLM。两版都恢复。"""
    bad = LLMResponse(
        tool_call=ToolCall(id="bad", name="nonexistent_tool", arguments={})
    )
    seq = [bad, _answer("recovered")]
    r1 = run_agent("q?", llm_client=FakeLLM(seq), tools=tools)
    r2 = run_agent_langgraph("q?", llm_client=FakeLLM(list(seq)), tools=tools)
    _assert_eq(r1, r2, "unknown_tool")
    assert r2.answer == "recovered"


def test_default_max_steps_is_8() -> None:
    """默认 max_steps = 8（两版共享常量）。"""
    assert DEFAULT_MAX_STEPS == 8
