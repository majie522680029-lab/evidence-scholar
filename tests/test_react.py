"""Tests for the ReAct agent loop (B3).

FakeLLM 注入预设返回序列，不调真 LLM/不占 GPU。和 A 阶段检索器测试、
B2 工具层测试同套路，秒级跑完。

覆盖 5 个场景：
1. 单跳收敛：FakeLLM 第 1 跳直接作答 → loop 1 步退出。
2. 两跳收敛：第 1 跳 tool_call + 第 2 跳作答 → 2 步退出，消息配对正确。
3. max_steps 兜底：FakeLLM 永远要工具 → 超限退出，answer=None。
4. 消息配对：第 2 跳 FakeLLM 能"看到"第 1 跳的 tool 结果（验证喂回对）。
5. 异常 tool_call：LLM 给未知工具名 → 循环不崩，错误喂回 LLM。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from evidence_scholar.agent.react import (
    DEFAULT_MAX_STEPS,
    AgentResult,
    LLMResponse,
    ToolCall,
    run_agent,
)
from evidence_scholar.agent.tools import RetrievalTools
from evidence_scholar.retrieval.schemas import RetrievalResult


# --- Fakes ---

class FakeLLM:
    """假 LLM：按预设序列返回 LLMResponse。

    记录每次 chat 的入参 messages，便于断言循环喂回的消息对不对。
    """

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
        # 拷贝避免测试改到循环内部状态。
        self.calls_messages.append([dict(m) for m in messages])
        if self._call_idx >= len(self._responses):
            # 序列耗尽默认"一直要工具"——用于测 max_steps 兜底。
            return LLMResponse(
                tool_call=ToolCall(id="exhausted", name="retrieve_hybrid",
                                   arguments={"query": "more"})
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


# --- 场景 1：单跳收敛 ---

def test_single_step_answer(tools: RetrievalTools) -> None:
    """LLM 第 1 跳直接作答 → 1 步退出，answer 非空。"""
    fake = FakeLLM([_answer("Paris")])
    result = run_agent("capital of France?", llm_client=fake, tools=tools)

    assert result.answer == "Paris"
    assert result.steps == 1
    assert result.stopped_reason == "answered"


# --- 场景 2：两跳收敛 ---

def test_two_step_convergence(tools: RetrievalTools) -> None:
    """第 1 跳 tool_call + 第 2 跳作答 → 2 步退出。"""
    fake = FakeLLM([_tool_call(), _answer("the answer")])
    result = run_agent("multi-hop q", llm_client=fake, tools=tools)

    assert result.answer == "the answer"
    assert result.steps == 2
    assert result.stopped_reason == "answered"
    # 第 1 跳确实调了检索器。
    assert len(tools._retriever.calls) == 1  # type: ignore[attr-defined]


def test_two_step_message_pairing(tools: RetrievalTools) -> None:
    """第 2 跳 LLM 能看到第 1 跳的 tool 结果（消息配对正确）。

    验证 messages 在第 2 跳调用时含：system+user+assistant(tool_call)+tool。
    """
    fake = FakeLLM([_tool_call(), _answer()])
    run_agent("q", llm_client=fake, tools=tools)

    # 第 2 次 chat 调用时的 messages 应有 4 条（system/user/assistant/tool）。
    second_call_msgs = fake.calls_messages[1]
    assert len(second_call_msgs) == 4
    roles = [m["role"] for m in second_call_msgs]
    assert roles == ["system", "user", "assistant", "tool"]
    # tool 消息的 tool_call_id 要和 assistant 的 tool_call.id 对上。
    asst_msg = second_call_msgs[2]
    tool_msg = second_call_msgs[3]
    assert asst_msg["tool_calls"][0]["id"] == tool_msg["tool_call_id"]


# --- 场景 3：max_steps 兜底 ---

def test_max_steps_fallback(tools: RetrievalTools) -> None:
    """LLM 永远要工具 → 超 max_steps 截停，answer=None。"""
    fake = FakeLLM([])  # 序列为空 → 默认一直要工具
    result = run_agent("q", llm_client=fake, tools=tools, max_steps=3)

    assert result.answer is None
    assert result.stopped_reason == "max_steps"
    assert result.steps == 3  # 刚好跑到上限


def test_default_max_steps_is_8() -> None:
    """DEFAULT_MAX_STEPS = 8（HotpotQA 多跳 2-3 跳够覆盖）。"""
    assert DEFAULT_MAX_STEPS == 8


# --- 场景 4：tool 结果回传 ---

def test_tool_result_visible_to_next_step(tools: RetrievalTools) -> None:
    """第 2 跳的 tool 消息 content 含第 1 跳检索返回的文档。"""
    fake = FakeLLM([_tool_call(q="hello"), _answer()])
    run_agent("q", llm_client=fake, tools=tools)

    second_call_msgs = fake.calls_messages[1]
    tool_content = second_call_msgs[3]["content"]
    # 检索返回的 d1/d2 标题应出现在喂回 LLM 的 tool 消息里。
    assert "Title d1" in tool_content
    assert "Title d2" in tool_content


# --- 场景 5：异常 tool_call ---

def test_unknown_tool_does_not_crash_loop(tools: RetrievalTools) -> None:
    """LLM 给未知工具名 → 循环不崩，错误喂回，下一跳作答。"""
    fake = FakeLLM([
        LLMResponse(tool_call=ToolCall(id="bad", name="not_a_tool",
                                       arguments={"query": "x"})),
        _answer("recovered"),
    ])
    result = run_agent("q", llm_client=fake, tools=tools)

    # 循环没崩，最终作答。
    assert result.answer == "recovered"
    assert result.stopped_reason == "answered"
    # trace 第 1 跳记了错误。
    assert result.trace[0].error is not None
    assert "Unknown tool" in result.trace[0].error


def test_bad_tool_args_fed_back(tools: RetrievalTools) -> None:
    """LLM 给空 query → tools.execute 抛错 → 错误塞回 tool 消息。"""
    fake = FakeLLM([
        LLMResponse(tool_call=ToolCall(id="bq", name="retrieve_hybrid",
                                       arguments={"query": ""})),
        _answer(),
    ])
    result = run_agent("q", llm_client=fake, tools=tools)

    # 第 2 跳的 tool 消息应是错误描述（而非检索结果）。
    second_tool_content = fake.calls_messages[1][3]["content"]
    assert "failed" in second_tool_content.lower() or "non-empty" in second_tool_content
    assert result.trace[0].error is not None


# --- 边界 ---

def test_max_steps_zero_rejected(tools: RetrievalTools) -> None:
    """max_steps <= 0 直接报错（防止配置写 0 导致永不循环）。"""
    fake = FakeLLM([_answer()])
    with pytest.raises(ValueError, match="max_steps"):
        run_agent("q", llm_client=fake, tools=tools, max_steps=0)
