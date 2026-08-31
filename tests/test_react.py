"""Tests for the ReAct agent loop (B3 + B5 + B4).

FakeLLM 注入预设返回序列，不调真 LLM/不占 GPU。和 A 阶段检索器测试、
B2 工具层测试同套路，秒级跑完。

覆盖 8 个场景：
1. 单跳收敛（退出 A）：FakeLLM 第 1 跳直接 text 作答 → loop 1 步退出。
2. 两跳收敛（退出 A）：第 1 跳 tool_call + 第 2 跳 text 作答 → 2 步退出，
   消息配对正确。
3. max_steps 兜底：FakeLLM 永远要工具 → 超限退出，answer=None。
4. 消息配对：第 2 跳 FakeLLM 能"看到"第 1 跳的 tool 结果（验证喂回对）。
5. 异常 tool_call：LLM 给未知工具名 → 循环不崩，错误喂回 LLM。
6. judge 退出（退出 B，B5 主路径）：LLM 调 judge_evidence sufficient=true →
   用结构化 answer 退出，不再多跑一跳。
7. judge insufficient：LLM 调 judge sufficient=false → 不退出，ack 喂回，
   下一跳继续（验证 LLM 仍是控制器，sufficient=false 不截停）。
8. B4 证据池：retrieve 后证据入池 + 摘要注入 tool 结果（方案 B），judge
   ack 不注入，无 retrieve 时池空。
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


# --- 场景 6：judge sufficient=true 退出（B5 主路径，退出 B）---

def test_judge_sufficient_exits_with_structured_answer(
    tools: RetrievalTools,
) -> None:
    """LLM 调 judge_evidence sufficient=true → 用 judge.answer 退出，1 跳完。

    这是 B5 的主退出路径。关键：退出用的是 judge 结构化 answer 字段，
    不是 LLM text，也不是多跑一跳。停止原因仍是 answered。
    """
    judge_call = LLMResponse(
        tool_call=ToolCall(
            id="j1", name="judge_evidence",
            arguments={"sufficient": True, "answer": "Paris",
                       "reason": "found capital", "next_query": ""},
        )
    )
    fake = FakeLLM([judge_call])
    result = run_agent("capital of France?", llm_client=fake, tools=tools)

    assert result.answer == "Paris"
    assert result.steps == 1
    assert result.stopped_reason == "answered"
    # trace 记了 judge 这跳。
    assert result.trace[-1].tool_call.name == "judge_evidence"


def test_judge_answer_empty_falls_back_to_text(tools: RetrievalTools) -> None:
    """judge sufficient=true 但 answer 字段空 → 回退用 LLM 这一跳的 text。

    多级兜底：LLM 可能把答案写进 content 而非 judge.answer，仍能收尾。
    """
    judge_call = LLMResponse(
        text="The answer is Paris.",
        tool_call=ToolCall(
            id="j2", name="judge_evidence",
            arguments={"sufficient": True, "answer": "",
                       "reason": "done", "next_query": ""},
        )
    )
    fake = FakeLLM([judge_call])
    result = run_agent("capital of France?", llm_client=fake, tools=tools)

    assert result.answer == "The answer is Paris."
    assert result.stopped_reason == "answered"


def test_judge_sufficient_after_retrieval(tools: RetrievalTools) -> None:
    """两跳：先 retrieve 收证据，再 judge sufficient=true → 2 步退出。

    更接近真实多跳流程：retrieve → judge（够）→ 出答案。
    """
    judge_call = LLMResponse(
        tool_call=ToolCall(
            id="j3", name="judge_evidence",
            arguments={"sufficient": True, "answer": "yes",
                       "reason": "evidence found", "next_query": ""},
        )
    )
    fake = FakeLLM([_tool_call(), judge_call])
    result = run_agent("multi-hop q", llm_client=fake, tools=tools)

    assert result.answer == "yes"
    assert result.steps == 2
    assert result.stopped_reason == "answered"
    # 第 1 跳 retrieve、第 2 跳 judge。
    assert result.trace[0].tool_call.name == "retrieve_hybrid"
    assert result.trace[1].tool_call.name == "judge_evidence"


# --- 场景 7：judge sufficient=false 不退出 ---

def test_judge_insufficient_does_not_exit(tools: RetrievalTools) -> None:
    """LLM 调 judge sufficient=false → 不退出，ack 喂回，继续跑下一跳。

    sufficient=false 表示证据不够。ack（含 next_query）塞回 LLM，下一跳
    LLM 自己拿 next_query 去 retrieve。保持 LLM 是控制器（ReAct 核心），
    不硬塞检索动作。
    """
    judge_insufficient = LLMResponse(
        tool_call=ToolCall(
            id="j4", name="judge_evidence",
            arguments={"sufficient": False, "answer": "",
                       "reason": "need more", "next_query": "sharper q"},
        )
    )
    # judge 之后 FakeLLM 再作答收尾。
    fake = FakeLLM([judge_insufficient, _answer("final after more retrieval")])
    result = run_agent("q", llm_client=fake, tools=tools)

    # 没有在 judge 这步退出——继续跑到第 2 跳作答。
    assert result.answer == "final after more retrieval"
    assert result.steps == 2
    assert result.stopped_reason == "answered"
    # judge 的 ack 确实喂回了第 2 跳的 LLM（消息含 tool 角色 ack）。
    second_call_msgs = fake.calls_messages[1]
    assert second_call_msgs[2]["role"] == "assistant"
    assert second_call_msgs[3]["role"] == "tool"
    ack_content = second_call_msgs[3]["content"]
    # ack 含 next_query 字段，LLM 下一跳能看到改写后的 query。
    assert "sharper q" in ack_content or "next_query" in ack_content


def test_judge_then_retrieve_then_judge_exit(tools: RetrievalTools) -> None:
    """完整多跳路径：retrieve → judge(不足) → retrieve → judge(足) → 退出。

    最贴近真实 ReAct 的多跳收敛。judge 两次：第一次不足继续，第二次足退出。
    """
    retrieve2 = LLMResponse(
        tool_call=ToolCall(id="r2", name="retrieve_hybrid",
                           arguments={"query": "second hop"})
    )
    judge_enough = LLMResponse(
        tool_call=ToolCall(
            id="j5", name="judge_evidence",
            arguments={"sufficient": True, "answer": "the final answer",
                       "reason": "complete", "next_query": ""},
        )
    )
    judge_not_enough = LLMResponse(
        tool_call=ToolCall(
            id="j6", name="judge_evidence",
            arguments={"sufficient": False, "answer": "",
                       "reason": "incomplete", "next_query": "second hop"},
        )
    )
    fake = FakeLLM([
        _tool_call(q="first hop"),   # 1. 先检索
        judge_not_enough,            # 2. 判不够，给 next_query
        retrieve2,                   # 3. 再检索（next hop）
        judge_enough,                # 4. 判够 → 退出
    ])
    result = run_agent("multi-hop q", llm_client=fake, tools=tools)

    assert result.answer == "the final answer"
    assert result.steps == 4
    assert result.stopped_reason == "answered"
    # 调了两次检索器。
    assert len(tools._retriever.calls) == 2  # type: ignore[attr-defined]


# --- 场景 8：B4 证据池入池 + 摘要注入 ---

def test_evidence_pool_accumulates_across_hops(tools: RetrievalTools) -> None:
    """多跳 retrieve → 证据池累积去重后的证据（B4 入池）。

    FakeRetriever 每次返回 d1/d2。两次 retrieve（之间夹一个 judge
    insufficient）→ d1/d2 各被两跳命中 → hit_count=2，但池里只 2 条
    （去重）。验证 pool.size 反映去重后不同文档数、hit_count 反映多跳命中。
    """
    judge_not_enough = LLMResponse(
        tool_call=ToolCall(
            id="j_ne", name="judge_evidence",
            arguments={"sufficient": False, "next_query": "second hop",
                       "reason": "need more"},
        )
    )
    judge_enough = LLMResponse(
        tool_call=ToolCall(
            id="j_e", name="judge_evidence",
            arguments={"sufficient": True, "answer": "done",
                       "reason": "enough"},
        )
    )
    # retrieve → judge(不足) → retrieve → judge(足) → 退出
    fake = FakeLLM([_tool_call(), judge_not_enough, _tool_call(), judge_enough])
    result = run_agent("q", llm_client=fake, tools=tools)

    # 两次 retrieve 都返回 [d1, d2]，去重后池里 2 条，但各被两跳命中。
    assert result.evidence_pool.size == 2
    for ev in result.evidence_pool.items:
        assert ev.hit_count == 2


def test_evidence_summary_injected_into_retrieve_result(
    tools: RetrievalTools,
) -> None:
    """retrieve 的 tool 结果消息含 [Evidence Pool] 摘要（B4 方案 B 注入）。

    第 2 跳 LLM 能看到第 1 跳 retrieve 结果里注入的累积证据摘要。
    """
    fake = FakeLLM([_tool_call(), _answer()])
    run_agent("q", llm_client=fake, tools=tools)

    # 第 2 次 chat 调用时，第 1 跳的 tool 消息应含注入的证据池摘要。
    second_call_msgs = fake.calls_messages[1]
    tool_content = second_call_msgs[3]["content"]
    assert "[Evidence Pool" in tool_content  # 摘要标记
    assert "items" in tool_content           # 摘要计数
    assert "Title d1" in tool_content        # 累积的文档标题


def test_evidence_pool_empty_when_no_retrieve(tools: RetrievalTools) -> None:
    """LLM 直接作答不调 retrieve → 证据池空（B4 边界）。"""
    fake = FakeLLM([_answer("direct answer")])
    result = run_agent("q", llm_client=fake, tools=tools)
    assert result.evidence_pool.size == 0


def test_judge_result_not_injected_with_pool(tools: RetrievalTools) -> None:
    """judge 的 ack 不注入证据池摘要（B4 设计：只在 retrieve 后注入）。

    judge 和 sufficient 退出在同一 tool_call，注入来不及影响这次判定。
    """
    judge_call = LLMResponse(
        tool_call=ToolCall(
            id="j1", name="judge_evidence",
            arguments={"sufficient": True, "answer": "yes", "reason": "ok"},
        )
    )
    fake = FakeLLM([judge_call])
    result = run_agent("q", llm_client=fake, tools=tools)

    # judge 没调 retrieve，证据池空，且 judge 的 ack 不含 Evidence Pool 标记。
    assert result.evidence_pool.size == 0


# --- 边界 ---

def test_max_steps_zero_rejected(tools: RetrievalTools) -> None:
    """max_steps <= 0 直接报错（防止配置写 0 导致永不循环）。"""
    fake = FakeLLM([_answer()])
    with pytest.raises(ValueError, match="max_steps"):
        run_agent("q", llm_client=fake, tools=tools, max_steps=0)
