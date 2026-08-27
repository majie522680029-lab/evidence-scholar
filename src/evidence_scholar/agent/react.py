"""ReAct agent loop (B3): Reason + Act over retrieval tools.

B3 是 v2 agent 的"大脑"——手写 ReAct 循环。为什么手写而非直接上
LangGraph：实习要求 #3 明确要"自己实现 ReAct Agent"。手写一遍才真懂
agent 的运行机制（消息配对、终止判定、循环兜底），C2 用 LangGraph
重写时才有对比、能讲"框架替我解决了什么"。

ReAct 循环（每跳 Reason + Act）：
    messages = [system + user]
    while step < max_steps:
        response = llm.chat(messages, tools)          # LLM 想一下
        if response 无 tool_call:
            return answer                              # LLM 自己答了 → 收
        result = tools.execute(tool_call)              # 执行 LLM 指派的动作
        messages += [assistant(tool_call), tool(result)]  # 喂回 LLM 观察
        step += 1
    return None (max_steps 兜底)                        # LLM 无限要工具 → 截停

关键设计：
- llm_client 是 Protocol，不绑死具体后端。B3 测试用 FakeLLM 注入，
  真跑用 OpenAICompatibleClient（B1 vLLM 起来后接）。这层抽象让循环
  逻辑可独立测试，不碰 GPU/网络。
- 循环退出两条件：LLM 主动停止要工具（成功）/ 超 max_steps（兜底）。
  B3 不加 Evidence Judge——LLM 自己决定答不答，B5 才加"判够不够"。
- 每跳记 trace（tool 名/参数/结果），是 B7 Langfuse 的本地雏形。
- tool_call 出错（未知工具/参数非法）不崩循环——把错误塞回 LLM 让它
  自己修正，这是 agent 鲁棒性的关键（LLM 会给错参数，不能一崩了之）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from evidence_scholar.agent.tools import RetrievalTools

logger = logging.getLogger(__name__)

# 默认最大跳数。HotpotQA 多跳问题通常 2-3 跳收敛；留 8 足够覆盖边界
# 情况，又防止 LLM 陷入无限调工具烧 token。
DEFAULT_MAX_STEPS = 8


class LLMClient(Protocol):
    """LLM 后端的抽象接口（B3 不绑死具体实现）。

    真跑用 OpenAICompatibleClient（调 vLLM 的 OpenAI 兼容 API）。
    测试用 FakeLLM 预设返回序列注入。两者都满足本 Protocol。
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> "LLMResponse":
        """One ReAct step: given the conversation so far, decide next action.

        Args:
            messages: 完整对话历史（system + user + 已有的 assistant/tool 轮）。
            tools: OpenAI tools 格式的工具签名列表。

        Returns:
            LLMResponse：含文本答案 或 一个 tool_call。
        """
        ...


@dataclass
class LLMResponse:
    """LLM 一跳的返回：要么给文本答案，要么给一个 tool_call。

    两者互斥：tool_call 非空表示 LLM 还要继续检索；为空且 text 非空表示
    LLM 决定作答。
    """

    text: str | None = None
    tool_call: "ToolCall | None" = None

    @property
    def wants_tool(self) -> bool:
        """LLM 是否要调工具（而非直接作答）。"""
        return self.tool_call is not None


@dataclass
class ToolCall:
    """LLM 请求的一次工具调用。id 用于和 tool 结果消息配对（OpenAI 格式）。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class StepTrace:
    """一跳的完整记录：LLM 调了什么工具、拿到什么、有无异常。B7 trace 雏形。"""

    step: int
    tool_call: ToolCall | None
    tool_result: str | None = None
    error: str | None = None


@dataclass
class AgentResult:
    """Agent 一次 run 的最终产物。"""

    answer: str | None
    steps: int
    stopped_reason: str  # "answered" | "max_steps" | "error"
    trace: list[StepTrace] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentState:
    """Agent 的对话状态 + 跨跳上下文（#5 Memory 的最小版，B4 会做厚）。

    B3 只记 messages 和步数。B4 会加：证据累积池、上下文窗压缩、
    跨跳摘要。先有最小可用版让 loop 能跑。
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0

    def append_assistant_tool_call(
        self, tool_call: ToolCall, text: str | None = None
    ) -> None:
        """把 LLM 这一跳的 tool_call 作为 assistant 消息塞回历史。

        OpenAI 格式要求 assistant 消息含 tool_calls 字段，且每个 tool_call
        带 id（和后续 tool 消息的 tool_call_id 配对）。
        """
        content: str | None = text if text else None
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(
                                tool_call.arguments, ensure_ascii=False
                            ),
                        },
                    }
                ],
            }
        )

    def append_tool_result(
        self, tool_call_id: str, result: str
    ) -> None:
        """把工具执行结果作为 tool 角色消息塞回历史。

        tool_call_id 必须和触发它的 assistant tool_call.id 对上，否则
        OpenAI API 会报错（vLLM 同样遵循此格式）。
        """
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            }
        )

    def append_assistant_text(self, text: str) -> None:
        """LLM 最终作答时，把答案作为 assistant 消息记下。"""
        self.messages.append(
            {"role": "assistant", "content": text}
        )


def run_agent(
    question: str,
    *,
    llm_client: LLMClient,
    tools: RetrievalTools,
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: str | None = None,
) -> AgentResult:
    """Run the ReAct loop until the LLM answers or max_steps is hit.

    Args:
        question: 用户问题。
        llm_client: LLM 后端（真跑用 OpenAICompatibleClient，测试用 FakeLLM）。
        tools: 已初始化的检索工具层（B2 的 RetrievalTools）。
        max_steps: 最大跳数兜底，防 LLM 无限调工具。
        system_prompt: 自定义 system 提示。默认用 ReAct agent 通用提示。

    Returns:
        AgentResult：含最终答案（或 None 若超限）、跳数、trace、完整 messages。
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")

    if system_prompt is None:
        system_prompt = _DEFAULT_SYSTEM_PROMPT

    state = AgentState(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    )

    trace: list[StepTrace] = []
    tool_schema = tools.schema

    while state.step < max_steps:
        response = llm_client.chat(state.messages, tools=tool_schema)

        # 分支 1：LLM 不再要工具 → 它在作答 → 成功退出。
        if not response.wants_tool:
            answer = response.text or ""
            state.append_assistant_text(answer)
            return AgentResult(
                answer=answer,
                steps=state.step + 1,
                stopped_reason="answered",
                trace=trace,
                messages=state.messages,
            )

        # 分支 2：LLM 要调工具 → 执行 → 喂回结果 → 下一跳。
        tool_call = response.tool_call
        assert tool_call is not None  # wants_tool 已保证

        step_record = StepTrace(step=state.step, tool_call=tool_call)
        # 先把 assistant 的 tool_call 塞回历史（即使执行可能出错，消息配对
        # 仍需完整——否则下一跳 LLM 看到一个悬空 tool_call 会困惑）。
        state.append_assistant_tool_call(tool_call, text=response.text)

        try:
            result = tools.execute(tool_call.name, tool_call.arguments)
            step_record.tool_result = result
            state.append_tool_result(tool_call.id, result)
        except (ValueError, KeyError) as error:
            # tool 执行失败（未知工具名/参数非法等）：不崩循环，把错误
            # 描述塞回 LLM，让它自己看到错误并修正下一跳。这是 agent
            # 鲁棒性的关键——LLM 经常传错参数，不能一崩了之。
            err_msg = f"Tool execution failed: {error}"
            step_record.error = err_msg
            logger.warning(
                "step %d tool %s failed: %s",
                state.step, tool_call.name, error,
            )
            state.append_tool_result(tool_call.id, err_msg)

        trace.append(step_record)
        state.step += 1

    # 走到这里 = LLM 一直要工具、超过 max_steps 仍没作答 → 兜底截停。
    return AgentResult(
        answer=None,
        steps=state.step,
        stopped_reason="max_steps",
        trace=trace,
        messages=state.messages,
    )


_DEFAULT_SYSTEM_PROMPT = (
    "You are an evidence-grounded research agent. To answer the user's "
    "question, you may call the retrieve_hybrid tool to search a document "
    "corpus. Reformulate the question into focused retrieval queries and "
    "gather evidence across multiple rounds if the question is multi-hop. "
    "When you have enough evidence, answer directly without calling any "
    "tool. If a tool call fails, read the error and adjust your next call. "
    "Keep answers grounded in retrieved evidence."
)
