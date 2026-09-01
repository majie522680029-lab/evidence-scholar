"""ReAct agent loop (B3 + B5 + B4 + B7): Reason + Act over retrieval tools + evidence judge + evidence pool + Langfuse tracing.

B3 是 v2 agent 的"大脑"——手写 ReAct 循环。为什么手写而非直接上
LangGraph：实习要求 #3 明确要"自己实现 ReAct Agent"。手写一遍才真懂
agent 的运行机制（消息配对、终止判定、循环兜底），C2 用 LangGraph
重写时才有对比、能讲"框架替我解决了什么"。

B5 给循环加了 Evidence Judge + Query Rewrite 的退出语义：LLM 不再靠
"停止调工具"作答，而是显式调 judge_evidence 工具，用结构化字段
（sufficient/next_query/answer）宣告"够不够"。判定结构走 tool-call
通道产出（B5 方案 3），绕开 Qwen3 thinking 混进 content 破坏 JSON 的
稳定性问题——hermes parser 在 decode 期就约束了 tool_call 参数。

B4 给循环加了证据累积池（EvidencePool）：替 agent 把每跳 retrieve
命中的证据去重累积、整理成摘要，注入到 retrieve 的 tool 结果消息里
（方案 B：系统自动注入，非 LLM 自己填）。LLM 下一跳看到的就不只是
"这次检索返回了啥"，还有"到目前为止累积的所有证据"的紧凑视图——
多跳金证据不丢、judge 组织答案有据可依、倾向简洁作答（对准 B6 暴露
的答案啰嗦问题）。命中实习要求 #4 Long Context / Memory Management。

ReAct 循环（每跳 Reason + Act）：
    messages = [system + user]
    pool = EvidencePool()
    while step < max_steps:
        response = llm.chat(messages, tools)          # LLM 想一下
        if response 无 tool_call:                     # 退出 A（B3 兜底）
            return response.text
        tool_call = response.tool_call
        result = tools.execute(tool_call)              # 执行 LLM 指派的动作
        if tool_call 是 retrieve:
            pool.add(result_docs, query, hop)           # B4：证据入池
            result += pool.summarize()                  # 方案 B：摘要注入
        messages += [assistant(tool_call), tool(result)]  # 喂回 LLM 观察
        if tool_call 是 judge 且 sufficient=true:      # 退出 B（B5 主路径）
            return judge.answer
        step += 1
    return None (max_steps 兜底)                        # LLM 无限要工具 → 截停

关键设计：
- llm_client 是 Protocol，不绑死具体后端。B3 测试用 FakeLLM 注入，
  真跑用 OpenAICompatibleClient（B1 vLLM 起来后接）。这层抽象让循环
  逻辑可独立测试，不碰 GPU/网络。
- 循环退出两路径：退出 A=LLM 主动停要工具（成功，text 作答）/ 退出 B=
  LLM 调 judge_evidence 且 sufficient=true（成功，结构化 answer 作答）。
  超 max_steps 仍兜底截停。两条成功路径并存增强鲁棒——LLM 忘了调
  judge 直接 text 作答也能收。
- judge sufficient=false 不退出：把 ack（含 next_query）塞回 LLM，
  让 LLM 下一跳自己拿 next_query 去 retrieve。保持 LLM 是控制器，
  这是 ReAct 的核心；不硬塞检索动作。
- B4 证据池注入点：retrieve 执行后把 pool.summarize() 拼进 tool 结果
  消息（方案 B）。judge 不注入——judge 和 sufficient 退出在同一个
  tool_call 里，注入也来不及影响这次判定；judge 靠上一跳 retrieve
  已注入的摘要作决策。故注入只在 retrieve 后，时机正确。
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

from evidence_scholar.agent.evidence_pool import EvidencePool
from evidence_scholar.agent.tools import RetrievalTools
from evidence_scholar.agent.trace import TraceContext

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
    # B4：累积证据池。B6 评测可读 pool.size 看证据利用情况，B7 trace 可
    # 序列化。默认空池（max_steps 兜底路径也返回池，便于统一分析）。
    evidence_pool: EvidencePool = field(default_factory=EvidencePool)


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
    trace_context: TraceContext | None = None,
) -> AgentResult:
    """Run the ReAct loop until the LLM answers or max_steps is hit.

    两条成功退出：A=LLM 不再调工具（text 作答，B3 兜底）/ B=LLM 调
    judge_evidence 且 sufficient=true（结构化 answer 作答，B5 主路径）。
    超 max_steps 兜底截停，answer=None。

    Args:
        question: 用户问题。
        llm_client: LLM 后端（真跑用 OpenAICompatibleClient，测试用 FakeLLM）。
        tools: 已初始化的检索工具层（B2/B5 的 RetrievalTools，含 retrieve
            + judge 两工具）。
        max_steps: 最大跳数兜底，防 LLM 无限调工具。
        system_prompt: 自定义 system 提示。默认用 ReAct agent 通用提示。
        trace_context: B7 Langfuse trace 上下文。None 时自动建（配了 key
            才真连云端，否则 no-op）。trace 挂了不崩主流程（TraceContext
            自降级 no-op）。

    Returns:
        AgentResult：含最终答案（或 None 若超限）、跳数、trace、完整 messages。
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")

    if system_prompt is None:
        system_prompt = _DEFAULT_SYSTEM_PROMPT

    # B7：开 trace 根 span（配了 Langfuse key 才真连云端，否则 no-op）。
    if trace_context is None:
        trace_context = TraceContext.start(question)

    state = AgentState(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    )

    trace: list[StepTrace] = []
    tool_schema = tools.schema
    # B4：跨跳累积证据池。每跳 retrieve 后入池，summarize() 注入到
    # retrieve 的 tool 结果消息（方案 B），judge 下一跳就能看到整理后证据。
    evidence_pool = EvidencePool()

    def _finish(
        *, answer: str | None, stopped_reason: str, steps: int
    ) -> AgentResult:
        """统一收尾：记 trace 最终输出 + flush，返回 AgentResult。"""
        trace_context.finish(
            answer=answer, stopped_reason=stopped_reason,
            steps=steps, evidence_pool_size=evidence_pool.size,
        )
        return AgentResult(
            answer=answer,
            steps=steps,
            stopped_reason=stopped_reason,
            trace=trace,
            messages=state.messages,
            evidence_pool=evidence_pool,
        )

    while state.step < max_steps:
        response = llm_client.chat(state.messages, tools=tool_schema)
        # B7：记 LLM 调用 span（含 token usage 算成本）。usage 可能 None
        # （FakeLLM 或客户端没暴露）——TraceContext 自己容错。
        trace_context.record_llm(
            model=getattr(llm_client, "_model", "unknown"),
            usage=getattr(response, "usage", None),
            response_preview=response.text or
                (str(response.tool_call) if response.tool_call else ""),
        )

        # 退出 A（B3 原有，保留为兜底）：LLM 不再要工具 → 它在作答 → 成功退出。
        # B5 下这是次要路径——主路径走 judge；但 LLM 若忘调 judge 直接 text
        # 作答，这条路径仍能收尾，增强鲁棒。
        if not response.wants_tool:
            answer = response.text or ""
            state.append_assistant_text(answer)
            return _finish(
                answer=answer, stopped_reason="answered",
                steps=state.step + 1,
            )

        # 分支 2：LLM 要调工具 → 执行 → 喂回结果 → 看是否 judge 退出。
        tool_call = response.tool_call
        assert tool_call is not None  # wants_tool 已保证

        # B7：开 step span（每跳一个，挂根 trace 下）。
        trace_context.start_step(state.step, tool_call)

        step_record = StepTrace(step=state.step, tool_call=tool_call)
        # 先把 assistant 的 tool_call 塞回历史（即使执行可能出错，消息配对
        # 仍需完整——否则下一跳 LLM 看到一个悬空 tool_call 会困惑）。
        state.append_assistant_tool_call(tool_call, text=response.text)

        executed_ok = True
        try:
            result = tools.execute(tool_call.name, tool_call.arguments)
            # B4 方案 B 注入点：retrieve 执行后，把命中证据入池，再把证据池
            # 摘要拼进 tool 结果消息。judge 不注入（judge 与 sufficient 退出在
            # 同一 tool_call，注入来不及影响这次判定；judge 靠上一跳 retrieve
            # 已注入的摘要决策，时机正确）。
            if tool_call.name == tools.HYBRID_TOOL_NAME:
                # last_retrieval 是 (query, raw_results)，_execute_hybrid 设的。
                last_query, last_results = tools.last_retrieval
                evidence_pool.add(
                    last_results, query=last_query, hop=state.step
                )
                # 摘要拼在 retrieve 结果后，LLM 下一跳（含 judge）能一并看到。
                result = result + "\n\n" + evidence_pool.summarize()
            step_record.tool_result = result
            state.append_tool_result(tool_call.id, result)
            # B7：记工具结果 span。
            trace_context.record_tool_result(result)
        except (ValueError, KeyError) as error:
            # tool 执行失败（未知工具名/参数非法等）：不崩循环，把错误
            # 描述塞回 LLM，让它自己看到错误并修正下一跳。是 agent 鲁棒
            # 性的关键——LLM 经常传错参数，不能一崩了之。
            executed_ok = False
            err_msg = f"Tool execution failed: {error}"
            step_record.error = err_msg
            logger.warning(
                "step %d tool %s failed: %s",
                state.step, tool_call.name, error,
            )
            state.append_tool_result(tool_call.id, err_msg)
            # B7：记工具错误 span（错误也记进 trace，便于失败分析）。
            trace_context.record_tool_result(err_msg, error=err_msg)

        # B7：结束当前 step span。
        trace_context.end_step()

        # 退出 B（B5 新增）：judge 判定 sufficient → 用结构化 answer 退出。
        # 只在执行成功时判（执行失败已塞回 LLM 自修正，不在此退出）。
        # sufficient=false 不退出：ack 含 next_query，下一跳 LLM 自己拿去
        # retrieve，保持 LLM 是控制器（ReAct 核心）。
        if executed_ok and tool_call.name == tools.JUDGE_TOOL_NAME:
            judge = tools.parse_judge(tool_call.arguments)
            if judge["sufficient"]:
                # 优先用 judge 结构化 answer；空则回退 LLM 这一跳的 text；
                # 再空则空串。多级兜底防 LLM 把答案塞错地方。
                answer = judge["answer"] or response.text or ""
                trace.append(step_record)
                state.step += 1
                return _finish(
                    answer=answer, stopped_reason="answered",
                    steps=state.step,
                )

        trace.append(step_record)
        state.step += 1

    # 走到这里 = LLM 一直要工具、超过 max_steps 仍没作答 → 兜底截停。
    return _finish(
        answer=None, stopped_reason="max_steps", steps=state.step,
    )


_DEFAULT_SYSTEM_PROMPT = (
    "You are an evidence-grounded research agent. To answer the user's "
    "question, call the retrieve_hybrid tool to search a document corpus; "
    "reformulate the question into focused retrieval queries and gather "
    "evidence across multiple rounds if the question is multi-hop. After "
    "each retrieval, the tool result includes an [Evidence Pool] summary of "
    "all evidence gathered so far across hops—use it to track what you "
    "know and avoid losing earlier evidence. When ready, call the "
    "judge_evidence tool to decide whether the evidence is sufficient: set "
    "sufficient=true and write the final answer in the answer field (keep "
    "the answer concise—the answer itself, not a full sentence), or set "
    "sufficient=false with a sharper next_query to retrieve more. You are "
    "done once judge_evidence returns sufficient=true. If a tool call fails, "
    "read the error and adjust your next call. Keep answers grounded in "
    "retrieved evidence."
)
