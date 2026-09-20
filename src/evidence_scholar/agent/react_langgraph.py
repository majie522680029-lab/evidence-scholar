"""LangGraph rewrite of the ReAct loop (C2, internship #7).

B3 的 run_agent 是手写 while 循环；本模块用 LangGraph 框架把同一逻辑
表达成状态图（state graph）：节点 = 函数，边 = 状态转移，条件边 = 根据
LLM 返回决定下一步。和 react.py 并存，用同一套 FakeLLM 测试验证行为一致。

为什么并存两版而非替换：
- 实习要求 #7 要用 LangGraph，但 #3 要"自己实现 ReAct"。两版并存才能
  在简历/面试讲清"手写版替我解决了什么、框架版带来了什么代价"。
- 手写版（react.py）是 ground truth：逻辑直接、无框架依赖、好调试。
  LangGraph 版（本文件）是对照：展示框架的图原语怎么表达循环。

图结构（等价于 react.py 的 while 循环）：
    START -> call_llm -> [route]
                        ├── no tool_call  -> END (退出 A, text 作答)
                        └── has tool_call -> execute_tool -> [route_after_tool]
                                                            ├── judge sufficient -> END (退出 B)
                                                            └── else -> call_llm (下一跳)

和 react.py 的逐行对应：
- call_llm 节点      = react.py:285  llm.chat(messages, tools)
- route 条件边       = react.py:298  if not response.wants_tool
- execute_tool 节点  = react.py:316-353  消息配对 + 工具执行 + B4 注入
- route_after_tool   = react.py:359-370  if judge sufficient
- recursion_limit    = react.py:284  while step < max_steps

关键设计（必须和 react.py 一致，否则行为不等价）：
1. 两条退出路径都保留：退出 A（LLM 不调工具，text 作答）/ 退出 B
   （judge sufficient=true，结构化 answer）。
2. B4 证据池注入时机：只在 retrieve 后注入，judge 不注入。
3. max_steps 兜底：用 recursion_limit（图遍历深度上限），每跳 2 节点，
   所以 recursion_limit = max_steps * 2 + 2（余量给 START/END）。
4. tool 错误不崩循环：未知工具/参数非法把错误塞回 LLM。
5. judge answer 多级兜底：judge.answer -> response.text -> ""。

LangGraph 替我们做的事（面试可讲"框架解决了什么"）：
- State reducer（add_messages）自动管消息累加，不用手动 append。
- checkpoint：compile(checkpointer=...) 可加持久化，支持中断/恢复。
- 图可视化：compiled.get_graph().draw_* 看节点拓扑。
- recursion_limit 统一兜底。

LangGraph 的代价（面试可讲"框架带来了什么"）：
- 抽象开销：State/reducer/节点函数多一层间接，调试不如手写直接。
- State 类型约束：要 TypedDict + Annotated，不如 list[dict] 灵活。
- 条件边返回字符串路由，拼错名只在运行时报错。
- State 在节点间是"旧值 + reducer 合并"，不如手写版的直接读写直觉。
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from evidence_scholar.agent.evidence_pool import EvidencePool
from evidence_scholar.agent.react import (
    DEFAULT_MAX_STEPS,
    AgentResult,
    LLMClient,
    LLMResponse,
    StepTrace,
    ToolCall,
    _DEFAULT_SYSTEM_PROMPT,
)
from evidence_scholar.agent.tools import RetrievalTools

logger = logging.getLogger(__name__)


class _AgentState(TypedDict):
    """LangGraph 的 agent 状态。

    messages 用 add_messages reducer：节点返回新消息，reducer 自动追加，
    不用手动维护历史。其它字段无 reducer（默认后者覆盖前者）。
    """
    messages: Annotated[list[dict[str, Any]], add_messages]
    step: int
    evidence_pool: EvidencePool
    answer: str | None
    stopped_reason: str
    trace: list[StepTrace]
    last_response: LLMResponse  # call_llm 写，route 读


def run_agent_langgraph(
    question: str,
    *,
    llm_client: LLMClient,
    tools: RetrievalTools,
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: str | None = None,
) -> AgentResult:
    """Run the ReAct loop via LangGraph. 行为等价于 react.run_agent。

    两版必须通过同一套 FakeLLM 测试（见 test_react_langgraph.py）。
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")
    if system_prompt is None:
        system_prompt = _DEFAULT_SYSTEM_PROMPT

    tool_schema = tools.schema

    # ---------------- 节点函数（读旧 state，返回要更新的字段） ----------------

    def call_llm(state: _AgentState) -> dict[str, Any]:
        """调 LLM。对应 react.py:285-304。

        若 LLM 不调工具（退出 A），顺便把 answer/stopped_reason 写进 state，
        route 只管路由。退出 A 的 assistant text 消息也写回（对应
        react.py:300 append_assistant_text）。
        """
        response = llm_client.chat(state["messages"], tools=tool_schema)
        updates: dict[str, Any] = {"last_response": response}
        if not response.wants_tool:
            # 退出 A：LLM 直接 text 作答（B3 兜底路径）
            answer = response.text or ""
            updates["messages"] = [{"role": "assistant", "content": answer}]
            updates["answer"] = answer
            updates["stopped_reason"] = "answered"
            updates["step"] = state["step"] + 1
        return updates

    def execute_tool(state: _AgentState) -> dict[str, Any]:
        """执行 tool_call + 消息配对 + B4 证据池注入。

        对应 react.py:316-353。退出 B（judge sufficient）也在这里判，
        因为 execute 后才有 judge 的结构化结果可读。
        """
        response = state["last_response"]
        tool_call = response.tool_call
        assert tool_call is not None  # route 已保证有 tool_call

        step = state["step"]
        evidence_pool = state["evidence_pool"]
        trace = list(state["trace"])  # 复制避免改到旧引用

        # assistant tool_call 消息（对应 react.py:316 append_assistant_tool_call）
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.text if response.text else None,
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

        new_msgs: list[dict[str, Any]] = [assistant_msg]

        # 执行工具（对应 react.py:319-337）
        step_record = StepTrace(step=step, tool_call=tool_call)
        executed_ok = True
        result: str
        try:
            result = tools.execute(tool_call.name, tool_call.arguments)
            # B4 证据池注入（对应 react.py:325-332）：只在 retrieve 后注入
            if tool_call.name == tools.HYBRID_TOOL_NAME:
                last_query, last_results = tools.last_retrieval
                evidence_pool.add(
                    last_results, query=last_query, hop=step
                )
                result = result + "\n\n" + evidence_pool.summarize()
            step_record.tool_result = result
        except (ValueError, KeyError) as error:
            # tool 错误不崩循环（对应 react.py:337-350）：错误塞回 LLM
            executed_ok = False
            result = f"Tool execution failed: {error}"
            step_record.error = result

        new_msgs.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        )
        trace.append(step_record)

        updates: dict[str, Any] = {
            "messages": new_msgs,
            "step": step + 1,
            "evidence_pool": evidence_pool,
            "trace": trace,
        }

        # 退出 B（对应 react.py:359-370）：judge sufficient -> 结构化 answer
        if executed_ok and tool_call.name == tools.JUDGE_TOOL_NAME:
            judge = tools.parse_judge(tool_call.arguments)
            if judge["sufficient"]:
                # 多级兜底（对应 react.py:364）
                answer = judge["answer"] or response.text or ""
                updates["answer"] = answer
                updates["stopped_reason"] = "answered"

        return updates

    # ---------------- 条件边路由（返回下一节点名字） ----------------

    def route(state: _AgentState) -> str:
        """LLM 调没调工具？对应 react.py:298。"""
        if not state["last_response"].wants_tool:
            return END  # 退出 A
        return "execute_tool"

    def route_after_tool(state: _AgentState) -> str:
        """judge sufficient？对应 react.py:359-370 的路由逻辑。

        execute_tool 已把 sufficient 判定写进 stopped_reason（若退出 B）。
        这里读 stopped_reason 判断是否到 END，否则回 call_llm 下一跳。
        """
        if state.get("stopped_reason") == "answered":
            return END  # 退出 B
        return "call_llm"

    # ---------------- 建图 ----------------
    graph = StateGraph(_AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tool", execute_tool)
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm", route, {END: END, "execute_tool": "execute_tool"}
    )
    graph.add_conditional_edges(
        "execute_tool", route_after_tool, {END: END, "call_llm": "call_llm"}
    )

    # recursion_limit 兜底（对应 react.py:284 while max_steps）：
    # 每跳走 call_llm -> execute_tool 2 节点，max_steps 跳 ≈ 2*max_steps 步。
    # +2 给 START 和 END 的边。超限 LangGraph 抛 GraphRecursionError，
    # 我们捕获转成 max_steps 兜底（answer=None）。
    # 注意：recursion_limit 是 invoke() 的参数，不是 compile() 的（1.2.x API）。
    compiled = graph.compile()

    initial: _AgentState = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "step": 0,
        "evidence_pool": EvidencePool(),
        "answer": None,
        "stopped_reason": "",
        "trace": [],
        "last_response": LLMResponse(),
    }

    try:
        final_state = compiled.invoke(
            initial, config={"recursion_limit": max_steps * 2 + 2}
        )
    except Exception:
        # GraphRecursionError 或其它异常 -> max_steps 兜底
        # （对应 react.py:376 return _finish(answer=None, "max_steps")）
        return AgentResult(
            answer=None,
            steps=max_steps,
            stopped_reason="max_steps",
            trace=[],
            messages=initial["messages"],
            evidence_pool=initial["evidence_pool"],
        )

    stopped = final_state.get("stopped_reason") or "max_steps"
    return AgentResult(
        answer=final_state.get("answer"),
        steps=final_state.get("step", 0),
        stopped_reason=stopped,
        trace=final_state.get("trace", []),
        messages=final_state.get("messages", []),
        evidence_pool=final_state.get("evidence_pool", EvidencePool()),
    )
