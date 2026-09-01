"""Langfuse tracing layer (B7): observe agent runs + send spans to Langfuse.

B7 给 agent 加全链路可观测性。地基是 react.py 的 StepTrace（每跳记
tool 名/参数/结果/错误的本地雏形），B7 把它接通专业 trace 平台
Langfuse 云版——浏览器看树状 span、token 成本、耗时。

设计要点（可观测性铁律）：
- **可选层**：Langfuse 没配 key / SDK 挂了 / 网络断，agent 照常跑。
  Trace 采集失败绝不拖垮主流程。TraceContext 所有方法是 no-op-safe。
- **不污染函数签名**：react.py 的 run_agent 只多一个可选 trace_context
  参数，trace 通过 TraceContext 对象在循环里手动开/关 span。
- **截断发云端**：文档 text 截断再发，省云额度 + 不泄露语料全文。
  只发 query/title/score/长度等元数据 + 截断片段。
- **keys 走环境变量**：LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL 从 env
  读，不写进代码/配置文件（和 GitHub PAT 同原则，跑完即弃）。

Langfuse 4.x OTel API 用法：
- start_as_current_observation 返回 context manager（要 with 用），但
  对循环不友好。改用从 span 对象的 start_observation 建子 span——
  LangfuseSpan 自带 start_observation 方法，自动挂到当前 span 下，
  直接返回 span 对象（有 .end()），无需 context manager 套娃。

trace 结构（树状）：
    trace: run_agent(question)
      ├─ span step-0 retrieve_hybrid   [query, top_k, #docs, latency]
      │    └─ span llm_call            [model, token usage]
      │    └─ span tool_result         [result preview]
      ├─ span step-1 judge_evidence    [sufficient, reason, answer]
      └─ ...
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# 发云端的文档/结果片段最大字符数。够 Langfuse 看清内容、不爆额度、
# 不泄露语料全文。比 evidence_pool 摘要的 400 略宽——trace 要能看清。
_MAX_TRACE_CHARS = 600


def _truncate(text: str | None, limit: int = _MAX_TRACE_CHARS) -> str:
    """截断长文本，补省略号。None 返回空串。"""
    if not text:
        return ""
    if len(text) > limit:
        return text[:limit] + "…"
    return text


class TraceContext:
    """一次 agent run 的 trace 上下文，封装 Langfuse span 管理。

    无 Langfuse 时（_langfuse 为 None）所有方法 no-op，不崩主流程。

    span 管理：根 span 在 start() 用 with-as-context 拿到（OTel context
    在 with 块内 active），但 react 循环跨多跳、不在单个 with 里。故用
    _root_span 对象的 start_observation 方法建子 span（自动挂根下，返回
    span 对象有 .end()，无需 context manager 套娃）。
    """

    def __init__(self, trace_name: str, question: str) -> None:
        """私有构造，用 TraceContext.start() 建实例。"""
        self._name = trace_name
        self._question = question
        self._langfuse = None
        self._trace_id: str | None = None
        self._root_span: Any = None  # 根 span（LangfuseSpan 对象）
        self._current_step_span: Any = None  # 当前 step 的 span

    @classmethod
    def start(
        cls, question: str, *, name: str = "agent_run"
    ) -> "TraceContext":
        """开始一次 trace。配了 Langfuse key 才真连云端，否则 no-op。

        通过环境变量 LANGFUSE_PUBLIC_KEY/SECRET_KEY 判定是否启用——
        有 key 就初始化 Langfuse 客户端并 auth_check；没 key 直接
        no-op，agent 照跑。BASE_URL 可选（JP/US region 切换）。
        """
        ctx = cls(name, question)
        if not (
            os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")
        ):
            # 没配 key → no-op 模式。agent 主流程不受影响。
            logger.debug("Langfuse keys not set, tracing disabled (no-op).")
            return ctx

        try:
            from langfuse import Langfuse

            base_url = os.environ.get("LANGFUSE_BASE_URL")
            # timeout=30：国内到 Langfuse 云（尤其 JP/US region）握手偶慢，
            # 默认 timeout 短会 SSL 超时误判。30s 足够且不阻塞主流程太久。
            ctx._langfuse = Langfuse(base_url=base_url, timeout=30) if base_url \
                else Langfuse(timeout=30)
            # auth_check 探活：key 错/host 错/网络断会抛，catch 住降级 no-op。
            ctx._langfuse.auth_check()
            # 起根 span：用 with 让 OTel context 在块内 active，拿到 span 对象
            # 后离开块（块内拿 span）。根 span 不在 with 结束时 end——finish 时
            # 手动 end，保证子 span 都挂它下面。
            with ctx._langfuse.start_as_current_observation(
                name=name,
                input={"question": _truncate(question, _MAX_TRACE_CHARS)},
            ) as root_span:
                ctx._root_span = root_span
                ctx._trace_id = ctx._langfuse.get_current_trace_id()
            logger.info(
                "Langfuse trace started: %s",
                ctx._langfuse.get_trace_url(trace_id=ctx._trace_id)
                if ctx._trace_id else "(no trace id)",
            )
        except Exception as error:  # 任何 Langfuse 故障都降级 no-op。
            logger.warning(
                "Langfuse init failed, tracing disabled: %s", error
            )
            ctx._langfuse = None
            ctx._root_span = None
        return ctx

    @property
    def enabled(self) -> bool:
        """trace 是否真连云端（False = no-op 模式）。"""
        return self._langfuse is not None

    @property
    def trace_url(self) -> str | None:
        """云端 trace 的 URL（no-op 模式返回 None）。"""
        if not self.enabled:
            return None
        try:
            # 4.x: get_trace_url 接受 trace_id 关键字参数；无 trace_id 时
            # 返回当前 context 的 trace url。
            return self._langfuse.get_trace_url(  # type: ignore[union-attr]
                trace_id=self._trace_id
            ) if self._trace_id else self._langfuse.get_trace_url()
        except Exception:
            return None

    def start_step(self, step: int, tool_call: Any) -> None:
        """开一个 step span（每跳一个），挂在根 span 下。

        用 _root_span.start_observation 建子 span——自动挂根下，返回
        span 对象（有 .end()），无需 context manager。
        """
        if not self.enabled or self._root_span is None:
            return
        try:
            name = getattr(tool_call, "name", "unknown")
            args = getattr(tool_call, "arguments", {})
            self._current_step_span = self._root_span.start_observation(
                name=f"step-{step} {name}",
                as_type="tool",
                input={
                    "step": step,
                    "tool": name,
                    "arguments": _truncate(
                        json.dumps(args, ensure_ascii=False, default=str)
                    ),
                },
            )
        except Exception as error:
            logger.debug("start_step failed (no-op): %s", error)

    def record_llm(
        self,
        *,
        model: str,
        usage: dict[str, int] | None,
        response_preview: str,
    ) -> None:
        """记 LLM 调用 span（含 token usage 算成本）。

        usage 是 OpenAI 格式 {prompt_tokens, completion_tokens, total_tokens}。
        从 _current_step_span 建子 span（挂在当前 step 下）。
        """
        if not self.enabled or self._current_step_span is None:
            return
        try:
            self._current_step_span.start_observation(
                name="llm_call", as_type="generation",
                model=model,
                input={"response_preview": _truncate(response_preview)},
                usage_details=usage,  # Langfuse 4.x 用 usage_details 而非 usage
            ).end()
        except Exception as error:
            logger.debug("record_llm failed (no-op): %s", error)

    def record_tool_result(
        self, result: str, *, error: str | None = None
    ) -> None:
        """记工具执行结果 span（截断防爆）。挂在当前 step 下。"""
        if not self.enabled or self._current_step_span is None:
            return
        try:
            payload = {"result": _truncate(result)}
            if error:
                payload["error"] = _truncate(error, 200)
            self._current_step_span.start_observation(
                name="tool_result", output=payload,
            ).end()
        except Exception as error_:
            logger.debug("record_tool_result failed (no-op): %s", error_)

    def end_step(self) -> None:
        """结束当前 step span。"""
        if not self.enabled or self._current_step_span is None:
            return
        try:
            self._current_step_span.end()
        except Exception as error:
            logger.debug("end_step failed (no-op): %s", error)
        self._current_step_span = None

    def finish(
        self, *, answer: str | None, stopped_reason: str,
        steps: int, evidence_pool_size: int,
    ) -> None:
        """trace 收尾：记最终输出到根 span，flush 到云端。

        注意：根 span 在 start() 的 with 块退出时已被 OTel 自动 end，
        故这里不再 .end()（会触发"Calling end() on an ended span"）。
        update 仍可写输出（Langfuse 4.x 支持在 span 结束后 update 元数据，
        flush 时一并提交）。
        """
        if not self.enabled:
            return
        try:
            # 4.x OTel API：根 span 用 update 写输出（span 已 end 仍可 update）。
            if self._root_span is not None:
                self._root_span.update(
                    output={
                        "answer": _truncate(answer),
                        "stopped_reason": stopped_reason,
                        "steps": steps,
                        "evidence_pool_size": evidence_pool_size,
                    }
                )
            self._langfuse.flush()  # type: ignore[union-attr]
        except Exception as error:
            logger.warning("Langfuse finish/flush failed: %s", error)
