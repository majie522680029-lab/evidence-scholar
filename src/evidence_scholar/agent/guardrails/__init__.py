"""Guardrails package (C3): input/tool/output safety layers.

当前实现：input_guard（简版，关键词正则 + 长度 + 空白）。
后续可选：tool_guard（工具调用参数校验）、output_guard（输出泄漏检测）。
"""

from evidence_scholar.agent.guardrails.input_guard import (
    GuardrailDecision,
    check_input,
)

__all__ = ["GuardrailDecision", "check_input"]
