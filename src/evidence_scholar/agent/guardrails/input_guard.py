"""Input guardrail: first-pass check on user question (C3 layer 1).

做什么：在 question 进 ReAct 循环之前过一道，拦两类低水平但高频的风险：
  1. 长度上限（防 token 轰炸 / 拖垮检索 / 撑爆上下文窗）
  2. prompt injection 关键词（模板攻击的兜底哨兵）

为什么是"哨兵"而不是"终极防御"——必须讲清楚（面试 / README 都要诚实标注）：
关键词/正则挡的是"字符串特征"，而 prompt injection 是"语义攻击"。
- 同义改写（"disregard prior" 无特征词）、换语言（中文"忽略上面指令"）、
  插空格/拆字（"i g n o r e"）都能绕过本层。
- 真正起作用的是系统设计：agent 工具集只读无副作用（retrieve/judge），
  即使被劫持也干不了破坏性的事。本层只是纵深防御（defense-in-depth）
  的最外层低成本闸，挡掉低水平模板攻击，省后面 LLM 算力，不是银弹。
- 语义层防御（LLM-as-judge）和 indirect injection 检测是后续可选增强。

返回结构：GuardrailDecision(action, reason)。不返回裸 bool，因为
- 不可追踪：block 了不知哪条规则命中
- 不可组合：后续加 tool/output guard 时多 decision 怎么聚合
reason 记命中规则名，全程可观测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 用户问题最大字符数。HotpotQA 问题通常 < 200 字符；留 2000 既宽松覆盖
# 长学术问题，又挡掉 token 轰炸 / 拖垮 BM25 的超长输入。
MAX_QUESTION_CHARS = 2000

# Prompt injection 关键词模式（按攻击语义意图分四类）。用正则而非纯
# 字符串 in：re.IGNORECASE 挡大小写，\b 词边界挡 "ignores" 误命中。
# 注意：这是"挡模板攻击"的特征词表，不是穷举——语义改写就绕过。
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # ① 指令否定类：试图覆盖 system prompt
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget)\b.*\b(previous|prior|above|instructions?|commands?)\b",
            re.IGNORECASE,
        ),
    ),
    # ② 身份重设类：试图改写 agent 角色
    (
        "identity_reset",
        re.compile(
            r"\b(you are now|act as|pretend you are|from now on you)\b",
            re.IGNORECASE,
        ),
    ),
    # ③ 系统提示泄露类：套取 system prompt
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(show|reveal|print|output|what are|repeat)\b.*\b(system prompt|instructions?|rules?|initial)\b"
            r"|^\s*system\s*:",
            re.IGNORECASE,
        ),
    ),
    # ④ 越狱/开发者模式类：经典越狱话术
    (
        "jailbreak",
        re.compile(
            r"\b(DAN|developer mode|jailbreak|do anything now|unrestricted mode)\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass(frozen=True)
class GuardrailDecision:
    """单条 guardrail 的判定结果。

    action="allow" 时 reason 为空串；action="block" 时 reason 记命中规则名
    （如 "too_long" / "instruction_override"），便于 trace 和回溯。
    """

    action: str  # "allow" | "block"
    reason: str

    @property
    def is_blocked(self) -> bool:
        """是否被拦截。"""
        return self.action == "block"


def check_input(question: str) -> GuardrailDecision:
    """对用户 question 做输入护栏检查。

    检查顺序：空/空白 -> 长度 -> injection 关键词。
    命中任一即 block，返回对应 reason。

    Args:
        question: 用户原始问题。

    Returns:
        GuardrailDecision：allow 放行进 ReAct 循环，block 直接拒绝。
    """
    # 空白问题：pydantic/schema 层也会挡，这里纵深防御再挡一次
    if not question or not question.strip():
        return GuardrailDecision(action="block", reason="empty_question")

    # 长度上限：防 token 轰炸 / 拖垮检索
    if len(question) > MAX_QUESTION_CHARS:
        return GuardrailDecision(action="block", reason="too_long")

    # Prompt injection 关键词：模板攻击兜底（语义改写会绕过，见模块 docstring）
    for rule_name, pattern in _INJECTION_PATTERNS:
        if pattern.search(question):
            return GuardrailDecision(action="block", reason=rule_name)

    return GuardrailDecision(action="allow", reason="")
