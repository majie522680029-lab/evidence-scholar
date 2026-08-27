"""OpenAI-compatible LLM client (talks to vLLM via the OpenAI SDK).

B3 的 LLMClient Protocol 的真实实现。vLLM 起的 OpenAI 兼容 API 和
OpenAI 官方接口同形，所以直接用 openai SDK 调——客户端代码和调云
OpenAI 一模一样，差别只是 base_url 指向本地 vLLM。

命中实习要求 #1（LLM API）：标准 OpenAI SDK 调用 + tool calling +
消息拼装全在这里。B1 起 vLLM 服务后，把 base_url 指向它即可联调。

设计要点：
- 只做"调一次 chat completion 并解析成 LLMResponse"这一件事，不
  碰循环逻辑（循环在 react.py）。保持单一职责。
- 解析 tool_calls：OpenAI 响应可能含 0 或多个 tool_calls。ReAct 每跳
  只取第一个（B3 单工具起步，多 tool_call 留后期）。
- arguments 是 JSON 字符串，要 json.loads 成 dict 才能喂给 tools.execute。
- Qwen3 thinking mode：vLLM serve 时若开 thinking，响应里 reasoning_content
  单独字段，不影响 content/tool_calls 解析。本客户端暂不处理 thinking
  文本（结构化输出场景用 /no_think，见 B5）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from evidence_scholar.agent.react import LLMResponse, ToolCall

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """LLMClient 实现：通过 OpenAI SDK 调 vLLM 的 OpenAI 兼容 API。

    Args:
        base_url: vLLM 服务地址，如 "http://127.0.0.1:8000/v1"。
        model: 模型名（vLLM --served-model-name，或权重路径）。
        api_key: vLLM 默认不校验，但 SDK 必须传非空，用占位 "EMPTY"。
        temperature: 采样温度。agent 推理用 0（确定性）便于复现评测。
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen3-8B",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """One chat completion call → parse into LLMResponse.

        Args:
            messages: 完整对话历史。
            tools: OpenAI tools 格式签名。

        Returns:
            LLMResponse：含文本答案 或 一个 tool_call。
        """
        # vLLM/OpenAI 兼容：tools 字段传 None 时表示不限制工具，这里始终
        # 把 tools schema 传进去，让模型知道有 retrieve_hybrid 可用。
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            temperature=self._temperature,
        )

        choice = response.choices[0]
        message = choice.message
        tool_calls = message.tool_calls or []

        # 分支 1：LLM 想调工具 → 取第一个 tool_call（B3 单工具起步）。
        if tool_calls:
            first = tool_calls[0]
            # arguments 是 JSON 字符串，解析成 dict；解析失败兜底成空 dict
            # 而非抛异常（让 react loop 把错误喂回 LLM 自我修正）。
            try:
                arguments = json.loads(first.function.arguments)
                if not isinstance(arguments, dict):
                    arguments = {}
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse tool arguments: %s",
                    first.function.arguments,
                )
                arguments = {}

            return LLMResponse(
                text=None,
                tool_call=ToolCall(
                    id=first.id,
                    name=first.function.name,
                    arguments=arguments,
                ),
            )

        # 分支 2：LLM 直接作答。
        return LLMResponse(text=message.content, tool_call=None)
