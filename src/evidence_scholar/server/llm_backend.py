"""Switchable LLM backend for the API server (C4).

为什么可切换：服务器 8000 端口不一定永远是 vLLM（实测现在 8000 被别的
服务占着，返回 HTML 404，不是 vLLM 的 JSON）。不能假定 vLLM 永远在跑。
所以用环境变量切：
- ES_LLM_MODE=fake（默认）：FakeLLM，不占卡，验证 API 骨架/缓存/存储
- ES_LLM_MODE=vllm：OpenAICompatibleClient，连 vLLM 的 OpenAI 兼容 API

这和 react.py 的 LLMClient Protocol 一致——run_agent 只认 Protocol，不
关心具体实现。fake/vllm 两者都满足 Protocol，agent 代码不用改。
"""

from __future__ import annotations

import os
from typing import Any

from evidence_scholar.agent.llm_client import OpenAICompatibleClient
from evidence_scholar.agent.react import LLMClient, LLMResponse, ToolCall


class _FakeLLM:
    """不占卡的假 LLM：永远返回一个 retrieve tool_call。

    用于 API 骨架开发——验证 /ask 链路、缓存、存储，不依赖 vLLM/GPU。
    真跑时切 ES_LLM_MODE=vllm。
    """

    def __init__(self) -> None:
        self._call_count = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        self._call_count += 1
        # 第 1 跳返回 retrieve tool_call，让 agent 走一跳后 max_steps 截停
        # （max_steps=1 时）。这样 /ask 能返回一个非空 AgentResult 用于
        # 验证链路，不用真 LLM。
        return LLMResponse(
            tool_call=ToolCall(
                id=f"fake_{self._call_count}",
                name="retrieve_hybrid",
                arguments={"query": "fake query for api test"},
            )
        )


def build_llm_backend() -> LLMClient:
    """根据 ES_LLM_MODE 环境变量构建 LLM 后端。

    fake（默认）→ _FakeLLM；vllm → OpenAICompatibleClient。
    vllm 模式的 base_url/model 可用 ES_VLLM_URL / ES_VLLM_MODEL 覆盖。
    """
    mode = os.environ.get("ES_LLM_MODE", "fake").lower()

    if mode == "vllm":
        base_url = os.environ.get("ES_VLLM_URL", "http://127.0.0.1:8000/v1")
        model = os.environ.get("ES_VLLM_MODEL", "Qwen3-8B")
        return OpenAICompatibleClient(base_url=base_url, model=model)

    if mode != "fake":
        raise ValueError(
            f"Unknown ES_LLM_MODE={mode!r}, expected 'fake' or 'vllm'."
        )

    return _FakeLLM()
