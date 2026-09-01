"""Tests for the Langfuse tracing layer (B7).

测两个核心保证（不调真 Langfuse 云端、不占网）：
1. no-op 模式（没配 key）：所有方法安全 no-op，不抛、不崩主流程。
2. enabled 模式（配了 key + fake langfuse）：调用序列正确。

可观测性铁律：trace 挂了绝不能拖垮 agent——这是测试重点。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from evidence_scholar.agent.trace import TraceContext, _truncate


# --- _truncate ---

def test_truncate_short_text_unchanged() -> None:
    assert _truncate("short") == "short"


def test_truncate_long_text_gets_ellipsis() -> None:
    long = "x" * 1000
    out = _truncate(long, limit=100)
    assert out.endswith("…")
    assert len(out) == 101  # 100 + 省略号


def test_truncate_none_returns_empty() -> None:
    assert _truncate(None) == ""


# --- 测试辅助：构造支持 with 语义的 fake ---

def _make_fake_langfuse(*, trace_id: str = "trace-123",
                        auth_fail: bool = False) -> tuple:
    """构造 fake langfuse 模块 + client，支持 start_as_current_observation
    的 with-as 语义（返回 context manager，__enter__ 返回 fake span）。
    fake span 有 start_observation（返回子 span，有 end）+ update + end。
    """
    fake_module = MagicMock()
    fake_client = MagicMock()
    if auth_fail:
        fake_client.auth_check.side_effect = RuntimeError("net down")
    else:
        fake_client.auth_check.return_value = None
    fake_client.get_current_trace_id.return_value = trace_id
    fake_client.get_trace_url.return_value = f"https://cloud.langfuse.com/trace/{trace_id}"

    # fake 根 span：start_as_current_observation 返回 context manager，
    # __enter__ 返回它自己（这样 with-as 拿到的就是根 span 对象）。
    fake_root_span = MagicMock()
    fake_root_span.update = MagicMock()
    fake_root_span.end = MagicMock()
    # start_observation 返回子 span（有 end/update）。
    fake_sub_span = MagicMock()
    fake_sub_span.end = MagicMock()
    fake_root_span.start_observation.return_value = fake_sub_span

    # start_as_current_observation 返回一个 context manager 对象。
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=fake_root_span)
    cm.__exit__ = MagicMock(return_value=None)
    fake_client.start_as_current_observation.return_value = cm

    fake_module.Langfuse.return_value = fake_client
    return fake_module, fake_client, fake_root_span, fake_sub_span


# --- no-op 模式（没配 key）---

def test_noop_when_keys_missing(monkeypatch) -> None:
    """没配 LANGFUSE_* key → TraceContext.enabled=False，所有方法 no-op。"""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    ctx = TraceContext.start("q")
    assert ctx.enabled is False
    assert ctx.trace_url is None
    # 所有方法调一遍都不抛——这是"trace 挂了不崩主流程"的保证。
    ctx.start_step(0, _FakeToolCall("retrieve_hybrid", {"query": "q"}))
    ctx.record_llm(model="m", usage=None, response_preview="resp")
    ctx.record_tool_result("result")
    ctx.end_step()
    ctx.finish(answer="a", stopped_reason="answered", steps=1, evidence_pool_size=2)


# --- enabled 模式（fake langfuse）---

class _FakeToolCall:
    """模拟 react.ToolCall，有 name/arguments 属性。"""
    def __init__(self, name: str, arguments: dict) -> None:
        self.name = name
        self.arguments = arguments


def test_enabled_when_keys_set(monkeypatch) -> None:
    """配了 key + langfuse 初始化成功 → enabled=True，trace_url 有值。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    import sys
    fake_module, fake_client, _, _ = _make_fake_langfuse()
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    ctx = TraceContext.start("my question")
    assert ctx.enabled is True
    assert ctx.trace_url == "https://cloud.langfuse.com/trace/trace-123"


def test_init_failure_falls_back_to_noop(monkeypatch) -> None:
    """langfuse 初始化/auth_check 抛错 → 降级 no-op，不崩。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    import sys
    fake_module, _, _, _ = _make_fake_langfuse(auth_fail=True)
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    ctx = TraceContext.start("q")
    assert ctx.enabled is False  # auth_check 抛了 → 降级 no-op


def test_enabled_mode_calls_spans(monkeypatch) -> None:
    """enabled 模式下 start_step/record_llm/record_tool_result/finish 都调对。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    import sys
    fake_module, fake_client, fake_root_span, fake_sub_span = _make_fake_langfuse()
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    ctx = TraceContext.start("q")
    assert ctx.enabled

    # start_step 应调根 span 的 start_observation 建 step span。
    ctx.start_step(0, _FakeToolCall("retrieve_hybrid", {"query": "test"}))
    assert fake_root_span.start_observation.call_count == 1

    # record_llm/record_tool_result 应调 step span 的 start_observation。
    ctx.record_llm(model="Qwen3", usage={"total_tokens": 100}, response_preview="resp")
    ctx.record_tool_result("doc content", error=None)
    assert fake_sub_span.start_observation.call_count == 2  # llm + tool_result

    ctx.end_step()
    # finish 应调根 span 的 update（写最终输出）+ langfuse.flush。
    # 不再调 root_span.end()——根 span 在 start() 的 with 块退出时
    # 已被 OTel 自动 end，重复 end 会触发警告。
    ctx.finish(answer="yes", stopped_reason="answered", steps=1, evidence_pool_size=2)
    fake_root_span.update.assert_called_once()
    fake_client.flush.assert_called_once()


# --- 截断防爆云端 ---

def test_long_result_truncated_in_trace(monkeypatch) -> None:
    """工具结果超长 → 发云端时被截断（防爆额度）。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    import sys
    fake_module, _, _, fake_sub_span = _make_fake_langfuse()
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    ctx = TraceContext.start("q")
    ctx.start_step(0, _FakeToolCall("retrieve_hybrid", {"query": "q"}))
    long_result = "x" * 5000
    ctx.record_tool_result(long_result)
    # 拿 step span.start_observation 最后一次调用的 output 参数
    last_call = fake_sub_span.start_observation.call_args
    output = last_call.kwargs.get("output")
    assert output is not None
    assert "…" in output["result"]
    assert len(output["result"]) < 700  # 截到 600+省略号
