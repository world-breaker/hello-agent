"""可靠性层测试：重试 / 分类 / 降级 / 熔断 / 超时 / 未知工具。

这些测试全部离线运行（不联网、不花钱），是「无 API key 也能验证核心逻辑」的兜底。
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from langchain_core.tools import StructuredTool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import load_config  # noqa: E402
from agent.resilience import ToolExecutor, classify_error, extract_tool_text  # noqa: E402
from agent.trace import Tracer  # noqa: E402


def make_tool(name: str, fail_times: int = 0, exc_factory=None, delay_s: float = 0.0, result: str = '{"ok": true}'):
    """构造一个「前 fail_times 次失败，之后成功」的工具。"""
    state = {"calls": 0}

    async def _arun(**kwargs):
        state["calls"] += 1
        if delay_s:
            await asyncio.sleep(delay_s)
        if state["calls"] <= fail_times:
            raise exc_factory() if exc_factory else RuntimeError("boom")
        return result

    tool = StructuredTool.from_function(
        coroutine=_arun, name=name, description=f"测试工具 {name}",
    )
    return tool, state


def make_executor(tmp_path, tools, **overrides):
    config = load_config()
    config["tools"]["retry"].update(overrides.pop("retry", {}))
    config["tools"].update(overrides)
    tracer = Tracer(tmp_path, console="quiet", preview_chars=100)
    executor = ToolExecutor({t.name: t for t in tools}, config, tracer)
    return executor, tracer


# ---------------- 失败分类 ----------------


def test_classify_error():
    retryable_patterns = ["timeout", "connection", "429"]
    assert classify_error(asyncio.TimeoutError(), retryable_patterns) == "retryable"
    assert classify_error(TimeoutError("timed out"), retryable_patterns) == "retryable"
    assert classify_error(ConnectionError("connection reset"), retryable_patterns) == "retryable"
    assert classify_error(RuntimeError("HTTP 429 rate limit"), retryable_patterns) == "retryable"
    # 参数类错误不该重试
    assert classify_error(TypeError("unexpected keyword argument 'x'"), retryable_patterns) == "permanent"
    assert classify_error(ValueError("bad expression"), retryable_patterns) == "permanent"
    # 未知异常保守按可重试处理
    assert classify_error(RuntimeError("something odd"), retryable_patterns) == "retryable"


# ---------------- 重试 ----------------


def test_retryable_failure_then_success(tmp_path):
    tool, state = make_tool("flaky", fail_times=1, exc_factory=lambda: TimeoutError("timed out"))
    executor, tracer = make_executor(tmp_path, [tool])

    outcome = asyncio.run(executor.execute("flaky", {}))

    assert outcome.ok is True
    assert outcome.attempts == 2                     # 第一次失败 + 第二次成功
    assert state["calls"] == 2
    assert executor.stats["retries"] == 1
    event_types = [e["type"] for e in tracer.events]
    assert event_types.count("action") == 2
    assert "retry" in event_types


def test_permanent_failure_is_not_retried(tmp_path):
    tool, state = make_tool("bad_args", fail_times=1, exc_factory=lambda: TypeError("unexpected argument 'x'"))
    executor, _ = make_executor(tmp_path, [tool])

    outcome = asyncio.run(executor.execute("bad_args", {}))

    assert outcome.ok is False
    assert outcome.error_class == "permanent"
    assert outcome.attempts == 1        # 参数错重试没意义
    assert state["calls"] == 1
    assert "参数或用法有误" in outcome.to_observation()


def test_retry_exhausted_returns_actionable_observation(tmp_path):
    tool, state = make_tool("down", fail_times=99, exc_factory=lambda: ConnectionError("connection refused"))
    executor, _ = make_executor(tmp_path, [tool], retry={"max_attempts": 3, "backoff_base_s": 0.01})

    outcome = asyncio.run(executor.execute("down", {}))

    assert outcome.ok is False
    assert outcome.attempts == 3
    assert state["calls"] == 3
    observation = outcome.to_observation()
    assert "已重试 3 次仍失败" in observation
    assert "不要再重试同样的调用" in observation


# ---------------- 降级 ----------------


def test_fallback_is_used_after_retries_exhausted(tmp_path):
    tool, _ = make_tool("search_knowledge", fail_times=99, exc_factory=lambda: TimeoutError("timed out"))
    config = load_config()
    config["tools"]["retry"]["backoff_base_s"] = 0.01
    tracer = Tracer(tmp_path, console="quiet")

    fallback_calls = {"n": 0}

    def _fallback(args):
        fallback_calls["n"] += 1
        return json.dumps({"matched": 1, "results": [{"source": "fallback.txt"}]}, ensure_ascii=False)

    executor = ToolExecutor({"search_knowledge": tool}, config, tracer, fallbacks={"search_knowledge": _fallback})
    outcome = asyncio.run(executor.execute("search_knowledge", {"query": "x"}))

    assert fallback_calls["n"] == 1
    assert outcome.ok is True
    assert outcome.degraded is True
    assert outcome.degraded_from == "MCP stdio 通道"
    assert "降级结果" in outcome.to_observation()
    assert executor.stats["degraded"] == 1
    assert "degrade" in [e["type"] for e in tracer.events]


# ---------------- 熔断 ----------------


def test_circuit_breaker_opens_after_consecutive_failures(tmp_path):
    tool, state = make_tool("broken", fail_times=99, exc_factory=lambda: ValueError("always bad"))
    executor, tracer = make_executor(tmp_path, [tool], circuit_breaker={"failure_threshold": 2})

    first = asyncio.run(executor.execute("broken", {}))
    second = asyncio.run(executor.execute("broken", {}))
    calls_before = state["calls"]
    third = asyncio.run(executor.execute("broken", {}))

    assert first.ok is False and second.ok is False
    assert third.error_class == "circuit_open"
    assert state["calls"] == calls_before          # 熔断后不再真正调用工具
    assert "已被临时禁用" in third.to_observation()
    assert executor.stats["circuit_opens"] == 1
    assert "circuit_open" in [e["type"] for e in tracer.events]


def test_circuit_breaker_resets_after_success(tmp_path):
    tool, _ = make_tool("flaky2", fail_times=1, exc_factory=lambda: ValueError("bad once"))
    executor, _ = make_executor(tmp_path, [tool], circuit_breaker={"failure_threshold": 2})

    asyncio.run(executor.execute("flaky2", {}))    # 失败 1 次（permanent，无重试）
    ok = asyncio.run(executor.execute("flaky2", {}))
    assert ok.ok is True
    assert executor.consecutive_failures["flaky2"] == 0


# ---------------- 超时 / 未知工具 ----------------


def test_timeout_is_enforced(tmp_path):
    tool, _ = make_tool("slow", delay_s=0.5)
    executor, _ = make_executor(tmp_path, [tool], timeout_s=0.05,
                                retry={"max_attempts": 1, "backoff_base_s": 0.01})

    outcome = asyncio.run(executor.execute("slow", {}))

    assert outcome.ok is False
    assert outcome.error_class == "timeout"
    assert "超时" in outcome.to_observation()


def test_unknown_tool_lists_available_tools(tmp_path):
    tool, _ = make_tool("calculator")
    executor, _ = make_executor(tmp_path, [tool])

    outcome = asyncio.run(executor.execute("weather_api", {}))

    assert outcome.ok is False
    assert outcome.error_class == "unknown_tool"
    assert "calculator" in outcome.to_observation()


# ---------------- 返回值归一化（真实 MCP 工具返回的是内容块列表） ----------------


def test_extract_tool_text_unwraps_mcp_content_blocks():
    blocks = [{"type": "text", "text": '{"matched": 2, "results": []}'}]
    assert extract_tool_text(blocks) == '{"matched": 2, "results": []}'
    # 纯字符串原样返回
    assert extract_tool_text("plain") == "plain"
    # dict / 其它结构退回 JSON 序列化
    assert extract_tool_text({"a": 1}) == '{"a": 1}'


def test_observation_uses_extracted_text(tmp_path):
    from langchain_core.tools import StructuredTool

    async def _arun(**kwargs):
        return [{"type": "text", "text": '{"matched": 1}'}]

    tool = StructuredTool.from_function(coroutine=_arun, name="mcp_like", description="模拟 MCP 工具")
    executor, _ = make_executor(tmp_path, [tool])

    outcome = asyncio.run(executor.execute("mcp_like", {}))

    assert outcome.ok is True
    assert outcome.to_observation() == '{"matched": 1}'   # 不带转义引号的干净原文
    assert "\\" not in outcome.to_observation()
