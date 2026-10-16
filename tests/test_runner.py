"""runner 层测试（离线）：状态映射、trace 落盘、指标聚合。

做法：把 build_llm 换成本地脚本模型、把 build_toolset 换成一个只含计算器的假工具集，
从而在「没有 API key、没有 rag-mcp-server」的环境里也能验证整条 runner 链路。
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import load_config  # noqa: E402
from agent.runner import STATUS_COMPLETED, STATUS_MAX_STEPS, run_task  # noqa: E402
from agent.toolkit import ToolSet  # noqa: E402
from agent.tools_builtin import calculator  # noqa: E402
from tests.fake_llm import ScriptedChatModel, tool_call  # noqa: E402


def patch_env(monkeypatch, tmp_path, responses, max_steps=3):
    """打桩：假模型 + 假工具集 + 假环境变量（不需要真实 key）。"""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-for-test-not-a-real-key")
    monkeypatch.setenv("MODEL", "scripted-model")

    config = load_config()
    config["agent"]["max_steps"] = max_steps
    config["tools"]["retry"]["backoff_base_s"] = 0.01
    config["trace"]["dir"] = str(tmp_path)
    config["trace"]["console"] = "quiet"

    @asynccontextmanager
    async def fake_toolset(cfg, tracer, rag_mode_override=None):
        tracer.event("tools_ready", tools=["calculator"], rag_mode="direct",
                     rag_note="测试替身（不连 MCP）", fallbacks=[])
        yield ToolSet(tools={"calculator": calculator}, rag_mode="direct", tool_names=["calculator"])

    monkeypatch.setattr("agent.runner.build_toolset", fake_toolset)
    monkeypatch.setattr(
        "agent.runner.build_llm",
        lambda settings: ScriptedChatModel(responses=responses),
    )
    return config


def read_events(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_completed_run_writes_trace_and_metrics(monkeypatch, tmp_path):
    config = patch_env(
        monkeypatch, tmp_path,
        [
            tool_call("calculator", {"expression": "6*7"}, "call_1"),
            AIMessage(content="6*7=42。"),
        ],
    )

    result = asyncio.run(run_task("算一下 6*7", config=config))

    assert result.status == STATUS_COMPLETED
    assert result.steps == 2
    assert "42" in result.final_answer
    assert result.rag_mode == "direct"
    assert result.metrics["tool_calls"] == 1
    assert result.metrics["tool_ok"] == 1
    assert result.metrics["tool_success_rate"] == 1.0

    events = read_events(Path(result.trace_path))
    assert events[0]["type"] == "run_start"
    assert events[-1]["type"] == "run_end"
    assert events[-1]["status"] == STATUS_COMPLETED
    assert events[-1]["metrics"]["tool_calls"] == 1
    # thought / action / observation 三类事件都在
    types = {e["type"] for e in events}
    assert {"thought", "action", "observation"} <= types

    # 安全：trace 里不能出现密钥（哪怕是打桩的假 key）
    raw = Path(result.trace_path).read_text(encoding="utf-8")
    assert "sk-dummy-for-test-not-a-real-key" not in raw


def test_max_steps_exceeded_status_and_explanation(monkeypatch, tmp_path):
    looping = [tool_call("calculator", {"expression": "1+1"}, f"call_{i}") for i in range(10)]
    config = patch_env(monkeypatch, tmp_path, looping, max_steps=2)

    result = asyncio.run(run_task("一直算", config=config))

    assert result.status == STATUS_MAX_STEPS
    assert result.steps == 2
    assert "已达最大步数上限 max_steps=2" in result.final_answer
    assert "未完成" in result.final_answer

    events = read_events(Path(result.trace_path))
    assert any(e["type"] == "max_steps_reached" for e in events)
    assert events[-1]["status"] == STATUS_MAX_STEPS


def test_missing_env_raises_readable_error(monkeypatch, tmp_path):
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL"):
        monkeypatch.delenv(name, raising=False)

    config = load_config()
    config["trace"]["dir"] = str(tmp_path)

    try:
        asyncio.run(run_task("任务", config=config))
        raise AssertionError("缺少 .env 配置时应当报错")
    except RuntimeError as exc:
        assert ".env.example" in str(exc)
        assert "OPENAI_API_KEY" in str(exc)
