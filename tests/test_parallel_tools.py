"""工具调用并发化测试（离线、无真实 LLM）。

验证三件事：
1. 同一轮的多个 tool_calls 会并发执行（总耗时 ≈ 最慢的一个，而不是累加）；
2. ToolMessage 的顺序与 tool_calls 顺序严格一致（MCP/协议要求一一对应）；
3. `parallel_tools: false` 时退回串行（行为可回退）。
"""

import asyncio
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import load_config  # noqa: E402
from agent.graph import build_agent_graph  # noqa: E402
from agent.resilience import ToolExecutor  # noqa: E402
from agent.trace import Tracer  # noqa: E402
from tests.fake_llm import ScriptedChatModel  # noqa: E402

DELAY_S = 0.2
TOOL_NAMES = ["slow_a", "slow_b", "slow_c", "slow_d"]


def make_slow_tool(name: str, result: str) -> StructuredTool:
    async def _arun(value: str = "") -> str:
        await asyncio.sleep(DELAY_S)
        return result

    return StructuredTool.from_function(
        coroutine=_arun, name=name, description=f"测试用慢工具 {name}"
    )


def make_tools() -> dict:
    return {
        name: make_slow_tool(name, name.split("_")[-1].upper())
        for name in TOOL_NAMES
    }


def multi_call_message() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {"value": name}, "id": f"call_{i}"}
            for i, name in enumerate(TOOL_NAMES)
        ],
    )


def build_env(tmp_path: Path, parallel: bool):
    config = load_config()
    config["agent"]["max_steps"] = 3
    config["agent"]["parallel_tools"] = parallel
    config["agent"]["max_parallel_tools"] = 4
    config["tools"]["timeout_s"] = 5
    config["tools"]["retry"]["max_attempts"] = 1
    config["memory"]["enabled"] = False  # 本测试不涉及记忆，避免写文件

    tracer = Tracer(tmp_path, console="quiet", run_id=f"par-{parallel}")
    executor = ToolExecutor(make_tools(), config, tracer)
    llm = ScriptedChatModel(
        responses=[multi_call_message(), AIMessage(content="四个工具都完成了")]
    )
    graph = build_agent_graph(llm, executor, tracer, config)
    return graph, tracer


def run_graph(graph) -> dict:
    return asyncio.run(
        graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "并发跑四个工具"}],
                "steps": 0,
                "status": "new",
            },
            config={"recursion_limit": 25},
        )
    )


def test_parallel_tools_run_concurrently(tmp_path):
    graph, tracer = build_env(tmp_path, parallel=True)

    started = time.perf_counter()
    state = run_graph(graph)
    parallel_elapsed = time.perf_counter() - started

    assert state["status"] == "completed"
    observations = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(observations) == len(TOOL_NAMES)
    # 顺序必须与 tool_calls 一致
    assert [m.content for m in observations] == ["A", "B", "C", "D"]

    event_types = [e["type"] for e in tracer.events]
    assert "parallel_tools" in event_types
    assert "parallel_tools_done" in event_types

    # 4 个 0.2s 的工具：并发应 ≈0.2s，串行才是 ≈0.8s
    assert parallel_elapsed < 0.5, f"并发执行耗时异常：{parallel_elapsed:.2f}s"


def test_sequential_mode_is_slower(tmp_path):
    graph, tracer = build_env(tmp_path, parallel=False)

    started = time.perf_counter()
    state = run_graph(graph)
    sequential_elapsed = time.perf_counter() - started

    assert state["status"] == "completed"
    assert "parallel_tools" not in [e["type"] for e in tracer.events]
    assert sequential_elapsed > 0.65, f"串行执行耗时异常：{sequential_elapsed:.2f}s"
