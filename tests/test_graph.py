"""图编排测试（离线，用脚本化假模型）：ReAct 循环 / 最大步数 / 工具失败恢复。

不需要 API key，也不需要 rag-mcp-server：这里只验证「循环控制 + 失败处理」这套骨架。
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import load_config  # noqa: E402
from agent.graph import build_agent_graph  # noqa: E402
from agent.resilience import ToolExecutor  # noqa: E402
from agent.tools_builtin import calculator  # noqa: E402
from agent.trace import Tracer  # noqa: E402
from tests.fake_llm import ScriptedChatModel, tool_call  # noqa: E402


def make_broken_tool(calls: dict):
    from langchain_core.tools import StructuredTool

    async def _arun(**kwargs):
        calls["n"] = calls.get("n", 0) + 1
        raise TimeoutError("上游超时")

    return StructuredTool.from_function(coroutine=_arun, name="broken", description="永远失败的工具")


class BrokenLLM(ScriptedChatModel):
    """总是抛异常的模型：用来验证 LLM 失败时的兜底路径。"""

    async def _agenerate(self, *args, **kwargs):
        raise RuntimeError("模型炸了")


def build_env(tmp_path, responses, max_steps=3, tools=None, llm=None):
    config = load_config()
    config["agent"]["max_steps"] = max_steps
    config["tools"]["retry"]["backoff_base_s"] = 0.01   # 测试里不要真等
    config["tools"]["retry"]["max_attempts"] = 3

    tracer = Tracer(tmp_path, console="quiet", run_id="graphtest")
    tool_map = tools if tools is not None else {"calculator": calculator}
    executor = ToolExecutor(tool_map, config, tracer)
    if llm is None:
        llm = ScriptedChatModel(responses=responses)
    graph = build_agent_graph(llm, executor, tracer, config)
    return graph, executor, tracer


def invoke(graph, task="测试任务"):
    return asyncio.run(
        graph.ainvoke(
            {"messages": [{"role": "user", "content": task}], "steps": 0, "status": "new"},
            config={"recursion_limit": 25},
        )
    )


# ---------------- 正常路径 ----------------


def test_immediate_answer_finishes_in_one_step(tmp_path):
    graph, executor, tracer = build_env(tmp_path, [AIMessage(content="直接回答：不需要工具")], max_steps=3)

    state = invoke(graph)

    assert state["status"] == "completed"
    assert state["steps"] == 1
    assert executor.stats["calls"] == 0
    assert [e["type"] for e in tracer.events].count("thought") == 1


def test_tool_call_then_answer(tmp_path):
    graph, executor, _ = build_env(
        tmp_path,
        [
            tool_call("calculator", {"expression": "(3+5)*2"}, "call_1"),
            AIMessage(content="计算结果是 16。"),
        ],
        max_steps=4,
    )

    state = invoke(graph)

    assert state["status"] == "completed"
    assert state["steps"] == 2
    observations = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(observations) == 1
    assert '"result": 16' in observations[0].content
    assert executor.stats["ok"] == 1


# ---------------- 最大步数上限 ----------------


def test_max_steps_stops_the_loop(tmp_path):
    """模型一直要求调工具时，必须在 max_steps 处被拦住，而不是无限循环或抛异常。"""
    looping = [tool_call("calculator", {"expression": "1+1"}, f"call_{i}") for i in range(10)]
    graph, executor, tracer = build_env(tmp_path, looping, max_steps=3)

    state = invoke(graph)

    assert state["steps"] == 3
    assert state["status"] == "in_progress"        # runner 会把它翻译成 max_steps_exceeded
    # 第 3 轮的 tool_call 没有执行 → 只执行了前两轮
    assert executor.stats["calls"] == 2
    assert len([m for m in state["messages"] if isinstance(m, ToolMessage)]) == 2
    last = state["messages"][-1]
    assert getattr(last, "tool_calls", None)       # 最后一条仍是「想调工具」的消息


# ---------------- 工具失败后的恢复路径 ----------------


def test_tool_failure_is_reported_to_model_then_recovers(tmp_path):
    calls: dict = {}
    tools = {"broken": make_broken_tool(calls), "calculator": calculator}
    graph, executor, tracer = build_env(
        tmp_path,
        [
            tool_call("broken", {}, "call_1"),
            AIMessage(content="工具不可用，我如实说明：拿不到这部分信息。"),
        ],
        max_steps=4,
        tools=tools,
    )

    state = invoke(graph)

    assert state["status"] == "completed"
    observations = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(observations) == 1
    # observation 必须是「可据以行动」的信息，而不是一句 "error"
    assert "已重试 3 次仍失败" in observations[0].content
    assert "不要再重试同样的调用" in observations[0].content
    assert calls["n"] == 3                              # 可重试失败确实重试了 3 次
    assert executor.stats["retries"] == 2               # 第 1、2 次失败后各重试一次
    assert "retry" in [e["type"] for e in tracer.events]


def test_unknown_tool_does_not_break_the_loop(tmp_path):
    graph, executor, _ = build_env(
        tmp_path,
        [
            tool_call("weather_api", {"city": "武汉"}, "call_1"),   # 模型幻觉出一个不存在的工具
            AIMessage(content="没有这个工具，我改用知识库或直接说明。"),
        ],
        max_steps=4,
    )

    state = invoke(graph)

    assert state["status"] == "completed"
    observations = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert "没有名为 `weather_api` 的工具" in observations[0].content
    assert executor.stats["failed"] == 1
    assert executor.stats["calls"] == 1


# ---------------- LLM 失败时的兜底 ----------------


def test_llm_failure_fallback_is_marked_as_system_fallback(tmp_path):
    """★ 模型调用失败时的兜底文案必须与「模型真实回答」区分开。

    兜底文案里写着「……这一步没有产生任何工具调用」，「没有」正好命中评测侧的否认词 →
    不打标记的话，一次都没答出来的失败运行会被算成「诚实拒答」（诚实率假绿）。
    同时：**失败时如实说明「模型调用失败」的行为不能变**。
    """
    graph, executor, tracer = build_env(tmp_path, [], llm=BrokenLLM())

    state = invoke(graph)

    assert state["status"] == "llm_error"
    last = state["messages"][-1]
    assert (getattr(last, "additional_kwargs", None) or {}).get("answer_source") == "system_fallback"
    assert "模型调用失败" in last.content          # 诚实行为保持不变
    assert "没有产生任何工具调用" in last.content  # 兜底文案本身也不改


def test_circuit_breaker_disables_repeatedly_failing_tool(tmp_path):
    """同一工具连续失败到阈值后，后续调用被直接拒绝（不再浪费步数）。"""
    calls: dict = {}
    tools = {"broken": make_broken_tool(calls)}
    graph, executor, tracer = build_env(
        tmp_path,
        [tool_call("broken", {}, f"call_{i}") for i in range(10)],
        max_steps=6,
        tools=tools,
    )

    state = invoke(graph)

    assert state["status"] == "in_progress"           # 6 步用尽
    assert executor.stats["circuit_opens"] == 1
    # 计数口径：每次「执行」失败算 1 次（不管它内部重试了几次），阈值 3 → 前 3 次执行各重试 3 轮 = 9 次真实调用；
    # 第 4、5 次执行时熔断已打开，被直接拒绝，没有再打到工具
    assert calls["n"] == 9
    observations = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    assert any("已被临时禁用" in text for text in observations)
