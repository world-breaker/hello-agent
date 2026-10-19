"""断点续跑（LangGraph checkpointer）的离线单测：核心是**客观**证据，不是「看起来能跑」。

怎么构造「中途中断」：
- 模型换成脚本化假模型（步数、工具调用全确定）；
- 一个工具在第 1 次调用时抛 `BaseException`（`ToolExecutor` 只兜 `Exception`）——
  等价于「执行到一半进程被 Ctrl-C / kill」，异常直接逃出 `run_task`；
- 然后**换一个新的 checkpointer 实例**（模拟新进程）用同一个 thread_id 再跑一次。

断言的三条硬证据：
1. 已完成的第 1 步没有被重放（`calculator` 调用次数 1 → 1，模型调用次数也只增加收尾那一次）；
2. 只补跑了失败的那一步（`crashy` 第 2 次调用成功）；
3. 最终 status=completed，且 trace 里有 `resume_detected` 事件与正确的 `next_nodes`。

两个后端（官方 sqlite / 自研 JSONL）跑**同一套**断言——这同时也是「自研 saver 与官方语义一致」的回归测试。
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.checkpointing import JsonlCheckpointer, list_threads, open_checkpointer  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.runner import STATUS_COMPLETED, run_task  # noqa: E402
from agent.toolkit import ToolSet  # noqa: E402
from agent.tools_builtin import calculator  # noqa: E402
from tests.fake_llm import ScriptedChatModel, tool_call  # noqa: E402

BACKENDS = ["sqlite", "jsonl"]
TASK = "先算 (3+5)*2，再调用 crashy 拿外部数据，最后汇总。"


class SimulatedCrash(BaseException):
    """模拟进程被中断（不是可恢复的工具失败，所以要绕过 except Exception）。"""


class Counter:
    """工具调用计数：**跨两次运行**累计，用它证明「重放步数 = 0」。"""

    def __init__(self):
        self.calculator = 0
        self.crashy = 0


def build_tools(counter: Counter):
    async def _calc(expression: str) -> str:
        counter.calculator += 1
        return calculator.invoke({"expression": expression})

    async def _crashy() -> str:
        counter.crashy += 1
        if counter.crashy == 1:
            raise SimulatedCrash("模拟：工具执行到一半进程被 kill")
        return "外部数据：revenue=42"

    return {
        "calculator": StructuredTool.from_function(coroutine=_calc, name="calculator", description="算术计算器"),
        "crashy": StructuredTool.from_function(coroutine=_crashy, name="crashy", description="第一次调用会崩"),
    }


def patch_env(monkeypatch, tmp_path, scripts, counter, backend="sqlite", max_steps=4):
    """打桩：假模型（每次运行取一份脚本）+ 假工具集 + 临时 checkpoint 路径。"""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-for-test-not-a-real-key")
    monkeypatch.setenv("MODEL", "scripted-model")

    config = load_config()
    config["agent"]["max_steps"] = max_steps
    config["tools"]["retry"]["backoff_base_s"] = 0.01
    config["trace"]["dir"] = str(tmp_path / "traces")
    config["trace"]["console"] = "quiet"
    config["agent"]["checkpoint"]["backend"] = backend
    config["agent"]["checkpoint"]["path"] = str(tmp_path / "checkpoints.sqlite")

    pending = list(scripts)
    models = []

    def fake_build_llm(settings):
        model = ScriptedChatModel(responses=pending.pop(0))
        models.append(model)
        return model

    tools = build_tools(counter)

    @asynccontextmanager
    async def fake_toolset(cfg, tracer, rag_mode_override=None):
        tracer.event("tools_ready", tools=sorted(tools), rag_mode="direct", rag_note="测试替身", fallbacks=[])
        yield ToolSet(tools=tools, rag_mode="direct", tool_names=sorted(tools))

    monkeypatch.setattr("agent.runner.build_llm", fake_build_llm)
    monkeypatch.setattr("agent.runner.build_toolset", fake_toolset)
    return config, models


def crash_scripts():
    """第 1 次运行：算数 → 调会崩的工具（第 2 个工具调用处中断）。"""
    return [
        [tool_call("calculator", {"expression": "(3+5)*2"}, "call_1"),
         tool_call("crashy", {}, "call_2")],
        # 第 2 次运行只需要「收尾」这一步：前面的步骤若被重放，模型调用次数会 > 1
        [AIMessage(content="汇总：计算结果是 16，外部数据 revenue=42。")],
    ]


def read_events(trace_path: str):
    return [json.loads(line) for line in Path(trace_path).read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("backend", BACKENDS)
def test_resume_does_not_replay_completed_steps(monkeypatch, tmp_path, backend):
    counter = Counter()
    config, models = patch_env(monkeypatch, tmp_path, crash_scripts(), counter, backend=backend)

    # ---------- 第 1 次运行：中途被中断 ----------
    with pytest.raises(SimulatedCrash):
        asyncio.run(run_task(TASK, config=config, thread_id="t-resume"))

    assert counter.calculator == 1          # 第 1 步的工具确实执行过
    assert counter.crashy == 1              # 第 2 步的工具崩了
    assert models[0].calls == 2             # 模型被调用 2 次（两个 thought 步骤）

    # ---------- 第 2 次运行：同 thread_id 续跑（新 checkpointer 实例 = 新进程） ----------
    result = asyncio.run(run_task(TASK, config=config, thread_id="t-resume", resume=True))

    assert result.status == STATUS_COMPLETED
    assert result.resumed is True
    assert result.resumed_from_step == 2          # 从「已完成 2 步」之后继续
    assert result.steps == 3                      # 总步数 = 2 + 本次 1 步
    assert result.metrics["steps_this_run"] == 1
    assert counter.calculator == 1                # ★ 硬证据：已完成的第 1 步没有被重放
    assert counter.crashy == 2                    # 只补跑了失败的那一步
    assert models[1].calls == 1                   # ★ 模型只被调用 1 次（收尾），不是把 2 步重算一遍

    events = read_events(result.trace_path)
    resume_events = [e for e in events if e["type"] == "resume_detected"]
    assert len(resume_events) == 1
    assert resume_events[0]["next_nodes"] == ["tools"]     # 上次停在 tools 节点前
    assert resume_events[0]["steps_done"] == 2
    assert events[-1]["type"] == "run_end"
    assert events[-1]["resumed"] is True
    assert events[-1]["steps_this_run"] == 1
    # trace 的步号接着上次：续跑那次运行从 step=3 开始，不是从 1 重新数
    assert min(e["step"] for e in events if e["type"] in ("llm_call", "thought")) == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_finished_thread_starts_a_new_turn(monkeypatch, tmp_path, backend):
    """已跑到 END 的 thread 再喂新任务 = 新一轮对话（消息追加），不会被误判成续跑。"""
    counter = Counter()
    scripts = [
        [AIMessage(content="第一轮回答")],
        [AIMessage(content="第二轮回答")],
    ]
    config, _ = patch_env(monkeypatch, tmp_path, scripts, counter, backend=backend)

    first = asyncio.run(run_task("任务一", config=config, thread_id="t-turn"))
    assert first.status == STATUS_COMPLETED
    assert first.metrics["resumed"] is False

    second = asyncio.run(run_task("任务二", config=config, thread_id="t-turn"))
    assert second.status == STATUS_COMPLETED
    assert second.metrics["resumed"] is False      # 图已到 END → 按新任务开始
    assert second.steps == 2                       # steps 是通道，跨轮累加
    assert second.final_answer == "第二轮回答"

    # 同 thread 的两条任务都留在消息历史里（thread = 一段会话）
    events = read_events(second.trace_path)
    assert all(e["type"] != "resume_detected" for e in events)


def test_checkpoint_disabled_by_default_path(monkeypatch, tmp_path):
    """不给 thread_id 时完全不挂 checkpointer：后端标记 disabled，且不产生任何 checkpoint 文件。"""
    counter = Counter()
    config, _ = patch_env(monkeypatch, tmp_path, [[AIMessage(content="直接回答")]], counter)

    result = asyncio.run(run_task("不给 thread 的任务", config=config))

    assert result.thread_id is None
    assert result.metrics["checkpoint_backend"] == "disabled"
    assert result.metrics["resumed"] is False
    assert not (tmp_path / "checkpoints.sqlite").exists()
    assert not (tmp_path / "checkpoints.jsonl").exists()


def test_explicit_sqlite_backend_raises_when_dependency_missing(monkeypatch, tmp_path):
    """显式要求 sqlite 时不允许静默降级；auto 才允许退到 JSONL。"""
    config = load_config()
    config["agent"]["checkpoint"]["backend"] = "sqlite"
    config["agent"]["checkpoint"]["path"] = str(tmp_path / "cp.sqlite")
    monkeypatch.setattr("agent.checkpointing.sqlite_saver_available", lambda: (False, "模拟缺依赖"))

    async def _open(backend):
        async with open_checkpointer(config, enabled=True) as (saver, back, note):
            return saver, back, note

    with pytest.raises(RuntimeError):
        asyncio.run(_open("sqlite"))

    config["agent"]["checkpoint"]["backend"] = "auto"
    saver, backend, note = asyncio.run(_open("auto"))
    assert backend == "jsonl"
    assert "降级" in note
    assert saver is not None


def test_jsonl_checkpointer_roundtrip_across_instances(monkeypatch, tmp_path):
    """JSONL 后端要真的能跨进程：新实例（新进程）读回旧实例写的 checkpoint。"""
    counter = Counter()
    config, _ = patch_env(monkeypatch, tmp_path, crash_scripts(), counter, backend="jsonl")

    with pytest.raises(SimulatedCrash):
        asyncio.run(run_task(TASK, config=config, thread_id="t-persist"))

    path = tmp_path / "checkpoints.jsonl"
    assert path.exists()
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["version"] == 1
    assert snapshot["backend"] == "jsonl"
    assert len(snapshot["storage"]) >= 2        # 至少：第 1 步超级步 + 第 2 步 agent 超级步

    reloaded = JsonlCheckpointer(path)
    assert reloaded.loaded_checkpoints == len(snapshot["storage"])
    assert asyncio.run(list_threads(reloaded)) == {"t-persist": len(snapshot["storage"])}
    # 读回来的状态能反序列化成真正的 LangChain 消息（不是字符串壳子）
    tuple_ = reloaded.get_tuple({"configurable": {"thread_id": "t-persist"}})
    messages = tuple_.checkpoint["channel_values"]["messages"]
    assert [type(m).__name__ for m in messages][:1] == ["HumanMessage"]
    assert getattr(messages[-1], "tool_calls", None)   # 最后一条是「想调 crashy」的 AIMessage
