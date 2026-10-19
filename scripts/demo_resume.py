"""断点续跑演示（离线、可复现、不花钱）：模拟「跑到一半进程被杀」→ 同 thread_id 重跑。

用法：
    .venv\\Scripts\\python.exe scripts\\demo_resume.py                      # 默认 sqlite 后端
    .venv\\Scripts\\python.exe scripts\\demo_resume.py --backend jsonl       # 走 MemorySaver + JSONL 后端

为什么用脚本化假模型 + 会崩溃一次的工具：
「中断」这件事必须**确定性地**发生，否则证据不可复现（真实 Ctrl-C 的时机、真实模型返回的步数都不受控）。
所以这里把模型换成按脚本吐消息的假模型、把一个工具换成「第 1 次调用就抛 BaseException」的替身：
第 1 次运行必然在第 2 步的工具执行处中断，第 2 次运行必然从那个位置接着跑。
BaseException 是刻意的——`ToolExecutor` 的 `except Exception` 不会兜住它，等价于进程被 kill（不是可恢复的工具失败）。

打印的三条硬证据：
1. `calculator` 在两次运行里的调用次数都是 1（第 1 步没有被重放）；
2. 第 2 次运行只补跑了失败的那一步（`crashy` 第 2 次调用成功），模型调用次数 = 1（只有收尾那一步）；
3. 第 2 次运行 status=completed，且 trace 里能看到 `resume_detected` 事件（next_nodes=["tools"]）。

顺带演示用量统计：脚本里给每条假消息写了 `usage_metadata`（**数字是写死的假值**，用来验证统计链路，
不是真实计费），模型名取 `deepseek-chat` 以命中 config.yaml 的单价表，于是你能看到
「tokens in/out + 估算成本」那一行；再跑一次
`scripts/usage_report.py --trace-dir data/demo_resume_traces` 就能看到累计层。
"""

import argparse
import asyncio
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402

from agent.config import load_config  # noqa: E402
from agent.runner import run_task  # noqa: E402
from agent.toolkit import ToolSet  # noqa: E402
from agent.tools_builtin import calculator  # noqa: E402
from tests.fake_llm import ScriptedChatModel, tool_call  # noqa: E402

TASK = "先算 (3+5)*2，再调用 crashy 拿一份外部数据，最后汇总。"

# 假 usage（只为演示统计链路，数字写死、可复现）
FAKE_INPUT_1, FAKE_OUTPUT_1 = 820, 45    # 第 1 步：查/算
FAKE_INPUT_2, FAKE_OUTPUT_2 = 900, 38    # 第 2 步：调用崩溃的工具
FAKE_INPUT_3, FAKE_OUTPUT_3 = 1100, 60   # 续跑后的收尾


def with_usage(message: AIMessage, input_tokens: int, output_tokens: int) -> AIMessage:
    message.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return message


class SimulatedCrash(BaseException):
    """模拟「Ctrl-C / 进程被 kill」：用 BaseException 绕过执行层的 except Exception。"""


CALLS = {"calculator": 0, "crashy": 0}
LLM_INSTANCES: list = []


def make_counted_calculator():
    async def _arun(expression: str) -> str:
        CALLS["calculator"] += 1
        return calculator.invoke({"expression": expression})

    return StructuredTool.from_function(coroutine=_arun, name="calculator", description="算术计算器（带调用计数）")


def make_crashy_tool():
    async def _arun() -> str:
        CALLS["crashy"] += 1
        if CALLS["crashy"] == 1:
            raise SimulatedCrash("模拟：工具执行中进程被中断（Ctrl-C / kill）")
        return "外部数据：revenue=42（第 2 次调用成功）"

    return StructuredTool.from_function(coroutine=_arun, name="crashy", description="第一次调用会崩溃的工具")


def install_fakes(scripts: list):
    """把 runner 的模型与工具集换成演示替身（不改产品代码）。"""
    import agent.runner as runner

    pending = list(scripts)

    def fake_build_llm(settings):
        model = ScriptedChatModel(responses=pending.pop(0))
        LLM_INSTANCES.append(model)
        return model

    @asynccontextmanager
    async def fake_toolset(cfg, tracer, rag_mode_override=None):
        tools = {"calculator": make_counted_calculator(), "crashy": make_crashy_tool()}
        tracer.event("tools_ready", tools=sorted(tools), rag_mode="direct",
                     rag_note="演示替身（不连 MCP，也不调真实模型）", fallbacks=[])
        yield ToolSet(tools=tools, rag_mode="direct", tool_names=sorted(tools))

    runner.build_llm = fake_build_llm
    runner.build_toolset = fake_toolset


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "sqlite", "jsonl"], default="sqlite")
    parser.add_argument("--thread-id", default="demo-resume")
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:1/v1")
    os.environ.setdefault("OPENAI_API_KEY", "sk-demo-not-a-real-key")
    os.environ.setdefault("MODEL", "deepseek-chat")   # 命中单价表，才能演示成本估算

    config = load_config()
    config["trace"]["console"] = "quiet"
    trace_dir = REPO_ROOT / "data" / "demo_resume_traces"
    config["trace"]["dir"] = str(trace_dir)
    config["tools"]["retry"]["backoff_base_s"] = 0.01
    config["agent"]["checkpoint"]["backend"] = args.backend
    config["agent"]["checkpoint"]["path"] = str(REPO_ROOT / "data" / f"demo_resume_{args.backend}.sqlite")

    # 演示要可重复：每次从干净的 checkpoint 与 trace 目录开始
    for stale in (REPO_ROOT / "data").glob(f"demo_resume_{args.backend}.sqlite*"):
        stale.unlink()
    for stale in (REPO_ROOT / "data").glob(f"demo_resume_{args.backend}.jsonl*"):
        stale.unlink()
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    install_fakes(
        [
            # 第 1 次运行：模型先算数、再调会崩溃的工具
            [with_usage(tool_call("calculator", {"expression": "(3+5)*2"}, "call_1"), FAKE_INPUT_1, FAKE_OUTPUT_1),
             with_usage(tool_call("crashy", {}, "call_2"), FAKE_INPUT_2, FAKE_OUTPUT_2)],
            # 第 2 次运行：只剩「收尾」这一步需要模型（前两步不该重放）
            [with_usage(AIMessage(content="汇总：计算结果是 16，外部数据 revenue=42。"), FAKE_INPUT_3, FAKE_OUTPUT_3)],
        ]
    )

    print("=" * 100)
    print(f"第 1 次运行：thread_id={args.thread_id}（模拟跑到一半进程被杀）")
    print("=" * 100)
    try:
        await run_task(TASK, config=config, thread_id=args.thread_id)
        print("!! 预期中的中断没有发生（演示脚本应当中断）")
        return 1
    except SimulatedCrash as exc:
        print(f"\n[中断] {type(exc).__name__}: {exc}（异常逃出 run_task，等价于进程结束）")
    first = dict(CALLS)
    print(f"[第 1 次运行结束] 工具调用计数：{first}；模型被调用 {LLM_INSTANCES[0].calls} 次")

    print()
    print("=" * 100)
    print(f"第 2 次运行：同 thread_id={args.thread_id} + --resume（新进程、从磁盘 checkpoint 续跑）")
    print("=" * 100)
    result = await run_task(TASK, config=config, thread_id=args.thread_id, resume=True)
    second = dict(CALLS)

    print()
    print("=" * 100)
    print("硬证据")
    print("=" * 100)
    print(f"1) calculator 调用次数：第 1 次运行 {first['calculator']} → 第 2 次运行 {second['calculator']}"
          f"（增量 {second['calculator'] - first['calculator']}，即第 1 步没有被重放）")
    print(f"2) crashy 调用次数：第 1 次运行 {first['crashy']} → 第 2 次运行 {second['crashy']}"
          f"（增量 {second['crashy'] - first['crashy']}，只补跑了失败的那一步）")
    print(f"3) 模型调用次数：第 1 次运行 {LLM_INSTANCES[0].calls} 次 → 第 2 次运行 {LLM_INSTANCES[1].calls} 次"
          f"（第 2 次只花在收尾上，没有把前面两步重算一遍）")
    print(f"4) 续跑结果：status={result.status} 总步数={result.steps} 本次执行步数="
          f"{result.metrics.get('steps_this_run')}（从第 {result.resumed_from_step} 步后继续）")
    print(f"5) checkpoint 后端={result.metrics.get('checkpoint_backend')}，trace={result.trace_path}")
    print(f"6) 续跑那次的用量：tokens in={result.usage['prompt_tokens']} out={result.usage['completion_tokens']}"
          f"，估算成本=${result.usage['estimated_cost_usd']}（只算了收尾那一步）")
    print()
    print(f"重放步数 = {second['calculator'] - first['calculator']}（0 表示已完成的步骤没有被重放）")
    print(f"trace 目录：{trace_dir}（累计层看它：scripts\\usage_report.py --trace-dir data\\demo_resume_traces）")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
