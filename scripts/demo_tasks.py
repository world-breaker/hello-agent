"""跑一组 demo 任务并保留真实 trace（用于 README 的「真实运行日志」）。

用法：
    .venv\\Scripts\\python.exe scripts\\demo_tasks.py                  # 默认 3 条（RAG 形态取 config 的 auto）
    .venv\\Scripts\\python.exe scripts\\demo_tasks.py --rag-mode mcp    # 强制走 MCP 通道
    .venv\\Scripts\\python.exe scripts\\demo_tasks.py --rag-mode direct # 强制进程内直连
    .venv\\Scripts\\python.exe scripts\\demo_tasks.py --only 3          # 只跑第 3 条

设计说明：三条任务分别覆盖「多步工具组合」「检索 + 算术」「知识库外的诚实降级」，
都是设计出来的固定输入，但**回答完全由模型现场生成**，因此日志可复现、结论不可捏造。
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.config import load_config  # noqa: E402
from agent.runner import run_task  # noqa: E402

DEMO_TASKS = [
    "先查知识库：LoRA 的典型配置里 r 和 α（alpha）分别是多少？然后用计算器算一下 α/r 的值，"
    "最后告诉我现在的时间（精确到秒）。",
    "知识库里说 chunk_size 的典型取值范围是多少？如果我按这个范围的上限，对一篇 5000 字符的文档做"
    "固定窗口切分、overlap 取 50，大约会产生多少个 chunk？请给出计算过程。",
    "知识库里有没有关于『武汉天气』的内容？如果没有，请直接说没有，不要编造。",
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-mode", choices=["auto", "mcp", "direct"], default=None)
    parser.add_argument("--only", type=int, default=None, help="只跑第 N 条（从 1 开始）")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    if args.max_steps:
        config["agent"]["max_steps"] = args.max_steps

    tasks = DEMO_TASKS if args.only is None else [DEMO_TASKS[args.only - 1]]
    results = []
    started = time.perf_counter()

    for index, task in enumerate(tasks, 1):
        print("\n" + "=" * 100)
        print(f"【Demo {index}/{len(tasks)}】{task}")
        print("=" * 100)
        result = await run_task(task, config=config, rag_mode=args.rag_mode)
        results.append(result)

    print("\n" + "=" * 100)
    print("DEMO 汇总")
    print("=" * 100)
    for index, result in enumerate(results, 1):
        metrics = result.metrics or {}
        print(
            f"Demo {index}: status={result.status} 步数={result.steps} "
            f"工具调用={metrics.get('tool_calls')} 成功={metrics.get('tool_ok')} 失败={metrics.get('tool_failed')} "
            f"重试={metrics.get('tool_retries')} 降级={metrics.get('tool_degraded')} "
            f"RAG形态={result.rag_mode} 耗时={result.elapsed_ms}ms"
        )
        print(f"        trace: {result.trace_path}")
    print(f"总耗时 {round(time.perf_counter() - started, 2)}s；trace 目录：{config['trace']['dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
