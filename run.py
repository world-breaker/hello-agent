"""命令行入口。

用法：
    .venv\\Scripts\\python.exe run.py "先查知识库 LoRA 的 r/alpha 取值，再用计算器算 alpha/r"
    .venv\\Scripts\\python.exe run.py "..." --rag-mode direct     # 不走 MCP，进程内直连检索
    .venv\\Scripts\\python.exe run.py "..." --rag-mode mcp        # 强制 MCP，连不上直接报错
    .venv\\Scripts\\python.exe run.py "..." --max-steps 4 --console verbose

断点续跑（以 thread_id 为键）：
    .venv\\Scripts\\python.exe run.py "..." --thread-id demo-1    # 该 thread 上没跑完就自动续跑
    .venv\\Scripts\\python.exe run.py "..." --thread-id demo-1 --resume   # 显式要求续跑
    .venv\\Scripts\\python.exe run.py --list-threads              # 看看有哪些 thread / 各有多少 checkpoint
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.checkpointing import list_threads, open_checkpointer  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.runner import STATUS_COMPLETED, run_task  # noqa: E402


async def _list_threads(config) -> int:
    """列出 checkpoint 里出现过的 thread_id（不对模型发任何请求）。"""
    async with open_checkpointer(config, enabled=True) as (saver, backend, note):
        if saver is None:
            print("断点续跑未启用（config.yaml → agent.checkpoint.enabled=false）")
            return 1
        counts = await list_threads(saver)
        print(f"后端={backend} 说明={note}")
        if not counts:
            print("（还没有任何 checkpoint）")
            return 0
        for thread_id, count in sorted(counts.items()):
            print(f"  {thread_id}: {count} 个 checkpoint")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="hello-agent：最小 ReAct 工具调用 Agent")
    parser.add_argument("task", nargs="?", help="要交给 Agent 的任务描述")
    parser.add_argument("--rag-mode", choices=["auto", "mcp", "direct"], default=None,
                        help="RAG 工具形态：mcp=经 MCP 客户端接入；direct=进程内直连；auto=先 MCP 失败则直连")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 config.yaml 的最大步数")
    parser.add_argument("--console", choices=["summary", "verbose", "quiet"], default=None)
    parser.add_argument("--config", default=None, help="指定配置文件路径")
    parser.add_argument("--thread-id", default=None,
                        help="断点续跑键：同一 thread_id 重跑会从最后一个完成的步骤继续，不重放已完成的工具调用")
    parser.add_argument("--resume", action="store_true",
                        help="显式要求续跑（该 thread 没有未完成状态时会提示并按新任务开始）")
    parser.add_argument("--list-threads", action="store_true", help="列出 checkpoint 里的 thread_id 后退出")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps is not None:
        config["agent"]["max_steps"] = args.max_steps

    if args.list_threads:
        return asyncio.run(_list_threads(config))

    if not args.task:
        parser.error("要么给一个任务描述，要么用 --list-threads")

    try:
        result = asyncio.run(
            run_task(
                args.task,
                config=config,
                rag_mode=args.rag_mode,
                console=args.console,
                thread_id=args.thread_id,
                resume=args.resume,
            )
        )
    except RuntimeError as exc:  # 通常是没填 .env
        print(f"[配置错误] {exc}")
        return 2

    return 0 if result.status == STATUS_COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())
