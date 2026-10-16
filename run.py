"""命令行入口。

用法：
    .venv\\Scripts\\python.exe run.py "先查知识库 LoRA 的 r/alpha 取值，再用计算器算 alpha/r"
    .venv\\Scripts\\python.exe run.py "..." --rag-mode direct     # 不走 MCP，进程内直连检索
    .venv\\Scripts\\python.exe run.py "..." --rag-mode mcp        # 强制 MCP，连不上直接报错
    .venv\\Scripts\\python.exe run.py "..." --max-steps 4 --console verbose
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import load_config  # noqa: E402
from agent.runner import STATUS_COMPLETED, run_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="hello-agent：最小 ReAct 工具调用 Agent")
    parser.add_argument("task", help="要交给 Agent 的任务描述")
    parser.add_argument("--rag-mode", choices=["auto", "mcp", "direct"], default=None,
                        help="RAG 工具形态：mcp=经 MCP 客户端接入；direct=进程内直连；auto=先 MCP 失败则直连")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 config.yaml 的最大步数")
    parser.add_argument("--console", choices=["summary", "verbose", "quiet"], default=None)
    parser.add_argument("--config", default=None, help="指定配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps is not None:
        config["agent"]["max_steps"] = args.max_steps

    try:
        result = asyncio.run(run_task(args.task, config=config, rag_mode=args.rag_mode, console=args.console))
    except RuntimeError as exc:  # 通常是没填 .env
        print(f"[配置错误] {exc}")
        return 2

    return 0 if result.status == STATUS_COMPLETED else 1


if __name__ == "__main__":
    sys.exit(main())
