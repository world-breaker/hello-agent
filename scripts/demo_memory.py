"""跨任务记忆演示（离线，不需要 API key / 不需要模型）。

模拟两个任务：
  任务 A：用户说「记住：我的默认语言是中文」→ 写入长期记忆；
  任务 B：用户问「我的默认语言是什么」→ 任务启动时按关键词召回并注入 system prompt。

以及两个边界：
  - 新进程/新实例仍能读到（JSONL 持久化）；
  - 与记忆无关的问题不会注入任何内容（不污染上下文）。

运行：
    .venv\\Scripts\\python.exe scripts\\demo_memory.py
    .venv\\Scripts\\python.exe scripts\\demo_memory.py --reset   # 先清空演示记忆
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.config import load_config  # noqa: E402
from agent.memory import MemoryStore, build_memory_context  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=str(REPO_ROOT / "data" / "demo_memory.jsonl"),
        help="演示用记忆文件（与正式 agent_memory.jsonl 隔离）",
    )
    parser.add_argument("--reset", action="store_true", help="先清空演示记忆")
    args = parser.parse_args()

    path = Path(args.path)
    if args.reset and path.exists():
        path.unlink()
        print(f"[reset] 已删除 {path}")

    config = load_config()
    config["memory"]["path"] = str(path)
    config["memory"]["enabled"] = True

    print("=" * 90)
    print("【任务 A】用户：记住，我的默认语言是中文；项目路径在 D:/programmer/LLM")
    print("=" * 90)
    store = MemoryStore(config)
    store.remember("用户的默认语言是中文，回答请用中文", key="lang", tags=["偏好", "语言"])
    store.remember("用户的项目路径是 D:/programmer/LLM", key="project_path", tags=["项目"])
    print(f"已写入 2 条记忆 → {path}")
    print(f"当前记忆条数：{len(store)}")

    print()
    print("=" * 90)
    print("【任务 B】用户：我的默认语言是什么？帮我按项目路径找一下代码")
    print("=" * 90)
    task = "我的默认语言是什么？帮我按项目路径找一下代码"
    # 模拟「新实例/新进程」：重新从 JSONL 加载
    fresh_store = MemoryStore(config)
    context, keys = build_memory_context(fresh_store, task, top_k=3, max_chars=800)
    print("自动召回并注入 system prompt 的内容：")
    print(context or "(无)")
    print(f"\n命中的记忆 key：{keys}")

    print()
    print("=" * 90)
    print("【边界】用户：今天武汉天气怎么样？（与记忆无关键词重叠 → 不注入）")
    print("=" * 90)
    unrelated, unrelated_keys = build_memory_context(
        fresh_store, "今天武汉天气怎么样", top_k=3
    )
    print(f"注入内容：{unrelated or '(无)'}")
    print(f"命中 key：{unrelated_keys}")

    print()
    print(f"[OK] 跨任务记忆演示完成；持久化文件：{path}")
    print("     换一个进程再运行本脚本，任务 B 仍能召回这两条记忆（JSONL 落盘）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
