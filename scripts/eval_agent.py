"""hello-agent 评测 runner：跑一整组 query，聚合指标并落盘报告。

用法：
    .venv\\Scripts\\python.exe scripts\\eval_agent.py                  # 用 config 里的 rag 形态
    .venv\\Scripts\\python.exe scripts\\eval_agent.py --rag-mode direct  # 强制进程内直连
    .venv\\Scripts\\python.exe scripts\\eval_agent.py --eval-set data/eval_queries.json
    .venv\\Scripts\\python.exe scripts\\eval_agent.py --categories multi_tool,retrieval_arithmetic
    .venv\\Scripts\\python.exe scripts\\eval_agent.py --ids q01,q02 --limit 2

前置：
    - 需要 .env 里配好 OpenAI 兼容端点（与 run.py 同套配置）；
    - 评测逻辑见 agent/eval_harness.py（纯函数，可离线单测）。

筛选参数（--ids / --categories / --limit）只影响「跑哪几条」，
不影响指标口径；筛出来的子集的分母会跟着变小，报告里每项都带 n 就是为了看这个。

★ 记忆类用例（category=memory）的两个特殊处理：
    1. **隔离记忆库**：带记忆标注的用例一律跑在**临时记忆库**上，
       绝不往用户真实的 `data/agent_memory.jsonl` 里写测试数据；
    2. **可选的 memory_seed**：用例可以声明「跑之前先往记忆库里放这些（可能过时的）记忆」，
       用来制造「召回错」的陷阱（例如 q57 预置一条旧项目路径）。
"""

import argparse
import asyncio
import copy
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.config import load_config  # noqa: E402
from agent.eval_harness import (  # noqa: E402
    EvalRecord,
    evaluate,
    extract_successful_tool_names,
    extract_tool_names,
    has_memory_ground_truth,
    load_eval_set,
    memory_state_from_store,
    memory_state_from_trace,
    merge_memory_states,
    render_markdown,
)
from agent.runner import run_task  # noqa: E402


def _filter_queries(
    queries: list,
    ids: str = "",
    categories: str = "",
    limit: int = 0,
) -> list:
    """按 id / 分类 / 条数筛评测集（纯函数，可单测）。"""
    if ids:
        wanted = {item.strip() for item in ids.split(",") if item.strip()}
        queries = [q for q in queries if str(q.get("id")) in wanted]
    if categories:
        wanted_cats = {item.strip() for item in categories.split(",") if item.strip()}
        queries = [q for q in queries if str(q.get("category")) in wanted_cats]
    if limit and limit > 0:
        queries = queries[:limit]
    return queries


def _seed_memory_file(path: Path, seed: list) -> None:
    """把用例声明的 `memory_seed` 写成一个记忆库 JSONL（预置「过时记忆」）。"""
    lines = []
    for item in seed or []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        lines.append(json.dumps({
            "key": str(item.get("key") or "seed"),
            "content": content,
            "tags": list(item.get("tags") or []),
            "created_at": "2026-01-01T00:00:00+08:00",
            "run_id": "eval_seed",
            "source": "user_stated",
            "verified": bool(item.get("verified", True)),
            "expires_at": "",
        }, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_memory_env(config: dict, query: dict, workdir: Path) -> tuple[dict, str]:
    """给带记忆标注的用例准备一个**隔离的临时记忆库**。

    返回 (本次运行用的配置, 记忆库路径)。没有记忆标注的用例 → 原配置 + 空路径
    （行为与以前完全一致：不碰用户的真实记忆库）。
    """
    if not has_memory_ground_truth(query):
        return config, ""
    run_config = copy.deepcopy(config)
    mem_path = Path(workdir) / f"memory_{query.get('id', 'case')}.jsonl"
    _seed_memory_file(mem_path, query.get("memory_seed") or [])
    memory_cfg = run_config.setdefault("memory", {})
    memory_cfg["enabled"] = True
    memory_cfg["tools"] = True
    memory_cfg["path"] = str(mem_path)
    return run_config, str(mem_path)


def build_memory_state(trace_path: str, memory_path: str = ""):
    """把一次运行的记忆状态拼起来：trace 证据 + 记忆库快照。"""
    state = merge_memory_states(
        memory_state_from_trace(trace_path),
        memory_state_from_store(memory_path) if memory_path else None,
    )
    return state


def _record_to_dict(record: EvalRecord) -> dict:
    """把 EvalRecord 转成可 JSON 序列化的 dict（记忆状态转成普通 dict）。"""
    data = dict(record.__dict__)
    state = data.get("memory_state")
    data["memory_state"] = state.to_dict() if state is not None else None
    return data


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-mode", choices=["auto", "mcp", "direct"], default=None)
    parser.add_argument("--eval-set", default=str(REPO_ROOT / "data" / "eval_queries.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "eval_results"))
    # ---- 跑哪几条（默认全跑；筛选只影响子集，不改指标口径）----
    parser.add_argument("--ids", default="", help="只跑这些 id（逗号分隔，如 q01,q02）")
    parser.add_argument("--categories", default="", help="只跑这些分类（逗号分隔）")
    parser.add_argument("--limit", type=int, default=0, help="最多跑前 N 条（0 = 不限）")
    args = parser.parse_args()

    config = load_config()
    queries = load_eval_set(args.eval_set)
    queries = _filter_queries(queries, args.ids, args.categories, args.limit)
    print(f"评测集：{len(queries)} 条；RAG 形态：{args.rag_mode or config['tools']['rag']['mode']}")

    records: list[EvalRecord] = []
    started = time.perf_counter()
    workdir = Path(tempfile.mkdtemp(prefix="hello-agent-eval-"))
    try:
        for idx, q in enumerate(queries, 1):
            print(f"\n[{idx}/{len(queries)}] {q['task'][:60]}…")
            # 记忆类用例跑在隔离的临时记忆库上（不污染用户真实记忆）
            run_config, memory_path = prepare_memory_env(config, q, workdir)
            res = await run_task(q["task"], config=run_config, rag_mode=args.rag_mode)
            m = res.metrics or {}
            usage = m.get("usage") or {}
            used = extract_tool_names(res.trace_path)
            used_ok = extract_successful_tool_names(res.trace_path)
            records.append(
                EvalRecord(
                    query=q,
                    status=res.status,
                    steps=res.steps,
                    tool_calls=m.get("tool_calls", 0),
                    tool_ok=m.get("tool_ok", 0),
                    tool_failed=m.get("tool_failed", 0),
                    tool_retries=m.get("tool_retries", 0),
                    tool_degraded=m.get("tool_degraded", 0),
                    circuit_opens=m.get("circuit_opens", 0),
                    tool_success_rate=m.get("tool_success_rate"),
                    used_tools=used,
                    # 「真正执行成功」的工具（observation ok=True）——工具选择准确率用这个口径
                    used_tools_ok=used_ok,
                    final_answer=res.final_answer,
                    # 回答来源：区分「模型真实回答」与「系统兜底文案」，评测侧据此算诚实率
                    final_answer_source=res.final_answer_source,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    estimated_cost_usd=usage.get("estimated_cost_usd"),
                    # 记忆状态：trace 证据 + 记忆库快照（老用例没有 → None → 不进记忆分母）
                    memory_state=build_memory_state(res.trace_path, memory_path),
                )
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    summary = evaluate(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"eval_{stamp}.json").write_text(
        json.dumps(
            {"summary": summary, "records": [_record_to_dict(r) for r in records]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md = render_markdown(summary, records)
    (out / f"eval_{stamp}.md").write_text(md, encoding="utf-8")

    print("\n" + "=" * 60)
    print("评测汇总")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n报告：{out / f'eval_{stamp}.md'}")
    print(f"总耗时 {round(time.perf_counter() - started, 2)}s")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
