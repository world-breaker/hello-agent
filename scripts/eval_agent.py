"""hello-agent 评测 runner：跑一整组 query，聚合指标并落盘报告。

用法：
    .venv\\Scripts\\python.exe scripts\\eval_agent.py                  # 用 config 里的 rag 形态
    .venv\\Scripts\\python.exe scripts\\eval_agent.py --rag-mode direct  # 强制进程内直连
    .venv\\Scripts\\python.exe scripts\\eval_agent.py --eval-set data/eval_queries.json

前置：
    - 需要 .env 里配好 OpenAI 兼容端点（与 run.py 同套配置）；
    - 评测逻辑见 agent/eval_harness.py（纯函数，可离线单测）。
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.config import load_config  # noqa: E402
from agent.eval_harness import (  # noqa: E402
    EvalRecord,
    evaluate,
    extract_tool_names,
    load_eval_set,
    render_markdown,
)
from agent.runner import run_task  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-mode", choices=["auto", "mcp", "direct"], default=None)
    parser.add_argument("--eval-set", default=str(REPO_ROOT / "data" / "eval_queries.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "eval_results"))
    args = parser.parse_args()

    config = load_config()
    queries = load_eval_set(args.eval_set)
    print(f"评测集：{len(queries)} 条；RAG 形态：{args.rag_mode or config['tools']['rag']['mode']}")

    records: list[EvalRecord] = []
    started = time.perf_counter()
    for idx, q in enumerate(queries, 1):
        print(f"\n[{idx}/{len(queries)}] {q['task'][:60]}…")
        res = await run_task(q["task"], config=config, rag_mode=args.rag_mode)
        m = res.metrics or {}
        usage = m.get("usage") or {}
        used = extract_tool_names(res.trace_path)
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
                final_answer=res.final_answer,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                estimated_cost_usd=usage.get("estimated_cost_usd"),
            )
        )

    summary = evaluate(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"eval_{stamp}.json").write_text(
        json.dumps({"summary": summary, "records": [r.__dict__ for r in records]}, ensure_ascii=False, indent=2),
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
