"""用量与成本汇总导出：扫 traces/*.jsonl，重算每次运行的 token 与估算成本，并给出累计。

用法：
    .venv\\Scripts\\python.exe scripts\\usage_report.py                     # 扫 config.yaml 的 trace 目录
    .venv\\Scripts\\python.exe scripts\\usage_report.py --trace-dir traces   # 指定目录
    .venv\\Scripts\\python.exe scripts\\usage_report.py --out eval_results   # 顺便导出 json + markdown

为什么是「扫 trace」而不是「记一份账本」：
trace 里已经有 `run_start.llm`（模型名）与每条 `llm_call.usage`，所以成本**任何时刻都能重算**——
换单价表、复查某次为什么贵、把某次运行摘出来单独算，都只依赖已经落盘的 JSONL，
不会出现「账本和 trace 对不上」的问题。累计值 = 目录里所有 run 的和（想只看某段时间就换目录/过滤文件名）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.config import load_config  # noqa: E402
from agent.usage import PRICING_UNIT, aggregate_usage, load_pricing, scan_trace_dir  # noqa: E402


def render_markdown(rows, total) -> str:
    lines = ["# hello-agent 用量与成本汇总", ""]
    lines.append(f"- 运行次数：**{total['runs']}**（其中 {total['runs_with_pricing']} 次命中单价表）")
    lines.append(f"- 累计 token：**{total['total_tokens']}**"
                 f"（in {total['total_prompt_tokens']} / out {total['total_completion_tokens']}）")
    lines.append(f"- 平均每条任务：**{total['avg_tokens_per_run']} tokens**"
                 f" / **${total['avg_cost_per_run_usd']}**")
    lines.append(f"- 累计估算成本：**${total['total_cost_usd']}**（{PRICING_UNIT}）")
    if total.get("note"):
        lines.append(f"- 注意：{total['note']}")
    lines.append("")
    lines.append("| # | run_id | status | steps | in | out | total | cost_usd | model | task |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        cost = "—" if r.get("estimated_cost_usd") is None else f"{r['estimated_cost_usd']:.6f}"
        task = (r.get("task") or "").replace("|", "/")[:30]
        lines.append(
            f"| {i} | {r.get('run_id')} | {r.get('status')} | {r.get('steps')} | {r.get('prompt_tokens')} "
            f"| {r.get('completion_tokens')} | {r.get('total_tokens')} | {cost} | {r.get('model')} | {task} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", default=None, help="trace 目录（默认取 config.yaml 的 trace.dir）")
    parser.add_argument("--out", default=None, help="导出目录（给定时写 usage_<时间戳>.json/.md）")
    args = parser.parse_args()

    config = load_config()
    trace_dir = Path(args.trace_dir) if args.trace_dir else Path(str(config["trace"]["dir"]))
    pricing = load_pricing(config)

    rows = scan_trace_dir(trace_dir, pricing)
    total = aggregate_usage(rows)

    print(f"trace 目录：{trace_dir}")
    print(f"单价表：{json.dumps({k: v for k, v in pricing.items()}, ensure_ascii=False)}（{PRICING_UNIT}）")
    print("-" * 100)
    for index, row in enumerate(rows, 1):
        cost = "—" if row.get("estimated_cost_usd") is None else f"${row['estimated_cost_usd']:.6f}"
        print(
            f"[{index:>2}] run={row.get('run_id')} model={row.get('model')} status={row.get('status')} "
            f"steps={row.get('steps')} tokens in/out={row.get('prompt_tokens')}/{row.get('completion_tokens')} "
            f"合计={row.get('total_tokens')} 成本={cost}"
        )
    print("-" * 100)
    print(f"累计：{total['runs']} 次运行，{total['total_tokens']} tokens，"
          f"估算成本 ${total['total_cost_usd']}（{total['runs_with_pricing']} 次命中单价表）")
    if total.get("note"):
        print(f"注意：{total['note']}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (out / f"usage_{stamp}.json").write_text(
            json.dumps({"summary": total, "runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / f"usage_{stamp}.md").write_text(render_markdown(rows, total), encoding="utf-8")
        print(f"已导出：{out / f'usage_{stamp}.json'} / {out / f'usage_{stamp}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
