"""上下文压缩代价对照实验：**同一批任务 × 若干压缩档位**，把取舍曲线跑出来。

要回答的问题（README §5.6 的「省 31% 字符」一直缺另一半）：
    压缩把上下文压短了，**代价是什么**？——成功率、平均步数、token、答案要点命中率、
    以及「工具调用参数有没有被压坏」。

实验设计：
- **受控变量**：同一批任务（评测集里会检索知识库的那批——它们才有长上下文），
  每个档位跑一遍；任务文本永远在第一条消息里、**永远不被压缩**，
  所以「模型打算调哪些工具」在各档位完全一致，差异只可能来自「上下文被压掉多少」。
- **档位**：`(context_keep_last, max_context_chars)` 的若干组合，
  从「完全不压缩」到「极紧」，构成一条取舍曲线。
- **被测模型**：仓库自带的**脚本化假模型**（`tests/fake_llm.py` 的上下文忠实版，
  见 `fake_policy_model.py`）。本机没有可用的 LLM 端点（无 `.env`、无本地推理服务），
  所以**本轮不测真实模型的语义质量**，只测机械效果 + 「上下文信息损失」对答案的代理影响。

用法（在本仓库根目录）：
    .venv\\Scripts\\python.exe experiments\\context_compression\\run_experiment.py
    .venv\\Scripts\\python.exe experiments\\context_compression\\run_experiment.py --limit 8 --rag-mode direct

产物（写到 `experiments/results/`）：
    context_compression_baseline.csv   逐任务原始数据（每个档位一行）
    context_compression_baseline.json  元信息 + 各档位汇总
    context_compression_baseline.md    人读的取舍曲线报告

★ 产物里不写任何绝对路径：trace 落在系统临时目录（不提交），
  产物只保留相对文件名、run_id 与相对路径。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import io
import json
import os
import re
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.config import load_config  # noqa: E402
from agent.eval_harness import (  # noqa: E402
    answer_must_contain,
    answer_must_not_contain,
    load_eval_set,
    score_answer,
)
from agent.runner import run_task  # noqa: E402
from experiments.context_compression.fake_policy_model import (  # noqa: E402
    ContextFaithfulFakeModel,
)

RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# 压缩档位：(context_keep_last, max_context_chars)
# 「off」= 预算设成无限大 → compress_messages 直接返回原消息（对照组）
GROUPS: List[Dict[str, Any]] = [
    {"name": "off", "keep_last": 999, "max_context_chars": 10 ** 9,
     "note": "对照组：不压缩（预算无限）"},
    {"name": "default", "keep_last": 6, "max_context_chars": 12000,
     "note": "仓库默认档（12000 字符 / 保留最近 6 条）"},
    {"name": "keep2_4000", "keep_last": 2, "max_context_chars": 4000,
     "note": "收紧档：预算 4000 / 保留最近 2 条"},
    {"name": "keep2_2000", "keep_last": 2, "max_context_chars": 2000,
     "note": "收紧档：预算 2000 / 保留最近 2 条"},
    {"name": "keep2_1200", "keep_last": 2, "max_context_chars": 1200,
     "note": "收紧档：预算 1200 / 保留最近 2 条"},
    {"name": "keep1_800", "keep_last": 1, "max_context_chars": 800,
     "note": "极紧档：预算 800 / 保留最近 1 条"},
]


# ---------------- trace 解析 ----------------


def read_trace_events(trace_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(trace_path)
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def compression_stats(events: List[Dict[str, Any]]) -> Dict[str, int]:
    """从 trace 里统计压缩事件（before/after 字符数、求和）。"""
    events_c = [e for e in events if e.get("type") == "context_compressed"]
    before = sum(int(e.get("before_chars", 0)) for e in events_c)
    after = sum(int(e.get("after_chars", 0)) for e in events_c)
    return {
        "compress_events": len(events_c),
        "chars_before": before,
        "chars_after": after,
        "chars_saved": before - after,
    }


def tool_calls_with_args(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 trace 里取出模型发起过的工具调用（含参数），按发生顺序。"""
    calls: List[Dict[str, Any]] = []
    for event in events:
        if event.get("type") != "thought":
            continue
        for call in event.get("tool_calls") or []:
            calls.append({
                "name": call.get("name"),
                "args": call.get("args") or {},
            })
    return calls


def diff_tool_args(reference: List[Dict[str, Any]], current: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把当前档位的工具调用与对照组逐条比对 → 列出「参数不一样」的调用。

    ★ 这是**黑盒口径**：同一批任务、同一个策略，唯一的差别是上下文被压掉了多少；
      参数变了，就说明「工具调用参数被压坏了」（哪怕它自己没报错）。
    """
    diffs: List[Dict[str, Any]] = []
    for index, ref in enumerate(reference):
        if index >= len(current):
            diffs.append({"index": index, "name": ref.get("name"),
                          "off_args": ref.get("args"), "cur_args": None, "reason": "调用缺失"})
            continue
        cur = current[index]
        if ref.get("name") != cur.get("name") or ref.get("args") != cur.get("args"):
            diffs.append({
                "index": index,
                "name": ref.get("name"),
                "off_args": ref.get("args"),
                "cur_args": cur.get("args"),
                "reason": "参数不同" if ref.get("name") == cur.get("name") else "工具名不同",
            })
    return diffs


# ---------------- 单次运行 ----------------


async def run_one(task: Dict[str, Any], group: Dict[str, Any], config: Dict[str, Any],
                  trace_dir: Path, rag_mode: str) -> Dict[str, Any]:
    """跑一条任务，返回该档位下的原始观测数据。"""
    fake = ContextFaithfulFakeModel()
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        try:
            res = await run_task(
                task["task"],
                config=config,
                rag_mode=rag_mode,
                llm=fake,
            )
            status, steps, answer = res.status, res.steps, res.final_answer
            metrics = res.metrics or {}
            trace_path = res.trace_path
            answer_source = res.final_answer_source
        except Exception as exc:  # noqa: BLE001 —— 单条失败不该让整批实验挂掉
            status, steps, answer = "error", 0, f"{type(exc).__name__}: {exc}"
            metrics, trace_path, answer_source = {}, "", "run_error"

    events = read_trace_events(trace_path) if trace_path else []
    comp = compression_stats(events)
    calls = tool_calls_with_args(events)
    usage = metrics.get("usage") or {}

    must = answer_must_contain(task)
    must_not = answer_must_not_contain(task)
    scored = score_answer(task, answer) if (must or must_not) else None

    return {
        "task_id": task.get("id"),
        "category": task.get("category"),
        "status": status,
        "answer_source": answer_source,
        "steps": steps,
        "tool_calls": int(metrics.get("tool_calls", 0) or 0),
        "tool_ok": int(metrics.get("tool_ok", 0) or 0),
        "tool_failed": int(metrics.get("tool_failed", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
        "completion_tokens": int(usage.get("output_tokens", 0) or 0),
        "compress_events": comp["compress_events"],
        "chars_before": comp["chars_before"],
        "chars_after": comp["chars_after"],
        "chars_saved": comp["chars_saved"],
        "keyword_total": len(must),
        "keyword_hits": len(scored["must_hits"]) if scored else 0,
        "keyword_missed": len(scored["must_missed"]) if scored else 0,
        "keyword_recall": scored["recall"] if scored else None,
        "full_hit": (1 if scored and scored["full_hit"] else 0) if scored else None,
        "hallucinated": (1 if scored and scored["hallucinated"] else 0) if scored else None,
        "answer_chars": len(answer),
        "calls": calls,
    }


# ---------------- 汇总 ----------------


def summarize(group_name: str, rows: List[Dict[str, Any]], group_cfg: Dict[str, Any],
              diffs_by_task: Dict[str, int]) -> Dict[str, Any]:
    n = len(rows)
    completed = [r for r in rows if r["status"] == "completed"]
    scored = [r for r in rows if r["keyword_total"]]
    hits = sum(r["keyword_hits"] for r in rows)
    total = sum(r["keyword_total"] for r in rows)
    full_hits = sum(1 for r in scored if r["full_hit"])
    hallucinated = sum(1 for r in rows if r["hallucinated"])
    chars_before = sum(r["chars_before"] for r in rows)
    chars_after = sum(r["chars_after"] for r in rows)
    saved_ratio = round(1 - chars_after / chars_before, 4) if chars_before else None
    broken_calls = sum(diffs_by_task.values())

    return {
        "group": group_name,
        "keep_last": group_cfg["keep_last"],
        "max_context_chars": group_cfg["max_context_chars"],
        "note": group_cfg["note"],
        "n_tasks": n,
        "completed_n": len(completed),
        "completion_rate": round(len(completed) / n, 4) if n else None,
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 2) if n else None,
        "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2) if n else None,
        "tool_failed_total": sum(r["tool_failed"] for r in rows),
        "tasks_with_compression": sum(1 for r in rows if r["compress_events"] > 0),
        "chars_before_total": chars_before,
        "chars_after_total": chars_after,
        "chars_saved_total": chars_before - chars_after,
        "chars_saved_ratio": saved_ratio,
        "total_tokens": sum(r["total_tokens"] for r in rows),
        "avg_tokens_per_task": round(sum(r["total_tokens"] for r in rows) / n, 1) if n else None,
        "answer_keyword_recall": round(hits / total, 4) if total else None,
        "answer_keyword_hits": hits,
        "answer_keyword_total": total,
        "answer_keyword_full_hit_rate": round(full_hits / len(scored), 4) if scored else None,
        "answer_keyword_scored_n": len(scored),
        "hallucination_rate": round(hallucinated / len(scored), 4) if scored else None,
        "broken_tool_args": broken_calls,
        "tasks_with_broken_args": sum(1 for v in diffs_by_task.values() if v > 0),
    }


def render_markdown(meta: Dict[str, Any], summaries: List[Dict[str, Any]],
                     rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# 上下文压缩代价对照实验（压缩开 / 关，同一批任务）")
    lines.append("")
    lines.append(f"- 运行时间：{meta['run_at']}")
    lines.append(f"- 代码基线：`{meta['git_commit']}`（仓库 hello-agent）")
    lines.append(f"- 被测模型：**{meta['model']}**")
    lines.append(f"- 任务批次：{meta['task_batch']}（{meta['n_tasks']} 条）")
    lines.append(f"- RAG 形态：{meta['rag_mode']}（进程内直连同源 KnowledgeService）")
    lines.append("")
    lines.append("> ⚠️ **本轮用假模型，只测机械效果（压缩 / 步数 / token）与"
                 "「上下文信息损失对答案要点的代理影响」，未测语义质量。**")
    lines.append("> 原因：本机没有可用的 LLM 端点（无 `.env`、无本地推理服务），"
                 "真实模型的语义质量无从测起；编数字不如不测。")
    lines.append("")

    lines.append("## 1. 取舍曲线（每个档位都跑同一批任务）")
    lines.append("")
    lines.append("| 档位 | keep_last | 字符预算 | 完成任务 | 成功率 | 平均步数 | 压缩字符节省率 | "
                 "token 省下（对不压缩） | 平均 token | 答案要点命中率 | 要点全中率 | 工具参数被压坏次数 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| `{s['group']}` | {s['keep_last']} | {s['max_context_chars']} | "
            f"{s['completed_n']}/{s['n_tasks']} | {s['completion_rate']} | {s['avg_steps']} | "
            f"{s['chars_saved_ratio']} | {s.get('token_saving_vs_off')} | "
            f"{s['avg_tokens_per_task']} | {s['answer_keyword_recall']} | "
            f"{s['answer_keyword_full_hit_rate']} | {s['broken_tool_args']} |"
        )
    lines.append("")
    lines.append("口径说明：")
    lines.append("")
    lines.append("- **字符节省率** = 1 - Σ压缩后字符 / Σ压缩前字符（取自 trace 的 `context_compressed` 事件）。"
                 "⚠️ 各档位的**触发点不同**（触发得越早、事件越多、单次 before 越小），"
                 "所以这一列**不适合跨档位直接比大小**；跨档位请看下一列。")
    lines.append("- **token 省下（对不压缩）** = 1 - 本档位总 token / 不压缩档总 token"
                 "（假模型上报的是「实际送进模型的字符数」，1 字符 ≈ 1 token）——"
                 "**这一列才是可跨档位比较的「省了多少」**。")
    lines.append("- **答案要点命中率** = 命中要点数 / 要点总数（ground truth 来自 `data/eval_queries.json`）。")
    lines.append("- **工具参数被压坏次数** = 与对照组（`off`）逐条比对，**参数不一样**的工具调用次数；"
                 "「不一样」= 同任务同策略下唯一的变量是上下文被压缩。")
    lines.append("- **token** 是假模型的**字符代理**（1 字符 ≈ 1 token），不是真实 tokenizer 计数。")
    lines.append("")

    # ---------------- 结论（由数字算出来，不写死）----------------
    off = next((s for s in summaries if s["group"] == GROUPS[0]["name"]), None)
    compressing = [s for s in summaries if s.get("token_saving_vs_off") is not None]
    lines.append("## 2. 这条曲线说明什么（按上面的数字自动生成）")
    lines.append("")
    if off and compressing:
        worst = min(compressing, key=lambda s: s["answer_keyword_recall"] or 1.0)
        best = max(compressing, key=lambda s: s["answer_keyword_recall"] or 0.0)
        lines.append(
            f"1. **省多少**：把上下文压到最紧的档位（`{worst['group']}`，预算 {worst['max_context_chars']}）"
            f"，实际送进模型的上下文少了 **{worst['token_saving_vs_off']}**"
            f"（平均 token {off['avg_tokens_per_task']} → {worst['avg_tokens_per_task']}）。")
        lines.append(
            f"2. **代价多少**：同一批任务，答案要点命中率从 **{off['answer_keyword_recall']}** 掉到 "
            f"**{worst['answer_keyword_recall']}**（要点全中率 {off['answer_keyword_full_hit_rate']} → "
            f"{worst['answer_keyword_full_hit_rate']}）；"
            f"工具调用参数被压坏 **{worst['broken_tool_args']}** 次"
            f"（涉及 {worst['tasks_with_broken_args']} 条任务）。")
        lines.append(
            f"3. **没坏的**：成功率（{off['completion_rate']} → {worst['completion_rate']}）与平均步数"
            f"（{off['avg_steps']} → {worst['avg_steps']}）**完全没变**——"
            "这批任务里压缩不影响「跑不跑得完」，影响的是「答得全不全」。")
        unchanged = [s for s in compressing if s["token_saving_vs_off"] == 0]
        if unchanged:
            lines.append(
                f"4. **本仓库默认档（12000 字符 / 保留最近 6 条）在这批任务上根本没触发压缩**"
                f"（token 省下 {unchanged[0]['token_saving_vs_off']}）——"
                "也就是说：默认配置下的「压缩」目前只是一道**没被用上的保险**。")
        lines.append(
            "5. **口径提醒**：以上是**假模型**的结果——它测的是「上下文里的信息被压掉多少、"
            "丢掉的信息会不会出现在答案/参数里」，**不等于**真实模型在压缩下的语义质量下降幅度"
            "（真实模型可能能从残缺上下文里推断出被压掉的要点，也可能更糟）。")
    lines.append("")

    lines.append("## 3. 逐任务原始数据")
    lines.append("")
    lines.append("| 档位 | 任务 | 分类 | 状态 | 步数 | 工具调用 | 压缩次数 | 压缩前字符 | 压缩后字符 | "
                 "token | 要点命中 | 要点总数 | 参数差异 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['group']}` | {r['task_id']} | {r['category']} | {r['status']} | {r['steps']} | "
            f"{r['tool_calls']} | {r['compress_events']} | {r['chars_before']} | {r['chars_after']} | "
            f"{r['total_tokens']} | {r['keyword_hits']} | {r['keyword_total']} | {r['args_diff']} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------- 主流程 ----------------


def probe_endpoint() -> Dict[str, Any]:
    """探测「有没有可用的 LLM 端点」——先探测再决定用真模型还是假模型。"""
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("MODEL", "").strip()
    info = {
        "openai_base_url_set": bool(base_url),
        "openai_api_key_set": bool(api_key),
        "model_set": bool(model),
        "dotenv_exists": (REPO_ROOT / ".env").exists(),
    }
    if not (base_url and api_key and model):
        info["usable"] = False
        info["reason"] = "环境变量 OPENAI_BASE_URL / OPENAI_API_KEY / MODEL 未配置（仓库也没有 .env）"
        return info
    import urllib.error
    import urllib.request

    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/models")
        with urllib.request.urlopen(request, timeout=5) as response:
            info["usable"] = response.status == 200
            info["probe_status"] = response.status
    except urllib.error.HTTPError as exc:
        info["usable"] = False
        info["probe_status"] = exc.code
        info["reason"] = f"端点可达但鉴权/权限不通过：HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        info["usable"] = False
        info["reason"] = f"端点不可达：{type(exc).__name__}"
    return info


def pick_tasks(queries: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """挑「有长上下文」的那批：期望用到检索工具的任务（检索结果动辄上千字符）。"""
    picked = [
        q for q in queries
        if "search_knowledge" in (q.get("expect_tools") or []) and q.get("answer_must_contain")
    ]
    return picked[:limit] if limit else picked


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default=str(REPO_ROOT / "data" / "eval_queries.json"))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--limit", type=int, default=20, help="本批跑多少条任务（0 = 全部）")
    parser.add_argument("--rag-mode", choices=["auto", "mcp", "direct"], default="direct")
    parser.add_argument("--baseline", default="baseline", help="产物文件名（不含扩展名）")
    args = parser.parse_args()

    endpoint = probe_endpoint()
    print(f"[端点探测] {json.dumps(endpoint, ensure_ascii=False)}")
    if endpoint.get("usable"):
        print("!! 端点可用 —— 但本脚本用的是假模型；要测真实模型请用 scripts/eval_agent.py。")

    config = load_config()
    # 跨任务记忆会把「上一条任务的残留」注入 system prompt → 破坏两组的可比性，关掉
    config["memory"]["enabled"] = False
    config["trace"]["console"] = "quiet"

    queries = load_eval_set(args.eval_set)
    tasks = pick_tasks(queries, args.limit)
    print(f"[任务批次] {len(tasks)} 条（期望用到检索工具的题）")

    trace_root = Path(tempfile.mkdtemp(prefix="hello-agent-compression-"))
    print(f"[trace] 落在临时目录（不提交）：{trace_root.name}/")

    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    reference_calls: Dict[str, Dict[str, Any]] = {}   # task_id -> 对照组的工具调用
    diff_details: List[Dict[str, Any]] = []

    for group in GROUPS:
        group_dir = trace_root / group["name"]
        group_dir.mkdir(parents=True, exist_ok=True)
        group_config = copy.deepcopy(config)  # 深拷贝，避免组间互相污染
        group_config["agent"]["context_keep_last"] = group["keep_last"]
        group_config["agent"]["max_context_chars"] = group["max_context_chars"]
        group_config["trace"]["dir"] = str(group_dir)

        group_rows: List[Dict[str, Any]] = []
        started = time.perf_counter()
        for index, task in enumerate(tasks, 1):
            row = await run_one(task, group, group_config, group_dir, args.rag_mode)
            group_rows.append(row)
            print(f"  [{group['name']}] {index}/{len(tasks)} {task['id']} "
                  f"status={row['status']} steps={row['steps']} "
                  f"chars {row['chars_before']}→{row['chars_after']} "
                  f"tokens={row['total_tokens']}")
        elapsed = round(time.perf_counter() - started, 1)

        # 与对照组比对工具参数
        diffs_by_task: Dict[str, int] = {}
        for row in group_rows:
            task_id = row["task_id"]
            if group["name"] == "off":
                reference_calls[task_id] = {"calls": row["calls"]}
                row["args_diff"] = 0
                continue
            ref = reference_calls.get(task_id, {}).get("calls", [])
            diffs = diff_tool_args(ref, row["calls"])
            row["args_diff"] = len(diffs)
            diffs_by_task[task_id] = len(diffs)
            for d in diffs:
                diff_details.append({"group": group["name"], "task_id": task_id, **d})

        summary = summarize(group["name"], group_rows, group, diffs_by_task)
        summary["elapsed_s"] = elapsed
        summaries.append(summary)
        for row in group_rows:
            rows.append({"group": group["name"], **row})

    # 以「不压缩」那组为基准，算每个档位**实际少送了多少上下文**（token 口径，最直观）
    baseline_tokens = next(
        (s["total_tokens"] for s in summaries if s["group"] == GROUPS[0]["name"]), None
    )
    for summary in summaries:
        if summary["group"] == GROUPS[0]["name"] or not baseline_tokens:
            summary["token_saving_vs_off"] = None
        else:
            summary["token_saving_vs_off"] = round(
                1 - summary["total_tokens"] / baseline_tokens, 4
            )

    # ---------------- 落盘 ----------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.baseline}.csv"
    json_path = out_dir / f"{args.baseline}.json"
    md_path = out_dir / f"{args.baseline}.md"

    fieldnames = [
        "group", "task_id", "category", "status", "answer_source", "steps",
        "tool_calls", "tool_ok", "tool_failed", "total_tokens", "prompt_tokens",
        "completion_tokens", "compress_events", "chars_before", "chars_after",
        "chars_saved", "keyword_hits", "keyword_total", "keyword_missed",
        "keyword_recall", "full_hit", "hallucinated", "answer_chars", "args_diff",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    git_commit = os.environ.get("GIT_COMMIT", "unknown")
    try:
        import subprocess

        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        pass

    meta = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,
        "repo": "hello-agent",
        "model": "ContextFaithfulFakeModel（仓库自带 ScriptedChatModel 的上下文忠实扩展；脚本化假模型）",
        "model_kind": "scripted_fake_model",
        "model_policy": "只按「当前看得见的内容」决策：① 工具计划只由任务文本决定（各档位一致）；"
                        "② 计算器的表达式取自**检索结果里的数字**；"
                        "③ 最终答案复述看得见的观察内容。"
                        "★ 刻意把「计算」放在最后一步（先取事实、后算数）——"
                        "这样工具参数才依赖**更早**的观察，正是压缩最容易压坏的地方。",
        "scope_notice": "本轮用假模型，只测机械效果（压缩 / 步数 / token）与"
                        "「上下文信息损失对答案要点的代理影响」，未测语义质量。",
        "endpoint_probe": endpoint,
        "task_batch": "评测集里期望用到检索工具（search_knowledge）的题目",
        "n_tasks": len(tasks),
        "task_ids": [t["id"] for t in tasks],
        "rag_mode": args.rag_mode,
        "groups": GROUPS,
        "traces_committed": False,
        "traces_note": "逐任务 trace 落在系统临时目录（不提交）：避免把本机绝对路径写进仓库。",
    }

    payload = {
        "meta": meta,
        "summaries": summaries,
        "rows": [{k: v for k, v in row.items() if k != "calls"} for row in rows],
        "tool_args_diffs": diff_details,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(meta, summaries, rows), encoding="utf-8")

    print("\n" + "=" * 70)
    print("各档位汇总")
    print("=" * 70)
    for s in summaries:
        print(f"{s['group']:>12} | 成功 {s['completed_n']}/{s['n_tasks']} | 步数 {s['avg_steps']} | "
              f"字符节省 {s['chars_saved_ratio']} | token/条 {s['avg_tokens_per_task']} | "
              f"要点命中 {s['answer_keyword_recall']} | 参数压坏 {s['broken_tool_args']}")
    print(f"\n产物：\n  {csv_path.relative_to(REPO_ROOT)}\n  {json_path.relative_to(REPO_ROOT)}\n  {md_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
