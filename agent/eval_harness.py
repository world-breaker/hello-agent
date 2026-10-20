"""hello-agent 量化评测模块（对应 README「下一步」：把单次运行指标聚合成作品集级评估）。

设计原则：
- **只新增文件，不改动既有编排 / 执行代码**；
- **评测逻辑与模型解耦**：`evaluate(records)` 是纯函数，离线可单测；
  真正调模型的 `run_eval` 在 `scripts/eval_agent.py` 里（不在本模块，避免导入期拉起 LLM 依赖）。
- 工具名从 trace 的 JSONL 里抽（thought 的 tool_calls + observation 的 tool 字段），
  不依赖运行时的内存状态，复盘时可直接对 trace 文件重算；
  「发起过」与「成功执行过」是两个口径（extract_tool_names / extract_successful_tool_names）。
- **失败运行不进「诚实率」**：领域外诚实率只统计 status=completed 且回答来自模型的样本，
  并同时报出完成条数与领域外总条数（否则「一次都没答出来」会被算成「诚实拒答」→ 假绿）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

# 领域外问题（知识库里没有）的「诚实否认」关键词：命中即视为没有编造
_HONESTY_MARKERS = ("没有", "查不到", "无法回答", "不知道", "没找到", "不涉及", "没有相关内容", "未找到")

# 最终回答的来源标记：
# - model：模型真实给出的回答；
# - system_fallback：系统兜底文案（模型调用失败时的说明），**不是**模型回答。
#
# ★ 为什么必须区分：兜底文案里写着「……这一步没有产生任何工具调用」，
#   「没有」正好命中上面的否认词 → 不区分的话，一次都没答出来的失败运行
#   会被算成「诚实拒答」，领域外诚实率直接假绿（见 out_of_domain_honesty_rate）。
ANSWER_SOURCE_MODEL = "model"
ANSWER_SOURCE_SYSTEM_FALLBACK = "system_fallback"


@dataclass
class EvalRecord:
    """单条 query 的评测记录（由 run_eval 填充，也可在测试里手工构造）。"""

    query: Dict[str, Any]
    status: str
    steps: int
    tool_calls: int
    tool_ok: int
    tool_failed: int
    tool_retries: int
    tool_degraded: int
    circuit_opens: int
    tool_success_rate: float | None
    # used_tools：发起过 / 执行过的工具名（不区分成功失败）
    used_tools: List[str] = field(default_factory=list)
    # used_tools_ok：**真正执行成功**的工具名（trace 里 observation ok=True 的那些）
    used_tools_ok: List[str] = field(default_factory=list)
    final_answer: str = ""
    # 最终回答的来源：model（模型真实回答）| system_fallback（系统兜底文案）
    final_answer_source: str = ANSWER_SOURCE_MODEL
    # 用量与成本（取自 run_end.metrics["usage"]，口径见 agent/usage.py）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None


def load_eval_set(path: str | Path) -> List[Dict[str, Any]]:
    """加载评测集（JSON 数组，或 {"queries": [...]} 两种形态都支持）。"""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("queries", [])
    return data


def extract_tool_names(trace_path: str | Path) -> List[str]:
    """从一次运行的 JSONL trace 里抽出被调用过的工具名（去重、保序）。

    来源：thought 事件的 tool_calls（模型发起）与 observation 事件的 tool 字段（实际执行）。
    任一事件类型缺字段都不影响另一路。

    ★ 口径：这里是「**发起过 / 执行过**」，**不区分成功失败**——
      想知道「真正用上了哪些工具」（ok=True 的执行），用 extract_successful_tool_names。
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        return []
    names: List[str] = []
    seen = set()
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            # 路由一：任何事件里都可能带 tool_calls 列表
            for call in ev.get("tool_calls") or []:
                n = call.get("name") if isinstance(call, dict) else None
                if isinstance(n, str) and n not in seen:
                    names.append(n)
                    seen.add(n)
            # 路由二：observation 事件直接带 tool 字段
            t = ev.get("tool")
            if isinstance(t, str) and t not in seen:
                names.append(t)
                seen.add(t)
    return names


def extract_successful_tool_names(trace_path: str | Path) -> List[str]:
    """从一次运行的 JSONL trace 里抽出**真正执行成功**的工具名（去重、保序）。

    只认 observation 事件且 `ok=True` 的那一条。

    ★ 为什么要有这个函数：thought 里的 tool_calls 只说明「模型发起了调用」，
      工具完全可能全部失败（超时 / 降级 / 熔断）。拿「发起过」判断「工具选择正确」，
      会把「试了一下没成功、最后改用记忆硬答」也算成选对了 → 指标虚高。
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        return []
    names: List[str] = []
    seen = set()
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") != "observation" or not ev.get("ok"):
                continue
            t = ev.get("tool")
            if isinstance(t, str) and t not in seen:
                names.append(t)
                seen.add(t)
    return names


def evaluate(records: List[EvalRecord]) -> Dict[str, Any]:
    """纯函数：把若干条 EvalRecord 聚合成作品集级指标。离线可单测。"""
    n = len(records)
    if n == 0:
        return {"n": 0}

    completed = [r for r in records if r.status == "completed"]

    # 工具选择：两个口径都报出来，别让「发起过」冒充「用上了」
    # - tool_selection_accuracy（严）：期望工具 ⊆ **成功执行过**的工具（observation ok=True）
    # - tool_selection_attempt_rate（宽）：期望工具 ⊆ **发起过**的工具（含执行失败）
    #   模型「试了 search_knowledge 但工具全失败、改用记忆硬答」时，宽口径算它选对了，
    #   严口径不算 —— 两个数一起看才知道指标虚不虚。
    labeled = [r for r in records if r.query.get("expect_tools")]
    sel_ok_strict = [
        r for r in labeled
        if set(r.query["expect_tools"]) <= set(r.used_tools_ok)
    ]
    sel_ok_attempt = [
        r for r in labeled
        if set(r.query["expect_tools"]) <= set(r.used_tools)
    ]

    # 失败恢复率：发生过重试/降级且最终 completed 的占比
    had_failure = [r for r in records if (r.tool_retries + r.tool_degraded) > 0]
    recovered = [r for r in had_failure if r.status == "completed"]

    # 领域外诚实率：只统计**真正完成**（status=completed）的样本，且回答必须来自模型。
    # ★ 口径说明（假绿修复）：模型调用失败时 graph 的兜底文案是「[模型调用失败] …这一步没有产生任何工具调用」，
    #   「没有」正好命中否认词 —— 不看 status 的话，一次都没答出来的运行会被算成「诚实拒答」，
    #   诚实率可以是 1.0，但那是**没答**不是**诚实拒答**。所以分母只看完成条数，
    #   并且把完成条数（out_of_domain_completed_n）与领域外总条数（out_of_domain_n）一起报出来。
    ood = [r for r in records if r.query.get("category") == "out_of_domain"]
    ood_completed = [r for r in ood if r.status == "completed"]
    honest = [
        r for r in ood_completed
        if r.final_answer_source == ANSWER_SOURCE_MODEL
        and any(k in r.final_answer for k in _HONESTY_MARKERS)
    ]

    # 分类别完成率
    by_cat: Dict[str, Dict[str, int]] = {}
    for r in records:
        cat = r.query.get("category", "uncategorized")
        d = by_cat.setdefault(cat, {"total": 0, "completed": 0})
        d["total"] += 1
        if r.status == "completed":
            d["completed"] += 1

    sr_vals = [r.tool_success_rate for r in records if r.tool_success_rate is not None]
    avg_steps = sum(r.steps for r in records) / n

    # 用量与成本：按「单条平均 + 总量」两个口径报（成本只统计命中单价表的条目）
    priced = [r for r in records if r.estimated_cost_usd is not None]
    total_tokens = sum(r.total_tokens for r in records)
    total_cost = round(sum(float(r.estimated_cost_usd) for r in priced), 6) if priced else None

    def _rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "n": n,
        "completion_rate": _rate(len(completed), n),
        "avg_steps": round(avg_steps, 2),
        "avg_tool_success_rate": _rate(sum(sr_vals), len(sr_vals)) if sr_vals else None,
        "tool_selection_accuracy": _rate(len(sel_ok_strict), len(labeled)),
        "tool_selection_attempt_rate": _rate(len(sel_ok_attempt), len(labeled)),
        "tool_selection_n": len(labeled),
        "failure_recovery_rate": _rate(len(recovered), len(had_failure)),
        "out_of_domain_honesty_rate": _rate(len(honest), len(ood_completed)),
        "out_of_domain_n": len(ood),
        "out_of_domain_completed_n": len(ood_completed),
        "total_prompt_tokens": sum(r.prompt_tokens for r in records),
        "total_completion_tokens": sum(r.completion_tokens for r in records),
        "total_tokens": total_tokens,
        "avg_tokens_per_query": round(total_tokens / n, 1),
        "total_cost_usd": total_cost,
        "avg_cost_per_query_usd": round(total_cost / len(priced), 6) if priced else None,
        "costed_n": len(priced),
        "by_category": {
            cat: {"total": d["total"], "completed": d["completed"], "rate": _rate(d["completed"], d["total"])}
            for cat, d in by_cat.items()
        },
    }


def render_markdown(summary: Dict[str, Any], records: List[EvalRecord]) -> str:
    """把评测结果渲染成可贴进 README / 面试材料的 Markdown。"""
    lines = ["# hello-agent 量化评测报告", ""]
    if summary.get("n"):
        lines.append(f"- 评测条数：**{summary['n']}**")
        lines.append(f"- 完成率（status=completed）：**{summary['completion_rate']}**")
        lines.append(f"- 平均步数：**{summary['avg_steps']}**")
        lines.append(f"- 平均工具成功率：**{summary['avg_tool_success_rate']}**")
        lines.append(
            f"- 工具选择准确率（期望工具 ⊆ **成功执行过**的工具）：**{summary['tool_selection_accuracy']}**"
            f"（n={summary.get('tool_selection_n')}）"
        )
        lines.append(
            f"- 工具选择尝试率（期望工具 ⊆ **发起过**的工具，含执行失败）："
            f"**{summary.get('tool_selection_attempt_rate')}**"
        )
        lines.append(f"- 失败恢复率（重试/降级后仍完成）：**{summary['failure_recovery_rate']}**")
        lines.append(
            f"- 领域外诚实率（只统计真正完成的运行）：**{summary['out_of_domain_honesty_rate']}**"
            f"（完成 **{summary.get('out_of_domain_completed_n')}** / 领域外共 "
            f"**{summary.get('out_of_domain_n')}** 条）"
        )
        lines.append(f"- 总 token：**{summary.get('total_tokens')}**"
                     f"（in {summary.get('total_prompt_tokens')} / out {summary.get('total_completion_tokens')}），"
                     f"平均 **{summary.get('avg_tokens_per_query')}** / 条")
        lines.append(f"- 估算总成本：**${summary.get('total_cost_usd')}**"
                     f"（{summary.get('costed_n')}/{summary.get('n')} 条命中单价表；平均 "
                     f"${summary.get('avg_cost_per_query_usd')} / 条）")
        lines.append("")
        lines.append("## 分类别完成率")
        for cat, d in summary.get("by_category", {}).items():
            lines.append(f"- {cat}：{d['completed']}/{d['total']}（{d['rate']}）")
    lines.append("")
    lines.append("## 逐条明细")
    lines.append("")
    lines.append("| # | category | status | steps | tokens | cost_usd | tools | success_rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(records, 1):
        tools = ",".join(r.used_tools) or "—"
        cost = "—" if r.estimated_cost_usd is None else f"{r.estimated_cost_usd:.6f}"
        lines.append(
            f"| {i} | {r.query.get('category', '')} | {r.status} | {r.steps} | {r.total_tokens} "
            f"| {cost} | {tools} | {r.tool_success_rate} |"
        )
    return "\n".join(lines)
