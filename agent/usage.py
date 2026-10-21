"""token 用量与成本估算：从模型响应的 usage 里取数，按「单任务」与「累计」两个层次聚合。

设计取舍（README §5.7 有对应说明）：
- **只加一层薄封装，不另起一套指标**：数字最终并进既有的 `run_end.metrics["usage"]`
  与 JSONL trace 的 `llm_call.usage`，和 `tool_calls / tool_success_rate` 一个口径；
- **事实源是 trace，不是内存状态**：`summarize_usage(events)` / `summarize_tool_outcomes(events)`
  都是纯函数，复盘时可以直接对 `traces/*.jsonl` 重算（`scripts/usage_report.py` 就是这么做的），
  所以「累计」不需要额外持久化，也就不存在账本与 trace 对不上的问题；
- **算不出价就只报 token，绝不编数**：单价表是配置项（`config.yaml → llm.pricing`，按模型名匹配），
  没命中就返回 `estimated_cost_usd = None`，并在控制台明确写「未配置单价」；
- **provider 差异**：优先读 LangChain 统一后的 `usage_metadata`；
  退回读 `response_metadata.token_usage`（OpenAI 原生字段名），两者都没有就记「本次无用量」。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

# 单价单位：USD / 1M tokens（配置里可覆盖）。
# 数值取自 2025-08 各厂公开定价页，**会变**，所以是配置项而不是硬编码常量：
# 要精确计费时改 config.yaml，而不是改代码。
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

PRICING_UNIT = "USD / 1M tokens"


def load_pricing(config: Dict[str, Any] | None) -> Dict[str, Dict[str, float]]:
    """从 config 取单价表（`llm.pricing`），缺省用内置表，按模型名覆盖合并。"""
    pricing = {name: dict(price) for name, price in DEFAULT_PRICING.items()}
    if not config:
        return pricing
    user_pricing = ((config.get("llm") or {}).get("pricing")) or {}
    for model, price in user_pricing.items():
        if not isinstance(price, dict):
            continue
        merged = dict(pricing.get(model, {}))
        for key in ("input", "output"):
            if price.get(key) is not None:
                merged[key] = float(price[key])
        pricing[str(model)] = merged
    return pricing


def price_for(model: str | None, pricing: Dict[str, Dict[str, float]] | None = None) -> Dict[str, float] | None:
    """查单价：先精确匹配模型名，再退到通配 `*`；都没有返回 None。"""
    table = pricing if pricing is not None else DEFAULT_PRICING
    if not model:
        return None
    entry = table.get(str(model))
    if entry is None:
        entry = table.get("*")
    if not entry:
        return None
    if entry.get("input") is None and entry.get("output") is None:
        return None
    return entry


def extract_usage(message: Any) -> Dict[str, int] | None:
    """从一条模型响应里取 prompt/completion tokens；取不到返回 None。

    两条来源（按优先级）：
    1. `usage_metadata`：LangChain 把各家字段统一后的形态（`input_tokens/output_tokens/total_tokens`）；
    2. `response_metadata.token_usage`：provider 原始字段（`prompt_tokens/completion_tokens`），
       有些 OpenAI 兼容端点只填这一处。
    """
    usage = getattr(message, "usage_metadata", None) or {}
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    total = usage.get("total_tokens")

    if prompt is None and completion is None:
        metadata = getattr(message, "response_metadata", None) or {}
        raw = metadata.get("token_usage") or metadata.get("usage") or {}
        prompt = raw.get("prompt_tokens", raw.get("input_tokens"))
        completion = raw.get("completion_tokens", raw.get("output_tokens"))
        total = raw.get("total_tokens")

    if prompt is None and completion is None:
        return None

    prompt_i = int(prompt or 0)
    completion_i = int(completion or 0)
    return {
        "input_tokens": prompt_i,
        "output_tokens": completion_i,
        "total_tokens": int(total) if total is not None else prompt_i + completion_i,
    }


def estimate_cost(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    pricing: Dict[str, Dict[str, float]] | None = None,
) -> tuple[float | None, Dict[str, float] | None]:
    """按单价表算成本；查不到单价时返回 `(None, None)`（只报 token，不编价格）。"""
    unit_price = price_for(model, pricing)
    if unit_price is None:
        return None, None
    cost = (int(input_tokens) * float(unit_price.get("input") or 0.0) +
            int(output_tokens) * float(unit_price.get("output") or 0.0)) / 1_000_000
    return round(cost, 6), unit_price


def summarize_usage(
    events: Iterable[Dict[str, Any]],
    model: str | None = None,
    pricing: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    """纯函数：把若干 trace 事件聚合成「单次运行」的用量与成本。

    输入是 `Tracer.events`（或从 JSONL 读回来的同结构事件），所以离线可重算、可单测。
    口径：
    - `llm_calls_with_usage` / `llm_calls_without_usage`：有/没有 usage 的模型调用次数
      （端点不返回用量时**明示**，而不是静默当 0）；
    - 成本 = 输入 token × 输入单价 + 输出 token × 输出单价；
    - 单价表没命中 → `estimated_cost_usd = None`，`pricing_found = False`。
    """
    input_tokens = 0
    output_tokens = 0
    with_usage = 0
    without_usage = 0

    for event in events:
        if event.get("type") != "llm_call":
            continue
        usage = event.get("usage") or {}
        if usage.get("input_tokens") is None and usage.get("output_tokens") is None:
            without_usage += 1
            continue
        with_usage += 1
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)

    cost, unit_price = estimate_cost(model, input_tokens, output_tokens, pricing)
    summary: Dict[str, Any] = {
        "model": model,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "llm_calls_with_usage": with_usage,
        "llm_calls_without_usage": without_usage,
        "pricing_found": unit_price is not None,
        "pricing_unit": PRICING_UNIT,
        "cost_input_usd": (
            round(input_tokens * float(unit_price.get("input") or 0.0) / 1_000_000, 6)
            if unit_price else None
        ),
        "cost_output_usd": (
            round(output_tokens * float(unit_price.get("output") or 0.0) / 1_000_000, 6)
            if unit_price else None
        ),
        "estimated_cost_usd": cost,
        "price_input_per_1m": float(unit_price["input"]) if unit_price and unit_price.get("input") is not None else None,
        "price_output_per_1m": float(unit_price["output"]) if unit_price and unit_price.get("output") is not None else None,
    }
    return summary


def summarize_tool_outcomes(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """纯函数：把 trace 里的工具结局聚合成「这次运行」的工具口径指标。

    为什么放在这里：和 token 用量一样，**事实源是 trace** —— 每次 `ToolExecutor.execute()`
    都会落一条 `observation` 事件（带 `ok` / `degraded` 字段），
    所以工具成功率也能事后对 `traces/*.jsonl` 重算，不必额外记账本。
    口径必须与 `agent/runner.py:_summarize_metrics` 完全一致，否则两处数字对不上。

    ★ 三种结局互斥且穷尽（一次调用恰好落一条 observation）：
    - `tool_ok`       主通道成功
    - `tool_degraded` **降级成功**：主通道坏、fallback 好 —— 调用方真的拿到了可用结果
    - `tool_failed`   最终失败，没拿到结果

    不变式：`tool_ok + tool_degraded + tool_failed == tool_calls`。
    - `tool_success_rate`      = (ok + degraded) / calls —— **降级成功算成功**
    - `tool_success_rate_main` = ok / calls —— 只看主通道健康度
    """
    calls = 0
    ok = 0
    degraded = 0
    failed = 0
    for event in events:
        if event.get("type") != "observation":
            continue
        calls += 1
        if not event.get("ok"):
            failed += 1
        elif event.get("degraded"):
            degraded += 1
        else:
            ok += 1
    succeeded = ok + degraded
    return {
        "tool_calls": calls,
        "tool_ok": ok,
        "tool_degraded": degraded,
        "tool_failed": failed,
        "tool_succeeded": succeeded,
        "tool_success_rate": round(succeeded / calls, 4) if calls else None,
        "tool_success_rate_main": round(ok / calls, 4) if calls else None,
    }


def format_usage_line(usage: Dict[str, Any], extra: str = "") -> str:
    """一次运行结束时打印的那一行摘要（tokens in/out、估算成本、步数…）。"""
    if not usage:
        return f"===== 用量：无数据（本次运行没有任何模型调用）{extra} ====="
    if usage.get("estimated_cost_usd") is None:
        cost_text = (
            f"估算成本=—（模型 `{usage.get('model')}` 不在单价表，仅报 token；"
            f"可在 config.yaml → llm.pricing 里加单价）"
        )
    else:
        cost_text = (
            f"估算成本=${usage['estimated_cost_usd']:.6f} USD"
            f"（{usage.get('model')}：输入 {usage.get('price_input_per_1m')}/"
            f"输出 {usage.get('price_output_per_1m')} {usage.get('pricing_unit')}）"
        )
    return (
        f"===== 用量：tokens in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')} "
        f"合计={usage.get('total_tokens')}（模型调用 {usage.get('llm_calls_with_usage')} 次有用量"
        f"/{usage.get('llm_calls_without_usage')} 次无用量） | {cost_text}{extra} ====="
    )


def model_from_trace(events: Iterable[Dict[str, Any]]) -> str | None:
    """从 trace 的 `run_start.llm` 里解析模型名（形如 `model=deepseek-chat base_url=…`）。

    这样「复盘重算成本」只需要 trace 文件本身，不需要再问当事人当时用的什么模型。
    """
    for event in events:
        if event.get("type") != "run_start":
            continue
        for token in str(event.get("llm") or "").split():
            if token.startswith("model="):
                return token.split("=", 1)[1] or None
    return None


def read_trace_events(path: str | Path) -> List[Dict[str, Any]]:
    """读一个 JSONL trace 文件（坏行跳过，不因为一行坏数据丢掉整份复盘）。"""
    events: List[Dict[str, Any]] = []
    path = Path(path)
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


def scan_trace_dir(trace_dir: str | Path, pricing: Dict[str, Dict[str, float]] | None = None) -> List[Dict[str, Any]]:
    """扫一个 trace 目录，给出「每次运行」一行的用量与成本明细（按文件名排序 = 时间序）。"""
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(trace_dir).glob("run_*.jsonl")):
        events = read_trace_events(path)
        if not events:
            continue
        model = model_from_trace(events)
        usage = summarize_usage(events, model=model, pricing=pricing)
        tool_outcomes = summarize_tool_outcomes(events)
        run_end = next((e for e in reversed(events) if e.get("type") == "run_end"), {})
        run_start = next((e for e in events if e.get("type") == "run_start"), {})
        rows.append(
            {
                "trace": path.name,
                "run_id": run_start.get("run_id") or run_end.get("run_id"),
                "task": run_start.get("task", ""),
                # 没有 run_end 说明这次运行没有收尾（进程被中断/被杀），如实标出来而不是留空
                "status": run_end.get("status") or "interrupted（无 run_end）",
                "steps": run_end.get("steps"),
                "thread_id": run_end.get("thread_id") or run_start.get("thread_id"),
                "resumed": run_end.get("resumed"),
                **usage,
                **tool_outcomes,
            }
        )
    return rows


def aggregate_usage(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """累计层：把若干「单次运行」的明细加成总数（作品集里「累计花了多少钱」就是它）。

    除了 token / 成本，也把**工具结局**累计起来（与单次运行同一口径）：
    `tool_success_rate` = (ok + degraded) / calls（降级成功算成功），
    `tool_success_rate_main` = ok / calls。
    """
    rows = list(rows)
    priced = [r for r in rows if r.get("estimated_cost_usd") is not None]
    total_cost = round(sum(float(r["estimated_cost_usd"]) for r in priced), 6) if priced else None
    total_tokens = sum(int(r.get("total_tokens") or 0) for r in rows)
    tool_calls = sum(int(r.get("tool_calls") or 0) for r in rows)
    tool_ok = sum(int(r.get("tool_ok") or 0) for r in rows)
    tool_degraded = sum(int(r.get("tool_degraded") or 0) for r in rows)
    tool_failed = sum(int(r.get("tool_failed") or 0) for r in rows)
    tool_succeeded = tool_ok + tool_degraded
    return {
        "runs": len(rows),
        "runs_with_pricing": len(priced),
        "total_prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
        "total_completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "avg_tokens_per_run": round(total_tokens / len(rows), 1) if rows else None,
        "avg_cost_per_run_usd": round(total_cost / len(priced), 6) if priced else None,
        "pricing_unit": PRICING_UNIT,
        "total_tool_calls": tool_calls,
        "total_tool_ok": tool_ok,
        "total_tool_degraded": tool_degraded,
        "total_tool_failed": tool_failed,
        "total_tool_succeeded": tool_succeeded,
        # 成功率 = (ok + degraded) / calls：降级成功**算成功**（调用方拿到了结果），
        # 但 degraded 单独计数，所以主通道健康度看 tool_success_rate_main。
        "tool_success_rate": round(tool_succeeded / tool_calls, 4) if tool_calls else None,
        "tool_success_rate_main": round(tool_ok / tool_calls, 4) if tool_calls else None,
        "note": ("有运行未命中单价表 → 累计成本只统计命中的那部分，token 是全量"
                 if priced and len(priced) != len(rows) else None),
    }
