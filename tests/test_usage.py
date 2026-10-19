"""token 用量与成本统计的离线单测（不联网、不花钱）。

覆盖三层：
1. 纯函数（单价查表 / 成本换算 / usage 抽取 / trace 聚合）—— 断言精确数字；
2. runner 集成（走真实 run_task 链路，假模型带 usage）—— 断言 metrics 与打印摘要；
3. 累计层（扫 trace 目录 + 汇总）—— 断言「多次运行的累计」。
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.eval_harness import EvalRecord, evaluate, render_markdown  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.runner import run_task  # noqa: E402
from agent.toolkit import ToolSet  # noqa: E402
from agent.tools_builtin import calculator  # noqa: E402
from agent.usage import (  # noqa: E402
    aggregate_usage,
    estimate_cost,
    extract_usage,
    format_usage_line,
    load_pricing,
    price_for,
    scan_trace_dir,
    summarize_usage,
)
from tests.fake_llm import ScriptedChatModel, tool_call  # noqa: E402

PRICING = {"demo-model": {"input": 1.0, "output": 2.0}}  # 单价取整便于手算：$1 / $2 每 1M tokens


# ---------------- 1. 纯函数 ----------------


def test_price_lookup_exact_and_wildcard():
    assert price_for("demo-model", PRICING) == {"input": 1.0, "output": 2.0}
    assert price_for("demo-model", {"*": {"input": 0.5, "output": 0.5}}) == {"input": 0.5, "output": 0.5}
    assert price_for("unknown-model", PRICING) is None
    assert price_for(None, PRICING) is None


def test_estimate_cost_math_is_exact():
    # 1M 输入 + 2M 输出 ×（$1 / $2 per 1M）= $1 + $4 = $5
    cost, unit = estimate_cost("demo-model", 1_000_000, 2_000_000, PRICING)
    assert cost == 5.0
    assert unit == {"input": 1.0, "output": 2.0}
    # 1234 in / 567 out → (1234*1 + 567*2) / 1e6 = 2368 / 1e6
    cost, _ = estimate_cost("demo-model", 1234, 567, PRICING)
    assert cost == round(2368 / 1_000_000, 6) == 0.002368


def test_estimate_cost_without_price_returns_none():
    assert estimate_cost("unknown-model", 1000, 1000, PRICING) == (None, None)


def test_extract_usage_prefers_usage_metadata():
    msg = AIMessage(content="hi", usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})
    assert extract_usage(msg) == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}


def test_extract_usage_falls_back_to_response_metadata():
    """有些 OpenAI 兼容端点只填 response_metadata.token_usage（prompt_tokens 口径）。"""
    msg = AIMessage(
        content="hi",
        response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
    )
    assert extract_usage(msg) == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


def test_extract_usage_missing_returns_none():
    assert extract_usage(AIMessage(content="no usage here")) is None


def test_summarize_usage_counts_and_flags_missing_usage():
    events = [
        {"type": "run_start"},
        {"type": "llm_call", "ok": True, "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}},
        {"type": "llm_call", "ok": True, "usage": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}},
        {"type": "llm_call", "ok": True, "usage": None},          # 端点没返回用量 → 明示，而不是当 0 悄悄算上
        {"type": "llm_call", "ok": False, "error": "boom"},        # 失败的调用同样算「无用量」
        {"type": "run_end"},
    ]
    usage = summarize_usage(events, model="demo-model", pricing=PRICING)
    assert usage["prompt_tokens"] == 150
    assert usage["completion_tokens"] == 30
    assert usage["total_tokens"] == 180
    assert usage["llm_calls_with_usage"] == 2
    assert usage["llm_calls_without_usage"] == 2
    assert usage["estimated_cost_usd"] == round((150 * 1.0 + 30 * 2.0) / 1_000_000, 6) == 0.00021
    assert usage["pricing_found"] is True


def test_summarize_usage_unknown_model_reports_tokens_only():
    events = [{"type": "llm_call", "ok": True, "usage": {"input_tokens": 9, "output_tokens": 1, "total_tokens": 10}}]
    usage = summarize_usage(events, model="mystery-model", pricing=PRICING)
    assert usage["total_tokens"] == 10
    assert usage["estimated_cost_usd"] is None
    assert usage["pricing_found"] is False
    assert "不在单价表" in format_usage_line(usage)


def test_format_usage_line_includes_numbers_and_cost():
    usage = summarize_usage(
        [{"type": "llm_call", "ok": True, "usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}}],
        model="demo-model",
        pricing=PRICING,
    )
    line = format_usage_line(usage, extra=" | 步数=2")
    assert "tokens in=1000 out=500 合计=1500" in line
    assert "$0.002000" in line
    assert "步数=2" in line


# ---------------- 2. runner 集成 ----------------

def _usage_message(text: str, in_tokens: int, out_tokens: int) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": in_tokens, "output_tokens": out_tokens,
                        "total_tokens": in_tokens + out_tokens},
    )


def patch_env(monkeypatch, tmp_path, responses, max_steps=3, model="deepseek-chat"):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-for-test-not-a-real-key")
    monkeypatch.setenv("MODEL", model)

    config = load_config()
    config["agent"]["max_steps"] = max_steps
    config["tools"]["retry"]["backoff_base_s"] = 0.01
    config["trace"]["dir"] = str(tmp_path)
    config["trace"]["console"] = "quiet"
    config["agent"]["checkpoint"]["path"] = str(tmp_path / "checkpoints.sqlite")

    @asynccontextmanager
    async def fake_toolset(cfg, tracer, rag_mode_override=None):
        tracer.event("tools_ready", tools=["calculator"], rag_mode="direct", rag_note="测试替身", fallbacks=[])
        yield ToolSet(tools={"calculator": calculator}, rag_mode="direct", tool_names=["calculator"])

    monkeypatch.setattr("agent.runner.build_toolset", fake_toolset)
    monkeypatch.setattr("agent.runner.build_llm", lambda settings: ScriptedChatModel(responses=responses))
    return config


def test_run_task_aggregates_tokens_and_cost(monkeypatch, tmp_path, capsys):
    """一次多步任务：2 次模型调用（1 次发起工具、1 次收尾），token 与成本都要落到 metrics。"""
    first = tool_call("calculator", {"expression": "6*7"}, "call_1")
    first.usage_metadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    config = patch_env(
        monkeypatch, tmp_path,
        [first, _usage_message("6*7=42。", in_tokens=300, out_tokens=20)],
    )

    result = asyncio.run(run_task("算一下 6*7", config=config))
    out = capsys.readouterr().out

    usage = result.usage
    assert usage["prompt_tokens"] == 400          # 100 + 300
    assert usage["completion_tokens"] == 30       # 10 + 20
    assert usage["total_tokens"] == 430
    assert usage["llm_calls_with_usage"] == 2
    # deepseek-chat 单价 0.27 / 1.10（config.yaml → llm.pricing）
    assert usage["estimated_cost_usd"] == round((400 * 0.27 + 30 * 1.10) / 1_000_000, 6)
    assert result.metrics["usage"]["total_tokens"] == 430
    assert result.metrics["usage"]["model"] == "deepseek-chat"

    # 控制台必须能看到一行摘要
    assert "tokens in=400 out=30 合计=430" in out
    assert "估算成本=$0.000141" in out

    # trace 里也带 usage（复盘时可直接重算）
    events = [json.loads(line) for line in Path(result.trace_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    llm_calls = [e for e in events if e["type"] == "llm_call"]
    assert [e["usage"]["input_tokens"] for e in llm_calls] == [100, 300]
    assert events[-1]["metrics"]["usage"]["total_tokens"] == 430


def test_run_task_without_usage_still_reports_tokens_zero(monkeypatch, tmp_path, capsys):
    """端点不返回用量时不报错、不编数：tokens=0 且明确写「无用量」。"""
    config = patch_env(monkeypatch, tmp_path, [AIMessage(content="直接回答")], max_steps=2)
    result = asyncio.run(run_task("随便问问", config=config))
    out = capsys.readouterr().out
    assert result.usage["total_tokens"] == 0
    assert result.usage["llm_calls_without_usage"] == 1
    assert result.usage["estimated_cost_usd"] == 0.0   # 单价命中了，token 为 0 → 成本 0
    assert "0 次有用量/1 次无用量" in out


# ---------------- 3. 累计层（扫 trace 求和） ----------------


def test_scan_trace_dir_and_aggregate_are_cumulative(tmp_path):
    """累计统计只依赖 trace 文件：换单价表 / 复盘旧运行都能重算。"""
    for index, (in_tokens, out_tokens) in enumerate([(1000, 100), (2000, 200)], start=1):
        path = tmp_path / f"run_20260101-00000{index}_run{index}.jsonl"
        events = [
            {"ts": 1, "run_id": f"run{index}", "step": 0, "type": "run_start",
             "task": f"任务{index}", "llm": "model=demo-model base_url=http://x api_key=sk-****", "max_steps": 3},
            {"ts": 2, "run_id": f"run{index}", "step": 1, "type": "llm_call", "ok": True,
             "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens,
                       "total_tokens": in_tokens + out_tokens}},
            {"ts": 3, "run_id": f"run{index}", "step": 1, "type": "run_end", "status": "completed", "steps": 1,
             "metrics": {}},
        ]
        path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")

    rows = scan_trace_dir(tmp_path, PRICING)
    assert [r["model"] for r in rows] == ["demo-model", "demo-model"]  # 模型名从 run_start.llm 解析
    assert rows[0]["total_tokens"] == 1100
    assert rows[0]["estimated_cost_usd"] == round((1000 * 1.0 + 100 * 2.0) / 1_000_000, 6)

    total = aggregate_usage(rows)
    assert total["runs"] == 2
    assert total["total_tokens"] == 3300                       # 1100 + 2200
    assert total["total_prompt_tokens"] == 3000
    assert total["total_completion_tokens"] == 300
    assert total["total_cost_usd"] == round((3000 * 1.0 + 300 * 2.0) / 1_000_000, 6) == 0.0036
    assert total["avg_tokens_per_run"] == 1650.0


def test_load_pricing_merges_config_overrides():
    pricing = load_pricing({"llm": {"pricing": {"deepseek-chat": {"output": 9.99}, "my-model": {"input": 0.01, "output": 0.02}}}})
    assert pricing["deepseek-chat"]["output"] == 9.99
    assert pricing["deepseek-chat"]["input"] == 0.27   # 未覆盖的字段保留内置值
    assert pricing["my-model"] == {"input": 0.01, "output": 0.02}


# ---------------- 4. 评测侧聚合（沿用 eval_harness 风格） ----------------


def test_evaluate_aggregates_tokens_and_cost():
    records = [
        EvalRecord(query={"category": "multi_tool"}, status="completed", steps=2, tool_calls=1, tool_ok=1,
                   tool_failed=0, tool_retries=0, tool_degraded=0, circuit_opens=0, tool_success_rate=1.0,
                   prompt_tokens=300, completion_tokens=20, total_tokens=320, estimated_cost_usd=0.0001),
        EvalRecord(query={"category": "retrieval_only"}, status="completed", steps=1, tool_calls=0, tool_ok=0,
                   tool_failed=0, tool_retries=0, tool_degraded=0, circuit_opens=0, tool_success_rate=None,
                   prompt_tokens=700, completion_tokens=30, total_tokens=730, estimated_cost_usd=None),
    ]
    summary = evaluate(records)
    assert summary["total_tokens"] == 1050
    assert summary["total_prompt_tokens"] == 1000
    assert summary["avg_tokens_per_query"] == 525.0
    assert summary["total_cost_usd"] == 0.0001     # 只统计命中单价表的那条
    assert summary["costed_n"] == 1
    md = render_markdown(summary, records)
    assert "总 token" in md and "估算总成本" in md
