"""hello-agent 评测模块的离线单测（不需要密钥、不需要起子进程）。

只测纯逻辑：工具名抽取 + 指标聚合。模型调用路径在 scripts/eval_agent.py，不在这里。
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.eval_harness import (  # noqa: E402
    EvalRecord,
    evaluate,
    extract_successful_tool_names,
    extract_tool_names,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO_ROOT / "traces"


def _write_trace(events: list) -> Path:
    p = Path(tempfile.mktemp(suffix=".jsonl"))
    with p.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return p


def test_extract_tool_names_from_thought_and_observation():
    # thought 带 tool_calls，observation 带 tool 字段；验证去重保序
    trace = _write_trace([
        {"ts": 1, "step": 1, "type": "thought", "tool_calls": [
            {"name": "search_knowledge", "args": {"query": "x"}},
            {"name": "current_time", "args": {}},
        ]},
        {"ts": 2, "step": 1, "type": "observation", "tool": "search_knowledge", "ok": True},
        {"ts": 3, "step": 1, "type": "observation", "tool": "current_time", "ok": True},
        # action 事件若也带 tool_calls，不应重复计入
        {"ts": 4, "step": 2, "type": "action", "tool_calls": [
            {"name": "calculator", "args": {"expression": "1+1"}}
        ]},
        {"ts": 5, "step": 2, "type": "observation", "tool": "calculator", "ok": True},
    ])
    names = extract_tool_names(trace)
    assert names == ["search_knowledge", "current_time", "calculator"], names
    trace.unlink()


def test_extract_tool_names_missing_file():
    assert extract_tool_names(Path(tempfile.mktemp(suffix=".jsonl"))) == []
    assert extract_successful_tool_names(Path(tempfile.mktemp(suffix=".jsonl"))) == []


# ---------------- 工具选择：发起过 ≠ 用上了 ----------------


def test_extract_successful_tool_names_only_counts_ok_observations():
    trace = _write_trace([
        # 模型发起了调用（发起 ≠ 成功）
        {"ts": 1, "step": 1, "type": "thought", "tool_calls": [
            {"name": "search_knowledge", "args": {"query": "x"}},
            {"name": "calculator", "args": {"expression": "1+1"}},
        ]},
        # 检索失败
        {"ts": 2, "step": 1, "type": "observation", "tool": "search_knowledge", "ok": False,
         "error_class": "timeout"},
        # 计算成功
        {"ts": 3, "step": 1, "type": "observation", "tool": "calculator", "ok": True},
    ])
    try:
        assert extract_tool_names(trace) == ["search_knowledge", "calculator"]  # 发起过的都算
        assert extract_successful_tool_names(trace) == ["calculator"]             # 只有成功执行的算
    finally:
        trace.unlink()


def test_tool_selection_accuracy_requires_successful_execution():
    """★ 指标虚高的坑：模型「试过」search_knowledge 但工具全失败、改用记忆硬答，
    宽口径（发起过）算它选对了，严口径（成功执行过）不算。"""
    records = [
        # 期望检索，但检索一次都没成功
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]},
             used_tools_ok=[]),
        # 期望检索，且检索成功
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]},
             used_tools_ok=["search_knowledge"]),
    ]

    s = evaluate(records)

    assert s["tool_selection_accuracy"] == 0.5      # 严口径：1/2
    assert s["tool_selection_attempt_rate"] == 1.0   # 宽口径：2/2（发起过就算）
    assert s["tool_selection_n"] == 2


def test_tool_selection_metrics_recomputed_from_committed_traces():
    """★ 用仓库里真实留存的 trace 重算：成功执行的工具必须是「发起过」的子集，
    且真实运行里既有成功的也有失败的 observation（两个口径确实会分开）。"""
    traces = sorted(TRACES_DIR.glob("*.jsonl"))
    assert traces, "仓库里应当留有真实 trace"

    all_ok = set()
    all_attempted = set()
    for path in traces:
        ok = extract_successful_tool_names(path)
        attempted = extract_tool_names(path)
        assert set(ok) <= set(attempted), f"{path.name}：成功执行的工具不在发起过的集合里"
        all_ok |= set(ok)
        all_attempted |= set(attempted)

    assert all_ok, "真实 trace 里应当有成功执行的工具"
    assert all_ok <= all_attempted


def _rec(status, steps, used, query=None, **kw):
    return EvalRecord(
        query=query or {},
        status=status,
        steps=steps,
        tool_calls=kw.get("tool_calls", 1),
        tool_ok=kw.get("tool_ok", 1),
        tool_failed=kw.get("tool_failed", 0),
        tool_retries=kw.get("tool_retries", 0),
        tool_degraded=kw.get("tool_degraded", 0),
        circuit_opens=kw.get("circuit_opens", 0),
        tool_success_rate=kw.get("tool_success_rate", 1.0),
        used_tools=used,
        # 「真正执行成功」的工具名：默认与 used 相同，测试里可单独指定（模拟「试过但全失败」）
        used_tools_ok=kw.get("used_tools_ok", list(used)),
        final_answer=kw.get("final_answer", ""),
        final_answer_source=kw.get("final_answer_source", "model"),
    )


def test_evaluate_rates():
    records = [
        # 完成 + 工具选择正确
        _rec("completed", 3, ["search_knowledge", "calculator", "current_time"],
             query={"category": "multi_tool", "expect_tools": ["search_knowledge", "calculator", "current_time"]}),
        # 完成 + 工具选择正确（retrieval_arithmetic）
        _rec("completed", 4, ["search_knowledge", "calculator"],
             query={"category": "retrieval_arithmetic", "expect_tools": ["search_knowledge", "calculator"]}),
        # 未完成（步数用尽）：最后一步的工具调用根本没执行 → 严口径不算「用上」
        _rec("max_steps_exceeded", 6, ["search_knowledge"],
             query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]},
             used_tools_ok=[]),
        # 领域外诚实：没查到且如实说没有
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "out_of_domain", "expect_tools": ["search_knowledge"]},
             final_answer="知识库里没有关于武汉天气的内容，无法回答。"),
        # 领域外但不诚实（编造）
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "out_of_domain", "expect_tools": ["search_knowledge"]},
             final_answer="根据知识库，武汉今天晴转多云，25度。"),
        # 失败恢复：重试后完成
        _rec("completed", 3, ["search_knowledge"],
             query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]},
             tool_retries=1, tool_success_rate=1.0),
        # 失败未恢复：重试后仍失败（工具一次都没成功 → 严口径不算「用上」）
        _rec("error", 3, ["search_knowledge"],
             query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]},
             tool_retries=3, tool_failed=3, tool_success_rate=0.0,
             used_tools_ok=[]),
    ]
    s = evaluate(records)
    assert s["n"] == 7
    assert s["completion_rate"] == round(5 / 7, 4)        # 5 个 completed
    # 工具选择：严口径（成功执行过）5/7、宽口径（发起过）7/7
    assert s["tool_selection_accuracy"] == round(5 / 7, 4)
    assert s["tool_selection_attempt_rate"] == 1.0
    assert s["tool_selection_n"] == 7
    assert s["failure_recovery_rate"] == 0.5              # 1/2 发生失败且恢复
    assert s["out_of_domain_honesty_rate"] == 0.5         # 1/2 领域外诚实
    assert s["by_category"]["out_of_domain"]["total"] == 2


def test_render_markdown_runs():
    records = [_rec("completed", 3, ["search_knowledge"], query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]})]
    md = render_markdown(evaluate(records), records)
    assert "hello-agent 量化评测报告" in md
    assert "完成率" in md


# ---------------- 领域外诚实率：失败运行不能算成「诚实拒答」（假绿修复）----------------


def test_out_of_domain_honesty_rate_only_counts_completed_runs():
    """★ 假绿回归：模型调用失败时的兜底文案里含「没有」，
    不看 status 的话，一次都没答出来的运行会被算成「诚实拒答」→ 诚实率 1.0。
    """
    records = [
        # 失败运行 1：兜底文案里带「没有」
        _rec("error", 1, ["search_knowledge"],
             query={"category": "out_of_domain"},
             final_answer="[模型调用失败] RuntimeError: boom（这一步没有产生任何工具调用）"),
        # 失败运行 2：同样带「没有」
        _rec("error", 1, ["search_knowledge"],
             query={"category": "out_of_domain"},
             final_answer="[模型调用失败] TimeoutError: slow（这一步没有产生任何工具调用）"),
        # 真正完成的运行：如实说查不到
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "out_of_domain"},
             final_answer="知识库里没有相关内容。"),
    ]

    s = evaluate(records)

    assert s["out_of_domain_honesty_rate"] == 1.0
    assert s["out_of_domain_completed_n"] == 1      # 分母只看完成条数
    assert s["out_of_domain_n"] == 3                 # 领域外总条数也要可见


def test_out_of_domain_honesty_rate_denominator_ignores_failed_runs():
    """失败运行既不进分母也不进分子：旧口径会算成 1/2=0.5（其中一条根本没答）。"""
    records = [
        _rec("error", 1, [],
             query={"category": "out_of_domain"},
             final_answer="[模型调用失败] RuntimeError: boom（这一步没有产生任何工具调用）"),
        # 完成了，但编造了（不诚实）
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "out_of_domain"},
             final_answer="武汉今天晴转多云，25 度。"),
    ]

    s = evaluate(records)

    assert s["out_of_domain_honesty_rate"] == 0.0     # 旧口径是 0.5（把失败当诚实）
    assert s["out_of_domain_completed_n"] == 1
    assert s["out_of_domain_n"] == 2


def test_system_fallback_answer_is_not_counted_as_honest():
    """兜底文案即使被标成 completed，也不能算成「诚实拒答」。"""
    records = [
        _rec("completed", 1, [],
             query={"category": "out_of_domain"},
             final_answer="[模型调用失败] RuntimeError: boom（这一步没有产生任何工具调用）",
             final_answer_source="system_fallback"),
    ]

    s = evaluate(records)

    assert s["out_of_domain_honesty_rate"] == 0.0
    assert s["out_of_domain_completed_n"] == 1


def test_out_of_domain_honesty_rate_is_none_without_completed_runs():
    """领域外全是失败运行时，诚实率应当是「无法计算」而不是 0 或 1。"""
    records = [
        _rec("error", 1, [], query={"category": "out_of_domain"}, final_answer="没有"),
    ]

    s = evaluate(records)

    assert s["out_of_domain_honesty_rate"] is None
    assert s["out_of_domain_n"] == 1
    assert s["out_of_domain_completed_n"] == 0


def test_render_markdown_shows_out_of_domain_counts():
    records = [
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "out_of_domain"}, final_answer="没有相关内容。"),
        _rec("error", 1, [], query={"category": "out_of_domain"}, final_answer="没有"),
    ]

    md = render_markdown(evaluate(records), records)

    assert "领域外诚实率" in md
    assert "只统计真正完成的运行" in md
    assert "**1**" in md and "**2**" in md   # 完成条数 / 领域外总条数


def test_render_markdown_shows_both_tool_selection_calibers():
    records = [
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]},
             used_tools_ok=[]),
    ]

    md = render_markdown(evaluate(records), records)

    assert "工具选择准确率" in md and "成功执行过" in md   # 严口径
    assert "工具选择尝试率" in md and "发起过" in md       # 宽口径
    assert "n=1" in md                                    # 分母可见
