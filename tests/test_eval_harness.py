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
    extract_tool_names,
    render_markdown,
)


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
        final_answer=kw.get("final_answer", ""),
    )


def test_evaluate_rates():
    records = [
        # 完成 + 工具选择正确
        _rec("completed", 3, ["search_knowledge", "calculator", "current_time"],
             query={"category": "multi_tool", "expect_tools": ["search_knowledge", "calculator", "current_time"]}),
        # 完成 + 工具选择正确（retrieval_arithmetic）
        _rec("completed", 4, ["search_knowledge", "calculator"],
             query={"category": "retrieval_arithmetic", "expect_tools": ["search_knowledge", "calculator"]}),
        # 未完成（步数用尽）
        _rec("max_steps_exceeded", 6, ["search_knowledge"],
             query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]}),
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
        # 失败未恢复：重试后仍失败
        _rec("error", 3, ["search_knowledge"],
             query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]},
             tool_retries=3, tool_failed=3, tool_success_rate=0.0),
    ]
    s = evaluate(records)
    assert s["n"] == 7
    assert s["completion_rate"] == round(5 / 7, 4)        # 5 个 completed
    assert s["tool_selection_accuracy"] == 1.0             # 所有 labeled 的期望工具 ⊆ used
    assert s["failure_recovery_rate"] == 0.5              # 1/2 发生失败且恢复
    assert s["out_of_domain_honesty_rate"] == 0.5         # 1/2 领域外诚实
    assert s["by_category"]["out_of_domain"]["total"] == 2


def test_render_markdown_runs():
    records = [_rec("completed", 3, ["search_knowledge"], query={"category": "retrieval_only", "expect_tools": ["search_knowledge"]})]
    md = render_markdown(evaluate(records), records)
    assert "hello-agent 量化评测报告" in md
    assert "完成率" in md
