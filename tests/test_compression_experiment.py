"""上下文压缩对照实验的**纯逻辑**单测（不联网、不跑模型、不起子进程）。

守住三件事：
1. 假模型的决策是确定的、且只依赖任务文本（受控变量的前提）；
2. 工具参数的「压坏」判定（与对照组逐条比对）口径正确；
3. trace 解析（压缩事件 / 工具调用）口径正确。

实验脚本本身（`experiments/context_compression/run_experiment.py`）要真跑一遍才有数字，
那部分**不在单测里**——数字在 `experiments/results/`。
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.context_compression.fake_policy_model import (  # noqa: E402
    ContextFaithfulFakeModel,
    _find_expression,
    _kb_text,
    _first_number,
    _topic,
    plan_tools,
)
from experiments.context_compression.run_experiment import (  # noqa: E402
    compression_stats,
    diff_tool_args,
    pick_tasks,
    tool_calls_with_args,
)
from agent.eval_harness import load_eval_set  # noqa: E402

EVAL_SET_PATH = REPO_ROOT / "data" / "eval_queries.json"


# ---------------- 工具计划（受控变量的前提）----------------


def test_plan_tools_is_deterministic_and_task_only():
    """同一个任务文本 → 同一个工具计划（所有压缩档位共用同一份计划）。"""
    task = "先查知识库：LoRA 的典型配置里 r 和 α 分别是多少？然后用计算器算一下 α/r，最后告诉我现在的时间。"
    first = plan_tools(task)
    assert first == ["search_knowledge", "current_time", "calculator"]
    assert plan_tools(task) == first


def test_plan_tools_skips_tools_for_clarification_tasks():
    """澄清类任务不该调工具（与评测集里 `expect_tools: []` 的口径一致）。"""
    for task in ("帮我优化一下这段文案。", "把价格改成 200 元。", "统计一下去年有多少用户。"):
        assert plan_tools(task) == [], task


def test_plan_tools_covers_each_tool_family():
    assert plan_tools("现在几点了？") == ["current_time"]
    assert plan_tools("计算 16 除以 8 的结果。") == ["calculator"]
    assert plan_tools("知识库里有没有关于武汉天气的内容？") == ["search_knowledge"]


def test_topic_is_short_and_strips_leading_instruction():
    """检索主题要短：否则被压掉的前 120 字符全是 query 回显，压缩代价就测不出来。"""
    topic = _topic("先查知识库：LoRA 的典型配置里 r 和 α（alpha）分别是多少？然后用计算器算一下 α/r 的值。")
    assert topic == "LoRA 的典型配置里 r 和 α（alpha）分别是多少"
    assert len(topic) <= 40


def test_find_expression_prefers_explicit_formula():
    assert _find_expression("用计算器算 1 / (2 - 2)。") == "1 / (2 - 2)"
    assert _find_expression("用计算器算一下 12 ** * 3 这个表达式。") == "12 ** * 3"
    # 没有现成算式 → None（走「按数字拼」的兜底分支）
    assert _find_expression("知识库里说的 chunk_size 是多少？") is None


# ---------------- 从观察里取知识库正文 ----------------


def test_kb_text_extracts_only_document_text():
    """只取 results[].text：外层 JSON 里有每次都变的 elapsed_ms，取了会造成假阳性。"""
    observation = (
        '{"query": "LoRA", "elapsed_ms": 107.61, "matched": 1, "results": ['
        '{"rank": 1, "score": 0.65, "text": "微调7B模型FP16需要约76GB显存"}]}'
    )
    text = _kb_text(observation)
    assert "76GB" in text
    assert "elapsed_ms" not in text


def test_kb_text_works_on_truncated_json():
    """工具输出进上下文前会被截断到 1200 字符 → JSON 不完整，正则也要抓得到。"""
    truncated = '{"query": "x", "results": [{"text": "典型值256-1024字符"'
    assert "256-1024" in _kb_text(truncated)


def test_first_number_skips_decimals():
    """相似度分数（0.6565）不是「知识库给的数字」，不能被当成依据。"""
    assert _first_number("score 0.6565 ...") is None
    assert _first_number("典型值256-1024字符") == 256


# ---------------- 参数「压坏」的判定 ----------------


def test_diff_tool_args_detects_changed_arguments():
    reference = [{"name": "calculator", "args": {"expression": "(0-16)"}}]
    changed = [{"name": "calculator", "args": {"expression": "(0-0)"}}]
    same = [{"name": "calculator", "args": {"expression": "(0-16)"}}]

    assert diff_tool_args(reference, changed) != []
    assert diff_tool_args(reference, same) == []
    # 调用缺失也要报出来（不能静默吞掉）
    assert diff_tool_args(reference, []) != []


# ---------------- trace 解析 ----------------


def test_compression_stats_sums_events():
    events = [
        {"type": "context_compressed", "before_chars": 3000, "after_chars": 1000},
        {"type": "context_compressed", "before_chars": 2000, "after_chars": 800},
        {"type": "thought", "tool_calls": []},
    ]
    stats = compression_stats(events)
    assert stats["compress_events"] == 2
    assert stats["chars_before"] == 5000
    assert stats["chars_after"] == 1800
    assert stats["chars_saved"] == 3200


def test_tool_calls_with_args_reads_thought_events():
    events = [
        {"type": "thought", "tool_calls": [{"name": "search_knowledge", "args": {"query": "x"}}]},
        {"type": "observation", "tool": "search_knowledge", "ok": True},
        {"type": "thought", "tool_calls": [{"name": "calculator", "args": {"expression": "1+1"}}]},
    ]
    calls = tool_calls_with_args(events)
    assert calls == [
        {"name": "search_knowledge", "args": {"query": "x"}},
        {"name": "calculator", "args": {"expression": "1+1"}},
    ]


# ---------------- 批次挑选 ----------------


def test_pick_tasks_selects_retrieval_tasks_only():
    queries = load_eval_set(EVAL_SET_PATH)
    picked = pick_tasks(queries, limit=0)
    assert picked
    assert all("search_knowledge" in (q.get("expect_tools") or []) for q in picked)
    # limit 生效
    assert len(pick_tasks(queries, limit=3)) == 3


# ---------------- 假模型本身 ----------------


def test_fake_model_emits_tool_calls_then_answer():
    """假模型：先按计划发工具调用，全部拿到结果后才给最终答案。"""
    from langchain_core.messages import HumanMessage, ToolMessage

    model = ContextFaithfulFakeModel()
    messages = [HumanMessage(content="计算 16 除以 8 的结果。")]
    first = model._next_message_for(messages)
    assert first.tool_calls and first.tool_calls[0]["name"] == "calculator"

    messages = messages + [
        first,
        ToolMessage(content='{"ok": true, "result": 2}', tool_call_id="call_1", name="calculator"),
    ]
    second = model._next_message_for(messages)
    assert not second.tool_calls
    assert "2" in second.content


def test_fake_model_usage_metadata_is_char_proxy():
    """token 上报是「字符数代理」：1 字符 ≈ 1 token（不是真实 tokenizer）。"""
    from langchain_core.messages import HumanMessage

    model = ContextFaithfulFakeModel()
    messages = [HumanMessage(content="计算 16 除以 8 的结果。")]
    message = model._next_message_for(messages)
    result = model._finish(messages, message)
    usage = result.generations[0].message.usage_metadata
    assert usage["input_tokens"] == len("计算 16 除以 8 的结果。")
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
