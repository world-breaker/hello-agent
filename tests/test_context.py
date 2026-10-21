"""上下文压缩单测（离线、无 LLM）。

重点验证：
1. 未超预算时完全不动（no-op）；
2. 超预算时只压缩内容、不删消息，且保序/保协议配对（tool_call_id ↔ tool_calls）；
3. 第一条用户任务与最近 N 条原样保留。
"""

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.context import compress_messages  # noqa: E402


def _pair(index: int, content_size: int = 300) -> tuple[AIMessage, ToolMessage]:
    call_id = f"call_{index}"
    ai = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge",
                "args": {"query": f"q{index}"},
                "id": call_id,
            }
        ],
    )
    tool = ToolMessage(
        content="检索结果" * (content_size // 4),
        tool_call_id=call_id,
        name="search_knowledge",
    )
    return ai, tool


def build_messages(n_pairs: int = 8) -> list:
    messages = [HumanMessage(content="任务：检索资料并总结，最后给出结论。")]
    for index in range(n_pairs):
        messages.extend(_pair(index))
    messages.append(AIMessage(content="最终回答：" + "总结" * 50))
    return messages


def test_noop_when_under_budget():
    messages = build_messages(n_pairs=2)
    output, stats = compress_messages(messages, max_chars=10**9, keep_last=3)

    assert stats["compressed"] is False
    assert output == messages
    assert stats["before_chars"] == stats["after_chars"]


def test_compresses_without_breaking_protocol():
    messages = build_messages(n_pairs=8)
    keep_last = 3
    max_chars = 2000

    output, stats = compress_messages(
        messages, max_chars=max_chars, keep_last=keep_last
    )

    assert stats["compressed"] is True
    assert stats["after_chars"] < stats["before_chars"]
    assert stats["after_chars"] <= max_chars
    assert stats["compacted_messages"] > 0

    # 第一条用户任务原样保留
    assert isinstance(output[0], HumanMessage)
    assert output[0].content == messages[0].content

    # 最近 N 条原样保留
    assert [m.content for m in output[-keep_last:]] == [
        m.content for m in messages[-keep_last:]
    ]

    # 协议配对：每个 ToolMessage 的 tool_call_id 都能在某个 AIMessage.tool_calls 中找到
    call_ids = set()
    for message in output:
        for call in getattr(message, "tool_calls", None) or []:
            call_ids.add(call.get("id"))
    for message in output:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in call_ids

    # 消息数量不变（只压内容、不删消息）
    assert len(output) == len(messages)


def test_compression_is_idempotent():
    messages = build_messages(n_pairs=8)
    first, stats_first = compress_messages(messages, max_chars=2000, keep_last=3)
    second, stats_second = compress_messages(first, max_chars=2000, keep_last=3)

    assert stats_first["compressed"] is True
    assert stats_second["compressed"] is False
    assert [m.content for m in second] == [m.content for m in first]


# ---------------- max_chars 是「目标预算」，不是硬上限 ----------------


def test_under_budget_reports_no_overrun():
    """压进预算时：over_budget=False、超出 0 字符、没用到底线。"""
    messages = build_messages(n_pairs=8)
    _, stats = compress_messages(messages, max_chars=2000, keep_last=3)

    assert stats["compressed"] is True
    assert stats["after_chars"] <= stats["max_chars"]
    assert stats["over_budget"] is False
    assert stats["over_budget_chars"] == 0
    assert stats["budget_floor_hit"] is False


def test_over_budget_is_reported_not_hidden():
    """★ 压不下来说得出来：三档预览用完仍然超预算时，stats 必须明确标出来。

    30 轮工具调用（≈9600 字符）在 2000 字符预算下，三档预览（120 / 40 / 0）都用完
    仍然超预算（压完 2720 字符）—— 这是**不能靠删消息解决的**（删了会破坏
    tool_calls ↔ ToolMessage 配对），所以必须把它变成**可感知**的信号，
    而不是让调用方以为「压进预算了」。
    """
    messages = build_messages(n_pairs=30)
    output, stats = compress_messages(messages, max_chars=2000, keep_last=3)

    assert stats["compressed"] is True
    assert stats["after_chars"] > stats["max_chars"]          # 确实超了
    assert stats["over_budget"] is True
    assert stats["over_budget_chars"] == stats["after_chars"] - stats["max_chars"]
    assert stats["budget_floor_hit"] is True                  # 三档预览用完仍超 → 触底
    # 行为没变：仍然只压内容、不删消息
    assert len(output) == len(messages)


def test_noop_case_reports_budget_fields():
    """没触发压缩时，预算相关字段也要在（缺字段会让下游统计写代码变脆）。"""
    messages = build_messages(n_pairs=2)
    _, stats = compress_messages(messages, max_chars=10**9, keep_last=3)

    assert stats["compressed"] is False
    assert stats["over_budget"] is False
    assert stats["over_budget_chars"] == 0
    assert stats["budget_floor_hit"] is False
