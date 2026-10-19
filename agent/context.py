"""上下文压缩：在送模型前把「历史消息」压到字符预算内。

为什么需要：
- ReAct 循环里每一步都会把完整 `messages` 再送一次模型；步数一多，token 成本线性上涨，
  最终会撞上模型上下文上限；
- 直接把旧消息删掉会破坏 tool_calls ↔ ToolMessage 的配对（协议要求一一对应），
  所以本模块**只压缩内容、不删除消息**：保留每条的 role / tool_call_id / name，
  把早期文本替换成短摘要占位。

为什么用「字符数」而不是 token 数：
- 不引入 tokenizer 依赖；字符数是保守代理（中文约 1 字 ≈ 1 token，英文约 4 字符 ≈ 1 token）。
- 真正需要精确计费时，可在 `_message_chars()` 处替换成 tokenizer。

配置：
    agent.max_context_chars: 12000   # 字符预算
    agent.context_keep_last: 6       # 最近 N 条原样保留
"""

import json
from typing import Any, Dict, List, Tuple

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def _text_of(message: AnyMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _message_chars(message: AnyMessage) -> int:
    """一条消息的字符成本：正文 + tool_calls 参数。"""
    total = len(_text_of(message))
    for call in getattr(message, "tool_calls", None) or []:
        try:
            total += len(json.dumps(call.get("args") or {}, ensure_ascii=False))
        except (TypeError, ValueError):
            total += len(str(call.get("args")))
    return total


def _with_content(message: AnyMessage, content: str) -> AnyMessage:
    """替换消息正文，保留 role / tool_call_id / name / tool_calls 等协议字段。"""
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    return message.copy(update={"content": content})  # pydantic v1 兜底


def _compact(message: AnyMessage, preview_chars: int) -> AnyMessage:
    """把一条消息压成短占位；不改变消息类型与工具调用配对信息。"""
    if isinstance(message, SystemMessage):
        return message  # 系统提示永远保留

    text = _text_of(message)

    if isinstance(message, HumanMessage):
        # 用户消息可能包含任务约束，保留稍长一点
        limit = max(preview_chars * 2, preview_chars)
        body = text if len(text) <= limit else text[:limit] + "…"
        return _with_content(message, f"[早期用户] {body}")

    if isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            names = [call.get("name") for call in tool_calls]
            # 注意：tool_calls 本身保留（ToolMessage 需要匹配 tool_call_id）
            return _with_content(message, f"[早期步骤] 请求工具：{names}")
        body = text if len(text) <= preview_chars else text[:preview_chars] + "…"
        return _with_content(message, f"[早期回答] {body}")

    if isinstance(message, ToolMessage):
        body = text if len(text) <= preview_chars else text[:preview_chars] + "…"
        name = getattr(message, "name", "tool")
        return _with_content(message, f"[早期工具结果已压缩] {name}: {body}")

    body = text if len(text) <= preview_chars else text[:preview_chars] + "…"
    return _with_content(message, f"[早期消息] {body}")


def compress_messages(
    messages: List[AnyMessage],
    max_chars: int = 12000,
    keep_last: int = 6,
    preview_chars: int = 120,
) -> Tuple[List[AnyMessage], Dict[str, Any]]:
    """把历史消息压到字符预算内。

    返回 (compressed_messages, stats)。stats 可直接写入 trace：
    before_chars / after_chars / compacted_messages / kept_messages / compressed。
    """
    messages = list(messages)
    before = sum(_message_chars(message) for message in messages)
    stats: Dict[str, Any] = {
        "compressed": False,
        "before_chars": before,
        "after_chars": before,
        "compacted_messages": 0,
        "kept_messages": len(messages),
        "total_messages": len(messages),
        "max_chars": max_chars,
    }

    if before <= max_chars or len(messages) <= keep_last + 1:
        return messages, stats

    head = messages[:1]                 # 第一条通常是用户任务，必须保留
    tail = messages[-keep_last:]        # 最近 N 条原样保留
    middle = messages[1:-keep_last]

    # 先按 preview_chars 压缩；若仍超预算，再逐步收紧 preview（只改内容、不删消息）
    for preview in (preview_chars, max(20, preview_chars // 3), 0):
        compacted = head + [_compact(message, preview) for message in middle] + tail
        after = sum(_message_chars(message) for message in compacted)
        if after <= max_chars or preview == 0:
            stats.update(
                {
                    "compressed": True,
                    "after_chars": after,
                    "compacted_messages": len(middle),
                    "kept_messages": len(head) + len(tail),
                    "total_messages": len(compacted),
                    "preview_chars": preview,
                }
            )
            return compacted, stats

    # 理论上到不了这里（preview=0 时已返回），保底返回
    return messages, stats
