"""上下文忠实的**脚本化假模型**：给「上下文压缩代价」对照实验当被测对象。

★ 为什么不能直接用 `tests/fake_llm.py` 那个「固定脚本」假模型：
    压缩实验要回答的是「信息被压掉之后，Agent 还做得对不对」。
    如果模型无视上下文、每次都吐同一段固定答案，那么压缩与否答案完全一样 →
    测出来的差异永远是 0，实验就没有意义。
    所以这里的假模型按 ReAct 策略**读上下文**：

    1. 看得到哪些工具已经返回结果 → 决定下一步该调哪个工具（计划只由**任务文本**决定，
       任务文本永远在第一条消息里、永远不被压缩 → 各组完全一致，是受控变量）；
    2. 工具参数要用到**检索结果里的数字** → 检索结果被压掉就取不到值（= 参数被压坏）；
    3. 最终答案只复述**当前看得见的**观察内容 → 观察被压短，答案要点就丢。

    ⚠️ 这是**探针式设计**：目的是把「信息损失」放大成可测量的差异，
       **不是**在模拟真实模型的语义能力。本轮实验因此只能证明
       「压缩确实会吃掉上下文里的信息，且吃掉多少是可测的」，
       **不能**证明「真实模型在压缩下语义质量下降多少」。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fake_llm import ScriptedChatModel, tool_call  # noqa: E402  （仓库自带的脚本化假模型）

# 触发哪个工具的关键词（只由任务文本决定 → 各压缩档位完全一致）
SEARCH_TRIGGERS = ("知识库", "检索", "查一下", "查知识", "搜", "资料")
CALC_TRIGGERS = ("算", "计算", "等于", "除以", "次方", "多少个", "多少倍", "几倍", "一半", "多大")
TIME_TRIGGERS = ("时间", "几点", "日期", "星期", "今天", "现在")

# 任务文本里的算式（优先用它当计算器参数）
_EXPR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^()]*\)|[-+*/().\d\s]{3,}")
# 两位以上的整数（用于从检索结果里取「知识库给出的数字」）
# ★ 负向前瞻：排除「0.6565」这种小数里的一段（那不是知识库给的典型值，是相似度分数）
_NUM_RE = re.compile(r"(?<![\d.])\d{2,}")
# 检索结果里的知识库正文（正则版：JSON 被截断也能抓）
_TEXT_FIELD_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _text_of(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", block)) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def plan_tools(task: str) -> List[str]:
    """按任务文本决定要依次用哪些工具。

    ★ 计划只看**任务文本**：任务文本在消息列表第一条，`compress_messages` 永远原样保留，
      所以所有压缩档位拿到的计划完全一样 → 「差异只可能来自上下文被压缩」这条控制变量成立。
    """
    plan: List[str] = []
    if any(k in task for k in SEARCH_TRIGGERS):
        plan.append("search_knowledge")
    if any(k in task for k in TIME_TRIGGERS):
        plan.append("current_time")
    if any(k in task for k in CALC_TRIGGERS):
        plan.append("calculator")
    return plan


def _find_expression(task: str) -> Optional[str]:
    """从任务文本里抠出算式（找不到返回 None）。"""
    best = None
    for match in _EXPR_RE.finditer(task):
        candidate = match.group(0).strip()
        if not candidate:
            continue
        has_digit = any(ch.isdigit() for ch in candidate)
        has_op = any(op in candidate for op in "+-*/^")
        if not (has_digit and has_op):
            continue
        if best is None or len(candidate) > len(best):
            best = candidate
    return best


def _first_number(text: str) -> Optional[int]:
    """取文本里第一个「两位以上的整数」（知识库给的典型值这类数字）。"""
    match = _NUM_RE.search(text or "")
    if not match:
        return None
    return int(match.group(0))


def _topic(task: str) -> str:
    """从任务文本里抽出「检索主题」（短的、稳定的）。

    ★ 为什么要短：检索工具的返回是一个 JSON，**开头会把 query 原样回显**。
      如果把整句任务当 query，压缩后的预览（前 120 字符）里全是任务回显，
      关键词照样命中 → 压缩的代价就测不出来了。主题短一点，
      预览才会伸进知识库正文，**要点才会真的被压掉**。
    """
    text = task.strip()
    for prefix in (
        "先查知识库：", "先查知识库", "查一下知识库里", "查一下知识库", "查知识库里", "查知识库",
        "检索知识库里", "检索知识库", "用知识库里的内容", "用知识库内容", "基于知识库",
        "根据知识库", "用知识库", "知识库里",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    for separator in ("？", "?", "，", ",", "、"):
        if separator in text:
            text = text.split(separator)[0]
            break
    return text.strip()[:40]


def _kb_text(observation: str) -> str:
    """从检索结果里取出**知识库正文**（results[].text）。

    ★ 为什么不用 json.loads：工具输出进上下文前会被截断到 1200 字符
      （agent/resilience.py 的限长）→ JSON 往往是不完整的，解析必失败。
      退而求其次用正则抓 `"text": "..."` 字段——截断的 JSON 也照样抓得到。
    ★ 为什么要剔掉外层 JSON：检索结果里还有 `elapsed_ms: 107.61` 这种**每次都变**的字段，
      直接在整段 JSON 里找数字会捡到它 → 同一批任务两次跑出来的工具参数不一样 →
      「参数被压坏」就成了假阳性。只看正文，结果才是确定的。
    """
    texts = [
        match.group(1)
        for match in _TEXT_FIELD_RE.finditer(observation or "")
    ]
    if texts:
        return "\n".join(texts)
    return observation or ""


class ContextFaithfulFakeModel(ScriptedChatModel):
    """按上下文决策的假模型（继承仓库自带的 ScriptedChatModel）。

    额外能力：
    - `emitted`：本实例发出过的工具调用（含参数），供实验脚本事后核对「参数有没有被压坏」；
    - `usage_metadata`：给出**字符代理**的 token 用量（1 字符 ≈ 1 token，与 agent/context.py
      的保守代理一致）→ token 指标随上下文变短而变少，**不是**真实 tokenizer 的计数。
    """

    responses: List[AIMessage] = []
    cursor: int = 0
    bound_tools: List[str] = []
    calls: int = 0
    emitted: List[Dict[str, Any]] = []

    # ---------- 决策 ----------

    def _task_text(self, messages: List[BaseMessage]) -> str:
        for message in messages:
            if isinstance(message, HumanMessage):
                return _text_of(message)
        return ""

    def _visible_observations(self, messages: List[BaseMessage]) -> List[Tuple[str, str]]:
        """当前上下文里**看得见**的工具结果（压缩后的占位预览也算「看得见」）。"""
        return [
            (str(getattr(m, "name", "")), _text_of(m))
            for m in messages
            if isinstance(m, ToolMessage)
        ]

    def _build_args(
        self,
        name: str,
        task: str,
        observations: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """构造工具参数——**关键点**：计算器的表达式会用「检索结果里的数字」。

        检索结果被压缩 → 那个数字看不见 → 表达式只能退化成占位值
        → **工具调用参数被压坏**（实验脚本会拿它和「不压缩」那组逐条比对）。
        ★ 只从 **search_knowledge** 的结果里取数字：时间工具的结果里有当天年份（2026），
          拿它当「知识库给的数字」会让参数看起来正常，其实已经丢了检索依据。
        """
        if name == "search_knowledge":
            return {"query": _topic(task), "top_k": 3}
        if name == "current_time":
            return {}
        if name == "calculator":
            expr = _find_expression(task)
            if expr:
                return {"expression": expr}
            # 任务里没有现成算式 → 用「任务里的数字」减「检索结果里的数字」
            kb_number = None
            for tool_name, obs in observations:
                if tool_name != "search_knowledge":
                    continue
                kb_number = _first_number(_kb_text(obs))
                if kb_number is not None:
                    break
            task_number = None
            numbers = _NUM_RE.findall(task)
            if numbers:
                task_number = int(numbers[-1])
            if task_number is None:
                task_number = 0
            if kb_number is None:
                # 检索结果里的数字被压没了 → 只能填 0（参数「退化」）
                return {"expression": f"({task_number}-0)"}
            return {"expression": f"({task_number}-{kb_number})"}
        return {}

    def _compose_answer(self, task: str, observations: List[Tuple[str, str]]) -> str:
        """最终答案 = **当前看得见的**观察内容。

        ★ 为什么直接复述观察（而不是「摘要」）：
          假模型没有语义能力，它唯一能做的、也是真实模型唯一能做的，就是
          **用看得见的内容作答**。观察被压短 → 答案就只剩那么短 → 要点自然丢。
          上限取 1200 字符，与工具输出进上下文时的限长一致（agent/resilience.py）。
        """
        if not observations:
            return (
                "为了给出准确答案，请补充具体需求："
                "需要哪一段文案/哪一个产品/哪一年/哪一个方案的具体信息？"
            )
        excerpts = [obs[:1200] for _, obs in observations]
        return "根据工具结果整理如下：\n" + "\n".join(excerpts)

    # ---------- 模型接口 ----------

    def _next_message_for(self, messages: List[BaseMessage]) -> AIMessage:
        task = self._task_text(messages)
        plan = plan_tools(task)
        observations = self._visible_observations(messages)
        done = {
            getattr(m, "name", "") for m in messages if isinstance(m, ToolMessage)
        }
        pending = [tool for tool in plan if tool not in done]
        if pending:
            name = pending[0]
            args = self._build_args(name, task, observations)
            self.emitted.append({"name": name, "args": args})
            return tool_call(name, args, call_id=f"call_{len(self.emitted)}")
        text = self._compose_answer(task, observations)
        return AIMessage(content=text)

    def _finish(self, messages: List[BaseMessage], message: AIMessage) -> ChatResult:
        prompt_chars = sum(len(_text_of(m)) for m in messages)
        completion_chars = len(_text_of(message))
        message.usage_metadata = {
            "input_tokens": prompt_chars,
            "output_tokens": completion_chars,
            "total_tokens": prompt_chars + completion_chars,
        }
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._finish(messages, self._next_message_for(messages))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._finish(messages, self._next_message_for(messages))
