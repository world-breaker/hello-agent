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

★ ground truth 字段（v2 评测集，40+ 条）：
    每条 query 可以带下面这些标注（**全部可选**，旧结构的老数据照样能算）：
    - `expect_tools` / `expected_tools`：期望用到的工具名列表。
      **空列表 = 这一问不该用工具**（模型该直接答或该澄清），与「没标注」是两回事；
      两个字段名都认（`expect_tools` 是原始字段，`expected_tools` 是新别名）。
    - `expected_refusal`：这一问模型**是否应当承认做不到 / 不知道**（领域外问题必须为 true）。
    - `answer_must_contain`：答案里必须出现的要点（1~3 个）。
      每一项可以用 `a|b|c` 写多个可接受写法（任一命中即算这条要点命中）——
      这样同一份 ground truth 能容忍「没有相关内容 / 查不到 / 未找到」等不同措辞。
    - `answer_must_not_contain`：不该出现的词（抓幻觉用，可空）。
      只放「**断言形状**」的词（如「收盘价为」「晴转多云」），不要放会被原样复述的题面词。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

# 领域外问题（知识库里没有）的「诚实否认」关键词：命中即视为没有编造
_HONESTY_MARKERS = ("没有", "查不到", "无法回答", "不知道", "没找到", "不涉及", "没有相关内容", "未找到")

# 「该拒答 / 承认做不到」的标记词（给 refusal_accuracy 用）。
# ★ 为什么要与 _HONESTY_MARKERS 分开：诚实率是老口径（宽松，含裸「没有」），
#   而「检索成功但片段没提到某个细节」的成功回答里也常出现「没有」——
#   若共用宽松词表，成功回答会被误判成「拒答」，false-positive 指标就成了假红。
#   这里只收「明确承认做不到」的措辞，仍然是启发式（README §5.9.3 有局限说明）。
_REFUSAL_MARKERS = (
    "无法回答",
    "不能回答",
    "无法提供",
    "不能提供",
    "无法确定",
    "无法判断",
    "无法计算",
    "无法获取",
    "查不到",
    "检索不到",
    "未找到",
    "没有找到",
    "没有相关内容",
    "没有关于",
    "知识库里没有",
    "知识库中没有",
    "不知道",
    "抱歉",
    "超出",
    "不涉及",
    "失败",
)

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


# ---------------- ground truth 读取（向后兼容：老条目只有 id/category/task/expect_tools）----------------


def _as_list(value: Any) -> List[str]:
    """把标注字段规整成字符串列表（None → 空列表，标量 → 单元素列表）。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def expected_tools(query: Dict[str, Any] | None) -> List[str]:
    """期望用到的工具名（`expect_tools` 原始字段 / `expected_tools` 新别名都认）。"""
    query = query or {}
    tools = query.get("expect_tools")
    if tools is None:
        tools = query.get("expected_tools")
    return _as_list(tools)


def has_tool_ground_truth(query: Dict[str, Any] | None) -> bool:
    """这条 query 是否**显式标注**了工具期望（区分「没标注」与「标注为空列表」）。"""
    query = query or {}
    return "expect_tools" in query or "expected_tools" in query


def expected_refusal(query: Dict[str, Any] | None) -> bool:
    """这一问是否应当「承认做不到 / 不知道」（领域外问题必须为 true）。"""
    query = query or {}
    value = query.get("expected_refusal")
    if value is None:
        value = query.get("expect_refusal")
    return bool(value)


def answer_must_contain(query: Dict[str, Any] | None) -> List[str]:
    """答案里必须出现的要点。"""
    return _as_list((query or {}).get("answer_must_contain"))


def answer_must_not_contain(query: Dict[str, Any] | None) -> List[str]:
    """答案里不该出现的词（抓幻觉）。"""
    return _as_list((query or {}).get("answer_must_not_contain"))


def match_keyword(answer: str, keyword: str) -> bool:
    """关键词命中判定。

    一项要点可以用 `a|b|c` 写多个可接受写法（任一命中即算这条要点命中），
    这样同一份 ground truth 能容忍「查不到 / 没有相关内容 / 未找到」等不同措辞。
    比较时忽略大小写（英文关键词如 chunk_size 常被写成 CHUNK_SIZE）。
    """
    text = (answer or "").lower()
    for alt in str(keyword).split("|"):
        alt = alt.strip()
        if alt and alt.lower() in text:
            return True
    return False


def score_answer(query: Dict[str, Any] | None, answer: str) -> Dict[str, Any]:
    """按 ground truth 给一条答案打分（纯函数，可离线单测）。

    返回：
        must_hits / must_missed：命中与未命中的要点；
        must_not_violated：命中的「不该出现」词；
        recall：要点命中率（命中数 / 要点数，无要点时为 None）；
        full_hit：要点是否全部命中；
        hallucinated：是否命中了「不该出现」的词。
    """
    must = answer_must_contain(query)
    must_not = answer_must_not_contain(query)
    hits = [kw for kw in must if match_keyword(answer, kw)]
    missed = [kw for kw in must if kw not in hits]
    violated = [kw for kw in must_not if match_keyword(answer, kw)]
    return {
        "must_hits": hits,
        "must_missed": missed,
        "must_not_violated": violated,
        "recall": (len(hits) / len(must)) if must else None,
        "full_hit": bool(must) and not missed,
        "hallucinated": bool(violated),
    }


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
    """纯函数：把若干条 EvalRecord 聚合成作品集级指标。离线可单测。

    ★ 每个比率型指标都带一个 `*_n` 分母——只报比率不报分母的指标没法判断可信度。
    """
    n = len(records)
    if n == 0:
        return {"n": 0}

    completed = [r for r in records if r.status == "completed"]

    # 工具选择：两个口径都报出来，别让「发起过」冒充「用上了」
    # - tool_selection_accuracy（严）：期望工具 ⊆ **成功执行过**的工具（observation ok=True）
    # - tool_selection_attempt_rate（宽）：期望工具 ⊆ **发起过**的工具（含执行失败）
    #   模型「试了 search_knowledge 但工具全失败、改用记忆硬答」时，宽口径算它选对了，
    #   严口径不算 —— 两个数一起看才知道指标虚不虚。
    labeled = [r for r in records if expected_tools(r.query)]
    sel_ok_strict = [
        r for r in labeled
        if set(expected_tools(r.query)) <= set(r.used_tools_ok)
    ]
    sel_ok_attempt = [
        r for r in labeled
        if set(expected_tools(r.query)) <= set(r.used_tools)
    ]

    # 「不该用工具」的题（显式标注了空列表）：模型真的用了工具 = 违规
    no_tool_expected = [
        r for r in records
        if has_tool_ground_truth(r.query) and not expected_tools(r.query)
    ]
    no_tool_violations = [r for r in no_tool_expected if r.used_tools_ok]

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

    # ---------------- 拒答 / 承认不知道（expected_refusal）----------------
    # 口径与诚实率完全一致：
    #   分母 = **真正完成**（status=completed）的运行（失败运行既不进分子也不进分母）；
    #   分子 = 其中「回答来自模型」且带拒答措辞的。
    # ★ 为什么分子还要看来源：模型调用失败时 graph 的兜底文案含「没有」这类否认词，
    #   把它算成「诚实拒答」就是假绿（见 §5.9.1 同一套修法）。
    refusal_expected = [r for r in records if expected_refusal(r.query)]
    refusal_scored = [r for r in refusal_expected if r.status == "completed"]
    refused = [
        r for r in refusal_scored
        if r.final_answer_source == ANSWER_SOURCE_MODEL and is_refusal(r.final_answer)
    ]

    # 反向检查：不该拒答的题却拒答了（词表启发式，有假阳性风险，README 有说明）
    no_refusal_expected = [r for r in records if not expected_refusal(r.query)]
    no_refusal_scored = [r for r in no_refusal_expected if r.status == "completed"]
    false_refusals = [
        r for r in no_refusal_scored
        if r.final_answer_source == ANSWER_SOURCE_MODEL and is_refusal(r.final_answer)
    ]

    # ---------------- 答案要点命中率 / 幻觉率 ----------------
    # 分母同样只看「真正完成」的运行（失败运行的空答案不该拉低要点命中率）；
    # 分子侧逐条按 ground truth 打分（score_answer）。
    keyword_scored = [r for r in completed if answer_must_contain(r.query)]
    keyword_hits = 0
    keyword_total = 0
    full_hit = 0
    per_query_recall: List[float] = []
    for r in keyword_scored:
        s = score_answer(r.query, r.final_answer)
        keyword_hits += len(s["must_hits"])
        keyword_total += len(answer_must_contain(r.query))
        if s["full_hit"]:
            full_hit += 1
        if s["recall"] is not None:
            per_query_recall.append(s["recall"])

    hallucination_scored = [r for r in completed if answer_must_not_contain(r.query)]
    hallucination_hits = []
    for r in hallucination_scored:
        s = score_answer(r.query, r.final_answer)
        if s["hallucinated"]:
            hallucination_hits.append({
                "id": r.query.get("id"),
                "violated": s["must_not_violated"],
            })

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
        # 工具选择（严 / 宽两套口径，分母相同）
        "tool_selection_accuracy": _rate(len(sel_ok_strict), len(labeled)),
        "tool_selection_attempt_rate": _rate(len(sel_ok_attempt), len(labeled)),
        "tool_selection_n": len(labeled),
        "tool_selection_accuracy_n": len(labeled),
        "tool_selection_attempt_rate_n": len(labeled),
        # 「不该用工具」的题：模型用了工具 = 违规
        "no_tool_expected_n": len(no_tool_expected),
        "no_tool_violation_rate": _rate(len(no_tool_violations), len(no_tool_expected)),
        "failure_recovery_rate": _rate(len(recovered), len(had_failure)),
        # 领域外诚实率（老口径，保留）
        "out_of_domain_honesty_rate": _rate(len(honest), len(ood_completed)),
        "out_of_domain_n": len(ood),
        "out_of_domain_completed_n": len(ood_completed),
        # 拒答 / 承认不知道（新口径，可跨 category 用）
        "refusal_accuracy": _rate(len(refused), len(refusal_scored)),
        "refusal_accuracy_n": len(refusal_scored),
        "refusal_expected_n": len(refusal_expected),
        "refusal_completed_n": len(refusal_scored),
        "refusal_false_positive_rate": _rate(len(false_refusals), len(no_refusal_scored)),
        "refusal_false_positive_n": len(no_refusal_scored),
        # 答案要点命中率 / 幻觉率
        "answer_keyword_recall": _rate(keyword_hits, keyword_total),
        "answer_keyword_recall_n": len(keyword_scored),
        "answer_keyword_hits": keyword_hits,
        "answer_keyword_total": keyword_total,
        "answer_keyword_full_hit_rate": _rate(full_hit, len(keyword_scored)),
        "answer_keyword_full_hit_n": len(keyword_scored),
        "hallucination_rate": _rate(len(hallucination_hits), len(hallucination_scored)),
        "hallucination_n": len(hallucination_scored),
        "hallucination_hits": hallucination_hits,
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


def is_refusal(answer: str) -> bool:
    """判断一段回答是不是「承认做不到 / 不知道」（词表启发式）。

    ★ 局限（README §5.9.3 有同款说明）：这是**关键词启发式**，不是语义判定——
      成功回答里出现「无法确认具体数值」这类措辞会被算成拒答（假阳性）。
      所以指标一定连着分母一起看，不单看比率。
    """
    text = answer or ""
    return any(marker in text for marker in _REFUSAL_MARKERS)


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
            f"（n={summary.get('tool_selection_attempt_rate_n')}）"
        )
        lines.append(
            f"- 「不该用工具」的题里真的用了工具的比例：**{summary.get('no_tool_violation_rate')}**"
            f"（n={summary.get('no_tool_expected_n')}）"
        )
        lines.append(f"- 失败恢复率（重试/降级后仍完成）：**{summary['failure_recovery_rate']}**")
        lines.append(
            f"- 领域外诚实率（只统计真正完成的运行）：**{summary['out_of_domain_honesty_rate']}**"
            f"（完成 **{summary.get('out_of_domain_completed_n')}** / 领域外共 "
            f"**{summary.get('out_of_domain_n')}** 条）"
        )
        lines.append(
            f"- 拒答准确率（该承认做不到时承认了，只统计完成的运行）："
            f"**{summary.get('refusal_accuracy')}**"
            f"（n={summary.get('refusal_accuracy_n')}；标注为应拒答共 "
            f"**{summary.get('refusal_expected_n')}** 条）"
        )
        lines.append(
            f"- 不该拒答却拒答的比例：**{summary.get('refusal_false_positive_rate')}**"
            f"（n={summary.get('refusal_false_positive_n')}）"
        )
        lines.append(
            f"- 答案要点命中率：**{summary.get('answer_keyword_recall')}**"
            f"（n={summary.get('answer_keyword_recall_n')} 条，"
            f"命中 {summary.get('answer_keyword_hits')} / {summary.get('answer_keyword_total')} 个要点；"
            f"全中比例 **{summary.get('answer_keyword_full_hit_rate')}**"
            f"（n={summary.get('answer_keyword_full_hit_n')}））"
        )
        lines.append(
            f"- 幻觉率（命中「不该出现」词的运行占比）：**{summary.get('hallucination_rate')}**"
            f"（n={summary.get('hallucination_n')}）"
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
