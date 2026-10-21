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
    - `expect_memory_content_contains` / `expect_memory_absent` / `expect_memory_wrong_not_used`：
      跨任务长期记忆的三类标注（**全部可选**，缺字段 = 「不检查该项」，不会把老用例判成失败）：
      · `expect_memory_content_contains` —— **该记**：期望长期记忆里出现的内容片段；
      · `expect_memory_absent`          —— **不该记**：期望记忆里**不**出现的片段（一次性凭据 / 临时指令 / 闲聊）；
      · `expect_memory_wrong_not_used`   —— **召回错**：记忆里存在但**不应被当事实采信**的内容，
        期望最终回答**不依赖**它（允许「提到了但已更正」，见 §score_memory）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

# ---------------- 记忆评测：常量 ----------------

# 记忆工具名（agent/memory.py 里注册的两个）
MEMORY_WRITE_TOOL = "remember_fact"
MEMORY_READ_TOOL = "recall_memory"

# 「提到了但已更正」的标记词。
# ★ 为什么需要它：判断「错误记忆有没有被当事实用」不能只看「答案里有没有出现那段文字」——
#   正确行为可能是**指出它过时**（「记忆里是 D:/old-project，但你这次给的是 D:/work/new-repo，
#   以本次为准」），这句话里确实出现了旧内容，却**没有拿它当事实**。
#   所以：旧内容出现的**同一句**里带这些词 → 记为「提到但已更正」（不算违规）。
#   ★ 局限：这是词表启发式，会有假阴性（模型用别的措辞说「以你这次为准」），
#   也可能有假阳性（模型说「以…为准」但其实在拿旧值当事实）—— 指标连着分母一起看。
_CORRECTION_MARKERS = (
    "以本次",
    "以这次",
    "以当前",
    "以任务",
    "为准",
    "已过时",
    "过时",
    "过期",
    "不再",
    "更正",
    "更新",
    "改成",
    "而不是",
    "冲突",
    "不一致",
    "忽略",
    "作废",
    "失效",
)

# 句子分隔符：用来判断「更正标记」与「旧内容」是否落在同一句里
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]")


@dataclass
class MemoryState:
    """一次运行里「记忆」这一侧的可观测状态。

    ★ 关键决定：**记忆状态从 trace + 记忆库快照来，不读运行时的内存对象**。
      这样「已有 trace 不用重跑也能重算记忆指标」，与工具指标同一套口径（可回放、可复盘）。
      证据有四路：
      ① `memory_injected` 事件（runner 在任务启动时写）：注入了几条、命中哪些 key；
         ★ 注意：这条事件**只记 key、不记内容**（注入文本本身在 system prompt 里，不落 trace），
         所以「自动注入的内容」无法离线复原——这是已知边界，README §5.9.5 里写明；
      ② `thought` / `action` 事件里 `remember_fact` 的调用参数 → **写了什么**；
      ③ `observation` 事件里 `remember_fact` / `recall_memory` 的结果 → **写成功没有 / 召回到了什么**；
      ④ 可选的**记忆库 JSONL 快照**（跑完之后的长期记忆里到底有什么）→ `store_contents`。
    """

    # 真正写进记忆的内容（trace 里 remember_fact 成功的那几次）
    written_contents: List[str] = field(default_factory=list)
    # 发起过 remember_fact 的内容（**可能写失败**，例如被「疑似密钥」拦下）
    attempted_contents: List[str] = field(default_factory=list)
    written_keys: List[str] = field(default_factory=list)
    # 本次显式召回到的内容（recall_memory 的返回）
    recalled_contents: List[str] = field(default_factory=list)
    # 任务启动时自动注入的记忆 key（内容不在 trace 里）
    injected_keys: List[str] = field(default_factory=list)
    injected_count: int = 0
    # 记忆库当前内容（可选，来自 JSONL 快照）
    store_contents: List[str] = field(default_factory=list)
    store_keys: List[str] = field(default_factory=list)
    # 证据来源标记：trace / store / trace+store
    source: str = ""

    @property
    def corpus(self) -> List[str]:
        """本次运行可见的「记忆语料」：写进去的 + 记忆库里有的 + 召回到的。"""
        return _dedup(self.written_contents + self.store_contents + self.recalled_contents)

    def to_dict(self) -> Dict[str, Any]:
        """可 JSON 序列化的视图（评测结果落盘用）。"""
        return {
            "written_contents": list(self.written_contents),
            "attempted_contents": list(self.attempted_contents),
            "written_keys": list(self.written_keys),
            "recalled_contents": list(self.recalled_contents),
            "injected_keys": list(self.injected_keys),
            "injected_count": self.injected_count,
            "store_contents": list(self.store_contents),
            "store_keys": list(self.store_keys),
            "source": self.source,
        }


def _dedup(items: Iterable[str]) -> List[str]:
    """去重并保序（空串丢掉）。"""
    seen = set()
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


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
    # 记忆这一侧的可观测状态（trace + 记忆库快照重算；None = 没有记忆证据 → 不进记忆指标分母）
    memory_state: MemoryState | None = None


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


# ---------------- 记忆 ground truth（向后兼容：缺字段 = 不检查该项）----------------


def expected_memory_content_contains(query: Dict[str, Any] | None) -> List[str]:
    """**该记**：期望长期记忆里出现的内容片段。"""
    return _as_list((query or {}).get("expect_memory_content_contains"))


def expected_memory_absent(query: Dict[str, Any] | None) -> List[str]:
    """**不该记**：期望记忆里**不**出现的内容片段。"""
    return _as_list((query or {}).get("expect_memory_absent"))


def expected_memory_wrong_not_used(query: Dict[str, Any] | None) -> List[str]:
    """**召回错**：记忆里存在但不应被当事实采信的内容（期望最终回答不依赖它）。"""
    return _as_list((query or {}).get("expect_memory_wrong_not_used"))


def has_memory_ground_truth(query: Dict[str, Any] | None) -> bool:
    """这条 query 是否**显式标注**了任何记忆期望（区分「没标注」与「标注了空列表」）。"""
    query = query or {}
    return any(
        key in query
        for key in (
            "expect_memory_content_contains",
            "expect_memory_absent",
            "expect_memory_wrong_not_used",
        )
    )


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


def score_answer(
    query: Dict[str, Any] | None,
    answer: str,
    memory_state: MemoryState | None = None,
) -> Dict[str, Any]:
    """按 ground truth 给一条答案打分（纯函数，可离线单测）。

    返回：
        must_hits / must_missed：命中与未命中的要点；
        must_not_violated：命中的「不该出现」词；
        recall：要点命中率（命中数 / 要点数，无要点时为 None）；
        full_hit：要点是否全部命中；
        hallucinated：是否命中了「不该出现」的词。
        memory：**给了 memory_state 时**额外挂上记忆打分结果（见 score_memory）；
            不给就不出现这个键（老调用方拿到的字典形状逐字不变）。
    """
    must = answer_must_contain(query)
    must_not = answer_must_not_contain(query)
    hits = [kw for kw in must if match_keyword(answer, kw)]
    missed = [kw for kw in must if kw not in hits]
    violated = [kw for kw in must_not if match_keyword(answer, kw)]
    result = {
        "must_hits": hits,
        "must_missed": missed,
        "must_not_violated": violated,
        "recall": (len(hits) / len(must)) if must else None,
        "full_hit": bool(must) and not missed,
        "hallucinated": bool(violated),
    }
    if memory_state is not None:
        result["memory"] = score_memory(query, answer, memory_state)
    return result


# ---------------- 记忆状态：从 trace / 记忆库快照重算 ----------------


def _parse_json_maybe(text: Any) -> Any:
    """尽力把一段文本解析成 JSON；解析不了返回 None（不抛异常）。"""
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# 结果预览可能被截断（工具结果限长 1200 字符），所以再兜一层正则抠 content 字段
_CONTENT_RE = re.compile(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _remember_result_key(preview: Any) -> str:
    """从 remember_fact 的返回里取 key（拿不到就返回空串）。"""
    data = _parse_json_maybe(preview)
    if isinstance(data, dict):
        return str(data.get("key") or "")
    return ""


def _recall_result_contents(preview: Any) -> List[str]:
    """从 recall_memory 的返回里取回召回到的内容。

    正常走 JSON 解析；预览被截断导致解析不了时，退化成正则抠 `"content"` 字段
    （尽力而为：抠不全就少判几条，不编内容）。
    """
    data = _parse_json_maybe(preview)
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return [
                str(item.get("content") or "").strip()
                for item in results
                if isinstance(item, dict) and str(item.get("content") or "").strip()
            ]
        return []
    if isinstance(preview, str):
        return [m.group(1) for m in _CONTENT_RE.finditer(preview) if m.group(1).strip()]
    return []


def _match_writes(
    calls: List[Tuple[str, str]],
    ok_keys: List[str],
) -> List[Tuple[str, str]]:
    """把「写成功的 observation」与「发起过的 remember_fact 调用」配对。

    - 先按 **key 精确配对**（调用显式给了 key 时两边一致）；
    - 剩下按**顺序**兜底配对（key 是存储层自动生成的 `mem_xxx`，调用参数里看不到）。

    ★ 为什么要配对：只有「写成功」的那次才算真的进了长期记忆；
      写失败（例如内容形似密钥被工具拒了）的内容**不算**「记住了」。
    """
    matched: List[Tuple[str, str]] = []
    used: set[int] = set()
    for key in ok_keys:
        if not key:
            continue
        for idx, (content, call_key) in enumerate(calls):
            if idx in used:
                continue
            if call_key and call_key == key:
                matched.append((content, key))
                used.add(idx)
                break
    matched_keys = {key for _, key in matched}
    spare = [key for key in ok_keys if key and key not in matched_keys]
    for key in spare:
        for idx, (content, _call_key) in enumerate(calls):
            if idx in used:
                continue
            matched.append((content, key))
            used.add(idx)
            break
    return matched


def memory_state_from_trace(trace_path: str | Path) -> MemoryState:
    """从一次运行的 JSONL trace 里重算「记忆状态」。

    证据来源（全部来自 trace，**不用重跑**也能重算）：
    - `thought` / `action` 事件里 `remember_fact` 的调用参数 → 写了什么；
    - `observation` 事件里 `remember_fact` 且 `ok=True` → 写成功没有（返回里带 key）；
    - `observation` 事件里 `recall_memory` 且 `ok=True` → 召回到了什么内容；
    - `memory_injected` 事件 → 任务启动时自动注入了几条、命中哪些 key。
    """
    state = MemoryState(source="trace")
    path = Path(trace_path)
    if not path.exists():
        return state

    calls: List[Tuple[str, str]] = []
    ok_keys: List[str] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            event_type = ev.get("type")
            if event_type in ("thought", "action"):
                for call in ev.get("tool_calls") or []:
                    if not isinstance(call, dict) or call.get("name") != MEMORY_WRITE_TOOL:
                        continue
                    args = call.get("args") or {}
                    content = str(args.get("content") or "").strip()
                    if not content:
                        continue
                    calls.append((content, str(args.get("key") or "").strip()))
                    state.attempted_contents.append(content)
            elif event_type == "observation":
                tool = ev.get("tool")
                if tool == MEMORY_WRITE_TOOL and ev.get("ok"):
                    ok_keys.append(_remember_result_key(ev.get("result_preview")))
                elif tool == MEMORY_READ_TOOL and ev.get("ok"):
                    state.recalled_contents.extend(
                        _recall_result_contents(ev.get("result_preview"))
                    )
            elif event_type == "memory_injected":
                state.injected_count = int(ev.get("count") or 0)
                state.injected_keys.extend(str(k) for k in (ev.get("keys") or []))

    matched = _match_writes(calls, ok_keys)
    state.written_contents = [content for content, _ in matched]
    state.written_keys = _dedup(
        [key for _, key in matched if key] + [key for key in ok_keys if key]
    )
    state.attempted_contents = _dedup(state.attempted_contents)
    state.recalled_contents = _dedup(state.recalled_contents)
    state.injected_keys = _dedup(state.injected_keys)
    return state


def memory_state_from_store(path: str | Path) -> MemoryState:
    """从记忆库 JSONL 快照里读「长期记忆里现在到底有什么」。

    这是**记忆内容的权威口径**（长期记忆是落盘的数据，不该只靠 trace 反推）。
    读的是快照，不改任何东西。
    """
    state = MemoryState(source="store")
    file_path = Path(path)
    if not file_path.exists():
        return state
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        content = str(data.get("content") or "").strip()
        if not content:
            continue
        state.store_contents.append(content)
        state.store_keys.append(str(data.get("key") or ""))
    state.store_contents = _dedup(state.store_contents)
    state.store_keys = _dedup(state.store_keys)
    return state


def merge_memory_states(*states: MemoryState | None) -> MemoryState:
    """把多个记忆状态合并成一个（trace 的 + 记忆库快照的）。"""
    merged = MemoryState()
    sources: List[str] = []
    for state in states:
        if state is None:
            continue
        merged.written_contents.extend(state.written_contents)
        merged.attempted_contents.extend(state.attempted_contents)
        merged.written_keys.extend(state.written_keys)
        merged.recalled_contents.extend(state.recalled_contents)
        merged.injected_keys.extend(state.injected_keys)
        merged.injected_count += state.injected_count
        merged.store_contents.extend(state.store_contents)
        merged.store_keys.extend(state.store_keys)
        if state.source and state.source not in sources:
            sources.append(state.source)
    merged.written_contents = _dedup(merged.written_contents)
    merged.attempted_contents = _dedup(merged.attempted_contents)
    merged.written_keys = _dedup(merged.written_keys)
    merged.recalled_contents = _dedup(merged.recalled_contents)
    merged.injected_keys = _dedup(merged.injected_keys)
    merged.store_contents = _dedup(merged.store_contents)
    merged.store_keys = _dedup(merged.store_keys)
    merged.source = "+".join(sources)
    return merged


# ---------------- 记忆打分 ----------------


def _match_any(texts: Iterable[str], fragment: str) -> bool:
    """片段是否出现在任一文本里（与答案要点同一套 `a|b` / 忽略大小写的匹配口径）。"""
    return any(match_keyword(text, fragment) for text in texts)


def _is_corrected_mention(answer: str, fragment: str) -> bool:
    """答案里出现了「不该采信的旧内容」，但**同一句**里有更正标记 → 记为「提到但已更正」。

    例：记忆里是 D:/old-project，你这次给的是 D:/work/new-repo，以本次为准
        —— 句子里带了「以本次为准」，说明模型是在**指出冲突**，不是拿旧值当事实。
    """
    text = answer or ""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if not match_keyword(sentence, fragment):
            continue
        if any(marker in sentence for marker in _CORRECTION_MARKERS):
            return True
    return False


def score_memory(
    query: Dict[str, Any] | None,
    answer: str,
    memory_state: MemoryState | None,
) -> Dict[str, Any]:
    """按 ground truth 给「记忆行为」打分（纯函数，可离线单测）。

    三类判定：
    1. **该记**（`expect_memory_content_contains`）：片段应出现在记忆语料里
       （本次写进去的 ∪ 记忆库快照里有的）；另外单独报「是不是真的写进去了」。
    2. **不该记**（`expect_memory_absent`）：片段**不应**出现在记忆语料里
       （写进去的 ∪ 记忆库里的 ∪ 召回到的）；若只是**发起过**但没写进去（如被密钥规则拦下），
       记在 `absent_attempted` 里（信息项，不算违规）。
    3. **召回错**（`expect_memory_wrong_not_used`）：片段是「记忆里有但不该采信」的内容。
       - 记忆里**根本没有**它 → 陷阱没踩到，**不计分**（记在 `wrong_not_present`）；
       - 答案里**没**用它 → 合格；
       - 答案里用了它，但同一句有更正标记 → 记为「提到但已更正」，合格；
       - 答案里直接拿它当事实 → 违规（`wrong_used`）。

    ★ 记忆状态为 None（没有 trace / 没有记忆库快照）→ `checked=False`，
      所有判定都不做（老用例缺新字段时不会被判成失败）。
    """
    contains = expected_memory_content_contains(query)
    absent = expected_memory_absent(query)
    wrong = expected_memory_wrong_not_used(query)

    result: Dict[str, Any] = {
        "checked": False,
        "remember_hits": [],
        "remember_missed": [],
        "write_hits": [],
        "write_missed": [],
        "absent_violations": [],
        "absent_attempted": [],
        "wrong_used": [],
        "wrong_present": [],
        "wrong_mentioned_corrected": [],
        "wrong_not_present": [],
        "remember_ok": True,
        "write_ok": True,
        "absent_ok": True,
        "wrong_ok": True,
        "pass": None,
    }
    if memory_state is None:
        return result

    result["checked"] = True
    corpus = memory_state.corpus
    written = memory_state.written_contents

    for fragment in contains:
        if _match_any(corpus, fragment):
            result["remember_hits"].append(fragment)
        else:
            result["remember_missed"].append(fragment)
        if _match_any(written, fragment):
            result["write_hits"].append(fragment)
        else:
            result["write_missed"].append(fragment)
    result["remember_ok"] = not result["remember_missed"]
    result["write_ok"] = not result["write_missed"]

    for fragment in absent:
        if _match_any(corpus, fragment):
            result["absent_violations"].append(fragment)
        elif _match_any(memory_state.attempted_contents, fragment):
            # 发起过但没写进去（如被「疑似密钥」规则拒了）→ 信息项，不算违规
            result["absent_attempted"].append(fragment)

    for fragment in wrong:
        if not _match_any(corpus, fragment):
            # 记忆里根本没有这条 → 陷阱没踩到，不进这一项的分母
            result["wrong_not_present"].append(fragment)
            continue
        result["wrong_present"].append(fragment)
        if not match_keyword(answer or "", fragment):
            continue  # 答案里没出现 → 没被当事实用，合格
        if _is_corrected_mention(answer or "", fragment):
            result["wrong_mentioned_corrected"].append(fragment)
        else:
            result["wrong_used"].append(fragment)

    result["absent_ok"] = not result["absent_violations"]
    result["wrong_ok"] = not result["wrong_used"]
    result["pass"] = bool(
        result["remember_ok"] and result["absent_ok"] and result["wrong_ok"]
    )
    return result


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

    # ---------------- 记忆评测：该记 / 不该记 / 召回错 ----------------
    # 分母口径与其它指标一致：**只统计真正完成的运行**（status=completed），
    # 并且必须**有记忆状态**（trace / 记忆库快照）才判；
    # 标了记忆 ground truth 却没进分母的运行（失败的运行、或没取到记忆状态的），
    # 单独报 `memory_not_scored_n`——既不进分子也不进分母（不装作「检查过且通过」）。
    memory_labeled = [r for r in records if has_memory_ground_truth(r.query)]
    memory_scored = [
        r for r in memory_labeled
        if r.memory_state is not None and r.status == "completed"
    ]
    memory_remember_hits = 0
    memory_remember_total = 0
    memory_write_hits = 0
    memory_absent_violations = 0
    memory_absent_total = 0
    memory_absent_attempted = 0
    memory_wrong_used = 0
    memory_wrong_present = 0
    memory_pass = 0
    memory_wrong_used_ids: List[Dict[str, Any]] = []
    memory_failures: List[Dict[str, Any]] = []
    for r in memory_scored:
        s = score_memory(r.query, r.final_answer, r.memory_state)
        memory_remember_hits += len(s["remember_hits"])
        memory_remember_total += len(s["remember_hits"]) + len(s["remember_missed"])
        memory_write_hits += len(s["write_hits"])
        memory_absent_violations += len(s["absent_violations"])
        memory_absent_total += len(expected_memory_absent(r.query))
        memory_absent_attempted += len(s["absent_attempted"])
        # 召回错：只对「记忆里确实存在」的错误内容计分（陷阱没踩到就不算分）
        memory_wrong_present += len(s["wrong_present"])
        memory_wrong_used += len(s["wrong_used"])
        if s["wrong_used"]:
            memory_wrong_used_ids.append({
                "id": r.query.get("id"),
                "used": s["wrong_used"],
            })
        if s["pass"]:
            memory_pass += 1
        else:
            memory_failures.append({
                "id": r.query.get("id"),
                "remember_missed": s["remember_missed"],
                "absent_violations": s["absent_violations"],
                "wrong_used": s["wrong_used"],
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
        # 记忆评测：该记 / 不该记 / 召回错（分母都只算「完成且有记忆状态」的运行）
        "memory_labeled_n": len(memory_labeled),
        "memory_scored_n": len(memory_scored),
        # 标了记忆标注但没进分母的运行数（失败运行 + 没取到记忆状态的）
        "memory_not_scored_n": len(memory_labeled) - len(memory_scored),
        # 该记：期望出现在记忆里的片段，命中了多少
        "memory_remember_rate": _rate(memory_remember_hits, memory_remember_total),
        "memory_remember_n": memory_remember_total,
        "memory_remember_hits": memory_remember_hits,
        # 该记：其中**真的写进了记忆**（remember_fact 写成功）的比例
        "memory_write_rate": _rate(memory_write_hits, memory_remember_total),
        "memory_write_n": memory_remember_total,
        "memory_write_hits": memory_write_hits,
        # 不该记：期望不出现却出现在记忆里的片段占比（越低越好）
        "memory_absent_violation_rate": _rate(memory_absent_violations, memory_absent_total),
        "memory_absent_n": memory_absent_total,
        "memory_absent_violations": memory_absent_violations,
        "memory_absent_attempted_n": memory_absent_attempted,
        # 召回错：记忆里确实存在的错误内容里，没被当事实用的比例
        "memory_wrong_not_used_rate": _rate(
            memory_wrong_present - memory_wrong_used, memory_wrong_present
        ),
        "memory_wrong_present_n": memory_wrong_present,
        "memory_wrong_used_n": memory_wrong_used,
        "memory_wrong_used": memory_wrong_used_ids,
        # 三项全过
        "memory_pass_rate": _rate(memory_pass, len(memory_scored)),
        "memory_pass_n": len(memory_scored),
        "memory_failures": memory_failures,
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
        # 记忆评测（分母为 0 时报 None = 算不出来，不报 0）
        lines.append(
            f"- 记忆「该记」命中率（期望出现在长期记忆里的片段）："
            f"**{summary.get('memory_remember_rate')}**"
            f"（n={summary.get('memory_remember_n')} 片段；"
            f"其中真正写进记忆的比例 **{summary.get('memory_write_rate')}**）"
        )
        lines.append(
            f"- 记忆「不该记」违规率（不该进记忆却进了的片段占比）："
            f"**{summary.get('memory_absent_violation_rate')}**"
            f"（n={summary.get('memory_absent_n')} 片段）"
        )
        lines.append(
            f"- 记忆「召回错」未采信率（记忆里有但没被当事实用的比例）："
            f"**{summary.get('memory_wrong_not_used_rate')}**"
            f"（n={summary.get('memory_wrong_present_n')} 片段，"
            f"其中被当事实用了 {summary.get('memory_wrong_used_n')} 个）"
        )
        lines.append(
            f"- 记忆三项全过率： **{summary.get('memory_pass_rate')}**"
            f"（n={summary.get('memory_pass_n')} 条记忆用例；"
            f"标了记忆标注但没进分母的 "
            f"{summary.get('memory_not_scored_n')} 条不计分）"
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
