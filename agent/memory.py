"""跨任务长期记忆：JSONL 持久化 + 关键词召回 + 任务启动时自动注入。

为什么需要它？
- 单次 ReAct 的 messages 只在**本次任务**内有效，任务结束就丢了；
- 用户说「记住：以后回答都用中文」「我的项目路径是 …」时，下一次任务应该能自动想起来，
  而不是每次重新交代。

设计取舍（面试可讲）：
- **JSONL 追加写**：一行一条记录，崩溃最多丢最后一行；便于 grep / jq / 人工排查；
- **同 key 后写覆盖**：`remember` 允许带 key，读取时以最后一条为准（天然 upsert，不引入数据库）；
- **召回用 BM25**（英文词 + 中文单字/二元组分词）：记忆条数通常几十~几百，
  零依赖、可解释、离线可跑；BM25 的 idf 让稀有词（项目名/术语）权重更高，
  比简单词重叠更符合「用户提到独特术语时应命中那条记忆」的直觉；
  要语义召回时再接 embedding（与 rag-eval 的混合检索同思路）；
- **只注入命中的记忆**：没有关键词重叠就不注入，避免把无关记忆塞进 prompt 污染上下文；
- **显式工具 + 自动注入**：模型可用 `remember_fact` 主动记，任务开始时系统自动召回 top-k。

记忆的「可信度」管理（v2）：
- **来源 `source`**：这条事实从哪来（用户明说 / 工具返回 / 模型推断），
  写进记录里，召回时能区分；
- **核验 `verified`**：默认 `False`（= 未经核验）。模型可以写入，但**只有显式标成
  `verified=True` 的记录**才能被 `only_verified=True` 的召回选中 ——
  「模型说过」与「这条事实被核对过」是两件事，指标上要能分开看；
- **过期 `ttl_seconds`**：可写的记录带 TTL，过期后**不参与召回**（加载时顺手清掉）；
- **删除 `forget(key)`**：记忆能写就能删（JSONL 是追加写，删除 = 重写文件）。

安全边界：
- 记忆是**用户数据**，不要写入密钥；`remember_fact` 会拒绝包含疑似密钥的内容。
"""

import json
import math
import re
import threading
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, tool

_ASCII_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 简单密钥特征：OPENAI_API_KEY=... / BEGIN PRIVATE KEY —— 记忆是长期落盘的，宁可拒绝也不要泄露
_SECRET_HINTS = ("OPENAI_API_KEY", "API_KEY=", "BEGIN PRIVATE KEY")
# OpenAI 风格密钥的**形状**：`sk-` 后跟一长串字母/数字（真实 key 至少 16 位）。
# ★ 为什么用形状而不是子串 "sk-"：risk-based / disk-based / task-based 这类正常英文词
#   里也含 "sk-"，子串匹配会把整条正常记忆误拒（记忆是长期数据，误拒比漏判更伤）。
#   形状匹配要求 "sk-" 前面是词边界、后面至少 16 位字母数字 → 上面那些词都不会命中。
_SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b", re.IGNORECASE)


def _tokens(text: str) -> List[str]:
    """中英混合分词：英文/数字整词 + 中文单字 + 中文二元组（与混合检索同一思路）。"""
    text = text or ""
    tokens = [m.group(0).lower() for m in _ASCII_RE.finditer(text)]
    cjk = _CJK_RE.findall(text)
    tokens.extend(cjk)
    tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


def _bm25_scores(
    query_tokens: List[str],
    docs_tokens: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Okapi BM25（纯 Python，零依赖）。

    为什么记忆召回要用 BM25 而不是简单的词重叠比例：
    重叠比例把「量化」和「NF4」当成同等重要；BM25 的 idf 会给稀有词更高权重，
    更符合「用户提到一个独特的项目名/术语时，应该命中那条记忆」的直觉。
    """
    n = len(docs_tokens)
    if n == 0 or not query_tokens:
        return []

    avgdl = sum(len(tokens) for tokens in docs_tokens) / n
    df: Dict[str, int] = {}
    for tokens in docs_tokens:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    scores: List[float] = []
    for tokens in docs_tokens:
        dl = len(tokens)
        score = 0.0
        for token in set(query_tokens):
            freq = tokens.count(token)
            if freq == 0:
                continue
            idf = math.log((n - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5) + 1.0)
            denom = freq + k1 * (1 - b + b * dl / (avgdl or 1.0))
            score += idf * (freq * (k1 + 1)) / denom
        scores.append(score)
    return scores


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _expires_at_iso(ttl_seconds: float) -> str:
    """TTL 的绝对到期时刻（带时区、到微秒）。

    存**绝对时刻**而不是「写入时的 TTL 秒数」：进程重启后仍能判断这条记录过没过期
    （存相对秒数的话，重启后就没有基准了）。
    """
    deadline = datetime.now().astimezone() + timedelta(seconds=float(ttl_seconds))
    return deadline.isoformat()


def _parse_iso(value: str) -> Optional[datetime]:
    """解析 ISO 时刻；解析不了就返回 None（脏数据不当成过期，而是当没有 TTL）。"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# 记忆库默认配置（config.yaml 的 memory 段缺省值在这里也留一份，方便测试与文档对齐）
DEFAULT_COMPACT_THRESHOLD = 2000
DEFAULT_TTL_SECONDS = 0  # 0 = 永不过期


@dataclass
class MemoryRecord:
    key: str
    content: str
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    run_id: str = ""
    # ★ v2 新增字段（老 JSONL 文件里没有 → from_dict 取默认值，**旧文件照样能读**）：
    source: str = ""        # 来源：user_stated / tool_output / inferred …
    verified: bool = False  # 是否已核验（默认 False = 未核验）
    expires_at: str = ""    # 绝对到期时刻（空 = 永不过期）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "content": self.content,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "run_id": self.run_id,
            "source": self.source,
            "verified": bool(self.verified),
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """从一行 JSON 反序列化。

        兼容旧格式：缺 `source` / `verified` / `expires_at` 的老记录一律取默认值
        （source=""、verified=False、expires_at=""），不会因为字段缺失而读不出来。
        """
        verified = data.get("verified", False)
        if isinstance(verified, str):  # 老数据里可能存成 "true"/"false"
            verified = verified.strip().lower() in ("true", "1", "yes")
        return cls(
            key=str(data.get("key") or ""),
            content=str(data.get("content") or ""),
            tags=list(data.get("tags") or []),
            created_at=str(data.get("created_at") or ""),
            run_id=str(data.get("run_id") or ""),
            source=str(data.get("source") or ""),
            verified=bool(verified),
            expires_at=str(data.get("expires_at") or ""),
        )


class MemoryStore:
    """长期记忆存储：JSONL 落盘 + 同 key 覆盖 + BM25 召回（可切回字面重叠）。

    - `remember()` 追加一条记录；
    - `recall()` 按 BM25（默认）或字面重叠打分返回 top-k；
    - `all()` 返回按时间排序的当前视图（同 key 只保留最后一条，**不含已过期记录**）；
    - `forget(key)` 删除一条并立即落盘。

    ★ 配置项 `compact_threshold`（旧名 `max_records`，仍兼容）：
      它的真实语义**不是**「最多存多少条记录」，而是
      「**JSONL 文件的行数**超过它就重写文件（同 key 只留最新）」——
      也就是「追加写导致的重复行攒到多少就合并一次」。
      记录条数（不同 key 的条数）本身**不受它限制**。
      旧名 `max_records` 名不副实（容易被读成容量上限），
      所以改名为 `compact_threshold`；老配置文件继续可用（读到旧键名时按旧值生效并给一条告警）。
    """

    def __init__(self, config: Dict[str, Any], path: str | Path | None = None):
        mem_cfg = config.get("memory", {}) or {}
        self.enabled = bool(mem_cfg.get("enabled", True))
        raw_path = path or mem_cfg.get("path", "./data/agent_memory.jsonl")
        self.path = Path(raw_path)
        self.max_content_chars = int(mem_cfg.get("max_content_chars", 2000))
        # 召回算法：bm25（默认，稀有词权重更高）| overlap（旧的字面重叠比例，便于 A/B）
        self.recall_mode = str(mem_cfg.get("recall_mode", "bm25")).lower()
        self.bm25_k1 = float(mem_cfg.get("bm25_k1", 1.5))
        self.bm25_b = float(mem_cfg.get("bm25_b", 0.75))
        self.run_id = ""

        # ★ 紧凑阈值：读新键名 compact_threshold；老键名 max_records 仍兼容（值照旧生效 + 告警）
        raw_threshold = mem_cfg.get("compact_threshold")
        legacy_threshold = mem_cfg.get("max_records")
        if legacy_threshold is not None:
            # 新键名要么没给、要么还是默认值 → 说明用户用的是老键名，以老值为准
            if raw_threshold is None or int(raw_threshold) == DEFAULT_COMPACT_THRESHOLD:
                raw_threshold = legacy_threshold
                warnings.warn(
                    "memory.max_records 已改名为 memory.compact_threshold"
                    "（语义不变：JSONL 行数超过它就重写文件、同 key 只留最新；"
                    "它**不是**记录条数上限）。请更新 config.yaml。",
                    DeprecationWarning,
                    stacklevel=2,
                )
        self.compact_threshold = max(
            1, int(raw_threshold if raw_threshold is not None else DEFAULT_COMPACT_THRESHOLD)
        )
        # 记忆默认存活时间（秒）：0 = 永不过期
        self.default_ttl_seconds = max(0.0, float(mem_cfg.get("ttl_seconds", DEFAULT_TTL_SECONDS) or 0))

        self._lock = threading.Lock()
        self._records: Dict[str, MemoryRecord] = {}
        self._line_count = 0
        if self.enabled:
            self._load()

    # ---------- backward compatibility ----------

    @property
    def max_records(self) -> int:
        """Deprecated alias for `compact_threshold` (kept so old code doesn't break).

        Note: the semantics are "how many lines the JSONL file has before it gets rewritten
        (keeping only the newest record per key)" — **not** a cap on the number of records.
        """
        return self.compact_threshold

    @max_records.setter
    def max_records(self, value: int) -> None:
        self.compact_threshold = max(1, int(value))

    # ---------- expiry determination ----------

    @staticmethod
    def _is_expired(record: MemoryRecord, now: Optional[datetime] = None) -> bool:
        """这条记录过期了吗？坏数据（解析不了到期时刻）按「没过期」处理。

        宁可让一条脏数据多活一会儿，也不要因为一个解析不了的字段把整库记忆清空。
        """
        if not record.expires_at:
            return False
        deadline = _parse_iso(record.expires_at)
        if deadline is None:
            return False
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return (now or datetime.now().astimezone()) >= deadline

    def _live_records(self) -> List[MemoryRecord]:
        """当前**有效**记录（剔除已过期的）。"""
        with self._lock:
            records = list(self._records.values())
        return [record for record in records if not self._is_expired(record)]

    # ---------- 持久化 ----------

    def _load(self) -> None:
        if not self.path.exists():
            return
        expired = 0
        with self._lock:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 坏行跳过：不要让一行损坏拖垮整个记忆库
                record = MemoryRecord.from_dict(data)
                if not record.key or not record.content:
                    continue
                if self._is_expired(record):
                    expired += 1  # 过期记录不加载（也不参与召回）
                    continue
                self._records[record.key] = record  # 同 key 后写覆盖
                self._line_count += 1
            if expired:
                # 加载时顺手清理：过期记录已经不参与召回，还留在文件里只是占地方
                self._compact_locked()

    def _compact_locked(self) -> None:
        """记录过多时重写文件（同 key 只留最新），把无限追加变成有界存储。

        顺带把**已过期**的记录从文件与内存视图里清掉（它们已经不参与召回，
        留着只是占磁盘；下次加载时也会被清掉）。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        expired_keys = [key for key, rec in self._records.items() if self._is_expired(rec)]
        for key in expired_keys:
            self._records.pop(key, None)
        ordered = sorted(self._records.values(), key=lambda r: r.created_at)
        with open(self.path, "w", encoding="utf-8") as f:
            for record in ordered:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._line_count = len(ordered)

    # ---------- 写入 ----------

    def remember(
        self,
        content: str,
        key: str = "",
        tags: Optional[List[str]] = None,
        run_id: str = "",
        source: str = "",
        verified: bool = False,
        ttl_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """写入一条记忆。

        - `source`：来源标记（user_stated / tool_output / inferred …），只做记录，不参与判断；
        - `verified`：是否**已核验**，默认 False（= 模型说的、未经核对）；
        - `ttl_seconds`：存活秒数；给 None 用配置里的默认值（`memory.ttl_seconds`，0 = 永不过期），
          给 0 表示这条永不过期，给正数表示到点后不再参与召回。
        """
        content = (content or "").strip()
        if not content:
            raise ValueError("记忆内容不能为空")
        if len(content) > self.max_content_chars:
            content = content[: self.max_content_chars] + "…（已截断）"
        lowered = content.lower()
        if any(hint.lower() in lowered for hint in _SECRET_HINTS) or _SECRET_PATTERN.search(content):
            raise ValueError("记忆内容疑似包含密钥（形似 sk-… 的串 / API_KEY=…），拒绝落盘")

        key = (key or "").strip() or f"mem_{uuid.uuid4().hex[:8]}"
        ttl = self.default_ttl_seconds if ttl_seconds is None else max(0.0, float(ttl_seconds))
        record = MemoryRecord(
            key=key,
            content=content,
            tags=[str(t).strip() for t in (tags or []) if str(t).strip()],
            created_at=_now_iso(),
            run_id=run_id or self.run_id,
            source=str(source or "").strip(),
            verified=bool(verified),
            expires_at=_expires_at_iso(ttl) if ttl > 0 else "",
        )

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            self._records[key] = record
            self._line_count += 1
            if self._line_count > max(self.compact_threshold, 1):
                self._compact_locked()

        return record.to_dict()

    def forget(self, key: str) -> bool:
        """删除一条记忆并**立即落盘**。返回是否真的删掉了（key 不存在返回 False）。

        JSONL 是追加写，删除没法「追加一行删除标记」，只能重写文件 ——
        记忆库是几百~几千条的量级，重写成本可以接受。
        """
        key = (key or "").strip()
        if not key:
            return False
        with self._lock:
            if key not in self._records:
                return False
            del self._records[key]
            self._compact_locked()  # 重写文件 → 删除被持久化
            return True

    # ---------- 读取 / 召回 ----------

    def all(self) -> List[Dict[str, Any]]:
        """当前视图（同 key 只保留最后一条，**不含已过期记录**）。"""
        records = sorted(self._live_records(), key=lambda r: r.created_at)
        return [r.to_dict() for r in records]

    def __len__(self) -> int:
        """有效记忆条数（已过期的不算）。"""
        return len(self._live_records())

    def recall(
        self,
        query: str,
        top_k: int = 5,
        only_verified: bool = False,
    ) -> List[Dict[str, Any]]:
        """按 BM25（默认）或字面重叠召回；没有命中就不返回（宁缺毋滥）。

        - BM25：稀有词（项目名 / 专有术语）权重更高；
        - overlap：旧的字面重叠比例，保留用于 A/B 对照。
        两种模式都在 **key 以子串形式出现在 query 里**时给一点额外加权
        （实现是 `record.key.lower() in query_lower`，即**子串包含**，不是全等 ——
        key="pre" 会被 query="preference" 命中）。
        - 已过期的记录不参与召回；
        - `only_verified=True` 时只返回 `verified=True` 的记录（默认 False = 全要）。
        """
        query_tokens = list(dict.fromkeys(_tokens(query)))  # 去重且保持顺序
        if not query_tokens:
            return []

        records = self._live_records()
        if only_verified:
            records = [record for record in records if record.verified]
        if not records:
            return []

        docs_tokens: List[List[str]] = []
        for record in records:
            tokens = _tokens(record.content) + _tokens(record.key)
            for tag in record.tags:
                tokens += _tokens(tag)
            docs_tokens.append(tokens)

        if self.recall_mode == "overlap":
            query_set = set(query_tokens)
            raw_scores = [
                len(query_set & set(tokens)) / len(query_set) for tokens in docs_tokens
            ]
        else:
            raw_scores = _bm25_scores(
                query_tokens, docs_tokens, k1=self.bm25_k1, b=self.bm25_b
            )

        scored: List[Tuple[float, MemoryRecord]] = []
        query_lower = (query or "").lower()
        for record, score in zip(records, raw_scores):
            if score <= 0:
                continue
            if record.key.lower() in query_lower:
                # ★ key **子串**命中（实现是子串包含，不是全等）：在 BM25 之外再加一点确定性权重
                score += 0.5
            scored.append((float(score), record))

        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        results = []
        for score, record in scored[: max(1, int(top_k))]:
            payload = record.to_dict()
            payload["score"] = round(score, 4)
            results.append(payload)
        return results


def build_memory_context(
    store: Optional[MemoryStore],
    task: str,
    top_k: int = 3,
    max_chars: int = 800,
    only_verified: bool = False,
) -> Tuple[str, List[str]]:
    """把与当前任务相关的记忆拼成一段可注入 system prompt 的文本。

    返回 (context_text, used_keys)；没有命中时返回 ("", [])。
    `only_verified=True` 时只注入**已核验**的记忆（默认 False = 全注入，与旧行为一致）。
    """
    if store is None or not store.enabled or len(store) == 0:
        return "", []

    hits = store.recall(task, top_k=top_k, only_verified=only_verified)
    if not hits:
        return "", []

    lines = [
        "【跨任务记忆】以下是之前任务中保存、且与本次任务相关的信息。",
        "它们仅供参考；若与本次检索结果冲突，以本次工具返回的内容为准。",
    ]
    for index, hit in enumerate(hits, 1):
        lines.append(f"{index}. [{hit['key']}] {hit['content']}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…（记忆内容已截断）"
    return text, [hit["key"] for hit in hits]


def build_memory_tools(store: MemoryStore) -> Dict[str, BaseTool]:
    """注册两个工具：`remember_fact`（写）与 `recall_memory`（读）。"""

    @tool
    def remember_fact(
        content: str,
        key: str = "",
        tags: str = "",
        source: str = "",
        verified: bool = False,
        ttl_seconds: float = 0,
    ) -> str:
        """把一条跨任务有用的事实写入长期记忆，下次任务会自动召回。

        适用场景：用户明确说「记住…」，或给出了以后还会用到的偏好/配置/结论。
        参数：content 是要记住的事实；key 是可选的稳定标识（同 key 会覆盖旧值）；
        tags 是逗号分隔的标签（如 "偏好,语言"）；
        source 是这条事实的来源（如 "用户明说" / "工具返回" / "模型推断"），只做记录；
        verified 表示这条事实**已核验**（默认 False = 未经核验，只有你能确认时才填 True）；
        ttl_seconds 是这条记忆的存活秒数（0 = 用配置默认值，配置也没设就永不过期）。
        """
        try:
            tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
            record = store.remember(
                content,
                key=key,
                tags=tag_list,
                source=source,
                verified=verified,
                # ttl_seconds=0 → 用配置里的默认值（配置也没设就永不过期）；
                # 给正数 → 这条记忆自己的存活时间
                ttl_seconds=ttl_seconds if ttl_seconds and float(ttl_seconds) > 0 else None,
            )
            return json.dumps(
                {
                    "ok": True,
                    "key": record["key"],
                    "total": len(store),
                    "source": record["source"],
                    "verified": record["verified"],
                },
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001 —— 工具错误要可读，不要打断任务
            return json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )

    @tool
    def recall_memory(query: str, top_k: int = 5, only_verified: bool = False) -> str:
        """检索之前任务保存的长期记忆，返回与 query 最相关的若干条。

        适用场景：用户提到「上次说过的」「我之前告诉你的」等跨任务信息时先查一下。
        没有命中时返回空列表（不要编造记忆）。
        only_verified=True 时只返回已核验的记录（默认 False = 全部返回）。
        """
        try:
            hits = store.recall(query, top_k=top_k, only_verified=only_verified)
            return json.dumps(
                {"ok": True, "query": query, "count": len(hits), "results": hits},
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )

    return {remember_fact.name: remember_fact, recall_memory.name: recall_memory}
