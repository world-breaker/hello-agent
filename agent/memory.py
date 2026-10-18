"""跨任务长期记忆：JSONL 持久化 + 关键词召回 + 任务启动时自动注入。

为什么需要它？
- 单次 ReAct 的 messages 只在**本次任务**内有效，任务结束就丢了；
- 用户说「记住：以后回答都用中文」「我的项目路径是 …」时，下一次任务应该能自动想起来，
  而不是每次重新交代。

设计取舍（面试可讲）：
- **JSONL 追加写**：一行一条记录，崩溃最多丢最后一行；便于 grep / jq / 人工排查；
- **同 key 后写覆盖**：`remember` 允许带 key，读取时以最后一条为准（天然 upsert，不引入数据库）；
- **召回用字面重叠**（英文词 + 中文单字/二元组）而不是向量：记忆条数通常几十~几百，
  零依赖、可解释、离线可跑；要语义召回时再接 embedding（与 rag-eval 的混合检索同思路）；
- **只注入命中的记忆**：没有关键词重叠就不注入，避免把无关记忆塞进 prompt 污染上下文；
- **显式工具 + 自动注入**：模型可用 `remember_fact` 主动记，任务开始时系统自动召回 top-k。

安全边界：
- 记忆是**用户数据**，不要写入密钥；`remember_fact` 会拒绝包含疑似密钥的内容。
"""

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool, tool

_ASCII_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 简单密钥特征：sk-xxx / OPENAI_API_KEY=... —— 记忆是长期落盘的，宁可拒绝也不要泄露
_SECRET_HINTS = ("sk-", "OPENAI_API_KEY", "API_KEY=", "BEGIN PRIVATE KEY")


def _tokens(text: str) -> List[str]:
    """中英混合分词：英文/数字整词 + 中文单字 + 中文二元组（与混合检索同一思路）。"""
    text = text or ""
    tokens = [m.group(0).lower() for m in _ASCII_RE.finditer(text)]
    cjk = _CJK_RE.findall(text)
    tokens.extend(cjk)
    tokens.extend(cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1))
    return tokens


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class MemoryRecord:
    key: str
    content: str
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "content": self.content,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            key=str(data.get("key") or ""),
            content=str(data.get("content") or ""),
            tags=list(data.get("tags") or []),
            created_at=str(data.get("created_at") or ""),
            run_id=str(data.get("run_id") or ""),
        )


class MemoryStore:
    """长期记忆存储：JSONL 落盘 + 同 key 覆盖 + 字面召回。

    - `remember()` 追加一条记录；
    - `recall()` 按关键词重叠打分返回 top-k；
    - `all()` 返回按时间排序的当前视图（同 key 只保留最后一条）。
    """

    def __init__(self, config: Dict[str, Any], path: str | Path | None = None):
        mem_cfg = config.get("memory", {}) or {}
        self.enabled = bool(mem_cfg.get("enabled", True))
        raw_path = path or mem_cfg.get("path", "./data/agent_memory.jsonl")
        self.path = Path(raw_path)
        self.max_records = int(mem_cfg.get("max_records", 2000))
        self.max_content_chars = int(mem_cfg.get("max_content_chars", 2000))
        self.run_id = ""

        self._lock = threading.Lock()
        self._records: Dict[str, MemoryRecord] = {}
        self._line_count = 0
        if self.enabled:
            self._load()

    # ---------- 持久化 ----------

    def _load(self) -> None:
        if not self.path.exists():
            return
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
                self._records[record.key] = record  # 同 key 后写覆盖
                self._line_count += 1

    def _compact_locked(self) -> None:
        """记录过多时重写文件（同 key 只留最新），把无限追加变成有界存储。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
    ) -> Dict[str, Any]:
        content = (content or "").strip()
        if not content:
            raise ValueError("记忆内容不能为空")
        if len(content) > self.max_content_chars:
            content = content[: self.max_content_chars] + "…（已截断）"
        lowered = content.lower()
        if any(hint.lower() in lowered for hint in _SECRET_HINTS):
            raise ValueError("记忆内容疑似包含密钥（sk- / API_KEY），拒绝落盘")

        key = (key or "").strip() or f"mem_{uuid.uuid4().hex[:8]}"
        record = MemoryRecord(
            key=key,
            content=content,
            tags=[str(t).strip() for t in (tags or []) if str(t).strip()],
            created_at=_now_iso(),
            run_id=run_id or self.run_id,
        )

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            self._records[key] = record
            self._line_count += 1
            if self._line_count > max(self.max_records, 1):
                self._compact_locked()

        return record.to_dict()

    # ---------- 读取 / 召回 ----------

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            records = sorted(self._records.values(), key=lambda r: r.created_at)
        return [r.to_dict() for r in records]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """按查询与记忆的字面重叠召回；没有重叠就不返回（宁缺毋滥）。"""
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return []

        scored: List[Tuple[float, MemoryRecord]] = []
        with self._lock:
            records = list(self._records.values())

        for record in records:
            record_tokens = set(_tokens(record.content)) | set(_tokens(record.key))
            for tag in record.tags:
                record_tokens |= set(_tokens(tag))
            if not record_tokens:
                continue
            overlap = query_tokens & record_tokens
            if not overlap:
                continue
            # 覆盖率：命中查询词的比例；再给 key/tag 精确命中一点加权
            score = len(overlap) / len(query_tokens)
            if record.key.lower() in (query or "").lower():
                score += 0.5
            scored.append((score, record))

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
) -> Tuple[str, List[str]]:
    """把与当前任务相关的记忆拼成一段可注入 system prompt 的文本。

    返回 (context_text, used_keys)；没有命中时返回 ("", [])。
    """
    if store is None or not store.enabled or len(store) == 0:
        return "", []

    hits = store.recall(task, top_k=top_k)
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
    def remember_fact(content: str, key: str = "", tags: str = "") -> str:
        """把一条跨任务有用的事实写入长期记忆，下次任务会自动召回。

        适用场景：用户明确说「记住…」，或给出了以后还会用到的偏好/配置/结论。
        参数：content 是要记住的事实；key 是可选的稳定标识（同 key 会覆盖旧值）；
        tags 是逗号分隔的标签（如 "偏好,语言"）。
        """
        try:
            tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
            record = store.remember(content, key=key, tags=tag_list)
            return json.dumps(
                {"ok": True, "key": record["key"], "total": len(store)},
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001 —— 工具错误要可读，不要打断任务
            return json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )

    @tool
    def recall_memory(query: str, top_k: int = 5) -> str:
        """检索之前任务保存的长期记忆，返回与 query 最相关的若干条。

        适用场景：用户提到「上次说过的」「我之前告诉你的」等跨任务信息时先查一下。
        没有命中时返回空列表（不要编造记忆）。
        """
        try:
            hits = store.recall(query, top_k=top_k)
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
