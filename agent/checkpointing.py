"""断点续跑：给 LangGraph 图装 checkpointer，以 `thread_id` 为续跑键。

为什么需要它（README §5.8）：
`messages / steps / status` 只活在内存里，进程一挂（Ctrl-C、OOM、部署重启）整轮任务白跑，
而且**已经执行过的工具调用会被重放**——对有副作用的工具这就是真金白银的损失。
装上 checkpointer 后，图每完成一个 super-step 就落一次状态；同一个 `thread_id`
再跑时 LangGraph 会从「最后一个完成的步骤」接着走（依据是 checkpoint 里的 `versions_seen`，
已经消化过当前通道版本的节点不会再跑）。

后端怎么选（按优先级，`agent.checkpoint.backend`）：
1. `sqlite`：官方 `langgraph-checkpoint-sqlite` 的 `AsyncSqliteSaver`（进程级持久化，公共 API 最稳）；
2. `jsonl`：**本项目自己实现的 `JsonlCheckpointer`**（`InMemorySaver` 子类 + JSONL 落盘）。
   为什么自己写一个：① 官方 sqlite 包不在依赖里时（轻量环境 / 离线）需要等价能力；
   ② 复用官方 `serde`（`JsonPlusSerializer`）序列化，所以 LangChain 消息、元组、日期都能原样回读；
   ③ 只有 ~90 行，续跑语义与 sqlite 后端一致（同一个测试对两个后端都跑）。
3. `auto`（默认）：先试 sqlite，不可用则自动退到 jsonl，并在 trace 里记下降级原因。

明确的能力边界（README 也写了）：
- checkpoint 是**状态**持久化，不是**副作用**持久化：工具副作用不保证幂等
  （写文件、下单这类工具在崩溃窗口内仍可能被执行两次），所以要配 `幂等键` 或
  「先写意图再执行」的补偿逻辑——这超出本项目范围；
- SQLite 单文件、单机、无跨节点协调；高并发多 Worker 场景应换成 PostgresSaver。
"""

from __future__ import annotations

import base64
import json
import threading
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Sequence, Tuple

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata, SerializerProtocol
from langgraph.checkpoint.memory import InMemorySaver

BACKEND_AUTO = "auto"
BACKEND_SQLITE = "sqlite"
BACKEND_JSONL = "jsonl"

# JSONL 落盘格式版本（改结构时 +1，便于老文件识别）
LEDGER_VERSION = 1


def sqlite_saver_available() -> Tuple[bool, str]:
    """探测官方 sqlite 后端是否可用，返回 (是否可用, 原因/说明)。"""
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401
        import aiosqlite  # noqa: F401
    except Exception as exc:  # noqa: BLE001 —— 缺依赖就是「不可用」，不是错误
        return False, f"langgraph-checkpoint-sqlite/aiosqlite 不可用：{type(exc).__name__}: {exc}"
    return True, "langgraph-checkpoint-sqlite.AsyncSqliteSaver"


def resolve_paths(config: Dict[str, Any]) -> Tuple[Path, Path]:
    """从配置解析出 sqlite / jsonl 两个候选落盘路径（同一个 path 换后缀）。"""
    cfg = ((config.get("agent") or {}).get("checkpoint")) or {}
    raw = Path(str(cfg.get("path", "./data/checkpoints.sqlite")))
    if not raw.is_absolute():
        base = config.get("_config_dir") or raw.parent
        raw = Path(base) / raw
    sqlite_path = raw if raw.suffix else raw.with_suffix(".sqlite")
    jsonl_path = sqlite_path.with_suffix(".jsonl")
    return sqlite_path, jsonl_path


class JsonlCheckpointer(InMemorySaver):
    """`InMemorySaver` + JSONL 落盘：进程重启后仍能按 thread_id 续跑。

    做法（为什么这么设计）：
    - **只继承、不重写语义**：`get_tuple / put / list` 的取值逻辑全部沿用官方实现，
      我们只在 `put / put_writes` 之后把内部那三张表（`storage / blobs / writes`）镜像到
      一个 JSONL 文件，启动时再读回来——续跑语义因此与官方 saver 完全一致；
    - **序列化交给官方 serde**：`JsonPlusSerializer` 能处理 LangChain 消息对象
      （`lc` 扩展类型），所以 checkpoint 里的 `messages` 能原样回读，
      不是把消息 `str()` 成字符串那种糊弄做法；msgpack 字节再 base64 进 JSON 行；
    - **写盘要原子**：先写 `.tmp` 再 `os.replace`，避免 Ctrl-C 恰好落在写一半时留下半截文件；
    - **全量重写而不是追加**：单任务状态很小（KB 级），全量重写换来「文件永远是自洽快照」，
      不用处理追加日志的截断/去重问题。规模上来了应换 sqlite/Postgres 后端。
    """

    def __init__(self, path: str | Path, *, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # 当前是单任务串行写；锁是防御性的
        self.flush_count = 0
        self.loaded_checkpoints = 0
        self._load()

    # ---------- 序列化 helpers ----------
    @staticmethod
    def _dump_typed(value: Tuple[str, bytes]) -> list:
        kind, data = value
        return [kind, base64.b64encode(data).decode("ascii")]

    @staticmethod
    def _load_typed(value: Sequence) -> Tuple[str, bytes]:
        return (str(value[0]), base64.b64decode(value[1]))

    # ---------- 落盘 ----------
    def snapshot(self) -> Dict[str, Any]:
        """把内部三张表导成可 JSON 化的快照。"""
        storage = [
            {
                "thread_id": thread_id,
                "ns": ns,
                "checkpoint_id": checkpoint_id,
                "checkpoint": self._dump_typed(payload[0]),
                "metadata": self._dump_typed(payload[1]),
                "parent": payload[2],
            }
            for thread_id, namespaces in self.storage.items()
            for ns, checkpoints in namespaces.items()
            for checkpoint_id, payload in checkpoints.items()
        ]
        blobs = [
            {
                "thread_id": thread_id,
                "ns": ns,
                "channel": channel,
                "version": version,
                "blob": self._dump_typed(payload),
            }
            for (thread_id, ns, channel, version), payload in self.blobs.items()
        ]
        writes = [
            {
                "thread_id": thread_id,
                "ns": ns,
                "checkpoint_id": checkpoint_id,
                "task_id": inner_key[0],
                "idx": inner_key[1],
                "channel": stored[1],
                "value": self._dump_typed(stored[2]),
                "path": stored[3],
            }
            for (thread_id, ns, checkpoint_id), inner in self.writes.items()
            for inner_key, stored in inner.items()
        ]
        return {
            "version": LEDGER_VERSION,
            "backend": BACKEND_JSONL,
            "storage": storage,
            "blobs": blobs,
            "writes": writes,
        }

    def flush(self) -> None:
        """原子写盘：tmp 文件 + os.replace。"""
        with self._lock:
            snapshot = self.snapshot()
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False)
            tmp.replace(self.path)
            self.flush_count += 1

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open(encoding="utf-8") as f:
                snapshot = json.load(f)
        except (json.JSONDecodeError, OSError):
            # 文件坏掉不能影响「这次能不能跑」：当作空状态（会在下一次 put 时被覆盖）
            return
        for item in snapshot.get("storage") or []:
            self.storage[item["thread_id"]][item["ns"]][item["checkpoint_id"]] = (
                self._load_typed(item["checkpoint"]),
                self._load_typed(item["metadata"]),
                item.get("parent"),
            )
            self.loaded_checkpoints += 1
        for item in snapshot.get("blobs") or []:
            self.blobs[(item["thread_id"], item["ns"], item["channel"], item["version"])] = self._load_typed(item["blob"])
        for item in snapshot.get("writes") or []:
            task_id = item["task_id"]
            self.writes[(item["thread_id"], item["ns"], item["checkpoint_id"])][(task_id, item["idx"])] = (
                task_id,
                item["channel"],
                self._load_typed(item["value"]),
                item.get("path", ""),
            )

    # ---------- 覆盖官方实现：写完就落盘 ----------
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        result = super().put(config, checkpoint, metadata, new_versions)
        self.flush()
        return result

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)
        self.flush()

    # 异步入口也走落盘逻辑（图是用 ainvoke 跑的，会调 aput/aput_writes）
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        self.flush()


@asynccontextmanager
async def open_checkpointer(
    config: Dict[str, Any],
    enabled: bool = True,
) -> AsyncIterator[Tuple[Any, str, str]]:
    """按配置打开 checkpointer，产出 `(saver, backend, note)`；退出时释放连接。

    - `enabled=False`（没给 thread_id）时产出 `(None, "disabled", "")`，图不带 checkpointer，
      与加这个功能之前的行为**逐字节一致**；
    - `backend: sqlite` 但依赖缺失 / 建连失败 → 直接抛错（显式配置就该显式失败）；
      `backend: auto` → 静默降级到 jsonl，并把原因写进 note（由 runner 记进 trace）。
    - `AsyncExitStack` 负责把 sqlite 连接的生命周期绑到本上下文退出，而不是绑到 yield 前后，
      这样调用方在 with 体内的任何异常（包括我们演示用的 BaseException）都不会被误当成「建连失败」。
    """
    checkpoint_cfg = ((config.get("agent") or {}).get("checkpoint")) or {}
    if not enabled or not bool(checkpoint_cfg.get("enabled", True)):
        yield None, "disabled", ""
        return

    backend = str(checkpoint_cfg.get("backend", BACKEND_AUTO)).lower()
    sqlite_path, jsonl_path = resolve_paths(config)

    async with AsyncExitStack() as stack:
        note = ""
        if backend in (BACKEND_AUTO, BACKEND_SQLITE):
            available, reason = sqlite_saver_available()
            if available:
                saver = None
                try:
                    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                    saver = await stack.enter_async_context(
                        AsyncSqliteSaver.from_conn_string(str(sqlite_path))
                    )
                except Exception as exc:  # noqa: BLE001 —— 只有「建不起来」才降级
                    if backend == BACKEND_SQLITE:
                        raise
                    note = f"sqlite 后端建连失败（{type(exc).__name__}: {exc}）→ 已降级到 MemorySaver + JSONL"
                if saver is not None:
                    yield saver, BACKEND_SQLITE, f"{reason}（{sqlite_path}）"
                    return
            else:
                if backend == BACKEND_SQLITE:
                    raise RuntimeError(f"配置要求 sqlite 后端，但{reason}")
                note = f"{reason} → 已降级到 MemorySaver + JSONL 落盘"

        saver = JsonlCheckpointer(jsonl_path)
        yield saver, BACKEND_JSONL, f"{note}（{jsonl_path}；启动时已载入 {saver.loaded_checkpoints} 个历史 checkpoint）"


async def list_threads(saver: Any) -> Dict[str, int]:
    """列出 saver 里每个 thread_id 的 checkpoint 数量（CLI `--list-threads` 用）。

    用异步接口 `alist`：官方 `AsyncSqliteSaver` 的同步 `list()` 在主线程里会直接抛
    `InvalidStateError`（它内部是 aiosqlite 连接，只能在事件循环里用）。
    """
    counts: Dict[str, int] = {}
    async for item in saver.alist(None):
        thread_id = item.config["configurable"]["thread_id"]
        counts[thread_id] = counts.get(thread_id, 0) + 1
    return counts
