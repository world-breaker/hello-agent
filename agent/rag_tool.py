"""把 rag-mcp-server 的 `search_knowledge` 接成本地 Agent 的工具——提供两种形态。

**形态 A：MCP 客户端（默认）**
    用 `langchain-mcp-adapters` 以 stdio 拉起 rag-mcp-server 子进程，
    工具调用走标准 MCP JSON-RPC 协议。好处是**进程隔离**：
    检索用的 BGE 模型、知识库索引都在子进程里，Agent 进程保持轻量；
    同一个 Server 还能同时被 Claude Desktop / Cursor 复用。
    ⚠️ 依赖现状：langchain-mcp-adapters 目前锁 `mcp<2`，所以客户端是 mcp 1.x，
    服务端是 mcp 2.x —— 两者能互通（互操作的依据是协议本身，不是 SDK 版本）。

**形态 B：进程内直连（direct）**
    把 rag-mcp-server 仓库加进 `sys.path`，直接调用同源的 `KnowledgeService`。
    两个用途：① MCP 通道故障时的**降级路径**；② 不想起子进程的轻量跑法。
    代价：BGE 模型会加载进 Agent 进程（约 5s、几百 MB 内存），且与 Agent 共享生命周期。
"""

import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import BaseTool, tool

from agent.trace import Tracer

SEARCH_TOOL_NAME = "search_knowledge"
_direct_service = None  # 进程内检索服务（懒加载，模型只加载一次）


# ---------------- 形态 A：MCP 客户端 ----------------


@asynccontextmanager
async def mcp_rag_tools(config: Dict[str, Any], tracer: Tracer):
    """以 stdio 连接 rag-mcp-server，产出 {工具名: LangChain 工具}。

    关键点：整个任务期间**复用同一个 MCP 会话**（`client.session(...)` 上下文），
    而不是「每个工具调用开一次会话」——否则每次调用都会重启 server 子进程、
    重新加载 BGE 模型 + 重建索引（每次多花好几秒），这是实打实的成本问题。

    ★ 连接超时（`tools.rag.mcp.connect_timeout_s`）：连接阶段 = 拉起子进程 + 握手
      （initialize）+ 拉工具列表。MCP Server 挂起时这一步会**一直等下去**，
      于是整个任务卡死。现在这段有硬上限：超时抛可读的 TimeoutError，
      `auto` 模式会被 toolkit 捕获并降级到进程内直连（trace 里留 toolset_fallback）。
      注意只给「连接阶段」加超时，**会话建立后的正常使用期不受影响**。
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    mcp_cfg = config["tools"]["rag"]["mcp"]
    server_name = "rag-knowledge"
    connect_timeout_s = float(mcp_cfg.get("connect_timeout_s") or 0)
    connection = {
        "transport": "stdio",
        "command": str(mcp_cfg["server_command"]),
        "args": [str(mcp_cfg["server_script"])],
        "cwd": str(mcp_cfg["cwd"]),
        # ★ 踩坑：MCP SDK 拉起子进程时**只传一小撮环境变量**（PATH 等），
        # PYTHONUTF8 / PYTHONIOENCODING 不在其中。不显式指定的话，子进程在中文
        # Windows 上会用 GBK 写 stderr，而客户端的 stderr 通常是 UTF-8 文件/日志面板
        # → 服务端日志整段乱码（协议本身走 stdout，不受影响，所以这是个"看起来吓人但功能正常"的坑）。
        "env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    }

    client = MultiServerMCPClient({server_name: connection})
    started = time.perf_counter()
    # AsyncExitStack：会话要活过 yield（任务期间复用），所以不能直接用 with
    async with AsyncExitStack() as stack:
        try:
            enter = stack.enter_async_context(client.session(server_name))
            if connect_timeout_s > 0:
                session = await asyncio.wait_for(enter, timeout=connect_timeout_s)
                tools = await asyncio.wait_for(load_mcp_tools(session), timeout=connect_timeout_s)
            else:
                session = await enter
                tools = await load_mcp_tools(session)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"MCP 连接超时（connect_timeout_s={connect_timeout_s}s）：{server_name} "
                f"没能在时限内完成「拉起子进程 + 握手 + 拉工具列表」。"
            ) from exc
        connect_ms = round((time.perf_counter() - started) * 1000, 2)
        tracer.event(
            "toolset",
            source="mcp",
            server=server_name,
            command=connection["command"],
            server_script=connection["args"],
            tools=[t.name for t in tools],
            connect_ms=connect_ms,
            connect_timeout_s=connect_timeout_s,
        )
        yield {t.name: t for t in tools}


# ---------------- 形态 B：进程内直连 ----------------


def get_direct_service(config: Dict[str, Any]):
    """懒加载同源检索服务（把 rag-mcp-server 仓库加进 sys.path 后 import）。"""
    global _direct_service
    if _direct_service is None:
        direct_cfg = config["tools"]["rag"]["direct"]
        repo_path = Path(str(direct_cfg["repo_path"]))
        config_path = repo_path / str(direct_cfg["config_path"])
        if not config_path.exists():
            raise FileNotFoundError(f"找不到同源检索仓库的配置：{config_path}")

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from rag.config import load_config as load_rag_config  # type: ignore
        from rag.service import KnowledgeService  # type: ignore

        started = time.perf_counter()
        service = KnowledgeService(load_rag_config(config_path))
        service.warmup()
        _direct_service = service
        _direct_service._warmup_seconds = round(time.perf_counter() - started, 3)  # 记录给 stats 用
    return _direct_service


def make_direct_search_tool(config: Dict[str, Any]) -> BaseTool:
    """进程内直连版 search_knowledge（与 MCP 版同名同返回结构，模型无感知）。"""

    @tool(SEARCH_TOOL_NAME)
    def search_knowledge(query: str, top_k: int = 3) -> str:
        """检索本地知识库（中文技术文档），返回最相关的原文片段与相似度分数（JSON）。

        当问题可能依赖本地资料时，先调用它拿证据再回答。top_k 默认 3。
        """
        service = get_direct_service(config)
        payload = service.search(query, top_k=top_k)
        return json.dumps(payload, ensure_ascii=False)

    return search_knowledge


def make_direct_fallback(config: Dict[str, Any]):
    """给 MCP 版工具准备的降级函数：MCP 通道重试耗尽后，切到进程内直连同源检索。

    注意它是**懒加载**的：正常情况下永远不会触发，也就不会把 BGE 模型加载进 Agent 进程。
    """

    def _fallback(args: Dict[str, Any]) -> str:
        direct_tool = make_direct_search_tool(config)
        return direct_tool.invoke(args or {})

    return _fallback
