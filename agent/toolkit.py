"""工具集组装：把内置工具与 RAG 工具装成一个可用的工具袋（异步上下文管理器）。

组装顺序与降级逻辑：
1. 内置工具（calculator / current_time）——无外部依赖，永远可用；
2. RAG 工具：
   - `mode: mcp`  → 只走 MCP，连接失败即报错（用于验证 MCP 链路本身）；
   - `mode: direct` → 只走进程内直连；
   - `mode: auto`（默认）→ 先试 MCP；连不上就自动降到直连，并在 trace 里留证据。
3. MCP 模式下**额外注册降级路径**：MCP 工具重试耗尽时，ToolExecutor 会切到直连检索。
"""

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from langchain_core.tools import BaseTool

from agent.memory import MemoryStore, build_memory_tools
from agent.rag_tool import (
    SEARCH_TOOL_NAME,
    make_direct_fallback,
    make_direct_search_tool,
    mcp_rag_tools,
)
from agent.tools_builtin import build_builtin_tools
from agent.trace import Tracer


@dataclass
class ToolSet:
    tools: Dict[str, BaseTool] = field(default_factory=dict)
    fallbacks: Dict[str, Callable[[Dict[str, Any]], Any]] = field(default_factory=dict)
    rag_mode: str = "none"  # mcp | direct | none
    rag_note: str = ""
    tool_names: List[str] = field(default_factory=list)
    memory: MemoryStore | None = None  # 跨任务长期记忆（未启用时为 None）


@asynccontextmanager
async def build_toolset(
    config: Dict[str, Any],
    tracer: Tracer,
    rag_mode_override: str | None = None,
):
    """产出组装好的工具集；退出时自动清理 MCP 会话/子进程。"""
    async with AsyncExitStack() as stack:
        toolset = ToolSet()
        toolset.tools.update(build_builtin_tools(config))

        # 跨任务长期记忆：默认启用；记录 run_id 便于追溯「这条记忆是哪个任务写下的」
        memory_cfg = config.get("memory", {}) or {}
        if bool(memory_cfg.get("enabled", True)):
            store = MemoryStore(config)
            store.run_id = tracer.run_id
            toolset.memory = store
            if bool(memory_cfg.get("tools", True)):
                toolset.tools.update(build_memory_tools(store))
            tracer.event(
                "memory_ready",
                enabled=True,
                path=str(store.path),
                records=len(store),
                tools=["remember_fact", "recall_memory"]
                if bool(memory_cfg.get("tools", True))
                else [],
            )

        rag_cfg = config["tools"]["rag"]
        mode = (rag_mode_override or str(rag_cfg["mode"])).lower()

        if mode in ("mcp", "auto"):
            try:
                mcp_tools = await stack.enter_async_context(mcp_rag_tools(config, tracer))
                toolset.tools.update(mcp_tools)
                toolset.rag_mode = "mcp"
                toolset.rag_note = "经 MCP stdio 连接 rag-mcp-server 子进程"
            except Exception as exc:  # noqa: BLE001
                if mode == "mcp":
                    raise
                tracer.event(
                    "toolset_fallback",
                    reason=f"MCP 接入失败，改用进程内直连：{type(exc).__name__}: {exc}",
                )

        if SEARCH_TOOL_NAME not in toolset.tools:
            toolset.tools[SEARCH_TOOL_NAME] = make_direct_search_tool(config)
            toolset.rag_mode = "direct"
            toolset.rag_note = "进程内直连 rag-mcp-server 的同源 KnowledgeService"
        else:
            # MCP 可用：注册「重试耗尽 → 直连」的降级路径
            toolset.fallbacks[SEARCH_TOOL_NAME] = make_direct_fallback(config)

        toolset.tool_names = sorted(toolset.tools)
        tracer.event(
            "tools_ready",
            tools=toolset.tool_names,
            rag_mode=toolset.rag_mode,
            rag_note=toolset.rag_note,
            fallbacks=sorted(toolset.fallbacks),
        )
        yield toolset
