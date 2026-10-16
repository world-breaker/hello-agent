"""RAG 工具两种形态的集成测试（会真的加载 BGE 模型 / 真的拉起 MCP 子进程）。

- 直连形态（direct）：默认就跑，前提是本机存在同源仓库 rag-mcp-server（否则 skip）；
- MCP 形态：默认 skip，设 `HELLO_AGENT_TEST_MCP=1` 才跑（因为它会启动子进程、加载模型，约 10s）。
  真实使用中的 MCP 端到端证据见 README 的 demo 日志（`--rag-mode mcp`）。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import load_config  # noqa: E402
from agent.rag_tool import make_direct_fallback, make_direct_search_tool  # noqa: E402
from agent.resilience import extract_tool_text  # noqa: E402
from agent.trace import Tracer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RAG_REPO = Path(load_config()["tools"]["rag"]["direct"]["repo_path"])
RAG_AVAILABLE = (RAG_REPO / "server.py").exists()

requires_rag_repo = pytest.mark.skipif(
    not RAG_AVAILABLE, reason="本机没有同源仓库 rag-mcp-server（集成测试跳过）"
)


@requires_rag_repo
def test_direct_form_returns_same_json_contract(tmp_path):
    """直连形态：与 MCP 形态同名同返回结构（模型在两种形态下无感知）。"""
    config = load_config()
    config["tools"]["rag"]["mode"] = "direct"
    tool = make_direct_search_tool(config)

    assert tool.name == "search_knowledge"
    payload = json.loads(tool.invoke({"query": "LoRA 微调的原理", "top_k": 2}))

    assert payload["matched"] >= 1
    assert payload["results"][0]["score"] >= payload["results"][-1]["score"]
    assert "source" in payload["results"][0]
    # 领域外问题必须走「空结果 + hint」，而不是硬凑
    empty = json.loads(tool.invoke({"query": "今天武汉的天气", "top_k": 2}))
    assert empty["matched"] == 0 and "hint" in empty


@requires_rag_repo
def test_direct_fallback_wiring_returns_same_contract(tmp_path):
    """★ 降级路径的真实接线：MCP 工具重试耗尽时 ToolExecutor 会调用这个 fallback。"""
    config = load_config()
    fallback = make_direct_fallback(config)

    payload = json.loads(fallback({"query": "RAG 的检索优化", "top_k": 2}))

    assert payload["matched"] >= 1
    assert payload["results"][0]["source"].endswith(".txt")


@requires_rag_repo
def test_auto_mode_falls_back_to_direct_when_mcp_is_unavailable(tmp_path):
    """auto 模式：MCP 连不上（故意把命令换成不存在的可执行文件）→ 自动降级直连，并在 trace 留证。"""
    from agent.toolkit import build_toolset

    config = load_config()
    config["tools"]["rag"]["mcp"]["server_command"] = str(tmp_path / "python-does-not-exist.exe")
    tracer = Tracer(tmp_path, console="quiet", run_id="fallback")

    async def _run():
        async with build_toolset(config, tracer, "auto") as toolset:
            assert toolset.rag_mode == "direct"          # 已降级
            raw = toolset.tools["search_knowledge"].invoke({"query": "Transformer 自注意力", "top_k": 2})
            return json.loads(raw)

    payload = asyncio.run(_run())

    assert payload["matched"] >= 1
    event_types = [e["type"] for e in tracer.events]
    assert "toolset_fallback" in event_types            # 降级原因也落进了 trace


@requires_rag_repo
@pytest.mark.skipif(
    os.environ.get("HELLO_AGENT_TEST_MCP") != "1",
    reason="MCP 形态集成测试默认跳过（设 HELLO_AGENT_TEST_MCP=1 运行）",
)
def test_mcp_form_end_to_end(tmp_path):
    """MCP 形态：stdio 拉起子进程 → 真实工具调用 → 返回 JSON。"""
    from agent.rag_tool import mcp_rag_tools

    config = load_config()

    async def _run():
        tracer = Tracer(tmp_path, console="quiet", run_id="mcptest")
        async with mcp_rag_tools(config, tracer) as tools:
            assert "search_knowledge" in tools
            raw = await tools["search_knowledge"].ainvoke({"query": "Transformer 的自注意力", "top_k": 2})
            # MCP 工具返回内容块列表 → 统一成文本后再解析（真实适配层细节）
            return json.loads(extract_tool_text(raw)), tracer

    payload, tracer = asyncio.run(_run())

    assert payload["matched"] >= 1
    assert payload["results"][0]["score"] > 0.3
    events = [e["type"] for e in tracer.events]
    assert "toolset" in events  # 记录了 MCP 服务器信息（命令、工具列表、连接耗时）
