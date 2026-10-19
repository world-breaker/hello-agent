"""LangGraph 编排：最小 ReAct 循环（agent → tools → agent …）+ 最大步数上限 + trace 埋点。

为什么用 StateGraph 手写循环，而不是一行 `create_react_agent(...)`：
- **最大步数要可控**：框架的 `recursion_limit` 是「图步数」的兜底保护，不等于业务上的
  「最多思考几轮」；本项目把业务上限做进路由函数（`max_steps`），到点强制收尾；
- **失败处理要显式**：工具执行节点的每一步都要被 trace 记录、要能重试/降级/熔断，
  这需要自己的执行层（见 `resilience.py`），而不是框架内建的「把错误字符串塞回去」；
- **状态要说得清**：`messages` + `steps` + `status` 三个字段就是全部状态，
  面试问「状态存在哪里」时可以一行一行指出来。

图结构：
    START → agent ──(有 tool_calls 且未超步数)→ tools → agent → … → END
                 └──(无 tool_calls / 超步数 / 模型故障)→ END
"""

import asyncio
import time
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.context import compress_messages
from agent.resilience import ToolExecutor
from agent.trace import Tracer
from agent.usage import extract_usage


class AgentState(TypedDict):
    """图的全部状态。

    - messages：对话与工具消息（`add_messages` 表示新消息追加而不是覆盖）；
    - steps：已经走了几轮「模型思考」——最大步数限制的依据；
    - status：new / in_progress / completed / llm_error。
    """

    messages: Annotated[List[AnyMessage], add_messages]
    steps: int
    status: str


def _text_of(message: AnyMessage) -> str:
    """把消息内容取成字符串（内容可能是 str 或内容块列表）。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def build_agent_graph(
    llm: BaseChatModel,
    executor: ToolExecutor,
    tracer: Tracer,
    config: Dict[str, Any],
    memory_context: str = "",
    checkpointer: Any = None,
):
    """构建并编译 ReAct 图。

    memory_context：跨任务记忆（由 runner 从 MemoryStore 召回），非空时追加到 system prompt。
    checkpointer：LangGraph 状态持久化（`agent/checkpointing.py`）。为 None 时行为与之前完全一致；
        给了就在每个 super-step 结束后落一次状态，配合 `config.configurable.thread_id` 支持断点续跑。
    """
    max_steps = int(config["agent"]["max_steps"])
    system_prompt = str(config["agent"]["system_prompt"])
    if memory_context:
        system_prompt = system_prompt + "\n\n" + memory_context

    parallel_tools = bool(config["agent"].get("parallel_tools", True))
    max_parallel_tools = max(1, int(config["agent"].get("max_parallel_tools", 4)))

    # ★ 关键一步：把工具 schema 绑定到模型。
    # 框架版（create_react_agent / create_agent）会自动做这件事，手写循环必须显式绑定，
    # 否则模型完全不知道工具存在 —— 它会「凭记忆直答」，而且答案看起来还挺像样，
    # 属于非常隐蔽的错误（第一版就是这样漏掉的，见 README「踩坑记录」）。
    tools = list(executor.tools.values())
    if tools:
        llm = llm.bind_tools(tools)

    # ---------- 节点 1：模型思考 ----------
    async def agent_node(state: AgentState) -> Dict[str, Any]:
        tracer.next_step()
        step = tracer.step

        # 送模型前压上下文：只压内容、不删消息（保住 tool_calls ↔ ToolMessage 配对）
        history, context_stats = compress_messages(
            list(state["messages"]),
            max_chars=int(config["agent"].get("max_context_chars", 12000)),
            keep_last=int(config["agent"].get("context_keep_last", 6)),
        )
        if context_stats.get("compressed"):
            tracer.event("context_compressed", **context_stats)
        messages = [SystemMessage(content=system_prompt)] + history

        started = time.perf_counter()
        try:
            ai_message: AIMessage = await llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 —— 模型侧故障不能让整个流程抛出去
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            tracer.event("llm_call", elapsed_ms=elapsed_ms, ok=False, error=f"{type(exc).__name__}: {exc}")
            fallback_text = (
                f"[模型调用失败] {type(exc).__name__}: {exc}\n"
                "（已按配置重试 llm.max_retries 次；这一步没有产生任何工具调用）"
            )
            return {
                "messages": [AIMessage(content=fallback_text)],
                "steps": state.get("steps", 0) + 1,
                "status": "llm_error",
            }

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        # token 用量：优先 usage_metadata，退回 response_metadata.token_usage（见 agent/usage.py）
        usage = extract_usage(ai_message)
        tracer.event(
            "llm_call",
            elapsed_ms=elapsed_ms,
            ok=True,
            usage=usage,
        )

        tool_calls = [
            {"name": call.get("name"), "args": call.get("args") or {}}
            for call in (getattr(ai_message, "tool_calls", None) or [])
        ]
        tracer.event("thought", content=_text_of(ai_message), tool_calls=tool_calls)

        status = "completed" if not tool_calls else "in_progress"
        return {
            "messages": [ai_message],
            "steps": state.get("steps", 0) + 1,
            "status": status,
        }

    # ---------- 节点 2：执行工具 ----------
    async def tools_node(state: AgentState) -> Dict[str, Any]:
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", None) or [])
        if not calls:
            return {"messages": [], "status": "in_progress"}

        started = time.perf_counter()

        # ★ 并发化：模型一轮可能同时给出多个互不依赖的 tool_calls（如同时查时间 + 算数）。
        #   串行执行会把延迟累加；并发执行的总延迟约等于最慢的那个工具。
        #   用 Semaphore 限制并发上限，避免模型一次吐出几十个调用把依赖打爆。
        if parallel_tools and len(calls) > 1:
            tracer.event(
                "parallel_tools",
                count=len(calls),
                max_parallel=max_parallel_tools,
                tools=[call.get("name") for call in calls],
            )
            semaphore = asyncio.Semaphore(max_parallel_tools)

            async def _run(call: Dict[str, Any]):
                async with semaphore:
                    return await executor.execute(call["name"], call.get("args") or {})

            # gather 保持输入顺序 → ToolMessage 与 tool_calls 一一对应（协议要求）
            outcomes = await asyncio.gather(*(_run(call) for call in calls))
            tracer.event(
                "parallel_tools_done",
                count=len(calls),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        else:
            # 单个调用或显式关闭并发时走串行路径（保留原行为）
            outcomes = []
            for call in calls:
                outcomes.append(
                    await executor.execute(call["name"], call.get("args") or {})
                )

        tool_messages: List[ToolMessage] = [
            ToolMessage(
                content=outcome.to_observation(),
                tool_call_id=call.get("id") or "",
                name=call.get("name") or "unknown",
            )
            for call, outcome in zip(calls, outcomes)
        ]
        return {"messages": tool_messages, "status": "in_progress"}

    # ---------- 路由：决定继续调工具还是收尾 ----------
    def route_after_agent(state: AgentState) -> str:
        if state.get("status") in ("completed", "llm_error"):
            return "end"
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "end"
        if state.get("steps", 0) >= max_steps:
            # 步数用尽：不再执行工具（否则第 max_steps+1 轮的观察永远没机会被消化）
            return "end"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer) if checkpointer is not None else graph.compile()

