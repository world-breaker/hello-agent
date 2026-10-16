"""单条任务的执行入口：组装配置 → 工具 → 模型 → 图 → 跑 → 收尾落 trace。"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from agent.config import describe_llm, get_llm_settings, load_config
from agent.graph import build_agent_graph
from agent.llm import build_llm
from agent.resilience import ToolExecutor
from agent.toolkit import build_toolset
from agent.trace import Tracer

# 结束状态的取值（对外契约，README 与测试都按这个口径）
STATUS_COMPLETED = "completed"
STATUS_MAX_STEPS = "max_steps_exceeded"
STATUS_LLM_ERROR = "llm_error"
STATUS_ERROR = "error"


@dataclass
class RunResult:
    run_id: str
    status: str
    steps: int
    final_answer: str
    trace_path: str
    elapsed_ms: float
    rag_mode: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _text_of(message: AnyMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", block)) if isinstance(block, dict) else str(block) for block in content
        )
    return str(content)


def _summarize_metrics(tracer: Tracer, executor: ToolExecutor) -> Dict[str, Any]:
    """从 trace 事件里聚合本次运行的指标（可直接用于「工具调用成功率 / 平均步数」统计）。"""
    counts: Dict[str, int] = {}
    for event in tracer.events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    tool_calls = executor.stats["calls"]
    return {
        "llm_calls": counts.get("thought", 0),
        "tool_calls": tool_calls,
        "tool_ok": executor.stats["ok"],
        "tool_failed": executor.stats["failed"],
        "tool_retries": executor.stats["retries"],
        "tool_degraded": executor.stats["degraded"],
        "circuit_opens": executor.stats["circuit_opens"],
        "tool_success_rate": round(executor.stats["ok"] / tool_calls, 4) if tool_calls else None,
        "event_counts": counts,
    }


def _failure_explanation(state: Dict[str, Any], max_steps: int) -> str:
    """走到最大步数时的收尾说明：把「为什么没答完」讲清楚，而不是假装完成。"""
    last = state["messages"][-1]
    pending = [call.get("name") for call in (getattr(last, "tool_calls", None) or [])]
    partial = _text_of(last).strip()
    lines = [
        f"（未完成）已达最大步数上限 max_steps={max_steps}，本任务被强制收尾。",
    ]
    if pending:
        lines.append(f"最后一步仍在请求调用工具：{pending}，但这些调用没有执行。")
    if partial:
        lines.append(f"最后一步模型已给出的部分内容：{partial}")
    lines.append("如需继续，请提高 config.yaml 的 agent.max_steps 或把任务拆小后重跑。")
    return "\n".join(lines)


async def run_task(
    task: str,
    config: Dict[str, Any] | None = None,
    rag_mode: str | None = None,
    console: str | None = None,
) -> RunResult:
    """跑一条任务，返回结构化结果（trace 落在 traces/ 下）。"""
    config = config or load_config()
    settings = get_llm_settings(config)

    if settings["missing"]:
        raise RuntimeError(
            "缺少 LLM 端点配置："
            + ", ".join(settings["missing"])
            + "。请复制 .env.example 为 .env 并填入 OpenAI 兼容端点的 BASE_URL / API_KEY / MODEL。"
        )

    if console:
        config["trace"]["console"] = console

    tracer = Tracer(
        trace_dir=config["trace"]["dir"],
        console=str(config["trace"]["console"]),
        preview_chars=int(config["trace"]["preview_chars"]),
    )
    started = time.perf_counter()
    tracer.event(
        "run_start",
        task=task,
        llm=describe_llm(settings),
        max_steps=int(config["agent"]["max_steps"]),
        rag_mode=rag_mode or config["tools"]["rag"]["mode"],
    )

    status = STATUS_ERROR
    error: str | None = None
    state: Dict[str, Any] = {}
    rag_mode_used = "none"
    metrics: Dict[str, Any] = {}
    executor: ToolExecutor | None = None

    try:
        async with build_toolset(config, tracer, rag_mode) as toolset:
            rag_mode_used = toolset.rag_mode
            executor = ToolExecutor(toolset.tools, config, tracer, toolset.fallbacks)
            llm = build_llm(settings)
            graph = build_agent_graph(llm, executor, tracer, config)

            state = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=task)],
                    "steps": 0,
                    "status": "new",
                },
                config={"recursion_limit": int(config["agent"]["recursion_limit"])},
            )
            status = str(state.get("status", STATUS_ERROR))
            if status == "in_progress":
                # 图停下来但状态还是「进行中」= 被最大步数拦住了
                status = STATUS_MAX_STEPS
                tracer.event(
                    "max_steps_reached",
                    max_steps=int(config["agent"]["max_steps"]),
                    steps=int(state.get("steps", 0)),
                )
            elif status == "llm_error":
                status = STATUS_LLM_ERROR
    except Exception as exc:  # noqa: BLE001 —— 任何未预期的异常都要落 trace 并给出可读结论
        status = STATUS_ERROR
        error = f"{type(exc).__name__}: {exc}"
        tracer.event("run_error", error=error)

    # ---------- 收尾 ----------
    messages: List[AnyMessage] = state.get("messages") or []
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    if status == STATUS_MAX_STEPS:
        final_answer = _failure_explanation(state, int(config["agent"]["max_steps"]))
    elif last_ai is not None:
        final_answer = _text_of(last_ai).strip()
    else:
        final_answer = error or "（没有产生任何回答）"

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics = _summarize_metrics(tracer, executor) if executor else {}

    tracer.event(
        "run_end",
        status=status,
        steps=int(state.get("steps", 0)),
        rag_mode=rag_mode_used,
        elapsed_ms=elapsed_ms,
        final_answer=final_answer,
        metrics=metrics,
        trace_path=str(tracer.path),
        error=error,
    )

    print(f"\n===== 最终回答（status={status}）=====")
    print(final_answer)
    print(
        f"===== 指标：步数={state.get('steps', 0)} 工具调用={metrics.get('tool_calls')} "
        f"失败={metrics.get('tool_failed')} 重试={metrics.get('tool_retries')} "
        f"降级={metrics.get('tool_degraded')} RAG形态={rag_mode_used} ====="
    )
    print(f"===== trace：{tracer.path} =====")

    return RunResult(
        run_id=tracer.run_id,
        status=status,
        steps=int(state.get("steps", 0)),
        final_answer=final_answer,
        trace_path=str(tracer.path),
        elapsed_ms=elapsed_ms,
        rag_mode=rag_mode_used,
        metrics=metrics,
        error=error,
    )


def run_task_sync(task: str, config: Dict[str, Any] | None = None, rag_mode: str | None = None,
                  console: str | None = None) -> RunResult:
    """同步包装（给不方便用 asyncio 的场景）。"""
    return asyncio.run(run_task(task, config=config, rag_mode=rag_mode, console=console))
