"""单条任务的执行入口：组装配置 → 工具 → 模型 → 图 → 跑 → 收尾落 trace。"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from agent.checkpointing import open_checkpointer
from agent.config import describe_llm, get_llm_settings, load_config
from agent.graph import build_agent_graph
from agent.llm import build_llm
from agent.memory import build_memory_context
from agent.resilience import ToolExecutor
from agent.toolkit import build_toolset
from agent.trace import Tracer
from agent.usage import format_usage_line, load_pricing, summarize_usage

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
    # 最终回答的来源：model（模型真实回答）| system_fallback（模型调用失败的系统兜底文案）
    # | max_steps_explanation（步数用尽的收尾说明）| run_error（异常收尾）
    # 评测侧靠它把「系统兜底」和「模型真实回答」区分开（见 agent/eval_harness.py）。
    final_answer_source: str = "model"
    # 断点续跑相关（不指定 thread_id 时 thread_id=None / resumed=False）
    thread_id: str | None = None
    resumed: bool = False
    resumed_from_step: int = 0
    usage: Dict[str, Any] = field(default_factory=dict)


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

    # 上下文压缩：记录压缩次数、最大节省字符数、最近一次的 before/after
    context_events = [event for event in tracer.events if event["type"] == "context_compressed"]
    context_saved = max(
        (
            int(event.get("before_chars", 0)) - int(event.get("after_chars", 0))
            for event in context_events
        ),
        default=0,
    )

    return {
        "llm_calls": counts.get("thought", 0),
        "tool_calls": tool_calls,
        "tool_ok": executor.stats["ok"],
        "tool_failed": executor.stats["failed"],
        "tool_retries": executor.stats["retries"],
        "tool_degraded": executor.stats["degraded"],
        "circuit_opens": executor.stats["circuit_opens"],
        "tool_success_rate": round(executor.stats["ok"] / tool_calls, 4) if tool_calls else None,
        "context_compressions": len(context_events),
        "context_saved_chars": context_saved,
        "context_last_before_chars": context_events[-1].get("before_chars") if context_events else None,
        "context_last_after_chars": context_events[-1].get("after_chars") if context_events else None,
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


async def _detect_resume(graph, invoke_config: Dict[str, Any], tracer: Tracer,
                         thread_id: str, resume: bool) -> tuple[bool, int, bool]:
    """判断这个 thread 上的 checkpoint 处于什么状态。

    判据用 LangGraph 自己的 `StateSnapshot.next`：**有下一个待执行节点 = 上次没跑完**
    （节点执行完但进程被杀/Ctrl-C，图停在中间）。`next` 为空说明上次已到 END。
    这个判据比读 `status` 字段可靠：`max_steps_exceeded` 收尾时 status 也是 in_progress，
    但图已经到 END，不该被当成「可以续跑」。

    返回 `(是否续跑, 已完成的步数, 这个 thread 之前有没有消息)`。
    """
    snapshot = await graph.aget_state(invoke_config)
    if snapshot is None:
        return False, 0, False
    pending_nodes = list(getattr(snapshot, "next", ()) or ())
    values = snapshot.values or {}
    steps_before = int(values.get("steps", 0) or 0)
    has_history = bool(values.get("messages"))
    tracer.step = steps_before  # trace 的步号接着上次，复盘时能看出「第几步是续跑/新一轮」
    if pending_nodes:
        tracer.event(
            "resume_detected",
            thread_id=thread_id,
            next_nodes=pending_nodes,
            steps_done=steps_before,
            messages_done=len(values.get("messages") or []),
        )
        return True, steps_before, has_history
    if resume:
        tracer.event(
            "resume_noop",
            thread_id=thread_id,
            reason="该 thread 没有未完成的 checkpoint（上次已到 END），本次按新任务开始",
        )
    return False, steps_before, has_history


async def run_task(
    task: str,
    config: Dict[str, Any] | None = None,
    rag_mode: str | None = None,
    console: str | None = None,
    thread_id: str | None = None,
    resume: bool = False,
) -> RunResult:
    """跑一条任务，返回结构化结果（trace 落在 traces/ 下）。

    thread_id：给了就挂 checkpointer（断点续跑），且本次运行属于这个 thread；
        若该 thread 上有「没跑完」的状态（上次中断/Ctrl-C），**自动从最后一个完成的步骤续跑**，
        input 传 None 让 LangGraph 只补跑未完成的节点 → 已完成的工具调用不会被重放。
    resume：把「续跑」这件事显式化（CLI `--resume`）。即使不加，只要检测到未完成状态也会续跑，
        因为这个参数只影响「没有可续跑状态时」的提示行为，不影响正常任务。
    """
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
        thread_id=thread_id,
    )

    status = STATUS_ERROR
    error: str | None = None
    state: Dict[str, Any] = {}
    rag_mode_used = "none"
    metrics: Dict[str, Any] = {}
    usage: Dict[str, Any] = {}
    executor: ToolExecutor | None = None
    checkpoint_backend = "disabled"
    resumed = False
    steps_before = 0

    try:
        async with build_toolset(config, tracer, rag_mode) as toolset:
            rag_mode_used = toolset.rag_mode

            # 跨任务记忆：只注入与本次任务关键词相关的 top-k 条，避免污染上下文
            memory_cfg = config.get("memory", {}) or {}
            memory_context, memory_keys = build_memory_context(
                toolset.memory,
                task,
                top_k=int(memory_cfg.get("inject_top_k", 3)),
                max_chars=int(memory_cfg.get("max_inject_chars", 800)),
            )
            if toolset.memory is not None:
                tracer.event(
                    "memory_injected",
                    count=len(memory_keys),
                    keys=memory_keys,
                    chars=len(memory_context),
                    total_records=len(toolset.memory),
                )

            executor = ToolExecutor(toolset.tools, config, tracer, toolset.fallbacks)

            # ★ 断点续跑：只有指定了 thread_id 才挂 checkpointer（默认路径行为完全不变）
            async with open_checkpointer(config, enabled=thread_id is not None) as (
                checkpointer,
                checkpoint_backend,
                checkpoint_note,
            ):
                tracer.event(
                    "checkpoint_ready",
                    backend=checkpoint_backend,
                    thread_id=thread_id,
                    path=str(config["agent"].get("checkpoint", {}).get("path")),
                    note=checkpoint_note,
                )
                llm = build_llm(settings)
                graph = build_agent_graph(
                    llm, executor, tracer, config, memory_context=memory_context,
                    checkpointer=checkpointer,
                )

                invoke_config: Dict[str, Any] = {
                    "recursion_limit": int(config["agent"]["recursion_limit"])
                }
                graph_input: Any = {
                    "messages": [HumanMessage(content=task)],
                    "steps": 0,
                    "status": "new",
                }

                if checkpointer is not None:
                    invoke_config["configurable"] = {"thread_id": thread_id}
                    resumed, steps_before, has_history = await _detect_resume(
                        graph, invoke_config, tracer, str(thread_id), resume
                    )
                    if resumed:
                        # input=None：LangGraph 从最后一个 checkpoint 继续，只补跑未完成的节点
                        graph_input = None
                    elif has_history:
                        # 上一轮已收尾，这次是同一 thread 上的新一轮：只追加消息，
                        # **不要带 steps/status**（那会把通道值重置，步数统计就断了）
                        graph_input = {"messages": [HumanMessage(content=task)]}

                state = await graph.ainvoke(graph_input, config=invoke_config)

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
    final_answer_source = "model"
    if status == STATUS_MAX_STEPS:
        final_answer = _failure_explanation(state, int(config["agent"]["max_steps"]))
        final_answer_source = "max_steps_explanation"
    elif last_ai is not None:
        final_answer = _text_of(last_ai).strip()
        # 模型调用失败时 graph 给兜底文案打了 answer_source 标记 → 这里原样传出去，
        # 让评测侧知道这条不是「模型真实回答」（否则失败运行会被算成诚实拒答）。
        source = (getattr(last_ai, "additional_kwargs", None) or {}).get("answer_source")
        final_answer_source = str(source) if source else "model"
    else:
        final_answer = error or "（没有产生任何回答）"
        final_answer_source = "run_error"

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics = _summarize_metrics(tracer, executor) if executor else {}

    # token / 成本：从 trace 的 llm_call 事件聚合（纯函数，事后可对 trace 文件重算）
    usage = summarize_usage(tracer.events, model=settings["model"], pricing=load_pricing(config))
    total_steps = int(state.get("steps", 0))
    steps_this_run = max(0, total_steps - steps_before)
    metrics.update(
        {
            "usage": usage,
            "thread_id": thread_id,
            "checkpoint_backend": checkpoint_backend,
            "resumed": resumed,
            "steps_resumed_from": steps_before if resumed else 0,
            "steps_this_run": steps_this_run,
        }
    )

    tracer.event(
        "run_end",
        status=status,
        steps=total_steps,
        steps_this_run=steps_this_run,
        rag_mode=rag_mode_used,
        elapsed_ms=elapsed_ms,
        final_answer=final_answer,
        metrics=metrics,
        thread_id=thread_id,
        resumed=resumed,
        resumed_from_step=(steps_before if resumed else 0),
        checkpoint_backend=checkpoint_backend,
        trace_path=str(tracer.path),
        error=error,
        answer_source=final_answer_source,
    )

    print(f"\n===== 最终回答（status={status}）=====")
    print(final_answer)
    print(
        f"===== 指标：步数={total_steps}（本次执行 {steps_this_run}）"
        f" 工具调用={metrics.get('tool_calls')} "
        f"失败={metrics.get('tool_failed')} 重试={metrics.get('tool_retries')} "
        f"降级={metrics.get('tool_degraded')} 上下文压缩={metrics.get('context_saved_chars')}字符 "
        f"RAG形态={rag_mode_used} ====="
    )
    print(format_usage_line(usage, extra=f" | 步数={total_steps} status={status}"))
    if thread_id:
        print(
            f"===== 断点续跑：thread_id={thread_id} 后端={checkpoint_backend} "
            f"本次是否续跑={'是（从第 %d 步后继续）' % steps_before if resumed else '否'} ====="
        )
    print(f"===== trace：{tracer.path} =====")

    return RunResult(
        run_id=tracer.run_id,
        status=status,
        steps=total_steps,
        final_answer=final_answer,
        trace_path=str(tracer.path),
        elapsed_ms=elapsed_ms,
        rag_mode=rag_mode_used,
        metrics=metrics,
        error=error,
        final_answer_source=final_answer_source,
        thread_id=thread_id,
        resumed=resumed,
        resumed_from_step=(steps_before if resumed else 0),
        usage=usage,
    )


def run_task_sync(task: str, config: Dict[str, Any] | None = None, rag_mode: str | None = None,
                  console: str | None = None, thread_id: str | None = None,
                  resume: bool = False) -> RunResult:
    """同步包装（给不方便用 asyncio 的场景）。"""
    return asyncio.run(
        run_task(task, config=config, rag_mode=rag_mode, console=console,
                 thread_id=thread_id, resume=resume)
    )
