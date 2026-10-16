"""工具执行层：超时 / 重试 / 降级 / 熔断 —— 本项目的可靠性核心。

失败分类（对应面试里的「失败如何分类」）：

| 类别 | 典型来源 | 处理策略 |
|---|---|---|
| **参数错误 / 工具不存在**（permanent） | 模型给了错参数、幻觉出一个不存在的工具 | **不重试**（重试同样的参数必然同样失败），把「正确用法」回给模型让它自己修 |
| **瞬时故障**（retryable） | 超时、连接失败、限流 429、5xx | **指数退避重试**，最多 max_attempts 次 |
| **重试仍失败**（exhausted） | 依赖长时间不可用 | **降级**：调用该工具注册的 fallback（如 RAG 的进程内直连检索）；没有 fallback 就返回「明确失败 + 可选出路」 |
| **同一工具连续失败**（熔断） | 依赖彻底坏了 | **本任务内临时禁用该工具**，在 observation 里告知模型不要再调，避免把步数浪费在必然失败的调用上 |

另外两条硬约束：
- **超时**：每次工具调用都有 `asyncio.wait_for` 超时，防止一个卡死的工具把整个任务拖死；
- **输出长度上限**：工具结果会直接进模型上下文，超长结果截断（截断处标注），把「上下文成本」握在自己手里。
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List

from langchain_core.tools import BaseTool

from agent.trace import Tracer

PREVIEW_CHARS = 1200  # 单个工具结果进入模型上下文前的长度上限


@dataclass
class ToolOutcome:
    """一次工具调用的最终结果（不抛异常，全部用结构化字段表达）。"""

    ok: bool
    tool: str
    result: str | None = None
    error: str | None = None
    error_class: str | None = None  # retryable | permanent | unknown_tool | circuit_open | timeout
    attempts: int = 1
    elapsed_ms: float = 0.0
    degraded: bool = False
    degraded_from: str | None = None
    attempts_detail: List[str] = field(default_factory=list)

    def to_observation(self) -> str:
        """转成喂回模型的 observation 文本：成功给结果，失败给「原因 + 可选出路」。"""
        if self.ok:
            prefix = ""
            if self.degraded:
                prefix = f"[降级结果] 主通道（{self.degraded_from}）不可用，已改用本地同源检索：\n"
            return prefix + (self.result or "")
        return self.error or "工具调用失败（无详细信息）"


def extract_tool_text(raw: Any) -> str:
    """把工具返回值统一成"喂给模型看的字符串"。

    ★ 真实的适配层细节（踩过）：`langchain-mcp-adapters` 包装的 MCP 工具返回的是
    **内容块列表**：`[{"type": "text", "text": "<JSON 字符串>"}]`。
    如果直接 `str()` 或 `json.dumps()`，模型看到的是带转义引号的 Python repr
    （`[{'type': 'text', 'text': '{\\"query\\": ...'}]`）——既费 token 又难读。
    所以这里要把 text 块拼回原文。
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: List[str] = []
        for block in raw:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        if parts:
            return "\n".join(parts)
    try:
        return json.dumps(raw, ensure_ascii=False, default=str)
    except TypeError:
        return str(raw)


def classify_error(exc: Exception, retryable_patterns: List[str]) -> str:
    """把异常分成 retryable / permanent。

    规则（保守优先）：
    - 超时/连接/限流/5xx → retryable；
    - 参数校验类错误（ValidationError / TypeError / ValueError）→ permanent，重试没意义；
    - 其余未知异常按 retryable 处理 —— 因为「重试一次」的代价远小于「直接放弃」。
    """
    name = type(exc).__name__.lower()
    message = f"{exc}".lower()

    permanent_names = ("validationerror", "typeerror", "valueerror", "keyerror", "attributeerror")
    if any(token in name for token in permanent_names):
        return "permanent"

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "retryable"

    haystack = f"{name} {message}"
    if any(pattern.lower() in haystack for pattern in retryable_patterns):
        return "retryable"
    return "retryable"


class ToolExecutor:
    """带重试/降级/熔断的工具执行器（每个任务一个实例，熔断状态不跨任务）。"""

    def __init__(
        self,
        tools: Dict[str, BaseTool],
        config: Dict[str, Any],
        tracer: Tracer,
        fallbacks: Dict[str, Callable[[Dict[str, Any]], Any]] | None = None,
    ):
        self.tools = tools
        self.config = config
        self.tracer = tracer
        self.fallbacks = fallbacks or {}

        tools_cfg = config["tools"]
        retry_cfg = tools_cfg["retry"]
        self.timeout_s = float(tools_cfg["timeout_s"])
        self.max_attempts = int(retry_cfg["max_attempts"])
        self.backoff_base_s = float(retry_cfg["backoff_base_s"])
        self.backoff_max_s = float(retry_cfg["backoff_max_s"])
        self.retryable_patterns = list(retry_cfg["retryable_patterns"])
        self.breaker_threshold = int(tools_cfg["circuit_breaker"]["failure_threshold"])

        self.consecutive_failures: Dict[str, int] = {}
        self.breaker_open: Dict[str, bool] = {}
        self.stats = {"calls": 0, "ok": 0, "failed": 0, "retries": 0, "degraded": 0, "circuit_opens": 0}

    # ---------- 对外入口 ----------

    async def execute(self, name: str, args: Dict[str, Any]) -> ToolOutcome:
        started = time.perf_counter()
        self.stats["calls"] += 1

        if name not in self.tools:
            self.stats["failed"] += 1
            outcome = ToolOutcome(
                ok=False,
                tool=name,
                error_class="unknown_tool",
                error=(
                    f"没有名为 `{name}` 的工具。可用工具：{sorted(self.tools)}。"
                    "请从这些工具里选择；如果都不适用，就直接说明无法完成。"
                ),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            self.tracer.event("observation", tool=name, ok=False, error_class=outcome.error_class,
                              error=outcome.error, attempts=0, elapsed_ms=outcome.elapsed_ms)
            return outcome

        if self.breaker_open.get(name):
            self.stats["failed"] += 1
            outcome = ToolOutcome(
                ok=False,
                tool=name,
                error_class="circuit_open",
                error=(
                    f"工具 `{name}` 在本任务中已被临时禁用（连续失败 {self.consecutive_failures.get(name, 0)} 次）。"
                    "请不要再次调用它：基于已有信息回答，或如实说明这一部分无法获取。"
                ),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            self.tracer.event("observation", tool=name, ok=False, error_class=outcome.error_class,
                              error=outcome.error, attempts=0, elapsed_ms=outcome.elapsed_ms)
            return outcome

        # ---- 主循环：尝试 → 失败分类 → 退避重试 ----
        last_error: str | None = None
        last_class: str | None = None
        attempts = 0

        while attempts < self.max_attempts:
            attempts += 1
            self.tracer.event("action", tool=name, args=args, attempt=attempts)
            try:
                raw = await asyncio.wait_for(self._invoke(name, args), timeout=self.timeout_s)
                text = self._stringify(raw)
                self._on_success(name)
                self.stats["ok"] += 1
                outcome = ToolOutcome(
                    ok=True,
                    tool=name,
                    result=text,
                    attempts=attempts,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                self._trace_observation(outcome, text)
                return outcome
            except asyncio.TimeoutError:
                last_error = f"工具 `{name}` 调用超时（>{self.timeout_s}s）"
                last_class = "timeout"
            except Exception as exc:  # noqa: BLE001 —— 这里就是要兜住一切，然后分类
                last_error = f"{type(exc).__name__}: {exc}"
                last_class = classify_error(exc, self.retryable_patterns)

            # 走到这里说明失败：记录 + 决定要不要重试
            if last_class == "permanent":
                self._on_failure(name)
                self.stats["failed"] += 1
                outcome = ToolOutcome(
                    ok=False,
                    tool=name,
                    error=(
                        f"工具 `{name}` 参数或用法有误（不重试）：{last_error}。"
                        "请修正参数后重新调用，或改用其他工具。"
                    ),
                    error_class="permanent",
                    attempts=attempts,
                    attempts_detail=[last_error or ""],
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                self._trace_observation(outcome)
                return outcome

            if attempts < self.max_attempts:
                self.stats["retries"] += 1
                backoff = min(self.backoff_base_s * (2 ** (attempts - 1)), self.backoff_max_s)
                self.tracer.event("retry", tool=name, attempt=attempts, error_class=last_class,
                                  error=last_error, sleep_s=round(backoff, 2))
                await asyncio.sleep(backoff)

        # ---- 重试耗尽：熔断计数 + 尝试降级 ----
        self._on_failure(name)
        self.stats["failed"] += 1

        if name in self.fallbacks:
            self.tracer.event("degrade", tool=name, reason=f"重试 {attempts} 次仍失败：{last_error}")
            try:
                raw = await asyncio.wait_for(self._call_fallback(name, args), timeout=self.timeout_s)
                text = self._stringify(raw)
                self.stats["degraded"] += 1
                self._on_success(name)  # 降级成功 → 重置该工具的连续失败计数
                outcome = ToolOutcome(
                    ok=True,
                    tool=name,
                    result=text,
                    attempts=attempts,
                    degraded=True,
                    degraded_from="MCP stdio 通道",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                self._trace_observation(outcome, text)
                return outcome
            except Exception as exc:  # noqa: BLE001
                self.tracer.event("degrade", tool=name, reason=f"降级路径同样失败：{type(exc).__name__}: {exc}")

        outcome = ToolOutcome(
            ok=False,
            tool=name,
            error_class=last_class,
            error=(
                f"工具 `{name}` 已重试 {attempts} 次仍失败（{last_class}）：{last_error}。"
                "请不要再重试同样的调用：可以换参数/换工具再试一次，或基于已有信息回答，并如实说明这部分无法获取。"
            ),
            attempts=attempts,
            attempts_detail=[last_error or ""],
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        self._trace_observation(outcome)
        return outcome

    # ---------- 内部 ----------

    async def _invoke(self, name: str, args: Dict[str, Any]) -> Any:
        tool = self.tools[name]
        # BaseTool.ainvoke 对「只有同步实现」的工具会自动丢到线程池，这里统一用 ainvoke
        return await tool.ainvoke(args)

    async def _call_fallback(self, name: str, args: Dict[str, Any]) -> Any:
        fallback = self.fallbacks[name]
        if asyncio.iscoroutinefunction(fallback):
            return await fallback(args)
        result = fallback(args)
        if asyncio.iscoroutine(result):  # 懒加载的异步包装
            return await result
        return result

    @staticmethod
    def _stringify(raw: Any) -> str:
        """工具返回可能是 str / dict / list（含 LangChain 的内容块），统一成字符串并限长。"""
        text = extract_tool_text(raw)
        if len(text) > PREVIEW_CHARS:
            text = text[:PREVIEW_CHARS] + f"\n...（结果过长已截断，原长 {len(text)} 字符）"
        return text

    def _on_success(self, name: str) -> None:
        self.consecutive_failures[name] = 0

    def _on_failure(self, name: str) -> None:
        count = self.consecutive_failures.get(name, 0) + 1
        self.consecutive_failures[name] = count
        if count >= self.breaker_threshold and not self.breaker_open.get(name):
            self.breaker_open[name] = True
            self.stats["circuit_opens"] += 1
            self.tracer.event("circuit_open", tool=name, consecutive_failures=count)

    def _trace_observation(self, outcome: ToolOutcome, result_text: str | None = None) -> None:
        self.tracer.event(
            "observation",
            tool=outcome.tool,
            ok=outcome.ok,
            error_class=outcome.error_class,
            error=outcome.error,
            attempts=outcome.attempts,
            degraded=outcome.degraded,
            elapsed_ms=outcome.elapsed_ms,
            result_preview=(result_text or None),
        )
