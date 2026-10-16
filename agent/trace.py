"""结构化 trace：每一步的 thought / action / observation 落 JSONL 文件 + 控制台摘要。

设计目标（对应面试里的「链路追踪怎么设计」）：
- **一次任务一个文件**：`traces/run_<时间戳>_<短 id>.jsonl`，跑完即可回放整个决策链；
- **一条事件一行 JSON**：便于 grep / jq / 后续做指标统计（平均步数、工具成功率、失败恢复率）；
- **统一字段**：ts / run_id / step / type + 各类型自己的字段，type 取值：
  run_start / llm_call / thought / action / retry / observation / degrade / circuit_open / run_end；
- **不留敏感数据**：所有内容过 `mask_secret()`，密钥永远进不了 trace；
- **长度可控**：超长内容截断到 preview_chars，既省磁盘也省事后阅读成本。
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from agent.config import mask_secret


class Tracer:
    """JSONL trace 记录器（同步写，单任务串行执行，无需锁）。"""

    def __init__(
        self,
        trace_dir: str | Path,
        console: str = "summary",
        preview_chars: int = 200,
        run_id: str | None = None,
    ):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = self.trace_dir / f"run_{stamp}_{self.run_id}.jsonl"
        self.console = console  # summary | verbose | quiet
        self.preview_chars = int(preview_chars)
        self.step = 0
        self.events: list[Dict[str, Any]] = []
        self.started = time.perf_counter()

    # ---------- 事件写入 ----------

    def event(self, type: str, **fields: Any) -> Dict[str, Any]:
        """追加一条事件：写文件 + 按需打印控制台摘要。"""
        payload = {
            "ts": round(time.time(), 3),
            "run_id": self.run_id,
            "step": self.step,
            "type": type,
            **{k: _clean(v) for k, v in fields.items()},
        }
        self.events.append(payload)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._console(payload)
        return payload

    def next_step(self) -> int:
        """进入下一步（每轮「模型思考 + 工具执行」算一步）。"""
        self.step += 1
        return self.step

    # ---------- 控制台摘要 ----------

    def _console(self, payload: Dict[str, Any]) -> None:
        if self.console == "quiet":
            return
        t = payload["type"]
        step = payload["step"]
        preview = self.preview_chars

        if t == "run_start":
            print(f"[run {self.run_id}] 任务：{payload['task']}")
            print(f"[run {self.run_id}] 端点：{payload['llm']} | max_steps={payload['max_steps']} | rag_mode={payload.get('rag_mode')}")
        elif t == "tools_ready":
            print(f"[run {self.run_id}] 工具就绪：{payload['tools']} | RAG 形态={payload['rag_mode']}（{payload.get('rag_note')}）")
        elif t == "thought":
            text = payload.get("content") or "(无文本，只发起了工具调用)"
            print(f"[step {step}] thought: {_clip(text, preview)}")
            for call in payload.get("tool_calls") or []:
                print(f"[step {step}]   → action: {call['name']}({_clip(json.dumps(call['args'], ensure_ascii=False), preview)})")
        elif t == "observation":
            status = "ok" if payload.get("ok") else f"FAIL({payload.get('error_class')})"
            print(
                f"[step {step}] observation[{payload['tool']}] {status} "
                f"in {payload.get('elapsed_ms')}ms (attempt={payload.get('attempts')}): "
                f"{_clip(payload.get('result_preview') or payload.get('error') or '', preview)}"
            )
        elif t == "retry":
            print(f"[step {step}] retry[{payload['tool']}] 第 {payload['attempt']} 次失败（{payload['error_class']}）：{_clip(payload['error'], preview)}")
        elif t == "degrade":
            print(f"[step {step}] degrade[{payload['tool']}] 重试耗尽 → 降级：{_clip(payload['reason'], preview)}")
        elif t == "circuit_open":
            print(f"[step {step}] circuit_open[{payload['tool']}] 连续失败 {payload['consecutive_failures']} 次，本任务内不再调用")
        elif t == "llm_call":
            usage = payload.get("usage") or {}
            tokens = f" tokens=in{usage.get('input_tokens', '?')}/out{usage.get('output_tokens', '?')}" if usage else ""
            print(f"[step {step}] llm_call {payload['elapsed_ms']}ms{tokens}")
        elif t == "run_end":
            print(
                f"[run {self.run_id}] 结束：status={payload['status']} steps={payload['steps']} "
                f"耗时={payload['elapsed_ms']}ms trace={payload['trace_path']}"
            )
            if self.console == "verbose":
                print(f"[run {self.run_id}] 最终回答：{payload.get('final_answer')}")

    def close(self) -> Path:
        return self.path


def _clean(value: Any) -> Any:
    """递归地：抹掉密钥 + 保证可 JSON 序列化。"""
    if isinstance(value, str):
        return mask_secret(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return mask_secret(str(value))


def _clip(text: str, limit: int) -> str:
    text = mask_secret(str(text)).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + f"...（截断，共 {len(text)} 字符）"
