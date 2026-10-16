"""trace 测试：JSONL 落地、字段契约、密钥不外泄、长度截断。"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.trace import Tracer  # noqa: E402


def read_events(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_trace_writes_jsonl_with_required_fields(tmp_path):
    tracer = Tracer(tmp_path, console="quiet", preview_chars=50, run_id="testrun")
    tracer.event("run_start", task="demo", llm="model=m base_url=http://x api_key=sk-****", max_steps=3, rag_mode="mcp")
    tracer.next_step()
    tracer.event("thought", content="我要先检索", tool_calls=[{"name": "search_knowledge", "args": {"query": "LoRA"}}],)
    tracer.event("observation", tool="search_knowledge", ok=True, attempts=1, elapsed_ms=12.3, result_preview="{}")
    tracer.event("run_end", status="completed", steps=1, final_answer="答案")

    events = read_events(tracer.path)
    assert [e["type"] for e in events] == ["run_start", "thought", "observation", "run_end"]
    for event in events:
        assert {"ts", "run_id", "step", "type"} <= set(event)
        assert event["run_id"] == "testrun"
    assert events[1]["step"] == 1 and events[1]["tool_calls"][0]["name"] == "search_knowledge"
    assert events[3]["final_answer"] == "答案"


def test_trace_never_leaks_api_key(tmp_path, monkeypatch):
    """★ 安全断言：哪怕代码里把密钥拼进了日志，落盘的 trace 里也必须只有 sk-****。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value-123456")
    tracer = Tracer(tmp_path, console="quiet", run_id="mask")

    tracer.event("run_start", task="demo", llm="endpoint with sk-super-secret-value-123456 inside", max_steps=1)
    tracer.event("observation", tool="llm", error="auth failed with key sk-super-secret-value-123456")

    raw = tracer.path.read_text(encoding="utf-8")
    assert "sk-super-secret-value-123456" not in raw
    assert raw.count("sk-****") >= 2
    for event in read_events(tracer.path):
        assert "sk-super-secret-value-123456" not in json.dumps(event, ensure_ascii=False)


def test_console_preview_is_truncated(tmp_path, capsys):
    tracer = Tracer(tmp_path, console="summary", preview_chars=20, run_id="clip")
    tracer.next_step()
    tracer.event("thought", content="这是一段很长的思考内容" * 10, tool_calls=[])

    out = capsys.readouterr().out
    assert "截断" in out
    assert len(out.splitlines()[0]) < 200


def test_one_file_per_run(tmp_path):
    first = Tracer(tmp_path, console="quiet")
    second = Tracer(tmp_path, console="quiet")
    first.event("run_start", task="a", llm="x", max_steps=1)
    second.event("run_start", task="b", llm="x", max_steps=1)
    assert first.path != second.path
    assert len(list(tmp_path.glob("run_*.jsonl"))) == 2
