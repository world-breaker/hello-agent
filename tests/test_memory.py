"""跨任务长期记忆单测（离线、无 LLM）：持久化 / 覆盖 / 召回 / 工具 / 注入。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import load_config  # noqa: E402
from agent.memory import (  # noqa: E402
    MemoryStore,
    build_memory_context,
    build_memory_tools,
)


def make_config(tmp_path: Path, **overrides) -> dict:
    config = load_config()
    config["memory"].update(
        {
            "enabled": True,
            "path": str(tmp_path / "agent_memory.jsonl"),
            "max_records": 50,
            "max_content_chars": 200,
        }
    )
    config["memory"].update(overrides)
    return config


def test_remember_and_recall_persist_across_instances(tmp_path):
    config = make_config(tmp_path)

    store1 = MemoryStore(config)
    store1.remember("用户的默认语言是中文", key="lang", tags=["偏好"])
    store1.remember("项目路径在 D:/programmer/LLM", key="project_path")

    # 新实例从同一个 JSONL 文件恢复（模拟进程重启）
    store2 = MemoryStore(config)
    assert len(store2) == 2

    hits = store2.recall("用户的语言偏好是什么", top_k=2)
    assert hits
    assert hits[0]["key"] == "lang"
    assert "中文" in hits[0]["content"]


def test_same_key_overwrites(tmp_path):
    config = make_config(tmp_path)
    store = MemoryStore(config)

    store.remember("旧值：默认语言英文", key="lang")
    store.remember("新值：默认语言中文", key="lang")

    assert len(store) == 1
    hits = store.recall("默认语言", top_k=5)
    assert hits[0]["content"] == "新值：默认语言中文"

    # 落盘后也只保留最新视图
    reloaded = MemoryStore(config)
    assert len(reloaded) == 1
    assert reloaded.all()[0]["content"].endswith("中文")


def test_empty_and_secret_content_rejected(tmp_path):
    store = MemoryStore(make_config(tmp_path))

    with pytest.raises(ValueError):
        store.remember("   ")

    with pytest.raises(ValueError):
        store.remember("我的 key 是 sk-1234567890abcdef")


def test_recall_without_overlap_returns_empty(tmp_path):
    store = MemoryStore(make_config(tmp_path))
    store.remember("Transformer 的自注意力机制", key="transformer")

    assert store.recall("今天武汉天气如何") == []


def test_memory_tools_roundtrip(tmp_path):
    store = MemoryStore(make_config(tmp_path))
    tools = build_memory_tools(store)
    assert set(tools) == {"remember_fact", "recall_memory"}

    saved = json.loads(
        tools["remember_fact"].invoke(
            {"content": "回答时要先检索知识库", "key": "rule", "tags": "规则,检索"}
        )
    )
    assert saved["ok"] is True
    assert saved["key"] == "rule"

    recalled = json.loads(
        tools["recall_memory"].invoke({"query": "检索 规则", "top_k": 3})
    )
    assert recalled["ok"] is True
    assert recalled["count"] >= 1
    assert recalled["results"][0]["key"] == "rule"


def test_build_memory_context_formats_and_truncates(tmp_path):
    config = make_config(tmp_path, max_inject_chars=120)
    store = MemoryStore(config)
    store.remember("用户的默认语言是中文，回答要用中文", key="lang")
    store.remember("用户的时区是 Asia/Shanghai", key="tz")

    context, keys = build_memory_context(store, "请用中文回答，时区是什么", top_k=2, max_chars=120)
    assert "跨任务记忆" in context
    assert "lang" in keys
    assert len(context) <= 140  # 120 + 截断标记的余量

    empty_context, empty_keys = build_memory_context(store, "足球比赛结果", top_k=2)
    assert empty_context == ""
    assert empty_keys == []


def test_disabled_memory_does_nothing(tmp_path):
    config = make_config(tmp_path, enabled=False)
    store = MemoryStore(config)
    assert len(store) == 0
    assert build_memory_context(store, "任何问题") == ("", [])
