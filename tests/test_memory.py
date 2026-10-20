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


def test_secret_detection_uses_shape_not_substring(tmp_path):
    """★ 误拒修复：子串匹配 "sk-" 会把 risk-based / disk-based 这类正常英文词当成密钥。

    现在按**形状**判断（`sk-` + 至少 16 位字母数字，且前面是词边界），
    正常英文词不再被拒，形似密钥的串仍然被拒。
    """
    store = MemoryStore(make_config(tmp_path))

    # 正常内容里含 "sk-"：必须能写进去（记忆是长期数据，误拒比漏判更伤）
    for content in (
        "用户偏好 risk-based 的评估流程",
        "部署在 disk-based 存储上",
        "任务标记是 task-based 的",
    ):
        store.remember(content)
    assert len(store) == 3

    # 形似密钥的串：仍然拒绝
    with pytest.raises(ValueError):
        store.remember("我的 key 是 sk-1234567890abcdef")
    with pytest.raises(ValueError):
        store.remember("token=sk-abcdefghijklmnopqrst")
    with pytest.raises(ValueError):
        store.remember("OPENAI_API_KEY=whatever")


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


def test_bm25_recall_on_labeled_set(tmp_path):
    """6 条记忆 / 6 个查询：BM25 Recall@1 应为 6/6。"""
    store = MemoryStore(make_config(tmp_path))
    memories = [
        ("lang", "用户的默认语言是中文"),
        ("project", "项目路径是 D:/programmer/LLM"),
        ("db", "数据库使用 PostgreSQL 16"),
        ("deploy", "服务部署在 Docker 容器里"),
        ("tz", "用户所在时区是 Asia/Shanghai"),
        ("editor", "常用编辑器是 VSCode"),
    ]
    for key, content in memories:
        store.remember(content, key=key)

    cases = [
        ("我的默认语言是什么", "lang"),
        ("项目路径在哪", "project"),
        ("用的什么数据库", "db"),
        ("服务怎么部署的", "deploy"),
        ("我的时区是什么", "tz"),
        ("常用编辑器是哪个", "editor"),
    ]
    hits = 0
    for query, expected in cases:
        results = store.recall(query, top_k=1)
        assert results, f"没有召回任何记忆：{query}"
        if results[0]["key"] == expected:
            hits += 1
    assert hits == len(cases), f"BM25 Recall@1 = {hits}/{len(cases)}"


def test_overlap_mode_still_available(tmp_path):
    """保留旧的 overlap 模式，便于做 A/B 对照。"""
    config = make_config(tmp_path, recall_mode="overlap")
    store = MemoryStore(config)
    store.remember("用户的默认语言是中文", key="lang")
    store.remember("项目路径是 D:/programmer/LLM", key="project")

    hits = store.recall("默认语言", top_k=1)
    assert hits[0]["key"] == "lang"


def test_bm25_score_is_positive(tmp_path):
    store = MemoryStore(make_config(tmp_path))
    store.remember("数据库使用 PostgreSQL 16", key="db")
    hits = store.recall("数据库", top_k=1)
    assert hits[0]["score"] > 0


def test_bm25_beats_overlap_on_long_distractor(tmp_path):
    """长度归一化场景：两条记忆命中同一组关键词，只差文档长度。

    - target 很短：「数据库 配置」；
    - distractor 在一大段无关内容里也提到「数据库 配置」；
    - overlap 只看命中词比例 → 两者打平，按写入顺序选中 distractor；
    - BM25 有文档长度归一化（b=0.75）→ 短而聚焦的 target 分数更高。

    这是「为什么要把字面重叠升级成 BM25」的可复现证据。
    """
    target = ("target", "数据库 配置")
    distractor = ("distractor", "数据库 配置 " + "无关内容" * 60)
    query = "数据库 配置"

    store_bm25 = MemoryStore(make_config(tmp_path / "bm25"))
    store_overlap = MemoryStore(
        make_config(tmp_path / "overlap", recall_mode="overlap")
    )
    # 先写 distractor：overlap 打平时按写入顺序取第一个
    for store in (store_bm25, store_overlap):
        store.remember(distractor[1], key=distractor[0])
        store.remember(target[1], key=target[0])

    bm25_top = store_bm25.recall(query, top_k=1)[0]
    overlap_top = store_overlap.recall(query, top_k=1)[0]

    assert bm25_top["key"] == "target", f"BM25 未选中短而精确的记忆：{bm25_top}"
    assert overlap_top["key"] == "distractor", (
        f"overlap 未被长文档带偏（本用例失去对照意义）：{overlap_top}"
    )
