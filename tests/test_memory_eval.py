"""记忆评测（该记 / 不该记 / 召回错）的离线单测。

覆盖三件事：
1. **打分口径**：`score_memory` 对三类标注的判定（正例 + 反例各至少一条）；
2. **记忆状态从哪来**：从 trace（`remember_fact` 调用与结果、`recall_memory` 返回、
   `memory_injected` 事件）与记忆库 JSONL 快照重算，**不用重跑**；
3. **向后兼容**：旧的 50 条用例不带新字段 → 不进记忆分母、判定结果不变。
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.eval_harness import (  # noqa: E402
    EvalRecord,
    MemoryState,
    evaluate,
    expected_memory_absent,
    expected_memory_content_contains,
    expected_memory_wrong_not_used,
    has_memory_ground_truth,
    load_eval_set,
    memory_state_from_store,
    memory_state_from_trace,
    merge_memory_states,
    score_answer,
    score_memory,
)

EVAL_SET_PATH = REPO_ROOT / "data" / "eval_queries.json"
MEMORY_FIELDS = (
    "expect_memory_content_contains",
    "expect_memory_absent",
    "expect_memory_wrong_not_used",
)


# ---------------- 小工具 ----------------


def _rec(query=None, answer="", state=None, status="completed", used=None):
    """构造一条 EvalRecord（默认回答来自模型）。"""
    return EvalRecord(
        query=query or {},
        status=status,
        steps=2,
        tool_calls=1,
        tool_ok=1,
        tool_failed=0,
        tool_retries=0,
        tool_degraded=0,
        circuit_opens=0,
        tool_success_rate=1.0,
        used_tools=used if used is not None else [],
        used_tools_ok=list(used or []),
        final_answer=answer,
        final_answer_source="model",
        memory_state=state,
    )


def _write_trace(tmp_path: Path, events: list) -> Path:
    path = tmp_path / "run.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return path


def _write_store(tmp_path: Path, records: list) -> Path:
    path = tmp_path / "memory.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# ---------------- 1. 该记：期望记忆里出现该内容 ----------------


def test_should_remember_hit_when_memory_contains_content():
    """正例：任务要求记住的事实真的写进了长期记忆。"""
    query = {"expect_memory_content_contains": ["中文"]}
    state = MemoryState(written_contents=["以后所有回答都用中文"], source="trace")

    s = score_memory(query, "已记住，以后用中文回答。", state)

    assert s["checked"] is True
    assert s["remember_hits"] == ["中文"]
    assert s["remember_missed"] == []
    assert s["write_hits"] == ["中文"]          # 确实是「写进去」的，不是本来就有
    assert s["write_missed"] == []
    assert s["pass"] is True


def test_should_remember_miss_when_nothing_was_written():
    """反例：任务明确要求记住，但记忆里没有 → 判失败。"""
    query = {"expect_memory_content_contains": ["D:/projects/hello-agent"]}
    state = MemoryState(written_contents=["今天天气不错"], source="trace")

    s = score_memory(query, "好的。", state)

    assert s["remember_missed"] == ["D:/projects/hello-agent"]
    assert s["remember_ok"] is False
    assert s["write_missed"] == ["D:/projects/hello-agent"]
    assert s["pass"] is False


def test_should_remember_counts_existing_store_content():
    """记忆库里**原本就有**该内容 → 也算「记忆里有」（但不算「这次写进去了」）。"""
    query = {"expect_memory_content_contains": ["中文"]}
    state = MemoryState(store_contents=["用户的回答语言偏好是中文"], source="store")

    s = score_memory(query, "好的。", state)

    assert s["remember_hits"] == ["中文"]
    assert s["write_hits"] == []          # 没有写入动作
    assert s["write_ok"] is False
    assert s["remember_ok"] is True


# ---------------- 2. 不该记：期望记忆里不出现该内容 ----------------


def test_should_not_remember_pass_when_absent_from_memory():
    """正例：一次性凭据没有进长期记忆。"""
    query = {"expect_memory_absent": ["tmp-8f3a9c2b7d6e", "临时访问令牌"]}
    state = MemoryState(written_contents=["用户偏好中文回答"], source="trace")

    s = score_memory(query, "7×8=56，令牌我只在这次用了。", state)

    assert s["absent_violations"] == []
    assert s["absent_ok"] is True
    assert s["pass"] is True


def test_should_not_remember_violation_when_written_into_memory():
    """反例：不该进长期记忆的东西被写进去了 → 判失败。"""
    query = {"expect_memory_absent": ["tmp-8f3a9c2b7d6e"]}
    state = MemoryState(written_contents=["临时访问令牌 tmp-8f3a9c2b7d6e"], source="trace")

    s = score_memory(query, "已记住。", state)

    assert s["absent_violations"] == ["tmp-8f3a9c2b7d6e"]
    assert s["absent_ok"] is False
    assert s["pass"] is False


def test_failed_write_is_not_a_violation_but_is_reported_as_attempted():
    """★ 发起过但**没写进去**（例如被「疑似密钥」规则拒了）→ 不算违规，但要报出来。

    口径：违规 = 真的进了长期记忆；「试了一下没写成」是另一回事。
    """
    query = {"expect_memory_absent": ["tmp-8f3a9c2b7d6e"]}
    state = MemoryState(
        attempted_contents=["临时访问令牌 tmp-8f3a9c2b7d6e"],
        written_contents=[],
        source="trace",
    )

    s = score_memory(query, "好的。", state)

    assert s["absent_violations"] == []
    assert s["absent_attempted"] == ["tmp-8f3a9c2b7d6e"]
    assert s["pass"] is True


# ---------------- 3. 召回错：记忆里有但不该被当事实用 ----------------


def test_wrong_memory_not_used_pass_when_answer_ignores_it():
    """正例：记忆里有旧路径，但回答用的是新路径 → 合格。"""
    query = {"expect_memory_wrong_not_used": ["D:/old-project"]}
    state = MemoryState(store_contents=["我的项目根目录是 D:/old-project"], source="store")

    s = score_memory(query, "已记住：以后项目根目录是 D:/work/new-repo。", state)

    assert s["wrong_present"] == ["D:/old-project"]     # 陷阱确实存在
    assert s["wrong_used"] == []
    assert s["wrong_ok"] is True
    assert s["pass"] is True


def test_wrong_memory_used_is_a_violation():
    """反例：直接拿记忆里的旧内容当事实 → 判失败。"""
    query = {"expect_memory_wrong_not_used": ["D:/old-project"]}
    state = MemoryState(store_contents=["我的项目根目录是 D:/old-project"], source="store")

    s = score_memory(query, "你的项目根目录是 D:/old-project。", state)

    assert s["wrong_used"] == ["D:/old-project"]
    assert s["wrong_ok"] is False
    assert s["pass"] is False


def test_wrong_memory_mentioned_but_corrected_is_not_a_violation():
    """★ 提到旧内容但**同一句**里说明了「以本次为准」→ 不算拿它当事实。

    这是「不依赖它」的判定：只看「答案里有没有那段文字」会把
    「正确指出冲突」误判成违规。
    """
    query = {"expect_memory_wrong_not_used": ["D:/old-project"]}
    state = MemoryState(store_contents=["我的项目根目录是 D:/old-project"], source="store")

    s = score_memory(
        query,
        "记忆里是 D:/old-project，但你这次给的是 D:/work/new-repo，以本次为准。",
        state,
    )

    assert s["wrong_used"] == []
    assert s["wrong_mentioned_corrected"] == ["D:/old-project"]
    assert s["pass"] is True


def test_wrong_memory_absent_from_memory_is_not_scored():
    """★ 记忆里**根本没有**那条错误内容 → 陷阱没踩到，不进这一项的分母。

    口径：`memory_wrong_present_n` 只数「确实存在」的那些，分母才不会被虚高。
    """
    query = {"expect_memory_wrong_not_used": ["D:/old-project"]}
    state = MemoryState(store_contents=["用户的回答语言偏好是中文"], source="store")

    s = score_memory(query, "已记住：以后用中文回答。", state)

    assert s["wrong_not_present"] == ["D:/old-project"]
    assert s["wrong_present"] == []
    assert s["wrong_used"] == []
    assert s["wrong_ok"] is True


# ---------------- 4. 缺字段 = 不检查（向后兼容）----------------


def test_missing_memory_fields_are_not_checked():
    """老用例没有记忆字段 → 什么都不检查，且**不算失败**。"""
    query = {"id": "q01", "category": "retrieval_only", "expect_tools": ["search_knowledge"]}
    state = MemoryState(written_contents=["随便什么"], source="trace")

    s = score_memory(query, "随便答一句", state)

    assert s["checked"] is True
    assert s["remember_missed"] == []
    assert s["absent_violations"] == []
    assert s["wrong_used"] == []
    assert s["pass"] is True


def test_no_memory_state_means_nothing_is_checked():
    """没有记忆状态（没有 trace / 没有记忆库快照）→ checked=False，判定结果为 None。

    ★ 这是「不把旧用例判成失败」的关键：老用例没有记忆证据时，
      记忆项既不算通过也不算失败，只是**没查**。
    """
    query = {"expect_memory_content_contains": ["中文"]}

    s = score_memory(query, "答", None)

    assert s["checked"] is False
    assert s["pass"] is None
    assert s["remember_missed"] == []


def test_score_answer_keeps_old_shape_and_adds_memory():
    """老调用方式（两参数）返回的字典形状不变；给了记忆状态才多一个 memory 键。"""
    query = {"answer_must_contain": ["中文"]}

    old = score_answer(query, "用中文回答")
    assert "memory" not in old
    assert old["must_hits"] == ["中文"]

    new = score_answer(query, "用中文回答", MemoryState(written_contents=["偏好中文"]))
    assert new["memory"]["checked"] is True
    assert new["must_hits"] == ["中文"]


def test_memory_ground_truth_accessors():
    q = {
        "expect_memory_content_contains": ["a", "b"],
        "expect_memory_absent": ["c"],
        "expect_memory_wrong_not_used": ["d"],
    }
    assert expected_memory_content_contains(q) == ["a", "b"]
    assert expected_memory_absent(q) == ["c"]
    assert expected_memory_wrong_not_used(q) == ["d"]
    assert has_memory_ground_truth(q) is True
    # 没标注 → 空列表 / False（老条目不受影响）
    assert expected_memory_content_contains({}) == []
    assert has_memory_ground_truth({}) is False
    # 标了空列表也算「标过」（区分「没标注」与「标了但为空」）
    assert has_memory_ground_truth({"expect_memory_absent": []}) is True


# ---------------- 5. 记忆状态从 trace 重算 ----------------


def test_memory_state_from_trace_reads_writes_recalls_and_injection(tmp_path):
    """★ 记忆判定基于 **trace 可见的记忆注入与召回**（不用重跑就能重算）。"""
    trace = _write_trace(tmp_path, [
        {"type": "memory_injected", "count": 1, "keys": ["lang_pref"], "chars": 42, "total_records": 3},
        {"type": "thought", "tool_calls": [
            {"name": "remember_fact", "args": {"content": "以后所有回答都用中文", "key": "lang"}},
        ]},
        {"type": "observation", "tool": "remember_fact", "ok": True,
         "result_preview": json.dumps({"ok": True, "key": "lang", "total": 4}, ensure_ascii=False)},
        {"type": "observation", "tool": "recall_memory", "ok": True,
         "result_preview": json.dumps(
             {"ok": True, "count": 1, "results": [{"key": "old", "content": "回答语言偏好是英文"}]},
             ensure_ascii=False)},
    ])

    state = memory_state_from_trace(trace)

    assert state.source == "trace"
    assert state.written_contents == ["以后所有回答都用中文"]
    assert state.written_keys == ["lang"]
    assert state.attempted_contents == ["以后所有回答都用中文"]
    assert state.recalled_contents == ["回答语言偏好是英文"]
    assert state.injected_keys == ["lang_pref"]
    assert state.injected_count == 1


def test_memory_state_from_trace_ignores_failed_writes(tmp_path):
    """写失败的 remember_fact 不算「写进记忆了」（只有 ok=True 的那次算）。"""
    trace = _write_trace(tmp_path, [
        {"type": "thought", "tool_calls": [
            {"name": "remember_fact", "args": {"content": "临时访问令牌 tmp-8f3a9c2b7d6e"}},
        ]},
        {"type": "observation", "tool": "remember_fact", "ok": False,
         "error_class": "permanent", "result_preview": None},
    ])

    state = memory_state_from_trace(trace)

    assert state.written_contents == []
    assert state.attempted_contents == ["临时访问令牌 tmp-8f3a9c2b7d6e"]


def test_memory_state_from_trace_handles_missing_file(tmp_path):
    state = memory_state_from_trace(tmp_path / "nope.jsonl")
    assert state.written_contents == []
    assert state.recalled_contents == []
    assert state.corpus == []


def test_memory_state_from_store_reads_snapshot(tmp_path):
    """记忆库快照 = 「长期记忆里现在到底有什么」（落盘数据，比 trace 更权威）。"""
    path = _write_store(tmp_path, [
        {"key": "lang", "content": "用户的回答语言偏好是中文", "tags": [], "created_at": "2026-01-01T00:00:00+08:00"},
        {"key": "old", "content": "用户的回答语言偏好是英文", "tags": [], "created_at": "2026-01-01T00:00:00+08:00"},
    ])

    state = memory_state_from_store(path)

    assert len(state.store_contents) == 2
    assert "用户的回答语言偏好是英文" in state.store_contents
    assert state.store_keys == ["lang", "old"]


def test_merge_memory_states(tmp_path):
    trace_state = MemoryState(written_contents=["a"], injected_keys=["k"], injected_count=1, source="trace")
    store_state = MemoryState(store_contents=["b"], source="store")

    merged = merge_memory_states(trace_state, store_state)

    assert merged.written_contents == ["a"]
    assert merged.store_contents == ["b"]
    assert merged.injected_count == 1
    assert merged.source == "trace+store"
    assert merged.corpus == ["a", "b"]


# ---------------- 6. evaluate 集成：分母与比率 ----------------


def test_evaluate_memory_metrics_use_documented_denominators():
    """记忆三项的分母：完成 + 有记忆状态；缺状态的不进分母（单独报 missing）。"""
    records = [
        # 该记：写进去了 → 通过
        _rec({"expect_memory_content_contains": ["中文"]},
             answer="已记住：以后用中文回答。",
             state=MemoryState(written_contents=["以后用中文回答"], source="trace")),
        # 该记：没写 → 不通过
        _rec({"expect_memory_content_contains": ["中文"]},
             answer="好的。",
             state=MemoryState(written_contents=[], source="trace")),
        # 不该记：没记 → 通过
        _rec({"expect_memory_absent": ["tmp-abc"]},
             answer="7×8=56",
             state=MemoryState(written_contents=["中文偏好"], source="trace")),
        # 不该记：记了 → 不通过
        _rec({"expect_memory_absent": ["tmp-abc"]},
             answer="好的。",
             state=MemoryState(written_contents=["tmp-abc"], source="trace")),
        # 召回错：旧内容在记忆里但没被当事实用 → 通过
        _rec({"expect_memory_wrong_not_used": ["D:/old-project"]},
             answer="已记住：以后用 D:/work/new-repo。",
             state=MemoryState(store_contents=["我的项目根目录是 D:/old-project"], source="store")),
        # 召回错：拿旧内容当事实 → 不通过
        _rec({"expect_memory_wrong_not_used": ["D:/old-project"]},
             answer="你的项目根目录是 D:/old-project。",
             state=MemoryState(store_contents=["我的项目根目录是 D:/old-project"], source="store")),
        # 标了记忆标注但**没有记忆状态** → 不进分母，单独报 missing
        _rec({"expect_memory_content_contains": ["中文"]}, answer="好的。", state=None),
        # 失败的运行 → 不进分母（口径与其它指标一致）
        _rec({"expect_memory_content_contains": ["中文"]},
             answer="[模型调用失败]",
             state=MemoryState(written_contents=[], source="trace"), status="error"),
    ]

    s = evaluate(records)

    assert s["memory_labeled_n"] == 8
    assert s["memory_scored_n"] == 6
    assert s["memory_not_scored_n"] == 2      # 1 条没状态 + 1 条失败运行
    # 该记：2 个片段，命中 1 个
    assert s["memory_remember_n"] == 2
    assert s["memory_remember_hits"] == 1
    assert s["memory_remember_rate"] == 0.5
    # 该记：真正写进去的
    assert s["memory_write_rate"] == 0.5
    # 不该记：2 个片段，违规 1 个
    assert s["memory_absent_n"] == 2
    assert s["memory_absent_violation_rate"] == 0.5
    # 召回错：2 个片段确实在记忆里，1 个被当事实用了
    assert s["memory_wrong_present_n"] == 2
    assert s["memory_wrong_used_n"] == 1
    assert s["memory_wrong_not_used_rate"] == 0.5
    # 三项全过：3/6
    assert s["memory_pass_n"] == 6
    assert s["memory_pass_rate"] == 0.5
    assert len(s["memory_failures"]) == 3


def test_evaluate_memory_metrics_are_none_when_no_memory_cases():
    """没有记忆用例 → 记忆指标是「算不出来」（None），不是 0（不假装查过）。"""
    records = [_rec({"category": "retrieval_only", "expect_tools": ["search_knowledge"]},
                    answer="查到了", used=["search_knowledge"])]

    s = evaluate(records)

    assert s["memory_labeled_n"] == 0
    assert s["memory_scored_n"] == 0
    assert s["memory_remember_rate"] is None
    assert s["memory_absent_violation_rate"] is None
    assert s["memory_wrong_not_used_rate"] is None
    assert s["memory_pass_rate"] is None


# ---------------- 7. 评测集文件：旧的 50 条不受影响 + 新的记忆用例齐全 ----------------


def test_eval_set_still_contains_the_original_50_cases_unchanged():
    """★ 旧用例**零回归**：q01~q50 仍在，字段一字未少（只新增了 memory 分类的用例）。"""
    queries = load_eval_set(EVAL_SET_PATH)
    by_id = {q["id"]: q for q in queries}

    for i in range(1, 51):
        qid = f"q{i:02d}"
        assert qid in by_id, f"{qid} 不见了"
        q = by_id[qid]
        # 旧用例的字段一个都不能少
        for key in ("id", "category", "task", "expect_tools", "expected_refusal",
                    "answer_must_contain", "answer_must_not_contain"):
            assert key in q, f"{qid} 缺字段 {key}"
        # 旧用例**不带**记忆标注（新的记忆字段是可选的，缺 = 不检查）
        for key in MEMORY_FIELDS:
            assert key not in q, f"{qid} 不该带记忆字段 {key}"
        assert not has_memory_ground_truth(q)


def test_eval_set_memory_cases_cover_three_kinds():
    """新的 memory 分类：三类各 ≥2 条，字段形状正确。"""
    queries = load_eval_set(EVAL_SET_PATH)
    memory_cases = [q for q in queries if q.get("category") == "memory"]

    assert len(memory_cases) >= 6, f"记忆用例至少 6 条，实际 {len(memory_cases)} 条"
    kinds = {
        "该记": [q for q in memory_cases if q.get("expect_memory_content_contains")],
        "不该记": [q for q in memory_cases if q.get("expect_memory_absent")],
        "召回错": [q for q in memory_cases if q.get("expect_memory_wrong_not_used")],
    }
    for name, items in kinds.items():
        assert len(items) >= 2, f"{name} 类用例至少 2 条，实际 {len(items)} 条"

    for q in memory_cases:
        for key in MEMORY_FIELDS:
            assert key in q, f"{q['id']} 缺 {key}"
            assert isinstance(q[key], list)
            assert all(isinstance(x, str) and x.strip() for x in q[key])
        # 记忆用例也必须能被老指标打分（字段形状与其它用例一致）
        assert isinstance(q["expect_tools"], list)
        assert isinstance(q["expected_refusal"], bool)
        assert 1 <= len(q["answer_must_contain"]) <= 3


def test_eval_set_total_and_ids_unique():
    """评测集总数 = 旧 50 + 新 8 = 58；id 唯一。"""
    queries = load_eval_set(EVAL_SET_PATH)
    assert len(queries) == 58
    ids = [q["id"] for q in queries]
    assert len(set(ids)) == len(ids)


def test_memory_cases_with_wrong_recall_have_a_seed():
    """「召回错」用例必须预置（seed）一条过时记忆，否则陷阱踩不到、分母为 0。"""
    queries = load_eval_set(EVAL_SET_PATH)
    for q in queries:
        if not q.get("expect_memory_wrong_not_used"):
            continue
        seed = q.get("memory_seed") or []
        assert seed, f"{q['id']} 声明了召回错，却没预置记忆"
        for item in seed:
            assert item.get("content")
            assert item.get("key")


# ---------------- 8. 评测 runner：记忆隔离 + 可序列化 ----------------


def _load_eval_script():
    import importlib.util

    path = REPO_ROOT / "scripts" / "eval_agent.py"
    spec = importlib.util.spec_from_file_location("eval_agent_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_runner_isolates_memory_store_for_memory_cases(tmp_path):
    """★ 记忆类用例跑在**临时记忆库**上：绝不往用户真实记忆库写测试数据。"""
    module = _load_eval_script()
    config = {"memory": {"enabled": True, "path": "./data/agent_memory.jsonl", "tools": True}}

    # 有记忆标注 → 隔离 + 预置
    run_config, path = module.prepare_memory_env(
        config, {"id": "q57", "expect_memory_wrong_not_used": ["x"],
                 "memory_seed": [{"key": "k", "content": "c"}]}, tmp_path,
    )
    assert path != "" and Path(path).exists()
    assert run_config["memory"]["path"] == path
    assert run_config["memory"]["path"] != "./data/agent_memory.jsonl"
    # 原配置不被就地改坏
    assert config["memory"]["path"] == "./data/agent_memory.jsonl"
    assert json.loads(Path(path).read_text(encoding="utf-8").strip())["content"] == "c"

    # 没有记忆标注 → 原样返回（老用例行为完全不变）
    same, same_path = module.prepare_memory_env(config, {"id": "q01"}, tmp_path)
    assert same is config
    assert same_path == ""


def test_eval_runner_record_to_dict_is_json_serializable(tmp_path):
    """记忆状态要能落进评测结果 JSON（不能是 dataclass 对象）。"""
    module = _load_eval_script()
    record = _rec({"expect_memory_content_contains": ["中文"]},
                  answer="已记住", state=MemoryState(written_contents=["中文"], source="trace"))

    data = module._record_to_dict(record)
    text = json.dumps(data, ensure_ascii=False)

    assert '"memory_state"' in text
    assert json.loads(text)["memory_state"]["written_contents"] == ["中文"]


# ---------------- 9. 端到端：模拟一次带记忆的运行 ----------------


def test_end_to_end_memory_eval_from_trace_and_store(tmp_path):
    """把「trace 证据 + 记忆库快照」拼起来 → 三类判定一次性跑通。"""
    trace = _write_trace(tmp_path, [
        {"type": "memory_injected", "count": 1, "keys": ["lang_pref"], "chars": 30, "total_records": 1},
        {"type": "thought", "tool_calls": [
            {"name": "remember_fact", "args": {"content": "以后所有回答都用中文", "key": "lang"}},
        ]},
        {"type": "observation", "tool": "remember_fact", "ok": True,
         "result_preview": json.dumps({"ok": True, "key": "lang", "total": 2}, ensure_ascii=False)},
    ])
    store = _write_store(tmp_path, [
        {"key": "lang", "content": "以后所有回答都用中文", "created_at": "2026-01-01T00:00:00+08:00"},
    ])

    state = merge_memory_states(memory_state_from_trace(trace), memory_state_from_store(store))
    query = {
        "id": "q51",
        "expect_memory_content_contains": ["中文"],
        "expect_memory_absent": [],
        "expect_memory_wrong_not_used": [],
    }

    s = score_memory(query, "已记住：以后所有回答都用中文。", state)

    assert s["pass"] is True
    assert s["write_hits"] == ["中文"]
    assert state.injected_keys == ["lang_pref"]


def test_real_run_with_memory_tool_is_scored_as_remembered(monkeypatch, tmp_path):
    """★ 端到端：真跑一次 Agent（假模型 + **真的** remember_fact 工具与 JSONL 记忆库），
    再用评测口径判定「该记的事实确实进了长期记忆」。

    这条测试保证「记忆评测链路」不是只在构造的字典上成立：
    真工具 → 真落盘 → 真 trace → 评测判定，四段都通。
    """
    import asyncio
    from contextlib import asynccontextmanager

    from langchain_core.messages import AIMessage

    from agent.config import load_config
    from agent.memory import MemoryStore, build_memory_tools
    from agent.runner import run_task
    from agent.toolkit import ToolSet
    from tests.fake_llm import ScriptedChatModel, tool_call

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-for-test-not-a-real-key")
    monkeypatch.setenv("MODEL", "scripted-model")

    memory_path = tmp_path / "memory.jsonl"
    config = load_config()
    config["agent"]["max_steps"] = 3
    config["memory"]["path"] = str(memory_path)
    config["memory"]["tools"] = True
    config["trace"]["dir"] = str(tmp_path / "traces")
    config["trace"]["console"] = "quiet"

    @asynccontextmanager
    async def fake_toolset(cfg, tracer, rag_mode_override=None):
        store = MemoryStore(cfg)
        store.run_id = tracer.run_id
        tools = build_memory_tools(store)
        tracer.event("memory_ready", enabled=True, path=str(store.path),
                     records=len(store), tools=list(tools))
        yield ToolSet(tools=tools, rag_mode="none", tool_names=list(tools), memory=store)

    monkeypatch.setattr("agent.runner.build_toolset", fake_toolset)
    scripted = ScriptedChatModel(responses=[
        tool_call("remember_fact", {"content": "以后所有回答都用中文", "key": "lang"}, "call_1"),
        AIMessage(content="已记住：以后所有回答都用中文。"),
    ])
    monkeypatch.setattr("agent.runner.build_llm", lambda settings: scripted)

    result = asyncio.run(run_task("记住：以后所有回答都用中文。", config=config))
    assert result.status == "completed"

    # 真的落盘了
    assert memory_path.exists()
    assert "以后所有回答都用中文" in memory_path.read_text(encoding="utf-8")

    # trace 里有记忆钩子与写入事件
    events = [json.loads(line) for line in
              Path(result.trace_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = {e["type"] for e in events}
    assert "memory_injected" in kinds
    assert any(e["type"] == "thought" and e.get("tool_calls") for e in events)

    # 评测口径：记忆里确实出现了该内容
    state = merge_memory_states(
        memory_state_from_trace(result.trace_path),
        memory_state_from_store(memory_path),
    )
    s = score_memory(
        {"id": "q51", "expect_memory_content_contains": ["中文"]},
        result.final_answer,
        state,
    )
    assert s["remember_hits"] == ["中文"]
    assert s["write_hits"] == ["中文"]
    assert s["pass"] is True



if __name__ == "__main__":
    pytest.main([__file__, "-q"])
