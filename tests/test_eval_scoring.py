"""评测打分器（ground truth 口径）的离线单测。

覆盖任务 A 新增的三件事：
1. **严 / 宽两套工具选择口径**（发起过 vs 真正执行成功）与分母可见；
2. **拒答口径**：`expected_refusal` 只统计**真正完成**且回答来自模型的运行（假绿修复不能退回去）；
3. **答案要点命中率 / 幻觉率**：按 ground truth 算，且每个比率都带分母 n。

另外守住两条「向后兼容」：
- 旧结构的老条目（只有 id/category/task/expect_tools）照样能算，新指标只是「没有」而不是「报错」；
- `expect_tools` 与 `expected_tools` 两个字段名都认。
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.eval_harness import (  # noqa: E402
    EvalRecord,
    answer_must_contain,
    answer_must_not_contain,
    evaluate,
    expected_refusal,
    expected_tools,
    has_tool_ground_truth,
    is_refusal,
    load_eval_set,
    match_keyword,
    render_markdown,
    score_answer,
)

EVAL_SET_PATH = REPO_ROOT / "data" / "eval_queries.json"


def _rec(status="completed", steps=2, used=None, query=None, **kw):
    """构造一条 EvalRecord（默认回答来自模型）。"""
    return EvalRecord(
        query=query or {},
        status=status,
        steps=steps,
        tool_calls=kw.get("tool_calls", 1),
        tool_ok=kw.get("tool_ok", 1),
        tool_failed=kw.get("tool_failed", 0),
        tool_retries=kw.get("tool_retries", 0),
        tool_degraded=kw.get("tool_degraded", 0),
        circuit_opens=kw.get("circuit_opens", 0),
        tool_success_rate=kw.get("tool_success_rate", 1.0),
        used_tools=used if used is not None else [],
        used_tools_ok=kw.get("used_tools_ok", list(used or [])),
        final_answer=kw.get("final_answer", ""),
        final_answer_source=kw.get("final_answer_source", "model"),
    )


# ---------------- ground truth 字段读取（向后兼容）----------------


def test_expected_tools_accepts_both_field_names():
    """`expect_tools`（原始字段）与 `expected_tools`（新别名）都要认。"""
    assert expected_tools({"expect_tools": ["calculator"]}) == ["calculator"]
    assert expected_tools({"expected_tools": ["calculator"]}) == ["calculator"]
    # 两个都在时以 `expect_tools` 为准（原始字段优先，兼容老数据）
    assert expected_tools({"expect_tools": ["a"], "expected_tools": ["b"]}) == ["a"]
    # 没标注 → 空列表
    assert expected_tools({}) == []
    assert expected_tools(None) == []


def test_has_tool_ground_truth_distinguishes_missing_from_empty():
    """「没标注工具期望」与「标注了空列表（不该用工具）」是两回事。"""
    assert has_tool_ground_truth({}) is False
    assert has_tool_ground_truth({"expect_tools": []}) is True
    assert has_tool_ground_truth({"expected_tools": []}) is True
    assert has_tool_ground_truth({"expect_tools": ["x"]}) is True


def test_old_style_entry_without_ground_truth_still_evaluates():
    """★ 向后兼容：老条目只有 id/category/task/expect_tools，不能报错，新指标应为「没有」。"""
    old_query = {"id": "q01", "category": "retrieval_only", "task": "查一下", "expect_tools": ["search_knowledge"]}
    records = [_rec("completed", 2, ["search_knowledge"], query=old_query, final_answer="查到了")]

    s = evaluate(records)

    assert s["n"] == 1
    assert s["tool_selection_accuracy"] == 1.0
    # 没有 ground truth 的那些指标：分母为 0 → 比率是 None（而不是 0，更不是假绿）
    assert s["refusal_accuracy"] is None
    assert s["refusal_accuracy_n"] == 0
    assert s["answer_keyword_recall"] is None
    assert s["answer_keyword_recall_n"] == 0
    assert s["hallucination_rate"] is None
    assert s["hallucination_n"] == 0
    assert s["no_tool_expected_n"] == 0


def test_ground_truth_accessors():
    q = {
        "expected_refusal": True,
        "answer_must_contain": ["a", "b"],
        "answer_must_not_contain": ["c"],
    }
    assert expected_refusal(q) is True
    assert expected_refusal({"expected_refusal": False}) is False
    assert expected_refusal({}) is False
    assert answer_must_contain(q) == ["a", "b"]
    assert answer_must_not_contain(q) == ["c"]
    assert answer_must_contain({}) == []


# ---------------- 工具选择：严 / 宽两套口径 + 分母 ----------------


def test_tool_selection_strict_and_wide_with_denominators():
    """严口径 = 成功执行过；宽口径 = 发起过；两个分母都要可见。"""
    records = [
        # 期望检索：发起过但全失败 → 严口径不算，宽口径算
        _rec("completed", 2, ["search_knowledge"],
             query={"expect_tools": ["search_knowledge"]}, used_tools_ok=[]),
        # 期望检索且成功
        _rec("completed", 2, ["search_knowledge"],
             query={"expect_tools": ["search_knowledge"]}, used_tools_ok=["search_knowledge"]),
        # 没标注工具期望 → 不进分母
        _rec("completed", 1, [], query={"category": "time_only"}),
    ]

    s = evaluate(records)

    assert s["tool_selection_accuracy"] == 0.5
    assert s["tool_selection_attempt_rate"] == 1.0
    assert s["tool_selection_n"] == 2
    assert s["tool_selection_accuracy_n"] == 2
    assert s["tool_selection_attempt_rate_n"] == 2


def test_no_tool_expected_violation_rate():
    """显式标注「不该用工具」的题：模型用了工具 = 违规（分母只算这类题）。"""
    records = [
        # 标注了空列表 + 模型真的调了工具 → 违规
        _rec("completed", 2, ["search_knowledge"], query={"expect_tools": []}),
        # 标注了空列表 + 没有调工具 → 不违规
        _rec("completed", 1, [], query={"expect_tools": []}),
        # 没标注 → 不进这个分母
        _rec("completed", 2, ["calculator"], query={"category": "calculator_only"}),
    ]

    s = evaluate(records)

    assert s["no_tool_expected_n"] == 2
    assert s["no_tool_violation_rate"] == 0.5


# ---------------- 拒答口径：只算真正完成的运行 ----------------


def test_refusal_accuracy_only_counts_completed_runs():
    """★ 假绿回归：失败运行的兜底文案里也有「没有」，不能算成「诚实拒答」。"""
    records = [
        # 失败运行（系统兜底文案，含「没有」）→ 不进分母
        _rec("error", 1, ["search_knowledge"],
             query={"category": "out_of_domain", "expected_refusal": True},
             final_answer="[模型调用失败] RuntimeError: boom（这一步没有产生任何工具调用）",
             final_answer_source="system_fallback"),
        # 完成了且如实拒答 → 分子 + 分母
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "out_of_domain", "expected_refusal": True},
             final_answer="知识库里没有关于武汉天气的内容，无法回答。"),
    ]

    s = evaluate(records)

    assert s["refusal_accuracy"] == 1.0
    assert s["refusal_accuracy_n"] == 1      # 分母只看完成的
    assert s["refusal_expected_n"] == 2      # 标注为应拒答的共 2 条
    assert s["refusal_completed_n"] == 1


def test_refusal_accuracy_is_none_when_nothing_completed():
    """全是失败运行时，拒答准确率应当是「无法计算」，不是 0 也不是 1。"""
    records = [
        _rec("error", 1, [], query={"expected_refusal": True}, final_answer="没有"),
    ]

    s = evaluate(records)

    assert s["refusal_accuracy"] is None
    assert s["refusal_accuracy_n"] == 0
    assert s["refusal_expected_n"] == 1


def test_system_fallback_is_not_counted_as_refusal():
    """兜底文案即使被标成 completed，也不能算成「诚实拒答」（与诚实率同一套修法）。

    口径：分母 = 真正完成的运行（1 条），分子 = 回答来自模型**且**带拒答措辞 → 0 条 → 0.0。
    """
    records = [
        _rec("completed", 1, [],
             query={"category": "out_of_domain", "expected_refusal": True},
             final_answer="[模型调用失败] RuntimeError: boom（这一步没有产生任何工具调用）",
             final_answer_source="system_fallback"),
    ]

    s = evaluate(records)

    assert s["refusal_accuracy"] == 0.0
    assert s["refusal_accuracy_n"] == 1


def test_refusal_false_positive_rate():
    """不该拒答的题却拒答了 → 计入 false positive（分母 = 非应拒答且完成的运行数）。"""
    records = [
        # 该答却拒答
        _rec("completed", 2, ["search_knowledge"],
             query={"expect_tools": ["search_knowledge"], "expected_refusal": False},
             final_answer="查不到相关内容，无法回答。"),
        # 该答且答了
        _rec("completed", 2, ["search_knowledge"],
             query={"expect_tools": ["search_knowledge"], "expected_refusal": False},
             final_answer="知识库说 chunk_size 典型值是 256-1024 字符。"),
    ]

    s = evaluate(records)

    assert s["refusal_false_positive_rate"] == 0.5
    assert s["refusal_false_positive_n"] == 2


def test_is_refusal_marker_matching():
    """拒答判定是词表启发式：认「承认做不到」的措辞，不认泛泛的「没有」。"""
    assert is_refusal("知识库里没有关于武汉天气的内容") is True
    assert is_refusal("查不到相关内容") is True
    assert is_refusal("计算失败：除数为 0") is True
    # 普通回答不算拒答
    assert is_refusal("chunk_size 典型值是 256-1024 字符") is False
    assert is_refusal("") is False


# ---------------- 答案要点命中率 / 幻觉率 ----------------


def test_answer_keyword_recall_counts_hits_and_misses():
    records = [
        # 两个要点，命中一个
        _rec("completed", 2, ["search_knowledge"],
             query={"answer_must_contain": ["LoRA", "低秩"]},
             final_answer="LoRA 是一种参数高效微调方法。"),
        # 两个要点全中
        _rec("completed", 2, ["search_knowledge"],
             query={"answer_must_contain": ["LoRA", "低秩"]},
             final_answer="LoRA 用低秩矩阵做微调。"),
        # 没标注要点 → 不进分母
        _rec("completed", 1, [], query={"category": "time_only"}, final_answer="现在 12:00:00"),
    ]

    s = evaluate(records)

    assert s["answer_keyword_recall_n"] == 2
    assert s["answer_keyword_hits"] == 3
    assert s["answer_keyword_total"] == 4
    assert s["answer_keyword_recall"] == 0.75
    assert s["answer_keyword_full_hit_rate"] == 0.5
    assert s["answer_keyword_full_hit_n"] == 2


def test_answer_keyword_recall_ignores_failed_runs():
    """失败运行的空答案不该拉低要点命中率（口径与诚实率一致：只算真正完成的）。"""
    records = [
        _rec("error", 1, [], query={"answer_must_contain": ["LoRA"]}, final_answer=""),
        _rec("completed", 2, ["search_knowledge"],
             query={"answer_must_contain": ["LoRA"]}, final_answer="LoRA 微调"),
    ]

    s = evaluate(records)

    assert s["answer_keyword_recall_n"] == 1
    assert s["answer_keyword_recall"] == 1.0


def test_empty_answer_scores_zero_not_crash():
    """完成但回答为空：要点全未命中，指标是 0 而不是异常。"""
    records = [
        _rec("completed", 2, ["search_knowledge"],
             query={"answer_must_contain": ["LoRA"]}, final_answer=""),
    ]

    s = evaluate(records)

    assert s["answer_keyword_recall"] == 0.0
    assert s["answer_keyword_recall_n"] == 1
    assert s["answer_keyword_full_hit_rate"] == 0.0


def test_hallucination_rate_flags_forbidden_words():
    records = [
        # 编造了「晴转多云」
        _rec("completed", 2, ["search_knowledge"],
             query={"answer_must_not_contain": ["晴转多云", "25度"]},
             final_answer="武汉今天晴转多云，25 度。"),
        # 正常拒答，没命中禁词
        _rec("completed", 2, ["search_knowledge"],
             query={"answer_must_not_contain": ["晴转多云", "25度"]},
             final_answer="知识库里没有关于武汉天气的内容。"),
        # 没标注禁词 → 不进分母
        _rec("completed", 1, [], query={"category": "time_only"}, final_answer="现在 12:00:00"),
    ]

    s = evaluate(records)

    assert s["hallucination_rate"] == 0.5
    assert s["hallucination_n"] == 2
    assert s["hallucination_hits"][0]["id"] is None
    assert s["hallucination_hits"][0]["violated"]


def test_keyword_alternatives_and_case_insensitive():
    """`a|b` 里任一命中即算这条要点命中；英文关键词忽略大小写。"""
    assert match_keyword("查不到相关内容", "没有关于|查不到|未找到") is True
    assert match_keyword("知识库里没有相关内容", "没有关于|没有相关内容|查不到") is True
    # 写法对不上就不算命中（ground truth 写的是「可接受的写法集合」）
    assert match_keyword("知识库里没有相关内容", "没有关于|查不到") is False
    assert match_keyword("随便一句", "没有关于|查不到") is False
    assert match_keyword("CHUNK_SIZE 典型 512", "chunk_size") is True


def test_score_answer_is_pure_function():
    query = {"answer_must_contain": ["a|b"], "answer_must_not_contain": ["x"]}
    s = score_answer(query, "a 出现了")
    assert s["must_hits"] == ["a|b"]
    assert s["must_missed"] == []
    assert s["full_hit"] is True
    assert s["hallucinated"] is False
    assert s["recall"] == 1.0

    s2 = score_answer(query, "x 出现了")
    assert s2["hallucinated"] is True
    assert s2["must_hits"] == []


# ---------------- 评测集文件本身的数据完整性 ----------------


def test_eval_set_file_is_valid_and_covers_ground_truth():
    """守住评测集规模与标注质量（防止以后有人把 ground truth 删掉）。"""
    queries = load_eval_set(EVAL_SET_PATH)

    assert len(queries) >= 40, f"评测集应至少 40 条，实际 {len(queries)} 条"

    ids = [q["id"] for q in queries]
    assert len(set(ids)) == len(ids), "id 必须唯一"

    categories = {q["category"] for q in queries}
    assert {"out_of_domain", "retrieval_only", "multi_tool"} <= categories
    # 新增的两个分类
    assert "clarification" in categories
    assert "tool_failure_recovery" in categories

    for q in queries:
        assert "task" in q and q["task"].strip(), f"{q.get('id')} 缺 task"
        assert "expect_tools" in q, f"{q.get('id')} 缺 expect_tools"
        assert isinstance(q["expect_tools"], list)
        assert 1 <= len(q["answer_must_contain"]) <= 3, f"{q.get('id')} 要点应 1~3 个"
        assert isinstance(q["answer_must_not_contain"], list)
        assert isinstance(q["expected_refusal"], bool)
        if q["category"] == "out_of_domain":
            assert q["expected_refusal"] is True, f"{q.get('id')} 领域外问题必须标 expected_refusal=true"


def test_eval_set_file_is_valid_json():
    raw = EVAL_SET_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert isinstance(data, list)
    assert len(data) >= 40


# ---------------- 报告渲染 ----------------


def test_render_markdown_shows_new_metrics_with_denominators():
    records = [
        _rec("completed", 2, ["search_knowledge"],
             query={"category": "out_of_domain", "expected_refusal": True,
                    "answer_must_contain": ["没有关于"],
                    "answer_must_not_contain": ["晴转多云"]},
             final_answer="知识库里没有关于武汉天气的内容。"),
    ]

    md = render_markdown(evaluate(records), records)

    assert "拒答准确率" in md
    assert "答案要点命中率" in md
    assert "幻觉率" in md
    assert "不该用工具" in md
    assert "n=1" in md


# ---------------- scripts/eval_agent.py 的筛选参数 ----------------


def _load_eval_script():
    path = REPO_ROOT / "scripts" / "eval_agent.py"
    spec = importlib.util.spec_from_file_location("eval_agent_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_queries_by_ids_categories_limit():
    module = _load_eval_script()
    queries = load_eval_set(EVAL_SET_PATH)

    picked = module._filter_queries(queries, ids="q01,q02")
    assert [q["id"] for q in picked] == ["q01", "q02"]

    picked = module._filter_queries(queries, categories="out_of_domain")
    assert picked and all(q["category"] == "out_of_domain" for q in picked)

    picked = module._filter_queries(queries, limit=3)
    assert len(picked) == 3

    # 默认不筛 → 全部
    assert len(module._filter_queries(queries)) == len(queries)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
