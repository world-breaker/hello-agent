# 上下文压缩代价对照实验（压缩开 / 关，同一批任务）

- 运行时间：2026-10-21 01:29:19
- 代码基线：`34edfe8`（仓库 hello-agent）
- 被测模型：**ContextFaithfulFakeModel（仓库自带 ScriptedChatModel 的上下文忠实扩展；脚本化假模型）**
- 任务批次：评测集里期望用到检索工具（search_knowledge）的题目（34 条）
- RAG 形态：direct（进程内直连同源 KnowledgeService）

> ⚠️ **本轮用假模型，只测机械效果（压缩 / 步数 / token）与「上下文信息损失对答案要点的代理影响」，未测语义质量。**
> 原因：本机没有可用的 LLM 端点（无 `.env`、无本地推理服务），真实模型的语义质量无从测起；编数字不如不测。

## 1. 取舍曲线（每个档位都跑同一批任务）

| 档位 | keep_last | 字符预算 | 完成任务 | 成功率 | 平均步数 | 压缩字符节省率 | token 省下（对不压缩） | 平均 token | 答案要点命中率 | 要点全中率 | 工具参数被压坏次数 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `off` | 999 | 1000000000 | 34/34 | 1.0 | 2.68 | None | None | 3993.8 | 0.7581 | 0.5882 | 0 |
| `default` | 6 | 12000 | 34/34 | 1.0 | 2.68 | None | 0.0 | 3993.8 | 0.7581 | 0.5882 | 0 |
| `keep2_4000` | 2 | 4000 | 34/34 | 1.0 | 2.68 | None | 0.0 | 3993.8 | 0.7581 | 0.5882 | 0 |
| `keep2_2000` | 2 | 2000 | 34/34 | 1.0 | 2.68 | None | 0.0 | 3993.8 | 0.7581 | 0.5882 | 0 |
| `keep2_1200` | 2 | 1200 | 34/34 | 1.0 | 2.68 | 0.7023 | 0.2927 | 2824.9 | 0.6129 | 0.4118 | 5 |
| `keep1_800` | 1 | 800 | 34/34 | 1.0 | 2.68 | 0.2846 | 0.2808 | 2872.2 | 0.6129 | 0.4118 | 5 |

口径说明：

- **字符节省率** = 1 - Σ压缩后字符 / Σ压缩前字符（取自 trace 的 `context_compressed` 事件）。⚠️ 各档位的**触发点不同**（触发得越早、事件越多、单次 before 越小），所以这一列**不适合跨档位直接比大小**；跨档位请看下一列。
- **token 省下（对不压缩）** = 1 - 本档位总 token / 不压缩档总 token（假模型上报的是「实际送进模型的字符数」，1 字符 ≈ 1 token）——**这一列才是可跨档位比较的「省了多少」**。
- **答案要点命中率** = 命中要点数 / 要点总数（ground truth 来自 `data/eval_queries.json`）。
- **工具参数被压坏次数** = 与对照组（`off`）逐条比对，**参数不一样**的工具调用次数；「不一样」= 同任务同策略下唯一的变量是上下文被压缩。
- **token** 是假模型的**字符代理**（1 字符 ≈ 1 token），不是真实 tokenizer 计数。

## 2. 这条曲线说明什么（按上面的数字自动生成）

1. **省多少**：把上下文压到最紧的档位（`keep2_1200`，预算 1200），实际送进模型的上下文少了 **0.2927**（平均 token 3993.8 → 2824.9）。
2. **代价多少**：同一批任务，答案要点命中率从 **0.7581** 掉到 **0.6129**（要点全中率 0.5882 → 0.4118）；工具调用参数被压坏 **5** 次（涉及 5 条任务）。
3. **没坏的**：成功率（1.0 → 1.0）与平均步数（2.68 → 2.68）**完全没变**——这批任务里压缩不影响「跑不跑得完」，影响的是「答得全不全」。
4. **本仓库默认档（12000 字符 / 保留最近 6 条）在这批任务上根本没触发压缩**（token 省下 0.0）——也就是说：默认配置下的「压缩」目前只是一道**没被用上的保险**。
5. **口径提醒**：以上是**假模型**的结果——它测的是「上下文里的信息被压掉多少、丢掉的信息会不会出现在答案/参数里」，**不等于**真实模型在压缩下的语义质量下降幅度（真实模型可能能从残缺上下文里推断出被压掉的要点，也可能更糟）。

## 3. 逐任务原始数据

| 档位 | 任务 | 分类 | 状态 | 步数 | 工具调用 | 压缩次数 | 压缩前字符 | 压缩后字符 | token | 要点命中 | 要点总数 | 参数差异 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `off` | q01 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6582 | 3 | 3 | 0 |
| `off` | q02 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4684 | 2 | 2 | 0 |
| `off` | q03 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2922 | 1 | 1 | 0 |
| `off` | q04 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2908 | 2 | 2 | 0 |
| `off` | q07 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6548 | 1 | 3 | 0 |
| `off` | q08 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4618 | 2 | 2 | 0 |
| `off` | q09 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2912 | 0 | 1 | 0 |
| `off` | q10 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2902 | 2 | 2 | 0 |
| `off` | q11 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6622 | 1 | 2 | 0 |
| `off` | q12 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4537 | 2 | 2 | 0 |
| `off` | q15 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2904 | 0 | 1 | 0 |
| `off` | q16 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2898 | 2 | 2 | 0 |
| `off` | q17 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `off` | q18 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2944 | 2 | 2 | 0 |
| `off` | q19 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `off` | q20 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `off` | q21 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2942 | 2 | 2 | 0 |
| `off` | q22 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `off` | q23 | retrieval_only | completed | 3 | 2 | 0 | 0 | 0 | 4483 | 2 | 2 | 0 |
| `off` | q24 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2916 | 2 | 2 | 0 |
| `off` | q25 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4680 | 2 | 2 | 0 |
| `off` | q26 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4582 | 1 | 2 | 0 |
| `off` | q27 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4558 | 1 | 2 | 0 |
| `off` | q28 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4507 | 2 | 2 | 0 |
| `off` | q29 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4507 | 1 | 2 | 0 |
| `off` | q35 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6524 | 1 | 2 | 0 |
| `off` | q36 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6466 | 1 | 2 | 0 |
| `off` | q37 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6652 | 1 | 2 | 0 |
| `off` | q44 | tool_failure_recovery | completed | 3 | 2 | 0 | 0 | 0 | 4524 | 2 | 2 | 0 |
| `off` | q46 | out_of_domain | completed | 3 | 2 | 0 | 0 | 0 | 2584 | 0 | 1 | 0 |
| `off` | q47 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2882 | 0 | 1 | 0 |
| `off` | q48 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2876 | 1 | 1 | 0 |
| `off` | q49 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 1494 | 0 | 1 | 0 |
| `off` | q50 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2900 | 0 | 1 | 0 |
| `default` | q01 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6582 | 3 | 3 | 0 |
| `default` | q02 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4684 | 2 | 2 | 0 |
| `default` | q03 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2922 | 1 | 1 | 0 |
| `default` | q04 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2908 | 2 | 2 | 0 |
| `default` | q07 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6548 | 1 | 3 | 0 |
| `default` | q08 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4618 | 2 | 2 | 0 |
| `default` | q09 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2912 | 0 | 1 | 0 |
| `default` | q10 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2902 | 2 | 2 | 0 |
| `default` | q11 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6622 | 1 | 2 | 0 |
| `default` | q12 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4537 | 2 | 2 | 0 |
| `default` | q15 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2904 | 0 | 1 | 0 |
| `default` | q16 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2898 | 2 | 2 | 0 |
| `default` | q17 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `default` | q18 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2944 | 2 | 2 | 0 |
| `default` | q19 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `default` | q20 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `default` | q21 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2942 | 2 | 2 | 0 |
| `default` | q22 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `default` | q23 | retrieval_only | completed | 3 | 2 | 0 | 0 | 0 | 4483 | 2 | 2 | 0 |
| `default` | q24 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2916 | 2 | 2 | 0 |
| `default` | q25 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4680 | 2 | 2 | 0 |
| `default` | q26 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4582 | 1 | 2 | 0 |
| `default` | q27 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4558 | 1 | 2 | 0 |
| `default` | q28 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4507 | 2 | 2 | 0 |
| `default` | q29 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4507 | 1 | 2 | 0 |
| `default` | q35 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6524 | 1 | 2 | 0 |
| `default` | q36 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6466 | 1 | 2 | 0 |
| `default` | q37 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6652 | 1 | 2 | 0 |
| `default` | q44 | tool_failure_recovery | completed | 3 | 2 | 0 | 0 | 0 | 4524 | 2 | 2 | 0 |
| `default` | q46 | out_of_domain | completed | 3 | 2 | 0 | 0 | 0 | 2584 | 0 | 1 | 0 |
| `default` | q47 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2882 | 0 | 1 | 0 |
| `default` | q48 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2876 | 1 | 1 | 0 |
| `default` | q49 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 1494 | 0 | 1 | 0 |
| `default` | q50 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2900 | 0 | 1 | 0 |
| `keep2_4000` | q01 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6582 | 3 | 3 | 0 |
| `keep2_4000` | q02 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4684 | 2 | 2 | 0 |
| `keep2_4000` | q03 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2922 | 1 | 1 | 0 |
| `keep2_4000` | q04 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2908 | 2 | 2 | 0 |
| `keep2_4000` | q07 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6548 | 1 | 3 | 0 |
| `keep2_4000` | q08 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4618 | 2 | 2 | 0 |
| `keep2_4000` | q09 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2912 | 0 | 1 | 0 |
| `keep2_4000` | q10 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2902 | 2 | 2 | 0 |
| `keep2_4000` | q11 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6622 | 1 | 2 | 0 |
| `keep2_4000` | q12 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4537 | 2 | 2 | 0 |
| `keep2_4000` | q15 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2904 | 0 | 1 | 0 |
| `keep2_4000` | q16 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2898 | 2 | 2 | 0 |
| `keep2_4000` | q17 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `keep2_4000` | q18 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2944 | 2 | 2 | 0 |
| `keep2_4000` | q19 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `keep2_4000` | q20 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `keep2_4000` | q21 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2942 | 2 | 2 | 0 |
| `keep2_4000` | q22 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `keep2_4000` | q23 | retrieval_only | completed | 3 | 2 | 0 | 0 | 0 | 4483 | 2 | 2 | 0 |
| `keep2_4000` | q24 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2916 | 2 | 2 | 0 |
| `keep2_4000` | q25 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4680 | 2 | 2 | 0 |
| `keep2_4000` | q26 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4582 | 1 | 2 | 0 |
| `keep2_4000` | q27 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4558 | 1 | 2 | 0 |
| `keep2_4000` | q28 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4507 | 2 | 2 | 0 |
| `keep2_4000` | q29 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4507 | 1 | 2 | 0 |
| `keep2_4000` | q35 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6524 | 1 | 2 | 0 |
| `keep2_4000` | q36 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6466 | 1 | 2 | 0 |
| `keep2_4000` | q37 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6652 | 1 | 2 | 0 |
| `keep2_4000` | q44 | tool_failure_recovery | completed | 3 | 2 | 0 | 0 | 0 | 4524 | 2 | 2 | 0 |
| `keep2_4000` | q46 | out_of_domain | completed | 3 | 2 | 0 | 0 | 0 | 2584 | 0 | 1 | 0 |
| `keep2_4000` | q47 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2882 | 0 | 1 | 0 |
| `keep2_4000` | q48 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2876 | 1 | 1 | 0 |
| `keep2_4000` | q49 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 1494 | 0 | 1 | 0 |
| `keep2_4000` | q50 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2900 | 0 | 1 | 0 |
| `keep2_2000` | q01 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6582 | 3 | 3 | 0 |
| `keep2_2000` | q02 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4684 | 2 | 2 | 0 |
| `keep2_2000` | q03 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2922 | 1 | 1 | 0 |
| `keep2_2000` | q04 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2908 | 2 | 2 | 0 |
| `keep2_2000` | q07 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6548 | 1 | 3 | 0 |
| `keep2_2000` | q08 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4618 | 2 | 2 | 0 |
| `keep2_2000` | q09 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2912 | 0 | 1 | 0 |
| `keep2_2000` | q10 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2902 | 2 | 2 | 0 |
| `keep2_2000` | q11 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6622 | 1 | 2 | 0 |
| `keep2_2000` | q12 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4537 | 2 | 2 | 0 |
| `keep2_2000` | q15 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2904 | 0 | 1 | 0 |
| `keep2_2000` | q16 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2898 | 2 | 2 | 0 |
| `keep2_2000` | q17 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `keep2_2000` | q18 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2944 | 2 | 2 | 0 |
| `keep2_2000` | q19 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `keep2_2000` | q20 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `keep2_2000` | q21 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2942 | 2 | 2 | 0 |
| `keep2_2000` | q22 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `keep2_2000` | q23 | retrieval_only | completed | 3 | 2 | 0 | 0 | 0 | 4483 | 2 | 2 | 0 |
| `keep2_2000` | q24 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2916 | 2 | 2 | 0 |
| `keep2_2000` | q25 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4680 | 2 | 2 | 0 |
| `keep2_2000` | q26 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4582 | 1 | 2 | 0 |
| `keep2_2000` | q27 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4558 | 1 | 2 | 0 |
| `keep2_2000` | q28 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4507 | 2 | 2 | 0 |
| `keep2_2000` | q29 | retrieval_arithmetic | completed | 3 | 2 | 0 | 0 | 0 | 4507 | 1 | 2 | 0 |
| `keep2_2000` | q35 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6524 | 1 | 2 | 0 |
| `keep2_2000` | q36 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6466 | 1 | 2 | 0 |
| `keep2_2000` | q37 | multi_tool | completed | 4 | 3 | 0 | 0 | 0 | 6652 | 1 | 2 | 0 |
| `keep2_2000` | q44 | tool_failure_recovery | completed | 3 | 2 | 0 | 0 | 0 | 4524 | 2 | 2 | 0 |
| `keep2_2000` | q46 | out_of_domain | completed | 3 | 2 | 0 | 0 | 0 | 2584 | 0 | 1 | 0 |
| `keep2_2000` | q47 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2882 | 0 | 1 | 0 |
| `keep2_2000` | q48 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2876 | 1 | 1 | 0 |
| `keep2_2000` | q49 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 1494 | 0 | 1 | 0 |
| `keep2_2000` | q50 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2900 | 0 | 1 | 0 |
| `keep2_1200` | q01 | multi_tool | completed | 4 | 3 | 2 | 3095 | 1026 | 3443 | 2 | 3 | 1 |
| `keep2_1200` | q02 | retrieval_arithmetic | completed | 3 | 2 | 1 | 1450 | 409 | 2594 | 1 | 2 | 0 |
| `keep2_1200` | q03 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2922 | 1 | 1 | 0 |
| `keep2_1200` | q04 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2908 | 2 | 2 | 0 |
| `keep2_1200` | q07 | multi_tool | completed | 4 | 3 | 2 | 3068 | 999 | 3403 | 1 | 3 | 1 |
| `keep2_1200` | q08 | retrieval_arithmetic | completed | 3 | 2 | 1 | 1416 | 375 | 2528 | 1 | 2 | 0 |
| `keep2_1200` | q09 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2912 | 0 | 1 | 0 |
| `keep2_1200` | q10 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2902 | 2 | 2 | 0 |
| `keep2_1200` | q11 | multi_tool | completed | 4 | 3 | 2 | 3122 | 1053 | 3489 | 0 | 2 | 0 |
| `keep2_1200` | q12 | retrieval_arithmetic | completed | 3 | 2 | 1 | 1397 | 356 | 2447 | 1 | 2 | 0 |
| `keep2_1200` | q15 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2904 | 0 | 1 | 0 |
| `keep2_1200` | q16 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2898 | 2 | 2 | 0 |
| `keep2_1200` | q17 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `keep2_1200` | q18 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2944 | 2 | 2 | 0 |
| `keep2_1200` | q19 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `keep2_1200` | q20 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2936 | 2 | 2 | 0 |
| `keep2_1200` | q21 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2942 | 2 | 2 | 0 |
| `keep2_1200` | q22 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2930 | 2 | 2 | 0 |
| `keep2_1200` | q23 | retrieval_only | completed | 3 | 2 | 1 | 1366 | 325 | 2393 | 1 | 2 | 0 |
| `keep2_1200` | q24 | retrieval_only | completed | 2 | 1 | 0 | 0 | 0 | 2916 | 2 | 2 | 0 |
| `keep2_1200` | q25 | retrieval_arithmetic | completed | 3 | 2 | 1 | 1457 | 416 | 2590 | 2 | 2 | 0 |
| `keep2_1200` | q26 | retrieval_arithmetic | completed | 3 | 2 | 1 | 1420 | 379 | 2492 | 0 | 2 | 0 |
| `keep2_1200` | q27 | retrieval_arithmetic | completed | 3 | 2 | 1 | 1401 | 360 | 2468 | 0 | 2 | 0 |
| `keep2_1200` | q28 | retrieval_arithmetic | completed | 3 | 2 | 1 | 1384 | 343 | 2417 | 1 | 2 | 0 |
| `keep2_1200` | q29 | retrieval_arithmetic | completed | 3 | 2 | 1 | 1391 | 350 | 2417 | 1 | 2 | 0 |
| `keep2_1200` | q35 | multi_tool | completed | 4 | 3 | 2 | 3030 | 961 | 3387 | 1 | 2 | 1 |
| `keep2_1200` | q36 | multi_tool | completed | 4 | 3 | 2 | 2997 | 928 | 3319 | 1 | 2 | 1 |
| `keep2_1200` | q37 | multi_tool | completed | 4 | 3 | 2 | 3110 | 1041 | 3511 | 1 | 2 | 1 |
| `keep2_1200` | q44 | tool_failure_recovery | completed | 3 | 2 | 1 | 1397 | 356 | 2434 | 2 | 2 | 0 |
| `keep2_1200` | q46 | out_of_domain | completed | 3 | 2 | 0 | 0 | 0 | 2584 | 0 | 1 | 0 |
| `keep2_1200` | q47 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2882 | 0 | 1 | 0 |
| `keep2_1200` | q48 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2876 | 1 | 1 | 0 |
| `keep2_1200` | q49 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 1494 | 0 | 1 | 0 |
| `keep2_1200` | q50 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 2900 | 0 | 1 | 0 |
| `keep1_800` | q01 | multi_tool | completed | 4 | 3 | 3 | 4443 | 2460 | 3529 | 2 | 3 | 1 |
| `keep1_800` | q02 | retrieval_arithmetic | completed | 3 | 2 | 2 | 2820 | 1837 | 2652 | 1 | 2 | 0 |
| `keep1_800` | q03 | out_of_domain | completed | 2 | 1 | 1 | 1299 | 1331 | 2954 | 1 | 1 | 0 |
| `keep1_800` | q04 | retrieval_only | completed | 2 | 1 | 1 | 1299 | 1331 | 2940 | 2 | 2 | 0 |
| `keep1_800` | q07 | multi_tool | completed | 4 | 3 | 3 | 4401 | 2418 | 3489 | 1 | 3 | 1 |
| `keep1_800` | q08 | retrieval_arithmetic | completed | 3 | 2 | 2 | 2756 | 1773 | 2586 | 1 | 2 | 0 |
| `keep1_800` | q09 | out_of_domain | completed | 2 | 1 | 1 | 1290 | 1322 | 2944 | 0 | 1 | 0 |
| `keep1_800` | q10 | retrieval_only | completed | 2 | 1 | 1 | 1296 | 1328 | 2934 | 2 | 2 | 0 |
| `keep1_800` | q11 | multi_tool | completed | 4 | 3 | 3 | 4459 | 2476 | 3575 | 0 | 2 | 0 |
| `keep1_800` | q12 | retrieval_arithmetic | completed | 3 | 2 | 2 | 2713 | 1730 | 2505 | 1 | 2 | 0 |
| `keep1_800` | q15 | out_of_domain | completed | 2 | 1 | 1 | 1289 | 1321 | 2936 | 0 | 1 | 0 |
| `keep1_800` | q16 | retrieval_only | completed | 2 | 1 | 1 | 1291 | 1323 | 2930 | 2 | 2 | 0 |
| `keep1_800` | q17 | retrieval_only | completed | 2 | 1 | 1 | 1325 | 1357 | 2962 | 2 | 2 | 0 |
| `keep1_800` | q18 | retrieval_only | completed | 2 | 1 | 1 | 1310 | 1342 | 2976 | 2 | 2 | 0 |
| `keep1_800` | q19 | retrieval_only | completed | 2 | 1 | 1 | 1326 | 1358 | 2968 | 2 | 2 | 0 |
| `keep1_800` | q20 | retrieval_only | completed | 2 | 1 | 1 | 1306 | 1338 | 2968 | 2 | 2 | 0 |
| `keep1_800` | q21 | retrieval_only | completed | 2 | 1 | 1 | 1317 | 1349 | 2974 | 2 | 2 | 0 |
| `keep1_800` | q22 | retrieval_only | completed | 2 | 1 | 1 | 1301 | 1333 | 2962 | 2 | 2 | 0 |
| `keep1_800` | q23 | retrieval_only | completed | 3 | 2 | 2 | 2657 | 1674 | 2451 | 1 | 2 | 0 |
| `keep1_800` | q24 | retrieval_only | completed | 2 | 1 | 1 | 1299 | 1331 | 2948 | 2 | 2 | 0 |
| `keep1_800` | q25 | retrieval_arithmetic | completed | 3 | 2 | 2 | 2827 | 1844 | 2648 | 2 | 2 | 0 |
| `keep1_800` | q26 | retrieval_arithmetic | completed | 3 | 2 | 2 | 2760 | 1777 | 2550 | 0 | 2 | 0 |
| `keep1_800` | q27 | retrieval_arithmetic | completed | 3 | 2 | 2 | 2727 | 1744 | 2526 | 0 | 2 | 0 |
| `keep1_800` | q28 | retrieval_arithmetic | completed | 3 | 2 | 2 | 2693 | 1710 | 2475 | 1 | 2 | 0 |
| `keep1_800` | q29 | retrieval_arithmetic | completed | 3 | 2 | 2 | 2707 | 1724 | 2475 | 1 | 2 | 0 |
| `keep1_800` | q35 | multi_tool | completed | 4 | 3 | 3 | 4344 | 2361 | 3473 | 1 | 2 | 1 |
| `keep1_800` | q36 | multi_tool | completed | 4 | 3 | 3 | 4296 | 2313 | 3405 | 1 | 2 | 1 |
| `keep1_800` | q37 | multi_tool | completed | 4 | 3 | 3 | 4464 | 2481 | 3597 | 1 | 2 | 1 |
| `keep1_800` | q44 | tool_failure_recovery | completed | 3 | 2 | 2 | 2698 | 1715 | 2492 | 2 | 2 | 0 |
| `keep1_800` | q46 | out_of_domain | completed | 3 | 2 | 0 | 0 | 0 | 2584 | 0 | 1 | 0 |
| `keep1_800` | q47 | out_of_domain | completed | 2 | 1 | 1 | 1276 | 1308 | 2914 | 0 | 1 | 0 |
| `keep1_800` | q48 | out_of_domain | completed | 2 | 1 | 1 | 1270 | 1302 | 2908 | 1 | 1 | 0 |
| `keep1_800` | q49 | out_of_domain | completed | 2 | 1 | 0 | 0 | 0 | 1494 | 0 | 1 | 0 |
| `keep1_800` | q50 | out_of_domain | completed | 2 | 1 | 1 | 1294 | 1326 | 2932 | 0 | 1 | 0 |
