"""hello-agent：最小可讲的 ReAct / 工具调用 Agent。

模块划分（每一层都能单独测、单独换）：
    config.py      —— 配置与密钥（.env + config.yaml），含密钥掩码
    trace.py       —— 结构化 trace（JSONL 文件 + 控制台摘要）
    resilience.py  —— 工具调用的重试 / 降级 / 熔断 / 超时（可靠性核心）
    tools_builtin.py —— 计算器（AST 白名单求值）与当前时间
    rag_tool.py    —— 经 MCP 客户端接入 rag-mcp-server 的 search_knowledge（含直连降级形态）
    toolkit.py     —— 组装工具集（异步上下文管理器，负责 MCP 会话生命周期）
    graph.py       —— LangGraph 编排：agent 节点 + tools 节点 + 最大步数路由
    runner.py      —— 单条任务的完整执行入口
"""

__version__ = "0.1.0"
