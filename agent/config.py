"""配置模块：读 .env（密钥）+ config.yaml（行为参数），并提供密钥掩码工具。

安全红线：
- 密钥只从环境变量/.env 读，**不写进代码、不写进日志、不写进 trace**；
- 任何要落盘/打印的字符串都先过 `mask_secret()`；
- `describe_llm()` 只输出 base_url 与 model 名，密钥一律显示为 sk-****。
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

DEFAULTS: Dict[str, Any] = {
    "llm": {
        "temperature": 0.0,
        "timeout_s": 60,
        "max_retries": 2,
    },
    "agent": {
        "max_steps": 6,
        "recursion_limit": 25,
        # 同一轮模型返回多个 tool_calls 时是否并发执行（Semaphore 限流）
        "parallel_tools": True,
        "max_parallel_tools": 4,
        "system_prompt": (
            "你是一个严谨的助手，可以调用工具。规则：\n"
            "1) 需要事实依据时先检索知识库（search_knowledge），基于返回片段回答，不要凭记忆编造；\n"
            "2) 需要算术时调用 calculator，不要心算；需要当前时间调用 current_time；\n"
            "3) 如果工具返回错误或空结果，先判断能否换参数/换工具重试一次，仍不行就如实说明「查不到」，不要编答案；\n"
            "4) 回答用中文，简洁，必要时说明依据来自哪个来源文件。"
        ),
    },
    "tools": {
        "timeout_s": 20,
        "retry": {
            "max_attempts": 3,
            "backoff_base_s": 0.5,
            "backoff_max_s": 4.0,
            "retryable_patterns": [
                "timeout",
                "timed out",
                "connection",
                "connect",
                "temporarily",
                "rate limit",
                "429",
                "502",
                "503",
                "504",
            ],
        },
        "circuit_breaker": {
            "failure_threshold": 3,
        },
        "rag": {
            "mode": "auto",  # auto（先 MCP，失败则直连）| mcp | direct
            "mcp": {
                "server_command": "../rag-mcp-server/.venv/Scripts/python.exe",
                "server_script": "../rag-mcp-server/server.py",
                "cwd": "../rag-mcp-server",
                "connect_timeout_s": 60,
            },
            "direct": {
                "repo_path": "../rag-mcp-server",
                "config_path": "config.yaml",
            },
        },
    },
    # 跨任务长期记忆（JSONL 落盘；任务启动时按关键词召回并注入 system prompt）
    "memory": {
        "enabled": True,
        "path": "./data/agent_memory.jsonl",
        "tools": True,            # 是否注册 remember_fact / recall_memory 工具
        "inject_top_k": 3,        # 任务启动时自动注入的相关记忆条数
        "max_inject_chars": 800,  # 注入文本长度上限（控制上下文成本）
        "max_content_chars": 2000,
        "max_records": 2000,      # 超过后自动 compaction（同 key 只留最新）
    },
    "trace": {
        "dir": "./traces",
        "console": "summary",  # summary | verbose | quiet
        "preview_chars": 200,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_dotenv_if_present(path: Path | None = None) -> None:
    """加载 .env（不覆盖已存在的环境变量）。没装 python-dotenv 时静默跳过。"""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:  # 不因为缺一个可选依赖就跑不起来
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """加载 config.yaml 并合并默认值；同时把 .env 读进环境变量。"""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    load_dotenv_if_present(config_path.parent / ".env")

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    config = _deep_merge(DEFAULTS, user_config)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = config_path.parent

    # trace 目录按配置文件所在目录解析（与工作目录无关）
    trace_dir = Path(str(config["trace"]["dir"]))
    if not trace_dir.is_absolute():
        trace_dir = config_path.parent / trace_dir
    config["trace"]["dir"] = str(trace_dir.resolve())

    # 长期记忆文件同样按配置文件所在目录解析
    memory_path = Path(str(config["memory"]["path"]))
    if not memory_path.is_absolute():
        memory_path = config_path.parent / memory_path
    config["memory"]["path"] = str(memory_path.resolve())

    # RAG 相关相对路径同样解析成绝对路径
    rag_cfg = config["tools"]["rag"]
    for key in ("server_command", "server_script", "cwd"):
        raw = Path(str(rag_cfg["mcp"][key]))
        # server_command 可能是 "python"（PATH 上的可执行文件）→ 只在像路径时才解析
        looks_like_path = any(sep in str(raw) for sep in ("/", "\\")) or str(raw).startswith(".")
        if looks_like_path and not raw.is_absolute():
            rag_cfg["mcp"][key] = str((config_path.parent / raw).resolve())
    direct_repo = Path(str(rag_cfg["direct"]["repo_path"]))
    if not direct_repo.is_absolute():
        rag_cfg["direct"]["repo_path"] = str((config_path.parent / direct_repo).resolve())

    return config


def get_llm_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """从环境变量读 LLM 端点配置（key 只留在内存里，不落盘）。"""
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("MODEL", "").strip()
    missing = [name for name, value in [("OPENAI_BASE_URL", base_url), ("OPENAI_API_KEY", api_key), ("MODEL", model)] if not value]
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "missing": missing,
        "temperature": float(config["llm"]["temperature"]),
        "timeout_s": float(config["llm"]["timeout_s"]),
        "max_retries": int(config["llm"]["max_retries"]),
    }


def mask_secret(text: Any, secret: str | None = None) -> str:
    """把字符串里的密钥替换成 sk-****。

    两道防线：
    1) 显式传入的密钥（调用方知道自己在处理什么）；
    2) 环境变量 OPENAI_API_KEY —— 兜底：任何拼进日志的字符串里出现密钥都会被抹掉。
    """
    result = str(text)
    secrets = {secret} if secret else set()
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        secrets.add(env_key)
    for item in secrets:
        if item and item in result:
            result = result.replace(item, "sk-****")
    return result


def describe_llm(settings: Dict[str, Any]) -> str:
    """可安全打印的端点描述（不含密钥）。"""
    return f"model={settings['model']} base_url={settings['base_url']} api_key=sk-****"
