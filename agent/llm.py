"""LLM 客户端：任何 OpenAI 兼容端点都能用（DeepSeek / 百炼 / vLLM / Ollama / OneAPI…）。

密钥来源只有一个：环境变量 `OPENAI_API_KEY`（.env 里填）。代码里没有任何默认 key，
也不会把 key 打印出来（见 `config.describe_llm`）。
"""

from typing import Any, Dict

from langchain_openai import ChatOpenAI


def build_llm(settings: Dict[str, Any]) -> ChatOpenAI:
    """按配置构建 ChatModel。

    - `temperature=0`：Agent 场景要可复现，尽量别加随机性；
    - `max_retries`：模型 HTTP 层重试（网络抖动、限流），与工具层的重试是两套独立机制；
    - `timeout`：单次模型调用超时，防止请求挂死。
    """
    return ChatOpenAI(
        model=settings["model"],
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        temperature=settings["temperature"],
        timeout=settings["timeout_s"],
        max_retries=settings["max_retries"],
    )
