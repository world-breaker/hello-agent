"""脚本化假模型：单测里替代真实 LLM（不花钱、不联网、结果确定）。

用法：把「模型应该依次返回什么」写成列表，脚本按顺序吐消息，用完后重复最后一条。
"""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """构造一条「只发起工具调用」的模型消息。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


class ScriptedChatModel(BaseChatModel):
    """按脚本返回 AIMessage 的假模型（同步/异步都实现，供 LangGraph 调用）。

    ★ 必须实现 `bind_tools`：真实图构建时会对模型调用它（把工具 schema 绑定给模型，
    否则模型不知道工具存在——见 README 踩坑记录）。假模型的脚本是写死的，不需要
    真的绑定，但 `BaseChatModel.bind_tools` 的默认实现是**直接抛 NotImplementedError**，
    不覆盖的话所有走「图构建」的用例都会红。这里只记录工具名，返回自身。
    """

    responses: List[AIMessage] = []
    cursor: int = 0
    bound_tools: List[str] = []
    calls: int = 0   # 模型被调用的次数（断点续跑测试用它证明「前面几步没有被重放」）

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        self.bound_tools = [getattr(tool, "name", str(tool)) for tool in tools]
        return self

    def _next_message(self) -> AIMessage:
        index = min(self.cursor, len(self.responses) - 1)
        self.cursor += 1
        self.calls += 1
        return self.responses[index]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])
