"""
tests/helpers.py

测试辅助对象集合。

职责：
- 提供假的 breaker、cache、memory、router
- 用固定行为替代真实依赖，保证测试稳定、可重复

使用场景：
- AgentLoop 测试
- API 测试
- 其他需要 mock 外部依赖的测试
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_data_agent.context.prompt_builder import PromptBuilder
from ai_data_agent.context.query_rewriter import QueryRewriter
from ai_data_agent.context.schema_context import SchemaContextBuilder
from ai_data_agent.memory.cache_memory import CacheMemory
from ai_data_agent.memory.work_memory import WorkMemory
from ai_data_agent.model_gateway.base_model import LLMResponse, Message
from ai_data_agent.orchestration.agent_loop import AgentLoop
from ai_data_agent.tools.tool_registry import ToolRegistry


@dataclass
class DummyBreaker:
    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)


class DummyMemory:
    def __init__(self, history: list[Message] | None = None) -> None:
        self.history = history or []
        self.added: list[tuple[str, str, str, dict[str, Any] | None]] = []

    def get_messages(self, conversation_id: str) -> list[Message]:
        return list(self.history)

    async def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.added.append((conversation_id, role, content, metadata))

    def clear(self, conversation_id: str) -> None:
        self.history.clear()


class DummyCache:
    def __init__(self, value: Any = None) -> None:
        self.value = value
        self.stored: dict[str, Any] = {}

    def make_key(self, *parts: Any) -> str:
        return "|".join(map(str, parts))

    def get(self, key: str) -> Any:
        return self.value

    def set(self, key: str, value: Any) -> None:
        self.stored[key] = value


class SequenceRouter:
    def __init__(self, responses: list[LLMResponse], embeddings: list[list[float]] | None = None) -> None:
        self._responses = responses
        self._embeddings = embeddings or [[0.1, 0.2, 0.3]]
        self.generate_calls: list[dict[str, Any]] = []

    async def generate(self, messages: list[Message], task_type: Any = None, **kwargs: Any) -> LLMResponse:
        self.generate_calls.append({"messages": messages, "task_type": task_type, "kwargs": kwargs})
        if not self._responses:
            raise AssertionError("No more fake LLM responses configured.")
        return self._responses.pop(0)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embeddings[0] for _ in texts]


def build_test_agent(
    *,
    router: Any,
    memory: Any | None = None,
    cache: Any | None = None,
    registry: ToolRegistry | None = None,
    breaker: Any | None = None,
) -> AgentLoop:
    return AgentLoop(
        prompt_builder=PromptBuilder(),
        query_rewriter=QueryRewriter(),
        schema_builder=SchemaContextBuilder(),
        memory=memory or DummyMemory(),
        cache=cache or CacheMemory(),
        work_memory=WorkMemory(),
        registry=registry or ToolRegistry(),
        router=router,
        breaker=breaker or DummyBreaker(),
    )
