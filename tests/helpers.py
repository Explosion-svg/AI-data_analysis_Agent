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

from ai_data_agent.model_gateway.base_model import LLMResponse, Message


@dataclass
class DummyBreaker:
    async def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)


class DummyMemory:
    def __init__(self, history: list[Message] | None = None) -> None:
        self.history = history or []
        self.added: list[tuple[str, str, str]] = []

    def get_messages(self, conversation_id: str) -> list[Message]:
        return list(self.history)

    def add(self, conversation_id: str, role: str, content: str) -> None:
        self.added.append((conversation_id, role, content))


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
