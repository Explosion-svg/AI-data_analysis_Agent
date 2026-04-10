"""
tests/unit/test_router.py

模型路由单元测试。

主要验证：
- 不同任务类型下的模型选择
- 主模型失败后的 fallback 行为
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from ai_data_agent.model_gateway.base_model import BaseLLM, LLMConfig, LLMResponse, Message
from ai_data_agent.model_gateway.router import ModelRouter, TaskType


class FakeLLM(BaseLLM):
    # 最小可用 LLM 假实现，用来验证路由与 fallback 逻辑。
    def __init__(self, name: str, should_fail: bool = False) -> None:
        self._name = name
        self._should_fail = should_fail
        self.calls: list[LLMConfig] = []

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, messages: list[Message], config: LLMConfig) -> LLMResponse:
        self.calls.append(config)
        if self._should_fail:
            raise RuntimeError(f"{self._name} failed")
        return LLMResponse(content=f"from-{self._name}", model=config.model)

    async def stream(self, messages: list[Message], config: LLMConfig) -> AsyncIterator[str]:
        yield "x"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    async def health_check(self) -> bool:
        return True


def build_router(registry: dict[str, BaseLLM]) -> ModelRouter:
    # 绕过 __post_init__，直接注入假的 registry，避免真实模型配置依赖。
    router = object.__new__(ModelRouter)
    router._registry = registry
    return router


@pytest.mark.asyncio
async def test_router_selects_openai_for_simple() -> None:
    # SIMPLE 任务优先走 openai。
    router = build_router({"openai": FakeLLM("openai"), "deepseek": FakeLLM("deepseek")})

    resp = await router.generate([Message(role="user", content="hi")], task_type=TaskType.SIMPLE)

    assert resp.content == "from-openai"


@pytest.mark.asyncio
async def test_router_falls_back_when_primary_fails() -> None:
    # 主模型失败时应自动 fallback 到次级模型。
    router = build_router({"openai": FakeLLM("openai", should_fail=True), "deepseek": FakeLLM("deepseek")})

    resp = await router.generate([Message(role="user", content="hi")], task_type=TaskType.COMPLEX)

    assert resp.content == "from-deepseek"
