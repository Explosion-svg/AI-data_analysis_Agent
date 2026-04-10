"""
tests/conftest.py

全局 pytest fixture 配置文件。

职责：
- 在每个测试前重置项目里的模块级单例，避免状态污染
- 提供测试运行所需的基础事件循环

为什么重要：
- 本项目大量使用全局单例，例如 cache、memory、router、tool registry
- 如果不在测试之间清理，测试可能第一次通过、第二次失败
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    from ai_data_agent import assembler
    from ai_data_agent.memory import cache_memory, conversation_memory
    from ai_data_agent.model_gateway import router
    from ai_data_agent.reliability import circuit_breaker
    from ai_data_agent.tools import tool_registry

    assembler._container = None
    cache_memory._cache = None
    conversation_memory._memory = None
    router._router = None
    tool_registry._registry = None
    circuit_breaker._breakers.clear()


@pytest.fixture
def event_loop() -> Any:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
