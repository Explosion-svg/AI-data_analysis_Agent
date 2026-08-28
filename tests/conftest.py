"""
tests/conftest.py

全局 pytest fixture 配置文件。

职责：
- 在每个测试前重置项目里的模块级单例，避免状态污染
- 在每个测试后清理 Chroma 客户端等持有文件锁的资源

为什么重要：
- 本项目大量使用全局单例，例如 cache、memory、router、tool registry、Chroma、warehouse
- 如果不在测试之间清理，测试可能第一次通过、第二次失败
- Chroma 客户端持有持久化目录的文件锁，Windows 下不关闭会锁住 tmp 目录
"""

from __future__ import annotations

from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def reset_singletons() -> Iterator[None]:
    """
    每个测试前重置模块级单例，测试后清理 Chroma 客户端与请求上下文。

    为什么在这里关闭 Chroma / 重置 ContextVar（P4-3）：
    - 原实现只重置了 assembler/cache/memory/router/registry/breaker，
      遗漏了 vector_store._client、warehouse._engine、request_context ContextVar。
    - 不重置 vector_store._client：上一个测试的 Chroma 客户端继续持有
      持久化目录文件锁，Windows 下 tmp 目录无法删除，后续测试报权限错误。
    - 不重置 request_context ContextVar：跨测试泄漏上次请求的租户/用户身份。
    """
    from ai_data_agent import assembler
    from ai_data_agent.context import request_context
    from ai_data_agent.infra import vector_store, warehouse
    from ai_data_agent.memory import cache_memory, conversation_memory, work_memory
    from ai_data_agent.model_gateway import router
    from ai_data_agent.reliability import circuit_breaker, concurrency
    from ai_data_agent.tools import tool_registry

    assembler._container = None
    cache_memory._cache = None
    conversation_memory._memory = None
    work_memory._work_memory = None
    router._router = None
    tool_registry._registry = None
    circuit_breaker._breakers.clear()
    # P4-7：并发限流器单例绑定创建时的事件循环，测试间不重置会导致
    # "Semaphore bound to a different event loop"（pytest-asyncio 每测试新 loop）。
    concurrency.reset_limiter()
    # P4-3：补充此前遗漏的单例重置
    vector_store._client = None
    warehouse._engine = None
    request_context._current_request_context.set(None)

    yield

    # 测试后清理：释放 Chroma 文件锁，清空可能泄漏的请求上下文
    vector_store.close_vector_store()
    request_context._current_request_context.set(None)
