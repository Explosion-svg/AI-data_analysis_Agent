"""
reliability/timeout.py — 异步超时控制
防止慢查询/工具调用卡死整个 Agent
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable

from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


class TimeoutError(asyncio.TimeoutError):
    """超时异常（包含上下文信息）。"""
    def __init__(self, name: str, timeout: float) -> None:
        super().__init__(f"'{name}' timed out after {timeout}s")
        self.name = name
        self.timeout = timeout


async def run_with_timeout(
    coro,
    timeout: float,
    name: str = "operation",
) -> Any:
    """
    在 timeout 秒内执行 coro，超时抛出 TimeoutError。
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("timeout.exceeded", name=name, timeout=timeout)
        raise TimeoutError(name, timeout)


def with_timeout(timeout: float, name: str | None = None) -> Callable:
    """
    装饰器：为异步函数添加超时控制。

    用法::
        @with_timeout(30.0, "sql_query")
        async def execute_sql(sql): ...
    """
    def decorator(fn: Callable) -> Callable:
        _name = name or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await run_with_timeout(
                fn(*args, **kwargs),
                timeout=timeout,
                name=_name,
            )

        return wrapper

    return decorator
