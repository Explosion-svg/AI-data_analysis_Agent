"""
reliability/retry.py — 指数退避重试
支持同步/异步函数，可指定可重试的异常类型
"""
from __future__ import annotations

import asyncio
import functools
import random
from typing import Any, Callable, Sequence, Type

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


def async_retry(
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    exceptions: Sequence[Type[BaseException]] = (Exception,),
    jitter: bool = True,
) -> Callable:
    """
    异步重试装饰器（指数退避 + 随机抖动）。

    用法::
        @async_retry(max_attempts=3, exceptions=(APIError,))
        async def call_llm(): ...
    """
    _max = max_attempts or settings.retry_max_attempts
    _base = base_delay or settings.retry_base_delay
    _max_delay = max_delay or settings.retry_max_delay

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, _max + 1):
                try:
                    return await fn(*args, **kwargs)
                except tuple(exceptions) as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt == _max:
                        logger.error(
                            "retry.exhausted",
                            fn=fn.__qualname__,
                            attempts=attempt,
                            error=str(exc),
                        )
                        raise
                    delay = min(_base * (2 ** (attempt - 1)), _max_delay)
                    if jitter:
                        delay *= 0.5 + random.random() * 0.5
                    logger.warning(
                        "retry.attempt",
                        fn=fn.__qualname__,
                        attempt=attempt,
                        next_in=round(delay, 2),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
