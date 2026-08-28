"""
reliability/retry.py — 异步重试装饰器（指数退避 + 随机抖动）

职责：
  为异步函数提供自动重试能力，使用指数退避算法控制重试间隔，
  通过随机抖动防止"惊群效应"（Thundering Herd Problem）。

指数退避算法（Exponential Backoff）：
  delay = base_delay × 2^(attempt - 1)
  例如（base_delay=1.0, max_attempts=4）：
  - 第 1 次失败后等待：1.0 × 2^0 = 1.0 秒
  - 第 2 次失败后等待：1.0 × 2^1 = 2.0 秒
  - 第 3 次失败后等待：1.0 × 2^2 = 4.0 秒（但受 max_delay 限制）
  - 第 4 次失败：超过最大次数，直接重新抛出异常

随机抖动（Jitter）：
  每次延迟乘以 0.5~1.0 之间的随机系数（等分抖动策略）：
    delay *= 0.5 + random.random() * 0.5
  效果：第 2 次等待可能是 1.0~2.0 秒，而不是固定 2.0 秒
  原因：防止多个客户端在同一时刻同时重试（如果 100 个客户端都在 t=2.0 秒重试，
        服务器会同时收到 100 个请求的重试洪峰）

与 CircuitBreaker 的配合：
  - retry 处理短暂故障（网络抖动、临时 429）
  - circuit_breaker 处理持续故障（服务宕机）
  - 建议：先 retry，retry 全部失败后 circuit_breaker 才记为一次"失败"
  - 实现：retry 的失败才会传播到 circuit_breaker 的 _on_failure

exceptions 参数：
  只对指定类型的异常重试，其他类型直接抛出。
  典型配置：exceptions=(RateLimitError, APITimeoutError) 只重试可恢复的 API 错误，
  不重试 APIError（如 400 Bad Request，重试也没有意义）。
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
    异步重试装饰器工厂（Decorator Factory）。

    使用方式（作为装饰器）：
        @async_retry(max_attempts=3, exceptions=(APIError, TimeoutError))
        async def call_llm(): ...

    使用方式（手动包裹）：
        retry_fn = async_retry(max_attempts=3)
        await retry_fn(some_async_fn)(arg1, arg2)

    @async_retry() 是最常用的方式（不带参数，全用默认值）：
        @async_retry()
        async def _generate_with_retry(...): ...

    参数设计（全部为 None 时使用 settings 中的全局配置）：
    - max_attempts：最大尝试次数（包含第一次，即最多重试 max_attempts-1 次）
    - base_delay：基础延迟（秒），指数退避的起点
    - max_delay：最大延迟上限（秒），防止退避时间无限增长
    - exceptions：只对这些类型的异常重试（其他异常直接传播）
    - jitter：是否添加随机抖动（强烈建议 True，防止惊群）

    Args:
        max_attempts: 最大尝试次数（None = settings.retry_max_attempts，默认 3）
        base_delay: 基础延迟秒数（None = settings.retry_base_delay，默认 1.0）
        max_delay: 最大延迟秒数（None = settings.retry_max_delay，默认 30.0）
        exceptions: 触发重试的异常类型序列（默认所有 Exception）
        jitter: 是否添加随机抖动（默认 True）

    Returns:
        装饰器函数（接受 async 函数并返回包装后的 async 函数）
    """
    _max = max_attempts or settings.retry_max_attempts
    _base = base_delay or settings.retry_base_delay
    _max_delay = max_delay or settings.retry_max_delay

    def decorator(fn: Callable) -> Callable:
        """
        将 async 函数包装为带重试能力的 async 函数。

        使用 @functools.wraps(fn) 保留原函数的 __name__、__doc__、__module__ 等属性，
        确保装饰后的函数在日志、调试、文档中仍然显示原始函数名。

        Args:
            fn: 要包装的 async 函数

        Returns:
            带重试能力的 async 函数包装器
        """
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            实际的重试执行逻辑。

            循环逻辑：
            1. attempt 从 1 开始，到 _max 结束
            2. 尝试调用 fn，成功则直接返回
            3. 如果失败且 attempt == _max：记录 EXHAUSTED 日志，重新抛出
            4. 如果失败且 attempt < _max：计算延迟，等待后进行下一次尝试

            延迟计算：
              delay = min(base × 2^(attempt-1), max_delay)
              if jitter: delay *= random.uniform(0.5, 1.0)

            注意：exceptions 参数使用 tuple() 转换，
            因为 Python 的 except 语句只接受 tuple，不接受 Sequence。
            `except tuple(exceptions) as exc` 等价于 `except (TypeA, TypeB, ...) as exc`。
            """
            last_exc: BaseException | None = None
            for attempt in range(1, _max + 1):
                try:
                    return await fn(*args, **kwargs)
                except tuple(exceptions) as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt == _max:
                        # 最后一次尝试也失败了，记录并重新抛出
                        logger.error(
                            "retry.exhausted",
                            fn=fn.__qualname__,
                            attempts=attempt,
                            error=str(exc),
                        )
                        raise

                    # 计算指数退避延迟（带上限）
                    delay = min(_base * (2 ** (attempt - 1)), _max_delay)
                    if jitter:
                        # 等分抖动：delay × [0.5, 1.0)，减少惊群效应
                        delay *= 0.5 + random.random() * 0.5

                    logger.warning(
                        "retry.attempt",
                        fn=fn.__qualname__,
                        attempt=attempt,
                        next_in=round(delay, 2),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc] # 理论上不会走到这里

        return wrapper

    return decorator
