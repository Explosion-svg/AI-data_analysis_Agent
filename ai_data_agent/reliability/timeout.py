"""
reliability/timeout.py — 异步超时控制

职责：
  为任意 async 操作提供超时保护，防止慢查询或外部服务无响应导致
  协程永久阻塞，进而拖垮整个 Agent 的响应能力。

核心机制（asyncio.wait_for）：
  asyncio.wait_for(coro, timeout=N) 在 N 秒内未完成时：
  1. 取消正在运行的协程（发送 CancelledError）
  2. 抛出 asyncio.TimeoutError

  本模块将 asyncio.TimeoutError 包装为自定义 TimeoutError，
  附加 name 和 timeout 属性，方便日志和上层异常处理识别。

使用方式：
  方式一：直接调用（适合动态超时场景）：
      result = await run_with_timeout(
          execute_query(sql), timeout=30.0, name="sql_query"
      )

  方式二：装饰器（适合固定超时场景）：
      @with_timeout(30.0, "sql_query")
      async def execute_query(sql: str) -> pd.DataFrame:
          ...

与 concurrency limiter 的配合：
  并发限制 → 超时控制 → circuit breaker → retry
  1. concurrency: 限制同时进入的请求数量（入口）
  2. timeout: 限制单次请求的最长等待时间（单请求保护）
  3. circuit_breaker: 持续失败后停止调用（出口）
  4. retry: 临时失败后重试（恢复策略）

注意事项：
  - asyncio.wait_for 的取消是协作式的（cooperative cancellation）
    - 如果被取消的协程内部有 try/except 捕获了 CancelledError 而不重新抛出，
      取消可能不会立即生效
  - 数据库查询超时后，数据库侧的查询可能仍在运行（游标/连接需要手动关闭）
  - SQL 工具的超时建议在 SQLTool 层通过 run_with_timeout 包裹整个执行过程
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable

from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


class TimeoutError(asyncio.TimeoutError):
    """
    超时异常，携带操作名称和超时时长信息。

    继承 asyncio.TimeoutError 的原因：
    - 保持与标准库的兼容性（调用者可以用 asyncio.TimeoutError 捕获）
    - 同时携带额外上下文信息（name、timeout），方便日志和监控

    与内置 TimeoutError 的区别：
    - Python 内置 TimeoutError 是 OSError 的子类（面向系统调用）
    - asyncio.TimeoutError 是 Exception 的子类（面向 asyncio 操作）
    - 本类继承 asyncio.TimeoutError，明确表示是异步等待超时

    属性：
        name: 超时的操作名称（如 "sql_tool"、"python_tool"）
        timeout: 配置的超时时长（秒）
    """
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
    在 timeout 秒内执行协程，超时则抛出 TimeoutError。

    包装 asyncio.wait_for 的原因：
    1. 统一日志：超时时自动记录 WARNING 日志（含操作名和超时时长）
    2. 异常封装：将 asyncio.TimeoutError 包装为更有信息量的 TimeoutError
    3. 接口简化：比直接调用 asyncio.wait_for 更易读

    Args:
        coro: 要执行的协程对象（注意：不是函数，是 fn(*args) 的结果）
        timeout: 最大等待秒数
        name: 操作名称（用于日志，建议与工具名一致，如 "sql_tool"）

    Returns:
        coro 的执行结果

    Raises:
        TimeoutError: 协程在 timeout 秒内未完成
        其他异常: coro 内部抛出的任意异常（透传，不拦截）
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("timeout.exceeded", name=name, timeout=timeout)
        raise TimeoutError(name, timeout)


def with_timeout(timeout: float, name: str | None = None) -> Callable:
    """
    装饰器工厂：为异步函数添加超时控制。

    与 run_with_timeout 的区别：
    - run_with_timeout：调用时传入已创建的协程对象（适合动态超时值）
    - with_timeout：在定义时就绑定超时值（适合静态超时场景，代码更简洁）

    使用方式::
        @with_timeout(30.0, "sql_query")
        async def execute_sql(sql: str) -> pd.DataFrame:
            ...
        # 等价于：
        async def execute_sql(sql: str) -> pd.DataFrame:
            return await run_with_timeout(
                _execute_sql_impl(sql), timeout=30.0, name="sql_query"
            )

    name 参数默认值：
    - None 时使用 fn.__qualname__（包含类名的全限定函数名）
    - 如 "SQLTool._run"，比 "execute_sql" 更具定位性

    @functools.wraps(fn)：
    - 保留原函数的 __name__、__doc__、__module__ 等属性
    - 使装饰后的函数在调试、日志中仍然显示原始函数名

    Args:
        timeout: 最大等待秒数（对所有调用生效）
        name: span/日志中的操作名称（None = fn.__qualname__）

    Returns:
        装饰器函数（接受 async 函数并返回带超时包装的 async 函数）
    """
    def decorator(fn: Callable) -> Callable:
        # 确定操作名称：显式指定 > 函数全限定名
        _name = name or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await run_with_timeout(
                fn(*args, **kwargs),  # 创建协程对象（此时不执行）
                timeout=timeout,
                name=_name,
            )

        return wrapper

    return decorator
