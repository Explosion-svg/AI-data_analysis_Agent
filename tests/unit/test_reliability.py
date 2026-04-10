"""
tests/unit/test_reliability.py

可靠性模块单元测试。

主要验证：
- SQL 安全防护
- 超时控制
- 自动重试
- 熔断器状态切换
"""

from __future__ import annotations

import asyncio

import pytest

from ai_data_agent.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitState,
)
from ai_data_agent.reliability.retry import async_retry
from ai_data_agent.reliability.sql_guard import SQLGuardError, validate_sql
from ai_data_agent.reliability.timeout import TimeoutError, run_with_timeout


def test_validate_sql_allows_select() -> None:
    # 普通只读查询应通过安全校验。
    assert validate_sql("SELECT * FROM sales") == "SELECT * FROM sales"


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "SELECT * FROM users; DELETE FROM sales",
        "SELECT * FROM users WHERE 1=1",
    ],
)
def test_validate_sql_blocks_dangerous_sql(sql: str) -> None:
    # 危险 SQL、拼接多语句、典型注入模式都必须被拦截。
    with pytest.raises(SQLGuardError):
        validate_sql(sql)


@pytest.mark.asyncio
async def test_run_with_timeout_success() -> None:
    # 正常在超时窗口内完成的协程应返回结果。
    result = await run_with_timeout(asyncio.sleep(0.01, result="ok"), timeout=0.1, name="sleep")
    assert result == "ok"


@pytest.mark.asyncio
async def test_run_with_timeout_raises_timeout_error() -> None:
    # 超时应转换为项目自定义的 TimeoutError，而不是裸 asyncio 异常。
    with pytest.raises(TimeoutError):
        await run_with_timeout(asyncio.sleep(0.05), timeout=0.01, name="slow")


@pytest.mark.asyncio
async def test_async_retry_eventually_succeeds() -> None:
    # 前两次失败、第三次成功时，应能自动重试并最终返回成功。
    state = {"count": 0}

    @async_retry(max_attempts=3, base_delay=0.001, max_delay=0.001, jitter=False)
    async def flaky() -> str:
        state["count"] += 1
        if state["count"] < 3:
            raise ValueError("fail")
        return "ok"

    assert await flaky() == "ok"
    assert state["count"] == 3


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_recovers() -> None:
    # 连续失败达到阈值后应打开，恢复窗口过去后允许再次尝试并恢复。
    breaker = CircuitBreaker("svc", failure_threshold=2, recovery_timeout=0.01)

    async def fail() -> None:
        raise RuntimeError("boom")

    async def succeed() -> str:
        return "ok"

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)

    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitBreakerError):
        await breaker.call(succeed)

    await asyncio.sleep(0.02)
    assert await breaker.call(succeed) == "ok"
    assert breaker.state == CircuitState.CLOSED
