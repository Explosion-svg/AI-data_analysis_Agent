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
from ai_data_agent.reliability.sql_guard import (
    SQLGuardError,
    enforce_allowed_tables,
    extract_referenced_tables,
    validate_sql,
)
from ai_data_agent.reliability.timeout import AsyncTimeoutError, run_with_timeout


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


def test_extract_referenced_tables_comma_join() -> None:
    # P2-13：逗号 join 的两张表都应被提取（旧正则只提取到第一张）。
    assert extract_referenced_tables("SELECT * FROM allowed, secret") == ["allowed", "secret"]


def test_extract_referenced_tables_quoted_identifier() -> None:
    # P2-13：引号标识符应被归一化提取，而不是提取为空。
    assert extract_referenced_tables('SELECT * FROM "secret"') == ["secret"]
    assert extract_referenced_tables("SELECT * FROM `secret`") == ["secret"]


def test_extract_referenced_tables_qualified_and_aliased() -> None:
    # schema 限定名保留；别名不当作表名。
    assert extract_referenced_tables(
        "SELECT * FROM public.users u JOIN public.orders o ON u.id = o.user_id"
    ) == ["public.users", "public.orders"]


def test_enforce_allowed_tables_blocks_comma_join_bypass() -> None:
    # P2-13：白名单开启时，逗号 join 访问白名单外表必须被拦截。
    with pytest.raises(SQLGuardError):
        enforce_allowed_tables("SELECT * FROM allowed, secret", ["allowed"])


def test_enforce_allowed_tables_blocks_quoted_bypass() -> None:
    # P2-13：引号标识符访问白名单外表必须被拦截。
    with pytest.raises(SQLGuardError):
        enforce_allowed_tables('SELECT * FROM "secret"', ["allowed"])


def test_enforce_allowed_tables_allows_whitelisted_comma_join() -> None:
    # P2-13：逗号 join 两张白名单表应放行（不误拦截合法查询）。
    enforce_allowed_tables("SELECT * FROM allowed, allowed2", ["allowed", "allowed2"])


@pytest.mark.asyncio
async def test_run_with_timeout_success() -> None:
    # 正常在超时窗口内完成的协程应返回结果。
    result = await run_with_timeout(asyncio.sleep(0.01, result="ok"), timeout=0.1, name="sleep")
    assert result == "ok"


@pytest.mark.asyncio
async def test_run_with_timeout_raises_timeout_error() -> None:
    # 超时应转换为项目自定义的 AsyncTimeoutError，而不是裸 asyncio 异常。
    with pytest.raises(AsyncTimeoutError):
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


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_allows_single_probe() -> None:
    # P2-11：半开态并发请求只放行一个试探，其余按 CircuitBreakerError 拒绝（防惊群）。
    breaker = CircuitBreaker("svc", failure_threshold=1, recovery_timeout=0.01)

    async def fail() -> None:
        raise RuntimeError("boom")

    async def slow_success() -> str:
        await asyncio.sleep(0.05)
        return "ok"

    # 一次失败即触发熔断
    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    assert breaker.state == CircuitState.OPEN

    # 等待进入 HALF_OPEN
    await asyncio.sleep(0.02)

    # 并发发起 10 个请求：应只有 1 个放行（试探），其余被拒绝
    results = await asyncio.gather(
        *[breaker.call(slow_success) for _ in range(10)],
        return_exceptions=True,
    )
    successes = [r for r in results if r == "ok"]
    rejects = [r for r in results if isinstance(r, CircuitBreakerError)]
    assert len(successes) == 1
    assert len(rejects) == 9
    # 试探成功 → 熔断器完全恢复
    assert breaker.state == CircuitState.CLOSED
