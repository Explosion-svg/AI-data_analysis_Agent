"""
reliability/circuit_breaker.py — 熔断器
状态机：CLOSED → OPEN → HALF_OPEN → CLOSED
防止雪崩效应
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum, auto
from typing import Any, Callable

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)


class CircuitState(Enum):
    CLOSED = auto()      # 正常
    OPEN = auto()        # 熔断，拒绝请求
    HALF_OPEN = auto()   # 尝试恢复


class CircuitBreakerError(RuntimeError):
    """熔断器开启，请求被拒绝。"""


class CircuitBreaker:
    """
    单个服务的熔断器。
    线程安全（asyncio 单线程）。
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self._recovery_timeout = recovery_timeout or settings.circuit_breaker_recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def _check_state(self) -> None:
        """检查是否可以从 OPEN 转换到 HALF_OPEN。"""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                async with self._lock:
                    if self._state == CircuitState.OPEN:
                        self._state = CircuitState.HALF_OPEN
                        logger.info(
                            "circuit_breaker.half_open",
                            name=self.name,
                        )
                        metrics.circuit_breaker_open.labels(service=self.name).set(0)

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """通过熔断器执行一次调用。"""
        await self._check_state()

        if self._state == CircuitState.OPEN:
            logger.warning("circuit_breaker.rejected", name=self.name)
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service unavailable. "
                f"Recovery in {self._recovery_timeout}s."
            )

        try:
            result = await fn(*args, **kwargs)
            await self._on_success()
            return result
        except CircuitBreakerError:
            raise
        except Exception as exc:
            await self._on_failure()
            raise exc

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("circuit_breaker.recovered", name=self.name)
                metrics.circuit_breaker_open.labels(service=self.name).set(0)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            logger.warning(
                "circuit_breaker.failure",
                name=self.name,
                count=self._failure_count,
                threshold=self._failure_threshold,
            )
            if self._failure_count >= self._failure_threshold:
                if self._state != CircuitState.OPEN:
                    self._state = CircuitState.OPEN
                    logger.error(
                        "circuit_breaker.opened",
                        name=self.name,
                        failures=self._failure_count,
                    )
                    metrics.circuit_breaker_open.labels(service=self.name).set(1)

    def reset(self) -> None:
        """手动重置熔断器（运维操作）。"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        metrics.circuit_breaker_open.labels(service=self.name).set(0)
        logger.info("circuit_breaker.reset", name=self.name)


# ── 全局熔断器注册表 ─────────────────────────────────────────────────────────

_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name)
    return _breakers[name]
