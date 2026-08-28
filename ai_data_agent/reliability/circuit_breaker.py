"""
reliability/circuit_breaker.py — 熔断器（Circuit Breaker）

职责：
  防止雪崩效应（Cascade Failure）：当下游服务（如 LLM API）连续失败时，
  暂时停止向其发送请求，让服务有时间恢复。

熔断器状态机（三态）：
  CLOSED → 正常工作，请求正常通过，记录失败次数
    ↓ 失败次数 >= failure_threshold
  OPEN → 熔断，拒绝所有请求，抛出 CircuitBreakerError
    ↓ 经过 recovery_timeout 秒后
  HALF_OPEN → 试探恢复，允许一个请求通过
    ↓ 成功           ↓ 失败
  CLOSED           OPEN

状态转换详细逻辑：
  CLOSED → OPEN：连续失败次数达到 failure_threshold
  OPEN → HALF_OPEN：上次失败时间距今超过 recovery_timeout 秒
  HALF_OPEN → CLOSED（恢复）：试探请求成功
  HALF_OPEN → OPEN（再次熔断）：试探请求失败

设计细节：
  - _lock（asyncio.Lock）保护状态转换，防止并发竞争条件
  - 成功时 CLOSED 状态下会递减失败计数（最低为 0），实现"渐进恢复"
  - HALF_OPEN 状态下只允许一个试探请求（asyncio.Lock 保证并发安全）
  - CircuitBreakerError 不被 _on_failure 统计（避免因熔断本身触发更多计数）

监控集成：
  - circuit_breaker_open Gauge：1 表示熔断器开启，0 表示正常
  - 通过 Prometheus Gauge 让 Grafana 展示熔断器状态历史

对比 retry 的关系：
  - retry：短暂故障（网络抖动）→ 立即重试几次
  - circuit_breaker：持续故障（服务宕机）→ 暂停请求，等待恢复
  - 两者配合使用：retry 负责短暂异常，circuit_breaker 负责持续异常
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
    """
    熔断器状态枚举。

    使用 auto() 而不是手动指定整数值，原因：
    - 状态值本身没有语义（不需要用数字表示大小关系）
    - auto() 自动生成唯一值，避免手动维护时出错
    - 日志和监控通过 state.name 展示（"CLOSED"/"OPEN"/"HALF_OPEN"），
      不依赖具体数值
    """
    CLOSED = auto()      # 正常工作：请求正常通过
    OPEN = auto()        # 熔断中：拒绝所有请求
    HALF_OPEN = auto()   # 试探恢复：允许一个试探请求


class CircuitBreakerError(RuntimeError):
    """
    熔断器开启时拒绝请求的异常。

    继承 RuntimeError 而不是 Exception 的原因：
    - AgentLoop 的异常处理会捕获并处理这个异常
    - 作为 RuntimeError 子类，更明确地表示"运行时的系统错误"
      而不是业务逻辑错误

    在 call() 中，这个异常不会触发 _on_failure（不被计入失败次数），
    原因是熔断器本身拒绝请求不应该被视为下游服务的新失败。
    """


class CircuitBreaker:
    """
    单个服务的熔断器实现（asyncio 原生支持，线程安全）。

    设计原则：
    - 每个服务（如 "llm"）有独立的熔断器实例
    - 熔断器实例通过全局注册表（_breakers）共享，保证同一服务的所有调用
      使用同一个熔断器状态
    - asyncio.Lock 保护所有状态修改，防止并发修改导致的竞态条件

    与超时的配合：
    - circuit_breaker 不包含超时功能
    - 超时由 reliability/timeout.py 的 run_with_timeout/with_timeout 处理
    - 建议在 circuit_breaker.call() 内部的 fn 里使用超时控制
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
    ) -> None:
        """
        初始化熔断器。

        Args:
            name: 服务名称（用于日志、指标标签和全局注册表键名）
            failure_threshold: 触发熔断的连续失败次数阈值
                              None 时使用 settings.circuit_breaker_failure_threshold（默认 5）
            recovery_timeout: 熔断后等待多少秒再尝试恢复（OPEN → HALF_OPEN）
                            None 时使用 settings.circuit_breaker_recovery_timeout（默认 60.0）
        """
        self.name = name
        self._failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self._recovery_timeout = recovery_timeout or settings.circuit_breaker_recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        # asyncio.Lock 确保并发请求不会同时修改状态（防止 HALF_OPEN 状态下多个请求同时试探）
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """
        当前熔断器状态（只读属性）。

        Returns:
            当前 CircuitState 枚举值
        """
        return self._state

    @property
    def is_open(self) -> bool:
        """
        判断熔断器是否处于开启（拒绝请求）状态。

        便捷属性，等价于 state == CircuitState.OPEN。
        在调用前快速检查时使用：if breaker.is_open: ...

        Returns:
            True 表示熔断器开启（服务不可用），False 表示可以发送请求
        """
        return self._state == CircuitState.OPEN

    async def _check_state(self) -> None:
        """
        检查并尝试从 OPEN 状态转换到 HALF_OPEN 状态。

        转换条件：当前是 OPEN 状态，且距离上次失败时间超过 recovery_timeout。

        双重检查（double-checked locking 模式）：
        - 外层检查：不加锁，快速判断是否需要进入锁（减少锁竞争）
        - 内层检查：加锁后再次检查，防止多个协程同时通过外层检查后重复触发转换

        使用 time.monotonic() 而不是 time.time()：
        - monotonic 是单调递增的，不受系统时钟调整影响（如 NTP 校时）
        - 适合用于"经过了多少时间"的计算（相对时间）
        """
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                async with self._lock:
                    # 内层再次检查（双重检查锁定）
                    if self._state == CircuitState.OPEN:
                        self._state = CircuitState.HALF_OPEN
                        logger.info(
                            "circuit_breaker.half_open",
                            name=self.name,
                        )
                        metrics.circuit_breaker_open.labels(service=self.name).set(0)

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        通过熔断器保护地执行一次异步调用。

        调用流程：
        1. _check_state()：检查是否可以从 OPEN 转换到 HALF_OPEN
        2. 如果是 OPEN：拒绝，抛出 CircuitBreakerError（不调用 fn）
        3. 执行 fn(*args, **kwargs)（CLOSED 或 HALF_OPEN 状态）
        4. 成功：_on_success()
           - HALF_OPEN → CLOSED（服务恢复正常）
           - CLOSED：递减失败计数（min 0）
        5. 失败：_on_failure()
           - 增加失败计数
           - 如果达到阈值：CLOSED/HALF_OPEN → OPEN（熔断）

        异常分类：
        - CircuitBreakerError：不触发 _on_failure（不是服务失败，是熔断拒绝）
        - 其他异常：触发 _on_failure，重新抛出

        Args:
            fn: 要调用的异步函数（通常是 router.generate 等）
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            fn 的返回值

        Raises:
            CircuitBreakerError: 熔断器开启，请求被拒绝
            任意 fn 可能抛出的异常（经过 _on_failure 处理后重新抛出）
        """
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
            raise  # 熔断器自身的异常不触发 _on_failure
        except Exception as exc:
            await self._on_failure()
            raise exc

    async def _on_success(self) -> None:
        """
        处理一次成功调用的状态更新。

        HALF_OPEN → CLOSED：
        - 试探成功，服务恢复正常
        - 重置失败计数（归零）
        - 更新 Prometheus 指标（circuit_breaker_open = 0）

        CLOSED → CLOSED（递减失败计数）：
        - 成功调用"部分抵消"之前的失败
        - 防止偶发的网络抖动导致熔断器"记仇太久"
        - 使用 max(0, ...) 确保计数不会变成负数
        """
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # 试探成功 → 完全恢复
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("circuit_breaker.recovered", name=self.name)
                metrics.circuit_breaker_open.labels(service=self.name).set(0)
            elif self._state == CircuitState.CLOSED:
                # 正常成功 → 递减失败计数（渐进恢复）
                self._failure_count = max(0, self._failure_count - 1)

    async def _on_failure(self) -> None:
        """
        处理一次失败调用的状态更新。

        CLOSED → OPEN（当 failure_count >= threshold）：
        - 记录上次失败时间（用于 OPEN → HALF_OPEN 的超时计算）
        - 更新 Prometheus 指标（circuit_breaker_open = 1）
        - 记录 ERROR 级别日志（触发熔断是严重事件，值得告警）

        HALF_OPEN → OPEN：
        - 试探失败，服务仍然不可用
        - 重置超时计时（_last_failure_time 更新为当前时间）
        - 让恢复探测从现在开始重新计时
        """
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
                    # 首次达到阈值，触发熔断
                    self._state = CircuitState.OPEN
                    logger.error(
                        "circuit_breaker.opened",
                        name=self.name,
                        failures=self._failure_count,
                    )
                    metrics.circuit_breaker_open.labels(service=self.name).set(1)

    def reset(self) -> None:
        """
        手动重置熔断器到初始 CLOSED 状态（运维操作）。

        使用场景：
        - 运维人员确认服务已恢复，手动解除熔断（不等 recovery_timeout）
        - 单元测试在每个测试用例前重置状态（确保测试隔离）

        注意：这是同步方法（不需要 await），因为它只是简单的字段赋值，
        不需要与其他协程协调（运维操作通常是串行的）。
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        metrics.circuit_breaker_open.labels(service=self.name).set(0)
        logger.info("circuit_breaker.reset", name=self.name)


# ── 全局熔断器注册表 ─────────────────────────────────────────────────────────

_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    """
    获取指定服务的熔断器实例（懒加载单例）。

    每个服务名对应一个独立的 CircuitBreaker 实例，
    多次调用 get_breaker("llm") 返回同一个实例，保证状态共享。

    典型使用方式（在 assembler.py 中）：
        breaker = get_breaker("llm")
        agent_loop = AgentLoop(breaker=breaker, ...)

    Args:
        name: 服务名称（如 "llm", "database"）

    Returns:
        对应服务的 CircuitBreaker 实例（如果不存在则创建）
    """
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name)
    return _breakers[name]
