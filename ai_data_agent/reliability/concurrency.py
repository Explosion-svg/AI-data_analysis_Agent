"""
reliability/concurrency.py — 并发限流与舱壁隔离（Bulkhead Pattern）

职责：
  为 Agent 内部各类资源（LLM、数据库、工具等）提供独立的并发上限控制，
  防止单一依赖因突发流量拖垮整个系统（舱壁隔离原则）。

舱壁隔离（Bulkhead Isolation）原理：
  如同船舶的水密隔仓，把资源消耗分隔在独立的"桶"（bucket）里：
  - agent_request：全局请求级别并发（最外层限制，防止整体过载）
  - llm：LLM 调用并发（防止 LLM API rate limit 被单进程打爆）
  - sql_query、python_analysis：各工具独立并发槽（计算密集型工具不互相争抢）

  好处：
  - 某一工具的并发耗尽不影响其他工具的执行
  - 各桶的配额可以独立调优（从 settings 读取）

Semaphore 超时（acquire timeout）：
  如果等待获取 Semaphore 超过 concurrency_acquire_timeout_seconds，
  抛出 ConcurrencyLimitExceeded 而非无限等待。
  这防止了请求在高负载时永久阻塞，保证服务端能快速返回 503 而不是超时。

与 circuit_breaker 的区别：
  - concurrency：限制"当前"并发量（入口流量控制）
  - circuit_breaker：在"持续失败"后打开（出口故障检测）
  两者配合：先并发限流，过了限制再交给 circuit_breaker 检测故障

设计细节：
  - asyncio.Semaphore 在 asyncio 事件循环内是协程安全的（不需要 asyncio.Lock）
  - asyncio.wait_for(sem.acquire(), timeout=...) 实现带超时的获取
  - limit() 使用 @asynccontextmanager，保证即使 yield 内抛异常也会 release
  - max(1, ...) 确保 Semaphore 值至少为 1（settings 配置为 0 时的安全兜底）
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


class ConcurrencyLimitExceeded(RuntimeError):
    """
    等待并发槽超时时抛出的异常。

    继承 RuntimeError 而不是 Exception：
    - 表示"系统资源不足"这类运行时错误，而不是业务逻辑错误
    - AgentLoop 的异常处理会捕获此异常，返回 503/busy 响应

    额外属性 bucket 和 timeout_seconds 方便日志排查：
    - bucket：哪个资源桶满了（"llm"、"sql_query" 等）
    - timeout_seconds：等待了多少秒才放弃
    """
    def __init__(self, bucket: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Concurrency limit exceeded for '{bucket}' after waiting {timeout_seconds}s."
        )
        self.bucket = bucket
        self.timeout_seconds = timeout_seconds


class ConcurrencyLimiter:
    """
    多桶并发限流器（Multi-Bucket Concurrency Limiter）。

    每个桶对应一类资源，通过独立的 asyncio.Semaphore 控制最大并发量。
    桶的配额从 settings 中读取，支持按环境（dev/prod）差异化配置。

    使用方式（在 BaseTool.run() 和 AgentLoop 中）：
        async with get_limiter().limit("sql_query"):
            result = await tool.execute(sql)

    桶命名规则：
    - 与工具 name 属性保持一致（"sql_query", "python_analysis" 等），
      未知工具自动降级到 "tool" 通用桶，保证不崩溃

    注意：ConcurrencyLimiter 实例通过 get_limiter() 懒加载为全局单例，
    因此所有协程共享同一个 Semaphore 池（不是 per-request 的）。
    """

    def __init__(self) -> None:
        """
        初始化所有资源桶的 Semaphore。

        每个 Semaphore 的初始值来自 settings 对应配置项。
        max(1, ...) 保证即使 settings 配置为 0，也至少有 1 个并发槽，
        防止系统完全死锁（配置错误的容错兜底）。

        桶的设计遵循从粗到细的层次：
        - agent_request：最外层（整体请求级别）
        - llm / embedding：LLM 层（API 调用级别）
        - tool：通用工具层（未注册专用桶的工具使用此桶）
        - sql_query / python_analysis 等：具体工具层（按工具名精确控制）
        """
        # 并发量的限制，最多允许多少个协程同时运行
        self._semaphores: dict[str, asyncio.Semaphore] = {
            "agent_request": asyncio.Semaphore(max(1, settings.agent_request_concurrency)),
            "llm": asyncio.Semaphore(max(1, settings.llm_concurrency)),
            "embedding": asyncio.Semaphore(max(1, settings.embedding_concurrency)),
            "tool": asyncio.Semaphore(max(1, settings.tool_concurrency)),
            "sql_query": asyncio.Semaphore(max(1, settings.sql_tool_concurrency)),
            "python_analysis": asyncio.Semaphore(max(1, settings.python_tool_concurrency)),
            "search_documents": asyncio.Semaphore(max(1, settings.rag_tool_concurrency)),
            "generate_chart": asyncio.Semaphore(max(1, settings.chart_tool_concurrency)),
            "get_schema": asyncio.Semaphore(max(1, settings.schema_tool_concurrency)),
        }

    @asynccontextmanager
    async def limit(self, bucket: str) -> AsyncIterator[None]:
        """
        获取指定资源桶的并发槽，超时则抛出 ConcurrencyLimitExceeded。

        使用 @asynccontextmanager 的原因：
        - 确保在 try/finally 中调用 sem.release()，即使 yield 内部抛出异常也不会遗漏
        - 语义清晰：async with limiter.limit("sql"):... 比手动 acquire/release 更安全

        超时机制：
        - asyncio.wait_for(sem.acquire(), timeout=...) 在等待超时时取消 acquire 协程
        - 超时后 Semaphore 不会被 acquire（自动回滚），无需手动 release
        - 超时说明系统当前负载过高，应快速失败（fail-fast）而非排队等待

        未知桶的降级策略：
        - self._semaphores.get(bucket) or self._semaphores["tool"]
        - 未注册的工具名（如自定义扩展工具）自动使用通用 "tool" 桶
        - 保证系统不会因为找不到桶而崩溃（防御性设计）

        日志说明：
        - concurrency.acquire（DEBUG）：记录获取时的剩余配额（available = 当前可用槽数）
        - concurrency.release（DEBUG）：释放后的剩余配额（用于调试并发问题）
        - concurrency.timeout（WARNING）：超时时记录桶名和等待时长

        Args:
            bucket: 资源桶名称（如 "sql_query"、"llm"）；未知名称降级到 "tool"

        Yields:
            None（只用于并发控制，不返回数据）

        Raises:
            ConcurrencyLimitExceeded: 等待 Semaphore 超过 concurrency_acquire_timeout_seconds
        """
        sem = self._semaphores.get(bucket) or self._semaphores["tool"]
        timeout = settings.concurrency_acquire_timeout_seconds
        try:
            await asyncio.wait_for(sem.acquire(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            logger.warning(
                "concurrency.timeout",
                bucket=bucket,
                timeout_seconds=timeout,
                available=self.available(bucket),
            )
            raise ConcurrencyLimitExceeded(bucket, timeout) from exc
        logger.debug("concurrency.acquire", bucket=bucket, available=self.available(bucket))
        try:
            yield
        finally:
            # 无论 yield 内部是否抛异常，都会执行 release
            sem.release()
            logger.debug("concurrency.release", bucket=bucket, available=self.available(bucket))

    def available(self, bucket: str) -> int:
        """
        返回指定桶当前可用的并发槽数（供日志/指标使用）。

        只读查询，不影响并发语义。asyncio.Semaphore 没有公开的剩余值读取
        接口，这里把对私有属性 `_value` 的访问收敛到唯一方法内（P4-7），
        避免散落在各处直接触碰私有属性。
        """
        sem = self._semaphores.get(bucket) or self._semaphores["tool"]
        return int(sem._value)  # type: ignore[attr-defined]  # noqa: SLF001


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_limiter: ConcurrencyLimiter | None = None


def get_limiter() -> ConcurrencyLimiter:
    """
    获取全局 ConcurrencyLimiter 单例（懒加载）。

    懒加载的原因：
    - ConcurrencyLimiter 内部的 asyncio.Semaphore 必须在 asyncio 事件循环内创建
    - 如果在模块导入时（即事件循环启动前）创建，部分环境会报错
    - 懒加载确保第一次调用时事件循环已经在运行

    单例的必要性：
    - 多次调用 get_limiter() 必须返回同一个实例
    - 如果每次返回新实例，各自的 Semaphore 独立，并发限制就失效了

    Returns:
        全局 ConcurrencyLimiter 实例（不存在则创建）
    """
    global _limiter
    if _limiter is None:
        _limiter = ConcurrencyLimiter()
    return _limiter


def reset_limiter() -> None:
    """
    重置全局 ConcurrencyLimiter 单例（P4-7）。

    使用场景：
    - 测试之间：asyncio.Semaphore 绑定创建时的事件循环，跨事件循环复用
      会报 "bound to a different event loop"。conftest 在每个测试前调用，
      确保新测试拿到绑定当前事件循环的新信号量。
    - 配置热更新后需要重建并发配额时。

    注意：调用时应确保没有协程正在 limit() 上下文内使用旧实例。
    """
    global _limiter
    _limiter = None
