"""
memory/factory.py — Memory/Cache 后端工厂

职责：
  根据配置（settings.memory_backend / settings.cache_backend）决定
  使用内存版还是 Redis 版的记忆/缓存实现，统一在这里做后端切换。

为什么用工厂模式？
  - 调用方（assembler）只需调用 build_xxx() 函数，不需要知道后端细节
  - 切换后端只需修改 .env 文件（MEMORY_BACKEND=redis），无需改代码
  - 每个工厂函数封装了该组件的 Redis 配置参数，保持 assembler 简洁

当前策略：
  - memory_backend="memory"（默认）：使用内存版，适合开发和单节点部署
  - memory_backend="redis"：使用 Redis 版，适合多节点水平扩展
  - cache_backend 独立配置，可以与 memory_backend 不同
    （如：对话记忆用内存，只有结果缓存用 Redis）

Redis 版特性：
  - fail_open=True：Redis 故障时降级到无缓存/无记忆，而不是报错
  - startup_check=True：启动时检查 Redis 连通性，提前暴露配置问题
  - 乐观锁（WATCH/MULTI/EXEC）：防止并发写冲突导致数据覆盖

注意：
  Redis 版的 conversation_memory 和 work_memory 需要序列化/反序列化，
  性能会略低于内存版，但支持跨进程共享和持久化。
"""
from __future__ import annotations

from ai_data_agent.config.config import settings
from ai_data_agent.memory.cache_memory import CacheMemory
from ai_data_agent.memory.conversation_memory import ConversationMemory
from ai_data_agent.memory.redis_cache_memory import RedisCacheMemory
from ai_data_agent.memory.redis_conversation_memory import RedisConversationMemory
from ai_data_agent.memory.redis_work_memory import RedisWorkMemory
from ai_data_agent.memory.work_memory import WorkMemory


def build_conversation_memory(*, router, breaker) -> ConversationMemory:
    """
    根据配置构建对话记忆实例。

    选择逻辑：
    - memory_backend="redis"：使用 Redis 持久化后端
      优点：多节点共享、重启不丢失、支持 TTL 自动清理
      需要：Redis 服务可用
    - 其他（默认 "memory"）：使用内存后端
      优点：零额外依赖、最低延迟
      缺点：进程重启后丢失、无法跨节点共享

    router 和 breaker 参数传递给 ConversationMemory 的原因：
    - 对话记忆在窗口溢出时需要调用 LLM 生成滚动摘要（异步）
    - breaker 包裹 LLM 调用，防止摘要生成失败影响主流程

    Args:
        router: ModelRouter 实例（用于 LLM 摘要调用）
        breaker: CircuitBreaker 实例（保护 LLM 调用）

    Returns:
        配置好的 ConversationMemory 实例（内存版或 Redis 版）
    """
    if settings.memory_backend == "redis":
        return RedisConversationMemory(
            max_turns=settings.conversation_max_turns,
            router=router,
            breaker=breaker,
            redis_url=settings.redis_url,
            # 环境前缀隔离不同部署环境（dev/staging/prod）的数据
            prefix=f"{settings.redis_conversation_prefix}:{settings.env.value}",
            ttl_seconds=settings.redis_conversation_ttl_seconds,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_connect_timeout,
            health_check_interval=settings.redis_health_check_interval,
            retry_on_timeout=settings.redis_retry_on_timeout,
            fail_open=settings.redis_cache_fail_open,
            startup_check=settings.redis_cache_startup_check,
        )
    return ConversationMemory(
        max_turns=settings.conversation_max_turns,
        router=router,
        breaker=breaker,
    )


def build_work_memory() -> WorkMemory:
    """
    根据配置构建工作记忆实例。

    工作记忆记录单次任务的执行轨迹（步骤、产物、发现等）。
    Redis 版支持跨进程访问，适合在多个 worker 之间共享任务状态。

    注意：工作记忆不需要 router/breaker，因为它不调用 LLM，
    只是纯粹的状态存储。

    Returns:
        配置好的 WorkMemory 实例（内存版或 Redis 版）
    """
    if settings.memory_backend == "redis":
        return RedisWorkMemory(
            redis_url=settings.redis_url,
            prefix=f"{settings.redis_work_prefix}:{settings.env.value}",
            ttl_seconds=settings.redis_work_ttl_seconds,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_connect_timeout,
            health_check_interval=settings.redis_health_check_interval,
            retry_on_timeout=settings.redis_retry_on_timeout,
            fail_open=settings.redis_cache_fail_open,
            startup_check=settings.redis_cache_startup_check,
        )
    return WorkMemory()


def build_cache_memory():
    """
    根据配置构建结果缓存实例。

    结果缓存独立于对话/工作记忆，可以使用不同的后端配置：
    - cache_backend="redis"：使用 Redis，支持跨节点共享缓存
    - cache_backend="memory"（默认）：使用内存 LRU+TTL 缓存

    独立配置的意义：
    - 某些部署场景可能只需要分布式缓存（多节点共享），但不需要分布式记忆
    - 允许灵活组合：如 memory_backend="memory" + cache_backend="redis"

    Returns:
        配置好的缓存实例（CacheMemory 内存版或 RedisCacheMemory Redis 版）
    """
    if settings.cache_backend == "redis":
        return RedisCacheMemory(
            redis_url=settings.redis_url,
            prefix=f"{settings.redis_cache_prefix}:{settings.env.value}",
            default_ttl=settings.cache_ttl_seconds,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_connect_timeout,
            health_check_interval=settings.redis_health_check_interval,
            retry_on_timeout=settings.redis_retry_on_timeout,
            fail_open=settings.redis_cache_fail_open,
            startup_check=settings.redis_cache_startup_check,
        )
    return CacheMemory(
        max_size=settings.cache_max_size,
        ttl_seconds=settings.cache_ttl_seconds,
    )
