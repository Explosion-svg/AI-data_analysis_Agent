"""
memory/cache_memory.py — LRU + TTL 结果缓存

职责：
  缓存 Agent 响应结果，避免对相同问题重复调用 LLM 和数据库。
  使用 LRU（最近最少使用）淘汰策略 + TTL（生存时间）过期机制。

TTL（Time-To-Live）：
  每个缓存项都有过期时间，超过 TTL 后视为失效。
  适合结果可能随时间变化的数据（如销售数据可能每天更新）。
  默认 TTL = settings.cache_ttl_seconds（5 分钟），可按需配置。

LRU（Least Recently Used）：
  当缓存达到最大容量时，优先淘汰"最久未被访问"的项。
  实现：使用 Python 的 OrderedDict（有序字典）模拟 LRU：
  - get() 命中后：move_to_end()（移到末尾 = 最近使用）
  - set() 时如果满了：删除第一个元素（末尾 = 最近使用，头部 = 最久未用）

为什么同时需要 TTL 和 LRU？
  - 只有 TTL：空间可能被旧数据占满，即使它们已经不常访问
  - 只有 LRU：热点数据永远不过期，但数据可能已过时
  - 两者结合：既控制空间，又保证数据时效性

生产环境替换：
  保持与 BaseCacheMemory 接口一致，可以无缝替换为 RedisCacheMemory（Redis 后端）
  而不需要修改任何调用方代码。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Any

from ai_data_agent.config.config import settings
from ai_data_agent.memory.interfaces import BaseCacheMemory
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """
    单个缓存项，包含值、过期时间和命中计数。

    为什么需要命中计数（hits）？
    - 用于诊断：了解哪些查询被频繁缓存命中
    - 便于优化缓存策略：高命中率的项可以考虑延长 TTL
    - 当前只写入日志，暂不用于自适应 TTL

    expires_at 使用 time.monotonic()（单调递增时钟）而非 time.time()（墙钟时间）：
    - monotonic 不受系统时钟调整影响（如 NTP 校时）
    - 适合用于相对时间计算（"再过多少秒过期"）
    - 不适合用于绝对时间（不能转换成可读日期）
    """

    value: Any           # 缓存的值（通常是 AgentResponse 对象）
    expires_at: float    # 过期时间（time.monotonic() 时间戳）
    hits: int = 0        # 命中次数，用于调试和统计

    @property
    def is_expired(self) -> bool:
        """
        检查缓存项是否已过期。

        使用 time.monotonic() 与存储的 expires_at 比较，
        不需要额外的时区或时钟漂移处理。

        Returns:
            True 表示已过期（应从缓存删除），False 表示仍有效
        """
        return time.monotonic() > self.expires_at


class CacheMemory(BaseCacheMemory):
    """
    基于 OrderedDict 的内存 LRU + TTL 缓存。

    这是 BaseCacheMemory 的内存实现，适合单进程、单节点部署。
    生产环境多节点部署时，需替换为 RedisCacheMemory（支持跨进程共享缓存）。

    并发安全性：
    - 当前实现依赖 Python GIL 提供基本的线程安全
    - 在 asyncio 单线程环境中，协程级别的并发访问是安全的
    - 如果未来迁移到多线程，需要添加 threading.Lock
    """

    def __init__(
        self,
        max_size: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        初始化 LRU + TTL 缓存。

        Args:
            max_size: 最大缓存条目数，None 时使用 settings.cache_max_size（默认 256）
            ttl_seconds: 默认生存时间（秒），None 时使用 settings.cache_ttl_seconds（默认 300）
        """
        self._max_size = max_size or settings.cache_max_size
        self._ttl = ttl_seconds or settings.cache_ttl_seconds
        # OrderedDict 保持插入/更新顺序，move_to_end() 把项移到末尾
        # 头部 = 最久未使用（LRU 淘汰候选），末尾 = 最近使用
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    @staticmethod
    def make_key(*parts: Any) -> str:
        """
        将任意参数序列化为稳定的缓存键（SHA-256 哈希）。

        使用 JSON 序列化 + SHA-256 哈希：
        - JSON 序列化：确保不同类型的参数可以稳定地转换为字符串
          sort_keys=True 确保字典键顺序一致（{"b":1,"a":2} 和 {"a":2,"b":1} 生成相同键）
          default=str 处理非 JSON 可序列化对象（如 datetime）
        - SHA-256 哈希：将任意长度的输入转换为固定长度（64 字符）的键
          碰撞概率极低（2^256 中的 1），实践中可忽略

        Args:
            *parts: 构成缓存键的各个部分（query、tenant_id、conversation_id 等）

        Returns:
            64 字符的 SHA-256 十六进制哈希字符串

        Example:
            >>> CacheMemory.make_key("agent", "public", "今年GMV多少", "conv-123")
            "8f3a2b1c..."
        """
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any:
        """
        获取缓存值，同时更新 LRU 顺序和命中计数。

        访问流程：
        1. 查找 key 是否存在
        2. 如果不存在：记录 cache_miss，返回 None
        3. 如果已过期：删除项，记录 cache_miss，返回 None
        4. 如果有效：move_to_end()（更新 LRU 顺序），++hits，记录 cache_hit，返回值

        注意：过期检测是懒惰的（lazy eviction），只在访问时才检查和删除过期项。
        定期清理通过 _evict_expired() 方法完成，在 set() 时触发。

        Args:
            key: 缓存键（通常是 make_key() 生成的哈希值）

        Returns:
            缓存值，未命中或过期时返回 None
        """
        entry = self._store.get(key)
        if entry is None:
            metrics.cache_misses_total.inc()
            return None
        if entry.is_expired:
            # 惰性删除过期项（不保留引用，GC 可以回收）
            del self._store[key]
            metrics.cache_misses_total.inc()
            return None
        # 命中：将此项移到末尾（标记为"最近使用"）
        self._store.move_to_end(key)
        entry.hits += 1
        metrics.cache_hits_total.inc()
        logger.debug("cache.hit", key=key[:16], hits=entry.hits)
        return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        写入缓存，自动触发过期项清理和 LRU 淘汰。

        写入流程：
        1. 先清理所有已过期的项（_evict_expired）
        2. 如果仍然超出 max_size，淘汰最久未使用的项（OrderedDict 头部）
        3. 写入新项（移到末尾 = 最近使用）

        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: 此项的生存时间（秒），None 时使用默认 TTL
        """
        # 先清理过期项，尽量为新项腾出空间（避免误淘汰有效项）
        self._evict_expired()
        exists = key in self._store
        if not exists and len(self._store) >= self._max_size:
            # LRU 淘汰：删除 OrderedDict 的第一个元素（最久未使用）。
            # 仅"新增 key"需要腾位；覆盖已有 key 不增加条目数，不该误驱逐无辜条目（P4-7）。
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
            logger.debug("cache.evict_lru", key=oldest_key[:16])

        ttl_s = ttl if ttl is not None else self._ttl
        self._store[key] = CacheEntry(
            value=value,
            expires_at=time.monotonic() + ttl_s,
        )
        if exists:
            # P4-7：覆盖已有 key 时 move_to_end，保持 LRU 顺序。
            # OrderedDict 对已有 key 赋值不会改变其位置——若不 move_to_end，
            # 热 key 会留在 LRU 头部，容量满时被误驱逐。
            self._store.move_to_end(key)
        logger.debug("cache.set", key=key[:16], ttl=ttl_s)

    def delete(self, key: str) -> None:
        """
        删除指定缓存项（如果不存在则静默忽略）。

        Args:
            key: 要删除的缓存键
        """
        self._store.pop(key, None)

    def clear(self) -> None:
        """清空所有缓存项（用于测试或强制刷新）。"""
        self._store.clear()
        logger.info("cache.cleared")

    def _evict_expired(self) -> None:
        """
        清理所有已过期的缓存项。

        遍历所有项，找出已过期的 key，然后批量删除。
        注意：不在遍历过程中直接删除（Python 不允许在迭代时修改字典）。

        此方法在每次 set() 前调用，保持缓存的相对新鲜度。
        如果缓存项数量很大，此操作的时间复杂度为 O(n)，
        但对于典型的缓存大小（256 项），性能可忽略。
        """
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]

    @property
    def size(self) -> int:
        """
        返回有效缓存项数量（排除过期项）。

        调用此属性会触发一次过期清理，确保返回的数量准确。
        """
        self._evict_expired()
        return len(self._store)

    def stats(self) -> dict[str, Any]:
        """
        返回缓存统计信息，用于 health_report() 和监控。

        调用时会先清理过期项，确保 size 反映实际有效条目数。

        Returns:
            字典，包含 size（当前有效项数）、max_size（最大容量）、ttl_seconds（默认 TTL）
        """
        self._evict_expired()
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }


# ── 全局单例 ──────────────────────────────────────────────────────────────────

_cache: CacheMemory | None = None


def get_cache() -> CacheMemory:
    """
    获取全局缓存单例（懒加载）。

    在 assembler 调用 _init_memory() 前，此函数会创建默认配置的缓存实例。
    assembler 会在初始化后把正式的缓存实例赋值给 _cache，
    后续调用此函数返回 assembler 装配的实例。

    Returns:
        全局唯一的 CacheMemory 实例
    """
    global _cache
    if _cache is None:
        _cache = CacheMemory()
    return _cache
