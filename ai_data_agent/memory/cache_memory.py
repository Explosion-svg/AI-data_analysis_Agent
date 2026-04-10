"""
memory/cache_memory.py — 结果缓存
TTL + LRU 淘汰策略，减少重复 LLM 调用和数据库查询
TTL:每个缓存项设置生存时间（TTL），到期后自动失效，无论是否被访问
LRU:当缓存空间不足时，优先淘汰最近最少使用的数据项。
支持 SQL 结果缓存 / RAG 结果缓存
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Any

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    # 缓存存储键值对中的值
    value: Any
    expires_at: float   # 过期时间
    hits: int = 0

    # 过期判断
    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class CacheMemory:
    """
    内存 LRU+TTL 缓存。
    生产环境可替换为 Redis（保持相同接口）。
    """

    def __init__(
        self,
        max_size: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._max_size = max_size or settings.cache_max_size
        self._ttl = ttl_seconds or settings.cache_ttl_seconds
        # 缓存存储结构
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    @staticmethod
    def make_key(*parts: Any) -> str:
        """
        参数--json序列化--hash--key
        将任意参数序列化为稳定的缓存键。
        """
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any:
        """缓存命中返回值，未命中或过期返回 None。"""
        entry = self._store.get(key)
        if entry is None:
            metrics.cache_misses_total.inc()
            return None
        if entry.is_expired:
            del self._store[key]
            metrics.cache_misses_total.inc()
            return None
        # LRU：移到末尾
        self._store.move_to_end(key)
        # 缓存命中次数
        entry.hits += 1
        metrics.cache_hits_total.inc()
        logger.debug("cache.hit", key=key[:16], hits=entry.hits)
        return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """写入缓存，自动淘汰过期项和最久未使用项。"""
        self._evict_expired()
        if len(self._store) >= self._max_size:
            # 淘汰最久未使用的项，OrderedDict第一个 = 最久未使用
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
            logger.debug("cache.evict_lru", key=oldest_key[:16])

        ttl_s = ttl if ttl is not None else self._ttl
        self._store[key] = CacheEntry(
            value=value,
            expires_at=time.monotonic() + ttl_s,
        )
        logger.debug("cache.set", key=key[:16], ttl=ttl_s)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()
        logger.info("cache.cleared")

    def _evict_expired(self) -> None:
        # 清理过期缓存
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]

    @property
    def size(self) -> int:
        self._evict_expired()
        return len(self._store)

    def stats(self) -> dict[str, Any]:
        self._evict_expired()
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }


# 全局单例
_cache: CacheMemory | None = None


def get_cache() -> CacheMemory:
    global _cache
    if _cache is None:
        _cache = CacheMemory()
    return _cache
