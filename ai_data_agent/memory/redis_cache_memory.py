"""
memory/redis_cache_memory.py — Redis 缓存实现

生产目标：
- 多实例共享缓存
- 默认 TTL，避免永久脏缓存
- Redis 故障时 fail-open，不阻塞主业务链路
- 保持与 BaseCacheMemory 兼容
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

from ai_data_agent.memory.interfaces import BaseCacheMemory
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)

try:
    from redis import Redis
    from redis.exceptions import RedisError
except Exception:  # pragma: no cover - 依赖可选
    Redis = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment]


class RedisCacheMemory(BaseCacheMemory):
    def __init__(
        self,
        redis_url: str,
        prefix: str = "ai_data_agent:cache",
        *,
        default_ttl: int = 300,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        health_check_interval: int = 30,
        retry_on_timeout: bool = True,
        fail_open: bool = True,
        startup_check: bool = True,
    ) -> None:
        if Redis is None:
            raise RuntimeError("redis package is required for RedisCacheMemory.")

        self._prefix = prefix.rstrip(":")
        self._default_ttl = max(1, int(default_ttl))
        self._fail_open = fail_open
        self._client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            health_check_interval=health_check_interval,
            retry_on_timeout=retry_on_timeout,
        )
        self._available = True

        if startup_check:
            self._ping_on_startup()

    @staticmethod
    def make_key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any:
        value = self._safe_call("get", key=key, fn=lambda: self._client.get(self._full_key(key)))
        if value is None:
            metrics.cache_misses_total.inc()
            return None

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("redis_cache.invalid_json", key=key[:16])
            metrics.cache_misses_total.inc()
            return None

        metrics.cache_hits_total.inc()
        logger.debug("redis_cache.hit", key=key[:16])
        return self._deserialize(parsed)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl_s = max(1, int(ttl if ttl is not None else self._default_ttl))
        payload = json.dumps(
            self._serialize(value),
            default=self._json_default,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._safe_call(
            "set",
            key=key,
            fn=lambda: self._client.set(self._full_key(key), payload, ex=ttl_s),
        )
        logger.debug("redis_cache.set", key=key[:16], ttl=ttl_s)

    def delete(self, key: str) -> None:
        self._safe_call("delete", key=key, fn=lambda: self._client.delete(self._full_key(key)))

    def clear(self) -> None:
        if not self._available and self._fail_open:
            logger.warning("redis_cache.clear_skipped", reason="cache_unavailable")
            return

        deleted = 0
        try:
            pipe = self._client.pipeline(transaction=False)
            batch_size = 0
            for key in self._client.scan_iter(match=f"{self._prefix}:*", count=500):
                pipe.delete(key)
                batch_size += 1
                deleted += 1
                if batch_size >= 500:
                    pipe.execute()
                    pipe = self._client.pipeline(transaction=False)
                    batch_size = 0
            if batch_size:
                pipe.execute()
            logger.info("redis_cache.cleared", keys=deleted)
        except RedisError as e:
            self._handle_error("clear", e, key=None)

    def stats(self) -> dict[str, Any]:
        size = 0
        sample_ttl = None
        available = self._available

        try:
            for idx, key in enumerate(self._client.scan_iter(match=f"{self._prefix}:*", count=200), start=1):
                size += 1
                if sample_ttl is None:
                    sample_ttl = self._client.ttl(key)
                if idx >= 1000:
                    break
        except RedisError as e:
            available = False
            self._handle_error("stats", e, key=None)

        return {
            "backend": "redis",
            "prefix": self._prefix,
            "default_ttl_seconds": self._default_ttl,
            "available": available,
            "size_estimate": size,
            "sample_ttl_seconds": sample_ttl,
            "fail_open": self._fail_open,
        }

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def _ping_on_startup(self) -> None:
        try:
            self._client.ping()
            logger.info("redis_cache.ready", prefix=self._prefix, ttl=self._default_ttl)
        except RedisError as e:
            self._available = False
            self._handle_error("startup_ping", e, key=None)
            if not self._fail_open:
                raise RuntimeError(f"Redis cache startup check failed: {e}") from e

    def _safe_call(self, op: str, *, key: str | None, fn) -> Any:
        if not self._available and self._fail_open:
            return None
        try:
            result = fn()
            self._available = True
            return result
        except RedisError as e:
            self._available = False
            self._handle_error(op, e, key=key)
            if self._fail_open:
                return None
            raise

    def _handle_error(self, op: str, error: Exception, *, key: str | None) -> None:
        logger.warning(
            "redis_cache.error",
            operation=op,
            key=(key[:16] if key else ""),
            error=str(error),
            fail_open=self._fail_open,
        )

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        """
        递归序列化缓存值，保证 dataclass / enum / 时间类型可稳定落盘。
        """
        if is_dataclass(value):
            return {
                "__type__": value.__class__.__name__,
                "payload": cls._serialize(asdict(value)),
            }
        if isinstance(value, dict):
            return {str(k): cls._serialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._serialize(v) for v in value]
        if isinstance(value, tuple):
            return [cls._serialize(v) for v in value]
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        return value

    @classmethod
    def _deserialize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("__type__") == "AgentResponse":
            from ai_data_agent.orchestration.agent_loop import AgentResponse

            payload = value.get("payload", {})
            return AgentResponse(**payload)
        return {k: cls._deserialize(v) for k, v in value.items()}

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        return str(value)
