"""
memory/redis_conversation_memory.py — Redis 会话记忆

目标：
- 多实例共享 conversation memory
- 保留现有滚动摘要和 pinned facts 策略
- Redis 故障时 fail-open，避免影响主对话链路
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from ai_data_agent.config.config import settings
from ai_data_agent.memory.conversation_memory import (
    ConversationMemory,
    ConversationState,
    Turn,
)
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

try:
    from redis import Redis
    from redis.exceptions import RedisError, WatchError
except Exception:  # pragma: no cover - optional dependency
    Redis = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment]
    WatchError = Exception  # type: ignore[assignment]


class RedisConversationMemory(ConversationMemory):
    def __init__(
        self,
        max_turns: int | None = None,
        *,
        router=None,
        breaker=None,
        redis_url: str,
        prefix: str = "ai_data_agent:conversation",
        ttl_seconds: int = 604800,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        health_check_interval: int = 30,
        retry_on_timeout: bool = True,
        fail_open: bool = True,
        startup_check: bool = True,
    ) -> None:
        super().__init__(max_turns=max_turns, router=router, breaker=breaker)
        if Redis is None:
            raise RuntimeError("redis package is required for RedisConversationMemory.")

        self._prefix = prefix.rstrip(":")
        self._ttl_seconds = max(300, int(ttl_seconds))
        self._fail_open = fail_open
        self._health_check_interval = float(health_check_interval)
        self._client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            health_check_interval=health_check_interval,
            retry_on_timeout=retry_on_timeout,
        )
        self._available = True
        self._last_ping = time.monotonic()
        # P4-7：_versions 从普通 dict 改为 OrderedDict 并施加与本地会话存储一致的
        # LRU 封顶，防止随会话数线性增长的无界内存泄漏。
        self._versions: OrderedDict[str, int] = OrderedDict()

        if startup_check:
            self._ping_on_startup()

    async def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # P2-15：Redis 读/写都是同步网络调用，包 to_thread 避免阻塞事件循环
        await asyncio.to_thread(self._ensure_loaded, conversation_id)
        await super().add(conversation_id, role, content, metadata)
        await asyncio.to_thread(self._persist_if_present, conversation_id)

    def get_messages(self, conversation_id: str):
        self._ensure_loaded(conversation_id)
        return super().get_messages(conversation_id)

    def get_turns(self, conversation_id: str):
        self._ensure_loaded(conversation_id)
        return super().get_turns(conversation_id)

    def clear(self, conversation_id: str) -> None:
        super().clear(conversation_id)
        self._versions.pop(conversation_id, None)
        self._safe_call("delete", key=conversation_id, fn=lambda: self._client.delete(self._full_key(conversation_id)))

    def list_conversations(self) -> list[str]:
        ids = set(super().list_conversations())
        try:
            for key in self._client.scan_iter(match=f"{self._prefix}:*", count=200):
                ids.add(key.split(f"{self._prefix}:", 1)[1])
        except RedisError as e:
            self._handle_error("list_conversations", e, key=None)
        return sorted(ids)

    def summary(self, conversation_id: str) -> dict:
        self._ensure_loaded(conversation_id)
        base = super().summary(conversation_id)
        base["backend"] = "redis" if self._available else "redis_unavailable"
        return base

    def _ensure_loaded(self, conversation_id: str) -> None:
        if conversation_id in self._store:
            return
        loaded = self._load_state(conversation_id)
        if loaded is not None:
            self._set_state(conversation_id, loaded)

    def _persist_if_present(self, conversation_id: str) -> None:
        state = self._store.get(conversation_id)
        if state is None:
            return
        self._safe_call(
            "persist",
            key=conversation_id,
            fn=lambda: self._persist_state_with_lock(conversation_id, state),
        )

    def _load_state(self, conversation_id: str) -> ConversationState | None:
        raw = self._safe_call(
            "get",
            key=conversation_id,
            fn=lambda: self._client.get(self._full_key(conversation_id)),
        )
        if not raw:
            return None
        try:
            state, version = self._deserialize_state(raw)
            self._set_version(conversation_id, version)
            return state
        except Exception as e:
            logger.warning(
                "redis_conversation_memory.deserialize_failed",
                conversation_id=conversation_id,
                error=str(e),
            )
            return None

    def _set_version(self, conversation_id: str, version: int) -> None:
        """写入会话版本号，并维护 LRU 封顶（P4-7：防止 _versions 无界增长）。"""
        self._versions[conversation_id] = version
        self._versions.move_to_end(conversation_id)
        while len(self._versions) > self._max_conversations:
            self._versions.popitem(last=False)

    def _full_key(self, conversation_id: str) -> str:
        return f"{self._prefix}:{conversation_id}"

    def _ping_on_startup(self) -> None:
        try:
            self._client.ping()
            logger.info("redis_conversation_memory.ready", prefix=self._prefix, ttl=self._ttl_seconds)
        except RedisError as e:
            self._available = False
            self._handle_error("startup_ping", e, key=None)
            if not self._fail_open:
                raise RuntimeError(f"Redis conversation memory startup check failed: {e}") from e

    def close(self) -> None:
        """
        关闭 Redis 客户端连接池（P2-20）。

        应用优雅关闭时由 AppContainer.shutdown() 调用。
        幂等：客户端已关闭时直接返回，可重复调用。
        """
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception as e:  # pragma: no cover - 防御性
            logger.warning("redis_conversation_memory.close_failed", error=str(e))
        self._client = None

    def _safe_call(self, op: str, *, key: str | None, fn) -> Any:
        if not self._available and self._fail_open:
            # P2-14：瞬时故障后的周期性恢复探测。
            # fail-open 是单向的（只有成功路径能翻回 True），
            # 因此不可用时周期性 ping，Redis 恢复后自动复位。
            if time.monotonic() - self._last_ping >= self._health_check_interval:
                try:
                    self._client.ping()
                    self._available = True
                    logger.info("redis_conversation_memory.recovered", prefix=self._prefix)
                except RedisError:
                    self._last_ping = time.monotonic()
            return None
        try:
            result = fn()
            self._available = True
            return result
        except RedisError as e:
            self._available = False
            self._last_ping = time.monotonic()
            self._handle_error(op, e, key=key)
            if self._fail_open:
                return None
            raise

    def _handle_error(self, op: str, error: Exception, *, key: str | None) -> None:
        logger.warning(
            "redis_conversation_memory.error",
            operation=op,
            key=key or "",
            error=str(error),
            fail_open=self._fail_open,
        )

    def _persist_state_with_lock(self, conversation_id: str, state: ConversationState) -> bool:
        key = self._full_key(conversation_id)
        expected_version = self._versions.get(conversation_id, 0)
        merged_state = state

        for _ in range(max(1, settings.redis_optimistic_lock_retries)):
            with self._client.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    current_state, current_version = self._deserialize_state(raw) if raw else (None, 0)

                    if current_version != expected_version:
                        if current_state is None:
                            expected_version = 0
                        else:
                            # P4-7：合并后按预算截断 recent_turns，避免并发合并
                            # 把窗口撑到 2 倍预算（本地 _roll 会截断，但合并路径不会）。
                            merged_state = self._merge_states(
                                current_state, merged_state, max_turns=self._max_turns
                            )
                            expected_version = current_version

                    payload = self._serialize_state(merged_state, expected_version + 1)
                    pipe.multi()
                    pipe.set(key, payload, ex=self._ttl_seconds)
                    pipe.execute()
                    self._set_version(conversation_id, expected_version + 1)
                    self._set_state(conversation_id, merged_state)
                    return True
                except WatchError:
                    continue
        logger.warning("redis_conversation_memory.cas_exhausted", conversation_id=conversation_id)
        return False

    @staticmethod
    def _serialize_state(state: ConversationState, version: int) -> str:
        return json.dumps(
            {"version": version, "state": asdict(state)},
            default=RedisConversationMemory._json_default,
            ensure_ascii=False,
        )

    @staticmethod
    def _deserialize_state(raw: str) -> tuple[ConversationState, int]:
        payload = json.loads(raw)
        version = int(payload.get("version", 0))
        state_payload = payload.get("state", payload)
        recent_turns = [
            Turn(
                role=item.get("role", ""),
                content=item.get("content", ""),
                timestamp=RedisConversationMemory._parse_datetime(item.get("timestamp")) or datetime.now(timezone.utc),
                metadata=dict(item.get("metadata", {})),
            )
            for item in state_payload.get("recent_turns", [])
        ]
        return ConversationState(
            recent_turns=recent_turns,
            rolling_summary=state_payload.get("rolling_summary", ""),
            pinned_facts=list(state_payload.get("pinned_facts", [])),
        ), version

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _merge_states(
        remote: ConversationState,
        local: ConversationState,
        *,
        max_turns: int,
    ) -> ConversationState:
        turns: list[Turn] = []
        seen: set[tuple[str, str, str]] = set()
        for turn in sorted(remote.recent_turns + local.recent_turns, key=lambda item: item.timestamp):
            key = (turn.role, turn.content, turn.timestamp.isoformat())
            if key in seen:
                continue
            seen.add(key)
            turns.append(turn)

        # P4-7：合并后按预算截断，防止并发合并把 recent_turns 撑到 2 倍预算。
        # 截断规则与 ConversationMemory._roll_recent_turns_into_summary 一致——
        # 优先保留最新的 user→assistant 完整对；从最旧端淘汰。
        max_messages = max(2, max_turns * 2)
        if len(turns) > max_messages:
            idx = 0
            while len(turns) - idx > max_messages and idx < len(turns):
                if (
                    idx + 1 < len(turns)
                    and turns[idx].role == "user"
                    and turns[idx + 1].role == "assistant"
                ):
                    idx += 2
                else:
                    idx += 1
            turns = turns[idx:]

        pinned_facts: list[str] = []
        for item in remote.pinned_facts + local.pinned_facts:
            if item and item not in pinned_facts:
                pinned_facts.append(item)

        rolling_summary = local.rolling_summary or remote.rolling_summary
        if remote.rolling_summary and local.rolling_summary:
            rolling_summary = local.rolling_summary if len(local.rolling_summary) >= len(remote.rolling_summary) else remote.rolling_summary

        return ConversationState(
            recent_turns=turns,
            rolling_summary=rolling_summary,
            pinned_facts=pinned_facts,
        )
