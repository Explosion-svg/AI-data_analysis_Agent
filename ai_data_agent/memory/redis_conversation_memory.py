"""
memory/redis_conversation_memory.py — Redis 会话记忆

目标：
- 多实例共享 conversation memory
- 保留现有滚动摘要和 pinned facts 策略
- Redis 故障时 fail-open，避免影响主对话链路
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
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
        self._client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            health_check_interval=health_check_interval,
            retry_on_timeout=retry_on_timeout,
        )
        self._available = True
        self._versions: dict[str, int] = {}

        if startup_check:
            self._ping_on_startup()

    async def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._ensure_loaded(conversation_id)
        await super().add(conversation_id, role, content, metadata)
        self._persist_if_present(conversation_id)

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
            self._store[conversation_id] = loaded

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
            self._versions[conversation_id] = version
            return state
        except Exception as e:
            logger.warning(
                "redis_conversation_memory.deserialize_failed",
                conversation_id=conversation_id,
                error=str(e),
            )
            return None

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
                            merged_state = self._merge_states(current_state, merged_state)
                            expected_version = current_version

                    payload = self._serialize_state(merged_state, expected_version + 1)
                    pipe.multi()
                    pipe.set(key, payload, ex=self._ttl_seconds)
                    pipe.execute()
                    self._versions[conversation_id] = expected_version + 1
                    self._store[conversation_id] = merged_state
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
                timestamp=RedisConversationMemory._parse_datetime(item.get("timestamp")) or datetime.utcnow(),
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
    def _merge_states(remote: ConversationState, local: ConversationState) -> ConversationState:
        turns: list[Turn] = []
        seen: set[tuple[str, str, str]] = set()
        for turn in sorted(remote.recent_turns + local.recent_turns, key=lambda item: item.timestamp):
            key = (turn.role, turn.content, turn.timestamp.isoformat())
            if key in seen:
                continue
            seen.add(key)
            turns.append(turn)

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
