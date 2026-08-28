"""
tests/unit/test_memory.py

内存状态模块单元测试。

主要验证：
- ConversationMemory 的会话隔离、窗口裁剪、清理与摘要
- CacheMemory 的命中、过期、LRU 淘汰
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
import pytest

from ai_data_agent.memory.cache_memory import CacheMemory
from ai_data_agent.memory.conversation_memory import ConversationMemory
from ai_data_agent.memory.conversation_memory import ConversationState, Turn
from ai_data_agent.memory import redis_conversation_memory as redis_conversation_memory_module
from ai_data_agent.memory.redis_conversation_memory import RedisConversationMemory
from ai_data_agent.memory.redis_work_memory import RedisWorkMemory
from ai_data_agent.memory.work_memory import WorkArtifact, WorkState, WorkStep


class _FakeRedisPipeline:
    def __init__(self, client: "_FakeRedisClient") -> None:
        self._client = client
        self._pending: tuple[str, str, int | None] | None = None

    def __enter__(self) -> "_FakeRedisPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def watch(self, key: str) -> None:
        self._key = key

    def get(self, key: str) -> str | None:
        return self._client.store.get(key)

    def multi(self) -> None:
        return None

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._pending = (key, value, ex)

    def execute(self) -> None:
        if self._client.raise_watch_once:
            self._client.raise_watch_once = False
            raise self._client.watch_error_type()
        if self._pending is not None:
            key, value, ex = self._pending
            self._client.store[key] = value
            self._client.expiry[key] = ex


class _FakeRedisClient:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.expiry: dict[str, int | None] = {}
        self.raise_watch_once = False
        self.watch_error_type = RuntimeError

    def pipeline(self) -> _FakeRedisPipeline:
        return _FakeRedisPipeline(self)

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def scan_iter(self, match: str | None = None, count: int | None = None):
        prefix = (match or "").rstrip("*")
        for key in list(self.store):
            if not prefix or key.startswith(prefix):
                yield key


def _build_redis_work_memory() -> RedisWorkMemory:
    memory = RedisWorkMemory.__new__(RedisWorkMemory)
    memory._store = {}
    memory._versions = {}
    memory._prefix = "test:work"
    memory._ttl_seconds = 123
    memory._fail_open = False
    memory._available = True
    memory._client = _FakeRedisClient()
    return memory


def _build_redis_conversation_memory() -> RedisConversationMemory:
    memory = RedisConversationMemory.__new__(RedisConversationMemory)
    memory._store = {}
    memory._versions = {}
    memory._prefix = "test:conversation"
    memory._ttl_seconds = 456
    memory._fail_open = False
    memory._available = True
    memory._client = _FakeRedisClient()
    return memory


@pytest.mark.asyncio
async def test_conversation_memory_isolation_and_trim() -> None:
    # 准备两个会话，其中 c1 故意写入超过窗口的数据。
    memory = ConversationMemory(max_turns=2)

    await memory.add("c1", "user", "u1")
    await memory.add("c1", "assistant", "a1")
    await memory.add("c1", "user", "u2")
    await memory.add("c1", "assistant", "a2")
    await memory.add("c1", "user", "u3")
    await memory.add("c1", "assistant", "a3")
    await memory.add("c2", "user", "other")

    c1_messages = memory.get_messages("c1")
    c2_messages = memory.get_messages("c2")

    # c1 现在会在最前面额外注入一条长期摘要 system message。
    assert c1_messages[0].role == "system"
    assert [m.content for m in c1_messages[1:]] == ["u2", "a2", "u3", "a3"]
    assert [m.content for m in c2_messages] == ["other"]


@pytest.mark.asyncio
async def test_conversation_memory_clear_and_summary() -> None:
    # 验证 summary 统计与 clear 行为。
    memory = ConversationMemory(max_turns=2)
    await memory.add("c1", "user", "hello")
    await memory.add("c1", "assistant", "world")

    assert memory.summary("c1") == {
        "conversation_id": "c1",
        "turns": 1,
        "messages": 2,
        "has_rolling_summary": False,
        "rolling_summary_chars": 0,
        "pinned_facts": 0,
    }

    memory.clear("c1")

    assert memory.get_messages("c1") == []


def test_cache_memory_hit_miss_ttl_and_lru() -> None:
    # 这个测试一次性覆盖缓存的四个关键行为：
    # 1. 未命中
    # 2. 命中
    # 3. LRU 淘汰
    # 4. TTL 过期
    cache = CacheMemory(max_size=2, ttl_seconds=1)
    key1 = cache.make_key("a")
    key2 = cache.make_key("b")
    key3 = cache.make_key("c")

    assert cache.get(key1) is None

    cache.set(key1, 1)
    cache.set(key2, 2)
    assert cache.get(key1) == 1

    cache.set(key3, 3)
    assert cache.get(key2) is None
    assert cache.get(key1) == 1
    assert cache.get(key3) == 3

    cache.set("temp", "x", ttl=0)
    time.sleep(0.01)
    assert cache.get("temp") is None


def test_redis_work_memory_merges_conflicting_updates() -> None:
    memory = _build_redis_work_memory()
    now = datetime.utcnow()
    remote = WorkState(
        conversation_id="c1",
        run_id="remote",
        status="running",
        original_query="remote query",
        findings=["remote finding"],
        selected_tables=["orders"],
        iterations=1,
        steps=[
            WorkStep(
                step_id="remote-step",
                iteration=1,
                tool="sql_query",
                status="done",
                started_at=now,
                finished_at=now + timedelta(seconds=1),
                result_summary="remote result",
            )
        ],
        artifacts=[
            WorkArtifact(
                artifact_id="remote-artifact",
                type="sql_result",
                preview="remote preview",
                created_at=now,
            )
        ],
        created_at=now,
        updated_at=now,
    )
    local = WorkState(
        conversation_id="c1",
        run_id="local",
        status="completed",
        original_query="local query",
        rewritten_query="rewritten",
        findings=["local finding"],
        selected_tables=["customers"],
        latest_sql="select * from customers",
        iterations=3,
        steps=[
            WorkStep(
                step_id="local-step",
                iteration=2,
                tool="python_analysis",
                status="done",
                started_at=now + timedelta(seconds=2),
                finished_at=now + timedelta(seconds=3),
                result_summary="local result",
            )
        ],
        artifacts=[
            WorkArtifact(
                artifact_id="local-artifact",
                type="chart",
                preview="local chart",
                created_at=now + timedelta(seconds=2),
            )
        ],
        final_answer="done",
        created_at=now + timedelta(seconds=1),
        updated_at=now + timedelta(seconds=3),
        completed_at=now + timedelta(seconds=3),
    )

    key = memory._full_key("c1")
    memory._client.store[key] = memory._serialize_state(remote, version=1)
    memory._versions["c1"] = 0

    assert memory._persist_state_with_lock("c1", local) is True

    persisted, version = memory._deserialize_state(memory._client.store[key])
    assert version == 2
    assert persisted.status == "completed"
    assert persisted.iterations == 3
    assert persisted.latest_sql == "select * from customers"
    assert persisted.final_answer == "done"
    assert persisted.findings == ["remote finding", "local finding"]
    assert persisted.selected_tables == ["orders", "customers"]
    assert {step.step_id for step in persisted.steps} == {"remote-step", "local-step"}
    assert {artifact.artifact_id for artifact in persisted.artifacts} == {
        "remote-artifact",
        "local-artifact",
    }
    assert memory._client.expiry[key] == 123


def test_redis_conversation_memory_retries_and_merges_conflicts() -> None:
    memory = _build_redis_conversation_memory()
    memory._client.raise_watch_once = True
    memory._client.watch_error_type = redis_conversation_memory_module.WatchError

    now = datetime.utcnow()
    remote = ConversationState(
        recent_turns=[
            Turn(role="user", content="remote question", timestamp=now),
        ],
        rolling_summary="older summary",
        pinned_facts=["tenant=acme"],
    )
    local = ConversationState(
        recent_turns=[
            Turn(role="assistant", content="local answer", timestamp=now + timedelta(seconds=1)),
        ],
        rolling_summary="older summary with more detail",
        pinned_facts=["currency=CNY"],
    )

    key = memory._full_key("c1")
    memory._client.store[key] = memory._serialize_state(remote, version=1)
    memory._versions["c1"] = 0

    assert memory._persist_state_with_lock("c1", local) is True

    persisted, version = memory._deserialize_state(memory._client.store[key])
    assert version == 2
    assert [turn.content for turn in persisted.recent_turns] == [
        "remote question",
        "local answer",
    ]
    assert persisted.rolling_summary == "older summary with more detail"
    assert persisted.pinned_facts == ["tenant=acme", "currency=CNY"]
    assert memory._client.expiry[key] == 456
