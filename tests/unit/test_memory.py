"""
tests/unit/test_memory.py

内存状态模块单元测试。

主要验证：
- ConversationMemory 的会话隔离、窗口裁剪、清理与摘要
- CacheMemory 的命中、过期、LRU 淘汰
"""

from __future__ import annotations

import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import pytest

from ai_data_agent.memory.cache_memory import CacheMemory
from ai_data_agent.memory.conversation_memory import ConversationMemory
from ai_data_agent.memory.conversation_memory import ConversationState, Turn
from ai_data_agent.memory import redis_conversation_memory as redis_conversation_memory_module
from ai_data_agent.memory.redis_cache_memory import RedisCacheMemory
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
        self.ping_fails = False

    def ping(self) -> None:
        # P2-14：模拟 Redis 故障（ping_fails=True 时抛 RedisError）
        if self.ping_fails:
            raise redis_conversation_memory_module.RedisError("redis down")

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
    # P2-18：本地读缓存使用 OrderedDict 以支持 LRU 驱逐（与生产实现一致）
    memory._store = OrderedDict()
    # P4-7：_versions 同样为 OrderedDict（生产实现支持 LRU 封顶）
    memory._versions = OrderedDict()
    memory._max_conversations = 1000
    memory._prefix = "test:work"
    memory._ttl_seconds = 123
    memory._fail_open = False
    memory._available = True
    memory._health_check_interval = 30.0
    memory._client = _FakeRedisClient()
    return memory


def _build_redis_conversation_memory() -> RedisConversationMemory:
    memory = RedisConversationMemory.__new__(RedisConversationMemory)
    # P2-18：本地读缓存使用 OrderedDict 以支持 LRU 驱逐（与生产实现一致）
    memory._store = OrderedDict()
    # P4-7：_versions 同样为 OrderedDict（生产实现支持 LRU 封顶）
    memory._versions = OrderedDict()
    memory._max_conversations = 1000
    # P4-7：合并路径会按 _max_turns 预算截断 recent_turns
    memory._max_turns = 20
    memory._prefix = "test:conversation"
    memory._ttl_seconds = 456
    memory._fail_open = False
    memory._available = True
    memory._health_check_interval = 30.0
    memory._client = _FakeRedisClient()
    return memory


def _build_redis_cache_memory() -> RedisCacheMemory:
    memory = RedisCacheMemory.__new__(RedisCacheMemory)
    memory._prefix = "test:cache"
    memory._default_ttl = 300
    memory._fail_open = False
    memory._health_check_interval = 30.0
    memory._available = True
    memory._last_ping = 0.0
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
    now = datetime.now(timezone.utc)
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

    now = datetime.now(timezone.utc)
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


def test_redis_conversation_memory_merge_truncates_to_budget() -> None:
    # P4-7：并发合并后 recent_turns 必须按 _max_turns 预算截断，
    # 否则会把窗口撑到 2 倍预算（本地 _roll 会截断，但合并路径不会）。
    memory = _build_redis_conversation_memory()  # _max_turns=20 → 40 条消息预算
    now = datetime.now(timezone.utc)
    remote_turns = [
        Turn(role="user" if i % 2 == 0 else "assistant", content=f"remote-{i}", timestamp=now + timedelta(seconds=i))
        for i in range(60)
    ]
    local_turns = [
        Turn(role="user" if i % 2 == 0 else "assistant", content=f"local-{i}", timestamp=now + timedelta(seconds=100 + i))
        for i in range(4)
    ]
    remote = ConversationState(recent_turns=remote_turns, rolling_summary="s", pinned_facts=[])
    local = ConversationState(recent_turns=local_turns, rolling_summary="s", pinned_facts=[])

    merged = memory._merge_states(remote, local, max_turns=memory._max_turns)

    # 64 条输入 → 截断到 40 条预算
    assert len(merged.recent_turns) == 40
    # 保留最新端（按时间戳排序，local 时间戳最新 → 排在末尾），最旧 24 条被淘汰
    assert merged.recent_turns[0].content == "remote-24"
    assert [t.content for t in merged.recent_turns[-4:]] == [t.content for t in local_turns]
    assert merged.recent_turns[-1].content == "local-3"


@pytest.mark.asyncio
async def test_conversation_memory_lru_evicts_oldest() -> None:
    # P2-18：进程内会话存储 LRU 封顶，超过上限驱逐最久未使用的会话。
    memory = ConversationMemory(max_turns=20, max_conversations=2)
    await memory.add("c1", "user", "u1")
    await memory.add("c2", "user", "u2")
    # 第 3 个会话加入后超出上限（2），最旧的 c1 应被驱逐。
    await memory.add("c3", "user", "u3")

    assert memory.get_messages("c1") == []
    assert len(memory.get_messages("c2")) > 0
    assert len(memory.get_messages("c3")) > 0


def test_redis_memory_recovers_after_transient_failure() -> None:
    # P2-14：fail-open 单向翻回问题——_available=False 后必须靠周期性 ping 复位。
    memory = _build_redis_work_memory()
    memory._fail_open = True  # 恢复探测仅在 fail-open 模式下进行
    memory._available = False
    memory._last_ping = time.monotonic() - memory._health_check_interval - 5  # 已进入探测窗口

    # 第一次调用：ping 成功 → available 复位（本次调用仍按不可用返回 None）
    assert memory._safe_call("get", key="k", fn=lambda: "v") is None
    assert memory._available is True
    # 第二次调用：走正常路径，返回真实结果
    assert memory._safe_call("get", key="k", fn=lambda: "v") == "v"
    assert memory._available is True


def test_redis_memory_ping_failure_keeps_unavailable() -> None:
    # P2-14：探测 ping 仍失败时应保持不可用，且不阻断调用（fail-open）。
    memory = _build_redis_work_memory()
    memory._fail_open = True
    memory._available = False
    memory._last_ping = time.monotonic() - memory._health_check_interval - 5
    memory._client.ping_fails = True

    assert memory._safe_call("get", key="k", fn=lambda: "v") is None
    assert memory._available is False


def test_redis_cache_poisoned_payload_treated_as_miss() -> None:
    # P2-19：发版后 schema 变更导致 payload 与当前结构不匹配时，
    # 反序列化异常应按 miss 处理并删除毒化条目，而不是该 key 持续 500。
    memory = _build_redis_cache_memory()
    key = "k1"
    memory._client.store[memory._full_key(key)] = "{not-json"

    assert memory.get(key) is None
    assert memory._full_key(key) not in memory._client.store
