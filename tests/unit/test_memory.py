"""
tests/unit/test_memory.py

内存状态模块单元测试。

主要验证：
- ConversationMemory 的会话隔离、窗口裁剪、清理与摘要
- CacheMemory 的命中、过期、LRU 淘汰
"""

from __future__ import annotations

import time

from ai_data_agent.memory.cache_memory import CacheMemory
from ai_data_agent.memory.conversation_memory import ConversationMemory


def test_conversation_memory_isolation_and_trim() -> None:
    # 准备两个会话，其中 c1 故意写入超过窗口的数据。
    memory = ConversationMemory(max_turns=2)

    memory.add("c1", "user", "u1")
    memory.add("c1", "assistant", "a1")
    memory.add("c1", "user", "u2")
    memory.add("c1", "assistant", "a2")
    memory.add("c1", "user", "u3")
    memory.add("c1", "assistant", "a3")
    memory.add("c2", "user", "other")

    c1_messages = memory.get_messages("c1")
    c2_messages = memory.get_messages("c2")

    # c1 只应保留最近 2 轮，c2 的数据不应被影响。
    assert [m.content for m in c1_messages] == ["u2", "a2", "u3", "a3"]
    assert [m.content for m in c2_messages] == ["other"]


def test_conversation_memory_clear_and_summary() -> None:
    # 验证 summary 统计与 clear 行为。
    memory = ConversationMemory(max_turns=2)
    memory.add("c1", "user", "hello")
    memory.add("c1", "assistant", "world")

    assert memory.summary("c1") == {
        "conversation_id": "c1",
        "turns": 1,
        "messages": 2,
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
