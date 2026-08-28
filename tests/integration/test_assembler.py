"""
tests/integration/test_assembler.py

应用装配器集成测试。

主要验证：
- AppContainer.startup 的阶段顺序
- health_report 的结构与关键字段
"""

from __future__ import annotations

import pytest

from ai_data_agent.assembler import AppContainer
from ai_data_agent.memory.cache_memory import CacheMemory
from ai_data_agent.memory.conversation_memory import ConversationMemory
from ai_data_agent.orchestration.executor import Executor
from ai_data_agent.orchestration.planner import Planner
from ai_data_agent.tools.tool_registry import ToolRegistry


@pytest.mark.asyncio
async def test_assembler_startup_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # 这个测试不验证真实外部依赖，而是验证容器启动时各阶段顺序是否正确。
    container = AppContainer()
    calls: list[str] = []

    async def fake_init_observability() -> None:
        calls.append("observability")

    async def fake_init_infra() -> None:
        calls.append("infra")

    async def fake_init_model_gateway() -> None:
        calls.append("model_gateway")

    async def fake_init_tools() -> None:
        calls.append("tools")

    async def fake_init_context() -> None:
        calls.append("context")

    async def fake_init_memory() -> None:
        calls.append("memory")

    async def fake_init_orchestration() -> None:
        calls.append("orchestration")

    async def fake_post_startup() -> None:
        calls.append("post_startup")

    monkeypatch.setattr(container, "_init_observability", fake_init_observability)
    monkeypatch.setattr(container, "_init_infra", fake_init_infra)
    monkeypatch.setattr(container, "_init_model_gateway", fake_init_model_gateway)
    monkeypatch.setattr(container, "_init_tools", fake_init_tools)
    monkeypatch.setattr(container, "_init_context", fake_init_context)
    monkeypatch.setattr(container, "_init_memory", fake_init_memory)
    monkeypatch.setattr(container, "_init_orchestration", fake_init_orchestration)
    monkeypatch.setattr(container, "_post_startup", fake_post_startup)

    await container.startup()

    assert calls == [
        "observability",
        "infra",
        "model_gateway",
        "tools",
        "context",
        "memory",
        "orchestration",
        "post_startup",
    ]


def test_assembler_health_report() -> None:
    # 手工填充最小组件后，health_report 应给出结构化健康状态。
    container = AppContainer()
    container.tool_registry = ToolRegistry()
    container.conversation_memory = ConversationMemory()
    container.cache = CacheMemory()
    container.work_memory = type("FakeWorkMemory", (), {"stats": lambda self: {"active_runs": 0}})()
    container.planner = Planner()
    container.executor = Executor()
    container.agent_loop = object()  # 这里只验证 health_report 结构，不依赖真实 AgentLoop 行为

    report = container.health_report()

    assert report["started"] is False
    assert report["tools"]["ready"] is True
    assert report["memory"]["conversation"] is True
    assert report["orchestration"]["agent_loop"] is True


@pytest.mark.asyncio
async def test_assembler_shutdown_closes_all_resources_in_order() -> None:
    # P2-20：shutdown 应关闭全部 4 类资源（LLM/Redis/Chroma/Warehouse），
    # 且按初始化逆序（LIFO）执行，最后重置启动标志。
    # （P4-7：OLTP database 死重模块已删除，仅剩 4 类资源。）
    container = AppContainer()
    closed: list[str] = []

    async def _close_llm() -> None:
        closed.append("llm")

    async def _close_redis() -> None:
        closed.append("redis")

    async def _close_chroma() -> None:
        closed.append("chroma")

    async def _close_warehouse() -> None:
        closed.append("warehouse")

    container._closers = lambda: [
        ("llm_clients", _close_llm),
        ("redis", _close_redis),
        ("chroma", _close_chroma),
        ("warehouse", _close_warehouse),
    ]
    container._started = True

    await container.shutdown()

    assert closed == ["llm", "redis", "chroma", "warehouse"]
    assert container._started is False


@pytest.mark.asyncio
async def test_assembler_startup_failure_triggers_cleanup() -> None:
    # P2-21：startup 中途失败时应无条件清理已初始化的资源，
    # 避免 K8s CrashLoop 重试时持续累积引擎/连接句柄。
    container = AppContainer()
    cleaned: list[str] = []

    async def _noop() -> None:
        return None

    async def _fail() -> None:
        raise RuntimeError("boom")

    async def _clean_chroma() -> None:
        cleaned.append("chroma")

    container._init_observability = _noop
    container._init_infra = _fail
    container._init_model_gateway = _noop
    container._init_tools = _noop
    container._init_context = _noop
    container._init_memory = _noop
    container._init_orchestration = _noop
    container._post_startup = _noop
    container._closers = lambda: [
        ("chroma", _clean_chroma),
    ]

    with pytest.raises(RuntimeError, match="boom"):
        await container.startup()

    assert cleaned == ["chroma"]
    assert container._started is False
