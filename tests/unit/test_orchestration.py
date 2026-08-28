"""
tests/unit/test_orchestration.py

编排层单元测试。

主要验证：
- Planner 的计划解析与降级
- Executor 的依赖传参与错误记录
"""

from __future__ import annotations

import asyncio
import pytest

from ai_data_agent.config.config import settings
from ai_data_agent.model_gateway.base_model import LLMResponse
from ai_data_agent.orchestration.executor import Executor
from ai_data_agent.orchestration.planner import Plan, PlanStep, Planner
from ai_data_agent.reliability.concurrency import ConcurrencyLimiter, ConcurrencyLimitExceeded
from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.tools.tool_registry import ToolRegistry


class FakeTool(BaseTool):
    # 伪工具：记录调用参数，并返回预设结果。
    def __init__(self, name: str, payload: object) -> None:
        self._name = name
        self.payload = payload
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    async def _run(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(success=True, data=self.payload, text="ok")


@pytest.mark.asyncio
async def test_planner_parses_code_fence_json(monkeypatch: pytest.MonkeyPatch) -> None:
    # 规划器常见场景：模型把 JSON 包在 ```json code fence 里。
    async def generate(*args, **kwargs):
        return LLMResponse(
            content="""```json
{"needs_rag": true, "needs_schema": true, "plan": [{"step": 1, "tool": "sql_query", "description": "run sql", "depends_on": []}], "complexity": "simple"}
```""",
            model="fake",
        )

    monkeypatch.setattr("ai_data_agent.orchestration.planner.get_router", lambda: type("R", (), {"generate": generate})())
    plan = await Planner().plan("问题", ["sql_query"])

    assert plan.needs_rag is True
    assert len(plan.steps) == 1
    assert plan.steps[0].tool == "sql_query"


@pytest.mark.asyncio
async def test_planner_falls_back_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    # 解析失败时应降级为空计划，而不是让整个链路中断。
    async def generate(*args, **kwargs):
        return LLMResponse(content="bad", model="fake")

    monkeypatch.setattr("ai_data_agent.orchestration.planner.get_router", lambda: type("R", (), {"generate": generate})())
    plan = await Planner().plan("问题", ["sql_query"])

    assert plan.is_empty is True
    assert plan.complexity == "simple"


@pytest.mark.asyncio
async def test_executor_passes_dependency_data(monkeypatch: pytest.MonkeyPatch) -> None:
    # Executor 应把上一步 SQL 的结果自动传给 python_analysis 这类依赖步骤。
    sql_tool = FakeTool("sql_query", [{"amount": 1}, {"amount": 2}])
    python_tool = FakeTool("python_analysis", 3)
    registry = ToolRegistry().register(sql_tool).register(python_tool)

    async def fake_generate(*args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt = messages[0].content if messages else ""
        if "python_analysis" in prompt:
            return LLMResponse(content='{"code": "result = 1"}', model="fake")
        return LLMResponse(content='{"sql": "SELECT amount FROM t"}', model="fake")

    monkeypatch.setattr("ai_data_agent.orchestration.executor.get_registry", lambda: registry)
    monkeypatch.setattr(
        "ai_data_agent.orchestration.executor.get_router",
        lambda: type("R", (), {"generate": fake_generate})(),
    )
    steps = [
        PlanStep(step=1, tool="sql_query", goal="select"),
        PlanStep(step=2, tool="python_analysis", goal="result = int(df['amount'].sum())", depends_on=[1]),
    ]

    result = await Executor().execute(Plan(steps=steps))

    assert result[0].done is True
    assert result[1].done is True
    assert python_tool.calls[0]["data"] == [{"amount": 1}, {"amount": 2}]
    assert "code" in python_tool.calls[0]


@pytest.mark.asyncio
async def test_executor_marks_missing_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # 如果计划里引用了未注册工具，Executor 应记录错误并继续。
    async def fake_generate(*args, **kwargs):
        return LLMResponse(content="{}", model="fake")

    monkeypatch.setattr("ai_data_agent.orchestration.executor.get_registry", lambda: ToolRegistry())
    monkeypatch.setattr(
        "ai_data_agent.orchestration.executor.get_router",
        lambda: type("R", (), {"generate": fake_generate})(),
    )
    steps = [PlanStep(step=1, tool="missing", goal="x")]

    result = await Executor().execute(Plan(steps=steps))

    assert result[0].done is True
    assert "not found" in result[0].error


@pytest.mark.asyncio
async def test_executor_parallelism_is_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowTool(BaseTool):
        def __init__(self) -> None:
            self.running = 0
            self.max_running = 0

        @property
        def name(self) -> str:
            return "search_documents"

        @property
        def description(self) -> str:
            return "slow"

        async def _run(self, **kwargs):
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            await asyncio.sleep(0.02)
            self.running -= 1
            return ToolResult(success=True, data=[kwargs.get("query")], text="ok")

    tool = SlowTool()
    registry = ToolRegistry().register(tool)

    async def fake_generate(*args, **kwargs):
        prompt = kwargs["messages"][0].content
        if "goal-1" in prompt:
            return LLMResponse(content='{"query": "q1"}', model="fake")
        if "goal-2" in prompt:
            return LLMResponse(content='{"query": "q2"}', model="fake")
        return LLMResponse(content='{"query": "q3"}', model="fake")

    monkeypatch.setattr("ai_data_agent.orchestration.executor.get_registry", lambda: registry)
    monkeypatch.setattr(
        "ai_data_agent.orchestration.executor.get_router",
        lambda: type("R", (), {"generate": fake_generate})(),
    )

    limiter = ConcurrencyLimiter()
    limiter._semaphores["search_documents"] = asyncio.Semaphore(1)
    monkeypatch.setattr("ai_data_agent.tools.base_tool.get_limiter", lambda: limiter)

    steps = [
        PlanStep(step=1, tool="search_documents", goal="goal-1"),
        PlanStep(step=2, tool="search_documents", goal="goal-2"),
        PlanStep(step=3, tool="search_documents", goal="goal-3"),
    ]

    result = await Executor().execute(Plan(steps=steps))

    assert all(step.done for step in result)
    assert tool.max_running == 1


@pytest.mark.asyncio
async def test_concurrency_limiter_times_out_when_bucket_is_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "concurrency_acquire_timeout_seconds", 0.01)
    limiter = ConcurrencyLimiter()
    limiter._semaphores["llm"] = asyncio.Semaphore(1)

    await limiter._semaphores["llm"].acquire()
    try:
        with pytest.raises(ConcurrencyLimitExceeded):
            async with limiter.limit("llm"):
                pass
    finally:
        limiter._semaphores["llm"].release()
