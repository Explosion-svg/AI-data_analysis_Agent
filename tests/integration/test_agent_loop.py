"""
tests/integration/test_agent_loop.py

Agent 主循环集成测试。

主要验证：
- 缓存命中分支
- 模型直接回答分支
- 工具调用后继续总结回答的 ReAct 核心路径
"""

from __future__ import annotations

import pytest

from ai_data_agent.model_gateway.base_model import LLMResponse, Message
from ai_data_agent.orchestration.agent_loop import AgentLoop, AgentResponse
from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.tools.tool_registry import ToolRegistry
from tests.helpers import DummyBreaker, DummyCache, DummyMemory, SequenceRouter


class FakeRAGTool(BaseTool):
    # 提供固定 RAG 文档，避免真实向量库依赖。
    @property
    def name(self) -> str:
        return "search_documents"

    @property
    def description(self) -> str:
        return "rag"

    async def _run(self, **kwargs):
        return ToolResult(success=True, data=[{"content": "doc", "score": 0.9, "metadata": {"source": "kb"}}], text="doc")


class FakeSQLTool(BaseTool):
    # 提供固定 SQL 结果，并记录调用参数。
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "sql_query"

    @property
    def description(self) -> str:
        return "sql"

    async def _run(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(success=True, data=[{"value": 1}], text="one row")


@pytest.mark.asyncio
async def test_agent_loop_returns_cached_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # 缓存命中时，AgentLoop 不应继续执行 ReAct 主循环。
    cached = AgentResponse(answer="cached", conversation_id="c1", success=True)
    cache = DummyCache(value=cached)
    agent = AgentLoop()

    async def should_not_run(*args, **kwargs):
        raise AssertionError("_react_loop should not be called on cache hit")

    monkeypatch.setattr("ai_data_agent.orchestration.agent_loop.get_cache", lambda: cache)
    monkeypatch.setattr(agent, "_react_loop", should_not_run)

    resp = await agent.run("query", "c1", use_cache=True)

    assert resp.answer == "cached"


@pytest.mark.asyncio
async def test_agent_loop_direct_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    # 模型如果直接输出最终答案，AgentLoop 应结束循环并写入 memory。
    router = SequenceRouter([LLMResponse(content="final answer", model="fake", tool_calls=None)])
    memory = DummyMemory()
    registry = ToolRegistry()
    agent = AgentLoop()

    async def rewrite(query: str):
        return {"rewritten": query, "alternatives": [], "keywords": [], "all_queries": [query]}

    async def build_schema(query: str):
        return "schema"

    monkeypatch.setattr(agent._query_rewriter, "rewrite", rewrite)
    monkeypatch.setattr(agent._schema_builder, "build", build_schema)
    monkeypatch.setattr(agent._prompt_builder, "build", lambda **kwargs: [Message(role="user", content=kwargs["query"])])
    monkeypatch.setattr("ai_data_agent.orchestration.agent_loop.get_memory", lambda: memory)
    monkeypatch.setattr("ai_data_agent.orchestration.agent_loop.get_registry", lambda: registry)
    monkeypatch.setattr("ai_data_agent.orchestration.agent_loop.get_router", lambda: router)
    monkeypatch.setattr("ai_data_agent.orchestration.agent_loop.get_breaker", lambda name: DummyBreaker())

    resp = await agent.run("query", "c1", use_cache=False)

    assert resp.success is True
    assert resp.answer == "final answer"
    assert memory.added == [("c1", "user", "query"), ("c1", "assistant", "final answer")]


@pytest.mark.asyncio
async def test_agent_loop_tool_call_then_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    # 这是 ReAct 的核心路径：模型先请求工具，再根据工具观察给出最终回答。
    sql_tool = FakeSQLTool()
    registry = ToolRegistry().register(FakeRAGTool()).register(sql_tool)
    router = SequenceRouter(
        [
            LLMResponse(
                content="need data",
                model="fake",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "sql_query", "arguments": '{"sql":"SELECT 1"}'},
                    }
                ],
            ),
            LLMResponse(content="analysis done", model="fake", tool_calls=None),
        ]
    )
    memory = DummyMemory()
    agent = AgentLoop()

    async def rewrite(query: str):
        return {"rewritten": query, "alternatives": [], "keywords": [], "all_queries": [query]}

    async def build_schema(query: str):
        return "schema"

    monkeypatch.setattr(agent._query_rewriter, "rewrite", rewrite)
    monkeypatch.setattr(agent._schema_builder, "build", build_schema)
    monkeypatch.setattr(agent._prompt_builder, "build", lambda **kwargs: [Message(role="user", content=kwargs["query"])])
    monkeypatch.setattr("ai_data_agent.orchestration.agent_loop.get_memory", lambda: memory)
    monkeypatch.setattr("ai_data_agent.orchestration.agent_loop.get_registry", lambda: registry)
    monkeypatch.setattr("ai_data_agent.orchestration.agent_loop.get_router", lambda: router)
    monkeypatch.setattr("ai_data_agent.orchestration.agent_loop.get_breaker", lambda name: DummyBreaker())

    resp = await agent.run("query", "c1", use_cache=False)

    assert resp.success is True
    assert resp.answer == "analysis done"
    assert resp.tool_calls == [{"tool": "sql_query", "args": {"sql": "SELECT 1"}, "success": True}]
    assert resp.data == [{"value": 1}]
    assert sql_tool.calls == [{"sql": "SELECT 1"}]
