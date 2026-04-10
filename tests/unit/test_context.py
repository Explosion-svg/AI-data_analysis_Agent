"""
tests/unit/test_context.py

上下文构建层单元测试。

主要验证：
- PromptBuilder 的消息拼装顺序
- QueryRewriter 的解析与降级逻辑
- SchemaContextBuilder 的选表与降级逻辑
"""

from __future__ import annotations

import pytest

from ai_data_agent.context.prompt_builder import PromptBuilder
from ai_data_agent.context.query_rewriter import QueryRewriter
from ai_data_agent.context.schema_context import SchemaContextBuilder
from ai_data_agent.model_gateway.base_model import LLMResponse, Message


def test_prompt_builder_orders_messages() -> None:
    # PromptBuilder 最重要的是消息顺序稳定，否则会直接影响模型行为。
    builder = PromptBuilder()
    history = [Message(role="assistant", content="old answer")]
    docs = [{"content": "doc body", "score": 0.9, "metadata": {"source": "kb"}}]

    messages = builder.build(
        "current question",
        rag_docs=docs,
        schema_context="table users(id int)",
        history=history,
    )

    assert messages[0].role == "system"
    assert messages[1].content == "old answer"
    assert "Relevant Knowledge Base Documents" in messages[2].content
    assert "Database Schema" in messages[3].content
    assert messages[4].role == "user"
    assert messages[4].content == "current question"


@pytest.mark.asyncio
async def test_query_rewriter_falls_back_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    async def bad_generate(*args, **kwargs):
        return LLMResponse(content="not json", model="fake")

    # 模型返回非法 JSON 时，不应抛错，而应退化回原始 query。
    monkeypatch.setattr("ai_data_agent.context.query_rewriter.get_router", lambda: type("R", (), {"generate": bad_generate})())
    result = await QueryRewriter().rewrite("销售趋势")

    assert result["rewritten"] == "销售趋势"
    assert result["all_queries"] == ["销售趋势"]


@pytest.mark.asyncio
async def test_query_rewriter_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    # 正常情况下应提取 rewritten / alternatives / keywords，并合并 all_queries。
    async def generate(*args, **kwargs):
        return LLMResponse(
            content='{"rewritten":"销售趋势分析","alternatives":["月度销售趋势"],"keywords":["销售","趋势"]}',
            model="fake",
        )

    monkeypatch.setattr("ai_data_agent.context.query_rewriter.get_router", lambda: type("R", (), {"generate": generate})())
    result = await QueryRewriter().rewrite("销售趋势")

    assert result["rewritten"] == "销售趋势分析"
    assert result["alternatives"] == ["月度销售趋势"]
    assert result["all_queries"] == ["销售趋势", "销售趋势分析", "月度销售趋势"]


@pytest.mark.asyncio
async def test_schema_context_prefers_semantic_search(monkeypatch: pytest.MonkeyPatch) -> None:
    # 当向量检索成功时，SchemaContextBuilder 应优先使用语义命中的表。
    async def get_table_names():
        return ["orders", "users", "products"]

    async def get_table_schema(table_name: str):
        return [{"name": "id", "type": "INTEGER", "nullable": False}]

    async def embed(texts):
        return [[0.1, 0.2]]

    monkeypatch.setattr("ai_data_agent.context.schema_context.warehouse.get_table_names", get_table_names)
    monkeypatch.setattr("ai_data_agent.context.schema_context.warehouse.get_table_schema", get_table_schema)
    monkeypatch.setattr("ai_data_agent.context.schema_context.get_router", lambda: type("R", (), {"embed": embed})())
    monkeypatch.setattr(
        "ai_data_agent.context.schema_context.vector_store.search_schema",
        lambda query_embedding, top_k: [{"metadata": {"table_name": "orders"}, "score": 0.9}],
    )

    result = await SchemaContextBuilder().build("订单分析", top_k=2)

    assert "Table: `orders`" in result
    assert "users" not in result


@pytest.mark.asyncio
async def test_schema_context_falls_back_to_keyword_match(monkeypatch: pytest.MonkeyPatch) -> None:
    # 当 embedding/向量检索失败时，应退化为关键词匹配，而不是直接返回空 schema。
    async def get_table_names():
        return ["orders", "users", "products"]

    async def get_table_schema(table_name: str):
        return [{"name": "id", "type": "INTEGER", "nullable": False}]

    async def embed(texts):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr("ai_data_agent.context.schema_context.warehouse.get_table_names", get_table_names)
    monkeypatch.setattr("ai_data_agent.context.schema_context.warehouse.get_table_schema", get_table_schema)
    monkeypatch.setattr("ai_data_agent.context.schema_context.get_router", lambda: type("R", (), {"embed": embed})())

    result = await SchemaContextBuilder().build("orders 表有哪些列")

    assert "Table: `orders`" in result
