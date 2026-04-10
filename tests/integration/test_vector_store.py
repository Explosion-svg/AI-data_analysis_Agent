"""
tests/integration/test_vector_store.py

向量库层集成测试。

主要验证：
- ChromaDB 临时目录初始化
- docs/schema 两类 collection 的 upsert 与 search
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_data_agent.config.config import settings
from ai_data_agent.infra import vector_store


@pytest.mark.asyncio
async def test_vector_store_upsert_and_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 使用临时目录初始化 Chroma，验证 docs/schema 两类 collection 的基本行为。
    persist_dir = tmp_path / "chroma"
    monkeypatch.setattr(settings, "chroma_persist_dir", str(persist_dir))
    monkeypatch.setattr(settings, "chroma_docs_collection", "test_docs")
    monkeypatch.setattr(settings, "chroma_schema_collection", "test_schema")

    await vector_store.init_vector_store()

    vector_store.upsert_docs(
        ids=["doc_1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["gross merchandise value means GMV"],
        metadatas=[{"source": "kb"}],
    )
    vector_store.upsert_schema(
        ids=["schema_sales"],
        embeddings=[[0.1, 0.2, 0.31]],
        documents=["Table sales: id(INT), amount(INT)"],
        metadatas=[{"table_name": "sales"}],
    )

    docs = vector_store.search_docs(query_embedding=[0.1, 0.2, 0.29], top_k=1)
    schema = vector_store.search_schema(query_embedding=[0.1, 0.2, 0.3], top_k=1)

    assert docs[0]["metadata"]["source"] == "kb"
    assert "GMV" in docs[0]["content"]
    assert schema[0]["metadata"]["table_name"] == "sales"
