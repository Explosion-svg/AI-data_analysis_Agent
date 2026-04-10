"""
infra/vector_store.py — 向量数据库封装（ChromaDB）
管理两个 collection：docs（RAG文档） + schema（数据库表结构语义）
schema用途：存表的语义信息，LLM根据NL，再embedding，找到对应的表，最后转SQL
接RAG向量检索

"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    # 获取数据库连接
    global _client
    if _client is None:
        raise RuntimeError("VectorStore not initialized. Call init_vector_store() first.")
    return _client


async def init_vector_store() -> None:
    # 初始化数据库操作接口
    global _client
    _client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    # 确保两个 collection 存在
    _client.get_or_create_collection(
        name=settings.chroma_docs_collection,
        metadata={"hnsw:space": "cosine"},
    )
    _client.get_or_create_collection(
        name=settings.chroma_schema_collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "vector_store.ready",
        persist_dir=settings.chroma_persist_dir,
        docs=settings.chroma_docs_collection,
        schema=settings.chroma_schema_collection,
    )


def get_docs_collection() -> chromadb.Collection:
    """
    HNSW：Hierarchical Navigable Small World，只适合小规模检索
    如果规模很大，需要RAG项目的复杂检索，hybrid+rerank等
    :return:
    """
    return _get_client().get_or_create_collection(
        name=settings.chroma_docs_collection,
        metadata={"hnsw:space": "cosine"},
    )


def get_schema_collection() -> chromadb.Collection:
    """
    HNSW：Hierarchical Navigable Small World
    是最近邻ANN算法
    适合小规模检索
    :return:
    """
    return _get_client().get_or_create_collection(
        name=settings.chroma_schema_collection,
        metadata={"hnsw:space": "cosine"},
    )


# ── 文档操作 ─────────────────────────────────────────────────────────────────

def upsert_docs(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> None:
    col = get_docs_collection()
    col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas or [{} for _ in ids],
    )
    logger.debug("vector_store.upsert_docs", count=len(ids))


def search_docs(
    query_embedding: list[float],
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    col = get_docs_collection()
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = col.query(**kwargs)
    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({"content": doc, "metadata": meta, "score": 1 - dist})
    return docs


# ── Schema 操作 ───────────────────────────────────────────────────────────────

def upsert_schema(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> None:
    col = get_schema_collection()
    col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas or [{} for _ in ids],
    )
    logger.debug("vector_store.upsert_schema", count=len(ids))


def search_schema(
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    col = get_schema_collection()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append({"content": doc, "metadata": meta, "score": 1 - dist})
    return items
