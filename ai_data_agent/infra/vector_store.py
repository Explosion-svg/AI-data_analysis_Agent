"""
infra/vector_store.py — 向量数据库封装（ChromaDB）

职责：
  封装 ChromaDB 操作，为 RAG 检索和 Schema 语义选表提供向量存储和查询能力。

为什么选择 ChromaDB？
  - 简单易用，零额外部署（可以 embedded 运行，无需独立服务）
  - PersistentClient 模式：数据持久化到磁盘，重启后不丢失
  - 支持余弦相似度（cosine）搜索，适合语义文本检索
  - 适合中小规模场景（几万到几十万向量）

两个 Collection 的设计：
  - docs collection：存储 RAG 知识库文档的 embedding
    用途：搜索与用户问题语义相关的内部文档（业务定义、指标口径等）
  - schema collection：存储数据库表/列描述的 embedding
    用途：根据用户问题找到最相关的表，动态构建 schema context

HNSW 算法说明：
  Hierarchical Navigable Small World（分层可导航小世界图）
  - ChromaDB 默认使用的 ANN（近似最近邻）算法
  - 优点：检索速度快、准确度高、内存友好
  - 适合场景：百万级以下的向量检索
  - 如果需要更大规模检索，应考虑迁移到 Milvus / Weaviate / Pinecone

注意：ChromaDB 目前只有同步接口，因此本模块的函数都是同步的，
在 async 代码中调用时会阻塞事件循环（HNSW + 磁盘 IO）。
调用方必须用 asyncio.to_thread() 包装（见 schema_context.py / rag_tool.py）。
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# 全局 ChromaDB 客户端单例（同步客户端）
_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """
    获取已初始化的 ChromaDB 客户端。

    内部辅助函数，在每个操作函数中调用，确保客户端已初始化。

    Raises:
        RuntimeError: 如果在 init_vector_store() 之前调用
    """
    global _client
    if _client is None:
        raise RuntimeError("VectorStore not initialized. Call init_vector_store() first.")
    return _client


async def init_vector_store() -> None:
    """
    初始化 ChromaDB 持久化客户端，并确保两个 Collection 存在。

    PersistentClient：
    - 数据存储到 chroma_persist_dir 指定的目录（如 ./data/chroma）
    - 重启后数据自动恢复，不需要重新导入文档
    - anonymized_telemetry=False：关闭匿名遥测数据上报（隐私保护）

    Collection 初始化：
    - get_or_create_collection：存在则获取，不存在则创建
    - metadata={"hnsw:space": "cosine"}：使用余弦相似度（适合文本 embedding）
    - 两个 collection 在应用启动时保证存在，后续操作不需要再检查是否存在

    注意：
    - ChromaDB 的异步支持还不完整，此函数虽标注 async 但内部是同步操作
    - 初始化操作只涉及本地文件 IO，速度很快，同步阻塞影响可忽略

    Raises:
        Exception: ChromaDB 目录权限问题或磁盘空间不足
    """
    global _client
    _client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    # 预先确保两个 collection 存在，避免后续每次操作都要检查
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


def close_vector_store() -> None:
    """
    关闭 ChromaDB 客户端，释放持久化目录文件锁与资源（P2-20）。

    幂等：未初始化或已关闭时直接返回。
    ChromaDB 不同版本关闭接口名称不同，这里做防御性探测：
    - clear_system_cache()：停止内部 System（新版本）
    - close()：显式关闭（部分版本提供）
    - 均不存在则仅释放模块单例引用
    """
    global _client
    if _client is None:
        return
    closer = getattr(_client, "clear_system_cache", None) or getattr(_client, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception as e:  # pragma: no cover - 防御性
            logger.warning("vector_store.close_failed", error=str(e))
    _client = None


def get_docs_collection() -> chromadb.Collection:
    """
    获取 RAG 文档 Collection。

    docs collection 存储知识库文档的 embedding，用于 RAG 检索。
    每次调用 get_or_create_collection 确保 collection 存在（幂等）。

    HNSW 余弦相似度适合文本 embedding 的原因：
    - 文本 embedding 向量的"方向"比"长度"更能表达语义
    - 余弦相似度只关注方向（两向量夹角），不受向量 L2 范数影响
    - 相比欧氏距离，对不同长度文本的 embedding 更鲁棒

    Returns:
        ChromaDB Collection 对象
    """
    return _get_client().get_or_create_collection(
        name=settings.chroma_docs_collection,
        metadata={"hnsw:space": "cosine"},
    )


def get_schema_collection() -> chromadb.Collection:
    """
    获取 Schema 语义索引 Collection。

    schema collection 存储表结构描述的 embedding，用于语义选表。
    格式：每条文档是 "Table {name}: {col1}({type1}), {col2}({type2})..."

    与 docs collection 共用相同的余弦相似度配置，
    但数据规模通常更小（只有几十到几百张表）。

    HNSW（Hierarchical Navigable Small World）：
    - 是一种近似最近邻搜索（ANN）算法
    - 在向量空间中构建多层图结构，实现对数复杂度的高效检索
    - 适合小到中等规模（百万以下），超大规模需要专业向量数据库

    Returns:
        ChromaDB Collection 对象
    """
    return _get_client().get_or_create_collection(
        name=settings.chroma_schema_collection,
        metadata={"hnsw:space": "cosine"},
    )


# ── 文档操作（RAG 知识库）────────────────────────────────────────────────────

def upsert_docs(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> None:
    """
    向 docs collection 插入或更新文档向量。

    upsert（update + insert）语义：
    - 如果 id 已存在：更新向量和文档内容
    - 如果 id 不存在：插入新记录
    这样支持增量更新，不需要先删后插。

    Args:
        ids: 文档唯一标识列表（如文件路径、文档 ID）
        embeddings: 对应的向量列表，每个向量维度由 embedding 模型决定
        documents: 文档原文列表（存储原始文本，用于返回时展示）
        metadatas: 元数据列表（如 source、timestamp），可为 None
    """
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
    """
    在 docs collection 中执行语义相似度搜索。

    返回格式：
        [{"content": "...", "metadata": {...}, "score": 0.95}, ...]

    score 计算：
    - ChromaDB 返回的是距离（distance），值越小表示越相似
    - score = 1 - distance（将距离转换为相似度分数）
    - 余弦距离 ∈ [0, 2]，转换后 score ∈ [-1, 1]
    - 实际场景中 score 通常 > 0，建议设置 score_threshold ≥ 0.5

    Args:
        query_embedding: 查询问题的向量表示
        top_k: 返回最相关的文档数量（默认 5）
        where: 元数据过滤条件（如 {"source": "sales_manual.pdf"}）

    Returns:
        按相似度降序排列的文档列表
    """
    col = get_docs_collection()
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = col.query(**kwargs)

    # 将 ChromaDB 结果转换为统一字典格式
    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({"content": doc, "metadata": meta, "score": 1 - dist})
    return docs


# ── Schema 操作（表结构语义索引）─────────────────────────────────────────────

def upsert_schema(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> None:
    """
    向 schema collection 插入或更新表结构向量。

    与 upsert_docs 功能相同，但操作的是 schema collection。
    通常在以下场景调用：
    - 应用启动时：index_all_tables() 批量索引所有表
    - Schema 变更后：重新索引受影响的表

    Args:
        ids: 表标识列表（如 "schema_orders"、"schema_sales"）
        embeddings: 表描述文本的向量列表
        documents: 表描述文本列表（"Table orders: id(INT), user_id(INT)..."）
        metadatas: 元数据列表，通常包含 {"table_name": "orders"}
    """
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
    """
    在 schema collection 中搜索与查询最相关的表。

    这是"智能选表"功能的核心实现：
    - 将用户问题向量化后，在表描述向量空间中搜索
    - 返回语义最相关的 top_k 张表的描述和元数据
    - SchemaContextBuilder 使用返回的 metadata.table_name 获取实际列信息

    与 search_docs 的区别：
    - search_schema 不支持 where 过滤（schema 数量少，无需过滤）
    - top_k 默认更大（10 vs 5），因为表选择需要更多候选才能覆盖边界情况

    Args:
        query_embedding: 查询问题的向量表示
        top_k: 返回最相关的表数量（默认 10）

    Returns:
        按相似度降序排列的表描述列表，每项包含 content、metadata、score
    """
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
