"""
context/schema_context.py — Schema 上下文构建器
动态选取与用户问题最相关的表，避免把全库 schema 塞入 prompt
支持语义搜索（向量库）和关键词匹配两种策略
"""
from __future__ import annotations

from ai_data_agent.infra import warehouse, vector_store
from ai_data_agent.model_gateway.router import get_router
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# 每表最多展示多少列
_MAX_COLS_PER_TABLE = 20
# prompt 中最多展示多少张表
_MAX_TABLES_IN_PROMPT = 8


class SchemaContextBuilder:
    """
    为 LLM 构建精简的 schema 上下文字符串。
    策略：
      1. 语义搜索 schema 向量库，找最相关的表
      2. 若向量库为空，退化为全量 schema（截断到 _MAX_TABLES_IN_PROMPT）
    """

    async def build(
        self,
        query: str,
        top_k: int = _MAX_TABLES_IN_PROMPT,
    ) -> str:
        """返回格式化的 schema 字符串。"""
        # 1. 获取所有表名
        try:
            all_tables = await warehouse.get_table_names()
        except Exception as e:
            logger.warning("schema_context.get_tables_failed", error=str(e))
            return ""

        if not all_tables:
            return "No tables found in the data warehouse."

        # 2. 语义检索相关表
        selected_tables = await self._select_relevant_tables(query, all_tables, top_k)

        # 3. 获取每张表的列信息
        lines = ["## Available Tables and Columns\n"]
        for table in selected_tables:
            try:
                cols = await warehouse.get_table_schema(table)
                col_lines = [
                    f"  - {c['name']} ({c['type']})"
                    + (" [NULL]" if c.get("nullable") else "")
                    for c in cols[:_MAX_COLS_PER_TABLE]
                ]
                lines.append(f"### Table: `{table}`")
                lines.extend(col_lines)
                if len(cols) > _MAX_COLS_PER_TABLE:
                    lines.append(f"  ... and {len(cols) - _MAX_COLS_PER_TABLE} more columns")
                lines.append("")
            except Exception as e:
                logger.warning("schema_context.table_failed", table=table, error=str(e))

        schema_str = "\n".join(lines)
        logger.debug(
            "schema_context.built",
            tables_selected=len(selected_tables),
            total_tables=len(all_tables),
        )
        return schema_str

    async def _select_relevant_tables(
        self,
        query: str,
        all_tables: list[str],
        top_k: int,
    ) -> list[str]:
        """使用向量相似度选择最相关的表。"""
        if len(all_tables) <= top_k:
            return all_tables

        # 尝试语义搜索
        try:
            router = get_router()
            embeddings = await router.embed([query])
            results = vector_store.search_schema(
                query_embedding=embeddings[0],
                top_k=top_k,
            )
            selected = [
                r["metadata"].get("table_name", "")
                for r in results
                if r["metadata"].get("table_name") in all_tables
            ]
            if selected:
                logger.debug("schema_context.semantic_selected", tables=selected)
                return selected
        except Exception as e:
            logger.debug("schema_context.semantic_failed", error=str(e))

        # 降级：关键词匹配
        query_lower = query.lower()
        keyword_matched = [t for t in all_tables if t.lower() in query_lower]
        if keyword_matched:
            return keyword_matched[:top_k]

        # 最终降级：返回前 top_k 张表
        return all_tables[:top_k]

    async def index_all_tables(self) -> None:
        """
        将所有表的 schema 信息向量化并存入 vector_store，
        供后续语义检索使用。应在数据仓库 schema 变化后调用。
        """
        try:
            router = get_router()
            tables = await warehouse.get_table_names()
            if not tables:
                return
            docs, ids, metas = [], [], []
            for table in tables:
                cols = await warehouse.get_table_schema(table)
                col_desc = ", ".join(f"{c['name']}({c['type']})" for c in cols)
                text = f"Table {table}: {col_desc}"
                docs.append(text)
                ids.append(f"schema_{table}")
                metas.append({"table_name": table})

            embeddings = await router.embed(docs)
            vector_store.upsert_schema(
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metas,
            )
            logger.info("schema_context.indexed", tables=len(tables))
        except Exception as e:
            logger.error("schema_context.index_failed", error=str(e))
