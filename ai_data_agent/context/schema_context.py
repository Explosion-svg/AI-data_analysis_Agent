"""
context/schema_context.py — Schema 上下文构建器

职责：
  动态选取与用户问题最相关的表 schema，格式化后注入 prompt，
  帮助 LLM 准确写出可执行的 SQL。

为什么不把所有表 schema 都注入 prompt？
  - 生产数据仓库通常有几十甚至上百张表，全部注入会耗尽 token 预算
  - 无关表的信息会干扰 LLM 的注意力，降低 SQL 生成质量
  - 按需注入（只注入相关的几张表）既节省 token 又提升精准度

表选择策略（三级降级）：
  1. 语义搜索（首选）：将 query 向量化后搜索 schema collection，找余弦相似度最高的表
  2. 关键词匹配（降级）：当向量库为空或搜索失败时，检查表名是否出现在 query 中
  3. 前 N 张表（兜底）：以上均失败时，直接取前 top_k 张表

单一职责设计：
  这个类只负责 schema 相关的事情：
  - 选表（_select_relevant_tables）
  - 格式化 schema 字符串（build）
  - 从 schema 文本中反解析表名（extract_table_names）

  不负责：
  - 执行 SQL（那是 warehouse 的职责）
  - 存储 schema（那是 vector_store 的职责）
  - 决定哪些信息进入 prompt（那是 prompt_builder 的职责）
"""
from __future__ import annotations

import asyncio
import re

from ai_data_agent.infra import warehouse, vector_store
from ai_data_agent.model_gateway.router import get_router
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# 每张表最多展示多少列（防止列很多的宽表撑爆 token）
_MAX_COLS_PER_TABLE = 20
# prompt 中最多展示多少张表（token 预算控制）
_MAX_TABLES_IN_PROMPT = 8


class SchemaContextBuilder:
    """
    为 LLM 构建精简的 schema 上下文字符串。

    这个类的核心价值是"选表"——从几十张表中选出与当前问题最相关的几张，
    避免把整个仓库的表结构塞入 prompt（既浪费 token 又干扰推理）。

    选表方法优先级（从高到低）：
      1. 语义向量搜索（精确）：需要提前建立 schema 向量索引
      2. 关键词匹配（快速）：检查表名是否出现在 query 文本中
      3. 取前 N 张（兜底）：当所有方法都失败时的最后防线

    典型使用方式（在 agent_loop 中）：
        schema_ctx = await schema_builder.build(query)
        tables = schema_builder.extract_table_names(schema_ctx)
        # schema_ctx 注入 prompt，tables 记录到 work_memory
    """

    async def build(
        self,
        query: str,
        top_k: int = _MAX_TABLES_IN_PROMPT,
    ) -> str:
        """
        为给定的用户问题构建格式化的 schema 字符串。

        执行流程：
          1. 获取数据仓库中的所有表名
          2. 通过 _select_relevant_tables 选择相关表（语义/关键词/兜底）
          3. 获取每张选中表的列信息（name, type, nullable）
          4. 格式化成 Markdown 风格的 schema 字符串

        格式示例：
            ## Available Tables and Columns

            ### Table: `sales`
              - date (DATE)
              - amount (DECIMAL) [NULL]
              - product_id (VARCHAR)
              ... and 5 more columns

        Args:
            query: 用户问题，用于语义匹配选表
            top_k: 最多选取多少张表（默认 8 张，受 _MAX_TABLES_IN_PROMPT 限制）

        Returns:
            格式化的 schema 字符串，空仓库时返回提示文本
        """
        # 1. 获取所有表名
        try:
            all_tables = await warehouse.get_table_names()
        except Exception as e:
            logger.warning("schema_context.get_tables_failed", error=str(e))
            return ""

        if not all_tables:
            return "No tables found in the data warehouse."

        # 2. 语义检索相关表（三级策略）
        selected_tables = await self._select_relevant_tables(query, all_tables, top_k)

        # 3. 获取每张表的列信息，格式化输出
        lines = ["## Available Tables and Columns\n"]
        for table in selected_tables:
            try:
                cols = await warehouse.get_table_schema(table)
                col_lines = [
                    f"  - {c['name']} ({c['type']})"
                    + (" [NULL]" if c.get("nullable") else "")
                    for c in cols[:_MAX_COLS_PER_TABLE]   # 最多展示 20 列
                ]
                lines.append(f"### Table: `{table}`")
                lines.extend(col_lines)
                # 如果列数超限，提示还有更多列
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
            schema_chars=len(schema_str),
        )
        return schema_str

    @staticmethod
    def extract_table_names(schema_context: str) -> list[str]:
        """
        从 schema prompt 文本中反解析出表名列表。

        为什么需要这个方法？
        - agent_loop 在构建 prompt 后，需要记录"本轮涉及了哪些表"到 work_memory
        - 直接把 schema_context 字符串存入 work_memory 太大，需要提取表名摘要
        - 把解析逻辑放在这里（而不是 agent_loop 里），符合单一职责原则

        依赖 build() 输出的稳定格式（"### Table: `table_name`"）：
        - 如果 build() 的格式改变，这里也需要同步修改
        - 通过正则表达式提取反引号内的表名

        Args:
            schema_context: build() 方法的输出字符串

        Returns:
            表名列表，如 ["sales", "products", "orders"]
        """
        return re.findall(r"### Table: `([^`]+)`", schema_context)

    async def _select_relevant_tables(
        self,
        query: str,
        all_tables: list[str],
        top_k: int,
    ) -> list[str]:
        """
        使用三级策略选择与用户问题最相关的表。

        策略优先级（从高到低）：

        1. 向量语义搜索（精确但需要预先建索引）：
           - 将 query 向量化
           - 在 schema collection 中搜索相似的表描述
           - 过滤掉不存在于当前仓库的表名
           - 优点：能处理同义词和语义相似（"营业额" = "sales amount"）

        2. 关键词匹配（快速，不依赖向量库）：
           - 检查每张表名是否出现在 query 字符串中
           - 例如 query="查看 orders 表的数据" 可以直接匹配 "orders" 表
           - 简单但不处理同义词

        3. 前 top_k 张表（兜底，确保总有结果）：
           - 当以上方法都没找到相关表时使用
           - 在小型数据仓库（表数量 ≤ top_k）时也直接走这里

        Args:
            query: 用户问题
            all_tables: 仓库中的所有表名
            top_k: 最多返回多少张表

        Returns:
            选中的表名列表
        """
        # 如果表总数不超过 top_k，直接全返回
        if len(all_tables) <= top_k:
            return all_tables

        # 尝试策略 1：语义向量搜索
        try:
            router = get_router()
            embeddings = await router.embed([query])
            # P2-16：ChromaDB 只有同步接口，包 to_thread 避免阻塞事件循环
            results = await asyncio.to_thread(
                vector_store.search_schema,
                query_embedding=embeddings[0],
                top_k=top_k,
            )
            # 过滤：只保留实际存在于仓库中的表（防止向量库过时）
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

        # 尝试策略 2：关键词匹配（表名是否出现在 query 中）
        query_lower = query.lower()
        keyword_matched = [t for t in all_tables if t.lower() in query_lower]
        if keyword_matched:
            return keyword_matched[:top_k]

        # 策略 3（兜底）：直接返回前 top_k 张表
        return all_tables[:top_k]

    async def index_all_tables(self) -> None:
        """
        将所有表的 schema 信息向量化并存入 ChromaDB vector store。

        这个方法在应用启动时由 assembler._post_startup() 调用一次。
        此后，每当数据仓库 schema 发生变化（新增/修改表），
        应通过管理接口重新调用此方法更新索引。

        索引内容格式：
          "Table {table_name}: {col1}({type1}), {col2}({type2}), ..."

        索引 ID 格式：
          "schema_{table_name}"（方便后续定向更新或删除）

        失败处理：
          - 如果仓库为空或连接失败，直接返回（不报错）
          - 错误被记录为 error 级别日志，但不抛出异常
          - assembler._post_startup() 中包装了 try/except，失败不阻断启动
        """
        try:
            router = get_router()
            tables = await warehouse.get_table_names()
            if not tables:
                return

            docs, ids, metas = [], [], []
            for table in tables:
                cols = await warehouse.get_table_schema(table)
                # 格式化为文本描述：让 embedding 模型能理解表的含义
                col_desc = ", ".join(f"{c['name']}({c['type']})" for c in cols)
                text = f"Table {table}: {col_desc}"
                docs.append(text)
                ids.append(f"schema_{table}")
                metas.append({"table_name": table})

            # 批量向量化（使用 embedding 模型）
            embeddings = await router.embed(docs)
            # 存入 ChromaDB（upsert = 更新已有 + 插入新增）
            # P2-16：ChromaDB 同步写入包 to_thread，避免阻塞事件循环
            await asyncio.to_thread(
                vector_store.upsert_schema,
                ids=ids,
                embeddings=embeddings,
                documents=docs,
                metadatas=metas,
            )
            logger.info("schema_context.indexed", tables=len(tables))
        except Exception as e:
            logger.error("schema_context.index_failed", error=str(e))
