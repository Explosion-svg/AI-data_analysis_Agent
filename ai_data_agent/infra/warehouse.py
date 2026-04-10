"""
infra/warehouse.py — 数据仓库连接器（OLAP）
支持 SQLite / PostgreSQL / ClickHouse / BigQuery / Snowflake
统一暴露 execute(sql) -> DataFrame 接口
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)

_engine: AsyncEngine | None = None


# ── 初始化OLAP数据库 ────────────────────────────────────────────
async def init_warehouse() -> None:
    # 数据库连接池
    global _engine
    kwargs: dict[str, Any] = {"future": True}
    if not settings.warehouse_url.startswith("sqlite"):
        kwargs.update(pool_pre_ping=True, pool_recycle=3600)
    _engine = create_async_engine(settings.warehouse_url, **kwargs)
    async with _engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("warehouse.ready", url=settings.warehouse_url)


async def close_warehouse() -> None:
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


def get_warehouse_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Warehouse not initialized. Call init_warehouse() first.")
    return _engine


async def execute(sql: str, params: dict | None = None) -> pd.DataFrame:
    """
    执行 SQL，返回 DataFrame。
    此方法仅执行，不做安全校验（安全校验由 sql_guard 负责）。
    """
    with metrics.sql_latency.time():
        engine = get_warehouse_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), params or {})
            rows = result.fetchall()
            columns = list(result.keys())
        df = pd.DataFrame(rows, columns=columns)
    metrics.sql_queries_total.inc()
    logger.debug("warehouse.execute", sql=sql[:200], rows=len(df))
    return df


async def get_table_names() -> list[str]:
    """返回数据仓库中所有表名。"""
    engine = get_warehouse_engine()
    dialect = engine.dialect.name
    if dialect in ("sqlite",):
        sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    elif dialect in ("postgresql", "clickhouse"):
        sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    else:
        sql = "SHOW TABLES"
    async with engine.connect() as conn:
        result = await conn.execute(text(sql))
        return [row[0] for row in result.fetchall()]


async def get_table_schema(table_name: str) -> list[dict]:
    """
    返回表的列信息列表：[{name, type, nullable}, ...]
    """
    engine = get_warehouse_engine()
    dialect = engine.dialect.name
    if dialect == "sqlite":
        sql = f"PRAGMA table_info({table_name})"
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            rows = result.fetchall()
        return [
            {"name": row[1], "type": str(row[2]), "nullable": not bool(row[3])}
            for row in rows
        ]
    else:
        sql = (
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            f"WHERE table_name = :table ORDER BY ordinal_position"
        )
        async with engine.connect() as conn:
            result = await conn.execute(text(sql), {"table": table_name})
            rows = result.fetchall()
        return [
            {"name": row[0], "type": row[1], "nullable": row[2] == "YES"}
            for row in rows
        ]


async def get_sample_rows(table_name: str, n: int = 3) -> pd.DataFrame:
    # 表名白名单：只允许合法标识符（防止注入）
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
        raise ValueError(f"Invalid table name: {table_name!r}")
    if not isinstance(n, int) or n < 1 or n > 1000:
        raise ValueError("n 必须是 1~1000 之间的整数")
    return await execute(f"SELECT * FROM {table_name} LIMIT {int(n)}")
