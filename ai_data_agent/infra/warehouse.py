"""
infra/warehouse.py — 数据仓库连接器（OLAP）

职责：
  管理 OLAP（分析型）数据库连接，提供统一的 execute(sql) → DataFrame 接口，
  以及 Schema 自省接口（列出表名、获取列信息、采样数据）。

数据库设计说明：
  - 本模块（OLAP）是系统中唯一的数据访问层，专门处理大批量分析查询
    （GROUP BY、聚合、宽表扫描），SQL 工具的 SELECT 全部走这里
  - 早期曾规划 OLTP 事务库（infra/database.py），但实际零调用者，
    已在 P4-7 删除，避免"把持 30 连接池、失败阻断启动"的死重

支持的数据库类型：
  - SQLite（开发/测试，无需额外服务）
  - PostgreSQL（通用，适合中等规模）
  - ClickHouse（高性能列式数据库，适合大规模分析）
  - BigQuery / Snowflake（通过对应驱动支持）

统一接口设计原则：
  execute(sql) 统一返回 pandas DataFrame，
  上层（tools/sql_tool.py）只需关心"如何处理 DataFrame"，
  不需要关心底层是什么数据库。

Schema 自省接口：
  get_table_names() / get_table_schema() / get_sample_rows()
  这些接口用于动态发现数据结构，支持 SchemaContextBuilder 的表选择和向量索引。
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sqlalchemy import event, make_url, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)

# 模块级异步引擎单例
_engine: AsyncEngine | None = None


# ── 初始化 / 关闭 ─────────────────────────────────────────────────────────────

async def init_warehouse() -> None:
    """
    初始化数据仓库异步引擎并执行健康检查。

    仓库连接只需要原始的 Connection 接口来执行 SQL（无需 Session/ORM）。

    SQLite 不支持连接池参数，因此根据 URL 前缀分别处理。
    pool_pre_ping=True 和 pool_recycle=3600 用于防止"僵尸连接"和连接老化。

    Raises:
        Exception: 数据库连接失败（配置错误或服务不可用）
    """
    global _engine
    kwargs: dict[str, Any] = {"future": True}
    if settings.warehouse_url.startswith("sqlite"):
        # P3-11：SQLite 默认 journal 模式为 rollback-journal，多连接并发写会报
        # "database is locked"。busy_timeout 让连接在锁释放前等待（毫秒），
        # 配合 WAL 模式（init 时设置一次，连接级），显著降低锁冲突。
        kwargs.update(connect_args={"timeout": 30})
    else:
        kwargs.update(pool_pre_ping=True, pool_recycle=3600)
    _engine = create_async_engine(settings.warehouse_url, **kwargs)
    # P2-24：引擎层强制只读（防御纵深，独立于 sql_guard 文本校验）。
    # 连接级强制只读，任何新建连接都会被应用，配合连接池复用。
    _enforce_readonly_on_connect(_engine)
    # 健康检查，确保连接可用
    async with _engine.connect() as conn:
        if settings.warehouse_url.startswith("sqlite"):
            await conn.execute(text("PRAGMA journal_mode = WAL"))
        await conn.execute(text("SELECT 1"))
    # P2-17：日志脱敏，避免数据库密码进入 JSON 日志（ELK/Loki 采集）
    logger.info(
        "warehouse.ready",
        url=make_url(settings.warehouse_url).render_as_string(hide_password=True),
    )


def _enforce_readonly_on_connect(engine: AsyncEngine) -> None:
    """
    为每个新建连接设置只读会话，让仓库引擎从连接层面拒绝写操作。

    SQLite：PRAGMA query_only = ON（等价于只读模式，写语句直接报错）
    PostgreSQL：default_transaction_read_only = on（事务级只读）

    用 SQLAlchemy 事件而不是在 init 时执行一次的原因：
    - PRAGMA/SET 是连接级状态，连接池复用/新建连接都需要生效
    - 事件钩子在每次建立连接时触发，覆盖所有池化连接
    """
    url = make_url(settings.warehouse_url)

    @event.listens_for(engine.sync_engine, "connect")
    def _apply_readonly(dbapi_connection, connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            if url.get_backend_name() == "sqlite":
                # P3-11：busy_timeout 连接级等待锁释放，避免并发 "database is locked"
                cursor.execute("PRAGMA busy_timeout = 30000")
                cursor.execute("PRAGMA query_only = ON")
            elif url.get_backend_name() == "postgresql":
                cursor.execute("SET default_transaction_read_only = on")
        finally:
            cursor.close()


async def close_warehouse() -> None:
    """
    关闭数据仓库连接池，释放所有连接。

    在应用关闭时由 assembler.shutdown() 调用。
    """
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


def get_warehouse_engine() -> AsyncEngine:
    """
    获取已初始化的仓库引擎。

    Raises:
        RuntimeError: 如果在 init_warehouse() 之前调用
    """
    if _engine is None:
        raise RuntimeError("Warehouse not initialized. Call init_warehouse() first.")
    return _engine


# ── 核心查询接口 ──────────────────────────────────────────────────────────────

async def execute(sql: str, params: dict | None = None) -> pd.DataFrame:
    """
    执行 SQL 语句，返回 pandas DataFrame。

    这是仓库层的核心接口，统一了不同数据库返回格式的差异：
    - 无论底层是 PostgreSQL、ClickHouse 还是 SQLite
    - 上层调用方总是得到一个标准的 pandas DataFrame

    注意：
    - 此方法只负责执行，不做 SQL 安全校验
    - 安全校验由 tools/sql_tool.py 中的 validate_sql() 负责
    - 这样保持了单一职责：仓库层只关心"如何执行"，安全层只关心"是否安全"

    指标记录：
    - metrics.sql_latency.time()：记录 SQL 执行延迟（Summary）
    - metrics.sql_queries_total.inc()：记录总查询次数（Counter）
    - metrics.sql_errors_total.inc()：执行失败时计数（P3-11，异常路径不再跳过）

    Args:
        sql: 要执行的 SQL 语句（已通过安全校验）
        params: SQL 参数字典（用于参数化查询，防止注入）

    Returns:
        查询结果 DataFrame，空结果时返回空 DataFrame（列名保留）
    """
    with metrics.sql_latency.time():
        try:
            engine = get_warehouse_engine()
            async with engine.connect() as conn:
                result = await conn.execute(text(sql), params or {})
                rows = result.fetchall()
                columns = list(result.keys())
            df = pd.DataFrame(rows, columns=columns)
        except Exception:
            # P3-11：失败路径不再跳过指标计数，SQL 错误率 = errors / (queries + errors)
            metrics.sql_errors_total.inc()
            raise
    metrics.sql_queries_total.inc()
    logger.debug("warehouse.execute", sql=sql[:200], rows=len(df))
    return df


# ── Schema 自省接口 ───────────────────────────────────────────────────────────

async def get_table_names() -> list[str]:
    """
    获取数据仓库中所有表名。

    使用各数据库的系统表或元数据接口查询表列表：
    - SQLite：sqlite_master 系统表
    - PostgreSQL / ClickHouse：information_schema.tables（public schema）
    - 其他：SHOW TABLES（MySQL 兼容语法）

    Returns:
        表名字符串列表，按数据库默认顺序排列
    """
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
    获取指定表的列信息。

    列信息用于：
    1. 在 prompt 中向 LLM 展示表结构（schema_context）
    2. 向量化存储为 schema embedding（用于语义选表）
    3. 验证 SQL 中引用的列名是否存在

    返回格式：
        [{"name": "col_name", "type": "VARCHAR", "nullable": True}, ...]

    各数据库实现：
    - SQLite：PRAGMA table_info() 返回 (cid, name, type, notnull, dflt, pk)
    - PostgreSQL 及其他：information_schema.columns 标准视图

    Args:
        table_name: 表名（不含 schema 前缀）

    Returns:
        列信息字典列表，每项包含 name、type、nullable 三个字段
    """
    engine = get_warehouse_engine()
    dialect = engine.dialect.name
    if dialect == "sqlite":
        sql = f"PRAGMA table_info({table_name})"
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            rows = result.fetchall()
        # PRAGMA table_info 返回格式：(cid, name, type, notnull, dflt_value, pk)
        return [
            {"name": row[1], "type": str(row[2]), "nullable": not bool(row[3])}
            for row in rows
        ]
    else:
        # 使用标准 SQL 信息模式视图（PostgreSQL、MySQL 均兼容）
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
    """
    从指定表中采样前 N 行数据，返回 DataFrame。

    主要用途：
    - schema_tool 的 sample_rows 动作（让 LLM 了解数据实际格式）
    - 调试时快速查看表数据

    安全措施：
    - 表名白名单验证：只允许合法 SQL 标识符（字母、数字、下划线）
      防止通过表名注入恶意 SQL（如 "users; DROP TABLE users"）
    - n 值范围限制：1 ≤ n ≤ 1000，防止误传超大值导致内存溢出

    注意：表名直接拼入 SQL 字符串（f-string），因此必须先验证格式。
    这是一种有意设计：LIMIT 子句不支持参数化，表名也不支持，
    所以必须在应用层做格式验证。

    Args:
        table_name: 要采样的表名（只允许合法标识符）
        n: 采样行数（默认 3，最大 1000）

    Returns:
        包含前 n 行数据的 DataFrame

    Raises:
        ValueError: 表名格式非法或 n 超出范围
    """
    # 白名单校验：只允许合法 SQL 标识符，防止 SQL 注入
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
        raise ValueError(f"Invalid table name: {table_name!r}")
    # 参数范围校验：防止超大采样导致内存溢出
    if not isinstance(n, int) or n < 1 or n > 1000:
        raise ValueError("n 必须是 1~1000 之间的整数")
    return await execute(f"SELECT * FROM {table_name} LIMIT {int(n)}")
