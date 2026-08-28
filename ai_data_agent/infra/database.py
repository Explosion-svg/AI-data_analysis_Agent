"""
infra/database.py — OLTP 数据库连接池（SQLAlchemy 2.0 异步）

职责：
  管理 OLTP（事务型）数据库的异步连接池，提供 Session 和 Connection 两种访问接口。

OLTP vs OLAP 的分工：
  - 这里是 OLTP（Online Transaction Processing）：适合行级、事务型操作
  - warehouse.py 是 OLAP（Online Analytical Processing）：适合大批量分析查询
  - 将两者分开的原因：避免长时间分析查询耗尽 OLTP 连接池，影响事务处理

支持的数据库类型：
  - SQLite（开发/测试）：使用 sqlite+aiosqlite 驱动
  - PostgreSQL（生产推荐）：使用 postgresql+asyncpg 驱动
  - MySQL：使用 mysql+aiomysql 驱动

连接池配置：
  - pool_size：常驻连接数（默认 10），即使空闲也保持连接
  - max_overflow：超出 pool_size 后的最大额外连接数（默认 20）
  - pool_pre_ping：每次取出连接时发送 SELECT 1 检测连接是否仍然有效
  - pool_recycle：超过 3600 秒的连接自动重建（防止被数据库服务器端超时断开）

使用方式：
  async with database.get_session() as session:
      result = await session.execute(text("SELECT 1"))

  async with database.get_connection() as conn:
      result = await conn.execute(text("SELECT * FROM users"))
"""
from __future__ import annotations

import contextlib
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import make_url, text

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# ── 模块级单例 ────────────────────────────────────────────────────────────────

# 全局异步数据库引擎（连接池的核心对象）
_engine: AsyncEngine | None = None
# Session 工厂（每次调用产生一个新 Session 对象，共享底层连接池）
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    """
    根据配置构建异步数据库引擎（连接池）。

    SQLite 特殊处理：
    - SQLite 不支持连接池参数（pool_size、max_overflow 等）
    - SQLite 本身是文件型数据库，连接开销极小，不需要连接池
    - 因此检测到 SQLite 时只传基础参数

    其他数据库（PostgreSQL、MySQL 等）：
    - pool_pre_ping=True：每次取用连接时先执行 SELECT 1 检测连通性
      避免使用已断开的"僵尸连接"导致请求失败
    - pool_recycle=3600：1 小时内的连接自动重建
      防止数据库服务器端（如 AWS RDS 默认 8 小时超时）主动断开连接

    Returns:
        配置好的 AsyncEngine 实例
    """
    kwargs: dict = {
        "echo": settings.db_echo,   # True 时打印所有 SQL（开发调试用，生产必须关闭）
        "future": True,             # 使用 SQLAlchemy 2.0 新式接口
    }
    # SQLite 不支持 pool_size 等参数，需要单独处理
    if not settings.database_url.startswith("sqlite"):
        kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return create_async_engine(settings.database_url, **kwargs)


# ── 初始化 / 关闭 ─────────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    初始化数据库引擎和 Session 工厂。

    在应用启动时由 assembler._init_infra() 调用一次。
    初始化后执行一次健康检查（SELECT 1），确保数据库连接正常。

    expire_on_commit=False：
      Session 提交后，已加载的 ORM 对象不会自动过期。
      在 async 环境中，过期的对象在 Session 关闭后无法再 lazy-load，
      会导致 MissingGreenlet 错误。因此关闭自动过期。

    autoflush=False：
      关闭自动 flush（自动把 pending changes 写入数据库）。
      在 async 场景中手动控制 flush 时机，避免意外的数据库写操作。

    Raises:
        Exception: 数据库连接失败（如配置错误、数据库服务不可用）
    """
    global _engine, _session_factory
    _engine = _build_engine()
    # async_sessionmaker：Session 工厂，每次 async with _session_factory() 产生新 Session
    _session_factory = async_sessionmaker(
        _engine, expire_on_commit=False, autoflush=False
    )
    # 健康检查：确认连接可用，如果失败会在启动阶段暴露问题
    async with _engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    # P2-17：日志脱敏，避免数据库密码进入 JSON 日志（ELK/Loki 采集）
    logger.info(
        "database.ready",
        url=make_url(settings.database_url).render_as_string(hide_password=True),
    )


async def close_db() -> None:
    """
    关闭数据库连接池，释放所有连接。

    在应用关闭时由 assembler.shutdown() 调用。
    dispose() 会等待所有活跃连接关闭后再释放池，
    确保正在执行的事务能够正常提交或回滚。
    """
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("database.closed")


def get_engine() -> AsyncEngine:
    """
    获取已初始化的数据库引擎。

    Raises:
        RuntimeError: 如果在 init_db() 之前调用
    """
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call init_db() first.")
    return _engine


# ── 数据访问接口 ──────────────────────────────────────────────────────────────

@contextlib.asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    获取数据库 Session（用于 ORM 操作 / 事务管理）。

    这是 OLTP 数据库的主要访问接口，适合：
    - ORM 模型的增删改查
    - 需要事务管理的操作（自动 commit/rollback）

    事务管理：
    - 正常退出：自动 commit（提交所有 pending 更改）
    - 异常退出：自动 rollback（回滚所有 pending 更改）
    - 两种情况都会关闭 Session（释放连接回池）

    使用方式：
        async with database.get_session() as session:
            user = await session.get(User, user_id)
            user.name = "新名字"
            # 退出 with 块时自动 commit

    Yields:
        AsyncSession 对象，可用于 ORM 操作

    Raises:
        RuntimeError: 如果数据库未初始化
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized.")
    async with _session_factory() as session:
        try:
            # 将 session 提供给调用方使用
            yield session
            # 正常退出：自动提交事务
            await session.commit()
        except Exception:
            # 出现任何异常：回滚事务，保证数据一致性
            await session.rollback()
            raise


@contextlib.asynccontextmanager
async def get_connection() -> AsyncIterator[AsyncConnection]:
    """
    获取原始数据库连接（用于执行原生 SQL）。

    适合：
    - 直接执行原生 SQL 语句（不经过 ORM）
    - 需要更精细控制 SQL 执行的场景
    - 批量导入、Schema 操作等特殊情况

    注意：这里不自动管理事务，调用方需要自行处理 commit/rollback。

    使用方式：
        async with database.get_connection() as conn:
            result = await conn.execute(text("SELECT * FROM users"))
            rows = result.fetchall()

    Yields:
        AsyncConnection 对象，可用于执行原生 SQL
    """
    async with get_engine().connect() as conn:
        yield conn
