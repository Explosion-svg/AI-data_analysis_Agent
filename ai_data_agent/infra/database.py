"""
infra/database.py — OLTP 数据库连接池（SQLAlchemy 2.0 异步）
支持 SQLite / PostgreSQL / MySQL
接SQL精准查询
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
from sqlalchemy import text

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# 数据库连接池
_engine: AsyncEngine | None = None
# async_sessionmaker是session工厂
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    # 构建数据库连接池
    kwargs: dict = {
        "echo": settings.db_echo,
        "future": True,
    }
    # SQLite 不支持 pool_size 等参数
    if not settings.database_url.startswith("sqlite"):
        kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return create_async_engine(settings.database_url, **kwargs)

# ── 初始化数据库 ────────────────────────────────────────────

async def init_db() -> None:
    """应用启动时调用，初始化引擎和会话工厂。"""
    global _engine, _session_factory
    _engine = _build_engine()
    # 创建session factory
    _session_factory = async_sessionmaker(
        _engine, expire_on_commit=False, autoflush=False
    )
    # 健康检查
    async with _engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    logger.info("database.ready", url=settings.database_url)


async def close_db() -> None:
    """应用关闭时释放连接池。"""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("database.closed")


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call init_db() first.")
    return _engine


# ── 数据库访问接口 ────────────────────────────────────────────

@contextlib.asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    数据库访问最核心的接口
    获取session，ORM / 事物管理
    :return:
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized.")
    async with _session_factory() as session:
        try:
            # 把session提供给调用方
            yield session
            # 自动提交事务
            await session.commit()
        except Exception:
            # 出错自动回滚
            await session.rollback()
            raise


@contextlib.asynccontextmanager
async def get_connection() -> AsyncIterator[AsyncConnection]:
    """
    获取 Connection，执行SQL
    :return:
    """
    async with get_engine().connect() as conn:
        yield conn
