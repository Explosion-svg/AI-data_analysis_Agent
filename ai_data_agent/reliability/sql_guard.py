"""
reliability/sql_guard.py — SQL 安全卫士
基于关键词黑名单 + sqlparse AST 双重校验
只允许 SELECT 语句（可配置只读模式）
"""
from __future__ import annotations

import re

import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DDL, DML

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)

# 危险关键词（不区分大小写）
_DANGEROUS_PATTERNS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|TRUNCATE|ALTER|CREATE|REPLACE|MERGE"
    r"|GRANT|REVOKE|EXEC|EXECUTE|CALL|SP_|XP_|LOAD\s+DATA|INTO\s+OUTFILE"
    r"|ATTACH\s+DATABASE|PRAGMA\s+(?!table_info|index_info)|DETACH)\b",
    re.IGNORECASE,
)

# SQL 注入常见模式
_INJECTION_PATTERNS = re.compile(
    r"(;\s*--|;\s*/\*|UNION\s+ALL\s+SELECT|UNION\s+SELECT|1\s*=\s*1|OR\s+1\s*=\s*1)",
    re.IGNORECASE,
)


class SQLGuardError(ValueError):
    """SQL 安全校验失败。"""


def validate_sql(sql: str) -> str:
    """
    校验 SQL 安全性，返回清理后的 SQL；不安全则抛出 SQLGuardError。
    """
    if not sql or not sql.strip():
        raise SQLGuardError("Empty SQL statement.")

    cleaned = sql.strip()

    # 1. 注入模式检测
    if _INJECTION_PATTERNS.search(cleaned):
        metrics.sql_blocked_total.inc()
        logger.warning("sql_guard.injection_pattern", sql=cleaned[:200])
        raise SQLGuardError("SQL injection pattern detected.")

    # 2. 只读模式：禁止危险关键词
    if settings.sql_readonly:
        m = _DANGEROUS_PATTERNS.search(cleaned)
        if m:
            metrics.sql_blocked_total.inc()
            logger.warning(
                "sql_guard.dangerous_keyword",
                keyword=m.group(0),
                sql=cleaned[:200],
            )
            raise SQLGuardError(
                f"Dangerous SQL keyword '{m.group(0)}' is not allowed in readonly mode."
            )

    # 3. sqlparse 级别：确保是 SELECT 语句
    if settings.sql_readonly:
        parsed: list[Statement] = sqlparse.parse(cleaned)
        for stmt in parsed:
            stmt_type = stmt.get_type()
            if stmt_type and stmt_type.upper() != "SELECT":
                metrics.sql_blocked_total.inc()
                logger.warning(
                    "sql_guard.non_select",
                    stmt_type=stmt_type,
                    sql=cleaned[:200],
                )
                raise SQLGuardError(
                    f"Only SELECT statements are allowed, got: {stmt_type}"
                )

    # 4. 多语句检测（防止 ; 分隔的注入）
    statements = [s for s in sqlparse.parse(cleaned) if s.get_type()]
    if len(statements) > 1:
        metrics.sql_blocked_total.inc()
        logger.warning("sql_guard.multiple_statements", sql=cleaned[:200])
        raise SQLGuardError("Multiple SQL statements are not allowed.")

    logger.debug("sql_guard.passed", sql=cleaned[:100])
    return cleaned
