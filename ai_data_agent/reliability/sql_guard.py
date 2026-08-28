"""
reliability/sql_guard.py — SQL 安全卫士（多层防御）

职责：
  在执行任何 SQL 之前进行安全校验，防止：
  1. SQL 注入攻击（Injection Attack）
  2. 危险 DDL/DML 操作（DROP、DELETE、UPDATE 等）
  3. 多语句注入（分号分隔的多条 SQL）
  4. 越权访问（访问不在白名单内的表）

多层防御架构（Defense in Depth）：
  Layer 1 → 注入模式正则检测（最快，拦截经典注入手法）
  Layer 2 → 危险关键词正则检测（拦截 DDL/写操作，只读模式下启用）
  Layer 3 → sqlparse AST 语法树校验（确认语句类型为 SELECT）
  Layer 4 → 多语句检测（防止分号分隔的复合注入）
  Layer 5 → 表名白名单校验（enforce_allowed_tables，应用层行级访问控制）

为什么需要多层而不是只用 sqlparse？
  - sqlparse 只能解析结构良好的 SQL，对畸形注入字符串可能解析为 None type
  - 正则是第一道门，速度快（比 sqlparse 快 10x）且对所有字符串有效
  - sqlparse AST 则能识别 sqlparse 无法用正则捕捉的结构性问题
  - 白名单是最后一道门，即使前几层出 bug，也能限制爆炸半径

sql_readonly 配置：
  - settings.sql_readonly=True（默认）：只允许 SELECT
  - settings.sql_readonly=False：允许所有 SQL（仅在受信任环境下使用）

审计指标：
  - metrics.sql_blocked_total：所有被拦截的 SQL 计数
  - 每次拦截都记录 WARNING 日志，包含拦截原因和 SQL 前 200 字符

PRAGMA 例外（SQLite）：
  _DANGEROUS_PATTERNS 对 PRAGMA 做了精细白名单：
  允许 PRAGMA table_info 和 PRAGMA index_info（查询 schema 信息），
  禁止其他 PRAGMA（如 PRAGMA journal_mode=DELETE 可能破坏数据库完整性）
"""
from __future__ import annotations

import re
from typing import Any

import sqlparse
from sqlparse import sql as _sql
from sqlparse import tokens as _tokens
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DDL, DML

from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics

logger = get_logger(__name__)

# ── 安全检测正则表达式 ────────────────────────────────────────────────────────

# 危险关键词黑名单（不区分大小写）
# 包含：DDL（DROP/ALTER/CREATE）、写操作（INSERT/UPDATE/DELETE/TRUNCATE）、
# 存储过程（EXEC/EXECUTE/CALL/SP_/XP_）、数据导出（LOAD DATA/INTO OUTFILE）、
# SQLite 特有危险操作（ATTACH/DETACH DATABASE）
# PRAGMA 例外：table_info 和 index_info 是只读 schema 查询，允许通过
_DANGEROUS_PATTERNS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|TRUNCATE|ALTER|CREATE|REPLACE|MERGE"
    r"|GRANT|REVOKE|EXEC|EXECUTE|CALL|SP_|XP_|LOAD\s+DATA|INTO\s+OUTFILE"
    r"|ATTACH\s+DATABASE|PRAGMA\s+(?!table_info|index_info)|DETACH)\b",
    re.IGNORECASE,
)

# SQL 注入经典模式检测
# - `;--` / `;/*`：注释截断（终止原始查询后执行注入内容）
# - `UNION ALL SELECT` / `UNION SELECT`：联合查询注入（获取其他表数据）
# - `1=1` / `OR 1=1`：永真条件（绕过 WHERE 过滤）
_INJECTION_PATTERNS = re.compile(
    r"(;\s*--|;\s*/\*|UNION\s+ALL\s+SELECT|UNION\s+SELECT|1\s*=\s*1|OR\s+1\s*=\s*1)",
    re.IGNORECASE,
)

# ── 异常类 ────────────────────────────────────────────────────────────────────

class SQLGuardError(ValueError):
    """
    SQL 安全校验失败时抛出的异常。

    继承 ValueError 的原因：
    - SQL 的内容不合法（值层面的错误），而不是程序逻辑错误
    - SQLTool._run() 捕获此异常，返回 ToolResult(success=False, error=...)
    - 上层调用者不需要特殊处理，只需知道 SQL 被拦截

    消息格式："{拦截原因}，附带原始 SQL 片段"，方便日志排查
    """


# ── 核心校验函数 ──────────────────────────────────────────────────────────────

def validate_sql(sql: str) -> str:
    """
    对 SQL 字符串进行四层安全校验，通过后返回清理后的 SQL。

    校验失败时抛出 SQLGuardError，同时：
    - 递增 metrics.sql_blocked_total 计数
    - 记录 WARNING 日志（含拦截原因和 SQL 前缀）

    校验顺序（从快到慢，从粗到细）：
    1. 空 SQL 检查：防止下游工具处理空字符串时崩溃
    2. 注入模式正则：O(n) 字符串扫描，速度最快，优先执行
    3. 危险关键词正则：仅 sql_readonly=True 时启用
    4. sqlparse AST 类型校验：确认语句 type 为 SELECT
    5. 多语句检测：确认只有一条语句（防 `;` 分隔注入）

    注意事项：
    - 只做语法/模式层面的安全校验，不执行 SQL
    - 不校验表/列名是否存在（那是数据库层面的工作）
    - 表名白名单校验由 enforce_allowed_tables() 单独处理

    Args:
        sql: 原始 SQL 字符串（可能来自 LLM 生成）

    Returns:
        strip() 后的 SQL 字符串（清理了首尾空白）

    Raises:
        SQLGuardError: 任意一层校验失败时抛出，消息说明具体失败原因
    """
    if not sql or not sql.strip():
        raise SQLGuardError("Empty SQL statement.")

    cleaned = sql.strip()

    # 1. 注入模式检测（正则快速扫描）
    if _INJECTION_PATTERNS.search(cleaned):
        metrics.sql_blocked_total.inc()
        logger.warning("sql_guard.injection_pattern", sql=cleaned[:200])
        raise SQLGuardError("SQL injection pattern detected.")

    # 2. 只读模式：禁止危险关键词（仅当 settings.sql_readonly=True）
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

    # 3. sqlparse AST 级别：确保所有语句都是 SELECT
    # sqlparse.parse() 返回语句列表，每个语句都需要检查
    # stmt.get_type() 返回 "SELECT"、"INSERT" 等，或 None（无法识别类型）
    # 注意：只对能识别类型的语句做强制检查（None type 交给多语句检测兜底）
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

    # 4. 多语句检测（防止 ; 分隔的注入，如 "SELECT 1; DROP TABLE users"）
    # 只统计有明确 type 的语句（过滤 sqlparse 解析出的空语句/注释）
    statements = [s for s in sqlparse.parse(cleaned) if s.get_type()]
    if len(statements) > 1:
        metrics.sql_blocked_total.inc()
        logger.warning("sql_guard.multiple_statements", sql=cleaned[:200])
        raise SQLGuardError("Multiple SQL statements are not allowed.")

    logger.debug("sql_guard.passed", sql=cleaned[:100])
    return cleaned


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def extract_referenced_tables(sql: str) -> list[str]:
    """
    用 sqlparse 语法树提取 SQL 中引用的表名（P2-13 重写）。

    相比旧的正则实现，现在可以正确处理：
    - 逗号 join 列表：`FROM allowed, secret`（旧正则只提取到 allowed，可跨租户绕过）
    - 引号标识符归一化：`FROM "secret"` / `` FROM `secret` ``（旧正则提取为空）
    - schema 限定名：`FROM public.users`
    - 子查询递归：`FROM (SELECT ... FROM x)` 会提取到 x

    使用场景：
    1. enforce_allowed_tables() 中用于白名单校验
    2. SQLTool._audit_sql() 中记录审计日志（哪些表被访问了）

    实现原理：
    - 递归扫描 token 树，遇到 FROM/JOIN 关键字时，取其后紧跟的表引用 token
      （Identifier 或 IdentifierList）
    - IdentifierList 拆分成多个 Identifier（逗号 join 的每一张表）
    - Identifier 内只收集点号连接的 Name/Symbol 作为表名，
      遇到空白/AS/别名即停止（避免把别名当表名）
    - Parenthesis（子查询/括号）递归进入，提取嵌套 FROM

    Args:
        sql: SQL 字符串（应已通过 validate_sql 清理）

    Returns:
        去重后的表名列表（保留首次出现顺序，去掉引号和反引号）
    """
    tables: list[str] = []
    parsed = sqlparse.parse(sql)
    if not parsed:
        return []
    _walk_table_refs(parsed[0].tokens, tables)

    # 去重（保留首次出现顺序）
    seen: set[str] = set()
    result: list[str] = []
    for table in tables:
        key = table.lower()
        if key not in seen:
            seen.add(key)
            result.append(table)
    return result


def _walk_table_refs(tokens: list, tables: list[str]) -> None:
    """
    递归扫描 token 序列，在 FROM/JOIN 关键字后收集表引用，并深入嵌套结构。

    只对 FROM/JOIN 关键字后的 token 做表引用处理；
    其他复合 token（Where/Comparison/Parenthesis 等）继续递归，
    以捕获子查询内部的 FROM/JOIN。
    """
    for i, tok in enumerate(tokens):
        if tok.is_keyword and tok.normalized.upper() in ("FROM", "JOIN"):
            j = i + 1
            while j < len(tokens) and tokens[j].is_whitespace:
                j += 1
            if j < len(tokens):
                _collect_identifier(tokens[j], tables)
        elif hasattr(tok, "tokens") and tok.tokens:
            _walk_table_refs(tok.tokens, tables)


def _collect_identifier(node: Any, tables: list[str]) -> None:
    """
    从单个表引用节点提取表名（支持 Identifier / IdentifierList / Parenthesis）。

    - IdentifierList：逗号 join，逐个拆分 Identifier 收集
    - Parenthesis：括号表达式（子查询），递归寻找内部 FROM/JOIN
    - Identifier：只收集点号连接的 Name/Symbol（含引号标识符归一化），
      遇到空白/AS/别名即停止
    """
    if isinstance(node, _sql.IdentifierList):
        for ident in node.get_identifiers():
            _collect_identifier(ident, tables)
        return
    if isinstance(node, _sql.Parenthesis):
        # 括号表达式（子查询等）：递归寻找其中的 FROM/JOIN
        _walk_table_refs(node.tokens, tables)
        return
    # Identifier：只收集点号连接的 Name/Symbol，遇到空白/AS/别名即停止
    parts: list[str] = []
    for sub in node.tokens:
        tt = sub.ttype
        if tt in (_tokens.Name, _tokens.Literal.String.Symbol):
            parts.append(sub.value)
        elif tt is _tokens.Punctuation and sub.value == ".":
            parts.append(".")
        elif isinstance(sub, _sql.Parenthesis):
            # 子查询（如 FROM (SELECT ...) AS alias）
            _walk_table_refs(sub.tokens, tables)
            break  # 子查询之后的 alias 不是表
        else:
            break  # 空白 / AS / 别名 → 表名结束
    name = "".join(parts).strip(".").strip('"`[]')
    if name:
        tables.append(name)


def enforce_allowed_tables(sql: str, allowed_tables: list[str]) -> None:
    """
    校验 SQL 只访问了允许白名单内的表，违规则抛出 SQLGuardError。

    使用场景：
    - 多租户环境：每个租户只能访问自己的表
    - 数据隔离：防止 Agent 意外访问敏感业务表
    - 开发保护：限制开发环境只能访问测试数据集

    当 allowed_tables 为空时直接跳过（表示不限制）：
    - 空列表意味着未配置白名单，遵循"不配置即不限制"语义
    - 这与"空列表=拒绝所有"语义相反，是为了方便默认部署

    大小写不敏感比较：
    - SQL 表名通常大小写不敏感
    - allowed_tables 和 referenced tables 都统一转小写比较
    - 避免因大小写差异导致误拦截

    Args:
        sql: 已通过 validate_sql() 清理的 SQL
        allowed_tables: 允许访问的表名列表（空列表 = 不限制）

    Raises:
        SQLGuardError: SQL 中引用了白名单之外的表
    """
    if not allowed_tables:
        return  # 未配置白名单，不限制

    allowed = {table.lower() for table in allowed_tables}
    referenced = extract_referenced_tables(sql)
    blocked = [table for table in referenced if table.lower() not in allowed]
    if blocked:
        metrics.sql_blocked_total.inc()
        logger.warning("sql_guard.table_not_allowed", tables=blocked, sql=sql[:200])
        raise SQLGuardError(
            f"Referenced table(s) not allowed by policy: {', '.join(blocked)}"
        )
