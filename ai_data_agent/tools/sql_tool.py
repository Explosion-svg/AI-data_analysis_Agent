"""
tools/sql_tool.py — SQL 执行工具（SQLTool）

职责：
  1. 安全校验：通过 sql_guard 多层检验，拒绝危险 SQL
  2. 限制结果集：自动注入 LIMIT 防止拉取超大结果集
  3. 执行查询：通过 warehouse（数据仓库适配层）执行 SQL
  4. 序列化输出：返回 markdown 格式的结果摘要 + dict records 格式的结构化数据
  5. 审计日志：记录完整的 SQL 审计轨迹（谁、访问了什么、结果如何）

安全流水线（Defence in Depth）：
  LLM 生成的 SQL → [validate_sql] → [enforce_allowed_tables] → LIMIT 注入 → warehouse.execute

LIMIT 注入策略：
  if max_rows > 0 and "limit" not in safe_sql.lower():
      safe_sql = f"SELECT * FROM ({safe_sql}) AS _q LIMIT {int(max_rows)}"

  包装成子查询的原因：
  - 原 SQL 可能有 ORDER BY（ORDER BY 必须在 LIMIT 之前，直接追加会出语法错误）
  - 子查询包装确保语义正确性（先排序，再取 top N 行）
  - int(max_rows) 防止浮点数注入（确保 LIMIT 值是整数）

输出格式（ToolResult）：
  - data: list[dict]（df.to_dict(orient="records")，每行是一个字典）
    → 供 python_analysis/generate_chart 工具直接使用
  - text: markdown 表格摘要
    → 供 LLM 消费（ReAct 循环的 Observation）

审计日志的必要性：
  数据分析系统一旦上线，SQL 审计是合规和安全必需品：
  - 追踪谁发起了哪些查询（request_id、user_id、tenant_id）
  - 统计哪些表被访问最多（用于权限审计）
  - 监控拦截率（outcome=blocked 的比例）

与 sql_guard 的关系：
  sql_guard 只做校验（返回 cleaned SQL 或抛异常），
  sql_tool 才做执行和输出格式化，两者职责分离。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.context.request_context import get_request_context
from ai_data_agent.observability.logger import get_logger
from ai_data_agent.observability.metrics import metrics
from ai_data_agent.reliability.sql_guard import (
    SQLGuardError,
    enforce_allowed_tables,
    extract_referenced_tables,
    validate_sql,
)
from ai_data_agent.reliability.timeout import run_with_timeout
from ai_data_agent.config.config import settings

logger = get_logger(__name__)


class SQLTool(BaseTool):
    """
    SQL 查询执行工具，集成安全检查和审计日志。

    工具名：sql_query
    并发槽：ConcurrencyLimiter 的 "sql_query" 桶（settings.sql_tool_concurrency）
    超时：settings.sql_query_timeout 秒（防止慢查询阻塞 Agent）
    """

    @property
    def name(self) -> str:
        """返回工具名称 "sql_query"，与 ConcurrencyLimiter 的桶名一致。"""
        return "sql_query"

    @property
    def description(self) -> str:
        """
        LLM 工具选择时参考的描述。

        描述要点：
        - 明确说明只支持 SELECT（LLM 不会尝试生成 INSERT/UPDATE）
        - 说明返回格式（数据表形式，供后续分析）
        - 给出典型使用场景（检索数据、聚合指标）
        """
        return (
            "Execute a SELECT SQL query against the data warehouse and return results as a table. "
            "Use this tool to retrieve data, aggregate metrics, or answer data-related questions."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """
        SQL 工具的参数 JSON Schema。

        参数：
        - sql（必填）：SELECT SQL 语句
        - max_rows（可选，默认 100）：最大返回行数
          → 防止 LLM 生成全表扫描查询导致结果集爆炸
        """
        return {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL SELECT statement to execute.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum number of rows to return (default: 100).",
                    "default": 100,
                },
            },
            "required": ["sql"],
        }

    async def _run(self, sql: str, max_rows: int = 100, **_: Any) -> ToolResult:
        """
        执行 SQL 查询的核心逻辑（四步流程）。

        步骤：
        1. 安全校验（validate_sql + enforce_allowed_tables）
           → 失败时记录审计日志（outcome=blocked）并返回 ToolResult(success=False)
        2. 注入 LIMIT（当 max_rows > 0 且 SQL 中没有 LIMIT）
           → 包装成子查询，确保 ORDER BY 语义正确
        3. 执行查询（warehouse.execute，带超时）
           → 超时或执行失败时记录审计日志（outcome=failed）
        4. 序列化输出（DataFrame → text + data）
           → 记录审计日志（outcome=success）

        **_ 参数：
        - 接收并忽略任何额外参数（Executor 可能注入 data 等额外字段）
        - 防止因参数不匹配导致 TypeError

        Args:
            sql: LLM 生成的 SELECT SQL 语句
            max_rows: 最大返回行数（默认 100，0 表示不限制）
            **_: 忽略的额外参数

        Returns:
            ToolResult：
            - 成功：success=True, data=list[dict], text=markdown 表格
            - 失败：success=False, error=失败原因
        """
        # Step 1: 安全校验（多层防御）
        try:
            safe_sql = validate_sql(sql)
            enforce_allowed_tables(safe_sql, settings.sql_allowed_tables)
        except SQLGuardError as e:
            self._audit_sql(sql=sql, safe_sql="", rows=0, outcome="blocked", error=str(e))
            return ToolResult(success=False, error=f"SQL safety check failed: {e}")

        # Step 2: 注入 LIMIT（防止超大结果集）
        # 使用子查询包装，确保 ORDER BY ... LIMIT 的语义正确
        if max_rows > 0 and "limit" not in safe_sql.lower():
            safe_sql = f"SELECT * FROM ({safe_sql}) AS _q LIMIT {int(max_rows)}"

        # Step 3: 执行查询（带超时保护）
        from ai_data_agent.infra import warehouse
        try:
            df: pd.DataFrame = await run_with_timeout(
                warehouse.execute(safe_sql),
                timeout=settings.sql_query_timeout,
                name="sql_tool",
            )
        except Exception as e:
            self._audit_sql(sql=sql, safe_sql=safe_sql, rows=0, outcome="failed", error=str(e))
            return ToolResult(success=False, error=f"SQL execution failed: {e}")

        # Step 4: 序列化输出
        rows, cols = df.shape
        if rows == 0:
            text = "Query returned no rows."
        else:
            text = (
                f"Query returned {rows} row(s), {cols} column(s).\n"
                f"{df.to_markdown(index=False)}"
            )
        self._audit_sql(sql=sql, safe_sql=safe_sql, rows=rows, outcome="success")

        return ToolResult(
            success=True,
            data=df.to_dict(orient="records"),   # list[dict] 格式，供 python_analysis/chart 消费
            text=text,                             # markdown 格式，供 LLM 消费
        )

    def _audit_sql(
        self,
        *,
        sql: str,
        safe_sql: str,
        rows: int,
        outcome: str,
        error: str = "",
    ) -> None:
        """
        记录 SQL 审计日志（合规和安全追踪）。

        审计内容：
        - 请求来源：request_id、user_id、tenant_id（从 RequestContext 获取）
        - 访问内容：tables（从 SQL 中提取的表名列表）
        - 执行结果：outcome、rows 返回行数、error 错误信息
        - SQL 快照：sql_preview（前 240 字符，不存全量避免日志过大）

        outcome 枚举：
        - "blocked"：被 sql_guard 拦截（注入/危险 SQL）
        - "failed"：执行失败（数据库错误/超时）
        - "success"：执行成功

        为什么不保存全量结果集：
        - 结果集可能有数千行，存到日志会导致日志大小爆炸
        - 合规审计只需要"谁访问了什么表、结果行数"，不需要具体数据
        - 敏感数据（如 PII）不应出现在日志中

        Args:
            sql: 原始（用户/LLM 提交的）SQL
            safe_sql: 通过安全校验后的 SQL（blocked 时为空字符串）
            rows: 返回的行数（成功时才有意义）
            outcome: 执行结果（blocked/failed/success）
            error: 错误信息（成功时为空字符串）
        """
        req_ctx = get_request_context()
        metrics.sql_audit_total.labels(outcome=outcome).inc()
        logger.info(
            "sql_tool.audit",
            request_id=req_ctx.request_id if req_ctx else "system",
            user_id=req_ctx.user_id if req_ctx else settings.default_user_id,
            tenant_id=req_ctx.tenant_id if req_ctx else settings.default_tenant_id,
            outcome=outcome,
            tables=extract_referenced_tables(safe_sql or sql),
            rows=rows,
            sql_preview=(safe_sql or sql)[:240],
            error=error[:240],
        )
