"""
tools/sql_tool.py — SQL 执行工具
整合 sql_guard 安全校验 + 数据仓库执行
返回 DataFrame 的 markdown 摘要
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.reliability.sql_guard import validate_sql, SQLGuardError
from ai_data_agent.reliability.timeout import run_with_timeout
from ai_data_agent.config.config import settings


class SQLTool(BaseTool):
    @property
    def name(self) -> str:
        return "sql_query"

    @property
    def description(self) -> str:
        return (
            "Execute a SELECT SQL query against the data warehouse and return results as a table. "
            "Use this tool to retrieve data, aggregate metrics, or answer data-related questions."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
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
        # 1. 安全校验
        try:
            safe_sql = validate_sql(sql)
        except SQLGuardError as e:
            return ToolResult(success=False, error=f"SQL safety check failed: {e}")

        # 2. 注入 LIMIT 防止拉取超大结果集
        if max_rows > 0 and "limit" not in safe_sql.lower():
            safe_sql = f"SELECT * FROM ({safe_sql}) AS _q LIMIT {int(max_rows)}"

        # 3. 执行（带超时）
        from ai_data_agent.infra import warehouse
        try:
            df: pd.DataFrame = await run_with_timeout(
                warehouse.execute(safe_sql),
                timeout=settings.sql_query_timeout,
                name="sql_tool",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"SQL execution failed: {e}")

        # 4. 序列化输出
        rows, cols = df.shape
        if rows == 0:
            text = "Query returned no rows."
        else:
            text = (
                f"Query returned {rows} row(s), {cols} column(s).\n"
                f"{df.to_markdown(index=False)}"
            )

        return ToolResult(
            success=True,
            data=df.to_dict(orient="records"),
            text=text,
        )
