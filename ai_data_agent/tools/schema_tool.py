"""
tools/schema_tool.py — 数据库 Schema 查询工具
让 LLM 能在对话中动态查询表结构
"""
from __future__ import annotations

from typing import Any

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.infra import warehouse


class SchemaTool(BaseTool):
    @property
    def name(self) -> str:
        return "get_schema"

    @property
    def description(self) -> str:
        return (
            "Query the data warehouse schema. "
            "List all tables, or get columns/sample rows for a specific table. "
            "Use this FIRST when you don't know which tables exist."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_tables", "describe_table", "sample_rows"],
                    "description": (
                        "Action: 'list_tables' to list all tables, "
                        "'describe_table' to get column info, "
                        "'sample_rows' to get a few sample rows."
                    ),
                },
                "table_name": {
                    "type": "string",
                    "description": "Table name (required for describe_table / sample_rows).",
                },
                "n_samples": {
                    "type": "integer",
                    "description": "Number of sample rows (default: 3).",
                    "default": 3,
                },
            },
            "required": ["action"],
        }

    async def _run(
        self,
        action: str,
        table_name: str | None = None,
        n_samples: int = 3,
        **_: Any,
    ) -> ToolResult:
        if action == "list_tables":
            tables = await warehouse.get_table_names()
            text = "Tables in warehouse:\n" + "\n".join(f"- {t}" for t in tables)
            return ToolResult(success=True, data=tables, text=text)

        if action in ("describe_table", "sample_rows"):
            if not table_name:
                return ToolResult(
                    success=False, error="table_name is required for this action."
                )

        if action == "describe_table":
            cols = await warehouse.get_table_schema(table_name)  # type: ignore[arg-type]
            lines = [f"  {c['name']} ({c['type']}) {'NULL' if c['nullable'] else 'NOT NULL'}"
                     for c in cols]
            text = f"Table `{table_name}` columns:\n" + "\n".join(lines)
            return ToolResult(success=True, data=cols, text=text)

        if action == "sample_rows":
            df = await warehouse.get_sample_rows(table_name, n=n_samples)  # type: ignore[arg-type]
            text = f"Sample rows from `{table_name}`:\n{df.to_markdown(index=False)}"
            return ToolResult(
                success=True,
                data=df.to_dict(orient="records"),
                text=text,
            )

        return ToolResult(success=False, error=f"Unknown action: '{action}'")
