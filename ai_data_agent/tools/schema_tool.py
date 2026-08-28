"""
tools/schema_tool.py — 数据库 Schema 查询工具（SchemaTool）

职责：
  让 LLM 能在对话中动态查询数据库结构信息：
  - 列出所有可用的表
  - 获取指定表的列名、类型、可空性
  - 获取指定表的样本数据行

在 Plan-and-Execute 中的作用：
  Planner 通常在 SQL 查询之前添加 get_schema 步骤，原因：
  - LLM 不知道数据库中有哪些表和列
  - 没有 schema 信息的 SQL 查询容易产生"表不存在"或"列名错误"的错误
  - schema 信息注入到 Executor._generate_params() 的 prompt 后，
    LLM 能生成更准确的 SQL

三种操作（action）：
  1. list_tables：列出所有表名
     → 用于"有哪些表"类问题，或当 LLM 不确定表名时先查一下

  2. describe_table（需要 table_name）：返回指定表的列定义
     → 格式：列名 (类型) NULL/NOT NULL
     → 帮助 LLM 了解可用列，生成正确的 SQL

  3. sample_rows（需要 table_name）：返回指定表的样本数据
     → 默认 3 行
     → 帮助 LLM 了解数据的实际内容和格式（如日期格式、枚举值等）

与 warehouse 的关系：
  SchemaTool 直接调用 infra/warehouse.py 中的 schema 查询方法，
  不经过 sql_guard（schema 查询不是用户输入的 SQL，是系统级操作）。
"""
from __future__ import annotations

from typing import Any

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.infra import warehouse


class SchemaTool(BaseTool):
    """
    数据库 Schema 查询工具，提供三种模式的结构查询。

    工具名：get_schema
    并发槽：ConcurrencyLimiter 的 "get_schema" 桶
    """

    @property
    def name(self) -> str:
        """返回工具名称 "get_schema"。"""
        return "get_schema"

    @property
    def description(self) -> str:
        """
        工具描述，明确说明什么时候用此工具。

        "Use this FIRST" 的设计意图：
        - 引导 LLM 在不确定表结构时优先查 schema
        - 比"先猜后改"更高效（减少 SQL 错误和重试次数）
        """
        return (
            "Query the data warehouse schema. "
            "List all tables, or get columns/sample rows for a specific table. "
            "Use this FIRST when you don't know which tables exist."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """
        Schema 工具的参数 JSON Schema。

        action 枚举：
        - "list_tables"：不需要 table_name
        - "describe_table"：需要 table_name（列定义）
        - "sample_rows"：需要 table_name + 可选 n_samples（样本数）

        JSON Schema 使用 enum 限制 action 的合法值，
        防止 LLM 生成 "get_columns" 等无效操作。
        """
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
        """
        根据 action 类型执行相应的 schema 查询操作。

        流程图（简化）：
          action == "list_tables" → warehouse.get_table_names() → 返回表名列表
          action == "describe_table" → 检查 table_name → warehouse.get_table_schema() → 返回列定义
          action == "sample_rows" → 检查 table_name → warehouse.get_sample_rows() → 返回样本数据
          其他 action → 返回 ToolResult(success=False, error="Unknown action")

        table_name 校验：
        - "describe_table" 和 "sample_rows" 都需要 table_name
        - 这两种操作共享同一个 table_name 必填校验（减少重复代码）
        - 校验失败立即返回 ToolResult(success=False)，不进入 warehouse 调用

        输出格式：
        list_tables：
          - data: list[str]（表名列表）
          - text: "Tables in warehouse:\n- table1\n- table2\n..."

        describe_table：
          - data: list[dict]（列定义，如 [{"name": "id", "type": "INTEGER", "nullable": False}]）
          - text: "Table `sales` columns:\n  id (INTEGER) NOT NULL\n  date (DATE) NULL\n..."

        sample_rows：
          - data: list[dict]（样本行的 records 格式）
          - text: "Sample rows from `sales`:\n| id | date | amount |\n|---|---|---|\n..."

        Args:
            action: 操作类型（list_tables / describe_table / sample_rows）
            table_name: 表名（describe_table 和 sample_rows 必填）
            n_samples: 样本行数（sample_rows 时使用，默认 3）
            **_: 忽略的额外参数

        Returns:
            ToolResult：
            - 成功：success=True, data=对应格式数据, text=格式化文本
            - 失败：success=False, error=失败原因
        """
        # 操作 1：列出所有表
        if action == "list_tables":
            tables = await warehouse.get_table_names()
            text = "Tables in warehouse:\n" + "\n".join(f"- {t}" for t in tables)
            return ToolResult(success=True, data=tables, text=text)

        # 需要 table_name 的操作：提前校验
        if action in ("describe_table", "sample_rows"):
            if not table_name:
                return ToolResult(
                    success=False, error="table_name is required for this action."
                )

        # 操作 2：描述表结构（列名、类型、可空性）
        if action == "describe_table":
            cols = await warehouse.get_table_schema(table_name)  # type: ignore[arg-type]
            # 格式化为 "列名 (类型) NULL/NOT NULL" 格式，直观易读
            lines = [f"  {c['name']} ({c['type']}) {'NULL' if c['nullable'] else 'NOT NULL'}"
                     for c in cols]
            text = f"Table `{table_name}` columns:\n" + "\n".join(lines)
            return ToolResult(success=True, data=cols, text=text)

        # 操作 3：获取样本数据行（帮助 LLM 理解数据格式）
        if action == "sample_rows":
            df = await warehouse.get_sample_rows(table_name, n=n_samples)  # type: ignore[arg-type]
            text = f"Sample rows from `{table_name}`:\n{df.to_markdown(index=False)}"
            return ToolResult(
                success=True,
                data=df.to_dict(orient="records"),  # list[dict] 格式
                text=text,
            )

        # 未知操作：返回错误（不应该发生，schema 的 enum 已经限制了合法值）
        return ToolResult(success=False, error=f"Unknown action: '{action}'")
