"""
memory/work_memory_summarizer.py — 写入工作记忆前的摘要器

这个模块只做一件事：把原始工具结果压缩成稳定、简短、适合进入 work_memory
的摘要文本。

设计边界：
- agent_loop 负责流程编排
- tool 自己负责返回原始 ToolResult
- 这个摘要器负责把“原始结果”转换为“工作记忆可存储摘要”

这样可以避免 orchestration 层继续堆积各种 if/else 格式化细节。
"""
from __future__ import annotations

from typing import Any


class WorkMemorySummarizer:
    """
    生成适合写入 WorkMemory 的稳定摘要。

    “稳定”有两个含义：
    1. 尽量保留固定句式，方便后续 prompt 消费和问题排查
    2. 不把大块原始内容直接塞进 work_memory，控制状态体积
    """

    @staticmethod
    def summarize_rows(rows: list[dict[str, Any]]) -> str:
        """
        压缩 SQL 查询结果。

        这里只保留三个层级的信息：
        - 返回行数
        - 列名概览
        - 首行预览

        这样既能让后续推理知道“查到了什么”，又不会把整批数据塞进
        work_memory，导致上下文膨胀。
        """
        if not rows:
            return "SQL returned 0 rows."

        first = rows[0]
        columns = list(first.keys())
        return (
            f"SQL returned {len(rows)} row(s). "
            f"Columns: {', '.join(columns[:12])}. "
            f"First row preview: {str(first)[:300]}"
        )

    @staticmethod
    def summarize_tool_result(
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
        observation: str,
    ) -> str:
        """
        生成单次工具调用的步骤摘要。

        这里强调“这一步发生了什么”，而不是原样复读 observation。
        原因是：
        - observation 往往更长，更像原始返回文本
        - work_memory 更适合存结构化、短句式的过程摘要
        """
        if tool_result is None:
            return f"{tool_name} failed before producing a ToolResult."

        if not tool_result.success:
            return f"{tool_name} failed: {tool_result.error or observation[:160]}"

        if tool_name == "sql_query":
            sql = tool_args.get("sql", "")
            rows = len(tool_result.data or []) if isinstance(tool_result.data, list) else "?"
            return f"Executed SQL query successfully, rows={rows}, sql={str(sql)[:200]}"

        if tool_name == "generate_chart":
            return "Chart generated successfully."

        return (tool_result.text or observation or f"{tool_name} succeeded.")[:240]
