"""
memory/work_memory_summarizer.py — 工作记忆摘要生成器

职责：
  将工具执行的原始结果压缩成适合写入 WorkMemory 的稳定、简短摘要文本。

为什么需要单独的摘要器？
  - agent_loop 负责流程编排（不应同时承担格式化细节）
  - tool 只负责返回原始 ToolResult（不关心如何被摘要）
  - 这个摘要器负责"中间转换"：原始结果 → 工作记忆可存储摘要

摘要原则（"稳定"的两层含义）：
  1. 固定句式：让工作记忆中的摘要格式一致，便于 prompt 消费和问题排查
  2. 体积控制：不把大块原始数据直接塞入工作记忆，防止状态无限膨胀

设计边界：
  - 只生成摘要文本，不直接操作 WorkMemory（不调用 work_memory.xxx()）
  - 所有方法都是静态方法，不需要实例状态，可直接调用
  - 保持轻量，不依赖 LLM（纯字符串处理）
"""
from __future__ import annotations

from typing import Any


class WorkMemorySummarizer:
    """
    生成适合写入 WorkMemory 的稳定摘要文本。

    所有方法均为静态方法（@staticmethod），因为摘要生成是纯函数，
    不依赖任何实例状态。这样可以直接通过类名调用：
    WorkMemorySummarizer.summarize_rows(rows)
    """

    @staticmethod
    def summarize_rows(rows: list[dict[str, Any]]) -> str:
        """
        将 SQL 查询结果压缩成简短的摘要文本。

        压缩策略（三个层级的信息）：
        1. 行数：让后续推理知道查询规模（0 行 vs 1000 行语义不同）
        2. 列名概览：让后续步骤知道可以引用哪些字段
        3. 首行预览：对数据格式的快速了解（数据类型、典型值）

        为什么不保存完整数据？
        - 完整结果集可能有数百行、几十列
        - 塞进 work_memory 会导致 prompt 超 token 预算
        - LLM 不需要完整数据，只需要"结果的关键特征"

        Args:
            rows: SQL 查询结果行列表（每行是字段名→值的字典）

        Returns:
            简短的摘要字符串，包含行数、列名、首行预览

        Example:
            >>> WorkMemorySummarizer.summarize_rows([{"date": "2026-01", "amount": 1000}])
            "SQL returned 1 row(s). Columns: date, amount. First row preview: {'date': '2026-01', 'amount': 1000}"
        """
        if not rows:
            return "SQL returned 0 rows."

        first = rows[0]
        columns = list(first.keys())
        return (
            f"SQL returned {len(rows)} row(s). "
            f"Columns: {', '.join(columns[:12])}. "   # 最多显示 12 列名
            f"First row preview: {str(first)[:300]}"   # 首行截断到 300 字符
        )

    @staticmethod
    def summarize_tool_result(
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: Any,
        observation: str,
    ) -> str:
        """
        为单次工具调用生成工作记忆步骤摘要。

        摘要强调"这一步发生了什么"（过程摘要），而不是简单复读 observation。
        原因：
        - observation 是原始返回文本，往往较长、包含完整数据
        - work_memory 步骤摘要应该是结构化、短句式的过程记录
        - 后续 prompt 注入时需要控制体积

        各工具的特化摘要：
        - sql_query：记录 SQL 语句（前 200 字符）和返回行数
        - generate_chart：简单确认图表生成成功
        - 其他工具：使用 tool_result.text 的前 240 字符

        失败处理：
        - tool_result 为 None（工具执行前已出错）：记录失败信息
        - tool_result.success=False（工具执行失败）：记录错误信息

        Args:
            tool_name: 工具名称（如 "sql_query"、"generate_chart"）
            tool_args: 工具参数字典（用于提取 SQL 等关键信息）
            tool_result: ToolResult 对象，可为 None（执行前出错时）
            observation: 原始 observation 文本（来自 ToolResult.to_observation()）

        Returns:
            适合存入 work_memory 步骤的摘要字符串
        """
        # 执行前就失败了（如工具不存在）
        if tool_result is None:
            return f"{tool_name} failed before producing a ToolResult."

        # 工具执行失败
        if not tool_result.success:
            return f"{tool_name} failed: {tool_result.error or observation[:160]}"

        # 各工具的专属摘要格式
        if tool_name == "sql_query":
            sql = tool_args.get("sql", "")
            # 计算返回行数（data 可能是列表或 None）
            rows = len(tool_result.data or []) if isinstance(tool_result.data, list) else "?"
            return f"Executed SQL query successfully, rows={rows}, sql={str(sql)[:200]}"

        if tool_name == "generate_chart":
            # 图表生成通常不需要摘要内容，只需确认成功
            return "Chart generated successfully."

        # 其他工具（python_analysis、search_documents、get_schema 等）
        # 使用 text 或 observation 的前 240 字符
        return (tool_result.text or observation or f"{tool_name} succeeded.")[:240]
