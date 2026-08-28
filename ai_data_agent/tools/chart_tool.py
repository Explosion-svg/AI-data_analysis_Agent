"""
tools/chart_tool.py — 图表生成工具（ChartTool）

职责：
  使用 Plotly Express 根据数据和图表配置生成交互式图表，
  输出 JSON 格式（可直接嵌入前端渲染，如 React Plotly 组件）。

输出格式（JSON）：
  fig.to_json() 返回完整的 Plotly figure JSON，包含：
  - data：数据系列（traces）
  - layout：图表布局（title、轴标签、图例等）
  - config：渲染配置

  前端接收此 JSON 后可直接使用 Plotly.react(element, data, layout, config) 渲染，
  无需后端额外处理。

饼图特殊处理：
  Plotly 饼图使用 names（分类）和 values（数值）而不是 x 和 y：
      px.pie(data_frame=df, names="category", values="amount")
  因此在 chart_type == "pie" 时，需要将 x → names，y → values。
  这个转换在 _run() 中完成（kwargs 的 pop + 重新赋值）。

与 sql_tool 的协作：
  sql_query → ToolResult.data (list[dict]) → ChartTool._run(data=...)
  Executor._inject_data() 负责将 sql_query 的结果注入 generate_chart 的 data 参数。
  LLM 不需要（也不应该）在参数中填写 data 字段（prompt 明确说明）。

延迟导入（plotly）：
  plotly 和 pandas 在 _run() 内部导入，原因：
  - 避免 import 阶段的重型库加载拖慢应用启动
  - 如果 plotly 未安装，只有在实际调用时才报错（不影响其他功能）
"""
from __future__ import annotations

import json
from typing import Any

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# 支持的图表类型集合（用于参数校验和 parameters_schema 的 enum）
_CHART_TYPES = {"bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "area"}


class ChartTool(BaseTool):
    """
    Plotly 交互式图表生成工具。

    工具名：generate_chart
    并发槽：ConcurrencyLimiter 的 "generate_chart" 桶
    """

    @property
    def name(self) -> str:
        """返回工具名称 "generate_chart"。"""
        return "generate_chart"

    @property
    def description(self) -> str:
        """工具描述，列出支持的图表类型和输出格式。"""
        return (
            "Generate an interactive chart using Plotly. "
            "Supports: bar, line, scatter, pie, histogram, box, area, heatmap. "
            "Returns chart JSON that can be rendered in the frontend."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """
        图表工具的参数 JSON Schema。

        核心参数：
        - chart_type（必填）：图表类型（enum 限制，防止 LLM 生成不支持的类型）
        - data（必填，但通常由 Executor 注入，LLM 不填）：数据记录列表

        可选参数：
        - x：X 轴列名
        - y：Y 轴列名（支持单列 string 或多列 array）
        - title：图表标题
        - color：颜色分组列名（如按 "category" 分组着色）
        - labels：轴标签别名映射（如 {"date": "日期", "amount": "金额"}）
        """
        return {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": list(_CHART_TYPES),
                    "description": "Type of chart to generate.",
                },
                "data": {
                    "type": "array",
                    "description": "List of records (dicts) to visualize.",
                    "items": {"type": "object"},
                },
                "x": {"type": "string", "description": "Column name for X axis."},
                "y": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "Column name(s) for Y axis.",
                },
                "title": {"type": "string", "description": "Chart title."},
                "color": {"type": "string", "description": "Column to use for color grouping."},
                "labels": {"type": "object", "description": "Axis label overrides."},
            },
            "required": ["chart_type", "data"],
        }

    async def _run(
        self,
        chart_type: str,
        data: list[dict[str, Any]],
        x: str | None = None,
        y: str | list[str] | None = None,
        title: str = "",
        color: str | None = None,
        labels: dict[str, str] | None = None,
        **_: Any,
    ) -> ToolResult:
        """
        使用 Plotly Express 生成图表，返回 JSON 格式的图表数据。

        参数校验：
        1. chart_type 必须在 _CHART_TYPES 集合中（快速失败，避免进入 try）
        2. data 不能为空（无数据无法生成图表）

        图表生成策略：
        - 使用 dict 映射 chart_type → px.xxx 函数（比 if/elif 更简洁）
        - kwargs 字典动态构建（只添加非 None 的参数）
        - heatmap 使用 lambda 包装（px.density_heatmap 与其他接口略有差异）

        饼图特殊处理：
        - Plotly pie 使用 names/values 而不是 x/y
        - 将 kwargs["x"] pop 后赋给 kwargs["names"]
        - 将 kwargs["y"] pop 后赋给 kwargs["values"]
        - 只处理 y 是字符串的情况（饼图单值轴）

        输出格式：
        - fig.to_json() 返回 JSON 字符串
        - json.loads() 将字符串转换为 Python dict（存储在 ToolResult.data）
        - 前端可直接接收 dict 格式（JSON 序列化时自动处理）

        Args:
            chart_type: 图表类型（bar/line/scatter/pie/histogram/box/area/heatmap）
            data: 数据记录列表（由 Executor._inject_data 注入）
            x: X 轴列名（可选）
            y: Y 轴列名或列名列表（可选）
            title: 图表标题（可选，默认为 chart_type 首字母大写）
            color: 颜色分组列名（可选）
            labels: 轴标签别名映射（可选）
            **_: 忽略的额外参数

        Returns:
            ToolResult：
            - 成功：success=True, data=Plotly JSON dict, text=摘要信息
            - 失败：success=False, error=错误信息
        """
        # 参数校验（快速失败）
        if chart_type not in _CHART_TYPES:
            return ToolResult(
                success=False,
                error=f"Unsupported chart type '{chart_type}'. Use one of: {_CHART_TYPES}",
            )
        if not data:
            return ToolResult(success=False, error="No data provided for chart.")

        try:
            import plotly.express as px
            import pandas as pd

            df = pd.DataFrame(data)

            # 动态构建 Plotly Express 参数（只添加非 None 值）
            kwargs: dict[str, Any] = {"data_frame": df, "title": title or chart_type.capitalize()}
            if x:
                kwargs["x"] = x
            if y:
                kwargs["y"] = y
            if color:
                kwargs["color"] = color
            if labels:
                kwargs["labels"] = labels

            # 图表类型 → Plotly Express 函数映射
            chart_fn = {
                "bar": px.bar,
                "line": px.line,
                "scatter": px.scatter,
                "pie": px.pie,
                "histogram": px.histogram,
                "box": px.box,
                "area": px.area,
                "heatmap": lambda **kw: px.density_heatmap(**kw),
            }.get(chart_type, px.bar)

            # 饼图特殊处理：Plotly 饼图 API 使用 names/values 而非 x/y
            if chart_type == "pie":
                if x:
                    kwargs["names"] = kwargs.pop("x")   # x → names（分类轴）
                if y and isinstance(y, str):
                    kwargs["values"] = kwargs.pop("y")  # y → values（数值轴）

            # 生成图表
            fig = chart_fn(**kwargs)

            # 序列化为 JSON（fig.to_json() 返回字符串，json.loads 转为 dict）
            chart_json = json.loads(fig.to_json())
            logger.debug("chart_tool.generated", chart_type=chart_type, rows=len(data))
            return ToolResult(
                success=True,
                data=chart_json,
                text=f"Chart '{title or chart_type}' generated with {len(data)} data points.",
            )
        except Exception as e:
            logger.error("chart_tool.error", error=str(e))
            return ToolResult(success=False, error=f"Chart generation failed: {e}")
