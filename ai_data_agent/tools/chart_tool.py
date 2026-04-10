"""
tools/chart_tool.py — 图表生成工具
使用 Plotly 生成交互式图表，输出 JSON（可直接嵌入前端）
"""
from __future__ import annotations

import json
from typing import Any

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

_CHART_TYPES = {"bar", "line", "scatter", "pie", "histogram", "box", "heatmap", "area"}


class ChartTool(BaseTool):
    @property
    def name(self) -> str:
        return "generate_chart"

    @property
    def description(self) -> str:
        return (
            "Generate an interactive chart using Plotly. "
            "Supports: bar, line, scatter, pie, histogram, box, area, heatmap. "
            "Returns chart JSON that can be rendered in the frontend."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
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

            kwargs: dict[str, Any] = {"data_frame": df, "title": title or chart_type.capitalize()}
            if x:
                kwargs["x"] = x
            if y:
                kwargs["y"] = y
            if color:
                kwargs["color"] = color
            if labels:
                kwargs["labels"] = labels

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

            # pie 使用 names / values 而非 x / y
            if chart_type == "pie":
                if x:
                    kwargs["names"] = kwargs.pop("x")
                if y and isinstance(y, str):
                    kwargs["values"] = kwargs.pop("y")

            fig = chart_fn(**kwargs)

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
