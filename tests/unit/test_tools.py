"""
tests/unit/test_tools.py

工具层单元测试。

主要验证：
- ToolRegistry 的注册与导出
- BaseTool 的公共包装逻辑
- SQLTool / PythonTool / ChartTool 的关键行为
"""

from __future__ import annotations

import pandas as pd
import pytest

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.tools.chart_tool import ChartTool
from ai_data_agent.tools.python_tool import PythonTool
from ai_data_agent.tools.sql_tool import SQLTool
from ai_data_agent.tools.tool_registry import ToolRegistry


class SuccessTool(BaseTool):
    # 一个最小成功工具，用于验证 BaseTool/Registry 的公共行为。
    @property
    def name(self) -> str:
        return "success_tool"

    @property
    def description(self) -> str:
        return "ok"

    async def _run(self, **kwargs):
        return ToolResult(success=True, data=kwargs, text="done")


class ErrorTool(BaseTool):
    # 一个故意抛异常的工具，用于验证 BaseTool.run 的兜底逻辑。
    @property
    def name(self) -> str:
        return "error_tool"

    @property
    def description(self) -> str:
        return "boom"

    async def _run(self, **kwargs):
        raise RuntimeError("boom")


class BusyTool(BaseTool):
    # 一个用于验证 P2-10 并发超限传播的工具。
    @property
    def name(self) -> str:
        return "busy_tool"

    @property
    def description(self) -> str:
        return "busy"

    async def _run(self, **kwargs):
        return ToolResult(success=True, text="done")


def test_tool_registry_register_get_and_export() -> None:
    # 注册后应能通过名字获取，并导出给 OpenAI function calling 的 schema。
    registry = ToolRegistry()
    registry.register(SuccessTool())

    assert registry.get("success_tool").name == "success_tool"
    assert "success_tool" in registry
    assert registry.to_openai_tools()[0]["function"]["name"] == "success_tool"


@pytest.mark.asyncio
async def test_base_tool_run_wraps_exceptions() -> None:
    # 工具内部抛异常时，不应直接把异常冒泡到上层。
    result = await ErrorTool().run()
    assert result.success is False
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_base_tool_run_propagates_concurrency_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # P2-10：并发超限属于系统过载，应原样上抛（由 503 处理器处理），
    # 而不是被吞成 ToolResult(success=False)。
    from contextlib import asynccontextmanager

    from ai_data_agent.reliability.concurrency import ConcurrencyLimitExceeded

    class FakeLimiter:
        @asynccontextmanager
        async def limit(self, bucket):
            raise ConcurrencyLimitExceeded(bucket, 0.1)
            yield  # pragma: no cover

    monkeypatch.setattr(
        "ai_data_agent.tools.base_tool.get_limiter", lambda: FakeLimiter()
    )

    with pytest.raises(ConcurrencyLimitExceeded):
        await BusyTool().run()


@pytest.mark.asyncio
async def test_sql_tool_adds_limit_and_serializes(monkeypatch: pytest.MonkeyPatch) -> None:
    # SQLTool 需要验证两件事：
    # 1. 自动追加 LIMIT，防止大查询
    # 2. DataFrame 能被正确序列化为 records
    captured: dict[str, str] = {}

    async def execute(sql: str):
        captured["sql"] = sql
        return pd.DataFrame([{"total": 10}])

    monkeypatch.setattr("ai_data_agent.tools.sql_tool.validate_sql", lambda sql: sql)
    monkeypatch.setattr("ai_data_agent.infra.warehouse.execute", execute)

    result = await SQLTool().run(sql="SELECT total FROM metrics", max_rows=5)

    assert result.success is True
    assert "LIMIT 5" in captured["sql"]
    assert result.data == [{"total": 10}]


@pytest.mark.asyncio
async def test_sql_tool_caps_max_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # P2-12：LLM 传入超配置上限的 max_rows 应被钳制到硬上限。
    from ai_data_agent.config.config import settings

    monkeypatch.setattr(settings, "sql_max_rows_cap", 5)
    captured: dict[str, str] = {}

    async def execute(sql: str):
        captured["sql"] = sql
        return pd.DataFrame([{"v": i} for i in range(100)])

    monkeypatch.setattr("ai_data_agent.tools.sql_tool.validate_sql", lambda sql: sql)
    monkeypatch.setattr("ai_data_agent.infra.warehouse.execute", execute)

    result = await SQLTool().run(sql="SELECT v FROM metrics", max_rows=99999)

    assert "LIMIT 5" in captured["sql"]   # LIMIT 注入使用封顶值
    assert len(result.data) == 5          # 结果集不超过硬上限


@pytest.mark.asyncio
async def test_sql_tool_truncates_large_results(monkeypatch: pytest.MonkeyPatch) -> None:
    # P2-12：SQL 自带超大 LIMIT（绕过子串检测）时，结果集仍被后截断。
    from ai_data_agent.config.config import settings

    monkeypatch.setattr(settings, "sql_max_rows_cap", 1000)
    captured: dict[str, str] = {}

    async def execute(sql: str):
        captured["sql"] = sql
        return pd.DataFrame([{"v": i} for i in range(5000)])

    monkeypatch.setattr("ai_data_agent.tools.sql_tool.validate_sql", lambda sql: sql)
    monkeypatch.setattr("ai_data_agent.infra.warehouse.execute", execute)

    result = await SQLTool().run(sql="SELECT v FROM metrics LIMIT 5000", max_rows=10)

    assert "LIMIT 10" not in captured["sql"]  # 已有 limit 子串，不重复注入
    assert len(result.data) == 10              # 结果集被后截断
    assert "5000" not in result.text           # 观测文本按截断后行数描述


@pytest.mark.asyncio
async def test_sql_tool_truncates_observation_text(monkeypatch: pytest.MonkeyPatch) -> None:
    # P2-12：观测文本超过字符上限时被截断，防止打爆 LLM 上下文。
    from ai_data_agent.config.config import settings

    monkeypatch.setattr(settings, "sql_observation_max_chars", 100)

    async def execute(sql: str):
        return pd.DataFrame([{"v": i} for i in range(1000)])

    monkeypatch.setattr("ai_data_agent.tools.sql_tool.validate_sql", lambda sql: sql)
    monkeypatch.setattr("ai_data_agent.infra.warehouse.execute", execute)

    result = await SQLTool().run(sql="SELECT v FROM metrics", max_rows=1000)

    assert len(result.text) <= 100
    assert len(result.data) == 1000


@pytest.mark.asyncio
async def test_python_tool_executes_dataframe_code() -> None:
    # data 参数应被注入为 DataFrame 变量 df，供代码直接使用。
    result = await PythonTool().run(
        code="result = int(df['amount'].sum())",
        data=[{"amount": 1}, {"amount": 2}],
    )

    assert result.success is True
    assert result.data == 3


@pytest.mark.asyncio
async def test_python_tool_blocks_unsafe_import() -> None:
    # Python 沙箱应禁止导入未列入白名单的模块。
    result = await PythonTool().run(code="import os\nresult = 1")
    assert result.success is False
    assert "not allowed" in result.error


@pytest.mark.asyncio
async def test_chart_tool_generates_json() -> None:
    # 图表工具成功时，返回结构应是 Plotly 可渲染 JSON。
    result = await ChartTool().run(
        chart_type="bar",
        data=[{"month": "Jan", "sales": 10}, {"month": "Feb", "sales": 12}],
        x="month",
        y="sales",
        title="Sales",
    )

    assert result.success is True
    assert result.data["layout"]["title"]["text"] == "Sales"


@pytest.mark.asyncio
async def test_chart_tool_rejects_empty_data() -> None:
    # 空数据不应生成图表。
    result = await ChartTool().run(chart_type="bar", data=[])
    assert result.success is False
    assert "No data" in result.error
