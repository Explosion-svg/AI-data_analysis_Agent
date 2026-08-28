"""
tests/integration/test_warehouse.py

数据仓库层集成测试。

主要验证：
- 真实 sqlite 临时库上的初始化、建表、查询
- 表名、schema、样本行与聚合查询能力
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ai_data_agent.config.config import settings
from ai_data_agent.infra import warehouse


@pytest.mark.asyncio
async def test_warehouse_execute_and_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 使用临时 sqlite 文件做真集成测试，验证 warehouse 的真实行为。
    db_path = tmp_path / "warehouse.db"
    monkeypatch.setattr(settings, "warehouse_url", f"sqlite+aiosqlite:///{db_path}")

    # P2-24：init_warehouse 之后引擎强制只读（连接层拒绝写）。
    # 因此先用一个独立可写引擎准备测试数据，再初始化仓库引擎。
    setup_engine = create_async_engine(settings.warehouse_url)
    async with setup_engine.begin() as conn:
        await conn.execute(text("CREATE TABLE sales (id INTEGER PRIMARY KEY, amount INTEGER, category TEXT)"))
        await conn.execute(text("INSERT INTO sales (amount, category) VALUES (10, 'A'), (20, 'B')"))
    await setup_engine.dispose()

    await warehouse.init_warehouse()
    try:
        engine = warehouse.get_warehouse_engine()
        # P2-24：引擎层只读，写操作必须在连接层被拒绝
        async with engine.begin() as conn:
            with pytest.raises(Exception):
                await conn.execute(text("INSERT INTO sales (amount, category) VALUES (99, 'X')"))

        tables = await warehouse.get_table_names()
        schema = await warehouse.get_table_schema("sales")
        sample = await warehouse.get_sample_rows("sales", n=1)
        result = await warehouse.execute("SELECT SUM(amount) AS total_amount FROM sales")

        assert "sales" in tables
        assert any(col["name"] == "amount" for col in schema)
        assert len(sample) == 1
        assert result.to_dict(orient="records") == [{"total_amount": 30}]
    finally:
        await warehouse.close_warehouse()
