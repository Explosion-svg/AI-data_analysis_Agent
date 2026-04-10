"""
tests/unit/test_benchmark_dataset.py

评估数据集单元测试。

主要验证：
- BenchmarkDataset 的增删查筛选
- 保存与加载
- 默认样例是否可用
"""

from __future__ import annotations

from pathlib import Path

from ai_data_agent.evaluation.benchmark_dataset import BenchmarkDataset, EvalCase, get_default_dataset


def test_benchmark_dataset_add_get_list_and_len() -> None:
    # 验证数据集的基础管理能力：新增、查询、筛选、统计数量。
    dataset = BenchmarkDataset()
    case1 = EvalCase(id="c1", question="q1", category="sql", difficulty="easy")
    case2 = EvalCase(id="c2", question="q2", category="rag", difficulty="hard")

    dataset.add(case1)
    dataset.add(case2)

    assert len(dataset) == 2
    assert dataset.get("c1").question == "q1"
    assert [c.id for c in dataset.list(category="sql")] == ["c1"]
    assert [c.id for c in dataset.list(difficulty="hard")] == ["c2"]


def test_benchmark_dataset_save_and_load(tmp_path: Path) -> None:
    # 验证数据集能序列化到 JSON 并正确恢复。
    dataset = BenchmarkDataset()
    dataset.add(
        EvalCase(
            id="c1",
            question="今年销量是多少",
            expected_tools=["sql_query"],
            tags=["sales"],
        )
    )
    path = tmp_path / "benchmark.json"

    dataset.save(path)
    loaded = BenchmarkDataset.load(path)

    assert len(loaded) == 1
    assert loaded.get("c1").expected_tools == ["sql_query"]
    assert loaded.get("c1").tags == ["sales"]


def test_get_default_dataset_contains_examples() -> None:
    # 默认数据集至少应包含若干预置评估样例。
    dataset = get_default_dataset()

    assert len(dataset) >= 4
    assert any(case.id == "rag_001" for case in dataset.list())
