"""
evaluation/benchmark_dataset.py — 基准测试数据集管理
存储测试问题 + 预期 SQL / 预期答案，用于回归评估
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    id: str
    question: str
    expected_sql: str | None = None
    expected_answer: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"   # easy | medium | hard
    tags: list[str] = field(default_factory=list)


class BenchmarkDataset:
    """测试用例管理器，支持从 JSON 文件加载和保存。"""

    def __init__(self) -> None:
        self._cases: dict[str, EvalCase] = {}

    def add(self, case: EvalCase) -> None:
        self._cases[case.id] = case

    def get(self, case_id: str) -> EvalCase:
        return self._cases[case_id]

    def list(
        self,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> list[EvalCase]:
        cases = list(self._cases.values())
        if category:
            cases = [c for c in cases if c.category == category]
        if difficulty:
            cases = [c for c in cases if c.difficulty == difficulty]
        return cases

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "id": c.id,
                "question": c.question,
                "expected_sql": c.expected_sql,
                "expected_answer": c.expected_answer,
                "expected_tools": c.expected_tools,
                "category": c.category,
                "difficulty": c.difficulty,
                "tags": c.tags,
            }
            for c in self._cases.values()
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkDataset":
        dataset = cls()
        data = json.loads(Path(path).read_text())
        for item in data:
            dataset.add(EvalCase(**item))
        return dataset

    def __len__(self) -> int:
        return len(self._cases)


def get_default_dataset() -> BenchmarkDataset:
    """内置示例测试集（可按需扩展）。"""
    ds = BenchmarkDataset()
    examples: list[dict[str, Any]] = [
        {
            "id": "sql_001",
            "question": "今年总销售额是多少？",
            "expected_sql": "SELECT SUM(amount) AS total_sales FROM sales WHERE YEAR(date) = YEAR(CURRENT_DATE)",
            "expected_tools": ["get_schema", "sql_query"],
            "category": "sql",
            "difficulty": "easy",
            "tags": ["sales", "aggregation"],
        },
        {
            "id": "sql_002",
            "question": "各产品类别的月度销售趋势",
            "expected_sql": None,
            "expected_tools": ["get_schema", "sql_query", "generate_chart"],
            "category": "visualization",
            "difficulty": "medium",
            "tags": ["sales", "trend", "chart"],
        },
        {
            "id": "python_001",
            "question": "计算销售额的同比增长率",
            "expected_sql": None,
            "expected_tools": ["sql_query", "python_analysis"],
            "category": "analysis",
            "difficulty": "hard",
            "tags": ["yoy", "growth", "python"],
        },
        {
            "id": "rag_001",
            "question": "什么是 GMV？",
            "expected_sql": None,
            "expected_tools": ["search_documents"],
            "category": "knowledge",
            "difficulty": "easy",
            "tags": ["definition", "rag"],
        },
    ]
    for item in examples:
        ds.add(EvalCase(**item))
    return ds
