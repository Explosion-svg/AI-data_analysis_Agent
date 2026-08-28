"""
evaluation/benchmark_dataset.py — 基准测试数据集管理

职责：
  管理 Agent 评估用的基准测试用例集（Benchmark Dataset），包含：
  - 用户问题（question）
  - 预期 SQL（expected_sql，可选）
  - 预期答案（expected_answer，可选）
  - 预期工具列表（expected_tools，用于工具命中率评估）
  - 用例元数据（category、difficulty、tags）

数据集设计原则：
  - EvalCase 是最小化的测试单元，只包含"输入+预期"，不包含实际执行结果
  - 实际结果由 EvalRunner 在运行时计算并存储在 EvalResult 中
  - 数据集可以序列化为 JSON 文件，支持版本控制和跨环境共享

difficulty 分级（easy/medium/hard）：
  - easy：单工具，直接问答（如"今年总销售额"）
  - medium：2-3 个工具，需要 SQL + 可能有图表（如"月度趋势图"）
  - hard：4+ 步骤，复杂分析（如"同比增长率计算"）

category 分类：
  - "sql"：纯 SQL 查询类问题
  - "visualization"：需要生成图表
  - "analysis"：需要 Python 分析（计算指标）
  - "knowledge"：知识库检索类问题（RAG）

JSON 持久化：
  save() 和 load() 支持将数据集持久化为 JSON 文件，
  方便团队共享测试用例，也方便在 CI/CD 中运行回归评估。
  JSON 格式（非二进制）便于 git diff 追踪变更。

内置默认数据集（get_default_dataset）：
  包含 4 个典型测试用例，覆盖 sql、visualization、analysis、knowledge 四类场景，
  可作为快速健康检查（smoke test）使用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """
    单个评估测试用例（输入+预期，不含实际结果）。

    字段说明：
    - id：全局唯一标识（如 "sql_001"、"rag_001"），用于报告中定位用例
    - question：模拟用户输入的自然语言问题
    - expected_sql：预期的 SQL 语句（可选，用于 SQL 准确率评估）
    - expected_answer：预期的答案文本（可选，用于答案质量评估）
    - expected_tools：预期使用的工具列表（用于工具命中率评估）
      → 例如 ["get_schema", "sql_query", "generate_chart"]
    - category：用例分类（sql/visualization/analysis/knowledge）
    - difficulty：难度级别（easy/medium/hard）
    - tags：自由标签（如 ["sales", "aggregation", "yoy"]）

    为什么 expected_sql 和 expected_answer 都是可选的：
    - 有些问题没有"唯一正确答案"（如趋势分析，图表形式多样）
    - 工具命中率（expected_tools）是更稳定、更易验证的评估指标
    - 精确答案匹配通常需要人工标注或 LLM-as-judge，成本更高
    """
    id: str
    question: str
    expected_sql: str | None = None
    expected_answer: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"   # easy | medium | hard
    tags: list[str] = field(default_factory=list)


class BenchmarkDataset:
    """
    测试用例集管理器，支持增删查、过滤、JSON 持久化。

    内部使用 dict[str, EvalCase] 存储（key 为 case.id）：
    - O(1) 按 ID 查找
    - 自动去重（同 ID 的用例覆盖旧的）
    - 保留插入顺序（Python 3.7+ dict 有序）

    典型使用流程::
        ds = BenchmarkDataset()
        ds.add(EvalCase(id="q1", question="...", expected_tools=["sql_query"]))
        cases = ds.list(category="sql", difficulty="easy")
        runner = EvalRunner()
        report = await runner.run(dataset=ds)
    """

    def __init__(self) -> None:
        """初始化空的测试用例字典。"""
        self._cases: dict[str, EvalCase] = {}

    def add(self, case: EvalCase) -> None:
        """
        添加或覆盖测试用例。

        覆盖行为（同 ID 的用例）：
        - 直接覆盖，不抛异常（类似 dict 的赋值语义）
        - 适用于更新预期答案或修改用例属性

        Args:
            case: 要添加的 EvalCase 对象
        """
        self._cases[case.id] = case

    def get(self, case_id: str) -> EvalCase:
        """
        按 ID 获取测试用例。

        Args:
            case_id: 用例 ID（如 "sql_001"）

        Returns:
            对应的 EvalCase 对象

        Raises:
            KeyError: 用例 ID 不存在
        """
        return self._cases[case_id]

    def list(
        self,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> list[EvalCase]:
        """
        获取用例列表，支持按分类和难度过滤。

        过滤逻辑：
        - 两个过滤条件都是 AND 关系（同时满足）
        - None 表示不过滤该维度（等价于"全选"）

        使用示例::
            easy_sql = ds.list(category="sql", difficulty="easy")
            all_cases = ds.list()  # 不过滤，返回全部

        Args:
            category: 按分类过滤（None = 不过滤）
            difficulty: 按难度过滤（None = 不过滤）

        Returns:
            满足条件的 EvalCase 列表（按插入顺序）
        """
        cases = list(self._cases.values())
        if category:
            cases = [c for c in cases if c.category == category]
        if difficulty:
            cases = [c for c in cases if c.difficulty == difficulty]
        return cases

    def save(self, path: str | Path) -> None:
        """
        将数据集序列化为 JSON 文件。

        文件格式：JSON 数组，每个元素是一个用例对象。
        使用 ensure_ascii=False 支持中文问题（不转义为 \\uXXXX）。
        使用 indent=2 格式化输出（便于 git diff 比较变更）。

        path.parent.mkdir(parents=True, exist_ok=True)：
        - 自动创建不存在的父目录
        - exist_ok=True 防止目录已存在时报错

        Args:
            path: 保存路径（字符串或 Path 对象）
        """
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
        """
        从 JSON 文件加载数据集。

        使用 @classmethod 而不是 __init__：
        - 是一种"替代构造函数"（Alternative Constructor）
        - 语义清晰：BenchmarkDataset.load(path) 比 BenchmarkDataset(path) 更直观
        - 允许 __init__ 保持简单（只接受无参或轻量参数）

        EvalCase(**item) 要求 JSON 中的字段名与 EvalCase 的字段名完全一致，
        这是一个简单约定（不做额外映射），适合内部使用。

        Args:
            path: JSON 文件路径（字符串或 Path 对象）

        Returns:
            加载了所有测试用例的 BenchmarkDataset 实例

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        dataset = cls()
        data = json.loads(Path(path).read_text())
        for item in data:
            dataset.add(EvalCase(**item))
        return dataset

    def __len__(self) -> int:
        """
        返回数据集中的用例数量，支持 len(ds) 语法。

        Returns:
            用例总数
        """
        return len(self._cases)


def get_default_dataset() -> BenchmarkDataset:
    """
    获取内置示例测试集（快速健康检查用）。

    包含 4 个典型测试用例，覆盖 4 种场景：
    1. sql_001（easy）：简单 SQL 聚合查询（年度总销售额）
       → 预期工具：get_schema + sql_query
    2. sql_002（medium）：月度趋势可视化
       → 预期工具：get_schema + sql_query + generate_chart
    3. python_001（hard）：同比增长率计算（需要 Python 分析）
       → 预期工具：sql_query + python_analysis
    4. rag_001（easy）：业务指标定义查询（知识库检索）
       → 预期工具：search_documents

    这 4 个用例可用于验证 Agent 的基本功能是否正常（smoke test），
    也可作为创建自定义数据集的参考。

    扩展建议：
    - 按业务场景添加更多用例（财务分析、用户行为分析等）
    - 增加边界用例（空表、超大结果集、复杂多表 JOIN 等）
    - 为每个用例标注 expected_sql，用于 SQL 准确率评估

    Returns:
        包含 4 个默认测试用例的 BenchmarkDataset 实例
    """
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
