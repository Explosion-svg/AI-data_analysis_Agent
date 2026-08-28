"""
evaluation/eval_runner.py — 评估运行器（Batch Evaluation Runner）

职责：
  批量运行 Agent，对预定义的测试用例集（BenchmarkDataset）进行自动化评估，
  计算关键质量指标：
  - 成功率（success rate）：Agent 是否成功返回答案
  - 工具命中率（tool hit rate）：是否使用了预期的工具
  - 平均延迟（avg latency）：端到端响应时间
  - 平均迭代次数（avg iterations）：ReAct 循环的效率

评估策略（工具命中率）：
  tool_hit 的判断逻辑：
  - 如果 case.expected_tools 非空：expected_tools ⊆ used_tools（所有预期工具都被使用）
  - 如果 case.expected_tools 为空：tool_hit = resp.success（只要成功返回就算命中）

  "⊆" 而不是 "=" 的原因：
  - Agent 可能额外使用了一些辅助工具（如 get_schema），这是合理的
  - 只要核心工具（如 sql_query、generate_chart）被使用，就算命中
  - 使用 "=" 会因为辅助工具而误报命中失败

并发控制：
  EvalRunner 使用 asyncio.Semaphore 限制并发评估用例数（默认 3），
  原因：
  - 每个用例都会创建一个 AgentLoop 并发起 LLM 调用
  - 过多并发会触发 LLM API rate limit
  - 3 是在速度和 API 限制之间的平衡点

use_cache=False：
  评估时强制禁用缓存，原因：
  - 缓存会导致相同问题每次返回相同答案，无法评估真实性能
  - 评估需要每次都走完整的 Agent 处理流程

离线模式（P4-5）：
  EvalRunner 支持通过 agent_factory 注入假 Agent（默认从容器取真实 AgentLoop）。
  注入假 Agent 后可以不调用任何真实 LLM 就能跑通评估链路——
  这是把评估接入 CI 的前提（真实 LLM 评估必须手动触发，不能进 CI）。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_data_agent.assembler import get_container
from ai_data_agent.evaluation.benchmark_dataset import BenchmarkDataset, EvalCase, get_default_dataset
from ai_data_agent.orchestration.agent_loop import AgentLoop, AgentResponse
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    """
    单个评估用例的执行结果。

    字段说明：
    - case_id：对应 EvalCase.id，用于在报告中定位用例
    - question：用户问题（冗余存储，方便报告打印，不用二次查询）
    - response：Agent 的完整响应（None 表示执行异常）
    - error：如果执行过程中抛出异常，记录异常信息
    - tool_hit：工具命中率评估结果（见 EvalRunner._run_case 的判断逻辑）
    - sql_hit：SQL 准确率评估结果（case.expected_sql 提供时评估，P4-5）
    - answer_hit：答案质量评估结果（case.expected_answer 提供时评估，P4-5）
    - sql_evaluated：本用例是否声明了 expected_sql（决定是否计入 SQL 命中率分母）
    - answer_evaluated：本用例是否声明了 expected_answer（决定是否计入答案命中率分母）
    - latency_ms：端到端延迟（包含等待 Semaphore 的时间）
    """
    case_id: str
    question: str
    response: AgentResponse | None = None
    error: str = ""
    tool_hit: bool = False          # 是否使用了预期工具
    sql_hit: bool = False           # 生成的 SQL 是否命中预期 SQL（归一化精确比较）
    answer_hit: bool = False        # 答案是否命中预期答案（归一化子串匹配）
    sql_evaluated: bool = False     # 是否评估了 SQL 命中
    answer_evaluated: bool = False  # 是否评估了答案命中
    latency_ms: float = 0.0


@dataclass
class EvalReport:
    """
    完整评估报告的汇总数据。

    字段说明：
    - total：测试用例总数
    - success：成功的用例数（response.success=True）
    - tool_hit_rate：工具命中率（tool_hit=True 的比例）
    - sql_hit_rate：SQL 准确率（sql_hit=True 的比例，仅统计提供了 expected_sql 的用例，P4-5）
    - answer_hit_rate：答案命中率（answer_hit=True 的比例，仅统计提供了 expected_answer 的用例，P4-5）
    - avg_latency_ms：所有用例的平均延迟（包含失败的用例）
    - avg_iterations：成功用例的平均 ReAct 迭代次数
    - results：每个用例的详细结果列表（用于逐用例分析）
    """
    total: int = 0
    success: int = 0
    tool_hit_rate: float = 0.0
    sql_hit_rate: float = 0.0
    answer_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_iterations: float = 0.0
    results: list[EvalResult] = field(default_factory=list)

    def print_summary(self) -> None:
        """
        打印格式化的评估报告到标准输出。

        报告格式：
        ==================================================
        Evaluation Report (N cases)
        ==================================================
        Success Rate   : X/N (XX.X%)
        Tool Hit Rate  : XX.X%
        SQL Hit Rate   : XX.X% (only cases with expected_sql)
        Answer Hit Rate: XX.X% (only cases with expected_answer)
        Avg Latency    : XXXX ms
        Avg Iterations : X.X
        ==================================================
          [✓] case_001: 今年总销售额是多少？
          [✗] case_002: 各产品类别的月度销售趋势  ERROR: LLM timeout

        ✓/✗ 标记基于 tool_hit（工具命中率），而不是 success：
        - 使用预期工具 = ✓（Agent 走了正确路径）
        - 没用预期工具 = ✗（Agent 可能用了错误策略）
        """
        print(f"\n{'='*50}")
        print(f"Evaluation Report ({self.total} cases)")
        print(f"{'='*50}")
        print(f"Success Rate   : {self.success}/{self.total} ({self.success/max(self.total,1)*100:.1f}%)")
        print(f"Tool Hit Rate  : {self.tool_hit_rate*100:.1f}%")
        print(f"SQL Hit Rate   : {self.sql_hit_rate*100:.1f}%")
        print(f"Answer Hit Rate: {self.answer_hit_rate*100:.1f}%")
        print(f"Avg Latency    : {self.avg_latency_ms:.0f} ms")
        print(f"Avg Iterations : {self.avg_iterations:.1f}")
        print(f"{'='*50}\n")
        for r in self.results:
            status = "✓" if r.tool_hit else "✗"
            err = f"  ERROR: {r.error}" if r.error else ""
            print(f"  [{status}] {r.case_id}: {r.question[:60]}{err}")


class EvalRunner:
    """
    批量评估 Agent 质量的运行器。

    并发设计：
    - 所有测试用例同时提交（asyncio.gather），受 Semaphore 限制并发数
    - 一个用例失败不影响其他用例（gather 不取消其他任务）

    使用方式::
        runner = EvalRunner(concurrency=3)
        report = await runner.run(dataset=get_default_dataset())
        report.print_summary()
    """

    def __init__(
        self,
        concurrency: int = 3,
        agent_factory: Callable[[], Any] | None = None,
    ) -> None:
        """
        初始化评估运行器。

        Args:
            concurrency: 同时运行的最大测试用例数（默认 3，防止 LLM API rate limit）
            agent_factory: AgentLoop 工厂函数（P4-5）。默认从全局容器取真实
                AgentLoop（get_container().get_agent_loop()，是容器单例）。
                测试/离线模式可注入假 Agent 工厂，避免打真实 LLM——
                这是评估能进入 CI 的前提。
        """
        self._concurrency = concurrency
        self._agent_factory = agent_factory or (lambda: get_container().get_agent_loop())

    async def run(
        self,
        dataset: BenchmarkDataset | None = None,
        conversation_prefix: str = "eval",
    ) -> EvalReport:
        """
        运行所有测试用例，返回评估报告。

        执行流程：
        1. 获取数据集（None 时使用内置默认数据集）
        2. 创建所有用例的 coroutine 任务列表
        3. asyncio.gather 并发执行（受 Semaphore 限制）
        4. 汇总计算评估报告

        conversation_prefix 的作用：
        - 每个用例的对话 ID = {prefix}_{case_id}
        - 不同用例使用不同对话 ID，确保状态隔离（不共享对话历史）
        - 使用前缀避免与正式对话 ID 冲突（eval_sql_001 vs real_user_xxx）

        Args:
            dataset: 测试数据集（None 使用内置默认数据集）
            conversation_prefix: 对话 ID 前缀（默认 "eval"）

        Returns:
            EvalReport：包含所有评估指标和详细结果
        """
        ds = dataset or get_default_dataset()
        cases = ds.list()

        logger.info("eval_runner.start", total=len(cases))

        sem = asyncio.Semaphore(self._concurrency)
        tasks = [
            self._run_case(case, f"{conversation_prefix}_{case.id}", sem)
            for case in cases
        ]
        results: list[EvalResult] = await asyncio.gather(*tasks)

        return self._compute_report(results)

    async def _run_case(
        self,
        case: EvalCase,
        conversation_id: str,
        sem: asyncio.Semaphore,
    ) -> EvalResult:
        """
        执行单个测试用例，在 Semaphore 保护的并发槽内运行。

        异常处理策略：
        - 捕获所有异常并记录到 EvalResult.error，不让异常传播
        - 这样单个用例失败不会影响其他用例的执行
        - gather 仍能收集到所有用例的结果

        工具命中率判断逻辑：
        - case.expected_tools 非空：
            used_tools = {tc["tool"] for tc in resp.tool_calls}（所有实际使用的工具）
            result.tool_hit = expected.issubset(used_tools)（预期工具 ⊆ 实际工具）
        - case.expected_tools 为空：
            result.tool_hit = resp.success（只判断是否成功）

        latency_ms 计算范围：
        - 从 async with sem 进入后开始（包含等待 sem 的时间不计入）
        - 实际是从 agent.run() 调用开始计时，到返回为止
        - 包含了 LLM 调用、工具执行的全部时间

        Args:
            case: 测试用例
            conversation_id: 独立对话 ID（隔离不同用例的状态）
            sem: 并发控制 Semaphore

        Returns:
            EvalResult：用例执行结果（包含成功/失败信息）
        """
        async with sem:
            # P4-5：Agent 构造移入 try 块——即使 get_agent_loop()/agent_factory()
            # 抛出异常，也只影响当前用例（写入 error），不再中断整个评估批次。
            # 注意：默认工厂取的是容器单例（不是"每用例独立实例"），用例间隔离
            # 依赖独立的 conversation_id，而不是独立的 AgentLoop 实例。
            result = EvalResult(case_id=case.id, question=case.question)
            try:
                agent = self._agent_factory()
                start = time.perf_counter()
                resp = await agent.run(
                    query=case.question,
                    conversation_id=conversation_id,
                    use_cache=False,  # 评估必须禁用缓存，确保每次走完整流程
                )
                result.response = resp
                result.latency_ms = (time.perf_counter() - start) * 1000
                result.sql_evaluated = bool(case.expected_sql)
                result.answer_evaluated = bool(case.expected_answer)

                # 工具命中率：实际使用的工具 ⊇ 预期工具（超集）
                if case.expected_tools:
                    used_tools = {tc["tool"] for tc in resp.tool_calls}
                    expected = set(case.expected_tools)
                    result.tool_hit = expected.issubset(used_tools)
                else:
                    # 无预期工具时，用成功状态替代
                    result.tool_hit = resp.success

                # SQL 准确率（P4-5）：提供 expected_sql 时评估。
                # 从 tool_calls 中提取 sql_query 的 sql 参数，与预期做归一化精确比较。
                if case.expected_sql:
                    result.sql_hit = self._sql_matches(
                        self._extract_sql(resp),
                        case.expected_sql,
                    )

                # 答案质量（P4-5）：提供 expected_answer 时评估。
                # 归一化后做子串包含匹配（宽松启发式，避免标点/大小写干扰）。
                if case.expected_answer:
                    result.answer_hit = self._answer_matches(
                        resp.answer,
                        case.expected_answer,
                    )

            except Exception as e:
                # 异常不传播：记录错误，保证其他用例不受影响
                result.error = str(e)
                result.latency_ms = (time.perf_counter() - start) * 1000
                logger.error("eval_runner.case_failed", case_id=case.id, error=str(e))

            logger.debug(
                "eval_runner.case_done",
                case_id=case.id,
                tool_hit=result.tool_hit,
                latency_ms=round(result.latency_ms),
            )
            return result

    @staticmethod
    def _compute_report(results: list[EvalResult]) -> EvalReport:
        """
        从所有用例结果计算汇总评估报告。

        各指标计算逻辑：
        - success：response 非 None 且 response.success=True 的用例数
        - tool_hit_rate：tool_hit=True 的比例（不管是否成功，命中率独立统计）
        - sql_hit_rate：仅统计提供了 expected_sql 的用例，sql_hit=True 的比例（P4-5）
        - answer_hit_rate：仅统计提供了 expected_answer 的用例，answer_hit=True 的比例（P4-5）
        - avg_latency：所有用例（包含失败的）的平均延迟
        - avg_iterations：只统计有 response 的用例的平均迭代次数
          （失败用例没有 iterations 数据，不参与平均计算）

        除零保护：
        - max(self.total, 1)：防止 total=0 时除零
        - max(sum(1 for r in results if r.response), 1)：防止无有效 response 时除零

        Args:
            results: 所有测试用例的执行结果列表

        Returns:
            EvalReport：汇总评估报告
        """
        total = len(results)
        if total == 0:
            return EvalReport()

        success = sum(1 for r in results if r.response and r.response.success)
        tool_hits = sum(1 for r in results if r.tool_hit)
        avg_latency = sum(r.latency_ms for r in results) / total
        avg_iters = (
            sum(r.response.iterations for r in results if r.response) /
            max(sum(1 for r in results if r.response), 1)
        )

        # P4-5：SQL/答案命中率只对声明了对应预期的用例求分母，
        # 未声明的用例（sql_evaluated/answer_evaluated=False）不稀释命中率。
        sql_cases = [r for r in results if r.sql_evaluated]
        answer_cases = [r for r in results if r.answer_evaluated]
        sql_hit_rate = (
            sum(1 for r in sql_cases if r.sql_hit) / len(sql_cases) if sql_cases else 0.0
        )
        answer_hit_rate = (
            sum(1 for r in answer_cases if r.answer_hit) / len(answer_cases)
            if answer_cases else 0.0
        )

        return EvalReport(
            total=total,
            success=success,
            tool_hit_rate=tool_hits / total,
            sql_hit_rate=sql_hit_rate,
            answer_hit_rate=answer_hit_rate,
            avg_latency_ms=avg_latency,
            avg_iterations=avg_iters,
            results=results,
        )

    # ── P4-5：SQL/答案匹配辅助函数 ──────────────────────────────────────────

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """SQL 归一化：小写 + 空白折叠（仅用于等价性粗比较，不做语义解析）。"""
        return " ".join(sql.lower().split())

    @staticmethod
    def _extract_sql(resp: AgentResponse) -> str | None:
        """从 Agent 响应的工具调用中提取第一条 sql_query 的 sql 参数。"""
        for tc in resp.tool_calls:
            if tc.get("tool") == "sql_query":
                args = tc.get("args") or {}
                sql = args.get("sql")
                if isinstance(sql, str) and sql.strip():
                    return sql
        return None

    @staticmethod
    def _sql_matches(used_sql: str | None, expected_sql: str) -> bool:
        """归一化精确比较生成的 SQL 与预期 SQL。"""
        if not used_sql:
            return False
        return EvalRunner._normalize_sql(used_sql) == EvalRunner._normalize_sql(expected_sql)

    @staticmethod
    def _answer_matches(answer: str, expected_answer: str) -> bool:
        """归一化后判断预期答案是否为实际答案的子串（宽松启发式）。"""
        a = " ".join(answer.lower().split())
        e = " ".join(expected_answer.lower().split())
        return bool(e) and e in a
