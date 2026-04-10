"""
evaluation/eval_runner.py — 评估运行器
批量运行 Agent，计算准确率、工具命中率、SQL 正确率
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ai_data_agent.evaluation.benchmark_dataset import BenchmarkDataset, EvalCase, get_default_dataset
from ai_data_agent.orchestration.agent_loop import AgentLoop, AgentResponse
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    case_id: str
    question: str
    response: AgentResponse | None = None
    error: str = ""
    tool_hit: bool = False          # 是否使用了预期工具
    latency_ms: float = 0.0


@dataclass
class EvalReport:
    total: int = 0
    success: int = 0
    tool_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_iterations: float = 0.0
    results: list[EvalResult] = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"\n{'='*50}")
        print(f"Evaluation Report ({self.total} cases)")
        print(f"{'='*50}")
        print(f"Success Rate   : {self.success}/{self.total} ({self.success/max(self.total,1)*100:.1f}%)")
        print(f"Tool Hit Rate  : {self.tool_hit_rate*100:.1f}%")
        print(f"Avg Latency    : {self.avg_latency_ms:.0f} ms")
        print(f"Avg Iterations : {self.avg_iterations:.1f}")
        print(f"{'='*50}\n")
        for r in self.results:
            status = "✓" if r.tool_hit else "✗"
            err = f"  ERROR: {r.error}" if r.error else ""
            print(f"  [{status}] {r.case_id}: {r.question[:60]}{err}")


class EvalRunner:
    """批量评估 Agent 质量。"""

    def __init__(self, concurrency: int = 3) -> None:
        self._concurrency = concurrency

    async def run(
        self,
        dataset: BenchmarkDataset | None = None,
        conversation_prefix: str = "eval",
    ) -> EvalReport:
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
        async with sem:
            agent = AgentLoop()
            start = time.perf_counter()
            result = EvalResult(case_id=case.id, question=case.question)
            try:
                resp = await agent.run(
                    query=case.question,
                    conversation_id=conversation_id,
                    use_cache=False,
                )
                result.response = resp
                result.latency_ms = (time.perf_counter() - start) * 1000

                # 工具命中率：实际使用的工具 ⊇ 预期工具
                if case.expected_tools:
                    used_tools = {tc["tool"] for tc in resp.tool_calls}
                    expected = set(case.expected_tools)
                    result.tool_hit = expected.issubset(used_tools)
                else:
                    result.tool_hit = resp.success

            except Exception as e:
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

        return EvalReport(
            total=total,
            success=success,
            tool_hit_rate=tool_hits / total,
            avg_latency_ms=avg_latency,
            avg_iterations=avg_iters,
            results=results,
        )
