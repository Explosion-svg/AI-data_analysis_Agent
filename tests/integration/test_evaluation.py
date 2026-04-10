"""
tests/integration/test_evaluation.py

评估运行器集成测试。

主要验证：
- EvalRunner.run 的聚合行为
- _run_case 的异常容错
- _compute_report 的统计逻辑
"""

from __future__ import annotations

import asyncio

import pytest

from ai_data_agent.evaluation.benchmark_dataset import BenchmarkDataset, EvalCase
from ai_data_agent.evaluation.eval_runner import EvalResult, EvalRunner
from ai_data_agent.orchestration.agent_loop import AgentResponse


@pytest.mark.asyncio
async def test_eval_runner_run_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    # 用假的 AgentLoop 替换真实 Agent，验证评估聚合逻辑是否正确。
    dataset = BenchmarkDataset()
    dataset.add(EvalCase(id="case1", question="q1", expected_tools=["sql_query"]))
    dataset.add(EvalCase(id="case2", question="q2", expected_tools=[]))

    class FakeAgentLoop:
        async def run(self, query: str, conversation_id: str, use_cache: bool = False) -> AgentResponse:
            if query == "q1":
                return AgentResponse(
                    answer="a1",
                    conversation_id=conversation_id,
                    iterations=2,
                    tool_calls=[{"tool": "sql_query", "args": {}, "success": True}],
                    success=True,
                )
            return AgentResponse(
                answer="a2",
                conversation_id=conversation_id,
                iterations=1,
                tool_calls=[],
                success=True,
            )

    monkeypatch.setattr("ai_data_agent.evaluation.eval_runner.AgentLoop", FakeAgentLoop)
    report = await EvalRunner(concurrency=2).run(dataset=dataset, conversation_prefix="t")

    assert report.total == 2
    assert report.success == 2
    assert report.tool_hit_rate == 1.0
    assert report.avg_iterations == 1.5
    assert len(report.results) == 2


@pytest.mark.asyncio
async def test_eval_runner_run_case_handles_agent_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    # 单个 case 失败时，不应让整个评估直接崩溃，而是写入 error 字段。
    class FakeAgentLoop:
        async def run(self, query: str, conversation_id: str, use_cache: bool = False) -> AgentResponse:
            raise RuntimeError("agent crashed")

    monkeypatch.setattr("ai_data_agent.evaluation.eval_runner.AgentLoop", FakeAgentLoop)
    case = EvalCase(id="case1", question="q1")
    result = await EvalRunner()._run_case(case, "conv_1", asyncio.Semaphore(1))

    assert result.case_id == "case1"
    assert "agent crashed" in result.error
    assert result.response is None


def test_eval_runner_compute_report_empty_and_non_empty() -> None:
    # _compute_report 是纯聚合逻辑，适合直接验证边界值与统计值。
    empty = EvalRunner._compute_report([])
    assert empty.total == 0

    non_empty = EvalRunner._compute_report(
        [
            EvalResult(
                case_id="a",
                question="q1",
                response=AgentResponse(answer="a", conversation_id="c1", iterations=2, success=True),
                tool_hit=True,
                latency_ms=100,
            ),
            EvalResult(
                case_id="b",
                question="q2",
                response=AgentResponse(answer="b", conversation_id="c2", iterations=4, success=False),
                tool_hit=False,
                latency_ms=300,
            ),
        ]
    )

    assert non_empty.total == 2
    assert non_empty.success == 1
    assert non_empty.tool_hit_rate == 0.5
    assert non_empty.avg_latency_ms == 200
    assert non_empty.avg_iterations == 3
