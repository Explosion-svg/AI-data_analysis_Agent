#!/usr/bin/env python3
"""
run_load_test.py

最小压测入口。

支持两种模式：
- asgi: 直接在本进程内挂载 FastAPI app，并用假 Agent 压测并发控制链路
- http: 打已有服务地址，观察真实环境下的吞吐、延迟和过载返回

示例：
    python3 run_load_test.py --mode asgi --requests 200 --concurrency 50
    python3 run_load_test.py --mode asgi --requests 200 --concurrency 50 --fake-latency-ms 200
    python3 run_load_test.py --mode http --url http://127.0.0.1:8000 --requests 500 --concurrency 100
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class RequestResult:
    status_code: int
    latency_ms: float
    ok: bool
    body_preview: str = ""


class FakeAgent:
    def __init__(self, response_cls: type, latency_ms: int = 50) -> None:
        self._response_cls = response_cls
        self._latency_ms = latency_ms

    async def run(
        self,
        query: str,
        conversation_id: str,
        request_context=None,
        use_cache: bool = True,
    ) -> Any:
        await asyncio.sleep(self._latency_ms / 1000)
        return self._response_cls(
            answer=f"echo: {query}",
            conversation_id=conversation_id,
            iterations=1,
            tool_calls=[],
            charts=[],
            data=[],
            latency_ms=float(self._latency_ms),
            success=True,
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Data Agent minimal load tester")
    parser.add_argument("--mode", choices=["asgi", "http"], default="asgi")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--query", default="load test query")
    parser.add_argument("--fake-latency-ms", type=int, default=50)
    parser.add_argument("--conversation-prefix", default="load")
    return parser.parse_args(argv)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def one_request(
    client: AsyncClient,
    *,
    idx: int,
    query: str,
    conversation_prefix: str,
) -> RequestResult:
    payload = {
        "query": query,
        "conversation_id": f"{conversation_prefix}-{idx}",
        "use_cache": False,
    }
    start = time.perf_counter()
    try:
        resp = await client.post("/api/v1/chat", json=payload)
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(
            status_code=resp.status_code,
            latency_ms=elapsed,
            ok=200 <= resp.status_code < 300,
            body_preview=resp.text[:200],
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(
            status_code=0,
            latency_ms=elapsed,
            ok=False,
            body_preview=str(e)[:200],
        )


async def run_load(
    client: AsyncClient,
    *,
    total_requests: int,
    concurrency: int,
    query: str,
    conversation_prefix: str,
) -> list[RequestResult]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(idx: int) -> RequestResult:
        async with sem:
            return await one_request(
                client,
                idx=idx,
                query=query,
                conversation_prefix=conversation_prefix,
            )

    tasks = [guarded(i) for i in range(total_requests)]
    return await asyncio.gather(*tasks)


def print_report(results: list[RequestResult], wall_time_s: float) -> None:
    latencies = [item.latency_ms for item in results]
    codes = Counter(item.status_code for item in results)
    success = sum(1 for item in results if item.ok)
    failures = len(results) - success
    throughput = len(results) / wall_time_s if wall_time_s > 0 else 0.0

    print("")
    print("Load Test Report")
    print("================")
    print(f"requests       : {len(results)}")
    print(f"success        : {success}")
    print(f"failures       : {failures}")
    print(f"wall_time_s    : {wall_time_s:.2f}")
    print(f"throughput_rps : {throughput:.2f}")
    print(f"latency_avg_ms : {statistics.fmean(latencies):.2f}" if latencies else "latency_avg_ms : 0.00")
    print(f"latency_p50_ms : {percentile(latencies, 0.50):.2f}")
    print(f"latency_p95_ms : {percentile(latencies, 0.95):.2f}")
    print(f"latency_p99_ms : {percentile(latencies, 0.99):.2f}")
    print("status_counts  : " + ", ".join(f"{code}={count}" for code, count in sorted(codes.items())))

    failure_samples = [item for item in results if not item.ok][:3]
    if failure_samples:
        print("")
        print("Failure Samples")
        print("---------------")
        for sample in failure_samples:
            print(f"status={sample.status_code} latency_ms={sample.latency_ms:.2f} body={sample.body_preview}")


async def main_async(args: argparse.Namespace) -> int:
    try:
        from httpx import ASGITransport, AsyncClient
    except ModuleNotFoundError:
        print("缺少依赖 httpx。请先安装：")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        return 2

    try:
        from ai_data_agent.api import chat_api
        from ai_data_agent.config.config import settings
        from ai_data_agent.main import create_app
        from ai_data_agent.orchestration.agent_loop import AgentResponse
    except ModuleNotFoundError as exc:
        print(f"缺少项目运行依赖: {exc.name}")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        return 2

    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    if args.mode == "asgi":
        app = create_app()
        app.dependency_overrides[chat_api._get_agent_loop] = lambda: FakeAgent(AgentResponse, args.fake_latency_ms)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            timeout=args.timeout_seconds,
            headers=headers,
        ) as client:
            start = time.perf_counter()
            results = await run_load(
                client,
                total_requests=args.requests,
                concurrency=args.concurrency,
                query=args.query,
                conversation_prefix=args.conversation_prefix,
            )
            wall = time.perf_counter() - start
        app.dependency_overrides.clear()
        print_report(results, wall)
        return 0 if all(item.ok for item in results) else 1

    async with AsyncClient(
        base_url=args.url.rstrip("/"),
        timeout=args.timeout_seconds,
        headers=headers,
    ) as client:
        start = time.perf_counter()
        results = await run_load(
            client,
            total_requests=args.requests,
            concurrency=args.concurrency,
            query=args.query,
            conversation_prefix=args.conversation_prefix,
        )
        wall = time.perf_counter() - start
    print_report(results, wall)
    return 0 if all(item.ok for item in results) else 1


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
