"""
tests/unit/test_zero_coverage.py

P4-4：零测试覆盖区补充测试。

此前完全没有行为型测试的区域，按 REVIEW_FINDINGS.md P4-4 逐项补齐：
- 真实 OpenAI 适配器（openai_model.py）：消息序列化、generate/stream/embed、
  错误映射（RateLimit/Timeout/APIError）、指标记录、健康检查、连接关闭
- observability 层：logger 幂等配置、metrics 计数、tracer NoOp/关闭
- rag_tool / schema_tool
- work_memory_summarizer
- request_context（租户隔离 / scoped key / ContextVar）

所有测试均为离线（不触碰真实网络/Redis/Chroma），用 monkeypatch 与假对象替代外部依赖。
"""

from __future__ import annotations

import httpx
import pandas as pd
import pytest
from openai import APIError, APITimeoutError, RateLimitError

from ai_data_agent.context.request_context import (
    RequestContext,
    clear_request_context,
    get_request_context,
    set_request_context,
)
from ai_data_agent.memory.work_memory_summarizer import WorkMemorySummarizer
from ai_data_agent.model_gateway.base_model import LLMConfig, Message
from ai_data_agent.model_gateway.openai_model import OpenAIModel, _to_openai_messages
from ai_data_agent.tools.base_tool import ToolResult
from ai_data_agent.tools.rag_tool import RAGTool
from ai_data_agent.tools.schema_tool import SchemaTool


# ── 假 OpenAI 客户端 ──────────────────────────────────────────────────────────


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list[_FakeToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage, finish_reason: str = "stop") -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20, total_tokens: int = 30) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeResponse:
    def __init__(self, choices: list[_FakeChoice], model: str = "fake-model", usage: _FakeUsage | None = None) -> None:
        self.choices = choices
        self.model = model
        self.usage = usage


class _FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeStreamChoice:
    def __init__(self, content: str) -> None:
        self.delta = _FakeDelta(content)


class _FakeStreamChunk:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeStreamChoice(content)]


class _FakeStream:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = [_FakeStreamChunk(c) for c in chunks]

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def __aiter__(self) -> "_FakeStream":
        self._it = iter(self._chunks)
        return self

    async def __anext__(self) -> _FakeStreamChunk:
        try:
            return next(self._it)  # type: ignore[arg-type]
        except StopIteration:
            raise StopAsyncIteration


class _FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingItem(v) for v in vectors]


class _FakeEmbeddings:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self._vectors = vectors or [[0.1, 0.2, 0.3]]
        self.calls: list[str] = []

    async def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
        self.calls = list(kwargs.get("input", []))  # type: ignore[arg-type]
        return _FakeEmbeddingResponse(self._vectors)


class _FakeModels:
    def __init__(self, ok: bool = True) -> None:
        self._ok = ok

    async def list(self) -> None:
        if not self._ok:
            raise RuntimeError("models endpoint down")


class _FakeCompletions:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[dict] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self._handler(kwargs)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions_handler, embeddings: _FakeEmbeddings | None = None, models: _FakeModels | None = None) -> None:
        self.chat = _FakeChat(_FakeCompletions(completions_handler))
        self.embeddings = embeddings or _FakeEmbeddings()
        self.models = models or _FakeModels()
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _build_model(client: _FakeClient) -> OpenAIModel:
    """构造 OpenAIModel 实例并替换其底层客户端为假对象（不触网）。"""
    model = OpenAIModel(api_key="test-key", api_base="http://localhost", model="gpt-4o-mini")
    model._client = client  # type: ignore[assignment]
    return model


# ── request_context ───────────────────────────────────────────────────────────


def test_request_context_valid_construction() -> None:
    ctx = RequestContext("r1", "u1", "t1")
    assert ctx.request_id == "r1"
    assert ctx.user_id == "u1"
    assert ctx.tenant_id == "t1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "a:b"),       # 冒号分隔符：可冒充面（P3-7）
        ("tenant_id", "a/b"),       # 斜杠
        ("tenant_id", "a b"),       # 空格
        ("tenant_id", "中文"),      # 非 ASCII
        ("tenant_id", "a\nb"),      # 控制字符：日志注入
        ("tenant_id", ""),          # 空
        ("tenant_id", "x" * 65),    # 超长
        ("user_id", "a:b"),
        ("user_id", ""),
        ("user_id", "x" * 65),
    ],
)
def test_request_context_rejects_unsafe_ident(field: str, value: str) -> None:
    kwargs = {"request_id": "r1", "user_id": "u1", "tenant_id": "t1"}
    kwargs[field] = value
    with pytest.raises(ValueError):
        RequestContext(**kwargs)


def test_request_context_scoped_conversation_id_prefixes_tenant() -> None:
    ctx = RequestContext("r1", "u1", "t1")
    assert ctx.scoped_conversation_id("conv-123") == "t1:conv-123"


def test_request_context_scoped_rejects_oversized_conversation_id() -> None:
    ctx = RequestContext("r1", "u1", "t1")
    with pytest.raises(ValueError):
        ctx.scoped_conversation_id("x" * 257)  # P4-7：长度上限


def test_request_context_scoped_keys_cannot_collide_across_tenants() -> None:
    # P3-7：tenant_id 字符集不含冒号，拼接键不可能发生三方碰撞。
    # 旧实现 "a:b"+"c" 与 "a"+"b:c" 碰撞的场景在字符集限制下不再存在。
    a = RequestContext("r", "u", "tenant")
    b = RequestContext("r", "u", "tenant-a")
    assert a.scoped_conversation_id("b:c") != b.scoped_conversation_id("c")


def test_request_context_set_get_clear_with_token() -> None:
    ctx = RequestContext("r1", "u1", "t1")
    token = set_request_context(ctx)
    assert get_request_context() == ctx
    clear_request_context(token)
    assert get_request_context() is None


def test_request_context_set_get_clear_without_token() -> None:
    ctx = RequestContext("r1", "u1", "t1")
    set_request_context(ctx)
    assert get_request_context() == ctx
    clear_request_context()
    assert get_request_context() is None


# ── work_memory_summarizer ────────────────────────────────────────────────────


def test_summarize_rows_empty() -> None:
    assert WorkMemorySummarizer.summarize_rows([]) == "SQL returned 0 rows."


def test_summarize_rows_with_data() -> None:
    text = WorkMemorySummarizer.summarize_rows(
        [{"date": "2026-01", "amount": 100}, {"date": "2026-02", "amount": 200}]
    )
    assert "2 row(s)" in text
    assert "date" in text
    assert "amount" in text
    assert "2026-01" in text  # 首行预览


def test_summarize_tool_result_none_tool_result() -> None:
    text = WorkMemorySummarizer.summarize_tool_result("sql_query", {}, None, "obs")
    assert "failed before producing a ToolResult" in text


def test_summarize_tool_result_failed() -> None:
    tr = ToolResult(success=False, error="boom")
    text = WorkMemorySummarizer.summarize_tool_result("sql_query", {}, tr, "obs")
    assert "boom" in text


def test_summarize_tool_result_sql_query_includes_sql_and_rows() -> None:
    tr = ToolResult(success=True, data=[{"a": 1}, {"a": 2}], text="...")
    text = WorkMemorySummarizer.summarize_tool_result(
        "sql_query", {"sql": "SELECT * FROM t"}, tr, "obs"
    )
    assert "rows=2" in text
    assert "SELECT * FROM t" in text


def test_summarize_tool_result_chart() -> None:
    tr = ToolResult(success=True, text="...")
    assert (
        WorkMemorySummarizer.summarize_tool_result("generate_chart", {}, tr, "obs")
        == "Chart generated successfully."
    )


def test_summarize_tool_result_other_tool_uses_text() -> None:
    tr = ToolResult(success=True, text="some result text")
    text = WorkMemorySummarizer.summarize_tool_result("search_documents", {}, tr, "obs")
    assert "some result text" in text


# ── rag_tool ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rag_tool_rejects_empty_query() -> None:
    result = await RAGTool().run(query="   ")
    assert result.success is False
    assert "Empty query" in result.error


@pytest.mark.asyncio
async def test_rag_tool_embedding_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenRouter:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding down")

    monkeypatch.setattr("ai_data_agent.tools.rag_tool.get_router", lambda: BrokenRouter())
    result = await RAGTool().run(query="hello")
    assert result.success is False
    assert "Embedding failed" in result.error


@pytest.mark.asyncio
async def test_rag_tool_vector_search_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubRouter:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2]]

    def broken_search(*args: object, **kwargs: object) -> object:
        raise RuntimeError("chroma down")

    monkeypatch.setattr("ai_data_agent.tools.rag_tool.get_router", lambda: StubRouter())
    monkeypatch.setattr("ai_data_agent.tools.rag_tool.vector_store.search_docs", broken_search)
    result = await RAGTool().run(query="hello")
    assert result.success is False
    assert "Vector search failed" in result.error


@pytest.mark.asyncio
async def test_rag_tool_no_docs_after_score_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubRouter:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2]]

    def search(*args: object, **kwargs: object) -> list[dict]:
        # 分数低于阈值，过滤后为空 → 正常返回空结果（不是错误）
        return [{"content": "low score doc", "score": 0.2, "metadata": {"source": "kb"}}]

    monkeypatch.setattr("ai_data_agent.tools.rag_tool.get_router", lambda: StubRouter())
    monkeypatch.setattr("ai_data_agent.tools.rag_tool.vector_store.search_docs", search)
    result = await RAGTool().run(query="hello", score_threshold=0.5)
    assert result.success is True
    assert result.data == []
    assert "No relevant documents found." in result.text


@pytest.mark.asyncio
async def test_rag_tool_filters_and_formats_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubRouter:
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2]]

    docs = [
        {"content": "doc a", "score": 0.8, "metadata": {"source": "kb1"}},
        {"content": "doc b", "score": 0.2, "metadata": {"source": "kb2"}},
    ]

    monkeypatch.setattr("ai_data_agent.tools.rag_tool.get_router", lambda: StubRouter())
    monkeypatch.setattr("ai_data_agent.tools.rag_tool.vector_store.search_docs", lambda *a, **k: docs)
    result = await RAGTool().run(query="hello", top_k=5, score_threshold=0.5)
    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["content"] == "doc a"
    assert "doc a" in result.text
    assert "doc b" not in result.text


# ── schema_tool ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_schema_tool_list_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    async def get_table_names() -> list[str]:
        return ["orders", "users"]

    monkeypatch.setattr("ai_data_agent.tools.schema_tool.warehouse.get_table_names", get_table_names)
    result = await SchemaTool().run(action="list_tables")
    assert result.success is True
    assert result.data == ["orders", "users"]
    assert "orders" in result.text


@pytest.mark.asyncio
async def test_schema_tool_describe_table_requires_name() -> None:
    result = await SchemaTool().run(action="describe_table")
    assert result.success is False
    assert "table_name is required" in result.error


@pytest.mark.asyncio
async def test_schema_tool_describe_table(monkeypatch: pytest.MonkeyPatch) -> None:
    async def get_table_schema(table_name: str) -> list[dict]:
        return [{"name": "id", "type": "INTEGER", "nullable": False}]

    monkeypatch.setattr("ai_data_agent.tools.schema_tool.warehouse.get_table_schema", get_table_schema)
    result = await SchemaTool().run(action="describe_table", table_name="users")
    assert result.success is True
    assert result.data[0]["name"] == "id"
    assert "id (INTEGER) NOT NULL" in result.text


@pytest.mark.asyncio
async def test_schema_tool_sample_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame([{"id": 1, "amount": 100}])

    async def get_sample_rows(table_name: str, n: int) -> pd.DataFrame:
        return df

    monkeypatch.setattr("ai_data_agent.tools.schema_tool.warehouse.get_sample_rows", get_sample_rows)
    result = await SchemaTool().run(action="sample_rows", table_name="orders", n_samples=3)
    assert result.success is True
    assert result.data == [{"id": 1, "amount": 100}]
    assert "orders" in result.text


@pytest.mark.asyncio
async def test_schema_tool_unknown_action() -> None:
    result = await SchemaTool().run(action="bogus")
    assert result.success is False
    assert "Unknown action" in result.error


# ── openai_model：消息序列化 ───────────────────────────────────────────────────


def test_to_openai_messages_omits_none_optional_fields() -> None:
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="", tool_calls=[{"id": "1"}]),
        Message(role="tool", content="res", name="sql_query", tool_call_id="1"),
    ]
    out = _to_openai_messages(msgs)
    # 只保留 role/content 与已设置的可选字段（None 字段不序列化，避免 400）
    assert out[0] == {"role": "system", "content": "sys"}
    assert set(out[1]) == {"role", "content"}
    assert out[2]["tool_calls"] == [{"id": "1"}]
    assert out[3]["name"] == "sql_query"
    assert out[3]["tool_call_id"] == "1"


# ── openai_model：generate ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_model_generate_maps_tool_calls() -> None:
    def handler(kwargs: dict) -> _FakeResponse:
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["max_tokens"] == 2048
        tc = _FakeToolCall("call_1", "sql_query", '{"sql": "SELECT 1"}')
        return _FakeResponse(
            [_FakeChoice(_FakeMessage(content=None, tool_calls=[tc]), finish_reason="tool_calls")],
            usage=_FakeUsage(),
        )

    client = _FakeClient(handler)
    model = _build_model(client)
    resp = await model.generate(
        [Message(role="user", content="hi")],
        LLMConfig(model="gpt-4o-mini", max_tokens=2048),
    )
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls[0]["id"] == "call_1"
    assert resp.tool_calls[0]["function"]["name"] == "sql_query"
    assert resp.tool_calls[0]["function"]["arguments"] == '{"sql": "SELECT 1"}'
    assert resp.total_tokens == 30


@pytest.mark.asyncio
async def test_openai_model_generate_returns_text_and_records_metrics() -> None:
    from prometheus_client import REGISTRY

    def handler(kwargs: dict) -> _FakeResponse:
        return _FakeResponse(
            [_FakeChoice(_FakeMessage(content="hello"))],
            model="gpt-4o-mini",
            usage=_FakeUsage(1, 2, 3),
        )

    client = _FakeClient(handler)
    model = _build_model(client)
    before = REGISTRY.get_sample_value("llm_tokens_total", {"model": "gpt-4o-mini"}) or 0.0
    resp = await model.generate(
        [Message(role="user", content="hi")], LLMConfig(model="gpt-4o-mini")
    )
    after = REGISTRY.get_sample_value("llm_tokens_total", {"model": "gpt-4o-mini"}) or 0.0
    assert resp.content == "hello"
    assert resp.model == "gpt-4o-mini"
    assert resp.total_tokens == 3
    assert after - before == 3  # P3-11：token 指标真实累计


@pytest.mark.asyncio
async def test_openai_model_generate_omits_optional_kwargs_when_unset() -> None:
    captured: dict = {}

    def handler(kwargs: dict) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse([_FakeChoice(_FakeMessage(content="ok"))])

    client = _FakeClient(handler)
    model = _build_model(client)
    await model.generate([Message(role="user", content="hi")], LLMConfig(model="gpt-4o-mini"))
    # stop/tools/tool_choice 未配置时不传给 API（None 字段会 400）
    assert "stop" not in captured
    assert "tools" not in captured
    assert "tool_choice" not in captured


@pytest.mark.asyncio
async def test_openai_model_generate_rate_limit_propagates() -> None:
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")

    def raise_rate_limit(kwargs: dict) -> object:
        raise RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)

    model = _build_model(_FakeClient(raise_rate_limit))
    with pytest.raises(RateLimitError):
        await model.generate([Message(role="user", content="hi")], LLMConfig(model="gpt-4o-mini"))


@pytest.mark.asyncio
async def test_openai_model_generate_timeout_propagates() -> None:
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")

    def raise_timeout(kwargs: dict) -> object:
        raise APITimeoutError(request=request)

    model = _build_model(_FakeClient(raise_timeout))
    with pytest.raises(APITimeoutError):
        await model.generate([Message(role="user", content="hi")], LLMConfig(model="gpt-4o-mini"))


@pytest.mark.asyncio
async def test_openai_model_generate_api_error_propagates() -> None:
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")

    def raise_api_error(kwargs: dict) -> object:
        raise APIError("bad request", request=request, body=None)

    model = _build_model(_FakeClient(raise_api_error))
    with pytest.raises(APIError):
        await model.generate([Message(role="user", content="hi")], LLMConfig(model="gpt-4o-mini"))


# ── openai_model：stream / embed / health / close ─────────────────────────────


@pytest.mark.asyncio
async def test_openai_model_stream_yields_content_chunks() -> None:
    def handler(kwargs: dict) -> _FakeStream:
        assert kwargs.get("stream") is True
        return _FakeStream(["你", "好", "！"])

    model = _build_model(_FakeClient(handler))
    chunks = [
        c
        async for c in model.stream(
            [Message(role="user", content="hi")], LLMConfig(model="gpt-4o-mini")
        )
    ]
    assert chunks == ["你", "好", "！"]


@pytest.mark.asyncio
async def test_openai_model_embed_returns_vectors() -> None:
    model = _build_model(
        _FakeClient(lambda kwargs: _FakeResponse([]), embeddings=_FakeEmbeddings([[1.0, 2.0]]))
    )
    out = await model.embed(["some text"])
    assert out == [[1.0, 2.0]]


@pytest.mark.asyncio
async def test_openai_model_health_check_ok() -> None:
    model = _build_model(_FakeClient(lambda kwargs: _FakeResponse([])))
    assert await model.health_check() is True


@pytest.mark.asyncio
async def test_openai_model_health_check_failure() -> None:
    model = _build_model(
        _FakeClient(lambda kwargs: _FakeResponse([]), models=_FakeModels(ok=False))
    )
    assert await model.health_check() is False


@pytest.mark.asyncio
async def test_openai_model_aclose_closes_httpx_client() -> None:
    client = _FakeClient(lambda kwargs: _FakeResponse([]))
    model = _build_model(client)
    await model.aclose()
    assert client.closed is True  # P2-20：优雅关闭必须释放 httpx 连接池


# ── observability：logger ─────────────────────────────────────────────────────


def test_logger_configure_idempotent_on_same_params(monkeypatch: pytest.MonkeyPatch) -> None:
    # P3-11：相同参数二次 configure 应幂等跳过（main.py 与 assembler 都会调用）。
    from ai_data_agent.observability import logger as logger_mod

    logger_mod._LOGGER_CONFIGURED = False
    logger_mod._LAST_CONFIG = None
    calls: list[dict] = []
    monkeypatch.setattr(logger_mod.structlog, "configure", lambda **kw: calls.append(kw))
    logger_mod.configure_logging(json_logs=True, log_level="INFO")
    logger_mod.configure_logging(json_logs=True, log_level="INFO")
    assert len(calls) == 1


def test_logger_configure_reapplies_when_params_change(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_data_agent.observability import logger as logger_mod

    logger_mod._LOGGER_CONFIGURED = False
    logger_mod._LAST_CONFIG = None
    calls: list[dict] = []
    monkeypatch.setattr(logger_mod.structlog, "configure", lambda **kw: calls.append(kw))
    logger_mod.configure_logging(json_logs=True, log_level="INFO")
    logger_mod.configure_logging(json_logs=False, log_level="DEBUG")
    assert len(calls) == 2


def test_logger_get_logger_returns_usable_logger() -> None:
    from ai_data_agent.observability.logger import get_logger

    log = get_logger("test.logger")
    # 返回的 logger 应可直接调用结构化日志方法，不抛异常
    log.debug("test.event", key="value")


# ── observability：metrics ────────────────────────────────────────────────────


def test_metrics_sql_error_counter_increments() -> None:
    # P3-11：SQL 失败路径现在会 inc() sql_errors_total，可与 sql_queries_total 算错误率。
    from prometheus_client import REGISTRY

    from ai_data_agent.observability.metrics import metrics

    before = REGISTRY.get_sample_value("sql_errors_total") or 0.0
    metrics.sql_errors_total.inc()
    after = REGISTRY.get_sample_value("sql_errors_total") or 0.0
    assert after - before == 1


def test_metrics_llm_labeled_counter_increments() -> None:
    from prometheus_client import REGISTRY

    from ai_data_agent.observability.metrics import metrics

    before = REGISTRY.get_sample_value("llm_tokens_total", {"model": "gpt-4o"}) or 0.0
    metrics.llm_tokens_total.labels(model="gpt-4o").inc(7)
    after = REGISTRY.get_sample_value("llm_tokens_total", {"model": "gpt-4o"}) or 0.0
    assert after - before == 7


# ── observability：tracer ─────────────────────────────────────────────────────


def test_tracer_span_noop_when_uninitialized() -> None:
    # 未初始化（enable_tracing=False / 未装 opentelemetry）时，span() 是 NoOp。
    from ai_data_agent.observability import tracer as tracer_mod

    tracer_mod._tracer = None
    tracer_mod._trace_module = None
    with tracer_mod.span("test.span", {"k": "v"}) as s:
        assert s is None


def test_tracer_shutdown_noop_when_uninitialized() -> None:
    from ai_data_agent.observability import tracer as tracer_mod

    tracer_mod._tracer = None
    tracer_mod._trace_module = None
    tracer_mod.shutdown_tracer()  # P3-11：未初始化时 shutdown 是 no-op，不应抛异常
    assert tracer_mod._tracer is None


def test_tracer_helpers_noop_when_uninitialized() -> None:
    from ai_data_agent.observability import tracer as tracer_mod

    tracer_mod._tracer = None
    tracer_mod._trace_module = None
    tracer_mod.record_exception(ValueError("boom"))  # 无活跃 span 时静默忽略，不应抛异常


def test_tracer_init_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_data_agent.config.config import settings
    from ai_data_agent.observability import tracer as tracer_mod

    monkeypatch.setattr(settings, "enable_tracing", False)
    monkeypatch.setattr(settings, "otlp_endpoint", None)
    tracer_mod._tracer = None
    tracer_mod._trace_module = None
    tracer_mod.init_tracer()
    assert tracer_mod._tracer is None
