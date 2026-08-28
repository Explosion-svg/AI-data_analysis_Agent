# AI Data Agent

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![CI](https://github.com/Explosion-svg/AI-data_analysis_Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Explosion-svg/AI-data_analysis_Agent/actions/workflows/ci.yml)

</div>

一个面向数据分析场景的多工具 AI Agent：输入自然语言问题，自动完成上下文准备、工具选择、SQL 查询、数据分析、图表生成与结果解释。

项目重点不只是"能回答问题"，而是把一个分析型 Agent 按清晰的工程边界落成代码--分层架构、组合根装配、可观测、可压测、可测试。

## Features

- 🗣️ **自然语言驱动** -- ReAct 风格多轮工具调用编排，自动规划执行路径
- 🔧 **多工具协作** -- SQL 查询、Python 分析（子进程沙箱隔离）、图表生成、Schema 检索、RAG
- 🧠 **分层记忆** -- Conversation Memory（聊过什么）+ Work Memory（任务进行到哪），支持内存/Redis 双后端
- 🔀 **多模型路由** -- OpenAI / DeepSeek / Anthropic / 本地模型，按任务复杂度选型，失败自动降级
- 🛡️ **工程可靠性** -- 熔断器、bulkhead 并发隔离、重试、超时、SQL 安全校验、结果缓存
- 📊 **可观测** -- structlog 结构化日志、Prometheus 指标、OpenTelemetry 链路追踪（可选）

## Architecture

8 层架构，请求链路自上而下：

| 层 | 目录 | 职责 |
|---|------|------|
| API | `api/` | HTTP 入口、鉴权、请求上下文 |
| Context | `context/` | query rewrite、prompt 构建、schema 上下文 |
| Tools | `tools/` | SQL / Python / chart / schema / RAG 执行能力 |
| Orchestration | `orchestration/` | ReAct 主循环、planner、executor |
| Memory | `memory/` | 会话记忆、工作记忆、缓存（内存/Redis 双后端） |
| Reliability | `reliability/` | 熔断、重试、超时、bulkhead、SQL guard |
| Observability | `observability/` | 日志、trace、metrics |
| Model Gateway | `model_gateway/` | LLM 适配器与路由 |
| Infra | `infra/` | DB / warehouse / vector store |

组件创建与业务运行严格分离：[`assembler.py`](ai_data_agent/assembler.py) 是唯一的 Composition Root，`AgentLoop` 通过构造函数接收全部依赖，业务层不自建组件。

## Repository Layout

```text
ai_data_agent/
├── api/                # HTTP 入口（chat / health / 会话管理）
├── config/             # pydantic-settings 配置管理
├── context/            # prompt、query rewrite、schema context、请求上下文
├── evaluation/         # 基准数据集与评测运行器
├── infra/              # DB / warehouse / ChromaDB vector store
├── memory/             # conversation / work memory / cache（内存 + Redis）
├── model_gateway/      # LLM 适配器与多模型路由
├── observability/      # structlog 日志、Prometheus 指标、OTel 追踪
├── orchestration/      # planner / executor / ReAct agent loop
├── reliability/        # breaker / retry / timeout / bulkhead / SQL guard
├── tools/              # SQL / Python / chart / schema / RAG
├── assembler.py        # Composition Root（唯一推荐装配入口）
└── main.py             # FastAPI 应用与启动入口
```

## Requirements

- Python 3.12+（CI 验证 3.13）
- 至少配置一种模型服务：OpenAI / DeepSeek / Anthropic / 兼容 OpenAI API 的本地服务（Ollama、LM Studio）
- 可选：Redis（memory/cache 后端切换时需要）

依赖清单见 [`requirements.txt`](requirements.txt)。

## Quick Start

### 1. 安装

```bash
python -m venv venv
source venv/bin/activate        # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
```

最小可用配置（OpenAI 示例）：

```env
OPENAI_API_KEY=sk-...
OPENAI_DEFAULT_MODEL=gpt-4o
OPENAI_FAST_MODEL=gpt-4o-mini
```

使用 DeepSeek / 本地模型时，按 [`config.py`](ai_data_agent/config/config.py) 中的字段名配置对应段落即可，无需全部填写。

### 3. 启动

```bash
uvicorn ai_data_agent.main:app --reload
```

服务默认监听 `http://127.0.0.1:8000`，交互式 API 文档在 `/docs`（Swagger）与 `/redoc`。

### Docker 部署

```bash
docker build -t ai-data-agent .
docker run -p 8000:8000 --env-file .env ai-data-agent
```

## Configuration

完整可配置项见 [`.env.example`](.env.example)（每项均有注释）。关键分组：

| 分组 | 说明 |
|------|------|
| LLM | 多提供方接入、温度、超时、重试次数 |
| Concurrency / Bulkhead | 各层并发上限与获取超时（单实例生产基线已给出） |
| Memory / Cache | `memory` \| `redis` 双后端选择、TTL、容量上限 |
| Reliability | 熔断阈值、恢复窗口、SQL/Python 超时、最大迭代数 |
| Observability | 日志级别/JSON、追踪开关、Prometheus 端口 |
| Security | `API_KEY`（留空则匿名访问）、`SQL_READONLY`、Python 沙箱开关 |

> **安全提示**：`API_KEY` 留空时 `/chat` 允许匿名调用。公网部署务必配置。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/chat` | 对话入口，返回分析结果与执行过程 |
| `GET` | `/api/v1/health` | 健康检查 |
| `DELETE` | `/api/v1/conversations/{conversation_id}` | 清除指定会话历史 |

**请求示例：**

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "query": "今年每个月的销售额趋势怎么样？",
    "conversation_id": "demo-session",
    "use_cache": true
  }'
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | `string` (1-4096 字符) | 用户问题，必填 |
| `conversation_id` | `string?` | 会话 ID，留空自动生成 UUID |
| `use_cache` | `bool` | 相同问题是否直接返回缓存（默认 `true`） |

配置 `API_KEY` 后所有请求需携带 `Authorization: Bearer <key>`；未配置时跳过鉴权。

## Memory Design

记忆拆成两层，避免把所有内容塞进一层历史导致 prompt 与状态边界混乱：

- **Conversation Memory** -- "聊过什么"：近期原始对话、LLM 滚动摘要、pinned facts
- **Work Memory** -- "任务做到哪"：query rewrite、schema、工具步骤、最近 SQL、数据摘要

两层均有内存与 Redis 两套实现，通过 `MEMORY_BACKEND` / `CACHE_BACKEND` 切换，由 [`memory/factory.py`](ai_data_agent/memory/factory.py) 统一装配。

## Python 沙箱

LLM 生成的 Python 代码在**一次性子进程**中执行（`python_tool.py`）：

- 代码与数据经 stdin JSON 传入，工作目录隔离在系统临时目录
- 硬超时（`PYTHON_EXEC_TIMEOUT`，默认 20s）到点杀进程，`while True` 无法冻结服务
- 敏感环境变量（API key 等）不透传给子进程
- stdout 与返回值均有字节数截断

## Testing

```bash
pytest                          # 全量（单元 + 集成，无需真实 LLM/Redis 密钥）
python run_tests.py             # 等价入口
```

测试套件对依赖全部做了 fake（FakeLLM、FakeRedis、临时 SQLite/Chroma），CI 无密钥可跑。压测：

```bash
python run_load_test.py --mode asgi --requests 200 --concurrency 50 --fake-latency-ms 100
python run_load_test.py --mode http --url http://127.0.0.1:8000 --requests 500 --concurrency 100
```

重点关注 `success/failures`、`status_counts` 中 503 占比、`latency_p95/p99_ms`。调参原则：503 多且依赖未满时提高 `AGENT_REQUEST_CONCURRENCY`；尾延迟恶化时下调 `LLM_CONCURRENCY`。

## Documentation

- [`Architecture.md`](Architecture.md) -- 架构设计详解
- [`TESTING.md`](TESTING.md) -- 测试体系说明
- [`TESTING_BEGINNER.md`](TESTING_BEGINNER.md) -- 测试入门指引
- [`REVIEW_FINDINGS.md`](REVIEW_FINDINGS.md) -- 全量代码审查与修复记录

## Development Notes

- 组件一律经 `assembler.py` 装配，不要在业务层直接 `AgentLoop()`
- `agent_loop.py` 只负责编排，不负责创建依赖
- 新工具注册到 `tool_registry`；新配置项进 `config.py` 并同步 `.env.example`
- 多 worker 部署时指标走 multiprocess 模式（已自动处理，无需手动配置）

## License

[MIT](#license)
