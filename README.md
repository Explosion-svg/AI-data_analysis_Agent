# AI 数据分析 Agent

> 企业级、8 层架构、高稳定性的 AI 数据分析智能体
> 输入自然语言问题，自动查询数据库、分析数据、生成图表，输出业务洞察。

---

## 目录

1. [项目简介](#1-项目简介)
2. [整体架构](#2-整体架构)
3. [目录结构详解](#3-目录结构详解)
4. [环境准备](#4-环境准备)
5. [安装步骤](#5-安装步骤)
6. [配置说明](#6-配置说明)
7. [快速启动](#7-快速启动)
8. [API 使用教程](#8-api-使用教程)
9. [各层代码详解](#9-各层代码详解)
10. [如何扩展：添加新工具](#10-如何扩展添加新工具)
11. [如何扩展：接入新的 LLM](#11-如何扩展接入新的-llm)
12. [数据接入指南](#12-数据接入指南)
13. [可观测性与监控](#13-可观测性与监控)
14. [安全设计](#14-安全设计)
15. [常见问题 FAQ](#15-常见问题-faq)
16. [性能调优建议](#16-性能调优建议)
17. [生产部署建议](#17-生产部署建议)

---

## 1. 项目简介

### 这是什么？

这是一个能用 **自然语言** 分析数据的 AI 智能体（Agent）。

**举个例子：**
- 你问："今年每个月的销售额趋势怎么样？"
- Agent 自动：
  1. 查看数据库表结构（知道有 `sales` 表）
  2. 写 SQL 查询月度销售数据
  3. 用 Python/pandas 计算趋势
  4. 用 Plotly 生成折线图
  5. 用自然语言解释分析结论

### 核心特性

| 特性 | 说明 |
|------|------|
| 自然语言查询 | 无需写 SQL，直接用中文/英文提问 |
| 多工具协作 | SQL、Python 分析、图表生成、文档检索自动组合 |
| 多模型支持 | OpenAI / DeepSeek / Claude / 本地 Ollama 均可 |
| 企业级可靠性 | 熔断器、重试、超时、SQL 安全防护 |
| 完整可观测性 | 结构化日志、Prometheus 指标、OpenTelemetry 链路追踪 |
| 多轮对话 | 记住上下文，支持追问 |

### 适合谁使用？

- **数据分析师**：用自然语言替代手工写 SQL
- **产品经理**：自助查看业务数据，无需找数据团队
- **开发者**：学习企业级 Agent 的架构和实现
- **初学者**：了解 LLM / RAG / 工具调用的完整工程实践

---

## 2. 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                     用户 / 前端                               │
└─────────────────────────┬────────────────────────────────────┘
                          │  HTTP POST /api/v1/chat
┌─────────────────────────▼────────────────────────────────────┐
│  Layer 1: API Layer (chat_api.py)                             │
│  唯一 HTTP 入口，只做请求接收/校验/返回，零业务逻辑           │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  Layer 2: Context Management (context/)                       │
│  Prompt 构建 | Query 改写 | Schema 上下文提取                 │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  Layer 3: Tool System (tools/)                                │
│  SQL工具 | Python工具 | 图表工具 | Schema工具 | RAG工具       │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  Layer 4: Orchestration (orchestration/)                      │
│  Planner 任务规划 | Executor 执行 | AgentLoop ReAct循环       │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  Layer 5: State & Memory (memory/)                            │
│  对话历史（多会话隔离）| LRU+TTL 结果缓存                     │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  Layer 6: Reliability (reliability/)                          │
│  重试 | SQL安全 | 熔断器 | 降级 | 超时控制                   │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  Layer 7: Observability (observability/)                      │
│  结构化日志(structlog) | Prometheus指标 | OpenTelemetry链路   │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  Layer 8: Model Gateway (model_gateway/)                      │
│  OpenAI | DeepSeek | Claude | 本地LLM | 智能路由 + Fallback  │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  Infra Layer (infra/)                                         │
│  OLTP数据库(SQLAlchemy) | 数据仓库 | 向量数据库(ChromaDB)    │
└──────────────────────────────────────────────────────────────┘
                          ▲
                          │ 贯穿所有层
┌─────────────────────────┴────────────────────────────────────┐
│  Assembler (assembler.py)                                     │
│  应用装配器：按正确顺序创建所有组件，统一管理生命周期         │
└──────────────────────────────────────────────────────────────┘
```

### 一次请求的完整流程

```
用户发送："今年销售趋势"
    ↓
[API] 接收请求，生成 conversation_id
    ↓
[Assembler] 提供已装配好的 AgentLoop
    ↓
[AgentLoop] 开始 ReAct 循环
    ↓
[QueryRewriter] 将问题改写为多个搜索查询，提高 RAG 召回
    ↓
[SchemaContext] 语义搜索最相关的数据表
    ↓
[RAGTool] 检索知识库中的相关文档
    ↓
[PromptBuilder] 组装 System Prompt + Schema + 文档 + 历史 + 用户问题
    ↓
[ModelRouter] 发送给 LLM（自动选择 GPT-4o / DeepSeek 等）
    ↓
[LLM] 决定：调用 get_schema 工具 → 查看表结构
    ↓
[SchemaTool] 返回表的列信息
    ↓
[LLM] 决定：调用 sql_query 工具 → 执行 SQL
    ↓
[SQLGuard] 校验 SQL 安全（只允许 SELECT）
    ↓
[SQLTool] 执行 SQL，返回 DataFrame
    ↓
[LLM] 决定：调用 generate_chart 工具 → 生成图表
    ↓
[ChartTool] 用 Plotly 生成 JSON 图表
    ↓
[LLM] 生成最终自然语言解读
    ↓
[ConversationMemory] 保存本轮对话
    ↓
[CacheMemory] 缓存结果（相同问题直接返回）
    ↓
[API] 返回：answer + charts + data + tool_calls
    ↓
用户看到：文字分析 + 交互式图表
```

---

## 3. 目录结构详解

```
program4/
├── ai_data_agent/              # 主包
│   ├── __init__.py
│   │
│   ├── assembler.py            # ⭐ 装配器：组装所有组件（Composition Root）
│   ├── main.py                 # 启动入口：FastAPI 应用工厂
│   │
│   ├── config/
│   │   └── config.py           # 全局配置（Pydantic Settings，读取 .env）
│   │
│   ├── api/
│   │   └── chat_api.py         # HTTP 路由（POST /chat, GET /health）
│   │
│   ├── context/                # 上下文管理层
│   │   ├── prompt_builder.py   # 组装 messages[] 发给 LLM
│   │   ├── query_rewriter.py   # 将用户问题改写为多个查询
│   │   └── schema_context.py   # 动态提取相关表结构
│   │
│   ├── tools/                  # 工具层（Agent 的"手"）
│   │   ├── base_tool.py        # 所有工具的抽象基类
│   │   ├── tool_registry.py    # 工具注册中心
│   │   ├── sql_tool.py         # SQL 查询工具
│   │   ├── python_tool.py      # Python 沙盒执行工具
│   │   ├── chart_tool.py       # Plotly 图表生成工具
│   │   ├── schema_tool.py      # 数据库结构查询工具
│   │   └── rag_tool.py         # 语义文档检索工具
│   │
│   ├── orchestration/          # 编排层（Agent 的"大脑"）
│   │   ├── planner.py          # 任务规划：把问题拆解为步骤
│   │   ├── executor.py         # 执行规划步骤
│   │   └── agent_loop.py       # ReAct 主循环
│   │
│   ├── memory/                 # 状态与记忆层
│   │   ├── conversation_memory.py  # 对话历史（滑动窗口）
│   │   └── cache_memory.py         # LRU+TTL 缓存
│   │
│   ├── reliability/            # 可靠性层（稳定性保障）
│   │   ├── retry.py            # 指数退避重试
│   │   ├── sql_guard.py        # SQL 安全卫士
│   │   ├── circuit_breaker.py  # 熔断器
│   │   ├── fallback.py         # 降级处理
│   │   └── timeout.py          # 超时控制
│   │
│   ├── observability/          # 可观测性层
│   │   ├── logger.py           # 结构化日志（structlog）
│   │   ├── tracer.py           # 分布式追踪（OpenTelemetry）
│   │   └── metrics.py          # Prometheus 指标
│   │
│   ├── model_gateway/          # 模型网关层
│   │   ├── base_model.py       # LLM 抽象接口
│   │   ├── openai_model.py     # OpenAI/DeepSeek/本地 适配器
│   │   └── router.py           # 智能模型路由 + Fallback
│   │
│   ├── infra/                  # 基础设施层
│   │   ├── database.py         # OLTP 数据库（SQLAlchemy 异步）
│   │   ├── warehouse.py        # OLAP 数据仓库（分析型查询）
│   │   └── vector_store.py     # 向量数据库（ChromaDB）
│   │
│   └── evaluation/             # 评估层
│       ├── benchmark_dataset.py # 测试用例管理
│       └── eval_runner.py       # 批量评估运行器
│
├── data/                        # 运行时数据目录（自动创建）
│   ├── agent.db                 # SQLite OLTP 数据库
│   ├── warehouse.db             # SQLite 数据仓库（示例）
│   └── chroma/                  # ChromaDB 向量库持久化
│
├── requirements.txt             # Python 依赖
└── .env.example                 # 环境变量模板（复制为 .env 后填写）
```

---

## 4. 环境准备

### 系统要求

| 项目 | 最低要求 | 推荐 |
|------|----------|------|
| Python | 3.10+ | 3.12 |
| 内存 | 2 GB | 8 GB+ |
| 磁盘 | 500 MB | 2 GB+ |
| 操作系统 | Windows 10 / macOS 12 / Ubuntu 20.04 | 任意 |

### 需要的账号/服务（至少一个）

| 服务 | 用途 | 获取地址 |
|------|------|----------|
| OpenAI | GPT-4o / Embedding | https://platform.openai.com |
| DeepSeek | 代码生成 LLM | https://platform.deepseek.com |
| Anthropic | Claude 系列 | https://console.anthropic.com |
| Ollama（免费） | 本地运行 LLM，无需云端 | https://ollama.ai |

> **完全免费方案**：只需安装 Ollama + 下载一个本地模型即可运行（不需要任何 API Key）

### 安装 Python（已有可跳过）

**Windows:**
1. 访问 https://www.python.org/downloads/
2. 下载 Python 3.12.x
3. 安装时勾选 "Add Python to PATH"
4. 验证：打开命令行输入 `python --version`

**macOS:**
```bash
brew install python@3.12
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install python3.12 python3.12-venv
```

---

## 5. 安装步骤

### 第一步：下载代码

```bash
# 进入你想存放代码的目录
cd E:/study/program/program4   # 已在此目录
```

### 第二步：创建虚拟环境

> 虚拟环境的作用：让这个项目的依赖和系统 Python 隔离，避免版本冲突

```bash
# 创建虚拟环境（只需执行一次）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 激活成功后，命令行前会出现 (venv) 前缀
```

### 第三步：安装依赖

```bash
# 升级 pip（避免旧版本问题）
pip install --upgrade pip

# 安装所有依赖
pip install -r requirements.txt

# 国内网络加速（可选）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装过程约 3-10 分钟，主要耗时在 `chromadb`、`pandas`、`plotly` 等大型包。

### 第四步：创建数据目录

```bash
mkdir -p data/chroma
```

### 第五步：配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 用文本编辑器打开 .env，填写你的 API Key
# Windows:
notepad .env

# macOS / Linux:
nano .env  # 或 vim .env
```

---

## 6. 配置说明

打开 `.env` 文件，按需填写以下配置：

### 最简配置（5分钟上手）

只需填写一个 LLM API Key 即可运行：

```env
# 选项A：使用 OpenAI（推荐）
OPENAI_API_KEY=sk-your-key-here

# 选项B：使用 DeepSeek（更便宜）
DEEPSEEK_API_KEY=sk-your-key-here

# 选项C：使用本地 Ollama（完全免费）
LOCAL_LLM_API_BASE=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen2.5:7b
```

### 完整配置说明

```env
# ════════════════════════════════════
# 应用基础配置
# ════════════════════════════════════
ENV=dev          # 环境：dev（开发）| staging（测试）| prod（生产）
DEBUG=false      # 是否开启调试模式（开发时可设为 true）
PORT=8000        # 服务端口

# API 访问密钥（留空则不需要认证，适合本地开发）
API_KEY=         # 例：API_KEY=mysecretkey123

# ════════════════════════════════════
# 数据库配置
# ════════════════════════════════════
# 开发用 SQLite（无需安装额外数据库）
DATABASE_URL=sqlite+aiosqlite:///./data/agent.db

# 生产用 PostgreSQL 示例：
# DATABASE_URL=postgresql+asyncpg://用户名:密码@localhost:5432/数据库名

# 数据仓库（你的业务数据在这里）
WAREHOUSE_URL=sqlite+aiosqlite:///./data/warehouse.db

# ════════════════════════════════════
# LLM 配置（填写至少一个）
# ════════════════════════════════════
OPENAI_API_KEY=sk-...
OPENAI_DEFAULT_MODEL=gpt-4o          # 复杂任务用这个
OPENAI_FAST_MODEL=gpt-4o-mini        # 简单任务用这个（省钱）

DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat

# ════════════════════════════════════
# 性能与稳定性配置
# ════════════════════════════════════
AGENT_MAX_ITERATIONS=10      # Agent 最多思考几轮（防死循环）
SQL_QUERY_TIMEOUT=30.0        # SQL 超时秒数
PYTHON_EXEC_TIMEOUT=20.0      # Python 代码执行超时
CACHE_TTL_SECONDS=300         # 缓存有效期（秒）
CONVERSATION_MAX_TURNS=20     # 保留最近多少轮对话

# ════════════════════════════════════
# 安全配置
# ════════════════════════════════════
SQL_READONLY=true    # true=只允许 SELECT，false=允许所有操作（危险！）
PYTHON_SANDBOX=true  # true=沙盒执行 Python
```

### 使用 Ollama（免费本地运行）

```bash
# 1. 安装 Ollama（macOS/Linux）
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: 访问 https://ollama.ai 下载安装包

# 2. 下载模型（选一个）
ollama pull qwen2.5:7b      # 中文效果好，7B 参数，需要 8GB 内存
ollama pull llama3.2:3b     # 英文效果好，3B 参数，需要 4GB 内存
ollama pull deepseek-r1:7b  # 推理能力强

# 3. 启动 Ollama 服务
ollama serve

# 4. 在 .env 中配置
LOCAL_LLM_API_BASE=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen2.5:7b
```

---

## 7. 快速启动

### 启动服务

```bash
# 确保虚拟环境已激活（看到命令行前有 (venv)）
# Windows:
venv\Scripts\activate

# 启动方式一：直接运行（推荐开发时使用）
python -m ai_data_agent.main

# 启动方式二：uvicorn 命令（更多控制选项）
uvicorn ai_data_agent.main:app --host 0.0.0.0 --port 8000 --reload

# 启动成功后，你会看到类似输出：
# {"event": "app.ready", "host": "0.0.0.0", "port": 8000}
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 验证服务正常运行

```bash
# 健康检查
curl http://localhost:8000/api/v1/health

# 返回：
# {"status": "ok", "version": "1.0.0", "env": "dev"}
```

### 访问 API 文档

浏览器打开：**http://localhost:8000/docs**

你会看到完整的 Swagger UI，可以直接在浏览器中测试所有接口。

---

## 8. API 使用教程

### 8.1 对话接口

**POST** `/api/v1/chat`

#### 请求格式

```json
{
  "query": "你的问题",
  "conversation_id": "可选，同一个ID表示同一个会话",
  "use_cache": true
}
```

#### 示例：第一次提问

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "数据库里有哪些表？",
    "conversation_id": "my-session-001"
  }'
```

#### 返回格式

```json
{
  "conversation_id": "my-session-001",
  "answer": "数据仓库中有以下表：\n- sales（销售记录）\n- orders（订单）\n- users（用户）",
  "iterations": 2,
  "tool_calls": [
    {
      "tool": "get_schema",
      "args": {"action": "list_tables"},
      "success": true
    }
  ],
  "charts": [],
  "data": [],
  "latency_ms": 1243.5,
  "success": true
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `conversation_id` | string | 会话 ID，用于追问 |
| `answer` | string | Agent 的自然语言回答 |
| `iterations` | int | Agent 思考了几轮才得出答案 |
| `tool_calls` | array | Agent 调用了哪些工具 |
| `charts` | array | 生成的 Plotly 图表 JSON（可直接渲染） |
| `data` | array | SQL 查询返回的原始数据（records 格式） |
| `latency_ms` | float | 总耗时（毫秒） |

#### 示例：数据查询

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "查询销售表的最近10条记录",
    "conversation_id": "my-session-001"
  }'
```

#### 示例：追问（利用对话记忆）

```bash
# 第一问
curl -X POST http://localhost:8000/api/v1/chat \
  -d '{"query": "去年总销售额是多少？", "conversation_id": "sess-abc"}'

# 追问（Agent 会记住上下文）
curl -X POST http://localhost:8000/api/v1/chat \
  -d '{"query": "那今年呢？同比增长了多少？", "conversation_id": "sess-abc"}'
```

#### 示例：请求生成图表

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -d '{
    "query": "用折线图展示每月销售额趋势",
    "conversation_id": "sess-abc"
  }'

# 返回的 charts 字段包含 Plotly JSON，前端可这样渲染：
# import Plotly from 'plotly.js'
# Plotly.newPlot('div-id', response.charts[0].data, response.charts[0].layout)
```

### 8.2 清除会话历史

```bash
curl -X DELETE http://localhost:8000/api/v1/conversations/my-session-001
```

### 8.3 使用 API Key 认证

如果在 `.env` 中设置了 `API_KEY`，所有请求需要带上 Bearer Token：

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query": "你好"}'
```

### 8.4 Python 客户端示例

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"
HEADERS = {"Content-Type": "application/json"}

def chat(query: str, conversation_id: str = "default") -> dict:
    resp = requests.post(
        f"{BASE_URL}/chat",
        headers=HEADERS,
        json={"query": query, "conversation_id": conversation_id},
    )
    resp.raise_for_status()
    return resp.json()

# 使用示例
result = chat("今年各月销售额是多少？")
print(result["answer"])

# 追问
result2 = chat("哪个月最高？", conversation_id=result["conversation_id"])
print(result2["answer"])
```

---

## 9. 各层代码详解

> 这一节面向想深入理解代码的学习者

### 9.1 装配器 (assembler.py) ⭐ 核心

**位置：** `ai_data_agent/assembler.py`

装配器是整个系统的 **"连线图"**，它解决一个关键问题：

> 所有组件都需要依赖其他组件，谁负责把它们组装在一起？

```
没有装配器（混乱）：
  AgentLoop → import ConversationMemory → import CacheMemory → 循环依赖？

有了装配器（清晰）：
  AppContainer.startup():
    1. 创建 ConversationMemory
    2. 创建 CacheMemory
    3. 创建 AgentLoop(memory, cache)  ← 注入依赖
```

**关键方法：**

```python
from ai_data_agent.assembler import get_container

# 获取容器（在应用启动后可从任何地方调用）
container = get_container()

# 获取 Agent
agent = container.get_agent_loop()

# 查看健康状态
print(container.health_report())
```

**装配顺序（严格按依赖关系）：**
```
Observability  →  日志最先初始化
Infra          →  数据库、向量库
ModelGateway   →  LLM 路由器
Tools          →  各种工具（依赖 infra + model_gateway）
Context        →  Prompt 构建（依赖 model_gateway）
Memory         →  对话记忆、缓存
Orchestration  →  规划器、执行器、Agent 循环（依赖上述所有）
```

### 9.2 ReAct Agent 循环 (agent_loop.py)

**ReAct = Reasoning（推理）+ Acting（行动）**

```
思考 → 行动 → 观察 → 再思考 → 再行动 ...
```

**代码流程（简化版）：**

```python
# orchestration/agent_loop.py 核心逻辑
while iteration < max_iterations:
    # 1. 让 LLM 思考，可能返回"调用某个工具"
    response = await llm.generate(messages, tools=all_tools)

    if not response.tool_calls:
        # LLM 认为已经得到答案，结束循环
        return final_answer

    # 2. 执行 LLM 决定调用的工具
    for tool_call in response.tool_calls:
        tool = registry.get(tool_call.name)
        result = await tool.run(**tool_call.args)

        # 3. 把工具结果作为"观察"加入对话
        messages.append({"role": "tool", "content": result.text})

    # 继续下一轮思考...
```

### 9.3 工具系统 (tools/)

每个工具遵循统一接口：

```python
# tools/base_tool.py
class BaseTool:
    name: str          # 工具唯一名称
    description: str   # 给 LLM 看的说明，决定 LLM 何时调用这个工具

    async def run(**kwargs) -> ToolResult:
        # 执行逻辑
        pass
```

**LLM 如何选择工具：**

系统会把所有工具的 `name` 和 `description` 传给 LLM，LLM 根据用户问题判断该用哪个工具。例如：

```
工具列表（发给 LLM）：
- sql_query: "Execute a SELECT SQL query against the data warehouse..."
- python_analysis: "Execute Python code for data analysis using pandas..."
- generate_chart: "Generate an interactive chart using Plotly..."
- get_schema: "Query the data warehouse schema. List all tables..."
- search_documents: "Search internal knowledge base using semantic search..."
```

### 9.4 SQL 安全防护 (sql_guard.py)

这是防止 AI 误操作数据库的关键：

```python
# reliability/sql_guard.py

# 黑名单关键词
_DANGEROUS = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|TRUNCATE|...)\b")

def validate_sql(sql: str) -> str:
    # 1. 检测危险关键词
    if _DANGEROUS.search(sql):
        raise SQLGuardError("危险操作被拒绝")

    # 2. 用 sqlparse 解析，确认是 SELECT
    if parsed.get_type() != "SELECT":
        raise SQLGuardError("只允许 SELECT 语句")

    # 3. 检测多语句注入（防止 ; DROP TABLE）
    if len(statements) > 1:
        raise SQLGuardError("不允许多条语句")

    return sql  # 通过检验
```

### 9.5 熔断器 (circuit_breaker.py)

当外部服务（如 OpenAI API）频繁失败时，熔断器自动"断开"，防止雪崩：

```
正常状态（CLOSED）：所有请求正常通过
         ↓ 失败次数 >= 5
熔断状态（OPEN）：拒绝所有请求，立即返回错误
         ↓ 等待 60 秒
尝试恢复（HALF_OPEN）：放行一个请求测试
         ↓ 成功
正常状态（CLOSED）：恢复正常
```

### 9.6 模型路由 (router.py)

根据任务类型选择最合适且最省钱的模型：

```python
# model_gateway/router.py

# 简单问题 → 便宜的小模型
simple task  → gpt-4o-mini (快，省钱)

# 复杂分析 → 强力大模型
complex task → gpt-4o (准确)

# 代码生成 → 专门的代码模型
code task    → deepseek-chat (代码能力强)
```

---

## 10. 如何扩展：添加新工具

> 假设我们要添加一个"天气查询工具"

### 步骤 1：创建工具文件

新建 `ai_data_agent/tools/weather_tool.py`：

```python
from ai_data_agent.tools.base_tool import BaseTool, ToolResult
import aiohttp

class WeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "get_weather"  # 工具唯一标识

    @property
    def description(self) -> str:
        return "Get current weather for a city."  # LLM 看到的描述

    @property
    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }

    async def _run(self, city: str, **kwargs) -> ToolResult:
        # 这里写实际的查询逻辑
        # 示例：调用免费天气 API
        async with aiohttp.ClientSession() as session:
            resp = await session.get(f"https://wttr.in/{city}?format=3")
            weather = await resp.text()

        return ToolResult(
            success=True,
            data={"city": city, "weather": weather},
            text=f"Weather in {city}: {weather}"
        )
```

### 步骤 2：在装配器中注册

打开 `ai_data_agent/assembler.py`，找到 `_init_tools` 方法，添加一行：

```python
async def _init_tools(self) -> None:
    from ai_data_agent.tools.sql_tool import SQLTool
    from ai_data_agent.tools.python_tool import PythonTool
    # ... 其他工具 ...
    from ai_data_agent.tools.weather_tool import WeatherTool  # ← 新增这行

    registry = ToolRegistry()
    registry.register(SQLTool())
    registry.register(PythonTool())
    # ... 其他注册 ...
    registry.register(WeatherTool())  # ← 新增这行
```

### 步骤 3：重启服务，测试

```bash
# 重启服务
python -m ai_data_agent.main

# 测试新工具
curl -X POST http://localhost:8000/api/v1/chat \
  -d '{"query": "北京今天天气怎么样？"}'
```

完成！Agent 会自动识别相关问题并调用新工具。

---

## 11. 如何扩展：接入新的 LLM

> 假设我们要接入智谱 AI（GLM-4）

### 步骤 1：创建适配器

由于智谱 AI 支持 OpenAI 兼容接口，可以直接复用 `OpenAIModel`：

打开 `ai_data_agent/model_gateway/openai_model.py`，在末尾添加：

```python
def build_zhipuai_model() -> OpenAIModel | None:
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        return None
    return OpenAIModel(
        api_key=api_key,
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4-flash",
        adapter_name="zhipuai",
    )
```

### 步骤 2：在路由器中注册

打开 `ai_data_agent/model_gateway/router.py`，修改 `_build_registry`：

```python
def _build_registry(self) -> None:
    from ai_data_agent.model_gateway.openai_model import (
        build_openai_model, build_deepseek_model,
        build_local_model, build_zhipuai_model  # ← 新增
    )
    for factory, key in [
        (build_openai_model, "openai"),
        (build_deepseek_model, "deepseek"),
        (build_local_model, "local"),
        (build_zhipuai_model, "zhipuai"),  # ← 新增
    ]:
        ...
```

### 步骤 3：在 .env 中配置

```env
ZHIPUAI_API_KEY=your-key-here
```

---

## 12. 数据接入指南

### 接入自己的业务数据库

#### 方案 A：SQLite（适合学习和小型项目）

1. 把你的数据导入 SQLite：
```bash
# 示例：把 CSV 文件导入 SQLite
python3 << 'EOF'
import sqlite3, pandas as pd

conn = sqlite3.connect("data/warehouse.db")
df = pd.read_csv("sales.csv")
df.to_sql("sales", conn, if_exists="replace", index=False)
print(f"导入 {len(df)} 行数据")
EOF
```

2. 在 `.env` 中确认：
```env
WAREHOUSE_URL=sqlite+aiosqlite:///./data/warehouse.db
```

#### 方案 B：PostgreSQL（适合生产环境）

1. 安装 PostgreSQL，创建数据库
2. 修改 `.env`：
```env
WAREHOUSE_URL=postgresql+asyncpg://username:password@localhost:5432/mydb
```

#### 方案 C：MySQL

```env
WAREHOUSE_URL=mysql+aiomysql://username:password@localhost:3306/mydb
```

### 接入知识库文档（RAG）

将文档向量化并存入 ChromaDB：

```python
# 运行此脚本将文档导入知识库
import asyncio
from ai_data_agent.assembler import startup

async def index_documents():
    container = await startup()
    router = container.get_router()

    # 你的文档
    docs = [
        "GMV（成交总额）是指平台上所有已下单商品的总金额，包含取消和退款订单。",
        "DAU（日活跃用户数）是指每天至少登录一次的不重复用户数量。",
        "销售额是指实际完成交付并支付的订单金额，不含退款。",
    ]
    doc_ids = [f"doc_{i}" for i in range(len(docs))]

    # 生成 embedding 并存入向量库
    embeddings = await router.embed(docs)

    from ai_data_agent.infra import vector_store
    vector_store.upsert_docs(
        ids=doc_ids,
        embeddings=embeddings,
        documents=docs,
        metadatas=[{"source": "业务定义手册"} for _ in docs],
    )
    print(f"成功索引 {len(docs)} 个文档")

asyncio.run(index_documents())
```

---

## 13. 可观测性与监控

### 查看日志

开发模式（彩色文本）：
```env
LOG_JSON=false
LOG_LEVEL=DEBUG
```

生产模式（JSON，便于日志系统解析）：
```env
LOG_JSON=true
LOG_LEVEL=INFO
```

日志示例输出：
```json
{"event": "agent_loop.tool_call", "tool": "sql_query", "iteration": 1, "timestamp": "2026-04-05T10:00:00Z"}
{"event": "tool.done", "tool": "sql_query", "success": true, "latency_ms": 234.5}
{"event": "agent_loop.final_answer", "iterations": 3, "conversation_id": "abc123"}
```

### Prometheus 指标

服务启动后，指标暴露在 `http://localhost:9090/metrics`

关键指标：

| 指标名 | 说明 |
|--------|------|
| `agent_requests_total` | 总请求数 |
| `agent_request_latency_seconds` | 请求耗时分布 |
| `llm_tokens_total{model}` | 各模型 token 消耗 |
| `tool_calls_total{tool_name}` | 各工具调用次数 |
| `sql_blocked_total` | 被 SQL 安全拦截的查询数 |
| `circuit_breaker_open{service}` | 熔断器是否打开（1=打开） |
| `cache_hits_total` / `cache_misses_total` | 缓存命中率 |

用 Grafana 可视化这些指标（可选）：
```bash
# 使用 Docker 快速启动 Prometheus + Grafana
docker-compose up -d  # 需要自行创建 docker-compose.yml
```

### OpenTelemetry 链路追踪

如果你有 Jaeger 或 Zipkin：
```env
ENABLE_TRACING=true
OTLP_ENDPOINT=http://localhost:4317
```

启动 Jaeger（Docker）：
```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

访问 http://localhost:16686 查看每次请求的完整调用链。

---

## 14. 安全设计

### SQL 安全（sql_guard.py）

系统默认只允许 `SELECT` 语句，防止 AI 误删数据。

工作原理（双重校验）：
1. **关键词黑名单**：检测 DROP、DELETE、UPDATE 等危险操作
2. **AST 解析**：用 sqlparse 解析语句结构，确认是 SELECT
3. **多语句检测**：防止 `SELECT 1; DROP TABLE users` 这类注入

如需允许写操作（谨慎！）：
```env
SQL_READONLY=false
```

### Python 沙盒（python_tool.py）

Python 代码在受限环境中执行：
- **只允许**：pandas、numpy、math、statistics 等安全模块
- **禁止**：os、subprocess、socket、open 等危险操作
- **超时**：默认 20 秒后强制终止

### API 认证

```env
API_KEY=your-strong-random-key-here
```

生成强密钥的方法：
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 15. 常见问题 FAQ

### Q: 启动时报 "No LLM adapter configured"

**原因**：没有设置任何 API Key。

**解决**：在 `.env` 中至少设置一个：
```env
OPENAI_API_KEY=sk-...     # 或
DEEPSEEK_API_KEY=sk-...   # 或
LOCAL_LLM_API_BASE=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen2.5:7b
```

---

### Q: 报错 `ModuleNotFoundError: No module named 'chromadb'`

**解决**：
```bash
pip install chromadb
# 或重新安装所有依赖
pip install -r requirements.txt
```

---

### Q: Agent 回答"我没有找到相关数据表"

**原因**：数据仓库为空，或 Agent 找不到合适的表。

**解决**：
1. 确认数据已导入数据仓库（`WAREHOUSE_URL` 对应的数据库）
2. 先问 Agent："数据库里有哪些表？"确认表是否存在
3. 如果表存在但 Agent 找不到，运行 schema 重索引：
```python
import asyncio
from ai_data_agent.assembler import startup

async def reindex():
    c = await startup()
    await c.schema_builder.index_all_tables()
    print("重索引完成")

asyncio.run(reindex())
```

---

### Q: SQL 查询被拦截，报 "Only SELECT statements are allowed"

**原因**：SQL 安全防护工作正常，Agent 尝试执行了非 SELECT 语句。

**这是正常行为**，说明安全防护在保护你的数据。

如果你确认需要执行写操作（如建表、插入测试数据），临时关闭：
```env
SQL_READONLY=false
```

---

### Q: 响应很慢，超过 60 秒

**原因可能有**：
1. LLM API 网络慢 → 可以设置国内代理或换用本地 Ollama
2. SQL 查询太慢 → 检查数据量，考虑加索引
3. Agent 迭代次数太多 → 减少 `AGENT_MAX_ITERATIONS`

**调试方法**：看日志中各步骤的 `latency_ms`：
```bash
# 过滤出 latency 超过 5 秒的日志
tail -f app.log | grep -E '"latency_ms":[0-9]{5}'
```

---

### Q: 如何限制 token 消耗（控制成本）

```env
# 减少最大 token 数
LLM_MAX_TOKENS=2048

# 使用更便宜的小模型处理简单任务
OPENAI_FAST_MODEL=gpt-4o-mini

# 增加缓存时间，减少重复请求
CACHE_TTL_SECONDS=3600

# 减少 Agent 迭代次数
AGENT_MAX_ITERATIONS=5
```

---

### Q: Windows 上 ChromaDB 安装失败

**解决**：
```bash
# 先安装 C++ 构建工具（如果还没有）
# 访问 https://visualstudio.microsoft.com/visual-cpp-build-tools/
# 下载安装 "Build Tools for Visual Studio"

# 然后重新安装
pip install chromadb --prefer-binary
```

---

### Q: 如何在不同项目/会话之间共享对话历史？

当前的 `ConversationMemory` 是内存存储，重启后清空。

要持久化对话历史，可以：
1. 将对话历史保存到数据库（扩展 `conversation_memory.py` 加入 SQLAlchemy 支持）
2. 使用 Redis 存储（修改 `CacheMemory` 使用 redis-py）

---

## 16. 性能调优建议

### 减少 LLM Token 消耗

| 配置项 | 建议值 | 说明 |
|--------|--------|------|
| `OPENAI_FAST_MODEL` | `gpt-4o-mini` | 简单问题用便宜模型 |
| `AGENT_MAX_ITERATIONS` | `5-8` | 减少思考轮数 |
| `CONVERSATION_MAX_TURNS` | `10` | 少保留历史 |
| `CACHE_TTL_SECONDS` | `600` | 更长的缓存时间 |

### 提升响应速度

1. **预热 Schema 索引**：启动时自动执行，无需手动操作
2. **开启结果缓存**：相同问题直接返回缓存（`use_cache=true`）
3. **使用连接池**：`DB_POOL_SIZE=20` 适合高并发
4. **本地 LLM**：Ollama + `qwen2.5:7b` 避免网络延迟

### 高并发场景

```env
WORKERS=4            # uvicorn worker 数（建议：CPU 核数 × 2）
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
CACHE_MAX_SIZE=1024
```

---

## 17. 生产部署建议

### 使用 Docker 部署

创建 `Dockerfile`：

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_data_agent/ ./ai_data_agent/
COPY .env .env

EXPOSE 8000 9090

CMD ["python", "-m", "ai_data_agent.main"]
```

```bash
# 构建镜像
docker build -t ai-data-agent .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  ai-data-agent
```

### 生产环境 .env 建议

```env
ENV=prod
DEBUG=false
LOG_JSON=true
LOG_LEVEL=INFO

# 使用强密钥
API_KEY=your-very-strong-random-key

# 生产数据库
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/agent
WAREHOUSE_URL=postgresql+asyncpg://user:pass@warehouse:5432/dw

# 关闭 Swagger UI（生产环境不暴露）
# 在 main.py 中 docs_url 已根据 is_prod 自动关闭

# 增加工作进程
WORKERS=4
```

### 安全检查清单

- [ ] 设置了强 `API_KEY`
- [ ] `SQL_READONLY=true`（除非明确需要写权限）
- [ ] `PYTHON_SANDBOX=true`
- [ ] 生产数据库使用专用只读账号
- [ ] `.env` 文件未提交到 Git（已加入 .gitignore）
- [ ] 日志中不包含敏感信息（API Key 等）

---

## 贡献指南

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/my-new-tool`
3. 编写代码和测试
4. 运行评估：`python -m pytest`
5. 提交 PR

## 许可证

MIT License

---


