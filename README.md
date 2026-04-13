# AI Data Agent

一个面向数据分析场景的多工具 AI Agent。用户输入自然语言问题，系统会自动完成上下文准备、工具选择、SQL 查询、数据分析、图表生成与结果解释。

这个项目的重点不只是“能回答问题”，而是把一个分析型 Agent 按较清晰的工程边界落成代码：`assembler` 负责装配，`orchestration` 负责编排，`tools` 负责执行能力，`memory` 负责会话与工作状态，`model_gateway` 负责模型路由，`reliability` 和 `observability` 负责稳定性与可观测性。

## Features

- 自然语言驱动的数据分析流程
- ReAct 风格的多轮工具调用编排
- SQL、Python、图表、Schema、RAG 工具协作
- 多模型接入与路由
- Conversation Memory + Work Memory 分层记忆
- 熔断、缓存、日志、指标等工程能力

## Architecture

核心请求链路如下：

1. `api/chat_api.py` 接收请求并校验参数
2. `assembler.py` 提供已装配完成的 `AgentLoop`
3. `orchestration/agent_loop.py` 执行 ReAct 主循环
4. `context/` 负责 query rewrite、prompt 构建、schema 上下文
5. `tools/` 执行 SQL、Python、图表、RAG 等工具
6. `memory/` 维护对话记忆和工作记忆
7. `model_gateway/` 选择并调用具体模型
8. `reliability/` 和 `observability/` 提供稳定性与监控支撑

当前项目还专门把“组件创建”和“业务运行”分开：

- `assembler.py` 是唯一推荐的组装入口
- `AgentLoop` 通过构造函数接收依赖，不在业务层自己实例化依赖

## Repository Layout

```text
ai_data_agent/
├── api/                # HTTP 入口
├── config/             # 配置管理
├── context/            # prompt、query rewrite、schema context
├── evaluation/         # 评估脚本
├── infra/              # DB / warehouse / vector store
├── memory/             # conversation memory / work memory / cache
├── model_gateway/      # LLM 适配器与路由
├── observability/      # 日志、trace、metrics
├── orchestration/      # planner / executor / agent loop
├── reliability/        # breaker / retry / timeout / SQL guard
├── tools/              # SQL / Python / chart / schema / RAG
├── assembler.py        # Composition Root
└── main.py             # 应用启动入口
```

## Requirements

- Python 3.10+
- 至少配置一种模型服务
  - OpenAI
  - DeepSeek
  - 本地兼容 OpenAI API 的模型服务

依赖列表见 `requirements.txt`。

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure

复制环境模板并填写至少一套模型配置：

```bash
cp .env.example .env
```

常见最小配置示例：

```env
OPENAI_API_KEY=your_key
OPENAI_DEFAULT_MODEL=gpt-4o
OPENAI_FAST_MODEL=gpt-4o-mini
```

如果你使用本地或其他提供方，按 `ai_data_agent/config/config.py` 中的字段名配置即可。

### 3. Run

```bash
uvicorn ai_data_agent.main:app --reload
```

默认接口：

- `POST /api/v1/chat`
- `GET /api/v1/health`
- `DELETE /api/v1/conversations/{conversation_id}`

## Example Request

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "今年每个月的销售额趋势怎么样？",
    "use_cache": true
  }'
```

## Memory Design

这个项目把记忆拆成两层：

- `conversation_memory`
  - 负责“聊过什么”
  - 保存近期原始对话、LLM 滚动摘要和 pinned facts
- `work_memory`
  - 负责“当前任务做到哪一步”
  - 保存 query rewrite、schema、工具步骤、最近 SQL、数据摘要等执行状态

这种拆法的目标是避免把所有内容都塞进一层历史里，导致 prompt 和状态边界混乱。

## Development Notes

- 推荐通过 `assembler.py` 装配组件，不要在业务层直接 `AgentLoop()`
- `agent_loop.py` 负责编排，不负责创建依赖
- `context`、`tools`、`memory` 内部逻辑应尽量保持单一职责

## Testing

如果环境已安装依赖，可以运行：

```bash
pytest
```

或：

```bash
python run_tests.py
```

## Related Docs

- `Architecture.md`
- `TESTING.md`
- `TESTING_BEGINNER.md`

## License

MIT
