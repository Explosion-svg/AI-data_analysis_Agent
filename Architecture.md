# AI 数据分析 Agent 架构（8 Layer Mapping）

# 完整项目结构

最终代码结构：

```
ai_data_agent/

├── api/
│   └── chat_api.py

├──	config/
│   └── config.py

├── context/
│   ├── prompt_builder.py
│   ├── query_rewriter.py
│   └── schema_context.py

├── tools/
|	├──	base_tool.py
|	├──	tool_registry.py
|	├──	rag_tool.py
│   ├── sql_tool.py
│   ├── python_tool.py
│   ├── chart_tool.py
│   └── schema_tool.py

├── orchestration/
│   ├── planner.py
│   ├── executor.py
│   └── agent_loop.py

├── memory/
│   ├── conversation_memory.py
│   └── cache_memory.py

├── reliability/
│   ├── retry.py
│   ├── sql_guard.py
|	├──	circuit_breaker.py
|	├──	fallback.py
│   └── timeout.py

├── observability/
│   ├── logger.py
│   ├── tracer.py
│   └── metrics.py

├── model_gateway/
│   ├── base_model.py
│   ├── openai_model.py
│   └── router.py

├── infra/
│   ├── database.py
│   ├── warehouse.py
│   └── vector_store.py

└── main.py
```

# 一、整体架构图（8层）

```
┌─────────────────────────────────────────────┐
│                Context Management           │
│    Prompt / Query Rewrite / Session Context │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│                 Tool System                 │
│ SQL Tool | Python Tool | Chart Tool | RAG   │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│             Execution Orchestration         │
│ Planner | Tool Router | Agent Loop          │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│                State & Memory               │
│ Conversation | Schema Memory | Cache        │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│               Reliability Layer             │
│ Retry | Guardrails | SQL Safety | Timeout   │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│          Evaluation & Observability         │
│ Logging | Metrics | Tracing | Eval Dataset  │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│                 Model Gateway               │
│ OpenAI | Deepseek | Local LLM | Router      │
└─────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────┐
│                  infra Layer                │
│ Database | Data Warehouse | Vector DB       │
└─────────────────────────────────────────────┘
```

------

# 整体调用流程

用户请求进入系统后的 **完整执行链路**：

```
User
 ↓
API (chat_api)
 ↓
Agent Loop
 ↓
Planner (任务规划)
 ↓
Executor (执行计划)
 ↓
Tools (RAG / SQL / Python / Chart)
 ↓
Infra (数据库 / 向量库)
 ↓
Model Gateway (LLM)
 ↓
Memory (保存对话)
 ↓
Observability (日志 / tracing)
```

# 二、逐层设计

## 1.API层

```
api/
└── chat_api.py
```

## 职责

系统 **唯一的HTTP入口**

负责：

```
1 接收用户请求
2 参数校验
3 调用 Agent
4 返回结果
```

参考代码逻辑：

```
@router.post("/chat")
async def chat(req: ChatRequest):

    response = agent_loop.run(
        query=req.query,
        conversation_id=req.conversation_id
    )

    return response
```

## API层边界

API层 **只做三件事**

```
接收请求
调用agent
返回结果
```

绝对不要：

```
❌ 写 SQL
❌ 调用 RAG
❌ 写 Prompt
❌ 业务逻辑
```

# 2.Context Management（上下文管理）

作用：

> 管理 **LLM输入上下文**

组件：

```
context/
    prompt_builder.py
    query_rewriter.py
    schema_context.py
```

## Context层边界

只负责：

```
Prompt Engineering
```

绝不：

```
❌ 调用数据库
❌ 执行SQL
❌ 调用工具
```

### prompt_builder.py

职责：

```
构建最终 prompt
```

 例如：

```
system prompt
+
用户问题
+
RAG文档
+
数据库schema
+
历史对话
```

最终生成：

```
messages[]
```

参考示例代码：

```
def build_prompt(query, docs, schema, history):

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "system", "content": docs},
        {"role": "system", "content": schema},
    ]
```

### query_rewriter.py

职责：

```
优化用户查询
```

典型策略：

```
Query Rewrite
Multi Query
Keyword Extract
```

例如：

```
原始问题:
今年销售怎么样

Rewrite:
2024年销售额
2024年销售趋势
年度销售统计
```

作用：

```
提高RAG召回率
```

### schema_context.py

职责：

```
构建数据库 schema context
```

例如：

```
tables:
sales
orders
users
```

生成：

```
Table sales
columns:
- id
- amount
- date
```

提供给 LLM：

```
让LLM写SQL
```

## 3.Tool System（工具系统）

Agent 的能力层。

### 职责和边界

Tools **只做执行**

```
输入
执行
输出
```

不要：

```
❌ 做规划
❌ 写prompt
❌ 决定是否执行
```

AI 数据分析 Agent 必备工具：

### SQL Tool

```
执行SQL
```

### Python Tool

```
Pandas分析
```

### Chart Tool

```
生成图表
```

### Data Profiling Tool

```
数据统计
```

代码结构：

```
tools/
	rag_tool.py
    sql_tool.py
    python_tool.py
    chart_tool.py
    schema_tool.py
```

工具接口统一：

```
class Tool:

    name: str
    description: str

    def run(input):
        ...
```

#### tool Registry.py

 tools 的管理层

例如：

class ToolRegistry:

    def __init__(self):
        self.tools = {}
    
    def register(self, tool):
        self.tools[tool.name] = tool
    
    def get(self, name):
        return self.tools.get(name)
    
    def list_tools(self):
        return list(self.tools.values())
注册工具例如：

```
registry = ToolRegistry()

registry.register(SQLTool())
registry.register(RAGTool())
registry.register(PythonTool())
```

#### base_tool.py

```
class BaseTool:

​    name: str
​    description: str

​    def run(self, input):
​        pass
.......
```

#### rag_tool.py

职责：

```
文档检索
```

流程：

```
query
 ↓
embedding
 ↓
vector search
 ↓
rerank
 ↓
返回文档
```

返回：

```
docs
```

#### sql_tool.py

职责：

```
执行 SQL
```

输入：

```
sql
```

执行：

```
warehouse.execute(sql)
```

返回：

```
table
```

#### python_tool.py

职责：

```
执行数据分析代码
```

例如：

```
pandas
numpy
统计分析
```

输入：

```
python code
```

返回：

```
analysis result
```

#### chart_tool.py

职责：

```
生成图表
```

例如：

```
matplotlib
plotly
```

输出：

```
chart json
chart image
```

#### schema_tool.py

职责：

```
查询数据库schema
```

例如：

```
show tables
describe table
```

### 4. Orchestration（执行编排）

这是 **Agent的大脑**。

职责：

```
理解任务
↓
规划步骤
↓
调用工具
↓
迭代
```

组件：

```
orchestration/

    planner.py
    executor.py
    agent_loop.py
```

典型步骤，实际根据决策调用：

```
User: 今年销售趋势

Planner:
Step1 查询SQL
Step2 数据分析
Step3 ...
Step4 解释
Step5 ...
```

Agent Loop：

```
while not done:
    plan
    call tool
    observe
```

#### planner.py

职责：

```
任务规划
```

##### 例如：

用户问：

```
今年销售额趋势
```

planner生成：

```
1 查询sales表
2 计算月销售
3 生成图表
```

输出：

```
plan
```

#### executor.py

职责：

```
执行 plan
```

例如：

```
step1 sql_tool
step2 python_tool
step3 ...
```

#### agent_loop.py

职责：

```
控制整个Agent循环
```

例如ReAct 循环：

```
while not finished:

    think
    choose_tool
    execute
    observe
```

流程：

```
LLM -> action
action -> tool
tool -> observation
observation -> LLM
```

### 5. State & Memory（状态与记忆）

Agent **必须有状态**。

否则每轮都像失忆。

#### 例如：

```
用户：

去年销售

下一轮：

那今年呢
```

Agent必须记住：

```
去年销售 -> orders表
```

#### conversation_memory.py

职责：

```
存储对话历史
```

##### 例如：

```
user: 销售多少
assistant: 10万
user: 那去年呢
```

帮助 LLM 理解：

```
上下文
```

#### cache_memory.py

职责：

```
缓存结果
```

##### 例如：

```
SQL结果缓存
RAG结果缓存
```

减少：

```
LLM调用
数据库查询
```

### 6. Reliability Layer（可靠性层）

这是 **Agent安全系统**

否则：

- SQL乱删
- LLM hallucination
- 死循环

组件：

```
reliability/

    retry.py
    timeout.py
    guardrails.py
    sql_safety.py
    circuit_breaker.py
```

#### retry.py

职责：

```
失败重试
```

例如：

```
LLM失败
API失败
SQL失败
```

#### sql_guard.py

职责：

```
SQL安全
```

防止：

```
drop table
delete
update
```

只允许：

```
select
```

#### timeout.py

职责：

```
防止任务卡死
```

例如：

```
python tool
SQL query
```

#### circuit_breaker.py

防止：

```
外部服务崩溃
```

例如：

```
OpenAI API 挂了
```

系统自动：

```
停止调用
```

#### fallback.py

例如：

```
GPT4失败
```

自动切换：

```
GPT4o
```

或：

```
RAG失败
```

返回：

```
LLM直接回答
```

### 7.Evaluation & Observability

生产级Agent必须可观测。

组件：

```
observability/

    logger.py
    tracer.py
    metrics.py
```

#### logger.py

记录：

```
用户问题
SQL
tool调用
错误
```

#### tracer.py

记录：

```
Agent执行链
```

例如：

```
chat_api
 → agent_loop
 → planner
 → sql_tool
```

类似：

```
LangSmith
OpenTelemetry
```

#### metrics.py

统计：

```
token
latency
success rate
tool usage
```

监控：

| 指标         | 说明   |
| ------------ | ------ |
| token消耗    | 成本   |
| tool调用次数 | 效率   |
| 失败率       | 稳定性 |
| SQL执行时间  | 性能   |

### evaluation

```
evaluation/
    benchmark_dataset.py
    metrics.py
    eval_runner.py
```

用于：

```
评估Agent准确率
```

#### benchmark_dataset.py

存：

```
测试问题
标准答案
```

例如：

```
question
expected_sql
expected_answer
```

------

#### metrics.py

计算：

```
accuracy
tool success rate
sql correctness
```

------

#### eval_runner.py

运行评估：

```
for q in dataset:
   run_agent(q)
   evaluate()
```

### 8.Model Gateway（模型网关）

职责：

```
统一LLM接口
```

支持：

```
OpenAI
Deepseek
Claude
Local LLM
```

代码结构：

```
model_gateway/

    base_model.py
    openai_model.py
    deepseek_model.py
    model_router.py
```

#### base_model.py

抽象接口：

```
generate()
stream()
...
```



统一接口：

```
class LLM:

    def generate(prompt):
        ...
```

这样可以：

```
随时换模型
```

#### router.py

模型路由：

例如：

```
simple task -> gpt4o-mini
complex -> gpt4
code -> deepseek
```

### 9. Infra层

```
infra/
├── database.py
├── warehouse.py
└── vector_store.py
```

负责：

```
外部系统连接
```

#### database.py

传统数据库：

```
Postgres
MySQL
```

------

#### warehouse.py

数据仓库：

```
BigQuery
Snowflake
Clickhouse
```

------

#### vector_store.py

里面放RAG vector DB存储文档，schema vector DB用来存储表结构：

```
collection:
   docs
   schema
   ...
```

统一管理：

```
docs_index
schema_index
```

向量数据库：

```
Chroma
Milvus
Weaviate
```

### 10. main.py

系统启动入口。

例如：

```
FastAPI
依赖注入
Agent初始化
```



# 核心设计原则

### 分层隔离

- 各层职责单一，互不穿透

上述的代码可以参考，结合你的知识库发挥你的想象

向量库运行后创建