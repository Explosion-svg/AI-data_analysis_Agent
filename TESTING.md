# AI 数据助手 Agent 测试说明书

## 1. 文档目的

本文档用于说明本项目的测试目标、测试范围、测试环境准备、测试执行步骤、测试结果判定标准，以及常见问题处理方式。

适用对象：

- 项目开发者
- 测试人员
- 需要验证模块稳定性的维护人员

本文档重点回答以下问题：

- 这个项目为什么要测试
- 每个模块应该测什么
- 具体应该怎么运行测试
- 测试失败后应如何定位

---

## 2. 项目测试目标

本项目是一个 AI 数据助手 Agent，包含 API、上下文构建、工具调用、编排循环、缓存记忆、可靠性控制、模型路由等多个层次。

测试目标不是只验证“程序能跑”，而是验证以下几类能力：

1. 逻辑正确
   - 输入合法时，模块输出应符合预期
   - 输入非法时，模块应做出安全、稳定的处理

2. 模块隔离
   - 一个模块的测试失败，不应由另一个模块的外部依赖导致
   - 例如测试 SQLGuard 时，不应依赖数据库连接

3. 接口稳定
   - API 返回结构、状态码、鉴权逻辑应稳定

4. Agent 主流程可回归
   - 后续修改代码后，能够快速检查是否破坏了已有行为

5. 高风险逻辑重点覆盖
   - SQL 安全校验
   - 缓存与会话隔离
   - 工具执行与异常兜底
   - Agent ReAct 循环

---

## 3. 测试分层说明

本项目建议采用以下 3 层测试。

### 3.1 单元测试

单元测试只验证某个模块自身逻辑，不依赖真实外部服务。

特点：

- 运行快
- 定位准
- 最适合做基础回归

适合的模块：

- `config`
- `memory`
- `reliability`
- `context`
- `tool_registry`
- `planner`
- `executor`
- `router`

### 3.2 集成测试

集成测试验证多个模块之间是否能协同工作。

特点：

- 关注模块配合而不是单点逻辑
- 比单元测试更接近真实运行

适合的模块：

- `agent_loop`
- `chat_api`
- 后续可扩展到 `warehouse`、`vector_store`

### 3.3 回归测试

回归测试用于在代码改动后，快速确认核心能力未退化。

本项目当前的回归测试主要依赖：

- `tests/` 下的 pytest 用例
- `evaluation/` 下的评估脚本

说明：

- `tests/` 更偏工程行为验证
- `evaluation/` 更偏 Agent 效果评估

---

## 4. 当前测试文件说明

当前已新增的测试文件如下：

```text
tests/
├── __init__.py
├── conftest.py
├── helpers.py
├── unit/
│   ├── test_config.py
│   ├── test_context.py
│   ├── test_benchmark_dataset.py
│   ├── test_memory.py
│   ├── test_orchestration.py
│   ├── test_reliability.py
│   ├── test_router.py
│   └── test_tools.py
└── integration/
    ├── test_agent_loop.py
    ├── test_assembler.py
    ├── test_chat_api.py
    ├── test_evaluation.py
    ├── test_main.py
    ├── test_vector_store.py
    └── test_warehouse.py
```

### 4.1 `tests/conftest.py`

作用：

- 每次测试前重置全局单例
- 避免以下状态污染后续测试：
  - 缓存单例
  - 会话记忆单例
  - 模型路由单例
  - 工具注册表单例
  - 熔断器全局状态

这是本项目测试中非常关键的一个文件，因为项目本身大量使用模块级单例。

### 4.2 `tests/helpers.py`

作用：

- 放置测试专用的假对象
- 用于替代真实依赖

例如：

- `DummyBreaker`
- `DummyMemory`
- `DummyCache`
- `SequenceRouter`

这些对象的目的，是让测试聚焦当前模块，而不是去真实调用外部模型、真实缓存或真实熔断器。

---

## 5. 各测试文件覆盖内容

## 5.1 `tests/unit/test_config.py`

测试内容：

- 配置默认值是否正确
- 环境变量是否能覆盖默认配置
- `llm_temperature` 非法时是否报错

判定标准：

- 默认配置应符合 `config.py` 中定义
- 环境变量生效
- 非法配置必须被拒绝

---

## 5.2 `tests/unit/test_memory.py`

测试内容：

### 对话记忆 `ConversationMemory`

- 不同 `conversation_id` 是否隔离
- 超过最大轮数后是否裁剪旧消息
- `clear()` 是否清空会话
- `summary()` 是否返回正确信息

### 缓存 `CacheMemory`

- 未命中时是否返回 `None`
- 命中时是否返回正确值
- TTL 过期后是否失效
- 超出容量后是否按 LRU 淘汰

判定标准：

- 不允许不同会话串数据
- 不允许缓存过期后仍被使用
- 不允许新数据写入后淘汰策略错误

---

## 5.2.1 `tests/unit/test_benchmark_dataset.py`

测试内容：

- `BenchmarkDataset` 的新增、查询、筛选、长度统计
- 数据集保存到 JSON 后再读取
- 默认数据集是否包含内置样例

判定标准：

- 评估数据集必须可持久化、可恢复、可筛选

---

## 5.3 `tests/unit/test_reliability.py`

测试内容：

### SQL 安全校验

- 合法 `SELECT` 能通过
- `DROP`、`DELETE`、多语句、典型注入语句会被拦截

### 超时控制

- 正常协程在超时时间内完成
- 超时协程会抛出 `TimeoutError`

### 重试逻辑

- 前几次失败、最后一次成功时应自动重试

### 熔断器

- 连续失败达到阈值后进入 `OPEN`
- `OPEN` 时拒绝请求
- 恢复时间后允许半开重试
- 成功后恢复到 `CLOSED`

判定标准：

- 安全模块宁可误拦截，也不能漏拦截危险 SQL
- 可靠性模块必须在失败路径上行为明确

---

## 5.4 `tests/unit/test_context.py`

测试内容：

### PromptBuilder

- message 顺序是否正确
- 是否正确拼接历史、RAG 文档、Schema、用户问题

### QueryRewriter

- 模型返回合法 JSON 时，是否正确解析
- 返回非法内容时，是否降级为原始 query

### SchemaContextBuilder

- 优先走语义搜索选择相关表
- 语义搜索失败时退化为关键词匹配

判定标准：

- 上下文构建顺序必须稳定
- 降级逻辑必须存在，不能因为上游失败直接崩溃

---

## 5.5 `tests/unit/test_tools.py`

测试内容：

### ToolRegistry

- 工具注册
- 工具查询
- 导出 OpenAI tool schema

### BaseTool

- 正常执行时返回成功结果
- 工具内部抛异常时，包装为失败结果，不应直接炸出异常

### SQLTool

- 安全校验通过后执行 SQL
- 无 `LIMIT` 时自动补充行数限制
- 返回结果能序列化为列表

### PythonTool

- 传入 `data` 后能构建 DataFrame
- 能执行分析代码并读取 `result`
- 非法模块导入会被拒绝

### ChartTool

- 能生成 Plotly 图表 JSON
- 空数据时应返回失败

判定标准：

- 工具层不能把异常直接传播给上层 Agent
- 工具输出应可被 Agent 消费

---

## 5.6 `tests/unit/test_orchestration.py`

测试内容：

### Planner

- 能解析模型返回的 JSON 计划
- 能处理 markdown code fence
- 非法 JSON 时降级为空计划

### Executor

- 能根据依赖步骤传递数据
- 缺失工具时记录错误而不是崩溃

判定标准：

- 编排层要保证步骤流转稳定
- 错误必须落到 plan step，而不是整条链路直接中断

---

## 5.7 `tests/unit/test_router.py`

测试内容：

- 不同任务类型时，路由是否选择正确模型
- 主模型失败时，是否 fallback 到次级模型

判定标准：

- 路由策略必须可预测
- fallback 必须有效

---

## 5.8 `tests/integration/test_agent_loop.py`

测试内容：

- 缓存命中时是否直接返回
- 模型直接回答时，是否正确写入会话记忆
- 模型发起工具调用后，是否执行工具并继续总结

判定标准：

- Agent 核心循环必须能完成：
  - 上下文准备
  - 工具调用
  - 工具观察
  - 最终回答

这是本项目最重要的测试之一。

---

## 5.9 `tests/integration/test_chat_api.py`

测试内容：

- `/api/v1/chat` 正常返回
- 空 `query` 时返回 422
- 配置了 API Key 时，未带 token 返回 401
- `/api/v1/health` 返回健康状态

判定标准：

- API 层只负责接入和返回，不应混入业务逻辑
- 状态码和响应结构必须稳定

---

## 5.10 `tests/integration/test_warehouse.py`

测试内容：

- 使用临时 sqlite 文件初始化 `warehouse`
- 真实创建表并插入测试数据
- 验证真实查询、表名读取、schema 读取、样本读取

判定标准：

- 仓库层应能在真实数据库上完成最基础的数据访问闭环

---

## 5.11 `tests/integration/test_vector_store.py`

测试内容：

- 使用临时目录初始化 ChromaDB
- 写入文档向量与 schema 向量
- 验证 `search_docs()` 和 `search_schema()` 的返回

判定标准：

- 向量库层应具备最基础的 upsert 和检索能力

---

## 5.12 `tests/integration/test_assembler.py`

测试内容：

- 验证 `AppContainer.startup()` 各阶段执行顺序
- 验证 `health_report()` 返回结构

说明：

- 该测试主要验证装配器逻辑本身
- 不依赖真实 LLM、真实数据库连接或真实向量库

判定标准：

- 启动顺序必须稳定
- 诊断结构字段必须完整

---

## 5.13 `tests/integration/test_evaluation.py`

测试内容：

- 验证 `EvalRunner.run()` 是否能正确聚合多个 case 的结果
- 验证 `_run_case()` 在 Agent 抛错时的容错行为
- 验证 `_compute_report()` 的统计字段

判定标准：

- 评估层不能只“能跑”，还要保证统计结果正确

---

## 5.14 `tests/integration/test_main.py`

测试内容：

- 验证 `lifespan()` 是否正确调用 startup/shutdown
- 验证 `create_app()` 在生产环境下关闭 docs/redoc
- 验证全局异常处理器的 500 JSON 返回

判定标准：

- 应用入口层必须保证生命周期逻辑和异常处理逻辑稳定

---

## 6. 测试环境准备

## 6.1 Python 版本

建议使用：

- Python 3.11 或 3.12

不建议使用过旧版本，否则可能出现：

- `typing` 行为差异
- `pytest-asyncio` 兼容性问题
- 部分依赖安装失败

## 6.2 安装依赖

在项目根目录执行：

```bash
python3 -m pip install -r requirements.txt
```

如果系统没有 `python3`，可以先确认：

```bash
python3 --version
```

如果没有 pip，可以先安装 pip 再继续。

## 6.3 建议使用虚拟环境

建议执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Windows PowerShell 可执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

## 7. 如何执行测试

## 7.1 运行全部测试

```bash
python3 -m pytest -q tests
```

说明：

- `-q` 表示简洁输出
- `tests` 表示运行当前测试目录

## 7.2 运行单个测试文件

例如只运行内存测试：

```bash
python3 -m pytest -q tests/unit/test_memory.py
```

例如只运行 AgentLoop 测试：

```bash
python3 -m pytest -q tests/integration/test_agent_loop.py
```

例如只运行基础设施与装配器测试：

```bash
python3 -m pytest -q tests/integration/test_warehouse.py tests/integration/test_vector_store.py tests/integration/test_assembler.py
```

## 7.3 运行单个测试函数

例如只运行 SQL 安全测试：

```bash
python3 -m pytest -q tests/unit/test_reliability.py -k validate_sql
```

## 7.4 查看详细输出

如果测试失败，需要更多信息，可执行：

```bash
python3 -m pytest tests -vv
```

说明：

- `-vv` 表示更详细的测试名称和执行过程

---

## 8. 如何理解测试结果

## 8.1 全部通过

示例：

```text
32 passed in 2.10s
```

表示：

- 当前测试样例全部通过
- 不代表系统完全没有问题
- 只代表当前已覆盖的行为未退化

## 8.2 存在失败

示例：

```text
2 failed, 30 passed
```

表示：

- 有两个测试场景的行为与预期不一致
- 应根据失败信息定位到具体模块

建议处理顺序：

1. 看失败文件名
2. 看失败函数名
3. 看断言失败内容
4. 看 traceback 指向的业务代码

---

## 9. 常见失败与排查方法

## 9.1 `No module named pytest`

原因：

- 当前 Python 环境没有安装测试依赖

处理：

```bash
python3 -m pip install -r requirements.txt
```

## 9.2 `ModuleNotFoundError` 或导入失败

原因：

- 没在项目根目录执行测试
- 虚拟环境未激活

处理：

确认当前目录为项目根目录，再执行：

```bash
pwd
python3 -m pytest -q tests
```

## 9.3 异步测试报错

原因：

- `pytest-asyncio` 未安装
- 插件版本不匹配

处理：

```bash
python3 -m pip install pytest pytest-asyncio
```

## 9.4 API 测试失败

可能原因：

- `httpx` 未安装
- FastAPI 相关依赖版本不兼容

处理：

```bash
python3 -m pip install httpx fastapi
```

## 9.5 ChartTool 测试失败

可能原因：

- `plotly` 未安装

处理：

```bash
python3 -m pip install plotly
```

## 9.6 Vector Store 测试失败

可能原因：

- `chromadb` 未安装
- 临时目录无写权限

处理：

```bash
python3 -m pip install chromadb
```

## 9.7 Warehouse 集成测试失败

可能原因：

- `sqlalchemy` 或 `aiosqlite` 未安装
- sqlite 文件目录不可写

处理：

```bash
python3 -m pip install sqlalchemy aiosqlite
```

---

## 10. 模块测试设计原则

后续如果你继续扩展测试，应遵循以下原则。

### 10.1 测行为，不测实现细节

例如：

- 应测“缓存过期后失效”
- 不应强依赖内部变量名字

### 10.2 单元测试尽量不连外网

本项目包含模型调用能力，但单元测试中不应真实访问：

- OpenAI
- DeepSeek
- 本地模型服务

原因：

- 慢
- 不稳定
- 结果不可重复
- 需要密钥

### 10.3 高风险路径优先覆盖

优先覆盖：

- 安全
- 超时
- 重试
- fallback
- 异常处理

### 10.4 测试要可重复执行

同样的代码、同样的输入，多次运行测试应得到一致结论。

因此：

- 避免依赖当前时间、外网、随机模型输出
- 尽量使用假对象和固定返回值

---

## 11. 后续建议补充的测试

当前已经有一版基础测试骨架，并且已经补齐：

- `tests/integration/test_warehouse.py`
- `tests/integration/test_vector_store.py`
- `tests/integration/test_assembler.py`

后续仍建议继续增强：

### 11.1 Evaluation 回归测试

建议新增：

- `tests/integration/test_eval_runner.py`

测试目标：

- 批量评估结果计算
- 成功率、工具命中率、平均延迟统计正确

---

## 12. 推荐测试执行流程

如果你是日常开发，建议这样做。

### 12.1 改动纯逻辑模块后

例如改了：

- `memory`
- `reliability`
- `context`

执行：

```bash
python3 -m pytest -q tests/unit
```

### 12.2 改动 Agent 主流程后

例如改了：

- `planner`
- `executor`
- `agent_loop`
- `router`

执行：

```bash
python3 -m pytest -q tests/unit/test_orchestration.py tests/unit/test_router.py tests/integration/test_agent_loop.py
```

### 12.3 改动接口层后

例如改了：

- `chat_api.py`
- `main.py`

执行：

```bash
python3 -m pytest -q tests/integration/test_chat_api.py
```

### 12.4 改动基础设施层后

例如改了：

- `warehouse.py`
- `vector_store.py`
- `assembler.py`

执行：

```bash
python3 -m pytest -q tests/integration/test_warehouse.py tests/integration/test_vector_store.py tests/integration/test_assembler.py
```

### 12.5 提交代码前

建议至少执行：

```bash
python3 -m pytest -q tests
```

---

## 13. 测试结论标准

一个模块达到“可接受”的最低测试标准，至少应满足：

1. 正常路径有测试
2. 异常路径有测试
3. 边界条件有测试
4. 不依赖外网和真实大模型
5. 能重复运行

对于本项目，以下模块属于高优先级，必须长期保持回归测试：

- `ai_data_agent/reliability/sql_guard.py`
- `ai_data_agent/memory/cache_memory.py`
- `ai_data_agent/memory/conversation_memory.py`
- `ai_data_agent/tools/sql_tool.py`
- `ai_data_agent/tools/python_tool.py`
- `ai_data_agent/infra/warehouse.py`
- `ai_data_agent/infra/vector_store.py`
- `ai_data_agent/assembler.py`
- `ai_data_agent/orchestration/agent_loop.py`
- `ai_data_agent/api/chat_api.py`

---

## 14. 附录：推荐命令清单

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

运行全部测试：

```bash
python3 -m pytest -q tests
```

运行单元测试：

```bash
python3 -m pytest -q tests/unit
```

运行集成测试：

```bash
python3 -m pytest -q tests/integration
```

查看详细结果：

```bash
python3 -m pytest tests -vv
```

按关键字筛选：

```bash
python3 -m pytest -q tests -k sql
```

使用一键脚本跑全部测试：

```bash
python3 run_tests.py all
```

使用一键脚本跑基础设施测试：

```bash
python3 run_tests.py infra
```

查看详细输出：

```bash
python3 run_tests.py all -v
```

---

## 15. 最后说明

这份说明书描述的是“如何使用当前这套测试”。

它不是一次性的文档。后续如果你继续新增：

- 新工具
- 新模型适配器
- 新缓存策略
- 新 API 路由

都应该同步补两件事：

1. 新增对应测试文件或测试函数
2. 更新本说明书中的测试范围和执行说明

这样测试体系才不会失效。
