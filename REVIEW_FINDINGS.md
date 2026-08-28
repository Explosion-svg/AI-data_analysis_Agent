# 项目问题清单（2026-08-28 全量审查）

> 审查方式：6 个独立代理分层深审（入口/API、Context/Memory、工具安全、编排/网关、可靠性/Infra/观测、测试/评测）+ 仓库层检查。
> 标注【实测】的问题均用项目自身代码复现验证过。测试套件当前 58/58 通过。
> 修复一项就勾掉一项：`- [ ]` → `- [x]`。

---

## P0 安全（最优先）

- [x] **P0-1 Python 沙箱可逃逸至宿主机 RCE**【实测】
  - 位置：`ai_data_agent/tools/python_tool.py:62-233`
  - 4 条逃逸路径均已复现：① `io` 在白名单里，`io.open` 就是 `builtins.open`（任意读写文件）；② 沙箱注入的真实 `pd`/`np` 模块对象，经 `pd.__dict__['__builtins__']['__import__']` 取回不受限解释器；③ `().__class__.__base__.__subclasses__()` 遍历拿到 `subprocess`（实测执行了系统命令并拿到回显）；④ 白名单含 `getattr`/`setattr`/`type`/`vars`，纯内省即可绕。
  - 结合 `/chat` 默认无鉴权（`config.py:73`，`chat_api.py:123`）+ RAG 文档注入通道，一条消息即可读走 `.env` 里的 API 密钥。
  - 修法：执行移出进程——一次性子进程/容器（无网络、只读 FS、rlimit/cgroup 限制），namespace 白名单无法修复此类逃逸。
  - 处理：`python_tool.py` 重构为一次性子进程执行（`asyncio.create_subprocess_exec` + JSON stdin/stdout 通道），主防线为进程隔离 + 敏感 env 过滤 + POSIX rlimit + 输出上限；白名单仅作纵深防御（移除 `io`）。4 条逃逸路径实测均无法触及宿主（密钥不泄入沙盒、宿主 env 不受影响）。

- [x] **P0-2 沙箱超时永远不会触发**【实测，两个代理独立确认】
  - 位置：`ai_data_agent/tools/python_tool.py:229` + `ai_data_agent/reliability/timeout.py:101`
  - `exec()` 同步阻塞事件循环，`asyncio.wait_for` 的定时器没有运行机会。实测 3 秒死循环在 1 秒超时设置下正常完成，无 `TimeoutError`。一个 `while True: pass` 请求冻结整个服务（所有请求、健康检查、指标全部停摆）。无内存上限（`[0]*10**10` 同样致命）。
  - 修法：进程隔离 + 硬 kill-on-timeout（与 P0-1 一并解决）；`asyncio.to_thread` 不够（线程无法强杀）。
  - 处理：子进程 + `wait_for(communicate())` 超时后 `terminate/kill` 硬杀；实测 1s 死循环 1.05s 被杀，事件循环全程保持活跃。

- [x] **P0-3 Demo 里 `eval()` 任意代码执行**
  - 位置：`Demo/mcp_server.py:8-11`
  - MCP 工具直接 `eval(expression)`，任何连接到 stdio server 的客户端可执行任意代码。属于复制粘贴陷阱。
  - 修法：删除，或换 `ast.literal_eval` / 安全表达式求值器。
  - 处理：替换为 AST 白名单求值器（仅数字字面量 + 四则运算），`__import__`/`open`/函数调用全部拒绝。

- [x] **P0-4 `/chat` 默认无鉴权**
  - 位置：`ai_data_agent/config/config.py:73`、`ai_data_agent/api/chat_api.py:123`
  - `api_key` 默认 `None` 时跳过鉴权，与 P0-1 组合即远程 RCE。
  - 修法：生产默认要求鉴权；API key 比较改用 `hmac.compare_digest`（`chat_api.py:126`，当前是非常数时间比较）。
  - 处理：config 增加 prod 强制鉴权校验（`env=prod` 且无 `api_key` 拒绝启动）；`chat_api.py` 改用 `hmac.compare_digest`。

---

## P1 仓库状态（GitHub 远端当前是坏的）

- [x] **P1-1 CI / Dockerfile / 12 个核心模块从未推送到 GitHub**
  - untracked：`.github/workflows/ci.yml`、`Dockerfile`、`.dockerignore`、`ai_data_agent/reliability/concurrency.py`、`ai_data_agent/memory/{factory,interfaces,redis_cache_memory,redis_conversation_memory,redis_work_memory}.py`、`ai_data_agent/orchestration/langgraph_agent_loop.py`、`ai_data_agent/context/request_context.py`、`run_load_test.py`
  - 后果：远端无 CI；`.env.example` 默认 `MEMORY_BACKEND=redis`，但 Redis 后端文件远端不存在——按默认配置启动直接 import 报错。
  - 修法：补 `git add` 后提交推送。

- [x] **P1-2 本地领先远端约 8000 行未提交**（96 文件，+7949/−1500）

- [x] **P1-3 80 个 `.pyc` + `.idea/` 已推送到 GitHub**
  - 修法：`git rm -r --cached` 清理 pycache 与 `.idea/`（`.gitignore` 已有规则但文件已被跟踪）。

- [x] **P1-4 git index 里有 3 个幽灵空文件**
  - `Demo/__init__/__init__.py`、`Demo/demo/__init__.py`、`ai_data_agent/orchestration/langgraph_agentloop.py` 呈 AD 状态（暂存后磁盘已删），直接 commit 会提交已删除的路径。
  - 修法：`git restore --staged` 清理。

- [x] **P1-5 `.gitignore` 的 `*.md` 全局忽略（仅 README 例外）**
  - 陷阱：以后任何文档（本清单、CHANGELOG、架构说明）都无法提交。
  - 修法：删掉 `*.md` / `!README.md` 规则；本文件就是受害者之一。

- [x] **P1-6 依赖声明与实际 import 不符**
  - `langgraph`（`langgraph_agent_loop.py:66`）和 `mcp`（`Demo/` 两文件）被 import 但不在 `requirements.txt`；`anthropic` 在 requirements 里但源码零 import。
  - 修法：删 `anthropic` 或接上；`langgraph`/`mcp` 视去留决定添加或删除模块。
  - 处理：删除死代码 `langgraph_agent_loop.py`（P4-6 决策）；requirements 移除 `anthropic`/`langgraph`、补充 `mcp`。

- [x] **P1-7 无 `.gitattributes`，CRLF/LF 警告刷屏**（Windows 开发典型问题）
  - 修法：加 `.gitattributes` 统一 `* text=auto eol=lf`。

---

## P2 生产可靠性

### 过载雪崩链（9→11 是连锁反应，需一起修）

- [ ] **P2-9 Router 对一切异常重试 + 全适配器 fallback**
  - 位置：`ai_data_agent/model_gateway/router.py:231, 298-331`
  - 默认 `exceptions=(Exception,)`（`retry.py:51`），本地信号量 1 秒超时（`ConcurrencyLimitExceeded`）也被当成模型失败触发全适配器扫射；最坏一次 Think 调 3×(1+适配器数) 次 API；最终报 `RuntimeError("All LLM adapters failed.")` 掩盖真实原因；每次失败还计入共享 `llm` 熔断器——5 次本地过载（LLM 本身健康）就让熔断器全局打开 60 秒。
  - 修法：`ConcurrencyLimitExceeded` 在 fallback 逻辑前捕获并原样抛出；重试限定 `(RateLimitError, APITimeoutError, APIConnectionError)`；退避 sleep 移出 `limit("llm")` 作用域；fallback 仅在供应商/传输错误时触发。

- [ ] **P2-10 过载返回 500 而非 503，429 不存在**
  - 位置：`ai_data_agent/tools/base_tool.py:249-268`、`ai_data_agent/orchestration/agent_loop.py`（`run()` 的兜底 except）
  - `ConcurrencyLimitExceeded`（RuntimeError 子类）被 `except Exception` 吞成 `ToolResult(success=False)` / `success=False`，`main.py:126` 的 503 处理器永远不触发；`concurrency.py:21,52` 文档声称返回 429——全仓库无任何 429。
  - 修法：在通用 except 前显式 `except ConcurrencyLimitExceeded: raise`；文档对齐。

- [ ] **P2-11 熔断器 HALF_OPEN 放行惊群**【实测】
  - 位置：`ai_data_agent/reliability/circuit_breaker.py:145-221`
  - `asyncio.Lock` 只保护 OPEN→HALF_OPEN 切换，不限制半开态准入。实测 OPEN 恢复期 10 个并发请求全部放行（10 个同时在飞、0 个拒绝），与模块自己的文档承诺（"HALF_OPEN 只允许一个试探请求"）相反。500 个排队请求会在恢复期同时打向未恢复的服务商并同步再跳闸。
  - 修法：锁内原子认领唯一试探槽位（`_probe_inflight` 标志或跨试探持锁），试探完成前其余 HALF_OPEN 请求按 `CircuitBreakerError` 拒绝。

### 数据与内存

- [ ] **P2-12 SQL 结果无硬上限**
  - 位置：`ai_data_agent/tools/sql_tool.py:154-155, 170-177`
  - LIMIT 注入条件是子串 `"limit"` 出现即跳过（列名 `limited`、注释 `/* limit */` 均可绕过）；`max_rows <= 0` 完全跳过（0 文档含义是"无限制"，且 `max_rows` 由 LLM 控制、无 schema 边界）。`SELECT * LIMIT 5000000` 全量 `fetchall()` + `to_dict` 内存翻倍 + `to_markdown` 逐行渲染进 LLM 观测——内存爆炸 + 上下文/token 成本爆炸。
  - 修法：Python 侧强制封顶（忽略超配置上限的 LIMIT、后截断 DataFrame、观测文本限字符数、`max_rows` 加 JSON Schema min/max）。

- [ ] **P2-13 SQL 表白名单可被绕过**【实测】
  - 位置：`ai_data_agent/reliability/sql_guard.py:76-79, 210-218`
  - 正则 `FROM|JOIN\s+[A-Za-z_]...` 提取不到逗号 join（`FROM allowed, secret` 只取到 `allowed`）和引号标识符（`FROM "secret"` 取到空）。开启 `sql_allowed_tables`（多租户隔离用途）时跨租户读数据。
  - 修法：改用 sqlparse token 树提取（含逗号列表和引号名归一化），或保守拒绝含引号标识符/逗号 join 的语句。

- [ ] **P2-14 Redis 一次瞬时故障永久失效**
  - 位置：`ai_data_agent/memory/redis_conversation_memory.py:162-174`（`redis_work_memory.py:251-263`、`redis_cache_memory.py:168-180` 同款）
  - `_safe_call` 的 fail-open 是单向的：`_available=False` 后只有成功路径能翻回 `True`，而成功路径已不可达。一次 Redis 超时后对话/缓存/工作记忆静默失效直到进程重启。
  - 修法：加恢复探测（`_available=False` 时周期性尝试 `ping()`，成功即复位 + 退避时间戳）。

- [ ] **P2-15 同步 redis-py 阻塞事件循环**
  - 位置：三个 `redis_*_memory.py`（`from_url` 同步客户端）；调用点 `agent_loop.py:225,271,320,347,451,463,476`、`chat_api.py:353-354`
  - ReAct 循环每次 work memory 变更跑同步 WATCH→GET→SET→EXEC 事务；`DELETE /conversations/{id}` 阻塞网络调用；Redis 抖动（2s socket timeout × retry）冻结整个 worker 数秒。
  - 修法：换 `redis.asyncio`，或全部包 `asyncio.to_thread`。

- [ ] **P2-16 ChromaDB 同步查询阻塞事件循环**
  - 位置：`ai_data_agent/infra/vector_store.py:26-28`（文档自称"不会阻塞"——不实）；调用点 `schema_context.py:197-201,261`、`rag_tool.py:170`
  - 每个请求的 schema 检索都在 loop 线程跑 HNSW + 磁盘查询。
  - 修法：`asyncio.to_thread` 包裹或提供 async 门面。

- [ ] **P2-17 数据库凭据写入日志**
  - 位置：`ai_data_agent/infra/database.py:120`、`warehouse.py:71`（INFO 级输出完整 URL，默认值含 `user:password`）
  - 生产环境密码进 JSON 日志（ELK/Loki 采集）。
  - 修法：`make_url(...).render_as_string(hide_password=True)` 或正则脱敏。

- [ ] **P2-18 进程内 conversation/work 存储无上限增长**
  - 位置：`ai_data_agent/memory/conversation_memory.py:169`、`work_memory.py:194`、Redis 变体的本地 dict（`redis_conversation_memory.py:112-117` 等）
  - 无 LRU/TTL/驱逐，按会话数线性膨胀（单会话状态 ~15-30KB）→ 长跑慢性 OOM。
  - 修法：会话 ID 维度 LRU 封顶；Redis 变体本地 dict 作为有界读缓存。

- [ ] **P2-19 缓存反序列化在 try 块外，毒化条目持续 500**
  - 位置：`ai_data_agent/memory/redis_cache_memory.py:77-86`（仅 `json.loads` 被守护，`_deserialize` 没有）；调用点 `agent_loop.py:223-234` 在 try 之外
  - 缓存 payload 与当前 `AgentResponse` 字段不匹配（发版后 schema 变更）→ 该 key 每次命中 TypeError → HTTP 500 直到 TTL 过期。
  - 修法：反序列化异常按 miss 处理；缓存查询移入错误处理路径。

### 生命周期

- [ ] **P2-20 优雅关闭只清理 2/5 类资源**
  - 位置：`ai_data_agent/assembler.py:188-191`
  - 只关 DB 和 warehouse；AsyncOpenAI 背后的 httpx 客户端、Redis 连接池、Chroma 客户端、模块单例（`_router`/`_memory`/`_cache` 等）全部泄漏。
  - 修法：适配器加 `aclose()`，`shutdown()` 统一调用并重置单例。

- [ ] **P2-21 启动中途失败零清理**
  - 位置：`ai_data_agent/main.py:75` + `assembler.py:185-186`
  - lifespan 在 `yield` 前抛异常则 `shutdown()` 不执行，且 `_started=False` 时 `shutdown()` 拒绝运行。K8s CrashLoop 重试累积引擎/连接。
  - 修法：跟踪已初始化组件，失败路径无条件清理。

- [ ] **P2-22 多 worker 下 3/4 Prometheus 指标静默丢失**
  - 位置：`ai_data_agent/assembler.py:219-229`（端口冲突吞掉且只记 DEBUG）；`main.py:19` 还推荐 `--workers 4`
  - 修法：冲突记 WARNING；用 `prometheus_client.multiprocess` 模式。

- [ ] **P2-23 内部异常原文返回给客户端**
  - 位置：`ai_data_agent/main.py:170-173`、`chat_api.py:263-267`
  - DB 连接错误会把 `postgresql://user:password@...` 回显在 HTTP 响应体里。
  - 修法：服务端记日志，客户端返回通用消息 + request_id。

- [ ] **P2-24 只读保护纯文本层，连接本身可写**
  - 位置：`ai_data_agent/reliability/sql_guard.py:139-170`（三层校验全 gated on `sql_readonly`，关掉即 DML/DDL 通行；`ATTACH 'evil.db'` 不带 `DATABASE` 关键字可绕过正则，目前仅靠 sqlparse 类型检查兜底）+ `warehouse.py:67`（引擎可写）
  - 修法：引擎层强制只读（SQLite `mode=ro` URI / Postgres 只读角色 / `SET default_transaction_read_only=on`）。

---

## P3 功能性 bug 与"虚假承诺"

- [ ] **P3-1 Multi-Query 多路检索从未接线**
  - `query_rewriter.py` 产出的 `all_queries`/`alternatives`/`keywords` 零消费者；`agent_loop.py:455-459` 读的 `reason` 键 `rewrite()` 从不返回；README 宣传的"多路并行检索"不存在。
  - 另：`query_rewriter.py:108` 裸 `json.loads` 不剥 ``` 围栏（`conversation_memory.py:646` 会剥）→ 围栏模型每次静默降级。

- [ ] **P3-2 QueryRewriter 把 OpenAI 模型名强塞给非 OpenAI 适配器**
  - 位置：`ai_data_agent/context/query_rewriter.py:104`
  - 无 `OPENAI_API_KEY` 的 DeepSeek/Ollama 部署下，请求带着 `model="gpt-4o-mini"` 发给 DeepSeek → 必 400 → 静默降级为原始 query，重写功能永久失效。
  - 修法：去掉 model 覆盖（SIMPLE 路由本就会选快速模型）。

- [ ] **P3-3 Planner"失败降级 ReAct"承诺失效**
  - 位置：`ai_data_agent/orchestration/planner.py:259-284`（step 解析在 try 外，模型漏 `"step"` 字段 → KeyError → 整请求 500）；`executor.py:130` 重复 step 号静默覆盖
  - 修法：step 构造纳入校验；`.get` + 类型纠正 + 去重重编号。

- [ ] **P3-4 Executor 部分失败 500 整个请求**
  - 位置：`ai_data_agent/orchestration/executor.py:160-173`（bare `gather`）、`:259`（步骤执行未守护，与 ReAct 路径 `agent_loop.py:672-680` 不对称）
  - 一步异常 → 全请求失败且不取消兄弟协程（孤儿步骤继续改状态）。
  - 修法：逐步 try/except 标记 `step.error`，或 `return_exceptions=True`。

- [ ] **P3-5 Planner/Executor/QueryRewriter 绕过注入的 router/breaker**
  - 位置：`planner.py:243,261`、`executor.py:126-127`、`query_rewriter.py:96,101` 全用全局 `get_router()`
  - 这些 LLM 调用无熔断保护（死供应商触发每请求全量重试扫射）；测试注入 mock router 时这些路径仍打真全局。
  - 修法：构造函数注入。

- [ ] **P3-6 Redis work memory 的 prompt 构建绕过 Redis**
  - 位置：`ai_data_agent/memory/redis_work_memory.py:76-83`（仅 `get_state`/`snapshot` 走 Redis）；继承的 `work_memory.py:552,609` 直读本地 dict
  - 重启后/跨 worker 时 `build_prompt_context` 返回空串——"跨进程共享"承诺不成立。
  - 修法：覆写两个方法走 Redis 加载路径。

- [ ] **P3-7 租户隔离可冒充**
  - 位置：`ai_data_agent/context/request_context.py:80`、`chat_api.py:133-165`
  - `X-Tenant-Id` 未校验字符集，`":"` 拼接有三方碰撞（`"a:b"+"c"` == `"a"+"b:c"`）；任何 key 持有者可声明任意租户读写/清除历史。
  - 修法：租户 ID 限 `[A-Za-z0-9_-]`，防碰撞 join，租户绑定到凭证。

- [ ] **P3-8 用户文本无围栏重注入 system 消息**
  - 位置：`prompt_builder.py:144-181`（work context/RAG/schema）、`work_memory.py:560`、`conversation_memory.py:529-540`（滚动摘要）
  - 用户文本进 system 角色（指令权重高于 user 轮），无任何"是数据不是指令"框架 → 注入面放大。
  - 修法：`<untrusted_context>` 式围栏 + 显式数据契约。

- [ ] **P3-9 沙箱 stdout/result 无上限**（`base_tool.py:248` 还会把完整参数 `str()` 后截断，大对象先全量序列化才切片，并把行级数据记入 debug 日志）

- [ ] **P3-10 config 默认值与 `.env.example` 矛盾**
  - `config.py:79,87` 默认 PostgreSQL（无 .env 时启动即失败），`.env.example` 说默认 SQLite；`env_file=".env"` 是 CWD 相对路径，换目录启动静默丢配置。
  - 修法：默认 SQLite + 绝对路径锚定项目根。

- [ ] **P3-11 观测项杂项**
  - Prometheus 指标：失败的 SQL 查询无处计数（`warehouse.py:131` 异常路径跳过）
  - tracer 无 shutdown/flush（SIGTERM 丢 ~5s 缓冲 span）
  - `metrics.py:53-55` 注释误导
  - 无 SQLite WAL/`busy_timeout`（两池同文件会 "database is locked"）
  - logger 配置调用两次（`main.py:64-68` + `assembler.py:210-215`），`cache_logger_on_first_use=True` 使第二次无效

---

## P4 测试与工程化

- [ ] **P4-1 CI 与实际环境脱节**
  - `ci.yml` 用 3.12，本地实际 3.13；`py_compile` 只查语法查不出缺依赖（langgraph/mcp 事故证明）；裸 `pytest` 从根目录收集，`run_load_test.py` 匹配 `*_test.py` glob 会被 import；requirements 全 `>=` 无锁定，不可复现。
  - 修法：`pytest.ini` 定 `testpaths = tests`；版本对齐；加 import 冒烟测试；锁定依赖。

- [ ] **P4-2 `run_tests.py` 分组已过期**：`integration` 组只跑 3/7 个集成文件却报绿；`load` 组指向的根本不是负载测试。修法：按目录推导或删除。

- [ ] **P4-3 conftest 单例重置不完整**：Chroma `_client`、warehouse 引擎、request_context ContextVar、metrics 单例未重置；`test_vector_store.py` 不关 Chroma 客户端（Windows 下锁 tmp 目录）。死代码 `event_loop` fixture 删除。

- [ ] **P4-4 零测试覆盖区**：真实 OpenAI 客户端（`openai_model.py`，含重试/流式/错误映射——最易碎的集成路径）、整个 observability 层、`rag_tool`/`schema_tool`、`work_memory_summarizer`、`request_context`。

- [ ] **P4-5 评测子系统可信度**
  - `eval_runner.py:227` agent 构造在 try 外（文档承诺"单用例失败不影响其他"失效）；`benchmark_dataset.py` 的 `expected_sql`/`expected_answer` 收集了但从不评估；"每用例独立 AgentLoop"注释不实（单例）；无离线模式（必打真实 LLM，不可进 CI）。
  - 修法：构造移入 try + `agent_factory` 注入；实现或删除未评估字段。

- [ ] **P4-6 `langgraph_agent_loop.py`（~1400 行死代码）建议删除**
  - 零引用、依赖未声明、无法 import；含真实 bug：超迭代路由到 `force_summarize` 时尾随 `tool_calls` 未被 tool 消息闭合 → OpenAI API 必 400（`langgraph_agent_loop.py:953` vs 手写循环 `agent_loop.py:390-418` 无此问题）；"与原循环等价"的文档说明不实。

- [ ] **P4-7 其他死代码/杂项**
  - `infra/database.py` 全死重（`get_session`/`get_connection` 零调用者）却把持 30 连接池、失败还能阻断启动；`close_db()` 不清 `_session_factory`
  - `with_fallback`（`fallback.py:45`）、`with_timeout` 装饰器（`timeout.py:107`）、`trace_async`/`get_current_span` 零使用
  - `python_sandbox` 配置项（`config.py:188`）零引用，给运维虚假安全感
  - CORS 硬编码 `allow_origins=["*"]` + `allow_credentials=True`（CORS 规范禁止的组合）且不可配置
  - `datetime.utcnow()` 弃用 ×36 警告（`conversation_memory.py:81`、`work_memory.py:61` 等）→ `datetime.now(timezone.utc)`
  - `conversation_memory.py:157` 文档"默认 10 轮" vs 实际 20
  - LRU 覆盖 bug：`cache_memory.py:184-196` 覆盖已有 key 不 `move_to_end`，容量满时热 key 覆盖误驱逐无辜条目
  - `timeout.py:52,102` 类名遮蔽内建 `TimeoutError`（3.13 两者同体），错误归属错乱
  - `main.py:194-201` `reload=True` 静默忽略 `workers`
  - `chat_api.py:173-177` 声明 400 实际从不产生（校验失败是 422），错误体两种 shape 不一致
  - `conversation_id` / `X-User-Id` / `X-Tenant-Id` 无长度上限（多 MB 输入进 dict key 和日志）
  - `agent_loop.py:224-234` 缓存命中路径跳过指标记录和历史写入，且无 singleflight（并发同 query 双跑）
  - `router.py:317-324` fallback 丢 `stop`/`top_p` 配置、SIMPLE 任务丢失快速模型选择
  - `executor.py:385-386` 假设 step 有序
  - `base_model.py:123` 的 `LLMConfig.timeout` 死配置（`openai_model.py:179-193` 从不传给 API 调用）
  - `query_rewriter.py:112-117` CJK 关键词切分无效（整句成一个"关键词"）
  - `conversation_memory.py:367-377` 同角色连续轮（失败重试）会使配对摘要错位
  - `request_context.py:141-146` `clear(None)` 覆盖而非 reset
  - `redis_conversation_memory.py:259-282` 并发合并后 `recent_turns` 不再截断（可达 2x 预算）+ `_versions` 无界泄漏
  - `circuit_breaker.py:264` 迟到失败持续推迟 HALF_OPEN；`:114-115` `or` 语义吞掉显式 0；`reset()` 无锁变更
  - `concurrency.py:168-190` 单例不跨事件循环重置（测试态）；`sem._value` 私有属性访问

---

## 建议修复顺序

1. **P1 仓库状态**（半天）——远端是坏的，后面每次修复都没有落点
2. **P0 安全**——子进程沙箱 + 真超时 + 鉴权
3. **P2 过载雪崩链**（P2-9/10/11 连锁，一起修）
4. 其余 P2 按清单推进，每修一类补对应测试（最缺：真实 OpenAI 客户端 mock 测试、observability 层）
5. P3/P4 穿插进行

## 已确认无问题的部分（避免误修）

- 测试套件本身：58 个行为型测试、依赖造假纪律好、无隐藏外部服务依赖
- ReAct 主循环：迭代上限、tool_call_id 协议、消息序列、memory 回写均正确
- 历史截断不会破坏 function-calling 配对（只持久化 user/assistant 文本轮）
- 缓存 key：SHA-256 + 租户隔离，无跨租户泄漏
- sql_guard 堆叠查询拦截（`SELECT 1; DROP TABLE x` 正确阻断，SQLite 驱动还有单语句兜底）
- SQL 审计日志（`sql_tool.py:186-233`）：含请求上下文/结果/表提取，不落全量 SQL 和数据
- chart_tool 输出纯 JSON（`to_json`），无 XSS/路径穿越面
- engine 单例 + 全异步驱动（无同步驱动混用）；retry 有界有抖动；`limit()` finally 释放
