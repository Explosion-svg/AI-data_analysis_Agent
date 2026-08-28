# AI 数据助手 Agent 新手测试执行手册

## 1. 这份手册是给谁看的

如果你现在的状态是下面这种，这份文档就是写给你的：

- 会看一点 Python，但不熟 pytest
- 知道项目里有 `tests/`，但不知道怎么跑
- 看得懂命令，但不知道每条命令的意义
- 测试失败后，不知道先看哪里

这份手册不讲测试设计理论，重点只做一件事：

按顺序带你把项目测试跑起来，并知道结果是什么意思。

---

## 2. 先理解一件事：测试到底是什么

你可以把测试理解成“提前写好的自动检查题”。

例如：

- 如果缓存过期了，系统还继续返回旧值，就是错
- 如果危险 SQL 没有被拦截，就是错
- 如果 API 在参数错误时没有返回正确状态码，就是错

测试的作用，是在你改代码后，自动帮你检查这些“不能错的地方”。

---

## 3. 你现在项目里已经有什么测试

项目里现在已经有两类测试：

### 3.1 单元测试

目录：

```text
tests/unit/
```

它们主要检查：

- 配置
- 缓存
- 会话记忆
- SQL 安全
- 超时/重试/熔断
- Prompt 构建
- 工具层
- 编排层
- 模型路由
- 评估数据集

### 3.2 集成测试

目录：

```text
tests/integration/
```

它们主要检查：

- Agent 主循环
- Chat API
- 数据仓库访问
- 向量库访问
- 装配器
- 评估运行器
- 应用生命周期

---

## 4. 第一步：打开终端并进入项目根目录

你应该先进入项目根目录，也就是包含这些文件的目录：

- `README.md`
- `requirements.txt`
- `ai_data_agent/`
- `tests/`

如果你当前就在项目目录，可以执行：

```bash
pwd
```

如果输出类似下面路径，就对了：

```text
/mnt/e/study/program/program4
```

如果不对，先切换目录：

```bash
cd /mnt/e/study/program/program4
```

---

## 5. 第二步：确认 Python 是否可用

执行：

```bash
python3 --version
```

如果输出类似：

```text
Python 3.11.9
```

说明 Python 正常。

如果提示命令不存在，你需要先安装 Python。

---

## 6. 第三步：创建虚拟环境

这一步不是绝对必须，但强烈建议。

原因：

- 避免污染系统 Python
- 避免项目之间依赖冲突
- 出问题时更容易重建环境

执行：

```bash
python3 -m venv .venv
```

激活虚拟环境：

Linux/macOS/Git Bash：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

激活成功后，终端前面一般会出现：

```text
(.venv)
```

---

## 7. 第四步：安装依赖

执行：

```bash
python3 -m pip install -r requirements.txt
```

这一步会安装：

- FastAPI
- pytest
- pytest-asyncio
- httpx
- sqlalchemy
- aiosqlite
- chromadb
- pandas
- plotly

如果安装完成没有报错，就可以继续。

---

## 8. 第五步：先跑一条最简单的测试

不要一上来就跑全量。

先跑一个最稳定、最容易理解的测试文件：

```bash
python3 -m pytest -q tests/unit/test_memory.py
```

如果看到类似：

```text
3 passed
```

说明：

- Python 环境正常
- pytest 正常
- 项目导入路径正常
- 测试目录结构正常

这一步成功非常重要。

---

## 9. 第六步：按模块逐步测试

建议按下面顺序执行。

### 9.1 先跑纯逻辑测试

```bash
python3 -m pytest -q tests/unit
```

这一步主要验证：

- 纯 Python 逻辑
- 不依赖真实数据库或真实模型的模块

如果这一步都没过，不建议继续跑集成测试。

### 9.2 再跑接口和主流程

```bash
python3 -m pytest -q tests/integration/test_chat_api.py tests/integration/test_agent_loop.py
```

这一步主要验证：

- API 层
- Agent 主循环

### 9.3 再跑基础设施集成测试

```bash
python3 -m pytest -q tests/integration/test_warehouse.py tests/integration/test_vector_store.py tests/integration/test_assembler.py
```

这一步主要验证：

- 临时 sqlite 数据库是否工作正常
- 临时 ChromaDB 是否工作正常
- 装配器流程是否正确

### 9.4 再跑评估相关测试

```bash
python3 -m pytest -q tests/unit/test_benchmark_dataset.py tests/integration/test_evaluation.py
```

这一步主要验证：

- 评估数据集的增删改查
- 评估报告统计是否正确

### 9.5 最后跑全部测试

```bash
python3 -m pytest -q tests
```

这样做的好处是：

- 一旦失败，你更容易知道是在哪一层出的错

---

## 10. 每条命令是什么意思

例如：

```bash
python3 -m pytest -q tests/unit/test_memory.py
```

含义如下：

- `python3`
  - 使用 Python 3 运行命令
- `-m pytest`
  - 告诉 Python 执行 pytest 模块
- `-q`
  - 简洁模式，输出更短
- `tests/unit/test_memory.py`
  - 只跑这一份测试文件

再例如：

```bash
python3 -m pytest tests -vv
```

含义如下：

- `tests`
  - 跑整个测试目录
- `-vv`
  - 显示更详细的测试名称和过程

---

## 11. 如果测试通过，代表什么

如果你看到：

```text
40 passed
```

它的意思不是“项目绝对没问题”。

它真正的意思是：

- 当前已经写出来的测试场景，全部通过了
- 至少说明这些核心行为没有被破坏

这已经很有价值，因为它能帮你发现回归问题。

---

## 12. 如果测试失败，先不要慌

测试失败时，你先按这个顺序看。

### 12.1 先看失败的是哪个文件

例如：

```text
FAILED tests/unit/test_reliability.py::test_validate_sql_blocks_dangerous_sql
```

这说明问题大概率在：

- `sql_guard.py`
- 或者该测试预期和代码实现不一致

### 12.2 再看失败的是哪个函数

例如：

```text
test_cache_memory_hit_miss_ttl_and_lru
```

说明故障可能和：

- 缓存命中
- TTL 过期
- LRU 淘汰

有关。

### 12.3 再看断言失败内容

例如：

```text
E       assert 2 == 3
```

这种通常表示：

- 测试预期值和真实输出不一致

### 12.4 最后看 traceback

traceback 最重要的地方，一般是最后几行。

你重点看：

- 报错在哪个 `.py` 文件
- 报错在第几行
- 抛出的异常是什么

---

## 13. 常见问题最简单处理方式

### 13.1 `No module named pytest`

执行：

```bash
python3 -m pip install -r requirements.txt
```

### 13.2 `No module named chromadb`

执行：

```bash
python3 -m pip install chromadb
```

### 13.3 `No module named aiosqlite`

执行：

```bash
python3 -m pip install aiosqlite
```

### 13.4 `ModuleNotFoundError: ai_data_agent`

原因通常是：

- 你没有在项目根目录执行测试

先执行：

```bash
pwd
```

确认当前目录是项目根目录后再重试。

### 13.5 某些测试第一次能过，第二次失败

这类问题通常和全局状态有关，例如：

- 缓存未清
- memory 未清
- router 单例未清

本项目已经在 `tests/conftest.py` 里做了重置，但如果你后续新增新的全局单例，也要记得补进去。

---

## 14. 你最常用的 6 条命令

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

跑单元测试：

```bash
python3 -m pytest -q tests/unit
```

跑集成测试：

```bash
python3 -m pytest -q tests/integration
```

跑全部测试：

```bash
python3 -m pytest -q tests
```

查看详细输出：

```bash
python3 -m pytest tests -vv
```

只跑和 SQL 有关的测试：

```bash
python3 -m pytest -q tests -k sql
```

---

## 15. 一键脚本怎么用

如果你不想记 pytest 命令，现在也可以直接用项目根目录里的脚本。

推荐使用跨平台脚本：

```bash
python3 run_tests.py unit
python3 run_tests.py integration
python3 run_tests.py infra
python3 run_tests.py evaluation
python3 run_tests.py all
```

如果你在 Linux、macOS 或 Git Bash 下，也可以执行：

```bash
bash run_tests.sh all
```

说明：

- `unit`：只跑单元测试
- `integration`：跑 API、AgentLoop、main 生命周期相关测试
- `infra`：跑 warehouse、vector_store、assembler 测试
- `evaluation`：跑 benchmark_dataset、eval_runner 测试
- `all`：跑全部测试

如果想看详细输出：

```bash
python3 run_tests.py all -v
```

---

## 16. 推荐你第一次就这样做

第一次执行时，按下面顺序最稳妥：

1. `python3 -m pip install -r requirements.txt`
2. `python3 -m pytest -q tests/unit/test_memory.py`
3. `python3 -m pytest -q tests/unit`
4. `python3 -m pytest -q tests/integration/test_chat_api.py tests/integration/test_agent_loop.py`
5. `python3 -m pytest -q tests/integration/test_warehouse.py tests/integration/test_vector_store.py tests/integration/test_assembler.py`
6. `python3 -m pytest -q tests/unit/test_benchmark_dataset.py tests/integration/test_evaluation.py`
7. `python3 -m pytest -q tests`

如果你照这个顺序执行，基本能比较清楚地定位是哪一层出问题。

---

## 17. 配套文档

如果你已经会执行测试，但想进一步理解“每个测试为什么这么设计”，再看这份更完整的说明书：

[TESTING.md](/mnt/e/study/program/program4/TESTING.md)
