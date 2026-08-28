"""
tools/python_tool.py — Python 代码沙盒执行工具（PythonTool）

职责：
  在受限的沙盒环境中执行 LLM 生成的 Python 代码，用于数据分析任务。
  典型用途：对 sql_query 返回的数据做统计分析（同比、环比、相关性等）。

沙盒安全机制（多层隔离）：
  1. 内置函数白名单（_SAFE_BUILTINS）：
     只允许数学/数据处理相关的内置函数（abs、sum、len 等），
     禁止危险内置（open、eval、exec、__import__ 直接调用等）。

  2. 自定义 __import__（_safe_import）：
     替换默认 __import__，只允许白名单模块导入。
     禁止：os、sys、subprocess、socket 等系统访问模块。
     允许：pandas、numpy、statistics、math、datetime 等数据分析模块。

  3. 独立命名空间（_build_sandbox_globals）：
     exec() 在自定义的 globs 字典中运行，不污染主程序命名空间。
     代码无法访问主程序中的变量（如数据库连接、API keys 等）。

  4. 超时保护（run_with_timeout）：
     防止 LLM 生成死循环（while True）或无限递归代码卡死 Agent。

约定（LLM 代码生成规范）：
  - 数据变量：`df`（pandas.DataFrame，由 Executor 注入 SQL 查询结果）
  - 输出变量：`result`（代码必须将最终结果赋给 result）
  - 中间输出：`print()`（标准输出会被捕获，包含在 text 中）

exec() 使用说明：
  _execute_code() 中使用 exec() 执行代码，这是有意为之（不是安全漏洞）：
  - exec() 被限制在 _build_sandbox_globals() 创建的受限命名空间内
  - __builtins__ 被替换为白名单副本（无法调用危险函数）
  - __import__ 被替换为 _safe_import（无法导入危险模块）
  - 标记 # noqa: S102 告知 Bandit 这是已知且受控的 exec 使用

redirect_stdout：
  contextlib.redirect_stdout(buf) 将 print() 输出重定向到 StringIO buffer，
  这样可以在不修改代码的情况下捕获 print 输出，包含在 ToolResult.text 中。

与 sql_tool 的协作：
  sql_query → ToolResult.data (list[dict]) → PythonTool._run(data=...) → df = pd.DataFrame(data)
  Executor._inject_data() 负责将 sql_query 的结果注入 python_analysis 的 data 参数。
"""
from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from typing import Any

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.reliability.timeout import run_with_timeout
from ai_data_agent.config.config import settings
from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)

# ── 沙盒安全配置 ──────────────────────────────────────────────────────────────

# 安全的内置函数白名单
# 原则：只允许纯计算函数（无副作用、不访问系统资源）
# 显式排除：open、input、eval、exec、compile、__import__、globals、locals
_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
    "pow", "print", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "str", "sum", "tuple", "type",
    "vars", "zip",
}

# 允许在沙盒内导入的安全模块白名单
# 选择标准：数据分析常用、无系统访问能力
# 禁止：os、sys、subprocess、socket、pathlib、shutil（可操作文件系统/网络）
_ALLOWED_MODULES = {
    "pandas", "pd",
    "numpy", "np",
    "statistics",
    "math",
    "collections",
    "itertools",
    "functools",
    "datetime",
    "re",
    "json",
    "csv",
    "io",
}


# ── 沙盒实现 ──────────────────────────────────────────────────────────────────

def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """
    自定义 __import__ 函数，只允许导入白名单内的模块。

    为什么需要替换 __import__：
    - 沙盒中的代码通过 `import os` 等语句最终都会调用 __import__
    - 将 __builtins__["__import__"] 替换为此函数后，所有 import 都经过此处过滤
    - 系统内置的 __import__ 可以导入任何模块，替换后只允许白名单

    base 提取逻辑：
    - `import pandas.core.frame` 时 name="pandas.core.frame"
    - base = name.split(".")[0] = "pandas"
    - 只检查顶层包名（不检查子模块），因为子模块不能绕过顶层包的限制

    Args:
        name: 模块名称（如 "pandas"、"os.path"）
        *args: 传给原始 __import__ 的其他参数（fromlist、level 等）
        **kwargs: 传给原始 __import__ 的关键字参数

    Returns:
        导入的模块对象

    Raises:
        ImportError: 尝试导入不在白名单内的模块
    """
    base = name.split(".")[0]
    if base not in _ALLOWED_MODULES:
        raise ImportError(f"Module '{name}' is not allowed in sandbox.")
    return __import__(name, *args, **kwargs)


def _build_sandbox_globals(extra_vars: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    构建沙盒执行的全局命名空间（globs 字典）。

    关键设计：
    1. __builtins__ 替换：
       - 不是 None（会导致一些基础操作无法工作）
       - 也不是完整的 __builtins__（太危险）
       - 而是从 _SAFE_BUILTINS 精心筛选的白名单副本
       - 白名单中额外加入 __import__ = _safe_import（必须在 __builtins__ 里才能被 import 语句找到）

    2. pandas/numpy 预注入：
       - 代码中可以直接使用 pd、pandas、np、numpy
       - 无需在沙盒代码中 import（import 也允许，两者都 OK）

    3. extra_vars（外部数据注入）：
       - 由 _run() 传入 {"df": pd.DataFrame(data)}
       - 代码中可以直接使用 df 变量
       - 不通过 import 机制，避免绕过 _safe_import

    __builtins__ 的两种形式：
    - 在主脚本中：__builtins__ 是模块对象（builtins 模块）
    - 在非主脚本中（包括 exec 上下文）：__builtins__ 是字典
    - 因此需要判断 isinstance(__builtins__, dict) 来选择不同的提取方式

    Args:
        extra_vars: 额外注入的全局变量（如 {"df": DataFrame}）

    Returns:
        沙盒全局命名空间字典
    """
    import pandas as pd
    import numpy as np

    # 只保留白名单 _SAFE_BUILTINS 里的内置函数，其他全部删掉
    # isinstance 检查处理 __builtins__ 可能是 dict 或 module 对象的情况
    builtins_dict: dict[str, Any] = (
        {k: __builtins__[k] for k in _SAFE_BUILTINS if k in __builtins__}  # type: ignore[index]
        if isinstance(__builtins__, dict)
        else {k: getattr(__builtins__, k) for k in _SAFE_BUILTINS if hasattr(__builtins__, k)}
    )
    # __import__ 必须放在 __builtins__ 内，Python exec 从这里查找它
    # 不能放在 globs 的顶层，因为 import 语句查找 __import__ 的机制是通过 __builtins__
    builtins_dict["__import__"] = _safe_import

    globs: dict[str, Any] = {
        "__builtins__": builtins_dict,
        "pd": pd,
        "pandas": pd,
        "np": np,
        "numpy": np,
    }
    if extra_vars:
        globs.update(extra_vars)
    return globs


async def _execute_code(
    code: str,
    extra_vars: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """
    在沙盒全局命名空间中编译并执行 Python 代码。

    执行流程：
    1. 构建沙盒命名空间（_build_sandbox_globals）
    2. 创建 StringIO 缓冲区（捕获 print 输出）
    3. redirect_stdout：将 sys.stdout 重定向到缓冲区
    4. compile + exec：编译并在沙盒命名空间中执行代码
    5. 提取 result 变量（约定代码将结果赋给 result）

    compile 的作用：
    - compile(code, "<sandbox>", "exec") 将字符串编译为代码对象
    - "<sandbox>" 是虚拟文件名（出现在 traceback 中，方便调试）
    - 先编译再 exec 可以提前捕获语法错误（SyntaxError）

    约定 result 变量：
    - 代码应该将最终输出赋给 result 变量
    - globs.get("result") 提取此变量（不存在时返回 None）
    - 如果代码没有 result，text 中只会有 stdout 输出

    # noqa: S102 说明：
    - Bandit 默认会警告 exec() 的使用（S102: use of exec）
    - 此处使用是安全的（沙盒已限制）且是必要的（动态代码执行核心功能）
    - noqa 注释告知 linter 跳过此检查

    Args:
        code: 要执行的 Python 代码字符串
        extra_vars: 注入沙盒命名空间的额外变量（如 {"df": DataFrame}）

    Returns:
        (stdout_output, result_variable) 元组
        - stdout_output: print() 输出的字符串
        - result_variable: 代码中 result 变量的值（不存在则为 None）

    Raises:
        任何代码执行中发生的异常（调用方负责捕获）
    """
    globs = _build_sandbox_globals(extra_vars)
    buf = io.StringIO()
    result = None
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<sandbox>", "exec"), globs)  # noqa: S102
        result = globs.get("result")  # 约定：最终结果赋给 `result` 变量
    except Exception:
        raise
    return buf.getvalue(), result


class PythonTool(BaseTool):
    """
    Python 代码沙盒执行工具，用于数据分析任务。

    工具名：python_analysis
    并发槽：ConcurrencyLimiter 的 "python_analysis" 桶
    超时：settings.python_exec_timeout 秒
    """

    @property
    def name(self) -> str:
        """返回工具名称 "python_analysis"。"""
        return "python_analysis"

    @property
    def description(self) -> str:
        """
        工具描述，明确说明可用变量和输出约定。

        关键信息：
        - `df` 变量来自前序 SQL 查询（Executor 自动注入）
        - `result` 变量是输出约定
        - print() 输出会被包含在结果中
        """
        return (
            "Execute Python code for data analysis using pandas and numpy. "
            "Available variables: `df` (pandas DataFrame from previous SQL query). "
            "Assign your final answer to `result`. Use print() for intermediate output."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """
        Python 工具的参数 JSON Schema。

        参数：
        - code（必填）：Python 代码字符串
        - data（可选）：数据记录列表
          → 通常不由 LLM 填写（Executor._inject_data 自动注入），
          → prompt 明确说明 "Do NOT include 'data' field"
        """
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use `df` for data, assign final result to `result`.",
                },
                "data": {
                    "type": "array",
                    "description": "Optional: list of records (dicts) to use as DataFrame `df`.",
                    "items": {"type": "object"},
                },
            },
            "required": ["code"],
        }

    async def _run(
        self,
        code: str,
        data: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> ToolResult:
        """
        在沙盒中执行 Python 代码，返回执行结果。

        执行流程：
        1. 验证 code 非空
        2. 如果有 data，创建 DataFrame 并注入沙盒（作为 df 变量）
        3. 带超时执行 _execute_code()
        4. 格式化输出（stdout + result 变量）

        结果格式化：
        - 如果 result 是 DataFrame → 转换为 markdown 表格 + dict records
        - 其他类型 → 直接 str() 转换
        - stdout 非空 → 前置 "Output:\n" 标签

        超时处理：
        - TimeoutError 返回 ToolResult(success=False, error=str(e))
        - 其他异常：捕获、记录日志、包含 traceback 在 error 字段中

        Args:
            code: Python 代码字符串
            data: 可选的数据记录列表（会被转换为 df=pd.DataFrame(data) 注入沙盒）
            **_: 忽略的额外参数

        Returns:
            ToolResult：
            - 成功：success=True, data=result, text=格式化输出
            - 失败：success=False, error=错误信息和 traceback
        """
        if not code.strip():
            return ToolResult(success=False, error="Empty code.")

        import pandas as pd

        extra_vars: dict[str, Any] = {}
        if data:
            # 将 list[dict] 转为 DataFrame，注入沙盒作为 df 变量
            extra_vars["df"] = pd.DataFrame(data)

        try:
            stdout, result = await run_with_timeout(
                _execute_code(code, extra_vars),
                timeout=settings.python_exec_timeout,
                name="python_tool",
            )
        except TimeoutError as e:
            # 超时：快速失败（fail-fast），不重试（代码逻辑问题，重试没意义）
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            # 代码执行错误：包含完整 traceback，方便 LLM 在下次尝试时修正
            tb = traceback.format_exc()
            logger.warning("python_tool.exec_error", error=str(e))
            return ToolResult(success=False, error=f"Execution error: {e}\n{tb}")

        # 格式化输出文本
        text_parts = []
        if stdout.strip():
            text_parts.append(f"Output:\n{stdout.strip()}")
        if result is not None:
            if isinstance(result, pd.DataFrame):
                # DataFrame 结果：转换为 markdown 格式（供 LLM）和 dict records（供下游工具）
                text_parts.append(f"Result (DataFrame):\n{result.to_markdown(index=False)}")
                result = result.to_dict(orient="records")
            else:
                text_parts.append(f"Result: {result}")

        return ToolResult(
            success=True,
            data=result,
            text="\n".join(text_parts) if text_parts else "Code executed successfully (no output).",
        )
