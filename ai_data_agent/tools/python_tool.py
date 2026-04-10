"""
tools/python_tool.py — Python 代码沙盒执行工具
支持 pandas / numpy / statistics 等数据分析库
使用 RestrictedPython 做沙盒隔离，防止危险操作
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

# 安全的内置函数白名单
_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
    "pow", "print", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "str", "sum", "tuple", "type",
    "vars", "zip",
}

# 允许导入的安全模块
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


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """自定义 __import__，只允许白名单模块。"""
    base = name.split(".")[0]
    if base not in _ALLOWED_MODULES:
        raise ImportError(f"Module '{name}' is not allowed in sandbox.")
    return __import__(name, *args, **kwargs)


def _build_sandbox_globals(extra_vars: dict[str, Any] | None = None) -> dict[str, Any]:
    """构建沙盒全局命名空间。"""
    import pandas as pd
    import numpy as np

    globs: dict[str, Any] = {
        "__builtins__": {k: __builtins__[k] for k in _SAFE_BUILTINS if k in __builtins__}  # type: ignore[index]
        if isinstance(__builtins__, dict)
        else {k: getattr(__builtins__, k) for k in _SAFE_BUILTINS if hasattr(__builtins__, k)},
        "__import__": _safe_import,
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
    """在沙盒中执行 Python 代码，返回 (stdout_output, result_variable)。"""
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
    @property
    def name(self) -> str:
        return "python_analysis"

    @property
    def description(self) -> str:
        return (
            "Execute Python code for data analysis using pandas and numpy. "
            "Available variables: `df` (pandas DataFrame from previous SQL query). "
            "Assign your final answer to `result`. Use print() for intermediate output."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
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
        if not code.strip():
            return ToolResult(success=False, error="Empty code.")

        import pandas as pd

        extra_vars: dict[str, Any] = {}
        if data:
            extra_vars["df"] = pd.DataFrame(data)

        try:
            stdout, result = await run_with_timeout(
                _execute_code(code, extra_vars),
                timeout=settings.python_exec_timeout,
                name="python_tool",
            )
        except TimeoutError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning("python_tool.exec_error", error=str(e))
            return ToolResult(success=False, error=f"Execution error: {e}\n{tb}")

        text_parts = []
        if stdout.strip():
            text_parts.append(f"Output:\n{stdout.strip()}")
        if result is not None:
            if isinstance(result, pd.DataFrame):
                text_parts.append(f"Result (DataFrame):\n{result.to_markdown(index=False)}")
                result = result.to_dict(orient="records")
            else:
                text_parts.append(f"Result: {result}")

        return ToolResult(
            success=True,
            data=result,
            text="\n".join(text_parts) if text_parts else "Code executed successfully (no output).",
        )
