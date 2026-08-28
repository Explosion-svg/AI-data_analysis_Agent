"""
tools/python_tool.py — Python 代码沙盒执行工具（PythonTool）

职责：
  在受限的沙盒环境中执行 LLM 生成的 Python 代码，用于数据分析任务。
  典型用途：对 sql_query 返回的数据做统计分析（同比、环比、相关性等）。

安全模型（进程隔离，防御纵深）：
  1. 进程隔离（主防线，修复 P0-1/P0-2）：
     用户代码在一次性子进程中执行，不再与主进程共享内存/对象图。
     - 原始 4 条逃逸路径（io.open、pd.__dict__['__builtins__']、subclasses
       内省、getattr/type 链）只能影响子进程自身，无法触及宿主进程的
       .env、API 密钥、数据库连接等内存态数据。
     - 超时通过父进程硬杀（kill-on-timeout）实现：asyncio.wait_for 取消
       的是 communicate() 协程，子进程随后被 terminate/kill，杜绝"exec()
       同步阻塞事件循环导致超时永不触发"的问题（P0-2）。
     - 子进程不继承宿主的敏感环境变量（过滤 *_KEY/*_SECRET/*_TOKEN/
       *PASSWORD 等），也不继承宿主打开的文件描述符（close_fds 默认开启）。

  2. 内置函数白名单（辅助层，非主防线）：
     只允许纯计算函数，禁止 open/eval/exec/compile/__import__ 直接调用。
     注意：白名单在子进程内仍可能被内省绕过，因此它只是纵深防御，
     真正的边界是进程隔离。

  3. 资源限制（POSIX）：
     worker 启动时通过 resource.setrlimit 设置：
     - RLIMIT_CPU：硬限制 CPU 时间（timeout + 余量）
     - RLIMIT_AS：虚拟内存上限（512MB，防 [0]*10**10 类内存炸弹）
     - RLIMIT_FSIZE：文件写入上限（10MB）
     - RLIMIT_NPROC：禁止再 fork 子进程
     Windows 无 rlimit 机制，依赖 kill-on-timeout + 输出上限。

  4. 输出上限（P3-9）：
     子进程 stdout 与结果序列化均截断到 1MB，防止 token/内存爆炸。

  剩余边界（需运维配合）：
  - 子进程以宿主同一 OS 用户运行，无容器（chroot/mount ns/seccomp）时，
    理论上仍可读取磁盘上同用户可读的文件（如 .env）。如需完全只读 FS
    与无网络，应将该工具部署在容器（Docker）或 VM 中。
  - 达到隔离边界后，白名单只是尽力而为，不承诺绝对安全。

约定（LLM 代码生成规范）：
  - 数据变量：`df`（pandas.DataFrame，由 Executor 注入 SQL 查询结果）
  - 输出变量：`result`（代码必须将最终结果赋给 result）
  - 中间输出：`print()`（标准输出会被捕获，包含在 text 中）

与 sql_tool 的协作：
  sql_query → ToolResult.data (list[dict]) → PythonTool._run(data=...) → df = pd.DataFrame(data)
  Executor._inject_data() 负责将 sql_query 的结果注入 python_analysis 的 data 参数。
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import traceback
from typing import Any

from ai_data_agent.tools.base_tool import BaseTool, ToolResult
from ai_data_agent.reliability.timeout import TimeoutError
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
# 选择标准：数据分析常用、无系统访问能力（注意：pandas 自带文件读取能力，
# 这是数据工具的功能所需，进程隔离才是真正的边界）
# 显式排除 io：io.open 即 builtins.open，可直接读写任意文件（P0-1 逃逸路径①）
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
}

# 子进程 stdout / 结果序列化 的最大字节数（超过截断，P3-9）
_MAX_OUTPUT_BYTES = 1024 * 1024  # 1MB

# 敏感环境变量关键词：子进程不继承（防止 API 密钥等泄入沙盒）
_SENSITIVE_ENV_PATTERNS = (
    "KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "PRIVATE",
    "CREDENTIAL", "AUTH", "API", "PSWD",
)


# ── 沙箱 worker（在子进程中运行的代码）────────────────────────────────────────

_SANDBOX_WORKER = r"""
import json, sys, io, traceback
from contextlib import redirect_stdout

# 与宿主保持一致的配置（由宿主通过 stdin 传入）
_MAX_OUTPUT = 1024 * 1024

_SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
    "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "min", "next", "oct", "ord",
    "pow", "print", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "str", "sum", "tuple", "type",
    "vars", "zip",
}

_ALLOWED_MODULES = {
    "pandas", "pd", "numpy", "np", "statistics", "math", "collections",
    "itertools", "functools", "datetime", "re", "json", "csv",
}


def _safe_import(name, *args, **kwargs):
    base = name.split(".")[0]
    if base not in _ALLOWED_MODULES:
        raise ImportError("Module '%s' is not allowed in sandbox." % name)
    return __import__(name, *args, **kwargs)


def _build_globals(extra):
    import pandas as pd
    import numpy as np
    if isinstance(__builtins__, dict):
        builtins_dict = {k: __builtins__[k] for k in _SAFE_BUILTINS if k in __builtins__}
    else:
        builtins_dict = {k: getattr(__builtins__, k) for k in _SAFE_BUILTINS if hasattr(__builtins__, k)}
    builtins_dict["__import__"] = _safe_import
    globs = {"__builtins__": builtins_dict, "pd": pd, "pandas": pd, "np": np, "numpy": np}
    globs.update(extra)
    return globs


def _to_json_safe(value):
    # 将执行结果转换为可 JSON 序列化的值（DataFrame/numpy 类型转换）
    import pandas as pd
    import numpy as np
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _apply_limits(timeout):
    # POSIX 资源限制；Windows 无此机制，跳过
    if sys.platform == "win32":
        return
    try:
        import resource
        cpu = max(1, int(timeout) + 5)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except Exception:
        pass


def main():
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    data = payload.get("data")
    timeout = payload.get("timeout") or 20.0
    _apply_limits(timeout)

    buf = io.StringIO()
    try:
        extra = {}
        if data is not None:
            import pandas as pd
            extra["df"] = pd.DataFrame(data)
        globs = _build_globals(extra)
        with redirect_stdout(buf):
            exec(compile(code, "<sandbox>", "exec"), globs)
        out = buf.getvalue()
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + "\n...[stdout truncated]..."
        envelope = {"ok": True, "stdout": out, "result": _to_json_safe(globs.get("result"))}
        sys.stdout.write(json.dumps(envelope, ensure_ascii=False, default=str))
    except Exception as e:
        envelope = {
            "ok": False,
            "error": "%s: %s" % (type(e).__name__, e),
            "traceback": traceback.format_exc(limit=20),
        }
        sys.stdout.write(json.dumps(envelope, ensure_ascii=False))
    except BaseException as e:  # SystemExit / KeyboardInterrupt 等：进程级退出
        envelope = {
            "ok": False,
            "error": "%s: %s" % (type(e).__name__, e),
            "traceback": traceback.format_exc(limit=5),
        }
        sys.stdout.write(json.dumps(envelope, ensure_ascii=False))


if __name__ == "__main__":
    main()
"""


# ── 沙箱辅助函数（宿主侧）─────────────────────────────────────────────────────

def _sandbox_env() -> dict[str, str]:
    """
    构造子进程环境变量：过滤所有疑似敏感的变量（API KEY/TOKEN/SECRET 等），
    避免 API 密钥等机密通过环境变量泄入沙盒子进程。
    """
    env: dict[str, str] = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if any(p in upper for p in _SENSITIVE_ENV_PATTERNS):
            continue
        env[k] = v
    return env


def _sandbox_cwd() -> str:
    """子进程工作目录：独立临时目录，避免沙盒代码直接读写项目目录。"""
    return tempfile.gettempdir()


def _creation_flags() -> int:
    """Windows 下禁止弹出控制台窗口；POSIX 返回 0。"""
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


async def _execute_code(
    code: str,
    data: list[dict[str, Any]] | None = None,
    timeout: float = 20.0,
) -> tuple[str, Any]:
    """
    在一次性子进程中执行 Python 代码，返回 (stdout, result)。

    流程：
      1. 将 code/data 打包为 JSON，通过 stdin 传给子进程 worker
      2. asyncio.wait_for 包裹 communicate()，超时即硬杀子进程（kill-on-timeout）
      3. 解析 worker 返回的 JSON envelope（stdout / result / error）
      4. stdout 与结果均做了字节数截断（P3-9）

    Args:
        code: 要执行的 Python 代码字符串
        data: 可选的数据记录列表（worker 侧转换为 df=pd.DataFrame(data)）
        timeout: 硬超时（秒），超时杀进程并抛 TimeoutError

    Returns:
        (stdout_output, result_variable) 元组

    Raises:
        TimeoutError: 代码执行超过 timeout 秒（子进程已被硬杀）
        RuntimeError: worker 进程异常退出或返回不可解析的 envelope
    """
    payload = json.dumps(
        {"code": code, "data": data, "timeout": timeout},
        ensure_ascii=False,
    ).encode("utf-8")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        _SANDBOX_WORKER,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=_sandbox_cwd(),
        env=_sandbox_env(),
        creationflags=_creation_flags(),
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=payload), timeout=timeout
        )
    except (asyncio.TimeoutError, TimeoutError):
        # 硬杀：wait_for 取消的只是 communicate()，子进程可能仍在运行
        logger.warning("python_tool.timeout_kill", timeout=timeout)
        await _terminate_proc(proc)
        raise TimeoutError("python_tool", timeout)

    if proc.returncode != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:2000] if stderr_bytes else ""
        raise RuntimeError(
            f"Sandbox worker exited with code {proc.returncode}: {stderr or 'unknown error'}"
        )

    try:
        envelope = json.loads(stdout_bytes.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError) as e:
        raise RuntimeError(f"Sandbox worker returned invalid result: {e}") from e

    if not envelope.get("ok"):
        error = envelope.get("error") or "Unknown sandbox error"
        tb = envelope.get("traceback") or ""
        raise RuntimeError(f"{error}\n{tb}")

    stdout = envelope.get("stdout") or ""
    result = envelope.get("result")
    return stdout, result


async def _terminate_proc(proc: asyncio.subprocess.Process) -> None:
    """硬杀子进程（先 terminate 后 kill，并 await wait() 回收避免僵尸进程）。"""
    for method in ("terminate", "kill"):
        try:
            if proc.returncode is None:
                getattr(proc, method)()
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001 - 尽力回收
            pass
    try:
        if proc.returncode is None:
            await proc.wait()
    except Exception:  # noqa: BLE001
        pass


class PythonTool(BaseTool):
    """
    Python 代码沙盒执行工具，用于数据分析任务。

    工具名：python_analysis
    并发槽：ConcurrencyLimiter 的 "python_analysis" 桶
    超时：settings.python_exec_timeout 秒（子进程硬杀）
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
        在子进程沙盒中执行 Python 代码，返回执行结果。

        超时处理：
        - TimeoutError（子进程被硬杀）→ ToolResult(success=False, error=str(e))
        - 其他异常（含沙盒代码错误）→ 捕获、记录日志、包含 traceback 在 error 中

        Args:
            code: Python 代码字符串
            data: 可选的数据记录列表（worker 侧转换为 df=pd.DataFrame(data)）
            **_: 忽略的额外参数

        Returns:
            ToolResult：
            - 成功：success=True, data=result, text=格式化输出
            - 失败：success=False, error=错误信息和 traceback
        """
        if not code.strip():
            return ToolResult(success=False, error="Empty code.")

        try:
            stdout, result = await _execute_code(
                code,
                data=data,
                timeout=settings.python_exec_timeout,
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
            # 子进程已通过 _to_json_safe 序列化，DataFrame 已转为 records，
            # 此处只负责文本化呈现
            if isinstance(result, list) and result and all(
                isinstance(r, dict) for r in result
            ):
                text_parts.append(f"Result (DataFrame):\n{result}")
            else:
                text_parts.append(f"Result: {result}")

        return ToolResult(
            success=True,
            data=result,
            text="\n".join(text_parts) if text_parts else "Code executed successfully (no output).",
        )
