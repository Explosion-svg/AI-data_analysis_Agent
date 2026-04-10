#!/usr/bin/env python3
"""
run_tests.py

项目测试一键执行脚本。

设计目标：
- 给不熟 pytest 命令的使用者一个统一入口
- 支持按测试层级执行
- 尽量跨平台，优先使用当前 Python 解释器

用法示例：
    python3 run_tests.py unit
    python3 run_tests.py integration
    python3 run_tests.py infra
    python3 run_tests.py evaluation
    python3 run_tests.py all
    python3 run_tests.py all -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

GROUPS: dict[str, list[str]] = {
    "unit": ["tests/unit"],
    "integration": [
        "tests/integration/test_chat_api.py",
        "tests/integration/test_agent_loop.py",
        "tests/integration/test_main.py",
    ],
    "infra": [
        "tests/integration/test_warehouse.py",
        "tests/integration/test_vector_store.py",
        "tests/integration/test_assembler.py",
    ],
    "evaluation": [
        "tests/unit/test_benchmark_dataset.py",
        "tests/integration/test_evaluation.py",
    ],
    "all": ["tests"],
}


def print_usage() -> None:
    print("用法: python3 run_tests.py [unit|integration|infra|evaluation|all] [-v]")
    print("")
    print("示例:")
    print("  python3 run_tests.py unit")
    print("  python3 run_tests.py infra")
    print("  python3 run_tests.py all -v")


def ensure_pytest_available() -> bool:
    # 先尝试使用当前 Python 环境导入 pytest。
    result = subprocess.run(
        [sys.executable, "-c", "import pytest"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True

    print("未检测到 pytest。请先安装依赖：")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    return False


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        print_usage()
        return 0

    group = argv[1]
    verbose = "-v" in argv[2:] or "--verbose" in argv[2:]

    if group not in GROUPS:
        print(f"未知测试分组: {group}")
        print_usage()
        return 2

    if not ensure_pytest_available():
        return 1

    targets = GROUPS[group]
    cmd = [sys.executable, "-m", "pytest"]
    cmd.append("-vv" if verbose else "-q")
    cmd.extend(targets)

    print("即将执行测试：")
    print("  " + " ".join(cmd))
    print("")

    completed = subprocess.run(cmd, cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
