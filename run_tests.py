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

# P4-2：分组按目录推导，避免新增测试文件被遗漏（原 integration 组只列了 3/7
# 个文件却报绿）。load 组指向的并非负载测试，已删除——真正的压测入口是
# run_load_test.py（独立脚本，不属于 pytest 套件）。
GROUPS: dict[str, list[str]] = {
    "unit": ["tests/unit"],
    "integration": ["tests/integration"],
    "evaluation": [
        "tests/unit/test_benchmark_dataset.py",
        "tests/integration/test_evaluation.py",
    ],
    "all": ["tests"],
}


def print_usage() -> None:
    print("用法: python3 run_tests.py [unit|integration|evaluation|all] [-v]")
    print("")
    print("示例:")
    print("  python3 run_tests.py unit")
    print("  python3 run_tests.py integration")
    print("  python3 run_tests.py all -v")
    print("")
    print("压测入口: python3 run_load_test.py --mode asgi --requests 200 --concurrency 50")


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
