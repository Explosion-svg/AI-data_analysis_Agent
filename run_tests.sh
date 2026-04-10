#!/usr/bin/env bash

# run_tests.sh
#
# Linux/macOS/Git Bash 下的一键测试入口。
# 实际执行逻辑委托给 run_tests.py，避免维护两套复杂脚本。

set -euo pipefail

python3 "$(dirname "$0")/run_tests.py" "$@"
