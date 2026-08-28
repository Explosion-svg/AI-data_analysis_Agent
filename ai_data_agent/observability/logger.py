"""
observability/logger.py — 结构化日志（Structlog）

职责：
  提供全局统一的结构化日志能力，支持：
  - 生产环境：JSON 格式（机器可读，方便日志收集系统解析）
  - 开发环境：彩色控制台格式（人类可读，方便开发调试）

为什么使用 structlog 而不是标准 logging：
  标准 logging 输出字符串（如 "2024-01-01 ERROR sql_guard: injection detected"），
  日志收集系统（ELK、Loki）需要用正则解析，容易出错。

  structlog 输出 JSON（如 {"ts":"2024-01-01", "level":"error", "event":"sql_guard.injection_detected",
  "sql":"SELECT..."}），日志收集系统可直接解析，支持按字段过滤和搜索。

结构化日志的核心优势：
  logger.warning("sql_guard.injection_pattern", sql=cleaned[:200])
  生成：{"event": "sql_guard.injection_pattern", "sql": "...", "level": "warning", "ts": "..."}
  可以直接在 Grafana 中按 event="sql_guard.injection_pattern" 过滤，
  或用 LogQL 查询 {event="sql_guard.injection_pattern"} | json | sql != ""

两阶段初始化策略：
  1. 模块导入阶段（_bootstrap_default_logging）：
     - 很多模块在导入时就执行 logger = get_logger(__name__)
     - 此时应用可能还未调用 configure_logging()
     - _bootstrap_default_logging 提供一个最小化的兜底配置
     - 保证模块级 logger 在正式配置前也可安全使用

  2. 应用启动阶段（configure_logging）：
     - 在 main.py 或 assembler.py 中调用一次
     - 覆盖 _bootstrap 的配置，应用生产级设置
     - json_logs=True（生产）或 False（开发）

contextvars 支持：
  structlog.contextvars.merge_contextvars 处理器自动将 structlog.contextvars
  中的变量（如 request_id、conversation_id）合并到每条日志中，
  无需在每次 logger.xxx() 调用时手动传入这些字段。
  在 AgentLoop.run() 中可以设置：
    structlog.contextvars.bind_contextvars(conversation_id=conversation_id)

cache_logger_on_first_use：
  - True（生产）：第一次使用后缓存 logger 实例，提升性能（避免重复构建处理器链）
  - False（开发/_bootstrap）：不缓存，确保配置变更立即生效
"""
from __future__ import annotations

import logging
import sys
from typing import Final

import structlog
from structlog.types import FilteringBoundLogger

# 日志是否已经初始化（避免重复配置）
_LOGGER_CONFIGURED: bool = False

# 默认日志级别（bootstrap 阶段使用）
_DEFAULT_LOG_LEVEL: Final[str] = "INFO"


def _bootstrap_default_logging() -> None:
    """
    为模块级 logger 提供安全的最小化兜底配置。

    触发时机：
    - 模块在导入阶段执行 logger = get_logger(__name__) 时
    - 如果 configure_logging() 尚未被调用
    - get_logger() 检测到 _LOGGER_CONFIGURED=False 时调用此函数

    为什么需要兜底而不是直接报错：
    - Python 模块导入是惰性的，但导入阶段的代码（如模块级 logger 创建）立即执行
    - 应用启动流程通常是：
        import modules → configure_logging() → start server
    - 在 configure_logging() 被调用前，如果有日志输出（如 settings 加载时），
      没有兜底配置会导致 structlog 内部状态不一致，可能抛出异常

    兜底配置特点：
    - 使用 ConsoleRenderer(colors=False)（纯文本，不依赖 termcolor）
    - cache_logger_on_first_use=False（不缓存，配置变更后立即生效）
    - 可以被后续的 configure_logging() 完全覆盖

    _LOGGER_CONFIGURED 检查确保只初始化一次（幂等操作）：
    - 如果已经被 configure_logging() 初始化过，直接返回
    - 防止多次调用时覆盖正式配置
    """
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
        processors=[
            structlog.contextvars.merge_contextvars,    # 合并上下文变量
            structlog.stdlib.add_logger_name,           # 添加 logger 名称
            structlog.stdlib.add_log_level,             # 添加日志级别
            structlog.processors.TimeStamper(fmt="iso", utc=True),  # ISO 时间戳
            structlog.dev.ConsoleRenderer(colors=False),  # 纯文本（兜底无颜色）
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, _DEFAULT_LOG_LEVEL.upper(), logging.INFO)
        ),
        context_class=dict,
        # logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,  # 不缓存，允许后续 configure_logging 覆盖
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, _DEFAULT_LOG_LEVEL.upper(), logging.INFO),
    )
    _LOGGER_CONFIGURED = True


def configure_logging(json_logs: bool = True, log_level: str = "INFO") -> None:
    """
    应用启动时调用一次，配置生产级或开发级日志格式。

    应在 main.py 或 assembler.py 中尽早调用（在任何 logger.xxx() 调用之前），
    确保所有后续日志都使用正式配置。

    处理器链（shared_processors）：
    1. merge_contextvars：合并 structlog.contextvars 中的上下文变量
       → request_id、conversation_id 等在每条日志中自动出现
    2. add_logger_name：添加 logger_name 字段（模块路径）
    3. add_log_level：添加 level 字段（"info"、"warning" 等）
    4. TimeStamper：添加 UTC ISO 8601 时间戳
    5. StackInfoRenderer：将 stack_info 转换为可读格式

    生产格式（json_logs=True）：
    - 添加 dict_tracebacks（将异常 traceback 转为 dict，JSON 友好）
    - 添加 JSONRenderer（最终输出 JSON 字符串）
    - 每条日志是一行 JSON，便于 ELK/Loki 解析

    开发格式（json_logs=False）：
    - 添加 ConsoleRenderer(colors=True)（彩色控制台输出）
    - 人类可读，包含颜色标注的级别、时间戳、事件名和结构化字段

    logging.basicConfig(force=True)：
    - force=True 强制接管 Python 标准 logging（覆盖已有的 handler）
    - 确保第三方库的 logging.getLogger().info() 输出也通过 structlog 格式化

    Args:
        json_logs: True 使用 JSON 格式（生产），False 使用彩色控制台（开发）
        log_level: 日志级别（"DEBUG"、"INFO"、"WARNING"、"ERROR"）
    """
    global _LOGGER_CONFIGURED
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,    # 合并上下文变量（request_id/conversation_id）
        structlog.stdlib.add_logger_name,           # 打印日志来自哪个 logger
        structlog.stdlib.add_log_level,             # 添加 level 字段
        structlog.processors.TimeStamper(fmt="iso", utc=True),  # ISO 8601 时间戳
        structlog.processors.StackInfoRenderer(),   # 解析堆栈信息
    ]

    if json_logs:
        # 生产环境：JSON 格式（机器可读）
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,   # 异常 traceback → dict
            structlog.processors.JSONRenderer(),    # 整体输出为 JSON 字符串
        ]
    else:
        # 开发环境：彩色控制台格式（人类可读）
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),  # 彩色控制台
        ]

    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),    # 兼容标准 logging
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        # logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,  # 生产环境缓存，提升性能
    )

    # 同步标准库 logging → structlog，接管 Python 整个标准日志系统
    # force=True：强制接管，即使已有 basicConfig 也会覆盖
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
        force=True,  # 强制接管全局
    )
    _LOGGER_CONFIGURED = True


def get_logger(name: str) -> FilteringBoundLogger:
    """
    获取指定名称的 structlog logger 实例。

    这是整个项目中获取 logger 的唯一入口，所有模块都通过此函数创建 logger：
        logger = get_logger(__name__)

    __name__ 作为 logger 名称的好处：
    - 日志中的 logger_name 字段自动显示模块路径（如 "ai_data_agent.tools.sql_tool"）
    - 方便通过模块路径过滤日志（如只看 reliability 模块的日志）

    懒加载初始化：
    - 第一次调用时如果 _LOGGER_CONFIGURED=False，自动调用 _bootstrap_default_logging()
    - 保证 logger 始终可用，不会因为初始化顺序问题崩溃

    FilteringBoundLogger 类型说明：
    - structlog.make_filtering_bound_logger 返回的类型
    - 支持 .debug()、.info()、.warning()、.error() 等方法
    - 支持关键字参数（结构化字段），如 logger.info("event", key=value)

    Args:
        name: logger 名称（通常传 __name__，即当前模块的全限定名）

    Returns:
        FilteringBoundLogger：structlog logger 实例，支持结构化日志记录
    """
    if not _LOGGER_CONFIGURED:
        _bootstrap_default_logging()
    return structlog.get_logger(name)
