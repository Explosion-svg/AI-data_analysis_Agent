"""
reliability/fallback.py — 降级处理
当主路径失败时，提供备选响应
"""
from __future__ import annotations

from typing import Any, Callable

from ai_data_agent.observability.logger import get_logger

logger = get_logger(__name__)


async def with_fallback(
    primary: Callable,
    fallback: Callable,
    *,
    label: str = "operation",
    exceptions: tuple = (Exception,),
) -> Any:
    """
    先尝试 primary，失败则执行 fallback。
    两者均失败则抛出原始异常。
    """
    try:
        return await primary()
    except exceptions as e:
        logger.warning(
            "fallback.triggered",
            label=label,
            error=str(e),
        )
        try:
            result = await fallback()
            logger.info("fallback.success", label=label)
            return result
        except Exception as fe:
            logger.error(
                "fallback.also_failed",
                label=label,
                fallback_error=str(fe),
            )
            raise e  # 抛出原始异常，保留语义
