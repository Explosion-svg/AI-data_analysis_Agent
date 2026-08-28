"""
reliability/fallback.py — 降级处理（Fallback Pattern）

职责：
  当主路径（primary）发生可预期的错误时，自动切换到备选路径（fallback），
  保证核心功能在部分故障下仍可用。

降级模式（Graceful Degradation）：
  - primary 成功 → 返回 primary 结果（正常路径）
  - primary 失败 + fallback 成功 → 返回 fallback 结果（降级路径，记录 WARNING）
  - primary 失败 + fallback 失败 → 重新抛出 primary 的原始异常

为什么抛出 primary 的异常而不是 fallback 的异常？
  - primary 失败是"根本原因"，fallback 失败只是"恢复也没成功"
  - 保留语义一致性：调用者期待的是 primary 的异常类型和消息
  - fallback 的失败原因记录在 ERROR 日志中，不丢失信息

适用场景：
  1. LLM 降级：主 LLM（GPT-4）失败 → 降级到本地模型
  2. 数据库降级：实时数据库失败 → 降级到缓存数据
  3. 规则降级：LLM 规划失败 → 降级到规则路由（ReAct 模式）
  4. 知识库降级：向量检索失败 → 降级到关键词检索

与 circuit_breaker 的区别：
  - with_fallback：单次调用级别，失败立即降级
  - circuit_breaker：跨调用级别，持续失败后才降级（暂停调用）

与 retry 的关系：
  建议先 retry（短暂故障可恢复），retry 全部失败后再 with_fallback（切换备选路径）

注意事项：
  - primary 和 fallback 都是零参数可调用对象（callable[[], Awaitable[Any]]）
  - 调用方负责通过闭包绑定参数（如 primary=lambda: call_llm(prompt)）
  - exceptions 参数控制哪些异常触发降级（默认所有 Exception）
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
    先尝试 primary，失败则执行 fallback；两者均失败则抛出原始异常。

    执行流程::
        ┌─────────────┐
        │ await primary() │
        └─────────────┘
              │ 成功
              ▼
        返回 primary 结果
              │ 失败（异常在 exceptions 内）
              ▼
        记录 WARNING 日志
        ┌─────────────┐
        │ await fallback() │
        └─────────────┘
              │ 成功
              ▼
        记录 INFO 日志，返回 fallback 结果
              │ 失败
              ▼
        记录 ERROR 日志，重新抛出 primary 的原始异常

    参数设计（primary/fallback 为零参数 callable）：
    - 设计为零参数而非接受任意参数，原因是参数绑定由调用方通过 lambda/functools.partial 完成
    - 例如：with_fallback(lambda: call_llm(prompt), lambda: call_local_model(prompt))
    - 这使 with_fallback 本身保持通用，不耦合具体函数签名

    exceptions 参数：
    - 只对指定类型的异常触发降级（默认所有 Exception）
    - 例如：exceptions=(ConnectionError, TimeoutError) 只对网络问题降级
    - 注意：这里是 tuple 类型（因为 except 语句只接受 tuple）

    Args:
        primary: 主路径的异步可调用对象（零参数 lambda 或 partial）
        fallback: 备选路径的异步可调用对象（零参数 lambda 或 partial）
        label: 操作标签（用于日志，如 "llm_generate"、"db_query"）
        exceptions: 触发降级的异常类型元组（默认 (Exception,)）

    Returns:
        primary 成功时返回 primary 的结果；primary 失败时返回 fallback 的结果

    Raises:
        原始 primary 的异常：当 primary 和 fallback 都失败时
        （注意：fallback 的异常被记录但不传播）
    """
    try:
        return await primary()
    except exceptions as e:
        # primary 失败，触发降级
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
            # fallback 也失败了，记录 ERROR 但抛出原始异常（primary 的 e）
            # 抛出 e 而不是 fe 的原因：
            # 1. 保留原始错误语义（调用方期待 primary 的错误类型）
            # 2. fallback 的失败原因已记录在日志中（fallback_error=str(fe)）
            logger.error(
                "fallback.also_failed",
                label=label,
                fallback_error=str(fe),
            )
            raise e  # 抛出原始异常，保留语义
