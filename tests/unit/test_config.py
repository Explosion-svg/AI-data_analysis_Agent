"""
tests/unit/test_config.py

配置模块单元测试。

主要验证：
- 默认配置值
- 环境变量覆盖
- 配置参数校验
"""

from __future__ import annotations

import pytest

from ai_data_agent.config.config import Env, Settings


def test_settings_default_values() -> None:
    # 验证未传环境变量时，配置对象能使用代码中的默认值。
    cfg = Settings()

    assert cfg.app_name == "AI Data Analysis Agent"
    assert cfg.env == Env.dev
    assert cfg.port == 8000


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # 验证环境变量会覆盖配置默认值。
    monkeypatch.setenv("APP_NAME", "Test Agent")
    monkeypatch.setenv("PORT", "9001")

    cfg = Settings()

    assert cfg.app_name == "Test Agent"
    assert cfg.port == 9001


def test_settings_temperature_validation() -> None:
    # 温度超出允许范围时，配置初始化应直接失败。
    with pytest.raises(ValueError):
        Settings(llm_temperature=3.0)
