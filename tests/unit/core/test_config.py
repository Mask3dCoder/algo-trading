"""Unit tests for configuration loader."""

from __future__ import annotations

from quantcore.core.config import get_config


def test_default_config_loads() -> None:
    cfg = get_config()
    assert cfg.database.host == "localhost"
    assert cfg.database.port == 5432


def test_database_name() -> None:
    cfg = get_config()
    assert cfg.database.name == "quantcore"


def test_redis_config() -> None:
    cfg = get_config()
    assert cfg.redis.port == 6379


def test_risk_limits_config() -> None:
    cfg = get_config()
    assert hasattr(cfg, "risk_limits")
