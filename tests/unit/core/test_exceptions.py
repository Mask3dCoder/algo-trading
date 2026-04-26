"""Unit tests for exception hierarchy."""

from __future__ import annotations

import pytest

from quantcore.core.exceptions import (
    BacktestError,
    BrokerError,
    ConfigError,
    DataError,
    DataSourceError,
    ExecutionError,
    FillError,
    LookaheadBiasDetected,
    OrderError,
    QuantCoreError,
    RiskLimitBreached,
    ValidationError,
)


def test_base_exception() -> None:
    err = QuantCoreError("test error", details={"key": "value"})
    assert str(err) == "test error"
    assert err.details["key"] == "value"


def test_exception_hierarchy() -> None:
    assert issubclass(ConfigError, QuantCoreError)
    assert issubclass(DataError, QuantCoreError)
    assert issubclass(DataSourceError, DataError)
    assert issubclass(ExecutionError, QuantCoreError)
    assert issubclass(OrderError, ExecutionError)
    assert issubclass(FillError, ExecutionError)
    assert issubclass(BrokerError, ExecutionError)
    assert issubclass(RiskLimitBreached, QuantCoreError)
    assert issubclass(ValidationError, QuantCoreError)
    assert issubclass(BacktestError, QuantCoreError)
    assert issubclass(LookaheadBiasDetected, BacktestError)


def test_risk_limit_breached() -> None:
    err = RiskLimitBreached(
        limit_name="max_drawdown",
        current_value=0.18,
        limit_value=0.15,
    )
    assert err.limit_name == "max_drawdown"
    assert err.current_value == 0.18
    assert err.limit_value == 0.15


def test_catch_by_base() -> None:
    with pytest.raises(QuantCoreError):
        raise DataError("data issue")
