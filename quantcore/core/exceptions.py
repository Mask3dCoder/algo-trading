"""Exception hierarchy for QuantCore."""

from __future__ import annotations


class QuantCoreError(Exception):
    """Base exception for all QuantCore errors."""

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConfigError(QuantCoreError):
    """Configuration loading or validation error."""


class DataError(QuantCoreError):
    """Data fetch, parse, or quality error."""


class DataSourceError(DataError):
    """Error communicating with a data source."""


class DataQualityError(DataError):
    """Data quality check failed."""


class DataQualityViolation(DataQualityError):
    """Market data failed a hard numerical sanity check."""


class ExecutionError(QuantCoreError):
    """Order execution or broker communication error."""


class OrderError(ExecutionError):
    """Order validation or submission error."""


class FillError(ExecutionError):
    """Fill simulation or reconciliation error."""


class BrokerError(ExecutionError):
    """Broker adapter connection or protocol error."""


class RiskLimitBreached(QuantCoreError):
    """A risk limit has been exceeded."""

    def __init__(
        self,
        message: str = "Risk limit breached",
        *,
        limit_name: str = "",
        current_value: float = 0.0,
        limit_value: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.limit_name = limit_name
        self.current_value = current_value
        self.limit_value = limit_value


class ValidationError(QuantCoreError):
    """Data or parameter validation error."""


class BacktestError(QuantCoreError):
    """Backtest engine error."""


class LookaheadBiasDetected(BacktestError):
    """Lookahead bias detected in backtest logic."""
