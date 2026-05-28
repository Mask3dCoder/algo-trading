"""Universal strategy factory — configure any strategy from YAML, no code required."""

from quantcore.strategy_factory.config_schema import (
    ExecutionSpec,
    FilterSpec,
    RegimeFilterSpec,
    SignalSpec,
    StrategyFactoryConfig,
)
from quantcore.strategy_factory.configurable_strategy import ConfigurableStrategy
from quantcore.strategy_factory.signal_engine import SignalEngine

__all__ = [
    "ConfigurableStrategy",
    "ExecutionSpec",
    "FilterSpec",
    "RegimeFilterSpec",
    "SignalEngine",
    "SignalSpec",
    "StrategyFactoryConfig",
]
