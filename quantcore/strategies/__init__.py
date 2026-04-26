"""Trading strategies: framework, equity, options, macro."""

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    StrategyMetrics,
    StrategyState,
    StrategyType,
    TradeSignal,
)
from quantcore.strategies.registry import clear, create, get, list_strategies, register

__all__ = [
    # Base
    "Strategy", "StrategyConfig", "StrategyMetrics",
    "StrategyState", "StrategyType", "TradeSignal",
    # Registry
    "register", "get", "list_strategies", "create", "clear",
]
