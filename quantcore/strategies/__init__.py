"""Trading strategies: framework, equity, options, macro."""

# Import sub-packages to trigger @register decorators at import time.
# Each sub-package's __init__.py imports its strategy modules, which
# call @register(...) at module level, populating the global registry.
from quantcore.strategies import (
    equity,  # noqa: F401
    macro,  # noqa: F401
    options,  # noqa: F401
)
from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    StrategyMetrics,
    StrategyState,
    StrategyStatus,
    StrategyType,
    TradeSignal,
)
from quantcore.strategies.registry import clear, create, get, list_strategies, register

__all__ = [
    # Base
    "Strategy", "StrategyConfig", "StrategyMetrics",
    "StrategyState", "StrategyStatus", "StrategyType", "TradeSignal",
    # Registry
    "register", "get", "list_strategies", "create", "clear",
]
