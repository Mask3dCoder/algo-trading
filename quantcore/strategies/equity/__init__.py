"""Equity strategies: momentum, stat-arb, mean-reversion, trend-following."""

from quantcore.strategies.equity.momentum import MomentumStrategy
from quantcore.strategies.equity.stat_arb import StatArbStrategy
from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
from quantcore.strategies.equity.trend_following import TrendFollowingStrategy

__all__ = [
    "MomentumStrategy",
    "StatArbStrategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy",
]
