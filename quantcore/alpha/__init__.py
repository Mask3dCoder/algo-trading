"""Alpha signal generators: momentum, pairs, stat-arb, residual, sentiment, FRED."""

from quantcore.alpha.fred_signals import (
    economic_momentum,
    financial_conditions,
    inflation_pressure,
    yield_curve_regime,
)
from quantcore.alpha.pairs_finder import CointegratedPair, PairsFinder
from quantcore.alpha.sentiment_momentum import SentimentMomentumStrategy
from quantcore.alpha.stat_arb_mean_reversion import StatArbMeanReversion

__all__ = [
    "CointegratedPair",
    "PairsFinder",
    "SentimentMomentumStrategy",
    "StatArbMeanReversion",
    "economic_momentum",
    "financial_conditions",
    "inflation_pressure",
    "yield_curve_regime",
]
