"""Equity strategies: momentum, stat-arb, mean-reversion, trend, factor, vol, volume."""

from quantcore.strategies.equity.cross_sectional_momentum import CrossSectionalMomentumStrategy
from quantcore.strategies.equity.earnings_momentum import EarningsMomentumStrategy
from quantcore.strategies.equity.low_volatility import LowVolatilityStrategy
from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
from quantcore.strategies.equity.momentum import MomentumStrategy
from quantcore.strategies.equity.multi_factor import MultiFactorStrategy
from quantcore.strategies.equity.multifactor_composite import MultiFactorCompositeStrategy
from quantcore.strategies.equity.opening_gap_fade import OpeningGapFadeStrategy
from quantcore.strategies.equity.quality_factor import QualityFactorStrategy
from quantcore.strategies.equity.quality_factor_rmw_cma import QualityRMWCMAStrategy
from quantcore.strategies.equity.stat_arb import StatArbStrategy
from quantcore.strategies.equity.trend_following import TrendFollowingStrategy
from quantcore.strategies.equity.tsmom import TSMOMStrategy
from quantcore.strategies.equity.value_factor import ValueFactorStrategy
from quantcore.strategies.equity.value_factor_hml import ValueHMLStrategy
from quantcore.strategies.equity.volatility_breakout import VolatilityBreakoutStrategy
from quantcore.strategies.equity.volume_profile import VolumeProfileStrategy

__all__ = [
    "CrossSectionalMomentumStrategy",
    "EarningsMomentumStrategy",
    "LowVolatilityStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MultiFactorCompositeStrategy",
    "MultiFactorStrategy",
    "OpeningGapFadeStrategy",
    "QualityFactorStrategy",
    "QualityRMWCMAStrategy",
    "StatArbStrategy",
    "TrendFollowingStrategy",
    "TSMOMStrategy",
    "ValueFactorStrategy",
    "ValueHMLStrategy",
    "VolatilityBreakoutStrategy",
    "VolumeProfileStrategy",
]
