"""Equity strategies: active survivors + archived shims."""

# Active survivors (passed honest sweep ci_lower > 0)
from quantcore.strategies.equity.full_universe_cross_sectional_momentum import FullUniverseCrossSectionalMomentum
from quantcore.strategies.equity.tsmom import TSMOMStrategy

# Archived — imported via compatibility shims for backward compat
from quantcore.strategies.equity.cross_sectional_momentum import CrossSectionalMomentumStrategy
from quantcore.strategies.equity.earnings_momentum import EarningsMomentumStrategy
from quantcore.strategies.equity.idiosyncratic_momentum import IdiosyncraticMomentumStrategy
from quantcore.strategies.equity.low_volatility import LowVolatilityStrategy
from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
from quantcore.strategies.equity.momentum import MomentumStrategy
from quantcore.strategies.equity.multi_factor import MultiFactorStrategy
from quantcore.strategies.equity.multifactor_composite import MultiFactorCompositeStrategy
from quantcore.strategies.equity.opening_gap_fade import OpeningGapFadeStrategy
from quantcore.strategies.equity.quality_factor import QualityFactorStrategy
from quantcore.strategies.equity.quality_factor_rmw_cma import QualityRMWCMAStrategy
from quantcore.strategies.equity.short_term_reversal import ShortTermReversalStrategy
from quantcore.strategies.equity.stat_arb import StatArbStrategy
from quantcore.strategies.equity.trend_following import TrendFollowingStrategy
from quantcore.strategies.equity.value_factor import ValueFactorStrategy
from quantcore.strategies.equity.value_factor_hml import ValueHMLStrategy
from quantcore.strategies.equity.volatility_breakout import VolatilityBreakoutStrategy
from quantcore.strategies.equity.volume_profile import VolumeProfileStrategy
from quantcore.strategies.equity.intraday_breakout import IntradayBreakoutStrategy  # noqa: F401
from quantcore.strategies.equity.intraday_mean_reversion import IntradayMeanReversionStrategy  # noqa: F401
from quantcore.strategies.equity.intraday_momentum import IntradayMomentumStrategy  # noqa: F401
from quantcore.strategies.equity.intraday_trend import IntradayTrendStrategy  # noqa: F401

__all__ = [
    # Active
    "FullUniverseCrossSectionalMomentum",
    "TSMOMStrategy",
    # Archived
    "CrossSectionalMomentumStrategy",
    "EarningsMomentumStrategy",
    "IdiosyncraticMomentumStrategy",
    "LowVolatilityStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MultiFactorStrategy",
    "MultiFactorCompositeStrategy",
    "OpeningGapFadeStrategy",
    "QualityFactorStrategy",
    "QualityRMWCMAStrategy",
    "ShortTermReversalStrategy",
    "StatArbStrategy",
    "TrendFollowingStrategy",
    "ValueFactorStrategy",
    "ValueHMLStrategy",
    "VolatilityBreakoutStrategy",
    "VolumeProfileStrategy",
]
