"""Macro strategies: active survivors + archived shims."""

# Active survivors (passed honest sweep ci_lower > 0)
from quantcore.strategies.macro.cta_trend import CTATrendStrategy
from quantcore.strategies.macro.risk_parity import RiskParityStrategy

# Archived — imported via compatibility shims for backward compat
from quantcore.strategies.macro.carry import CarryStrategy
from quantcore.strategies.macro.carry_trend_filter import CarryTrendFilterStrategy
from quantcore.strategies.macro.cross_sectional_factor_neutral import CrossSectionalFactorNeutralStatArb
from quantcore.strategies.macro.cross_sectional_momentum import CrossSectionalMomentumStrategy
from quantcore.strategies.macro.factor_residual import FactorResidualStrategy
from quantcore.strategies.macro.factor_residual_qp import FactorResidualQPStrategy
from quantcore.strategies.macro.factor_residual_strict import FactorResidualStrictStrategy
from quantcore.strategies.macro.factor_residual_v3 import FactorResidualStrategyV3
from quantcore.strategies.macro.fred_regime_strategy import FREDRegimeStrategy
from quantcore.strategies.macro.regime_adaptive_momentum import RegimeAdaptiveMomentumStrategy
from quantcore.strategies.macro.state_selective_momentum import StateSelectiveMomentumStrategy
from quantcore.strategies.macro.true_carry_strategy import TrueCarryStrategy
from quantcore.strategies.macro.vix_call_ladder import VIXCallLadderStrategy
from quantcore.strategies.macro.volatility_risk_premium import VolatilityRiskPremiumStrategy

__all__ = [
    # Active
    "CTATrendStrategy",
    "RiskParityStrategy",
    # Archived
    "CarryStrategy",
    "CarryTrendFilterStrategy",
    "CrossSectionalFactorNeutralStatArb",
    "CrossSectionalMomentumStrategy",
    "FactorResidualStrategy",
    "FactorResidualQPStrategy",
    "FactorResidualStrictStrategy",
    "FactorResidualV3Strategy",
    "FREDRegimeStrategy",
    "RegimeAdaptiveMomentumStrategy",
    "StateSelectiveMomentumStrategy",
    "TrueCarryStrategy",
    "VIXCallLadderStrategy",
    "VolatilityRiskPremiumStrategy",
]
