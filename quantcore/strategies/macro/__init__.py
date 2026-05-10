"""Macro strategies: risk parity, CTA trend, carry, factor residual, etc."""

from quantcore.strategies.macro.carry import CarryStrategy
from quantcore.strategies.macro.cta_trend import CTATrendStrategy
from quantcore.strategies.macro.risk_parity import RiskParityStrategy
from quantcore.strategies.macro.carry_trend_filter import CarryTrendFilterStrategy
from quantcore.strategies.macro.factor_residual import FactorResidualStrategy
from quantcore.strategies.macro.factor_residual_strict import FactorResidualStrictStrategy
from quantcore.strategies.macro.regime_adaptive_momentum import RegimeAdaptiveMomentumStrategy
from quantcore.strategies.macro.state_selective_momentum import StateSelectiveMomentumStrategy
from quantcore.strategies.macro.true_carry_strategy import TrueCarryStrategy
from quantcore.strategies.macro.cross_sectional_momentum import CrossSectionalMomentumStrategy
from quantcore.strategies.macro.cross_sectional_momentum_v2 import CrossSectionalMomentumV2Strategy
from quantcore.strategies.macro.factor_residual_qp import FactorResidualQPStrategy

__all__ = [
    "CTATrendStrategy",
    "CarryStrategy",
    "RiskParityStrategy",
    "CarryTrendFilterStrategy",
    "FactorResidualStrategy",
    "FactorResidualStrictStrategy",
    "RegimeAdaptiveMomentumStrategy",
    "StateSelectiveMomentumStrategy",
    "TrueCarryStrategy",
    "CrossSectionalMomentumStrategy",
    "CrossSectionalMomentumV2Strategy",
    "FactorResidualQPStrategy",
]
