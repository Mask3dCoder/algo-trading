"""Macro strategies: risk parity, CTA trend, carry, factor residual, etc."""

from quantcore.strategies.macro.carry import CarryStrategy
from quantcore.strategies.macro.carry_trend_filter import CarryTrendFilterStrategy

# DISABLED — CrossSectionalMomentumV2Strategy registration removed per
# prospective_trials.csv entry #1 (REJECTED). Class preserved for reference.
# from quantcore.strategies.macro.cross_sectional_momentum_v2 import CrossSectionalMomentumV2Strategy
from quantcore.strategies.macro.cross_sectional_factor_neutral import (
    CrossSectionalFactorNeutralStatArb,
)
from quantcore.strategies.macro.cross_sectional_momentum import CrossSectionalMomentumStrategy
from quantcore.strategies.macro.cta_trend import CTATrendStrategy
from quantcore.strategies.macro.factor_residual import FactorResidualStrategy
from quantcore.strategies.macro.factor_residual_qp import FactorResidualQPStrategy
from quantcore.strategies.macro.factor_residual_strict import FactorResidualStrictStrategy
from quantcore.strategies.macro.factor_residual_v3 import FactorResidualStrategyV3
from quantcore.strategies.macro.fred_regime_strategy import FREDRegimeStrategy
from quantcore.strategies.macro.regime_adaptive_momentum import RegimeAdaptiveMomentumStrategy
from quantcore.strategies.macro.risk_parity import RiskParityStrategy
from quantcore.strategies.macro.simple_test_strategy import SimpleBuyHoldStrategy
from quantcore.strategies.macro.state_selective_momentum import StateSelectiveMomentumStrategy
from quantcore.strategies.macro.true_carry_strategy import TrueCarryStrategy
from quantcore.strategies.macro.vix_call_ladder import VIXCallLadderStrategy
from quantcore.strategies.macro.volatility_risk_premium import VolatilityRiskPremiumStrategy

__all__ = [
    "CTATrendStrategy",
    "CarryStrategy",
    "RiskParityStrategy",
    "CarryTrendFilterStrategy",
    "FactorResidualStrategy",
    "FactorResidualStrictStrategy",
    "FactorResidualStrategyV3",
    "FREDRegimeStrategy",
    "RegimeAdaptiveMomentumStrategy",
    "SimpleBuyHoldStrategy",
    "StateSelectiveMomentumStrategy",
    "TrueCarryStrategy",
    "VIXCallLadderStrategy",
    "CrossSectionalMomentumStrategy",
    "CrossSectionalFactorNeutralStatArb",
    # "CrossSectionalMomentumV2Strategy",  # disabled per prospective trial #1
    "FactorResidualQPStrategy",
    "VolatilityRiskPremiumStrategy",
]
