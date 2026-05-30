"""Advanced risk models for statistical arbitrage portfolios.

Implements:
  - PCA/ICA factor decomposition
  - Extreme Value Theory (EVT) for tail risk
  - Correlation regime detection
  - DCC-GARCH for dynamic correlations
  - Risk decomposition
"""

from quantcore.risk.advanced_risk import (
    CorrelationRegime,
    CorrelationRegimeDetector,
    DCCGARCHModel,
    ICARiskModel,
    PCAResult,
    PCARiskModel,
    RiskDecomposer,
    RiskDecomposition,
    TailRiskMetrics,
    TailRiskModel,
    compute_portfolio_es,
    compute_portfolio_var,
)

__all__ = [
    "PCARiskModel",
    "ICARiskModel",
    "TailRiskModel",
    "CorrelationRegimeDetector",
    "DCCGARCHModel",
    "RiskDecomposer",
    "PCAResult",
    "TailRiskMetrics",
    "CorrelationRegime",
    "RiskDecomposition",
    "compute_portfolio_var",
    "compute_portfolio_es",
]
