"""Statistical arbitrage signals: pairs trading, basket arbitrage, factor signals."""

from quantcore.signals.statistical.basket import (
    adr_arb_signal,
    basket_spread_signal,
    etf_arb_signal,
    fair_value,
    index_arb_signal,
)
from quantcore.signals.statistical.factors import (
    accruals_signal,
    analyst_revision_signal,
    composite_factor,
    earnings_growth_signal,
    ev_ebitda_signal,
    factor_ic,
    fcf_yield_signal,
    gross_margin_signal,
    pb_signal,
    pe_signal,
    quintile_spread,
    revenue_growth_signal,
    roe_signal,
    roic_signal,
    rolling_factor_ic,
    short_interest_signal,
)
from quantcore.signals.statistical.kalman_hedge import (
    KalmanHedgeRatio,
    adaptive_kalman_hedge_ratio,
    kalman_hedge_ratio_series,
)
from quantcore.signals.statistical.ml_factors import (
    AdaptiveFactorCombiner,
    EnsembleFactorCombiner,
    GradientBoostingFactorCombiner,
    NeuralNetworkFactorCombiner,
    RandomForestFactorCombiner,
    create_ml_factor_combiner,
    evaluate_factor_combiner,
)
from quantcore.signals.statistical.pairs import (
    CointPair,
    cost_adjusted_threshold,
    halflife_size,
    kalman_hedge_ratio,
    pairs_signal,
    scan_cointegrated_pairs,
    spread_zscore,
)

__all__ = [
    # Pairs
    "scan_cointegrated_pairs", "kalman_hedge_ratio", "spread_zscore",
    "pairs_signal", "halflife_size", "cost_adjusted_threshold", "CointPair",
    # Kalman
    "KalmanHedgeRatio", "kalman_hedge_ratio_series", "adaptive_kalman_hedge_ratio",
    # Basket
    "fair_value", "etf_arb_signal", "index_arb_signal",
    "adr_arb_signal", "basket_spread_signal",
    # Factors
    "pe_signal", "pb_signal", "ev_ebitda_signal", "fcf_yield_signal",
    "roe_signal", "roic_signal", "gross_margin_signal", "accruals_signal",
    "revenue_growth_signal", "earnings_growth_signal",
    "analyst_revision_signal", "short_interest_signal",
    "composite_factor", "factor_ic", "rolling_factor_ic", "quintile_spread",
    # ML Factors
    "GradientBoostingFactorCombiner", "RandomForestFactorCombiner",
    "NeuralNetworkFactorCombiner", "EnsembleFactorCombiner",
    "AdaptiveFactorCombiner", "create_ml_factor_combiner", "evaluate_factor_combiner",
]
