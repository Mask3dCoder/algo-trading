"""Statistical arbitrage signals: pairs trading, basket arbitrage, factor signals."""

from quantcore.signals.statistical.pairs import (
    scan_cointegrated_pairs,
    kalman_hedge_ratio,
    spread_zscore,
    pairs_signal,
    halflife_size,
    cost_adjusted_threshold,
    CointPair,
)
from quantcore.signals.statistical.basket import (
    fair_value,
    etf_arb_signal,
    index_arb_signal,
    adr_arb_signal,
    basket_spread_signal,
)
from quantcore.signals.statistical.factors import (
    pe_signal, pb_signal, ev_ebitda_signal, fcf_yield_signal,
    roe_signal, roic_signal, gross_margin_signal, accruals_signal,
    revenue_growth_signal, earnings_growth_signal,
    analyst_revision_signal, short_interest_signal,
    composite_factor, factor_ic, rolling_factor_ic, quintile_spread,
)

__all__ = [
    # Pairs
    "scan_cointegrated_pairs", "kalman_hedge_ratio", "spread_zscore",
    "pairs_signal", "halflife_size", "cost_adjusted_threshold", "CointPair",
    # Basket
    "fair_value", "etf_arb_signal", "index_arb_signal",
    "adr_arb_signal", "basket_spread_signal",
    # Factors
    "pe_signal", "pb_signal", "ev_ebitda_signal", "fcf_yield_signal",
    "roe_signal", "roic_signal", "gross_margin_signal", "accruals_signal",
    "revenue_growth_signal", "earnings_growth_signal",
    "analyst_revision_signal", "short_interest_signal",
    "composite_factor", "factor_ic", "rolling_factor_ic", "quintile_spread",
]
