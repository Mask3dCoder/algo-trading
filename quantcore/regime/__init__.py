"""Market regime detection: HMM, features, changepoint, volatility regimes."""

from quantcore.regime.changepoint import (
    ChangepointResult,
    bocpd,
    cusum,
    pelt_changepoints,
    sprt_test,
)
from quantcore.regime.features import (
    advance_decline_ratio,
    adx,
    bid_ask_spread,
    compute_all_features,
    credit_spread,
    hurst_exponent,
    realized_volatility,
    rolling_correlation,
    turnover_ratio,
    yield_curve_curvature,
    yield_curve_slope,
)
from quantcore.regime.hmm import (
    RegimeHMM,
    RegimeResult,
    select_hmm_states,
)
from quantcore.regime.volatility_regime import (
    VolRegime,
    VolRegimeResult,
    garch_regime,
    vix_regime,
    vol_percentile_regime,
)

__all__ = [
    # hmm
    "RegimeResult", "RegimeHMM", "select_hmm_states",
    # features
    "realized_volatility", "rolling_correlation", "adx", "hurst_exponent",
    "bid_ask_spread", "turnover_ratio", "advance_decline_ratio",
    "credit_spread", "yield_curve_slope", "yield_curve_curvature",
    "compute_all_features",
    # changepoint
    "ChangepointResult", "pelt_changepoints", "bocpd", "cusum", "sprt_test",
    # volatility_regime
    "VolRegime", "VolRegimeResult", "vix_regime", "garch_regime",
    "vol_percentile_regime",
]
