"""Technical signal indicators: trend, mean-reversion, volume, volatility, momentum."""

from quantcore.signals.technical.trend import (
    sma, ema, dema, tema, hull_ma, kaufman_ama,
    macd, adx, supertrend,
    linear_regression_signal, breakout_signal, zscore_normalize,
)
from quantcore.signals.technical.mean_reversion import (
    bollinger_bands, rsi, rsi_signal,
    stochastic, stochastic_signal, williams_r,
    cci, ma_zscore, ou_halflife_signal,
)
from quantcore.signals.technical.volume import (
    vwap_deviation, obv, volume_price_trend, ad_line,
    chaikin_money_flow, mfi, mfi_signal,
    unusual_volume, obv_divergence, ease_of_movement,
)
from quantcore.signals.technical.volatility import (
    atr, atr_normalized, historical_vol, historical_vol_signal,
    vol_regime, vol_regime_signal,
    vix_term_structure_signal, rv_iv_spread,
    vol_of_vol, parkinson_vol, garman_klass_vol,
)
from quantcore.signals.technical.momentum import (
    cross_sectional_momentum, time_series_momentum,
    dual_momentum, idiosyncratic_momentum,
    high_52w_proximity, vol_scaled_momentum,
    roc, roc_signal, cmo, cmo_signal,
)

__all__ = [
    # Trend
    "sma", "ema", "dema", "tema", "hull_ma", "kaufman_ama",
    "macd", "adx", "supertrend",
    "linear_regression_signal", "breakout_signal", "zscore_normalize",
    # Mean reversion
    "bollinger_bands", "rsi", "rsi_signal",
    "stochastic", "stochastic_signal", "williams_r",
    "cci", "ma_zscore", "ou_halflife_signal",
    # Volume
    "vwap_deviation", "obv", "volume_price_trend", "ad_line",
    "chaikin_money_flow", "mfi", "mfi_signal",
    "unusual_volume", "obv_divergence", "ease_of_movement",
    # Volatility
    "atr", "atr_normalized", "historical_vol", "historical_vol_signal",
    "vol_regime", "vol_regime_signal",
    "vix_term_structure_signal", "rv_iv_spread",
    "vol_of_vol", "parkinson_vol", "garman_klass_vol",
    # Momentum
    "cross_sectional_momentum", "time_series_momentum",
    "dual_momentum", "idiosyncratic_momentum",
    "high_52w_proximity", "vol_scaled_momentum",
    "roc", "roc_signal", "cmo", "cmo_signal",
]
