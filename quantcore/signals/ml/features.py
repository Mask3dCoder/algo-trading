"""ML feature engineering: technical, inter-market, calendar, cross-sectional features."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quantcore.signals.technical.trend import sma, ema, macd, adx
from quantcore.signals.technical.mean_reversion import rsi, bollinger_bands, cci
from quantcore.signals.technical.volume import obv, ad_line, mfi
from quantcore.signals.technical.volatility import atr, historical_vol
from quantcore.signals.technical.momentum import roc, cmo


# ---------------------------------------------------------------------------
# Feature Groups
# ---------------------------------------------------------------------------


@dataclass
class TechnicalFeatures:
    """Generate technical indicator features at multiple windows."""

    close: NDArray
    high: NDArray | None = None
    low: NDArray | None = None
    volume: NDArray | None = None
    windows: list[int] = field(default_factory=lambda: [5, 10, 21, 63])

    def compute(self) -> dict[str, NDArray]:
        """Compute all technical features. Returns dict of feature_name → array."""
        features: dict[str, NDArray] = {}
        n = len(self.close)

        # Trend features
        for w in self.windows:
            features[f"sma_{w}"] = sma(self.close, w)
            features[f"ema_{w}"] = ema(self.close, w)
            features[f"price_sma_{w}_ratio"] = self.close / np.where(
                features[f"sma_{w}"] > 0, features[f"sma_{w}"], np.nan
            )
            features[f"roc_{w}"] = roc(self.close, w)
            features[f"cmo_{w}"] = cmo(self.close, min(w, 63))

        # Mean-reversion features
        for w in [14, 21]:
            features[f"rsi_{w}"] = rsi(self.close, w)
            bb_upper, bb_mid, bb_lower, pct_b, bw = bollinger_bands(self.close, w)
            features[f"bb_pct_b_{w}"] = pct_b
            features[f"bb_bandwidth_{w}"] = bw

        if self.high is not None and self.low is not None:
            for w in [14, 21]:
                features[f"cci_{w}"] = cci(self.high, self.low, self.close, w)
                adx_val, plus_di, minus_di = adx(self.high, self.low, self.close, w)
                features[f"adx_{w}"] = adx_val
                features[f"plus_di_{w}"] = plus_di
                features[f"minus_di_{w}"] = minus_di
                features[f"atr_{w}"] = atr(self.high, self.low, self.close, w)

                if self.volume is not None:
                    features[f"mfi_{w}"] = mfi(self.high, self.low, self.close, self.volume, w)

        # MACD features
        macd_line, signal_line, histogram = macd(self.close)
        features["macd_line"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = histogram

        # Volatility features
        hv = historical_vol(self.close, windows=[5, 10, 21, 63])
        for w, vol_arr in hv.items():
            features[f"hist_vol_{w}"] = vol_arr

        # Volume features
        if self.volume is not None:
            for w in self.windows:
                features[f"vol_adv_{w}"] = sma(self.volume, w)
                features[f"vol_ratio_{w}"] = self.volume / np.where(
                    features[f"vol_adv_{w}"] > 0, features[f"vol_adv_{w}"], np.nan
                )
            features["obv"] = obv(self.close, self.volume)
            if self.high is not None and self.low is not None:
                features["ad_line"] = ad_line(self.high, self.low, self.close, self.volume)

        return features


@dataclass
class InterMarketFeatures:
    """Rolling correlations with benchmark, sector ETFs, VIX."""

    returns: NDArray
    benchmark_returns: NDArray | None = None
    sector_returns: dict[str, NDArray] | None = None
    windows: list[int] = field(default_factory=lambda: [21, 63])

    def compute(self) -> dict[str, NDArray]:
        features: dict[str, NDArray] = {}
        n = len(self.returns)

        if self.benchmark_returns is not None:
            for w in self.windows:
                corr = self._rolling_corr(self.returns, self.benchmark_returns, w)
                features[f"bench_corr_{w}"] = corr
                beta = self._rolling_beta(self.returns, self.benchmark_returns, w)
                features[f"bench_beta_{w}"] = beta

        if self.sector_returns is not None:
            for name, sec_ret in self.sector_returns.items():
                for w in self.windows:
                    corr = self._rolling_corr(self.returns, sec_ret, w)
                    features[f"corr_{name}_{w}"] = corr

        return features

    @staticmethod
    def _rolling_corr(x: NDArray, y: NDArray, window: int) -> NDArray:
        n = len(x)
        result = np.full(n, np.nan, dtype=float)
        for i in range(window - 1, n):
            xw = x[i - window + 1:i + 1]
            yw = y[i - window + 1:i + 1]
            valid = ~(np.isnan(xw) | np.isnan(yw))
            if np.sum(valid) >= window // 2:
                corr = np.corrcoef(xw[valid], yw[valid])[0, 1]
                result[i] = corr if not np.isnan(corr) else 0.0
        return result

    @staticmethod
    def _rolling_beta(x: NDArray, y: NDArray, window: int) -> NDArray:
        n = len(x)
        result = np.full(n, np.nan, dtype=float)
        for i in range(window - 1, n):
            xw = x[i - window + 1:i + 1]
            yw = y[i - window + 1:i + 1]
            valid = ~(np.isnan(xw) | np.isnan(yw))
            if np.sum(valid) >= window // 2:
                cov = np.cov(xw[valid], yw[valid])
                var_y = cov[1, 1]
                result[i] = cov[0, 1] / var_y if var_y > 0 else 0.0
        return result


@dataclass
class CalendarFeatures:
    """Calendar-based features: day of week, month, days to events."""

    dates: NDArray  # Array of datetime-like objects or ordinal integers
    n: int = 0

    def compute(self) -> dict[str, NDArray]:
        features: dict[str, NDArray] = {}
        n = len(self.dates)
        day_of_week = np.zeros(n, dtype=float)
        month = np.zeros(n, dtype=float)

        for i in range(n):
            try:
                d = self.dates[i]
                if hasattr(d, "weekday"):
                    day_of_week[i] = d.weekday() / 6.0  # Normalize to [0, 1]
                    month[i] = (d.month - 1) / 11.0  # Normalize to [0, 1]
            except Exception:
                pass

        features["day_of_week"] = day_of_week
        features["month"] = month

        return features


# ---------------------------------------------------------------------------
# Cross-Sectional Normalization
# ---------------------------------------------------------------------------


def cross_sectional_normalize(
    feature_matrix: NDArray,
) -> NDArray:
    """Cross-sectionally z-score a feature matrix.

    Args:
        feature_matrix: (n_periods, n_assets) feature values.

    Returns:
        Z-scored matrix. Each row (cross-section) has mean≈0, std≈1.
    """
    n_periods, n_assets = feature_matrix.shape
    result = np.full_like(feature_matrix, np.nan)

    for t in range(n_periods):
        row = feature_matrix[t, :]
        valid = ~np.isnan(row)
        if np.sum(valid) < 5:
            continue
        m = np.mean(row[valid])
        s = np.std(row[valid], ddof=1)
        if s > 0:
            result[t, valid] = (row[valid] - m) / s

    return np.clip(result, -5.0, 5.0)


# ---------------------------------------------------------------------------
# Feature Matrix Builder
# ---------------------------------------------------------------------------


def build_feature_matrix(
    close: NDArray,
    high: NDArray | None = None,
    low: NDArray | None = None,
    volume: NDArray | None = None,
    benchmark_returns: NDArray | None = None,
    dates: NDArray | None = None,
) -> dict[str, NDArray]:
    """Build comprehensive feature dictionary from OHLCV data.

    Returns dict of feature_name → 1D array of length n.
    """
    features: dict[str, NDArray] = {}

    # Technical features
    tech = TechnicalFeatures(close=close, high=high, low=low, volume=volume)
    features.update(tech.compute())

    # Inter-market features
    if benchmark_returns is not None:
        returns = np.full(len(close), np.nan)
        returns[1:] = close[1:] / close[:-1] - 1.0
        im = InterMarketFeatures(returns=returns, benchmark_returns=benchmark_returns)
        features.update(im.compute())

    # Calendar features
    if dates is not None:
        cal = CalendarFeatures(dates=dates, n=len(close))
        features.update(cal.compute())

    return features
