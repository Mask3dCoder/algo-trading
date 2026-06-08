"""Quality factor via Fama-French RMW and CMA regression.

Proxy "quality" by regressing each stock's excess returns against the
Fama-French RMW (Robust Minus Weak profitability) and CMA (Conservative
Minus Aggressive investment) factors.  Quality score = 0.5*beta_RMW +
0.5*beta_CMA.  Long top quintile, short bottom quintile, inv-vol
weighted.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("equity_quality_rmw_cma")
class QualityRMWCMAStrategy(Strategy):
    """Quality factor via RMW/CMA regression.

    Config params:
        regression_days: trailing days for factor regression (default 252)
        top_pct: fraction of universe for long leg (default 0.20)
        bottom_pct: fraction of universe for short leg (default 0.20)
        vol_lookback: days for volatility estimation (default 60)
        rebalance_freq: rebalance frequency in trading days (default 21)
        min_price: minimum stock price filter (default 5.0)
        min_volume: minimum average daily dollar volume filter (default 5e6)
        rmw_weight: weight of RMW beta in quality score (default 0.5)
        cma_weight: weight of CMA beta in quality score (default 0.5)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.regression_days = int(config.params.get("regression_days", 252))
        self.top_pct = float(config.params.get("top_pct", 0.20))
        self.bottom_pct = float(config.params.get("bottom_pct", 0.20))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.min_price = float(config.params.get("min_price", 5.0))
        self.min_volume = float(config.params.get("min_volume", 5e6))
        self.rmw_weight = float(config.params.get("rmw_weight", 0.5))
        self.cma_weight = float(config.params.get("cma_weight", 0.5))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        volume = data.get("volume")
        factor_df = data.get("factor_returns")
        if close is None or factor_df is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq and self._current_weights is not None:
            return _signals_from_weights(self._current_weights, symbols, self.config.max_position_notional)

        min_required = max(self.regression_days, self.vol_lookback) + 1
        if n_periods < min_required:
            return []

        dates = data.get("dates")
        if dates is None:
            return []

        try:
            import pandas as pd
            factor_slice = factor_df.loc[:dates[-1]]
            factor_aligned = factor_slice.reindex(dates[-self.regression_days:])
            if factor_aligned.isna().any().any():
                factor_aligned = factor_aligned.ffill().bfill()
            mkt_rf = factor_aligned["Mkt-RF"].values
            rmw = factor_aligned["RMW"].values
            cma = factor_aligned["CMA"].values
            rf = factor_aligned["RF"].values
        except (KeyError, AttributeError):
            return []

        # Stock excess returns
        raw_rets = np.diff(close[-self.regression_days - 1:], axis=0) / np.maximum(
            close[-self.regression_days - 1:-1], 1e-10
        )
        excess_rets = raw_rets - rf[:len(raw_rets), np.newaxis]

        # Regress each stock on [1, Mkt-RF, SMB, HML, RMW, CMA]
        # We use all 5 factors for a proper regression, but only extract RMW and CMA
        factor_df_aligned = factor_aligned
        smb = factor_df_aligned["SMB"].values if "SMB" in factor_df_aligned.columns else np.zeros_like(mkt_rf)
        hml = factor_df_aligned["HML"].values if "HML" in factor_df_aligned.columns else np.zeros_like(mkt_rf)

        X = np.column_stack([np.ones(len(mkt_rf)), mkt_rf, smb, hml, rmw, cma])
        try:
            betas = np.linalg.lstsq(X, excess_rets, rcond=None)[0]
        except np.linalg.LinAlgError:
            return []

        rmw_betas = betas[4]  # RMW loading
        cma_betas = betas[5]  # CMA loading

        # Quality score
        quality_score = self.rmw_weight * rmw_betas + self.cma_weight * cma_betas

        # Universe filters
        valid = np.ones(n_assets, dtype=bool)
        valid &= close[-1] >= self.min_price
        if volume is not None and volume.ndim == 2 and volume.shape[0] >= 21:
            avg_dollar_vol = np.mean(volume[-21:] * close[-21:], axis=0)
            valid &= avg_dollar_vol >= self.min_volume

        quality_score[~valid] = np.nan
        n_valid = int(np.sum(valid))
        if n_valid < 4:
            return []

        n_long = max(1, int(n_valid * self.top_pct))
        n_short = max(1, int(n_valid * self.bottom_pct))

        valid_indices = np.where(valid)[0]
        sorted_idx = valid_indices[np.argsort(quality_score[valid_indices])]

        short_leg = sorted_idx[:n_short]
        long_leg = sorted_idx[-n_long:]

        overlap = set(long_leg) & set(short_leg)
        if overlap:
            short_leg = np.array([i for i in short_leg if i not in overlap])

        if len(long_leg) == 0 or len(short_leg) == 0:
            return []

        # Inv-vol weighting
        vol_slice = np.diff(close[-self.vol_lookback:], axis=0) / np.maximum(
            close[-self.vol_lookback:-1], 1e-10
        )
        vols = np.std(vol_slice, axis=0, ddof=1) * np.sqrt(252)
        vols = np.maximum(vols, 1e-4)

        inv_vol_long = 1.0 / vols[long_leg]
        inv_vol_short = 1.0 / vols[short_leg]

        weights = np.zeros(n_assets)
        weights[long_leg] = inv_vol_long / np.sum(inv_vol_long) / 2.0
        weights[short_leg] = -inv_vol_short / np.sum(inv_vol_short) / 2.0

        self._current_weights = weights
        self._days_since_rebalance = 0

        return _signals_from_weights(weights, symbols, self.config.max_position_notional)


def _signals_from_weights(
    weights: NDArray, symbols: list[str], max_notional: float,
) -> list[TradeSignal]:
    signals: list[TradeSignal] = []
    for i in range(len(weights)):
        if abs(weights[i]) > 1e-8:
            signals.append(TradeSignal(
                symbol=symbols[i],
                direction=1 if weights[i] > 0 else -1,
                strength=min(abs(weights[i]) * 2, 1.0),
                size=abs(weights[i]) * float(max_notional),
            ))
    return signals
