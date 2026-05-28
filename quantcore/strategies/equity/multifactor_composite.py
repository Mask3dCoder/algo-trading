"""Multi-factor composite (Smart Beta).

Combine four signals: time-series momentum, cross-sectional momentum
rank, value (HML beta), quality (RMW+CMA beta).  Each signal converted
to percentile rank [0,1], averaged into a composite, then long top
decile / short bottom decile, equal-weighted, dollar-neutral.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("equity_multifactor_composite")
class MultiFactorCompositeStrategy(Strategy):
    """Multi-factor composite: TSMOM + CS Mom + Value + Quality.

    Config params:
        lookback_days: return lookback for momentum signals (default 252)
        vol_lookback: window for vol estimation (default 60)
        regression_days: trailing days for factor regression (default 252)
        top_pct: fraction of universe for long leg (default 0.10)
        bottom_pct: fraction of universe for short leg (default 0.10)
        rebalance_freq: rebalance frequency in trading days (default 21)
        min_price: minimum stock price filter (default 5.0)
        min_volume: minimum average daily dollar volume filter (default 5e6)
        mom_weight: weight of TSMOM signal (default 0.25)
        cs_weight: weight of cross-sectional momentum (default 0.25)
        value_weight: weight of value signal (default 0.25)
        quality_weight: weight of quality signal (default 0.25)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback_days = int(config.params.get("lookback_days", 252))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))
        self.regression_days = int(config.params.get("regression_days", 252))
        self.top_pct = float(config.params.get("top_pct", 0.10))
        self.bottom_pct = float(config.params.get("bottom_pct", 0.10))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.min_price = float(config.params.get("min_price", 5.0))
        self.min_volume = float(config.params.get("min_volume", 5e6))
        self.mom_weight = float(config.params.get("mom_weight", 0.25))
        self.cs_weight = float(config.params.get("cs_weight", 0.25))
        self.value_weight = float(config.params.get("value_weight", 0.25))
        self.quality_weight = float(config.params.get("quality_weight", 0.25))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        volume = data.get("volume")
        factor_df = data.get("factor_returns")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq and self._current_weights is not None:
            return _signals_from_weights(self._current_weights, symbols, self.config.max_position_notional)

        min_required = max(self.lookback_days, self.vol_lookback, self.regression_days) + 1
        if n_periods < min_required:
            return []

        # Universe filters
        valid = np.ones(n_assets, dtype=bool)
        valid &= close[-1] >= self.min_price
        if volume is not None and volume.ndim == 2 and volume.shape[0] >= 21:
            avg_dollar_vol = np.mean(volume[-21:] * close[-21:], axis=0)
            valid &= avg_dollar_vol >= self.min_volume

        n_valid = int(np.sum(valid))
        if n_valid < 10:
            return []

        # --- Signal 1: TSMOM (12-month trailing return) ---
        ret_12m = close[-1] / np.maximum(close[-self.lookback_days - 1], 1e-10) - 1.0
        tsmom_signal = ret_12m.copy()

        # --- Signal 2: Cross-sectional momentum rank ---
        cs_signal = ret_12m.copy()

        # --- Signal 3: Value (HML beta) ---
        value_signal = np.zeros(n_assets)
        quality_signal = np.zeros(n_assets)

        dates = data.get("dates")
        if dates is not None and factor_df is not None:
            try:
                factor_slice = factor_df.loc[:dates[-1]]
                factor_aligned = factor_slice.reindex(dates[-self.regression_days:])
                if factor_aligned.isna().any().any():
                    factor_aligned = factor_aligned.ffill().bfill()

                mkt_rf = factor_aligned["Mkt-RF"].values
                hml = factor_aligned["HML"].values
                rf = factor_aligned["RF"].values

                raw_rets = np.diff(close[-self.regression_days - 1:], axis=0) / np.maximum(
                    close[-self.regression_days - 1:-1], 1e-10
                )
                excess_rets = raw_rets - rf[:len(raw_rets), np.newaxis]

                # Value regression: [1, Mkt-RF, HML]
                X_val = np.column_stack([np.ones(len(mkt_rf)), mkt_rf, hml])
                betas_val = np.linalg.lstsq(X_val, excess_rets, rcond=None)[0]
                value_signal = betas_val[2]

                # Quality regression: [1, Mkt-RF, SMB, HML, RMW, CMA]
                smb = factor_aligned["SMB"].values if "SMB" in factor_aligned.columns else np.zeros_like(mkt_rf)
                rmw = factor_aligned["RMW"].values if "RMW" in factor_aligned.columns else np.zeros_like(mkt_rf)
                cma = factor_aligned["CMA"].values if "CMA" in factor_aligned.columns else np.zeros_like(mkt_rf)

                X_qual = np.column_stack([np.ones(len(mkt_rf)), mkt_rf, smb, hml, rmw, cma])
                betas_qual = np.linalg.lstsq(X_qual, excess_rets, rcond=None)[0]
                quality_signal = 0.5 * betas_qual[4] + 0.5 * betas_qual[5]

            except (KeyError, AttributeError, np.linalg.LinAlgError):
                pass

        # Convert each signal to percentile rank [0, 1] among valid assets
        def _pct_rank(x: NDArray) -> NDArray:
            ranks = np.zeros(len(x))
            valid_idx = np.where(valid)[0]
            if len(valid_idx) == 0:
                return ranks
            x_valid = x[valid_idx]
            order = np.argsort(np.argsort(x_valid)).astype(float)
            ranks[valid_idx] = order / max(len(valid_idx) - 1, 1)
            return ranks

        pct_tsmom = _pct_rank(tsmom_signal)
        pct_cs = _pct_rank(cs_signal)
        pct_value = _pct_rank(value_signal)
        pct_quality = _pct_rank(quality_signal)

        composite = (
            self.mom_weight * pct_tsmom
            + self.cs_weight * pct_cs
            + self.value_weight * pct_value
            + self.quality_weight * pct_quality
        )

        composite[~valid] = np.nan

        n_long = max(1, int(n_valid * self.top_pct))
        n_short = max(1, int(n_valid * self.bottom_pct))

        valid_indices = np.where(valid)[0]
        sorted_idx = valid_indices[np.argsort(composite[valid_indices])]

        short_leg = sorted_idx[:n_short]
        long_leg = sorted_idx[-n_long:]

        overlap = set(long_leg) & set(short_leg)
        if overlap:
            short_leg = np.array([i for i in short_leg if i not in overlap])

        if len(long_leg) == 0 or len(short_leg) == 0:
            return []

        # Equal-weighted, dollar-neutral
        weights = np.zeros(n_assets)
        weights[long_leg] = 1.0 / len(long_leg) / 2.0
        weights[short_leg] = -1.0 / len(short_leg) / 2.0

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
