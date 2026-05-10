"""Multi-factor composite: value + momentum + quality + low-vol factor diversification.

The core strategy institutional desks actually run: combine orthogonal factor
signals into a single composite alpha, then weight long-short by inverse
volatility. Diversification across factors reduces drawdown and improves
Sharpe vs any single factor.

Factors combined:
 - Value: book-to-market / price-derived cheapness
 - Momentum: 12-1 cross-sectional momentum (skip 1 month)
 - Quality: profitability + stability + low accruals
 - Low-vol: betting-against-beta anomaly

Each factor z-scored cross-sectionally, then equal-weighted by default
(IC-weighting available via params).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    TradeSignal,
)
from quantcore.strategies.registry import register


@register("equity_multi_factor")
class MultiFactorStrategy(Strategy):
    """Multi-factor composite: value + momentum + quality + low-vol.

    Config params:
        lookback: window for factor computation (default 252)
        n_long: number of long positions (default 5)
        n_short: number of short positions (default 5)
        vol_lookback: window for volatility estimation (default 63)
        rebalance_freq: rebalance frequency in trading days (default 21)
        value_weight: weight of value signal (default 0.25)
        momentum_weight: weight of momentum signal (default 0.25)
        quality_weight: weight of quality signal (default 0.25)
        lowvol_weight: weight of low-vol signal (default 0.25)
        ic_weighting: use IC-weighting instead of equal (default False)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback = int(config.params.get("lookback", 252))
        self.n_long = int(config.params.get("n_long", 5))
        self.n_short = int(config.params.get("n_short", 5))
        self.vol_lookback = int(config.params.get("vol_lookback", 63))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.value_weight = float(config.params.get("value_weight", 0.25))
        self.momentum_weight = float(config.params.get("momentum_weight", 0.25))
        self.quality_weight = float(config.params.get("quality_weight", 0.25))
        self.lowvol_weight = float(config.params.get("lowvol_weight", 0.25))
        self.ic_weighting = bool(config.params.get("ic_weighting", False))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None
        self._factor_ics: dict[str, float] = {}

    def _value_signal(self, close: NDArray) -> NDArray:
        """Value signal: negative distance from rolling mean (cheaper = higher)."""
        n_periods, n_assets = close.shape
        score = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 20:
                continue
            score[a] = -(valid[-1] - np.mean(valid)) / np.mean(valid) if np.mean(valid) > 0 else 0.0

        return _zscore(score)

    def _momentum_signal(self, close: NDArray) -> NDArray:
        """12-1 momentum: skip 1 month, cumulative return."""
        n_periods, n_assets = close.shape
        skip_days = 21
        score = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < skip_days + 20:
                continue
            cum_ret = valid[-(skip_days + 1)] / valid[0] - 1.0 if len(valid) > skip_days else 0.0
            score[a] = cum_ret

        return _zscore(score)

    def _quality_signal(self, close: NDArray) -> NDArray:
        """Quality signal: hit rate (fraction of positive return days)."""
        n_periods, n_assets = close.shape
        score = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 20:
                continue
            rets = np.diff(valid) / valid[:-1]
            score[a] = np.mean(rets > 0)

        return _zscore(score)

    def _lowvol_signal(self, close: NDArray) -> NDArray:
        """Low-vol signal: negative realised vol (low vol = high score)."""
        n_periods, n_assets = close.shape
        score = np.zeros(n_assets)
        lb = min(self.vol_lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 20:
                continue
            rets = np.diff(valid) / valid[:-1]
            score[a] = -np.std(rets, ddof=1)

        return _zscore(score)

    def _estimate_ics(self, close: NDArray) -> dict[str, float]:
        """Estimate rolling IC for each factor (rank correlation with forward returns).

        Simplified: uses last quarter's data.
        """
        n_periods, n_assets = close.shape
        fwd_days = 21
        if n_periods < self.lookback + fwd_days:
            return {"value": 0.1, "momentum": 0.1, "quality": 0.1, "lowvol": 0.1}

        # Forward returns
        end_idx = n_periods - fwd_days
        start_idx = max(self.lookback, end_idx - 63)
        if end_idx <= start_idx:
            return {"value": 0.1, "momentum": 0.1, "quality": 0.1, "lowvol": 0.1}

        fwd_rets = np.zeros(n_assets)
        for a in range(n_assets):
            if close[end_idx, a] > 0 and close[end_idx + fwd_days - 1, a] > 0:
                fwd_rets[a] = close[end_idx + fwd_days - 1, a] / close[end_idx, a] - 1.0

        fwd_rank = np.argsort(np.argsort(fwd_rets)).astype(float)

        vc = self._value_signal(close)
        mc = self._momentum_signal(close)
        qc = self._quality_signal(close)
        lc = self._lowvol_signal(close)

        def _spearman(x: NDArray, y: NDArray) -> float:
            rx = np.argsort(np.argsort(x)).astype(float)
            n = len(x)
            if n < 3:
                return 0.0
            d = rx - y
            return 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))

        return {
            "value": _spearman(vc, fwd_rank),
            "momentum": _spearman(mc, fwd_rank),
            "quality": _spearman(qc, fwd_rank),
            "lowvol": _spearman(lc, fwd_rank),
        }

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        signals: list[TradeSignal] = []

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq and self._current_weights is not None:
            return signals

        if n_periods < self.lookback:
            return signals

        value_s = self._value_signal(close)
        momentum_s = self._momentum_signal(close)
        quality_s = self._quality_signal(close)
        lowvol_s = self._lowvol_signal(close)

        if self.ic_weighting:
            self._factor_ics = self._estimate_ics(close)
            ic = self._factor_ics
            ic_sum = sum(abs(ic[k]) for k in ic)
            if ic_sum > 1e-10:
                vw = abs(ic["value"]) / ic_sum
                mw = abs(ic["momentum"]) / ic_sum
                qw = abs(ic["quality"]) / ic_sum
                lw = abs(ic["lowvol"]) / ic_sum
                # Preserve sign: positive IC means factor predicts forward returns
                composite = (
                    np.sign(ic["value"]) * vw * value_s
                    + np.sign(ic["momentum"]) * mw * momentum_s
                    + np.sign(ic["quality"]) * qw * quality_s
                    + np.sign(ic["lowvol"]) * lw * lowvol_s
                )
            else:
                composite = (value_s + momentum_s + quality_s + lowvol_s) / 4.0
        else:
            composite = (
                self.value_weight * value_s
                + self.momentum_weight * momentum_s
                + self.quality_weight * quality_s
                + self.lowvol_weight * lowvol_s
            )

        # Vol for inverse-vol weighting
        vol_start = max(0, n_periods - self.vol_lookback)
        daily_rets = np.zeros((n_periods - vol_start, n_assets))
        for i in range(vol_start + 1, n_periods):
            daily_rets[i - vol_start] = close[i] / close[i - 1] - 1.0
        vols = np.std(daily_rets, axis=0, ddof=1)
        vols[vols < 1e-10] = 1e-10

        ranks = np.argsort(np.argsort(composite))
        n_long_actual = min(self.n_long, n_assets)
        n_short_actual = min(self.n_short, n_assets - n_long_actual)
        long_mask = ranks >= (n_assets - n_long_actual)
        short_mask = ranks < n_short_actual

        inv_vol = 1.0 / vols
        inv_vol[~long_mask & ~short_mask] = 0.0
        total_inv_vol = np.sum(inv_vol[long_mask]) + np.sum(inv_vol[short_mask])
        if total_inv_vol < 1e-10:
            return signals

        weights = np.zeros(n_assets)
        if np.any(long_mask):
            weights[long_mask] = inv_vol[long_mask] / np.sum(inv_vol[long_mask]) / 2.0
        if np.any(short_mask):
            weights[short_mask] = -inv_vol[short_mask] / np.sum(inv_vol[short_mask]) / 2.0

        self._current_weights = weights
        self._days_since_rebalance = 0

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])
        for i in range(n_assets):
            if abs(weights[i]) > 1e-8:
                direction = 1 if weights[i] > 0 else -1
                signals.append(TradeSignal(
                    symbol=symbols[i],
                    direction=direction,
                    strength=min(abs(weights[i]) * 2, 1.0),
                    size=abs(weights[i]) * float(self.config.max_position_notional),
                    metadata={
                        "composite_rank": int(ranks[i]),
                        "value_s": float(value_s[i]),
                        "momentum_s": float(momentum_s[i]),
                        "quality_s": float(quality_s[i]),
                        "lowvol_s": float(lowvol_s[i]),
                    },
                ))

        return signals


def _zscore(x: NDArray) -> NDArray:
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s
