"""Betting-against-beta (low volatility anomaly): long low-vol, short high-vol.

Frazzini & Pedersen (2014): constrained investors lever up risky assets,
making them overpriced and low-vol assets underpriced. Betting against
beta earns positive risk-adjusted returns.

Implementation: rank assets by realised volatility, long the lowest-vol
quintile, short the highest-vol quintile, with minimum-variance weighting.
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


@register("equity_low_volatility")
class LowVolatilityStrategy(Strategy):
    """Betting against beta: long low-vol, short high-vol, min-variance weighted.

    Config params:
        vol_lookback: window for volatility estimation (default 63)
        min_var_shrinkage: shrinkage towards equal weight (default 0.5)
        n_long: number of low-vol longs (default 5)
        n_short: number of high-vol shorts (default 5)
        rebalance_freq: rebalance frequency in trading days (default 21)
        vol_target: target portfolio vol (annualised, default 0.10)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.vol_lookback = int(config.params.get("vol_lookback", 63))
        self.min_var_shrinkage = float(config.params.get("min_var_shrinkage", 0.5))
        self.n_long = int(config.params.get("n_long", 5))
        self.n_short = int(config.params.get("n_short", 5))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.vol_target = float(config.params.get("vol_target", 0.10))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None

    def _min_variance_weights(self, cov: NDArray, shrinkage: float) -> NDArray:
        """Minimum variance portfolio with shrinkage towards equal weight.

        w = (1 - shrinkage) * w_mv + shrinkage * w_ew
        where w_mv = (Sigma^-1 * 1) / (1' Sigma^-1 1)
        """
        n = cov.shape[0]
        if n == 0:
            return np.zeros(n)

        try:
            inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-6)
        except np.linalg.LinAlgError:
            return np.ones(n) / n

        ones = np.ones(n)
        w_mv = inv_cov @ ones / (ones @ inv_cov @ ones)

        # Long-only constraint for min-var leg
        w_mv = np.maximum(w_mv, 0.0)
        total = np.sum(w_mv)
        if total > 1e-10:
            w_mv /= total
        else:
            w_mv = np.ones(n) / n

        w_ew = np.ones(n) / n
        w = (1.0 - shrinkage) * w_mv + shrinkage * w_ew
        total_w = np.sum(w)
        if total_w > 1e-10:
            w /= total_w

        return w

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

        vol_start = max(0, n_periods - self.vol_lookback)
        if n_periods - vol_start < 20:
            return signals

        # Compute volatilities
        daily_rets = np.zeros((n_periods - vol_start, n_assets))
        for i in range(vol_start, n_periods - 1):
            daily_rets[i - vol_start] = close[i + 1] / close[i] - 1.0
        vols = np.std(daily_rets, axis=0, ddof=1) * np.sqrt(252)
        vols = np.maximum(vols, 1e-10)

        # Rank by volatility (ascending: low vol = high rank)
        vol_ranks = np.argsort(np.argsort(vols))

        # Low-vol longs: lowest vol quintile
        n_long_actual = min(self.n_long, n_assets)
        n_short_actual = min(self.n_short, n_assets - n_long_actual)
        long_mask = vol_ranks < n_long_actual     # Low vol rank = low vol
        short_mask = vol_ranks >= (n_assets - n_short_actual)  # High vol rank = high vol

        # Compute sub-covariance matrices for min-var weighting
        long_indices = np.where(long_mask)[0]
        short_indices = np.where(short_mask)[0]

        weights = np.zeros(n_assets)

        if len(long_indices) > 0:
            long_cov = np.cov(daily_rets[:, long_indices], rowvar=False)
            if long_cov.ndim < 2:
                long_cov = np.array([[long_cov]])
            long_w = self._min_variance_weights(long_cov, self.min_var_shrinkage)
            weights[long_indices] = long_w / 2.0

        if len(short_indices) > 0:
            short_cov = np.cov(daily_rets[:, short_indices], rowvar=False)
            if short_cov.ndim < 2:
                short_cov = np.array([[short_cov]])
            short_w = self._min_variance_weights(short_cov, self.min_var_shrinkage)
            weights[short_indices] = -short_w / 2.0

        # Vol-target scaling
        port_var = weights @ np.cov(daily_rets, rowvar=False) @ weights if n_assets > 1 else 0.0
        port_vol = np.sqrt(max(port_var, 0.0)) * np.sqrt(252)
        if port_vol > 1e-10:
            scale = min(self.vol_target / port_vol, 2.0)
            weights *= scale

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
                    metadata={"vol_rank": int(vol_ranks[i]), "ann_vol": float(vols[i])},
                ))

        return signals
