"""Cross-sectional quality factor: long profitable/stable, short junk.

Novy-Marx (2013) gross profitability premium: firms with high gross
profitability (revenue - COGS) / assets earn higher returns. Extended
with earnings stability and low accruals (Sloan 1996).

Uses fundamental data when available; proxy indicators when not:
 - Profitability proxy: consistency of positive returns (hit rate)
 - Stability proxy: coefficient of variation of returns (lower = more stable)
 - Accruals proxy: negative serial correlation of returns (mean-reverting
   returns imply lower earnings quality / higher accruals)
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


@register("equity_quality_factor")
class QualityFactorStrategy(Strategy):
    """Cross-sectional quality: long profitable/stable, short junk.

    Config params:
        lookback: window for quality proxy computation (default 252)
        n_long: number of long positions (default 5)
        n_short: number of short positions (default 5)
        vol_lookback: window for volatility estimation (default 63)
        rebalance_freq: rebalance frequency in trading days (default 21)
        profitability_weight: weight of profitability signal (default 0.4)
        stability_weight: weight of stability signal (default 0.35)
        accruals_weight: weight of accruals signal (default 0.25)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback = int(config.params.get("lookback", 252))
        self.n_long = int(config.params.get("n_long", 5))
        self.n_short = int(config.params.get("n_short", 5))
        self.vol_lookback = int(config.params.get("vol_lookback", 63))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.profitability_weight = float(config.params.get("profitability_weight", 0.4))
        self.stability_weight = float(config.params.get("stability_weight", 0.35))
        self.accruals_weight = float(config.params.get("accruals_weight", 0.25))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None

    def _profitability_proxy(self, close: NDArray) -> NDArray:
        """Profitability proxy: fraction of positive return days (hit rate).

        High hit rate = consistently profitable operations.
        """
        n_periods, n_assets = close.shape
        hit_rate = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 20:
                continue
            rets = np.diff(valid) / valid[:-1]
            hit_rate[a] = np.mean(rets > 0)

        return hit_rate

    def _stability_proxy(self, close: NDArray) -> NDArray:
        """Stability proxy: negative coefficient of variation of returns.

        Lower CV = more consistent earnings. Negated so higher = better quality.
        """
        n_periods, n_assets = close.shape
        stability = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 20:
                continue
            rets = np.diff(valid) / valid[:-1]
            mean_r = np.mean(rets)
            std_r = np.std(rets, ddof=1)
            if std_r > 1e-10:
                stability[a] = -std_r / max(abs(mean_r), 1e-10)
            else:
                stability[a] = 0.0

        return stability

    def _accruals_proxy(self, close: NDArray) -> NDArray:
        """Accruals proxy: negative serial correlation of returns.

        Higher serial correlation → more persistent earnings (lower accruals).
        Mean-reverting returns → high accruals (cash not yet received).
        Negated autocorrelation so that lower autocorrelation → lower quality.
        """
        n_periods, n_assets = close.shape
        accrual_score = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 30:
                continue
            rets = np.diff(valid) / valid[:-1]
            if np.std(rets[:-1], ddof=1) < 1e-10 or np.std(rets[1:], ddof=1) < 1e-10:
                continue
            corr = np.corrcoef(rets[:-1], rets[1:])[0, 1]
            accrual_score[a] = corr if not np.isnan(corr) else 0.0

        return accrual_score

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

        gross_profit = data.get("gross_profit")
        total_assets = data.get("total_assets")

        if (
            gross_profit is not None and total_assets is not None
            and gross_profit.ndim == 2 and total_assets.ndim == 2
            and gross_profit.shape == total_assets.shape
        ):
            profitability = gross_profit[-1] / np.maximum(total_assets[-1], 1e-10)
        else:
            profitability = self._profitability_proxy(close)

        stability = self._stability_proxy(close)
        accruals = self._accruals_proxy(close)

        # Composite quality score (accruals: higher serial corr = low accruals = good)
        composite = (
            self.profitability_weight * _zscore(profitability)
            + self.stability_weight * _zscore(stability)
            + self.accruals_weight * _zscore(accruals)
        )

        # Volatility for inverse-vol weighting
        vol_start = max(0, n_periods - self.vol_lookback)
        daily_rets = np.zeros((n_periods - vol_start, n_assets))
        for i in range(vol_start + 1, n_periods):
            daily_rets[i - vol_start] = close[i] / close[i - 1] - 1.0
        vols = np.std(daily_rets, axis=0, ddof=1)
        vols[vols < 1e-10] = 1e-10

        # Rank by composite quality (high quality = high composite)
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
                    metadata={"quality_rank": int(ranks[i]), "composite": float(composite[i])},
                ))

        return signals


def _zscore(x: NDArray) -> NDArray:
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s
