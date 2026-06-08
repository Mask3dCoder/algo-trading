"""Cross-sectional equity momentum (clean Jegadeesh-Titman).

Rank stocks by 12-month return (t-12 to t-1), long top decile, short
bottom decile, equal-weighted, dollar-neutral.  The most-reproduced
anomaly in empirical asset pricing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("equity_cross_sectional_momentum")
class CrossSectionalMomentumStrategy(Strategy):
    """Cross-sectional momentum: long winners, short losers.

    Config params:
        lookback_days: return lookback in trading days (default 252)
        top_pct: fraction of universe for long leg (default 0.10)
        bottom_pct: fraction of universe for short leg (default 0.10)
        rebalance_freq: rebalance frequency in trading days (default 21)
        min_price: minimum stock price filter (default 5.0)
        min_volume: minimum average daily dollar volume filter (default 5e6)
        min_history: minimum bars of history required (default 200)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback_days = int(config.params.get("lookback_days", 252))
        self.top_pct = float(config.params.get("top_pct", 0.10))
        self.bottom_pct = float(config.params.get("bottom_pct", 0.10))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.min_price = float(config.params.get("min_price", 5.0))
        self.min_volume = float(config.params.get("min_volume", 5e6))
        self.min_history = int(config.params.get("min_history", 200))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        volume = data.get("volume")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq and self._current_weights is not None:
            return _signals_from_weights(self._current_weights, symbols, self.config.max_position_notional)

        if n_periods < self.lookback_days + 1 or n_periods < self.min_history:
            return []

        # Universe filters
        valid = np.ones(n_assets, dtype=bool)
        valid &= close[-1] >= self.min_price
        if volume is not None and volume.ndim == 2 and volume.shape[0] >= 21:
            avg_dollar_vol = np.mean(volume[-21:] * close[-21:], axis=0)
            valid &= avg_dollar_vol >= self.min_volume

        n_valid = int(np.sum(valid))
        if n_valid < 4:
            return []

        # 12-month return (t-12 to t-1, skipping most recent day)
        ret_12m = close[-2] / np.maximum(close[-self.lookback_days - 1], 1e-10) - 1.0
        ret_12m[~valid] = np.nan

        # Rank by return
        n_long = max(1, int(n_valid * self.top_pct))
        n_short = max(1, int(n_valid * self.bottom_pct))

        # Sort valid assets by return
        valid_indices = np.where(valid)[0]
        sorted_idx = valid_indices[np.argsort(ret_12m[valid_indices])]

        short_leg = sorted_idx[:n_short]
        long_leg = sorted_idx[-n_long:]

        # Ensure no overlap
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
