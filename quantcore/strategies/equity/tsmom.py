"""Time-Series Momentum (TSMOM) — per-stock trend following.

Moskowitz, Ooi, Pedersen (2012): each stock's own past return predicts
its future direction.  Long stocks with positive 12-month return, short
those with negative.  Volatility-scaled to target 15% annualized vol
per position.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("equity_tsmom")
class TSMOMStrategy(Strategy):
    """Time-series momentum: long positive-trend, short negative-trend.

    Config params:
        lookback_days: trailing return window in trading days (default 252)
        vol_lookback: window for vol estimation in trading days (default 60)
        target_annual_vol: annualized vol target per position (default 0.15)
        rebalance_freq: rebalance frequency in trading days (default 21)
        min_price: minimum stock price filter (default 5.0)
        min_volume: minimum average daily dollar volume filter (default 5e6)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback_days = int(config.params.get("lookback_days", 252))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))
        self.target_annual_vol = float(config.params.get("target_annual_vol", 0.15))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.min_price = float(config.params.get("min_price", 5.0))
        self.min_volume = float(config.params.get("min_volume", 5e6))
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

        min_required = max(self.lookback_days, self.vol_lookback) + 1
        if n_periods < min_required:
            return []

        # Daily returns
        rets = np.diff(close, axis=0) / np.maximum(close[:-1], 1e-10)

        # Annualized vol from trailing vol_lookback days
        vol_slice = rets[-self.vol_lookback:]
        ann_vol = np.std(vol_slice, axis=0, ddof=1) * np.sqrt(252)
        ann_vol = np.maximum(ann_vol, 1e-4)

        # Filters
        valid = np.ones(n_assets, dtype=bool)
        if close.shape[0] > 0:
            valid &= close[-1] >= self.min_price
        if volume is not None and volume.ndim == 2 and volume.shape[0] >= 21:
            avg_dollar_vol = np.mean(volume[-21:] * close[-21:], axis=0)
            valid &= avg_dollar_vol >= self.min_volume

        # 12-month trailing return
        ret_12m = close[-1] / np.maximum(close[-self.lookback_days - 1], 1e-10) - 1.0

        # Direction: long if positive, short if negative
        signal = np.sign(ret_12m)
        signal[~valid] = 0.0

        if np.all(signal == 0):
            return []

        # Vol-scaled weights: target vol / realized vol per position
        raw_weights = signal * self.target_annual_vol / ann_vol
        raw_weights[~valid] = 0.0

        total = np.sum(np.abs(raw_weights))
        if total < 1e-10:
            return []

        weights = raw_weights / total
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
