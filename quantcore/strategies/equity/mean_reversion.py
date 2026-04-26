"""Short-term mean reversion strategy: multi-indicator, 1-3 day hold.

Combines RSI, Bollinger Band deviations, and z-score of price relative
to VWAP to identify overextended assets likely to revert. Positions held
1-3 days with profit targets and stop losses.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    StrategyType,
    TradeSignal,
)
from quantcore.strategies.registry import register


def _rolling_mean(x: NDArray, window: int) -> NDArray:
    n = len(x)
    result = np.full(n, np.nan, dtype=float)
    cumsum = np.cumsum(x)
    for i in range(window - 1, n):
        result[i] = (cumsum[i] - (cumsum[i - window] if i >= window else 0)) / window
    return result


def _rolling_std(x: NDArray, window: int) -> NDArray:
    mean = _rolling_mean(x, window)
    n = len(x)
    result = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        result[i] = np.std(x[i - window + 1:i + 1], ddof=1)
    return result


@register("equity_mean_reversion")
class MeanReversionStrategy(Strategy):
    """Short-term mean reversion with multi-indicator confirmation.

    Config params:
        rsi_window: RSI lookback (default 14)
        rsi_oversold: RSI oversold threshold (default 25)
        rsi_overbought: RSI overbought threshold (default 75)
        bb_window: Bollinger Band window (default 20)
        bb_std: Bollinger Band standard deviations (default 2.0)
        vwap_window: VWAP-like rolling mean window (default 20)
        z_entry: z-score entry threshold (default 2.0)
        z_exit: z-score exit threshold (default 0.5)
        hold_days: max hold period (default 3)
        stop_loss_pct: stop loss percentage (default 2.0)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.rsi_window = int(config.params.get("rsi_window", 14))
        self.rsi_oversold = float(config.params.get("rsi_oversold", 25))
        self.rsi_overbought = float(config.params.get("rsi_overbought", 75))
        self.bb_window = int(config.params.get("bb_window", 20))
        self.bb_std = float(config.params.get("bb_std", 2.0))
        self.vwap_window = int(config.params.get("vwap_window", 20))
        self.z_entry = float(config.params.get("z_entry", 2.0))
        self.z_exit = float(config.params.get("z_exit", 0.5))
        self.hold_days = int(config.params.get("hold_days", 3))
        self.stop_loss_pct = float(config.params.get("stop_loss_pct", 2.0))
        self._entry_days: dict[str, int] = {}

    def _compute_rsi(self, prices: NDArray) -> NDArray:
        n = len(prices)
        rsi = np.full(n, 50.0, dtype=float)
        if n < self.rsi_window + 1:
            return rsi

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:self.rsi_window])
        avg_loss = np.mean(losses[:self.rsi_window])

        for i in range(self.rsi_window, n - 1):
            avg_gain = (avg_gain * (self.rsi_window - 1) + gains[i]) / self.rsi_window
            avg_loss = (avg_loss * (self.rsi_window - 1) + losses[i]) / self.rsi_window
            if avg_loss < 1e-10:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

        return rsi

    def _bb_deviation(self, prices: NDArray) -> NDArray:
        mean = _rolling_mean(prices, self.bb_window)
        std = _rolling_std(prices, self.bb_window)
        dev = np.full(len(prices), 0.0, dtype=float)
        valid = ~np.isnan(mean) & (std > 1e-10)
        dev[valid] = (prices[valid] - mean[valid]) / (self.bb_std * std[valid])
        return dev

    def _vwap_zscore(self, prices: NDArray) -> NDArray:
        mean = _rolling_mean(prices, self.vwap_window)
        std = _rolling_std(prices, self.vwap_window)
        z = np.full(len(prices), 0.0, dtype=float)
        valid = ~np.isnan(mean) & (std > 1e-10)
        z[valid] = (prices[valid] - mean[valid]) / std[valid]
        return z

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 2:
            close = close[:, 0]

        n = len(close)
        if n < max(self.rsi_window, self.bb_window, self.vwap_window) + 1:
            return []

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["asset_0"])

        rsi = self._compute_rsi(close)
        bb_dev = self._bb_deviation(close)
        vwap_z = self._vwap_zscore(close)

        symbol = symbols[0] if isinstance(symbols, list) else symbols

        # Composite score: sum of standardized indicators
        composite = 0.0
        n_indicators = 0

        # RSI signal: oversold → buy, overbought → sell
        if rsi[-1] < self.rsi_oversold:
            composite += (self.rsi_oversold - rsi[-1]) / self.rsi_oversold
            n_indicators += 1
        elif rsi[-1] > self.rsi_overbought:
            composite -= (rsi[-1] - self.rsi_overbought) / (100.0 - self.rsi_overbought)
            n_indicators += 1

        # BB deviation: below lower band → buy
        if abs(bb_dev[-1]) > 1.0:
            composite -= bb_dev[-1]  # negative dev → positive composite (buy)
            n_indicators += 1

        # VWAP z-score: below mean → buy
        if abs(vwap_z[-1]) > self.z_exit:
            composite -= vwap_z[-1] * 0.5
            n_indicators += 1

        if n_indicators < 2:
            return signals  # Need at least 2 indicators agreeing

        # Normalize composite to [-1, 1]
        composite = np.clip(composite / 3.0, -1.0, 1.0)

        if abs(composite) < 0.3:
            return signals  # Signal too weak

        direction = 1 if composite > 0 else -1
        strength = abs(composite)

        # Position sizing: inverse confidence scaling
        size = strength * self.config.max_position_notional * self.config.risk_budget_pct

        # Set stop loss and take profit
        stop_loss = close[-1] * (1 - direction * self.stop_loss_pct / 100.0)
        take_profit = close[-1] * (1 + direction * self.stop_loss_pct / 100.0)

        signals.append(TradeSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "rsi": float(rsi[-1]),
                "bb_dev": float(bb_dev[-1]),
                "vwap_z": float(vwap_z[-1]),
                "composite": float(composite),
            },
        ))

        # Check for exits on held positions
        for sym, pos in list(self._positions.items()):
            if sym == symbol and abs(pos) > 1e-8:
                days_held = self._entry_days.get(sym, 0) + 1
                self._entry_days[sym] = days_held

                if days_held >= self.hold_days or abs(vwap_z[-1]) < self.z_exit:
                    signals.append(TradeSignal(
                        symbol=sym, direction=0, strength=0.0, size=0.0,
                        metadata={"action": "mean_rev_exit", "days_held": days_held},
                    ))
                    self._entry_days.pop(sym, None)

        return signals
