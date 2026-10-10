"""Intraday mean reversion: short-lookback indicators, 30-90 min hold.

Optimized for 1-min and 5-min bars. Uses fast RSI, Bollinger Band
deviations, and VWAP z-score to catch overextended intraday moves
that are likely to revert within the hour.

Differs from the daily mean_reversion strategy in:
- Shorter lookback windows (7/10/10 vs 14/20/20)
- More sensitive entry thresholds (1.5σ vs 2.0σ)
- Holds measured in bars not days (30-90 bars)
- Tighter stop loss (0.5% vs 2.0%)
- Higher signal frequency (checks every bar)
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


def _rolling(x: NDArray, window: int, fn) -> NDArray:
    """Apply a rolling function over the last axis."""
    n = len(x)
    result = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        result[i] = fn(x[i - window + 1:i + 1])
    return result


@register("intraday_mean_reversion")
class IntradayMeanReversionStrategy(Strategy):
    """Intraday mean reversion with fast multi-indicator confirmation.

    Config params:
        rsi_window: RSI lookback (default 7)
        rsi_oversold: RSI oversold → buy (default 35)
        rsi_overbought: RSI overbought → sell (default 65)
        bb_window: Bollinger Band window (default 10)
        bb_std: Band std multiplier (default 1.5)
        vwap_window: Rolling VWAP window (default 10)
        z_entry: z-score entry threshold (default 1.5)
        z_exit: z-score exit threshold (default 0.3)
        hold_bars: max bars to hold before forced exit (default 60)
        stop_loss_pct: stop loss % from entry (default 0.5)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.rsi_window = int(config.params.get("rsi_window", 7))
        self.rsi_oversold = float(config.params.get("rsi_oversold", 35))
        self.rsi_overbought = float(config.params.get("rsi_overbought", 65))
        self.bb_window = int(config.params.get("bb_window", 10))
        self.bb_std = float(config.params.get("bb_std", 1.5))
        self.vwap_window = int(config.params.get("vwap_window", 10))
        self.z_entry = float(config.params.get("z_entry", 1.5))
        self.z_exit = float(config.params.get("z_exit", 0.3))
        self.hold_bars = int(config.params.get("hold_bars", 60))
        self.stop_loss_pct = float(config.params.get("stop_loss_pct", 0.5))
        self._entry_bars: dict[str, int] = {}
        self._entry_price: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

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
        mean = _rolling(prices, self.bb_window, np.mean)
        std = _rolling(prices, self.bb_window, lambda x: np.std(x, ddof=1))
        dev = np.full(len(prices), 0.0, dtype=float)
        valid = ~np.isnan(mean) & (std > 1e-10)
        dev[valid] = (prices[valid] - mean[valid]) / (self.bb_std * std[valid])
        return dev

    def _vwap_zscore(self, prices: NDArray) -> NDArray:
        mean = _rolling(prices, self.vwap_window, np.mean)
        std = _rolling(prices, self.vwap_window, lambda x: np.std(x, ddof=1))
        z = np.full(len(prices), 0.0, dtype=float)
        valid = ~np.isnan(mean) & (std > 1e-10)
        z[valid] = (prices[valid] - mean[valid]) / std[valid]
        return z

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 2:
            close = close[:, 0]

        n = len(close)
        min_bars = max(self.rsi_window, self.bb_window, self.vwap_window) + 1
        if n < min_bars:
            return []

        symbols = data.get("symbols", ["asset_0"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols
        price = float(close[-1])

        rsi = self._compute_rsi(close)
        bb_dev = self._bb_deviation(close)
        vwap_z = self._vwap_zscore(close)

        signals: list[TradeSignal] = []

        # ---- Exit checks first (risk management) ----
        for sym, pos in list(self._positions.items()):
            if abs(pos) < 1e-8:
                continue

            bars_held = self._entry_bars.get(sym, 0) + 1
            self._entry_bars[sym] = bars_held
            entry_px = self._entry_price.get(sym, price)

            exit_reason = None

            # Stop loss hit
            if (pos > 0 and price <= entry_px * (1 - self.stop_loss_pct / 100)) or (pos < 0 and price >= entry_px * (1 + self.stop_loss_pct / 100)):
                exit_reason = "stop_loss"
            # Mean reversion complete
            elif abs(vwap_z[-1]) < self.z_exit:
                exit_reason = "mean_reverted"
            # Max hold expired
            elif bars_held >= self.hold_bars:
                exit_reason = "max_hold"

            if exit_reason:
                signals.append(TradeSignal(
                    symbol=sym, direction=0, strength=0.0, size=0.0,
                    metadata={
                        "action": "intraday_exit",
                        "reason": exit_reason,
                        "bars_held": bars_held,
                    },
                ))
                self._entry_bars.pop(sym, None)
                self._entry_price.pop(sym, None)

        # ---- Entry: only if not already in a position for this symbol ----
        if symbol in self._positions and abs(self._positions[symbol]) > 1e-8:
            return signals

        # Composite score
        composite = 0.0
        n_indicators = 0

        if rsi[-1] < self.rsi_oversold:
            composite += (self.rsi_oversold - rsi[-1]) / self.rsi_oversold
            n_indicators += 1
        elif rsi[-1] > self.rsi_overbought:
            composite -= (rsi[-1] - self.rsi_overbought) / (100.0 - self.rsi_overbought)
            n_indicators += 1

        if abs(bb_dev[-1]) > 1.0:
            composite -= bb_dev[-1]
            n_indicators += 1

        if abs(vwap_z[-1]) > self.z_exit:
            composite -= vwap_z[-1] * 0.5
            n_indicators += 1

        if n_indicators < 2:
            return signals

        composite = np.clip(composite / 3.0, -1.0, 1.0)
        if abs(composite) < 0.25:
            return signals

        direction = 1 if composite > 0 else -1
        strength = float(abs(composite))

        size = strength * self.config.max_position_notional * self.config.risk_budget_pct

        stop_loss = price * (1 - direction * self.stop_loss_pct / 100.0)
        take_profit = price * (1 + direction * self.stop_loss_pct * 1.5 / 100.0)

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

        self._entry_bars[symbol] = 0
        self._entry_price[symbol] = price

        return signals
