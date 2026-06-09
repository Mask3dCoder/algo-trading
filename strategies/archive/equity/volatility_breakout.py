"""Volatility breakout: Bollinger squeeze + Keltner expansion (TTM Squeeze style).

Volatility contraction (Bollinger Bands inside Keltner Channels) precedes
volatility expansion and directional moves. When the squeeze fires
(Bollinger Bands expand outside Keltner Channels), enter in the direction
of momentum. Exit on opposing momentum or squeeze restart.

Uses existing signals from quantcore.signals.technical for RSI, Bollinger
Bands, and ATR.
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


@register("equity_volatility_breakout")
class VolatilityBreakoutStrategy(Strategy):
    """TTM Squeeze-style vol breakout: squeeze detection + momentum entry.

    Config params:
        bb_period: Bollinger Band period (default 20)
        bb_std: Bollinger Band standard deviations (default 2.0)
        keltner_period: Keltner Channel ATR period (default 20)
        keltner_atr_mult: Keltner Channel ATR multiple (default 1.5)
        momentum_period: lookback for momentum direction (default 12)
        rsi_period: RSI period for confirmation (default 14)
        stop_atr_mult: stop loss ATR multiple (default 2.0)
        max_hold_days: max days in position (default 15)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.bb_period = int(config.params.get("bb_period", 20))
        self.bb_std = float(config.params.get("bb_std", 2.0))
        self.keltner_period = int(config.params.get("keltner_period", 20))
        self.keltner_atr_mult = float(config.params.get("keltner_atr_mult", 1.5))
        self.momentum_period = int(config.params.get("momentum_period", 12))
        self.rsi_period = int(config.params.get("rsi_period", 14))
        self.stop_atr_mult = float(config.params.get("stop_atr_mult", 2.0))
        self.max_hold_days = int(config.params.get("max_hold_days", 15))
        self._squeeze_state: dict[str, str] = {}  # "squeezing" or "fired"
        self._entry_days: dict[str, int] = {}

    def _sma(self, x: NDArray, period: int) -> NDArray:
        n = len(x)
        result = np.full(n, np.nan, dtype=float)
        if n < period:
            return result
        cumsum = np.cumsum(x)
        result[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period
        return result

    def _atr(self, high: NDArray, low: NDArray, close: NDArray, period: int) -> NDArray:
        n = len(close)
        tr = np.zeros(n, dtype=float)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        atr = np.full(n, np.nan, dtype=float)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def _compute_rsi(self, prices: NDArray, period: int) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 2:
            close = close[:, 0]

        high = data.get("high", close * 1.005)
        low = data.get("low", close * 0.995)
        if high.ndim == 2:
            high = high[:, 0]
        if low.ndim == 2:
            low = low[:, 0]

        n = len(close)
        min_period = max(self.bb_period, self.keltner_period, self.momentum_period, self.rsi_period) + 1
        if n < min_period:
            return []

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["asset_0"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols

        # Compute indicators
        bb_mid = self._sma(close, self.bb_period)
        std_vals = np.full(n, np.nan, dtype=float)
        for i in range(self.bb_period - 1, n):
            std_vals[i] = np.std(close[i - self.bb_period + 1:i + 1], ddof=1)
        bb_upper = bb_mid + self.bb_std * std_vals
        bb_lower = bb_mid - self.bb_std * std_vals
        bb_width = bb_upper - bb_lower

        atr = self._atr(high, low, close, self.keltner_period)
        keltner_mid = self._sma(close, self.keltner_period)
        keltner_upper = keltner_mid + self.keltner_atr_mult * atr
        keltner_lower = keltner_mid - self.keltner_atr_mult * atr
        keltner_width = keltner_upper - keltner_lower

        # Squeeze detection: BB width < Keltner width
        if np.isnan(bb_width[-1]) or np.isnan(keltner_width[-1]):
            return signals

        squeezing = bb_width[-1] < keltner_width[-1]
        prev_squeezing = (
            not np.isnan(bb_width[-2]) and not np.isnan(keltner_width[-2])
            and bb_width[-2] < keltner_width[-2]
        ) if n >= 2 else False

        # Momentum direction: linear regression slope
        mom_window = close[-self.momentum_period:]
        if len(mom_window) >= 2:
            x = np.arange(len(mom_window), dtype=float)
            x_mean = np.mean(x)
            y_mean = np.mean(mom_window)
            ss_xy = np.sum((x - x_mean) * (mom_window - y_mean))
            ss_xx = np.sum((x - x_mean) ** 2)
            slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
            mom_direction = 1 if slope > 0 else -1
        else:
            mom_direction = 0

        # RSI confirmation
        rsi_val = self._compute_rsi(close, self.rsi_period)

        state = self._squeeze_state.get(symbol, "none")
        held = symbol in self._positions and abs(self._positions.get(symbol, 0)) > 1e-8

        # Check exits first
        if held and symbol in self._entry_days:
            self._entry_days[symbol] += 1
            exit_reason = None
            if self._entry_days[symbol] >= self.max_hold_days:
                exit_reason = "max_hold"
            elif squeezing and not prev_squeezing:
                exit_reason = "squeeze_restart"
            elif symbol in self._positions:
                pos_dir = np.sign(self._positions[symbol])
                if pos_dir > 0 and mom_direction < 0 and rsi_val < 40:
                    exit_reason = "momentum_reversal"
                elif pos_dir < 0 and mom_direction > 0 and rsi_val > 60:
                    exit_reason = "momentum_reversal"

            if exit_reason:
                signals.append(TradeSignal(
                    symbol=symbol, direction=0, strength=0.0, size=0.0,
                    metadata={"action": "vol_breakout_exit", "reason": exit_reason},
                ))
                self._entry_days.pop(symbol, None)

        # Squeeze fired: was squeezing, now not
        fired = prev_squeezing and not squeezing

        if fired:
            self._squeeze_state[symbol] = "fired"

            # Entry in momentum direction with RSI confirmation
            confirmed = (
                (mom_direction > 0 and rsi_val > 50)
                or (mom_direction < 0 and rsi_val < 50)
            )

            if confirmed and not np.isnan(atr[-1]) and atr[-1] > 1e-10:
                risk = self.stop_atr_mult * atr[-1]
                position_size = (
                    self.config.max_position_notional * self.config.risk_budget_pct
                ) / (risk * close[-1]) if close[-1] > 0 else 0.0

                stop = close[-1] - mom_direction * self.stop_atr_mult * atr[-1]

                signals.append(TradeSignal(
                    symbol=symbol,
                    direction=mom_direction,
                    strength=min(abs(slope) / max(np.std(close[-self.momentum_period:]), 1e-10), 1.0),
                    size=position_size * close[-1],
                    stop_loss=stop,
                    metadata={
                        "squeeze_fired": True,
                        "rsi": float(rsi_val),
                        "atr": float(atr[-1]),
                        "bb_width": float(bb_width[-1]),
                        "keltner_width": float(keltner_width[-1]),
                    },
                ))
                self._entry_days[symbol] = 0

        elif squeezing:
            self._squeeze_state[symbol] = "squeezing"

        return signals


def _zscore(x: NDArray) -> NDArray:
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s
