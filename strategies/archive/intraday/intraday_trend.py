"""Intraday trend following: dual EMA crossover on 5-min bars.

Captures intraday directional moves using fast/slow EMA crossovers
filtered by a medium-term trend EMA. Uses ATR-based position sizing,
pyramiding, and trailing stops. Designed for 5-min resampled bars.
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


def _rolling_atr(high: NDArray, low: NDArray, close: NDArray, window: int) -> NDArray:
    n = len(close)
    tr = np.zeros(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.full(n, np.nan, dtype=float)
    atr[window - 1] = np.mean(tr[:window])
    for i in range(window, n):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr


@register("intraday_trend")
class IntradayTrendStrategy(Strategy):
    """Intraday dual-EMA trend following with ATR sizing and pyramiding.

    Config params:
        fast_window: fast EMA period (default 8)
        slow_window: slow EMA period (default 21)
        trend_window: trend filter EMA period (default 50)
        atr_window: ATR calculation window (default 14)
        atr_risk_units: ATR multiplier for position sizing (default 2.0)
        max_pyramid_units: max pyramid entries (default 3)
        pyramid_atr_add: ATR multiple to trigger pyramid add (default 0.5)
        stop_atr_multiple: stop distance in ATR (default 2.0)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.fast_window = int(config.params.get("fast_window", 8))
        self.slow_window = int(config.params.get("slow_window", 21))
        self.trend_window = int(config.params.get("trend_window", 50))
        self.atr_window = int(config.params.get("atr_window", 14))
        self.atr_risk_units = float(config.params.get("atr_risk_units", 2.0))
        self.max_pyramid_units = int(config.params.get("max_pyramid_units", 3))
        self.pyramid_atr_add = float(config.params.get("pyramid_atr_add", 0.5))
        self.stop_atr_multiple = float(config.params.get("stop_atr_multiple", 2.0))
        self._entry_price: dict[str, float] = {}
        self._units_held: dict[str, int] = {}
        self._trailing_stop: dict[str, float] = {}

    def _ema(self, prices: NDArray, window: int) -> NDArray:
        alpha = 2.0 / (window + 1)
        ema = np.full(len(prices), np.nan, dtype=float)
        ema[0] = float(prices[0])
        for i in range(1, len(prices)):
            ema[i] = alpha * float(prices[i]) + (1 - alpha) * ema[i - 1]
        return ema

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 2:
            close_arr = close[:, 0]
        else:
            close_arr = close

        high = data.get("high", close_arr * 1.005)
        low = data.get("low", close_arr * 0.995)
        if high.ndim == 2:
            high = high[:, 0]
        if low.ndim == 2:
            low = low[:, 0]

        n = len(close_arr)
        min_bars = max(self.trend_window, self.slow_window) + 1
        if n < min_bars:
            return []

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["asset_0"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols
        price = float(close_arr[-1])

        fast_ema = self._ema(close_arr, self.fast_window)
        slow_ema = self._ema(close_arr, self.slow_window)
        trend_ema = self._ema(close_arr, self.trend_window)
        atr = _rolling_atr(high, low, close_arr, self.atr_window)

        if np.isnan(atr[-1]) or atr[-1] < 1e-10:
            return signals

        trend_up = price > trend_ema[-1] if not np.isnan(trend_ema[-1]) else None
        trend_down = price < trend_ema[-1] if not np.isnan(trend_ema[-1]) else None

        fast_above = fast_ema[-1] > slow_ema[-1]
        fast_below = fast_ema[-1] < slow_ema[-1]
        prev_fast_above = fast_ema[-2] > slow_ema[-2]
        prev_fast_below = fast_ema[-2] < slow_ema[-2]

        bullish_cross = fast_above and not prev_fast_above
        bearish_cross = fast_below and not prev_fast_below

        units = self._units_held.get(symbol, 0)
        entry = self._entry_price.get(symbol, 0.0)
        trail = self._trailing_stop.get(symbol, 0.0)

        # --- Trail stop check ---
        if units > 0:
            new_trail = price - self.stop_atr_multiple * float(atr[-1])
            if new_trail > trail or trail == 0.0:
                self._trailing_stop[symbol] = new_trail
                trail = new_trail
            if price <= trail:
                signals.append(TradeSignal(
                    symbol=symbol, direction=0, strength=0.0, size=0.0,
                    metadata={"action": "trail_stop", "reason": "trailing_stop_long"},
                ))
                self._units_held.pop(symbol, None)
                self._entry_price.pop(symbol, None)
                self._trailing_stop.pop(symbol, None)
                return signals
        elif units < 0:
            new_trail = price + self.stop_atr_multiple * float(atr[-1])
            if new_trail < trail or trail == 0.0:
                self._trailing_stop[symbol] = new_trail
                trail = new_trail
            if price >= trail:
                signals.append(TradeSignal(
                    symbol=symbol, direction=0, strength=0.0, size=0.0,
                    metadata={"action": "trail_stop", "reason": "trailing_stop_short"},
                ))
                self._units_held.pop(symbol, None)
                self._entry_price.pop(symbol, None)
                self._trailing_stop.pop(symbol, None)
                return signals

        # --- Entry: bullish crossover in uptrend ---
        if bullish_cross and trend_up and units <= 0:
            risk_per_unit = self.atr_risk_units * float(atr[-1])
            if risk_per_unit > 0:
                pos_size = (float(self.config.max_position_notional) * float(self.config.risk_budget_pct)) / risk_per_unit
            else:
                pos_size = 0.0
            stop = price - self.stop_atr_multiple * float(atr[-1])

            signals.append(TradeSignal(
                symbol=symbol,
                direction=1,
                strength=1.0,
                size=pos_size * price,
                stop_loss=stop,
                metadata={"atr": float(atr[-1]), "units": 1, "entry": "bullish_cross"},
            ))
            self._entry_price[symbol] = price
            self._units_held[symbol] = 1
            self._trailing_stop[symbol] = stop

        # --- Pyramiding: add to winning long ---
        elif (
            trend_up
            and units > 0
            and units < self.max_pyramid_units
            and price > entry + self.pyramid_atr_add * float(atr[-1]) * units
        ):
            risk_per_unit = self.atr_risk_units * float(atr[-1])
            if risk_per_unit > 0:
                pos_size = (float(self.config.max_position_notional) * float(self.config.risk_budget_pct)) / risk_per_unit
            else:
                pos_size = 0.0

            signals.append(TradeSignal(
                symbol=symbol,
                direction=1,
                strength=0.5,
                size=pos_size * price,
                stop_loss=trail if trail > 0 else price - self.stop_atr_multiple * float(atr[-1]),
                metadata={"atr": float(atr[-1]), "units": units + 1, "pyramid": True},
            ))
            self._units_held[symbol] = units + 1

        # --- Entry: bearish crossover in downtrend ---
        elif bearish_cross and trend_down and units >= 0:
            risk_per_unit = self.atr_risk_units * float(atr[-1])
            if risk_per_unit > 0:
                pos_size = (float(self.config.max_position_notional) * float(self.config.risk_budget_pct)) / risk_per_unit
            else:
                pos_size = 0.0
            stop = price + self.stop_atr_multiple * float(atr[-1])

            signals.append(TradeSignal(
                symbol=symbol,
                direction=-1,
                strength=1.0,
                size=pos_size * price,
                stop_loss=stop,
                metadata={"atr": float(atr[-1]), "units": -1, "entry": "bearish_cross"},
            ))
            self._entry_price[symbol] = price
            self._units_held[symbol] = -1
            self._trailing_stop[symbol] = stop

        # --- Exit: opposing crossover ---
        elif units > 0 and bearish_cross:
            signals.append(TradeSignal(
                symbol=symbol, direction=0, strength=0.0, size=0.0,
                metadata={"action": "trend_exit", "reason": "bearish_cross"},
            ))
            self._units_held.pop(symbol, None)
            self._entry_price.pop(symbol, None)
            self._trailing_stop.pop(symbol, None)

        elif units < 0 and bullish_cross:
            signals.append(TradeSignal(
                symbol=symbol, direction=0, strength=0.0, size=0.0,
                metadata={"action": "trend_exit", "reason": "bullish_cross"},
            ))
            self._units_held.pop(symbol, None)
            self._entry_price.pop(symbol, None)
            self._trailing_stop.pop(symbol, None)

        return signals
