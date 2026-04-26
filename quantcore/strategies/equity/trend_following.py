"""Multi-timeframe trend following strategy with ATR sizing and pyramiding.

Captures trends across multiple timeframes using moving average crossovers,
sizes positions by ATR (volatility-adjusted), and allows pyramiding
(adding to winning positions) up to a max unit count.
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


def _rolling_atr(high: NDArray, low: NDArray, close: NDArray, window: int) -> NDArray:
    """Compute Average True Range."""
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


@register("equity_trend_following")
class TrendFollowingStrategy(Strategy):
    """Multi-timeframe trend following with ATR sizing and pyramiding.

    Config params:
        fast_window: fast MA window (default 20)
        slow_window: slow MA window (default 50)
        trend_window: long-term trend filter (default 200)
        atr_window: ATR calculation window (default 20)
        atr_risk_units: ATR units for position sizing (default 2)
        max_pyramid_units: max units to pyramid (default 4)
        pyramid_atr_add: ATR multiple to add pyramid (default 0.5)
        stop_atr_multiple: stop loss ATR multiple (default 2.0)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.fast_window = int(config.params.get("fast_window", 20))
        self.slow_window = int(config.params.get("slow_window", 50))
        self.trend_window = int(config.params.get("trend_window", 200))
        self.atr_window = int(config.params.get("atr_window", 20))
        self.atr_risk_units = float(config.params.get("atr_risk_units", 2.0))
        self.max_pyramid_units = int(config.params.get("max_pyramid_units", 4))
        self.pyramid_atr_add = float(config.params.get("pyramid_atr_add", 0.5))
        self.stop_atr_multiple = float(config.params.get("stop_atr_multiple", 2.0))
        self._entry_price: dict[str, float] = {}
        self._units_held: dict[str, int] = {}

    def _ema(self, prices: NDArray, window: int) -> NDArray:
        alpha = 2.0 / (window + 1)
        ema = np.full(len(prices), np.nan, dtype=float)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

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
        if n < max(self.trend_window, self.slow_window) + 1:
            return []

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["asset_0"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols

        # Compute indicators
        fast_ema = self._ema(close, self.fast_window)
        slow_ema = self._ema(close, self.slow_window)
        trend_ema = self._ema(close, self.trend_window)
        atr = _rolling_atr(high, low, close, self.atr_window)

        if np.isnan(atr[-1]) or atr[-1] < 1e-10:
            return signals

        # Trend direction
        trend_up = close[-1] > trend_ema[-1] if not np.isnan(trend_ema[-1]) else None
        trend_down = close[-1] < trend_ema[-1] if not np.isnan(trend_ema[-1]) else None

        # MA crossover signal
        fast_above_slow = fast_ema[-1] > slow_ema[-1]
        fast_below_slow = fast_ema[-1] < slow_ema[-1]
        prev_fast_above = fast_ema[-2] > slow_ema[-2]
        prev_fast_below = fast_ema[-2] < slow_ema[-2]

        bullish_cross = fast_above_slow and not prev_fast_above
        bearish_cross = fast_below_slow and not prev_fast_below

        units = self._units_held.get(symbol, 0)
        entry = self._entry_price.get(symbol, 0.0)

        # Entry: bullish crossover in uptrend
        if bullish_cross and trend_up and units <= 0:
            risk_per_unit = self.atr_risk_units * atr[-1]
            position_size = (self.config.max_position_notional * self.config.risk_budget_pct) / risk_per_unit
            stop = close[-1] - self.stop_atr_multiple * atr[-1]

            signals.append(TradeSignal(
                symbol=symbol,
                direction=1,
                strength=1.0,
                size=position_size * close[-1],
                stop_loss=stop,
                metadata={"atr": float(atr[-1]), "units": 1},
            ))
            self._entry_price[symbol] = float(close[-1])
            self._units_held[symbol] = 1

        # Pyramiding: add to winning long position
        elif (
            trend_up
            and units > 0
            and units < self.max_pyramid_units
            and close[-1] > entry + self.pyramid_atr_add * atr[-1] * units
        ):
            risk_per_unit = self.atr_risk_units * atr[-1]
            position_size = (self.config.max_position_notional * self.config.risk_budget_pct) / risk_per_unit

            stop = close[-1] - self.stop_atr_multiple * atr[-1]
            signals.append(TradeSignal(
                symbol=symbol,
                direction=1,
                strength=0.5,
                size=position_size * close[-1],
                stop_loss=stop,
                metadata={"atr": float(atr[-1]), "units": units + 1, "pyramid": True},
            ))
            self._units_held[symbol] = units + 1

        # Entry: bearish crossover in downtrend
        elif bearish_cross and trend_down and units >= 0:
            risk_per_unit = self.atr_risk_units * atr[-1]
            position_size = (self.config.max_position_notional * self.config.risk_budget_pct) / risk_per_unit
            stop = close[-1] + self.stop_atr_multiple * atr[-1]

            signals.append(TradeSignal(
                symbol=symbol,
                direction=-1,
                strength=1.0,
                size=position_size * close[-1],
                stop_loss=stop,
                metadata={"atr": float(atr[-1]), "units": -1},
            ))
            self._entry_price[symbol] = float(close[-1])
            self._units_held[symbol] = -1

        # Exit: opposing crossover
        elif units > 0 and bearish_cross:
            signals.append(TradeSignal(
                symbol=symbol, direction=0, strength=0.0, size=0.0,
                metadata={"action": "trend_exit", "reason": "bearish_cross"},
            ))
            self._units_held.pop(symbol, None)
            self._entry_price.pop(symbol, None)

        elif units < 0 and bullish_cross:
            signals.append(TradeSignal(
                symbol=symbol, direction=0, strength=0.0, size=0.0,
                metadata={"action": "trend_exit", "reason": "bullish_cross"},
            ))
            self._units_held.pop(symbol, None)
            self._entry_price.pop(symbol, None)

        return signals
