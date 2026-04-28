"""Intraday breakout: Donchian channel breakout with scaled entry.

Uses a rolling N-bar high/low range as support/resistance. Enters on a
confirmed break with half size, adds the second half on a retest of the
breakout level. Risk-managed with stops at the opposite side of the range
and a time-based exit near session close. Designed for 5-min resampled bars.
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


def _rolling_high(prices: NDArray, window: int) -> NDArray:
    result = np.full(len(prices), np.nan, dtype=float)
    for i in range(window - 1, len(prices)):
        result[i] = np.max(prices[i - window + 1:i + 1])
    return result


def _rolling_low(prices: NDArray, window: int) -> NDArray:
    result = np.full(len(prices), np.nan, dtype=float)
    for i in range(window - 1, len(prices)):
        result[i] = np.min(prices[i - window + 1:i + 1])
    return result


def _rolling_atr(high: NDArray, low: NDArray, close: NDArray, window: int) -> NDArray:
    n = len(close)
    tr = np.zeros(n, dtype=float)
    tr[0] = float(high[0]) - float(low[0])
    for i in range(1, n):
        tr[i] = max(
            float(high[i]) - float(low[i]),
            abs(float(high[i]) - float(close[i - 1])),
            abs(float(low[i]) - float(close[i - 1])),
        )
    atr = np.full(n, np.nan, dtype=float)
    atr[window - 1] = np.mean(tr[:window])
    for i in range(window, n):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr


@register("intraday_breakout")
class IntradayBreakoutStrategy(Strategy):
    """Donchian channel breakout with scaled entry and tight risk management.

    Config params:
        range_bars: bars for the Donchian channel (default 20)
        entry_buffer_pct: % beyond channel to confirm breakout (default 0.1)
        stop_buffer_pct: % inside opposite channel side to place stop (default 0.05)
        target_atr_multiple: profit target in ATR multiples (default 2.0)
        atr_window: ATR period (default 14)
        first_scale_pct: % of full size on initial break (default 50)
        session_bars: bars in a full session for time-based exit (default 78 for 5-min)
        exit_before_close: bars before session end to force flatten (default 10)
        cooldown_bars: bars to wait after an exit before re-entering (default 5)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.range_bars = int(config.params.get("range_bars", 20))
        self.entry_buffer_pct = float(config.params.get("entry_buffer_pct", 0.1))
        self.stop_buffer_pct = float(config.params.get("stop_buffer_pct", 0.05))
        self.target_atr_multiple = float(config.params.get("target_atr_multiple", 2.0))
        self.atr_window = int(config.params.get("atr_window", 14))
        self.first_scale_pct = float(config.params.get("first_scale_pct", 50.0)) / 100.0
        self.session_bars = int(config.params.get("session_bars", 78))
        self.exit_before_close = int(config.params.get("exit_before_close", 10))
        self.cooldown_bars = int(config.params.get("cooldown_bars", 5))

        self._entry_price: dict[str, float] = {}
        self._entry_direction: dict[str, int] = {}
        self._first_scale_filled: dict[str, bool] = {}
        self._breach_level: dict[str, float] = {}
        self._bars_since_exit: dict[str, int] = {}
        self._bars_held: dict[str, int] = {}

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        high = data.get("high")
        low = data.get("low")
        if close is None:
            return []

        if close.ndim == 2:
            close_arr = close[:, 0]
            high_arr = high[:, 0] if high is not None else close_arr * 1.005
            low_arr = low[:, 0] if low is not None else close_arr * 0.995
        else:
            close_arr = close
            high_arr = high if high is not None else close_arr * 1.005
            low_arr = low if low is not None else close_arr * 0.995

        n = len(close_arr)
        min_bars = max(self.range_bars, self.atr_window) + 1
        if n < min_bars:
            return []

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["asset_0"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols
        price = float(close_arr[-1])

        atr = _rolling_atr(high_arr, low_arr, close_arr, self.atr_window)
        if np.isnan(atr[-1]) or atr[-1] < 1e-10:
            return signals

        # Channel from previous range_bars bars (exclude current bar so close can
        # actually break above / below the channel).
        if n < self.range_bars + 1:
            return signals
        ch = float(np.max(high_arr[-self.range_bars - 1 : -1]))
        cl = float(np.min(low_arr[-self.range_bars - 1 : -1]))
        channel_width = ch - cl
        if channel_width < 1e-8:
            return signals

        # --- Position management ---
        pos = self._positions.get(symbol, 0.0)
        direction = self._entry_direction.get(symbol, 0)
        entry_px = self._entry_price.get(symbol, price)
        self._bars_held[symbol] = self._bars_held.get(symbol, 0) + 1

        # Time-based exit near session close
        bar_in_session = n % self.session_bars
        if abs(pos) > 1e-8 and bar_in_session >= self.session_bars - self.exit_before_close:
            signals.append(TradeSignal(
                symbol=symbol, direction=0, strength=0.0, size=0.0,
                metadata={"action": "session_exit", "reason": "near_close", "bar": bar_in_session},
            ))
            self._clear_state(symbol)
            self._bars_since_exit[symbol] = 0
            return signals

        # Stop loss: opposite side of channel
        if abs(pos) > 1e-8:
            stop_hit = False
            if direction > 0:
                stop_level = cl * (1 - self.stop_buffer_pct / 100.0)
                target = entry_px + self.target_atr_multiple * float(atr[-1])
                if price <= stop_level:
                    stop_hit = True
                elif price >= target:
                    # Take profit at ATR target
                    signals.append(TradeSignal(
                        symbol=symbol, direction=0, strength=0.0, size=0.0,
                        metadata={"action": "take_profit", "reason": "atr_target"},
                    ))
                    self._clear_state(symbol)
                    self._bars_since_exit[symbol] = 0
                    return signals
            elif direction < 0:
                stop_level = ch * (1 + self.stop_buffer_pct / 100.0)
                target = entry_px - self.target_atr_multiple * float(atr[-1])
                if price >= stop_level:
                    stop_hit = True
                elif price <= target:
                    signals.append(TradeSignal(
                        symbol=symbol, direction=0, strength=0.0, size=0.0,
                        metadata={"action": "take_profit", "reason": "atr_target"},
                    ))
                    self._clear_state(symbol)
                    self._bars_since_exit[symbol] = 0
                    return signals

            if stop_hit:
                signals.append(TradeSignal(
                    symbol=symbol, direction=0, strength=0.0, size=0.0,
                    metadata={"action": "stop_loss", "reason": "channel_stop"},
                ))
                self._clear_state(symbol)
                self._bars_since_exit[symbol] = 0
                return signals

            # Second scale: retest of breakout level for remaining position
            if direction > 0 and not self._first_scale_filled.get(symbol, False):
                breach_level = self._breach_level.get(symbol, ch)
                if price <= breach_level * 1.001:
                    # Retest — add second scale
                    remaining = 1.0 - self.first_scale_pct
                    size = (remaining * float(self.config.max_position_notional)
                            * float(self.config.risk_budget_pct))
                    stop = cl * (1 - self.stop_buffer_pct / 100.0)
                    signals.append(TradeSignal(
                        symbol=symbol, direction=1, strength=0.7,
                        size=size, stop_loss=stop,
                        metadata={"action": "second_scale", "atr": float(atr[-1])},
                    ))
                    self._first_scale_filled[symbol] = True
            elif direction < 0 and not self._first_scale_filled.get(symbol, False):
                breach_level = self._breach_level.get(symbol, cl)
                if price >= breach_level * 0.999:
                    remaining = 1.0 - self.first_scale_pct
                    size = (remaining * float(self.config.max_position_notional)
                            * float(self.config.risk_budget_pct))
                    stop = ch * (1 + self.stop_buffer_pct / 100.0)
                    signals.append(TradeSignal(
                        symbol=symbol, direction=-1, strength=0.7,
                        size=size, stop_loss=stop,
                        metadata={"action": "second_scale", "atr": float(atr[-1])},
                    ))
                    self._first_scale_filled[symbol] = True

            return signals

        # Cooldown after exit
        bars_since = self._bars_since_exit.get(symbol, self.cooldown_bars + 1)
        self._bars_since_exit[symbol] = bars_since + 1
        if bars_since < self.cooldown_bars:
            return signals

        # --- Entry: breakout above channel ---
        entry_threshold_high = ch * (1 + self.entry_buffer_pct / 100.0)
        entry_threshold_low = cl * (1 - self.entry_buffer_pct / 100.0)

        # Confirm with volume or momentum (price above prior bar close)
        if price > entry_threshold_high and price > float(close_arr[-2]):
            scale_size = self.first_scale_pct
            size = (scale_size * float(self.config.max_position_notional)
                    * float(self.config.risk_budget_pct))
            stop = cl * (1 - self.stop_buffer_pct / 100.0)

            signals.append(TradeSignal(
                symbol=symbol, direction=1, strength=0.8,
                size=size, stop_loss=stop,
                metadata={
                    "action": "breakout_long",
                    "channel_high": ch, "channel_low": cl,
                    "atr": float(atr[-1]),
                },
            ))
            self._entry_price[symbol] = price
            self._entry_direction[symbol] = 1
            self._first_scale_filled[symbol] = False
            self._breach_level[symbol] = ch
            self._bars_held[symbol] = 0

        elif price < entry_threshold_low and price < float(close_arr[-2]):
            scale_size = self.first_scale_pct
            size = (scale_size * float(self.config.max_position_notional)
                    * float(self.config.risk_budget_pct))
            stop = ch * (1 + self.stop_buffer_pct / 100.0)

            signals.append(TradeSignal(
                symbol=symbol, direction=-1, strength=0.8,
                size=size, stop_loss=stop,
                metadata={
                    "action": "breakout_short",
                    "channel_high": ch, "channel_low": cl,
                    "atr": float(atr[-1]),
                },
            ))
            self._entry_price[symbol] = price
            self._entry_direction[symbol] = -1
            self._first_scale_filled[symbol] = False
            self._breach_level[symbol] = cl
            self._bars_held[symbol] = 0

        return signals

    def _clear_state(self, symbol: str) -> None:
        self._entry_price.pop(symbol, None)
        self._entry_direction.pop(symbol, None)
        self._first_scale_filled.pop(symbol, None)
        self._breach_level.pop(symbol, None)
        self._bars_held.pop(symbol, None)
