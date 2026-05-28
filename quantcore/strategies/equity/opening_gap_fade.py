"""Opening gap fade: intraday mean-reversion on overnight gaps.

Core hypothesis:
  Stocks that gap significantly at the open tend to partially revert during
  the first 30-60 minutes of trading as overnight orders are absorbed and
  price discovery normalises.

Entry (first tick of the session):
  gap_pct = (today_open - prev_close) / prev_close * 100
  Filter: |gap_pct| > threshold (default 2%), ADV > $10M, price > $10
  Fade direction: positive gap → short, negative gap → long
  Strength: min(|gap_pct| / 5.0, 1.0)
  Max 10 concurrent positions, max 5% portfolio per trade.

Exit (checked every tick):
  1. Time exit: current_time >= entry_time + exit_minutes → flatten
  2. Retracement target: price retraces 50% of original gap → take profit
  3. Stop loss: 1.5% adverse move from entry price
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("opening_gap_fade")
class OpeningGapFadeStrategy(Strategy):
    """Fade opening gaps with time-based and retracement exits.

    Data dict expectations:
        open:       NDArray (n_periods, n_assets) — today's open prices
        close:      NDArray (n_periods, n_assets) — running close prices
        prev_close: dict[str, float] — {ticker: yesterday_close}
        current_time: datetime — current ET wall-clock time
        symbols:    list[str] — ticker symbols

    Config params:
        gap_threshold_pct:  min |gap%| to trade (default 2.0)
        exit_minutes:       minutes after open to force flatten (default 30)
        retracement_target: fraction of gap to use as profit target (default 0.50)
        max_positions:      max concurrent positions (default 10)
        max_position_pct:   max fraction of portfolio per position (default 0.05)
        min_adv_dollars:    min avg daily volume in dollars (default 10_000_000)
        min_price:          min stock price to trade (default 10.0)
        stop_loss_pct:      stop loss distance from entry (default 1.5)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        self.gap_threshold_pct = float(config.params.get("gap_threshold_pct", 2.0))
        self.exit_minutes = int(config.params.get("exit_minutes", 30))
        self.retracement_target = float(config.params.get("retracement_target", 0.50))
        self.max_positions = int(config.params.get("max_positions", 10))
        self.max_position_pct = float(config.params.get("max_position_pct", 0.05))
        self.min_adv_dollars = float(config.params.get("min_adv_dollars", 10_000_000))
        self.min_price = float(config.params.get("min_price", 10.0))
        self.stop_loss_pct = float(config.params.get("stop_loss_pct", 1.5))

        # Per-symbol session state
        self._entry_price: dict[str, float] = {}
        self._gap_pct: dict[str, float] = {}
        self._entry_time: datetime | None = None
        self._traded_today: bool = False
        self._prev_close: dict[str, float] = {}

    def _reset_session(self) -> None:
        """Reset state at start of a new trading day."""
        self._entry_price.clear()
        self._gap_pct.clear()
        self._entry_time = None
        self._traded_today = False
        self._prev_close.clear()

    def on_data(self, data: dict[str, NDArray | dict | datetime]) -> list[TradeSignal]:
        close = data.get("close")
        open_prices = data.get("open")
        symbols = data.get("symbols", [])
        current_time = data.get("current_time")
        prev_close_raw = data.get("prev_close")

        if close is None or open_prices is None or not symbols:
            return []
        if not isinstance(current_time, datetime):
            return []
        if not isinstance(prev_close_raw, dict):
            return []

        # Detect new trading day → reset
        if self._entry_time is not None and current_time.date() != self._entry_time.date():
            self._reset_session()

        if close.ndim == 1:
            close = close.reshape(-1, 1)
        if open_prices.ndim == 1:
            open_prices = open_prices.reshape(-1, 1)

        n_periods, n_assets = close.shape
        if n_periods < 2:
            return []

        self._prev_close = prev_close_raw
        signals: list[TradeSignal] = []

        # --- Exit logic: check all open positions first ---
        exit_symbols: list[str] = []
        for sym in list(self._entry_price.keys()):
            idx = None
            for i, s in enumerate(symbols):
                if s == sym:
                    idx = i
                    break
            if idx is None:
                continue

            entry_px = self._entry_price[sym]
            gap = self._gap_pct[sym]
            current_px = float(close[-1, idx])

            # Determine direction from gap sign: positive gap → short (-1), negative → long (+1)
            direction = -1 if gap > 0 else 1

            # Time exit
            assert self._entry_time is not None
            minutes_held = (current_time - self._entry_time).total_seconds() / 60.0
            if minutes_held >= self.exit_minutes:
                signals.append(TradeSignal(
                    symbol=sym, direction=0, strength=0.0, size=0.0,
                    metadata={"action": "time_exit", "minutes_held": round(minutes_held, 1)},
                ))
                exit_symbols.append(sym)
                continue

            # Retracement target (50% of gap)
            if direction > 0:  # long (fading negative gap)
                target_px = entry_px + abs(gap / 100.0 * entry_px) * self.retracement_target
                if current_px >= target_px:
                    signals.append(TradeSignal(
                        symbol=sym, direction=0, strength=0.0, size=0.0,
                        metadata={"action": "take_profit", "reason": "retracement_target"},
                    ))
                    exit_symbols.append(sym)
                    continue
            else:  # short (fading positive gap)
                target_px = entry_px - abs(gap / 100.0 * entry_px) * self.retracement_target
                if current_px <= target_px:
                    signals.append(TradeSignal(
                        symbol=sym, direction=0, strength=0.0, size=0.0,
                        metadata={"action": "take_profit", "reason": "retracement_target"},
                    ))
                    exit_symbols.append(sym)
                    continue

            # Stop loss
            stop_dist = entry_px * self.stop_loss_pct / 100.0
            if direction > 0:
                if current_px <= entry_px - stop_dist:
                    signals.append(TradeSignal(
                        symbol=sym, direction=0, strength=0.0, size=0.0,
                        metadata={"action": "stop_loss", "reason": "stop_pct"},
                    ))
                    exit_symbols.append(sym)
            else:
                if current_px >= entry_px + stop_dist:
                    signals.append(TradeSignal(
                        symbol=sym, direction=0, strength=0.0, size=0.0,
                        metadata={"action": "stop_loss", "reason": "stop_pct"},
                    ))
                    exit_symbols.append(sym)

        # Clean up exited positions
        for sym in exit_symbols:
            self._entry_price.pop(sym, None)
            self._gap_pct.pop(sym, None)

        # --- Entry logic: only once per day, on first tick ---
        if self._traded_today:
            return signals

        # Check time window: entry only in first 5 minutes after open (09:30-09:35 ET)
        if current_time.hour > 9 or (current_time.hour == 9 and current_time.minute > 35):
            return signals

        self._traded_today = True
        self._entry_time = current_time

        # Volume data for ADV filter (use close * volume if available)
        volume = data.get("volume")

        open_row = open_prices[-1]
        close_row = close[-1]

        # Count existing positions
        n_existing = len(self._entry_price)

        candidates: list[tuple[str, float, float, int]] = []  # (sym, gap, open_px, idx)

        for i in range(min(n_assets, len(symbols))):
            sym = symbols[i]
            if sym in self._entry_price:
                continue

            prev_px = self._prev_close.get(sym)
            if prev_px is None or prev_px < self.min_price:
                continue

            today_open = float(open_row[i])
            if today_open < self.min_price:
                continue

            gap_pct = (today_open - prev_px) / prev_px * 100.0
            if abs(gap_pct) < self.gap_threshold_pct:
                continue

            # ADV filter: approximate from volume if available
            if volume is not None:
                vol_arr = volume[:, i] if volume.ndim == 2 else volume
                avg_vol = float(np.mean(vol_arr[-20:])) if len(vol_arr) >= 20 else float(np.mean(vol_arr))
                avg_price = float(np.mean(close[-20:, i])) if n_periods >= 20 else float(close_row[i])
                adv_dollars = avg_vol * avg_price
                if adv_dollars < self.min_adv_dollars:
                    continue

            candidates.append((sym, gap_pct, today_open, i))

        # Rank by absolute gap size (biggest gaps first)
        candidates.sort(key=lambda c: abs(c[1]), reverse=True)

        for sym, gap_pct, today_open, idx in candidates:
            if n_existing >= self.max_positions:
                break

            # Direction: fade the gap
            direction = -1 if gap_pct > 0 else 1
            strength = min(abs(gap_pct) / 5.0, 1.0)

            # Size: max_position_pct of portfolio
            max_notional = float(self.config.max_position_notional)
            size = self.max_position_pct * max_notional

            signals.append(TradeSignal(
                symbol=sym,
                direction=direction,
                strength=strength,
                size=size,
                metadata={
                    "action": "gap_fade_entry",
                    "gap_pct": round(gap_pct, 2),
                    "today_open": round(today_open, 2),
                    "prev_close": round(self._prev_close.get(sym, 0), 2),
                },
            ))

            self._entry_price[sym] = today_open
            self._gap_pct[sym] = gap_pct
            n_existing += 1

        return signals
