"""Opening gap fade: fade overnight gaps > 2% with time-based and profit exits.

Core hypothesis:
  Stocks that open with a significant overnight gap tend to retrace toward
  the prior close during the first 30 minutes of the regular trading session.
  The strategy shorts positive gaps and goes long on negative gaps, exiting
  by 10:00 AM ET or when a 0.5% profit target is hit.

Entry (first tick after 09:30 ET):
  gap  = (session_open - prev_close) / prev_close
  Select if abs(gap) > gap_threshold_pct (default 2.0%) and pre-market
  volume is below a multiple of the 20-day average.

Exit (checked every tick):
  1. Time: current_time >= entry_time + exit_minutes
  2. Profit target: gain from entry >= 0.5%
  3. Retracement: price has retraced retracement_target of the original gap
  4. Stop loss: 1.5% adverse move from entry price
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("opening_gap_fade")
class OpeningGapFadeStrategy(Strategy):
    """Fade opening gaps with time, profit-target, and retracement exits.

    Data dict expectations:
        close:           NDArray (n_periods, n_assets) or (n_periods,)
        open:            NDArray (n_periods, n_assets) or (n_periods,)
        prev_close:      dict[str, float] — {ticker: yesterday_close}
        current_time:    datetime — current ET wall-clock time
        symbols:         list[str] — ticker symbols
        session_open:    optional dict[str, float] — {ticker: session open price}
        volume:          optional NDArray — aligned volume bars
        ctx:             optional dict — tick-level context (e.g. pre-market vol)

    Config params:
        gap_threshold_pct:   min |gap%| to trade (default 2.0)
        exit_minutes:        minutes after open to force flatten (default 30)
        retracement_target:  fraction of gap to use as profit target (default 0.50)
        max_positions:       max concurrent positions (default 10)
        max_position_pct:    max % of equity per trade (default 0.05)
        stop_loss_pct:       stop loss distance from entry (default 1.5)
        pre_market_vol_mult: pre-market volume threshold multiple (default 1.5)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        self.gap_threshold_pct = float(config.params.get("gap_threshold_pct", 2.0))
        self.exit_minutes = int(config.params.get("exit_minutes", 30))
        self.retracement_target = float(config.params.get("retracement_target", 0.50))
        self.max_positions = int(config.params.get("max_positions", 10))
        self.max_position_pct = float(config.params.get("max_position_pct", 0.05))
        self.stop_loss_pct = float(config.params.get("stop_loss_pct", 1.5))
        self.pre_market_vol_mult = float(config.params.get("pre_market_vol_mult", 1.5))

        # Per-symbol session state
        self._entry_price: dict[str, float] = {}
        self._gap_pct: dict[str, float] = {}
        self._prev_close_px: dict[str, float] = {}
        self._entry_time: dict[str, datetime] = {}
        self._session_open_map: dict[str, float] = {}
        self._active_symbols: set[str] = set()
        self._current_date: datetime | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reset_session(self, date: datetime) -> None:
        """Reset per-session state at the start of a new trading day."""
        self._entry_price.clear()
        self._gap_pct.clear()
        self._prev_close_px.clear()
        self._entry_time.clear()
        self._session_open_map.clear()
        self._active_symbols.clear()
        self._current_date = date

    @staticmethod
    def _slippage_mult(current_time: datetime) -> float:
        """2× slippage during the first 5 minutes of the session (opening auction)."""
        if not isinstance(current_time, datetime):
            return 1.0
        # 09:30 – 09:35 ET
        if current_time.hour == 9 and current_time.minute < 35:
            return 2.0
        return 1.0

    def _passes_volume_filter(self, sym: str, idx: int, data: dict[str, Any], ctx: dict[Any, Any] | None) -> bool:  # noqa: C901,PLR0912
        """Return True if pre-market volume is below threshold multiple of avg."""
        pre_market_vol: float | None = None
        vol_20d_avg: float | None = None

        context = ctx if isinstance(ctx, dict) else {}

        # 1) Tick context (preferred)
        if context:
            pmv_val = context.get("pre_market_volume")
            avg_val = context.get("volume_20d_avg")

            if isinstance(pmv_val, dict):
                pre_market_vol = pmv_val.get(sym)  # type: ignore[assignment]
            elif pmv_val is not None:
                pre_market_vol = float(pmv_val)  # type: ignore[assignment]

            if isinstance(avg_val, dict):
                vol_20d_avg = avg_val.get(sym)  # type: ignore[assignment]
            elif avg_val is not None:
                vol_20d_avg = float(avg_val)  # type: ignore[assignment]

        # 2) Fallback to flat data keys
        if pre_market_vol is None:
            pmv = data.get("pre_market_volume")
            if pmv is not None:
                if hasattr(pmv, "shape") and pmv.ndim == 2:  # type: ignore[union-attr]
                    pre_market_vol = float(pmv[-1, idx])  # type: ignore[index]
                elif hasattr(pmv, "shape") and pmv.ndim == 1:  # type: ignore[union-attr]
                    pre_market_vol = float(pmv[-1])  # type: ignore[index]
                else:
                    pre_market_vol = float(pmv)  # type: ignore[arg-type]

        if vol_20d_avg is None:
            vma = data.get("volume_20d_avg")
            if vma is not None:
                if hasattr(vma, "shape") and vma.ndim == 2:  # type: ignore[union-attr]
                    vol_20d_avg = float(vma[-1, idx])  # type: ignore[index]
                elif hasattr(vma, "shape") and vma.ndim == 1:  # type: ignore[union-attr]
                    vol_20d_avg = float(vma[-1])  # type: ignore[index]
                else:
                    vol_20d_avg = float(vma)  # type: ignore[arg-type]

        # 3) Volume derived from bar data (last 20 avg)
        if vol_20d_avg is None:
            volume = data.get("volume")
            if volume is not None:
                if volume.ndim == 2 and volume.shape[0] >= 20:
                    vol_20d_avg = float(np.mean(volume[-20:, idx]))
                elif volume.ndim == 1 and volume.shape[0] >= 20:
                    vol_20d_avg = float(np.mean(volume[-20:]))

        # If we have both values, apply filter: pre_market_vol < mult * avg
        if pre_market_vol is not None and vol_20d_avg is not None and vol_20d_avg > 0:
            return float(pre_market_vol) < self.pre_market_vol_mult * float(vol_20d_avg)

        # Absent volume data → permissive pass
        return True

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:  # type: ignore[override]
        close = data.get("close")
        open_arr = data.get("open")
        session_open_raw = data.get("session_open")
        symbols = data.get("symbols", [])
        current_time = data.get("current_time")  # type: ignore[arg-type]
        prev_close_raw = data.get("prev_close")
        ctx = data.get("ctx") if isinstance(data.get("ctx"), dict) else {}

        if close is None or not symbols:
            return []

        if not isinstance(current_time, datetime):
            return []

        # Reset on new trading day
        if self._current_date is None or current_time.date() != self._current_date.date():
            self._reset_session(current_time)

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        if open_arr is not None and open_arr.ndim == 1:
            open_arr = open_arr.reshape(-1, 1)

        n_periods, n_assets = close.shape
        if n_periods < 1:
            return []

        # ---- Resolve session_open per symbol ----
        session_open: dict[str, float] = {}
        if isinstance(session_open_raw, dict):
            session_open = session_open_raw  # type: ignore[assignment]
        elif open_arr is not None:
            # Best-effort fallback: use the first open bar for the day
            for i in range(min(n_assets, len(symbols))):
                try:
                    session_open[symbols[i]] = float(open_arr[0, i])
                except Exception:
                    session_open[symbols[i]] = float(open_arr[-1, i])

        # ---- Resolve prev_close per symbol ----
        prev_close: dict[str, float] = {}
        if isinstance(prev_close_raw, dict):
            prev_close = prev_close_raw  # type: ignore[assignment]

        signals: list[TradeSignal] = []
        current_px_row = close[-1]

        # ------------------------------------------------------------------
        # 1) Exit logic: handle active positions first
        # ------------------------------------------------------------------
        exit_symbols: list[str] = []
        for sym in list(self._active_symbols):
            if sym not in self._entry_price:
                continue

            idx = -1
            for j, s in enumerate(symbols):
                if s == sym:
                    idx = j
                    break
            if idx == -1:
                continue

            entry_px = self._entry_price[sym]
            gap = self._gap_pct[sym]
            current_px = float(current_px_row[idx])

            # Direction: positive gap → short (-1), negative → long (+1)
            direction = -1 if gap > 0 else 1

            exit_needed = False
            exit_reason = ""

            # a) Time exit
            entry_time = self._entry_time.get(sym)
            if entry_time is not None:
                target_time = entry_time + timedelta(minutes=self.exit_minutes)
                if current_time >= target_time:
                    exit_needed = True
                    exit_reason = "time_exit"

            # b) Profit target: 0.5% profit from entry
            if not exit_needed:
                if direction > 0:  # long
                    profit_pct = (current_px - entry_px) / entry_px
                else:
                    profit_pct = (entry_px - current_px) / entry_px
                if profit_pct >= 0.005:
                    exit_needed = True
                    exit_reason = "profit_target"

            # c) Retracement target (configurable fraction of the gap)
            if not exit_needed:
                gap_size = abs(gap / 100.0 * entry_px)
                retrace_dist = gap_size * self.retracement_target
                if direction > 0:
                    if current_px >= entry_px + retrace_dist:
                        exit_needed = True
                        exit_reason = "retracement_target"
                else:
                    if current_px <= entry_px - retrace_dist:
                        exit_needed = True
                        exit_reason = "retracement_target"

            # d) Stop loss
            if not exit_needed:
                stop_dist = entry_px * self.stop_loss_pct / 100.0
                if direction > 0:
                    if current_px <= entry_px - stop_dist:
                        exit_needed = True
                        exit_reason = "stop_loss"
                else:
                    if current_px >= entry_px + stop_dist:
                        exit_needed = True
                        exit_reason = "stop_loss"

            if exit_needed:
                signals.append(TradeSignal(
                    symbol=sym,
                    direction=0,
                    strength=0.0,
                    size=0.0,
                    metadata={
                        "action": "gap_fade_exit",
                        "gap_pct": round(float(gap), 2),
                        "session_open": round(float(self._session_open_map.get(sym, 0.0)), 2),
                        "prev_close": round(float(self._prev_close_px.get(sym, 0.0)), 2),
                        "signal_reason": exit_reason,
                    },
                ))
                exit_symbols.append(sym)

        # Clean up exited positions
        for sym in exit_symbols:
            self._active_symbols.discard(sym)
            self._entry_price.pop(sym, None)
            self._gap_pct.pop(sym, None)
            self._entry_time.pop(sym, None)
            self._session_open_map.pop(sym, None)
            self._prev_close_px.pop(sym, None)

        # ------------------------------------------------------------------
        # 2) Entry logic
        # ------------------------------------------------------------------
        # Only enter during first minutes of session (09:30 – 09:35 ET)
        if current_time.hour > 9 or (current_time.hour == 9 and current_time.minute > 35):
            return signals
        if len(self._active_symbols) >= self.max_positions:
            return signals

        for i in range(min(n_assets, len(symbols))):
            sym = symbols[i]
            if sym in self._active_symbols:
                continue

            s_open = session_open.get(sym)
            prev_px = prev_close.get(sym)
            if s_open is None or prev_px is None or s_open <= 0 or prev_px <= 0:
                continue

            gap_pct = (s_open - prev_px) / prev_px * 100.0
            if abs(gap_pct) < self.gap_threshold_pct:
                continue

            # Volume filter
            if not self._passes_volume_filter(sym, i, data, ctx):  # type: ignore[arg-type]
                continue

            # Direction: fade the gap
            direction = -1 if gap_pct > 0 else 1
            strength = min(abs(gap_pct) / 5.0, 1.0)

            # Size
            max_notional = float(self.config.max_position_notional)
            size = self.max_position_pct * max_notional

            # Stop loss
            stop_loss = s_open * (1 - direction * self.stop_loss_pct / 100.0)

            # Slippage during opening auction
            slippage = self._slippage_mult(current_time)

            signals.append(TradeSignal(
                symbol=sym,
                direction=direction,
                strength=strength,
                size=size,
                stop_loss=stop_loss,
                metadata={
                    "action": "gap_fade_entry",
                    "gap_pct": round(float(gap_pct), 2),
                    "session_open": round(float(s_open), 2),
                    "prev_close": round(float(prev_px), 2),
                    "signal_reason": "gap_above_threshold",
                    "slippage_mult": slippage,
                },
            ))

            # Update state
            self._active_symbols.add(sym)
            self._entry_price[sym] = s_open
            self._gap_pct[sym] = gap_pct
            self._entry_time[sym] = current_time
            self._session_open_map[sym] = s_open
            self._prev_close_px[sym] = prev_px

            if len(self._active_symbols) >= self.max_positions:
                break

        return signals

    # ── Intraday backtest engine hook ────────────────────────────────────────

    def on_bar_intraday(self, bar, ctx):
        """Intraday bar handler for IntradayBacktestEngine.

        Processes a single 5-minute OHLCV bar wrapped in its session-aware
        TickContext.  Entry is gated to the first RTH bar; exit is checked
        every bar.  This is the native path for the opening gap fade.
        """
        current_time = bar.timestamp

        # Reset on new trading day
        if self._current_date is None or current_time.date() != self._current_date.date():
            self._reset_session(current_time)

        symbol = bar.symbol
        current_px = float(bar.close)
        trade_signals: list[TradeSignal] = []

        # ── Exit logic for active positions ────────────────────────────────────
        if symbol in self._active_symbols:
            entry_px = self._entry_price[symbol]
            gap = self._gap_pct[symbol]
            direction = -1 if gap > 0 else 1
            exit_needed = False
            exit_reason = ""

            # a) Time exit
            entry_time = self._entry_time.get(symbol)
            if entry_time is not None:
                target_time = entry_time + timedelta(minutes=self.exit_minutes)
                if current_time >= target_time:
                    exit_needed = True
                    exit_reason = "time_exit"

            # b) Profit target: 0.5% profit from entry
            if not exit_needed:
                if direction > 0:  # long
                    profit_pct = (current_px - entry_px) / entry_px
                else:
                    profit_pct = (entry_px - current_px) / entry_px
                if profit_pct >= 0.005:
                    exit_needed = True
                    exit_reason = "profit_target"

            # c) Stop loss: 1.5% adverse move from entry
            if not exit_needed:
                stop_dist = entry_px * self.stop_loss_pct / 100.0
                if direction > 0:
                    if current_px <= entry_px - stop_dist:
                        exit_needed = True
                        exit_reason = "stop_loss"
                else:
                    if current_px >= entry_px + stop_dist:
                        exit_needed = True
                        exit_reason = "stop_loss"

            if exit_needed:
                trade_signals.append(TradeSignal(
                    symbol=symbol,
                    direction=0,
                    strength=0.0,
                    size=0.0,
                    metadata={
                        "action": "gap_fade_exit",
                        "gap_pct": round(float(gap), 2),
                        "session_open": round(float(self._session_open_map.get(symbol, 0.0)), 2),
                        "prev_close": round(float(self._prev_close_px.get(symbol, 0.0)), 2),
                        "signal_reason": exit_reason,
                    },
                ))
                self._active_symbols.discard(symbol)
                self._entry_price.pop(symbol, None)
                self._gap_pct.pop(symbol, None)
                self._entry_time.pop(symbol, None)
                self._session_open_map.pop(symbol, None)
                self._prev_close_px.pop(symbol, None)

        # ── Entry logic: only in first 5 min of RTH ──────────────────────────
        can_enter = (
            symbol not in self._active_symbols
            and len(self._active_symbols) < self.max_positions
        )
        if can_enter:
            # Hard gate: only enter during the first 5 minutes of RTH (9:30-9:35)
            if current_time.hour == 9 and current_time.minute <= 35:
                # TickContext gap filter (checks is_first_bar, pre-mkt volume)
                if ctx.gap_filter_passed(self.gap_threshold_pct, avg_pre_mkt_vol=None):
                    s_open = float(ctx.session_open)
                    prev_px = float(ctx.prev_close)
                    gap_pct = ctx.overnight_gap_pct

                    if abs(gap_pct) >= self.gap_threshold_pct:
                        direction = -1 if gap_pct > 0 else 1
                        strength = min(abs(gap_pct) / 5.0, 1.0)
                        max_notional = float(self.config.max_position_notional)
                        size = self.max_position_pct * max_notional
                        stop_loss = s_open * (1.0 - direction * self.stop_loss_pct / 100.0)
                        slippage = self._slippage_mult(current_time)

                        trade_signals.append(TradeSignal(
                            symbol=symbol,
                            direction=direction,
                            strength=strength,
                            size=size,
                            stop_loss=stop_loss,
                            metadata={
                                "action": "gap_fade_entry",
                                "gap_pct": round(float(gap_pct), 2),
                                "session_open": round(float(s_open), 2),
                                "prev_close": round(float(prev_px), 2),
                                "signal_reason": "gap_above_threshold",
                                "slippage_mult": slippage,
                            },
                        ))

                        self._active_symbols.add(symbol)
                        self._entry_price[symbol] = s_open
                        self._gap_pct[symbol] = gap_pct
                        self._entry_time[symbol] = current_time
                        self._session_open_map[symbol] = s_open
                        self._prev_close_px[symbol] = prev_px

        return trade_signals
