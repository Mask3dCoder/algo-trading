"""Framework strategy adapter — bridges quantcore.strategies.base.Strategy
to the backtest engine's Strategy Protocol and to the paper trader's
generate_signals interface."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantcore.backtest.engine import (
    OHLCV,
    Fill,
    PortfolioState,
    SignalData,
)
from quantcore.strategies.base import Strategy as FrameworkStrategy
from quantcore.strategies.base import TradeSignal

_DIRECTION_MAP: dict[int, str] = {1: "long", -1: "short", 0: "flat"}


class FrameworkStrategyAdapter:
    """Adapts a quantcore.strategies.base.Strategy to the backtest Protocol.

    Accumulates per-symbol OHLCV bars into numpy arrays, calls the framework
    strategy's ``on_data`` with aligned 2D arrays, and converts emitted
    ``TradeSignal`` objects into ``SignalData`` for the engine.

    Two engine entry points are provided:

    - ``on_snapshot(snapshot, portfolio)`` — the PREFERRED cross-sectional
      path. The engine calls it once per timestamp with every universe
      symbol's bar. Without it the engine falls back to the legacy
      ``on_bar(representative_bar)`` path, which delivers only ONE symbol's
      bar per timestamp (``MarketDataSnapshot.representative_bar`` returns
      ``symbols[0]``), so multi-symbol strategies never see the rest of the
      universe and the adapter's alignment check blocks ``on_data`` forever.
    - ``on_bar(bar, portfolio)`` — legacy single-symbol compatibility, kept
      for callers that drive the adapter bar-by-bar.

    Parameters
    ----------
    strategy:
        An initialized framework strategy instance (or one that will be
        initialized on first bar).
    symbols:
        The symbols the adapter should buffer and pass to the strategy.
    """

    def __init__(self, strategy: FrameworkStrategy, symbols: list[str]) -> None:
        self._strategy = strategy
        self._symbols = symbols
        self._strategy.initialize()
        self._buffers: dict[str, dict[str, list[float]]] = {
            sym: defaultdict(list) for sym in symbols
        }
        self._last_bar_time: datetime | None = None
        # Preallocated (capacity, n_assets) assembly cache. Filled
        # incrementally as bars arrive; each on_data call receives fresh
        # array copies sliced from this cache. This replaces the former
        # per-call column_stack over the full history, which was O(history)
        # per bar (O(n^2) per pass).
        self._field_arrays: dict[str, NDArray[np.floating]] | None = None
        self._field_arrays_capacity = 0
        self._field_arrays_filled = 0

    # ------------------------------------------------------------------
    # Strategy Protocol
    # ------------------------------------------------------------------

    def on_snapshot(self, snapshot: Any, portfolio: PortfolioState) -> list[SignalData]:
        """Engine-native path: consume the full cross-sectional slice.

        Appends every universe symbol's bar from the immutable snapshot. A
        universe symbol with no bar in this snapshot is causally
        forward-filled from its last seen bar (open=high=low=close=prev_close,
        volume=0) so the per-symbol buffers stay aligned and ``on_data``
        keeps receiving rectangular arrays across suspensions/missing days.
        A symbol that has never traded yet is NOT filled (nothing to fill
        from) and keeps the adapter silent until it first appears.
        """
        saw_any = False
        for sym in self._symbols:
            bar = snapshot.get_bar(sym)
            if bar is not None:
                self._append_bar(sym, bar)
                saw_any = True
            else:
                self._ffill_symbol(sym)
        if saw_any:
            self._last_bar_time = snapshot.timestamp
        return self._aligned_on_data(portfolio)

    def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list[SignalData]:
        sym = bar.symbol
        if sym not in self._buffers:
            return []

        # Accumulate
        self._append_bar(sym, bar)
        self._last_bar_time = bar.timestamp

        return self._aligned_on_data(portfolio)

    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        # order_id formats (FillSimulator appends _buy or _sell):
        #   ord_{timestamp}_{symbol}_{action}_{side}
        #   ord_eop_{symbol}_{action1}_{action2}_{side}
        # In all formats, parts[2] is the symbol and parts[-1] is the side.
        parts = fill.order_id.split("_")
        symbol = parts[2]  # always the symbol position
        side = "buy" if parts[-1] == "buy" else "sell"
        self._strategy.on_fill(symbol, float(fill.price), float(fill.quantity), side)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append_bar(self, sym: str, bar: OHLCV) -> None:
        self._buffers[sym]["open"].append(float(bar.open))
        self._buffers[sym]["high"].append(float(bar.high))
        self._buffers[sym]["low"].append(float(bar.low))
        self._buffers[sym]["close"].append(float(bar.close))
        self._buffers[sym]["volume"].append(float(bar.volume))

    def _ffill_symbol(self, sym: str) -> None:
        """Causally continue ``sym``'s series from its last seen bar."""
        closes = self._buffers[sym]["close"]
        if not closes:
            return  # never traded yet — nothing to fill from
        pc = float(closes[-1])
        self._buffers[sym]["open"].append(pc)
        self._buffers[sym]["high"].append(pc)
        self._buffers[sym]["low"].append(pc)
        self._buffers[sym]["close"].append(pc)
        self._buffers[sym]["volume"].append(0.0)

    def _aligned_on_data(self, portfolio: PortfolioState) -> list[SignalData]:
        """Call ``on_data`` once all symbol buffers are aligned (once per period).

        This prevents strategies with internal counters from seeing N calls
        per period (one per symbol) in multi-asset backtests.
        """
        lengths = [len(self._buffers[s]["close"]) for s in self._symbols]
        if min(lengths) != max(lengths):
            return []

        data = self._assemble_data()
        if data is None:
            return []

        # Inject current portfolio equity so strategies can size dynamically
        data["equity"] = portfolio.total_value

        trade_signals = self._strategy.on_data(data)
        return [self._to_signal_data(s) for s in trade_signals]

    def _assemble_data(self) -> dict[str, NDArray] | None:
        """Build 2D arrays (n_periods, n_assets) aligned to min length.

        Arrays are assembled from a preallocated cache that is filled
        incrementally (only newly arrived bars are copied in), then handed
        to the strategy as fresh array copies — the same no-aliasing
        contract the former per-call column_stack provided, without its
        O(history) cost per bar.
        """
        lengths = [len(self._buffers[s]["close"]) for s in self._symbols]
        min_len = min(lengths)
        if min_len < 2:
            return None

        n_assets = len(self._symbols)
        fields = ("open", "high", "low", "close", "volume")

        if (
            self._field_arrays is None
            or self._field_arrays_capacity < min_len
        ):
            new_capacity = max(64, 2 * (min_len + 16))
            self._field_arrays = {
                f: np.empty((new_capacity, n_assets), dtype=float) for f in fields
            }
            self._field_arrays_capacity = new_capacity
            self._field_arrays_filled = 0

        if min_len > self._field_arrays_filled:
            # Copy only the rows that arrived since the last assembly.
            for j, sym in enumerate(self._symbols):
                buf = self._buffers[sym]
                for f in fields:
                    self._field_arrays[f][self._field_arrays_filled:min_len, j] = (
                        buf[f][self._field_arrays_filled:min_len]
                    )
            self._field_arrays_filled = min_len

        data: dict[str, NDArray | dict | datetime] = {}
        for f in fields:
            data[f] = self._field_arrays[f][:min_len].copy()

        data["symbols"] = list(self._symbols)

        # Inject prev_close (previous day's close) for gap-fade backtesting
        prev_close: dict[str, float] = {}
        for s in self._symbols:
            closes = self._buffers[s]["close"]
            if len(closes) >= 2:
                prev_close[s] = float(closes[-2])
        data["prev_close"] = prev_close

        # Inject current_time for time-based exit logic
        if self._last_bar_time is not None:
            data["current_time"] = self._last_bar_time

        return data

    def _to_signal_data(self, ts: TradeSignal) -> SignalData:
        direction = _DIRECTION_MAP.get(ts.direction, "flat")
        # Coerce numpy scalars to Python types
        symbol = str(ts.symbol) if not isinstance(ts.symbol, str) else ts.symbol
        notional = float(ts.size) if ts.size and float(ts.size) > 0 else None
        return SignalData(
            symbol=symbol,
            direction=direction,
            strength=float(ts.strength),
            strategy_id=self._strategy.name,
            notional_size=notional,
            metadata=dict(ts.metadata),
            instrument_type=ts.instrument_type,
        )


# ---------------------------------------------------------------------------
# Live Strategy Adapter — bridges framework Strategy to paper trader
# ---------------------------------------------------------------------------


class LiveStrategyAdapter:
    """Adapts a framework ``Strategy`` to the paper trader's
    ``generate_signals(feed) -> list[dict]`` interface.

    Accumulates per-symbol OHLCV bars from the feed, builds aligned 2D numpy
    arrays, calls ``strategy.on_data()``, and converts ``TradeSignal``
    objects into the dict format the paper trader expects.

    Parameters
    ----------
    strategy:
        An initialized framework strategy instance.
    symbols:
        The symbols the adapter should buffer and pass to the strategy.
    resample_minutes:
        If set (e.g., 5), aggregates raw 1-min bars into higher-timeframe
        bars before passing them to the strategy. Reduces noise and aligns
        signal frequency with the strategy's intended timeframe.
    """

    def __init__(
        self,
        strategy: FrameworkStrategy,
        symbols: list[str],
        resample_minutes: int = 1,
    ) -> None:
        self._strategy = strategy
        self._symbols = symbols
        self._resample = max(1, resample_minutes)
        self._strategy.initialize()
        self._buffers: dict[str, dict[str, list[float]]] = {
            sym: {
                "open": [], "high": [], "low": [], "close": [],
                "volume": [], "timestamp": [],
            }
            for sym in symbols
        }
        # Per-symbol raw bar accumulator for resampling
        self._raw: dict[str, dict[str, list[float]]] = {
            sym: {
                "open": [], "high": [], "low": [], "close": [],
                "volume": [], "timestamp": [],
            }
            for sym in symbols
        }
        self._current_direction: dict[str, int] = {}
        self._warmed = False

    # ------------------------------------------------------------------
    # Paper trader interface
    # ------------------------------------------------------------------

    def generate_signals(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Called by the paper trader's main loop.

        Accepts the latest bar events (already fetched by the caller) and
        returns signals in the format the paper trader expects::

            {"symbol": "SPY", "direction": "long", "price": 530.0,
             "strength": 0.8, "notional_size": 5000.0,
             "stop_loss": 520.0, "take_profit": 545.0}
        """
        if not events:
            return []

        for event in events:
            self._ingest_event(event)

        data = self._assemble_data()
        if data is None:
            return []

        trade_signals = self._strategy.on_data(data)
        return self._to_dict_signals(trade_signals)

    def warmup(self, historical_bars: list[dict[str, Any]]) -> None:
        """Preload historical bars to warm up the strategy.

        Feed bars in chronological order. After warmup the strategy can
        generate signals from the first live bar.

        Parameters
        ----------
        historical_bars:
            List of bar dicts with at least ``symbol`` and ``close`` fields.
        """
        for bar in historical_bars:
            self._ingest_event(bar)
        self._warmed = True

    def on_fill(self, fill: Any) -> None:
        """Forward fill events to the framework strategy.

        Handles IBKR Fill objects, PaperBroker fill dicts, and generic
        objects with symbol/price/quantity/side attributes.
        """
        symbol, price, qty, side = self._extract_fill_fields(fill)
        if symbol and price > 0 and qty != 0:
            self._strategy.on_fill(symbol, price, qty, side)

    @property
    def warmed(self) -> bool:
        return self._warmed

    @property
    def positions(self) -> dict[str, float]:
        return self._strategy.positions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ingest_event(self, event: dict[str, Any]) -> None:
        sym = event.get("symbol", "")
        if sym not in self._buffers:
            return

        ts = event.get("timestamp")
        price = float(event.get("price", event.get("close", 0)))
        if price <= 0:
            return

        if self._resample > 1:
            self._ingest_raw(sym, ts, price, event)
        else:
            self._ingest_direct(sym, ts, price, event)

    def _ingest_direct(
        self, sym: str, ts: Any, price: float, event: dict[str, Any],
    ) -> None:
        """Append a bar directly to main buffers (no resampling)."""
        ts_list = self._buffers[sym]["timestamp"]
        if ts and ts_list and ts_list[-1] == ts:
            return

        self._buffers[sym]["open"].append(float(event.get("open", price)))
        self._buffers[sym]["high"].append(float(event.get("high", price)))
        self._buffers[sym]["low"].append(float(event.get("low", price)))
        self._buffers[sym]["close"].append(float(event.get("close", price)))
        self._buffers[sym]["volume"].append(float(event.get("volume", 0)))
        if ts:
            ts_list.append(ts)
        else:
            ts_list.append(datetime.now(UTC))

    def _ingest_raw(
        self, sym: str, ts: Any, price: float, event: dict[str, Any],
    ) -> None:
        """Accumulate raw 1-min bars; emit a resampled bar every N bars."""
        raw = self._raw[sym]
        raw_ts = raw["timestamp"]

        # Dedup on raw buffer
        if ts and raw_ts and raw_ts[-1] == ts:
            return

        raw["open"].append(float(event.get("open", price)))
        raw["high"].append(float(event.get("high", price)))
        raw["low"].append(float(event.get("low", price)))
        raw["close"].append(float(event.get("close", price)))
        raw["volume"].append(float(event.get("volume", 0)))
        if ts:
            raw_ts.append(ts)
        else:
            raw_ts.append(datetime.now(UTC))

        if len(raw["close"]) < self._resample:
            return

        # Build resampled OHLC bar
        r_open = raw["open"][0]
        r_high = max(raw["high"])
        r_low = min(raw["low"])
        r_close = raw["close"][-1]
        r_volume = sum(raw["volume"])
        r_ts = raw_ts[-1]

        # Dedup against main buffers (same resampled bar should not re-enter)
        main_ts = self._buffers[sym]["timestamp"]
        if main_ts and main_ts[-1] == r_ts:
            for k in raw:
                raw[k].clear()
            return

        self._buffers[sym]["open"].append(r_open)
        self._buffers[sym]["high"].append(r_high)
        self._buffers[sym]["low"].append(r_low)
        self._buffers[sym]["close"].append(r_close)
        self._buffers[sym]["volume"].append(r_volume)
        main_ts.append(r_ts)

        # Clear raw accumulator for next bar
        for k in raw:
            raw[k].clear()

    def _assemble_data(self) -> dict[str, Any] | None:
        """Build 2D arrays (n_periods, n_assets) aligned to min length."""
        lengths = [len(self._buffers[s]["close"]) for s in self._symbols]
        min_len = min(lengths)
        if min_len < 2:
            return None

        fields = ["open", "high", "low", "close", "volume"]
        data: dict[str, Any] = {}

        for field in fields:
            cols = []
            for s in self._symbols:
                cols.append(self._buffers[s][field][:min_len])
            data[field] = np.column_stack(cols)

        data["symbols"] = list(self._symbols)
        return data

    def _to_dict_signals(
        self, trade_signals: list[TradeSignal],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for ts in trade_signals:
            symbol = (
                str(ts.symbol) if not isinstance(ts.symbol, str)
                else ts.symbol
            )
            direction = ts.direction

            # Only emit when direction changes
            prev = self._current_direction.get(symbol)
            if prev == direction:
                continue
            self._current_direction[symbol] = direction

            # Latest price for this symbol
            price = (
                float(self._buffers[symbol]["close"][-1])
                if self._buffers[symbol]["close"] else 0.0
            )

            sig: dict[str, Any] = {
                "symbol": symbol,
                "direction": _DIRECTION_MAP.get(direction, "flat"),
                "price": price,
                "strength": float(ts.strength),
            }

            if ts.size and float(ts.size) > 0:
                sig["notional_size"] = float(ts.size)
            if ts.stop_loss and float(ts.stop_loss) > 0:
                sig["stop_loss"] = float(ts.stop_loss)
            if ts.take_profit and float(ts.take_profit) > 0:
                sig["take_profit"] = float(ts.take_profit)

            results.append(sig)

        return results

    @staticmethod
    def _extract_fill_fields(fill: Any) -> tuple[str, float, float, str]:
        """Extract symbol, price, qty, side from various fill formats."""
        # IBKR fill (ib_insync.objects.Fill)
        if hasattr(fill, "contract") and hasattr(fill.contract, "symbol"):
            symbol = fill.contract.symbol
            price = (
                float(fill.execution.price) if fill.execution else 0.0
            )
            qty = (
                float(fill.execution.shares) if fill.execution else 0.0
            )
            side = fill.execution.side if fill.execution else ""
            return symbol, price, qty, side

        # Dict format (PaperBroker)
        if isinstance(fill, dict):
            return (
                str(fill.get("symbol", "")),
                float(fill.get("price", 0)),
                float(fill.get("quantity", 0)),
                str(fill.get("side", "")),
            )

        # Generic object fallback
        symbol = str(getattr(fill, "symbol", ""))
        price = float(getattr(fill, "price", 0) or 0)
        qty = float(getattr(fill, "quantity", 0) or 0)
        side = str(getattr(fill, "side", ""))
        return symbol, price, qty, side
