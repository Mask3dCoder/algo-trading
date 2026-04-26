"""Bar aggregator — accumulates ticks into OHLCV bars with VWAP/TWAP."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from decimal import Decimal

from quantcore.core.types import OHLCV, Tick


class BarAggregator:
    """Accumulate ticks into OHLCV bars at configurable intervals.

    Supports 1s, 1min, 5min, etc. bars with running VWAP and TWAP.
    """

    def __init__(self, interval_seconds: int = 60) -> None:
        self.interval = interval_seconds
        self._bars: dict[str, dict] = defaultdict(self._new_bar)
        self._current_bar_start: dict[str, datetime] = {}

    def _new_bar(self) -> dict:
        return {
            "open": None, "high": Decimal("-inf"),
            "low": Decimal("inf"), "close": Decimal("0"),
            "volume": 0, "vwap_numerator": 0.0, "vwap_denominator": 0,
            "twap_numerator": 0.0, "twap_count": 0,
            "tick_count": 0,
        }

    def _bar_key(self, timestamp: datetime) -> datetime:
        """Compute the start of the bar period for a given timestamp."""
        ts = timestamp.timestamp()
        interval_ts = int(ts // self.interval) * self.interval
        return datetime.fromtimestamp(interval_ts)

    def add_tick(self, tick: Tick) -> OHLCV | None:
        """Add a tick. Returns a completed OHLCV bar if the bar just closed, else None."""
        # Convert tick timestamp (microseconds) to datetime
        ts = datetime.utcfromtimestamp(tick.timestamp_us / 1e6)
        bar_start = self._bar_key(ts)
        symbol = tick.symbol

        completed_bar = None

        # Check if we crossed a bar boundary
        if symbol in self._current_bar_start:
            if bar_start != self._current_bar_start[symbol]:
                # Close the current bar
                completed_bar = self._finalize_bar(symbol, self._current_bar_start[symbol])
                # Start a new bar
                self._bars[symbol] = self._new_bar()

        self._current_bar_start[symbol] = bar_start
        bar = self._bars[symbol]

        price = float(tick.price)
        size = tick.size

        if bar["open"] is None:
            bar["open"] = tick.price
        bar["high"] = max(bar["high"], tick.price)
        bar["low"] = min(bar["low"], tick.price)
        bar["close"] = tick.price
        bar["volume"] += size
        bar["vwap_numerator"] += price * size
        bar["vwap_denominator"] += size
        bar["twap_numerator"] += price
        bar["twap_count"] += 1
        bar["tick_count"] += 1

        return completed_bar

    def _finalize_bar(self, symbol: str, bar_start: datetime) -> OHLCV:
        """Finalize and return the current bar as an OHLCV object."""
        bar = self._bars[symbol]
        vwap = Decimal(str(bar["vwap_numerator"] / bar["vwap_denominator"])) if bar["vwap_denominator"] > 0 else Decimal("0")
        twap = Decimal(str(bar["twap_numerator"] / bar["twap_count"])) if bar["twap_count"] > 0 else Decimal("0")

        return OHLCV(
            symbol=symbol,
            timestamp=bar_start,
            open=bar["open"] or Decimal("0"),
            high=bar["high"] if bar["high"] != Decimal("-inf") else Decimal("0"),
            low=bar["low"] if bar["low"] != Decimal("inf") else Decimal("0"),
            close=bar["close"],
            volume=bar["volume"],
        )

    def flush(self) -> list[OHLCV]:
        """Force-finalize all current bars. Useful for shutdown."""
        bars = []
        for symbol in list(self._current_bar_start.keys()):
            bar = self._finalize_bar(symbol, self._current_bar_start[symbol])
            bars.append(bar)
        self._bars.clear()
        self._current_bar_start.clear()
        return bars

    @property
    def running_vwap(self) -> dict[str, float]:
        """Return running VWAP for each symbol's current bar."""
        result = {}
        for symbol, bar in self._bars.items():
            if bar["vwap_denominator"] > 0:
                result[symbol] = bar["vwap_numerator"] / bar["vwap_denominator"]
        return result

    @property
    def running_twap(self) -> dict[str, float]:
        """Return running TWAP for each symbol's current bar."""
        result = {}
        for symbol, bar in self._bars.items():
            if bar["twap_count"] > 0:
                result[symbol] = bar["twap_numerator"] / bar["twap_count"]
        return result
