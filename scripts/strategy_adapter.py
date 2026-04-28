"""Bridge between Strategy ABC (on_data) and BacktestEngine Protocol (on_bar).

The engine drives on_bar(bar, portfolio) -> list[SignalData] row-by-row.
The base Strategy ABC expects on_data(data_dict) -> list[TradeSignal] with full arrays.

Supports both single-symbol (1D arrays) and multi-symbol (2D arrays) strategies.
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

from quantcore.backtest.engine import SignalData
from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal


class StrategyAdapter:
    """Wraps a Strategy ABC so the engine can drive it row-by-row.

    For multi-symbol strategies: buffers bars within each timestamp,
    builds 2D arrays (bars x symbols), and calls on_data when complete.

    Usage:
        adapter = StrategyAdapter(MomentumStrategy, cfg, n_symbols=6)
        engine = BacktestEngine(strategy=adapter, ...)
    """

    def __init__(self, strategy_cls: type[Strategy], config: StrategyConfig,
                 n_symbols: int = 1) -> None:
        self._strat = strategy_cls(config)
        self._strat.initialize()
        self._n_symbols = n_symbols

        # Row-by-row accumulation (per-bar, per-symbol)
        self._data_bars: list[OrderedDict] = []
        self._current_ts = None
        self._current_bar_data: OrderedDict[str, dict] = OrderedDict()
        self._symbols: list[str] = []

        # Cached signals from last on_data call
        self._last_trade_signals: list[TradeSignal] = []
        self._bar_idx = 0

    def on_bar(self, bar, portfolio) -> list[SignalData]:
        sym = str(getattr(bar, 'symbol', 'TEST'))
        ts = getattr(bar, 'timestamp', None)

        # Detect timestamp boundary
        if self._current_ts is not None and ts != self._current_ts:
            # Flush the previous timestamp to data
            self._finalize_timestamp()

        self._current_ts = ts

        # Accumulate this bar's values
        self._current_bar_data[sym] = {
            "close": float(bar.close),
            "high": float(getattr(bar, 'high', float(bar.close))),
            "low": float(getattr(bar, 'low', float(bar.close))),
            "open": float(getattr(bar, 'open', float(bar.close))),
            "volume": float(getattr(bar, 'volume', 0)),
            "implied_vol": float(getattr(bar, 'implied_vol', 0.25)),
            "yields": float(getattr(bar, 'yields', 0.03)),
            "earnings_upcoming": float(getattr(bar, 'earnings_upcoming', 0.0)),
        }
        if sym not in self._symbols:
            self._symbols.append(sym)

        self._bar_idx += 1

        # Only produce signals when we have enough bars
        if len(self._data_bars) < 2:
            return []

        # Build data dict and call strategy
        try:
            data = self._build_data_dict()
            self._last_trade_signals = self._strat.on_data(data) or []
        except Exception:
            self._last_trade_signals = []

        # Convert to SignalData, filter for current symbol
        result = []
        for tsig in self._last_trade_signals:
            if tsig.symbol != sym and self._n_symbols > 1:
                continue  # multi-symbol: only emit signals for current bar's symbol
            result.append(self._convert(tsig))

        return result

    def on_fill(self, fill, portfolio) -> None:
        qty = float(fill.quantity)
        side = "buy" if fill.order_id.endswith("_buy") else "sell"
        try:
            self._strat.on_fill(fill.symbol, float(fill.price), qty, side)
        except Exception:
            pass

    def _finalize_timestamp(self) -> None:
        """Store the completed timestamp's data and prepare next."""
        if self._current_bar_data:
            self._data_bars.append(self._current_bar_data)
        self._current_bar_data = OrderedDict()

    def _build_data_dict(self) -> dict:
        """Build the data dict from accumulated bar data.

        For n_symbols=1: returns 1D arrays (n_bars,)
        For n_symbols>1: returns 2D arrays (n_bars, n_symbols)
        """
        # Include all bars up to now
        bars = self._data_bars

        if self._n_symbols == 1:
            return self._build_1d(bars)
        else:
            return self._build_2d(bars)

    def _build_1d(self, bars: list) -> dict:
        """Build 1D data dict (single-symbol)."""
        n = len(bars)
        if n == 0:
            return {"close": np.array([], dtype=float), "symbols": []}

        close = np.array([list(b.values())[0]["close"] for b in bars], dtype=float)
        high = np.array([list(b.values())[0]["high"] for b in bars], dtype=float)
        low = np.array([list(b.values())[0]["low"] for b in bars], dtype=float)
        open_ = np.array([list(b.values())[0]["open"] for b in bars], dtype=float)

        data: dict = {
            "close": close,
            "high": high,
            "low": low,
            "open": open_,
            "symbols": list(self._symbols),
        }
        # Optional fields
        vols = [list(b.values())[0]["volume"] for b in bars]
        if any(v > 0 for v in vols):
            data["volume"] = np.array(vols, dtype=float)
        ivs = [list(b.values())[0]["implied_vol"] for b in bars]
        if any(v != 0.25 for v in ivs):
            data["implied_vol"] = np.array(ivs, dtype=float)
        return data

    def _build_2d(self, bars: list) -> dict:
        """Build 2D data dict (multi-symbol)."""
        n_bars = len(bars)
        symbols = self._symbols
        n_sym = len(symbols)

        if n_bars == 0:
            return {"close": np.array([[]], dtype=float), "symbols": symbols}

        sym_idx = {s: i for i, s in enumerate(symbols)}

        def _fill(field: str, default: float = 0.0) -> np.ndarray:
            arr = np.full((n_bars, max(n_sym, 1)), default, dtype=float)
            for bi, bar_data in enumerate(bars):
                for sym, vals in bar_data.items():
                    si = sym_idx.get(sym, 0)
                    arr[bi, si] = vals.get(field, default)
            return arr

        data: dict = {
            "close": _fill("close"),
            "high": _fill("high"),
            "low": _fill("low"),
            "open": _fill("open"),
            "volume": _fill("volume"),
            "implied_vol": _fill("implied_vol", 0.25),
            "symbols": symbols,
        }
        return data

    def _convert(self, ts: TradeSignal) -> SignalData:
        dir_map = {1: "long", -1: "short", 0: "flat"}
        return SignalData(
            symbol=ts.symbol,
            direction=dir_map.get(ts.direction, "flat"),
            strength=ts.strength,
            strategy_id=self._strat.name,
        )

    def get_strategy(self) -> Strategy:
        return self._strat
