# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
"""equity_factor_neutral.py — Market-neutral multi-factor sleeve.

Integrates with the event-driven BacktestEngine via the `on_snapshot` protocol.
Strict point-in-time discipline:

  * Factors use ONLY closes accumulated up to and including the current snapshot.
  * Trading is suppressed until `oos_start` (the OOS window) — the pre-OOS period
    is pure warm-up history, not a tradeable in-sample.
  * Rebalance every `rebalance_days` trading days: exit ALL open positions, then
    enter the new long/short selection (no position carry-over, no look-ahead).
  * Positions are equal-notional, gross ~= `gross_target` (long + short), net ~0.

The real production panel has no sector/market_cap, so neutralisation is
cross-sectional only (see quantcore/signals/equity_factors.py).
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from quantcore.backtest.engine import SignalData
from quantcore.signals.equity_factors import (
    build_signal,
    compute_factors_vectorized,
)

_log = logging.getLogger(__name__)

# 252 trading days ~ 12 months; +21 skip; +60 beta window; +buffer.
_MIN_BARS = 252 + 21 + 60 + 5


class EquityFactorNeutral:
    """Multi-factor market-neutral strategy (engine-compatible `on_snapshot`)."""

    def __init__(
        self,
        name: str = "equity_factor_neutral",
        *,
        oos_start: date | None = None,
        rebalance_days: int = 5,
        long_frac: float = 0.10,
        short_frac: float = 0.10,
        capital: float = 1_000_000.0,
        gross_target: float = 1.0,
        max_notional_per_name_pct: float = 0.02,
        lookback: int = 252,
        skip: int = 21,
        window: int = 60,
    ) -> None:
        self.name = name
        self.oos_start = oos_start
        self.rebalance_days = max(1, int(rebalance_days))
        self.long_frac = long_frac
        self.short_frac = short_frac
        self.capital = float(capital)
        self.gross_target = float(gross_target)
        self.max_notional_per_name = self.capital * max_notional_per_name_pct
        self.lookback = lookback
        self.skip = skip
        self.window = window
        self._trade_log: list[dict] = []
        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        """Clear all state (call between walk-forward folds)."""
        self._panel: list[tuple[pd.Timestamp, dict[str, float]]] = []
        self._bar_count = 0
        self._held: set[str] = set()
        self._trade_log = []

    # -- engine interface ----------------------------------------------------
    def on_snapshot(self, snapshot, portfolio) -> list[SignalData]:
        ts = snapshot.timestamp
        # Accumulate closes for EVERY symbol present at this (past) timestamp.
        row: dict[str, float] = {}
        for sym, bar in snapshot.bars.items():
            try:
                row[sym] = float(bar.close)
            except (TypeError, ValueError):
                continue
        if not row:
            return []
        self._panel.append((pd.Timestamp(ts), row))
        self._bar_count += 1

        # Warm-up not reached, or before OOS window -> no trading (history only).
        in_sample = self.oos_start is not None and ts.date() < self.oos_start
        if self._bar_count < _MIN_BARS or in_sample:
            return []
        if self._bar_count % self.rebalance_days != 0:
            return []

        return self._rebalance(ts)

    def on_fill(self, fill, portfolio) -> None:
        """Record executed entries/exits for the trade log (OOS evidence)."""
        try:
            sym = self._parse_symbol(fill.order_id)
            side = "buy" if str(fill.side).lower().startswith("buy") else "sell"
            self._trade_log.append({
                "date": pd.Timestamp(getattr(fill, "timestamp", None) or datetime.now()).date().isoformat(),
                "symbol": sym,
                "side": side,
                "price": float(fill.price),
                "quantity": float(fill.quantity),
            })
        except Exception:
            pass

    # -- core ----------------------------------------------------------------
    def _rebalance(self, ts) -> list[SignalData]:
        # Need at least _MIN_BARS of accumulated history before any factor is
        # well-defined. Guard on the FULL panel length, not the sliced tail.
        if len(self._panel) < _MIN_BARS:
            return []
        # Trailing window must cover lookback + skip + window (>= _MIN_BARS).
        n_need = self.lookback + self.skip + self.window + 5
        tail = self._panel[-n_need:]
        if len(tail) < _MIN_BARS:
            return []

        idx = [d for d, _ in tail]
        syms = sorted({s for _, r in tail for s in r})
        data = {s: [r.get(s, np.nan) for _, r in tail] for s in syms}
        wide = pd.DataFrame(data, index=pd.DatetimeIndex(idx)).sort_index()

        factors = compute_factors_vectorized(
            wide, lookback=self.lookback, skip=self.skip, window=self.window
        )
        targets = build_signal(
            factors,
            long_frac=self.long_frac,
            short_frac=self.short_frac,
        )
        new_long = set(targets["long"])
        new_short = set(targets["short"])
        if not new_long and not new_short:
            # Nothing tradable -> just hold flat on what we have.
            return []

        # Equal-notional sizing: gross = gross_target * capital, split long/short.
        k = max(len(new_long), len(new_short), 1)
        notional_per_name = self.capital * 0.5 * self.gross_target / k
        notional_per_name = min(notional_per_name, self.max_notional_per_name)

        signals: list[SignalData] = []
        # 1) Exit everything currently held (clean rebalance).
        for sym in sorted(self._held):
            signals.append(SignalData(
                symbol=sym, direction="flat", strength=0.0,
                strategy_id=self.name, instrument_type="equity",
            ))
        # 2) Enter new longs / shorts.
        for sym in sorted(new_long):
            signals.append(SignalData(
                symbol=sym, direction="long", strength=1.0,
                strategy_id=self.name, notional_size=notional_per_name,
                instrument_type="equity",
            ))
        for sym in sorted(new_short):
            signals.append(SignalData(
                symbol=sym, direction="short", strength=1.0,
                strategy_id=self.name, notional_size=notional_per_name,
                instrument_type="equity",
            ))

        self._held = new_long | new_short
        return signals

    @staticmethod
    def _parse_symbol(order_id: str) -> str:
        base = order_id.rsplit("_", 2)[0]
        parts = base.split("_", 2)
        return parts[2] if len(parts) > 2 else base

    # -- reporting -----------------------------------------------------------
    def export_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_log, columns=["date", "symbol", "side", "price", "quantity"])
