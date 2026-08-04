# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment.
# Ensure Alpaca keys are in .env for live deployment.

"""equity_pairs.py - PIT-correct cointegrated pairs trading sleeve.

Integrates with the event-driven BacktestEngine via the `on_snapshot` protocol,
exactly like quantcore/strategies/equity/equity_factor_neutral.py. Strict
point-in-time discipline:

  * Prices are accumulated bar-by-bar from the first row of the fold (the
    gate loads a 395-day pre-OOS buffer). Trading is SUPPRESSED until `oos_start`;
    the pre-OOS rows are used ONLY to warm up rolling statistics, never to trade.
  * The hedge ratio, spread mean and spread std are recomputed from a rolling
    60-bar window of ACCUMULATED history at each rebalance. Nothing is read from
    data/pairs_found.csv except the *list* of pairs to trade (the cointegration
    relationship). This kills the look-ahead leakage that a full-sample hedge
    ratio would introduce.
  * Rebalance every `rebalance_days` bars: emit long/short/flat SignalData per
    leg; the engine queues fills for the next bar's open (no intra-bar leakage).

Data policy:
  * Pair list is read from data/pairs_found.csv. The `MA,DOW` pair is dropped
    because DOW is absent from the real panel.
  * DISJOINT PAIRS ONLY. The BacktestEngine holds ONE net position per symbol,
    and its `flat` signal closes the ENTIRE net position in that symbol. If two
    pairs shared a symbol (e.g. SHW in both (ROK,SHW) and (UNH,SHW)), one pair's
    exit would forcibly flatten the other pair's leg and corrupt both P&L and the
    trade count. We therefore greedily select a maximal set of pairs with no
    shared symbols. This is the correct, leak-free implementation on this engine.
  * Uses ONLY [symbol, timestamp, OHLCV] - no fabricated fundamentals.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from quantcore.backtest.engine import SignalData

_log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _select_disjoint(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Greedy maximal set of pairs with no symbol shared between any two pairs."""
    chosen: list[tuple[str, str]] = []
    used: set[str] = set()
    for a, b in pairs:
        if a in used or b in used:
            continue
        chosen.append((a, b))
        used.add(a)
        used.add(b)
    return chosen


class EquityPairs:
    """Cointegrated pairs trading with rolling-window OLS hedge ratio (engine-compatible)."""

    def __init__(
        self,
        name: str = "equity_pairs",
        *,
        oos_start: date | None = None,
        rebalance_days: int = 5,
        capital: float = 1_000_000.0,
        gross_target: float = 1.0,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        warmup_bars: int = 60,
        pair_list_path: str = "data/pairs_found.csv",
    ) -> None:
        self.name = name
        self.oos_start = pd.Timestamp(oos_start).date() if oos_start is not None else None
        self.rebalance_days = max(1, int(rebalance_days))
        self.capital = float(capital)
        self.gross_target = float(gross_target)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.warmup_bars = int(warmup_bars)

        # --- Load pair list (selection only; stats are recomputed PIT) ----------
        path = Path(pair_list_path)
        if not path.is_absolute():
            path = path if path.exists() else REPO_ROOT / pair_list_path
        if not path.exists():
            raise RuntimeError(f"Pair list missing: {path}. Refusing to fabricate inputs.")
        raw = pd.read_csv(path)
        # Header is stock_a, stock_b, ... (NOT symbol_a/symbol_b).
        all_pairs: list[tuple[str, str]] = []
        for _, row in raw.iterrows():
            a, b = str(row["stock_a"]).strip(), str(row["stock_b"]).strip()
            if a == "DOW" or b == "DOW":
                _log.info("Dropping pair %s,%s (DOW absent from panel).", a, b)
                continue
            all_pairs.append((a, b))

        # Disjoint selection so no symbol is shared across pairs (engine contract).
        self.pairs = _select_disjoint(all_pairs)
        if not self.pairs:
            raise ValueError("No tradable disjoint pairs after dropping DOW.")
        self.n_pairs = len(self.pairs)
        _log.info(
            "equity_pairs: %d raw pairs -> %d disjoint tradable pairs: %s",
            len(all_pairs), self.n_pairs, self.pairs,
        )

        # Equal gross notional spread across pairs; half per leg (dollar-neutral).
        self.notional_per_pair = (self.capital * self.gross_target) / self.n_pairs
        self.target_notional_per_leg = self.notional_per_pair / 2.0

        self.reset()

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> None:
        # Per-pair rolling price history: list of (date, price_a, price_b).
        self._hist: dict[tuple[str, str], list[tuple[pd.Timestamp, float, float]]] = {
            p: [] for p in self.pairs
        }
        # Tracked position per pair: (qty_a, qty_b) in shares.
        self._pos: dict[tuple[str, str], tuple[float, float]] = {
            p: (0.0, 0.0) for p in self.pairs
        }
        self._bar_count = 0
        self._trade_log: list[dict] = []

    # -- engine interface ----------------------------------------------------
    def on_snapshot(self, snapshot, portfolio) -> list[SignalData]:
        ts = snapshot.timestamp
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)

        # 1) Accumulate closes for every pair leg present at this (past) bar.
        #    This runs for ALL bars, including pre-OOS warm-up history.
        bars = snapshot.bars
        for (sym_a, sym_b) in self.pairs:
            ba = bars.get(sym_a)
            bb = bars.get(sym_b)
            if ba is None or bb is None:
                continue
            try:
                pa = float(ba.close)
                pb = float(bb.close)
            except (TypeError, ValueError):
                continue
            if pa > 0 and pb > 0:
                self._hist[(sym_a, sym_b)].append((ts, pa, pb))

        self._bar_count += 1

        # 2) Suppress trading before OOS window / off rebalance cadence.
        in_sample = self.oos_start is not None and ts.date() < self.oos_start
        if in_sample or self._bar_count % self.rebalance_days != 0:
            return []

        # 3) Rebalance: z-score entry/exit per pair.
        signals: list[SignalData] = []
        for (sym_a, sym_b) in self.pairs:
            hist = self._hist[(sym_a, sym_b)]
            if len(hist) < self.warmup_bars:
                continue  # not enough warm-up history yet

            prices_a = np.array([h[1] for h in hist[-self.warmup_bars:]], dtype=float)
            prices_b = np.array([h[2] for h in hist[-self.warmup_bars:]], dtype=float)
            if np.std(prices_a) < 1e-9 or np.std(prices_b) < 1e-9:
                continue

            # Rolling OLS hedge ratio: price_b = alpha + hr * price_a.
            hr = float(np.cov(prices_a, prices_b, ddof=1)[0, 1] / np.var(prices_a, ddof=1))
            spread = prices_b - hr * prices_a
            spread_mean = float(np.mean(spread))
            spread_std = float(np.std(spread, ddof=1))
            if spread_std < 1e-9:
                continue

            # Current spread from the latest accumulated bar.
            cur_a, cur_b = float(hist[-1][1]), float(hist[-1][2])
            cur_spread = cur_b - hr * cur_a
            z = (cur_spread - spread_mean) / spread_std

            cur_qa, cur_qb = self._pos[(sym_a, sym_b)]

            if z < -self.entry_z:
                # Long the spread: buy B, sell A.
                tgt_qa = -self.target_notional_per_leg / cur_a
                tgt_qb = self.target_notional_per_leg / cur_b
            elif z > self.entry_z:
                # Short the spread: sell B, buy A.
                tgt_qa = self.target_notional_per_leg / cur_a
                tgt_qb = -self.target_notional_per_leg / cur_b
            elif abs(z) <= self.exit_z:
                tgt_qa, tgt_qb = 0.0, 0.0  # exit / flatten
            else:
                tgt_qa, tgt_qb = cur_qa, cur_qb  # hold band: keep position

            # Emit a SignalData only for legs whose target changed.
            if abs(tgt_qa - cur_qa) > 1e-9:
                self._emit_leg(signals, sym_a, cur_a, tgt_qa)
                self._pos[(sym_a, sym_b)] = (tgt_qa, self._pos[(sym_a, sym_b)][1])
            if abs(tgt_qb - cur_qb) > 1e-9:
                self._emit_leg(signals, sym_b, cur_b, tgt_qb)
                self._pos[(sym_a, sym_b)] = (self._pos[(sym_a, sym_b)][0], tgt_qb)

        return signals

    def _emit_leg(self, signals: list[SignalData], sym: str, price: float, target_qty: float) -> None:
        if abs(target_qty) < 1e-9:
            direction = "flat"
        else:
            direction = "long" if target_qty > 0 else "short"
        notional = abs(target_qty) * price
        signals.append(SignalData(
            symbol=sym,
            direction=direction,
            strength=1.0,
            strategy_id=self.name,
            notional_size=notional,
            instrument_type="equity",
        ))

    def on_fill(self, fill, portfolio) -> None:
        """Record the executed fill for the OOS trade log (gate evidence)."""
        try:
            sym = self._parse_symbol(fill.order_id)
            side_raw = str(fill.side).upper()
            side = "BUY" if side_raw.endswith("BUY") else (
                "SELL" if side_raw.endswith("SELL") else str(fill.side)
            )
            ts = getattr(fill, "timestamp", None) or pd.Timestamp.now()
            self._trade_log.append({
                "date": pd.Timestamp(ts).date().isoformat(),
                "symbol": sym,
                "side": side,
                "price": float(fill.price),
                "quantity": float(fill.quantity),
            })
        except Exception:
            pass

    @staticmethod
    def _parse_symbol(order_id: str) -> str:
        # order_id = "ord_<timestamp>_<SYMBOL>_<suffix>" (timestamp has no underscores).
        parts = str(order_id).split("_")
        if len(parts) >= 3 and parts[0] == "ord":
            return parts[2]
        return order_id

    # -- reporting -----------------------------------------------------------
    def export_trade_log(self) -> pd.DataFrame:
        cols = ["date", "symbol", "side", "price", "quantity"]
        if not self._trade_log:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self._trade_log, columns=cols)
        df["date"] = pd.to_datetime(df["date"])
        return df
