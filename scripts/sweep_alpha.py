#!/usr/bin/env python
"""Phase 4: Generic alpha sweep harness — switch the core signal.

Same walk-forward + Pareto-filter + regime-overlay machinery as
scripts/sweep_vol_target_momentum.py, but the alpha is pluggable via a factory.
This is the "pivot" tool: when momentum/vol-target fails, point it at a
different signal without rewriting the harness.

Signals
-------
  vol_target_momentum  -> VolTargetMomentumStrategy (drop-in; proves the
                          generic harness reproduces sweep_vol_target_momentum)
  stat_arb / residual_momentum
                        -> existing batch alphas wrapped by BatchStrategyAdapter
                          so they run through BacktestEngine's PIT-guarded loop.
                          Requires a REAL priced panel (--data), not the
                          single-name OHLCV synthetic universe.

BatchStrategyAdapter
--------------------
Wraps any batch alpha exposing ``compute_target_weights(as_of, training_start)
-> dict[symbol, weight]`` into the ``on_snapshot`` Protocol. It emits long/short
SignalData from the target weights at each rebalance bar. Construction details
of each batch alpha differ, so the caller supplies a ``factory(panel)`` that
builds the concrete instance.

Usage
-----
  python scripts/sweep_alpha.py --factory vol_target_momentum
  python scripts/sweep_alpha.py --factory residual_momentum --data real_panel.csv
  python scripts/sweep_alpha.py --factory vol_target_momentum --sample 6   # smoke
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "synthetic_universe"
REPORTS = REPO / "reports"

TRAIN_START, TRAIN_END = "2018-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2024-12-31"
MAX_DD_LIMIT, MIN_TRADES = 0.40, 50


# ---------------------------------------------------------------------------
# Generic backtest
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, strat, regime=None) -> dict:
    from quantcore.backtest.engine import BacktestEngine
    if regime is not None:
        strat._regime = regime  # regime overlay applied post-hoc
    engine = BacktestEngine(strategy=strat, initial_cash=1_000_000.0,
                            slippage_bps=5.0, commission_per_share=0.005, sizing_pct=1.0)
    result = engine.run(df)
    rep = engine.summary()
    eq = result["equity"].values.astype(float)
    peak = np.maximum.accumulate(eq)
    mdd = abs(float(((eq - peak) / peak).min())) if len(eq) else 0.0
    calmar = (rep.annual_return / mdd) if mdd > 1e-9 else 0.0
    return {
        "oos_total_return": float(rep.total_return), "oos_sharpe": float(rep.sharpe),
        "oos_max_dd": float(rep.max_drawdown), "oos_calmar": float(calmar),
        "trade_count": int(rep.n_trades), "win_rate": float(rep.win_rate),
        "avg_hold_days": round(len(result) / max(int(rep.n_trades), 1), 2),
    }


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def factory_vol_target_momentum(params: dict):
    from quantcore.alpha.vol_target_momentum import VolTargetMomentumStrategy
    return VolTargetMomentumStrategy(
        momentum_lookback=params["momentum_lookback"],
        target_vol=params["vol_target_annual"],
        vol_window=params["vol_estimation_window"],
        rebalance_frequency=params["rebalance_frequency"],
        n_select=5, base_notional=250_000.0,
    )


class BatchStrategyAdapter:
    """Adapt a batch alpha (compute_target_weights) to the on_snapshot Protocol."""

    def __init__(self, batch_instance, rebalance_frequency=21, n_select=5,
                 base_notional=250_000.0, regime_filter=None):
        self._batch = batch_instance
        self._rebal = rebalance_frequency
        self._n = n_select
        self._base = base_notional
        self._regime = regime_filter
        self._bar = 0

    def on_snapshot(self, snapshot, portfolio):
        from quantcore.backtest.engine import SignalData
        self._bar += 1
        as_of = pd.Timestamp(snapshot.timestamp)
        if self._regime is not None and not self._regime.is_risk_on(snapshot.timestamp):
            return []
        if self._bar % self._rebal != 0:
            return []
        try:
            weights = self._batch.compute_target_weights(
                as_of, as_of - pd.DateOffset(years=3))
        except Exception:
            return []
        if not weights:
            return []
        items = sorted(weights.items(), key=lambda kv: kv[1])
        shorts, longs = items[: self._n], items[-self._n:]
        per = self._base / (2.0 * self._n)
        sigs = []
        for s, w in longs:
            sigs.append(SignalData(symbol=s, direction="long", strength=abs(w),
                                   strategy_id="adapter", notional_size=per))
        for s, w in shorts:
            sigs.append(SignalData(symbol=s, direction="short", strength=abs(w),
                                   strategy_id="adapter", notional_size=per))
        return sigs

    def on_bar(self, bar, portfolio):
        return []

    def on_fill(self, fill, portfolio):
        return None


def factory_residual_momentum(panel: pd.DataFrame):
    """Build a swept ResidualMomentum via BatchStrategyAdapter. Requires --data."""
    from quantcore.alpha.residual_momentum import ResidualMomentumStrategy

    def _make(params: dict) -> BatchStrategyAdapter:
        batch = ResidualMomentumStrategy(stock_prices=panel)  # wire extra ctor args as needed
        return BatchStrategyAdapter(batch, rebalance_frequency=params["rebalance_frequency"],
                                     n_select=5, base_notional=250_000.0)
    return _make


def factory_stat_arb(panel: pd.DataFrame):
    """Build a swept StatArb via the native on_snapshot strategy.

    Pairs are discovered ONCE on the point-in-time training window
    (2018-2021) so no OOS data leaks into pair selection. The returned
    builder constructs ``StatArbMeanReversionSnapshot`` with the swept
    stat-arb parameters. Requires --data (a long/tidy priced panel).
    """
    from quantcore.alpha.stat_arb_snapshot import (
        StatArbMeanReversionSnapshot, pairs_from_panel)

    pairs = pairs_from_panel(panel, train_start="2018-01-01", train_end="2021-12-31")
    if not pairs:
        raise RuntimeError("PairsFinder found 0 cointegrated pairs on training window")

    def _make(params: dict) -> StatArbMeanReversionSnapshot:
        return StatArbMeanReversionSnapshot(
            pairs=pairs,
            entry_threshold=params.get("entry_threshold", 2.0),
            lookback_zscore=params.get("lookback_zscore", 20),
            max_pairs=params.get("max_pairs", 10),
            rebalance_freq=params.get("rebalance_freq", 60),
            stop_threshold=params.get("stop_threshold", 4.0),
            max_pair_notional=params.get("max_pair_notional", 50_000.0),
        )
    return _make


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep(factory, param_grid: dict, df_test: pd.DataFrame, regime=None, sample=None) -> pd.DataFrame:
    rows = []
    combos = list(product(*param_grid.values()))
    if sample:
        combos = combos[:sample]
    print(f"[sweep_alpha] {len(combos)} combos")
    for i, combo in enumerate(combos, 1):
        params = dict(zip(param_grid.keys(), combo))
        try:
            strat = factory(params)
            m = run_backtest(df_test, strat, regime=regime)
        except Exception as exc:
            m = {k: float("nan") for k in
                 ["oos_total_return", "oos_sharpe", "oos_max_dd", "oos_calmar", "win_rate"]}
            m.update(trade_count=0, avg_hold_days=0.0, _error=str(exc)[:100])
        rows.append({"run_id": i, **params, **m})
        if i % 6 == 0 or i == len(combos):
            print(f"  run {i:3d}: Sharpe={m['oos_sharpe']:.2f} MaxDD={m['oos_max_dd']*100:5.1f}% "
                  f"trades={m['trade_count']}")
    out = pd.DataFrame(rows)
    out["survivor"] = (out["oos_max_dd"] <= MAX_DD_LIMIT) & (out["trade_count"] >= MIN_TRADES)
    return out.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factory", default="vol_target_momentum",
                    choices=["vol_target_momentum", "residual_momentum", "stat_arb"])
    ap.add_argument("--data", default=None, help="Priced panel CSV for batch alphas")
    ap.add_argument("--sample", type=int, default=None, help="limit combos (smoke test)")
    ap.add_argument("--regime", action="store_true", help="apply yield-curve overlay")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.factory == "stat_arb":
        param_grid = {
            "entry_threshold": [1.5, 2.0, 2.5],
            "lookback_zscore": [20, 40, 60],
            "max_pairs": [5, 10, 15],
            "rebalance_freq": [20, 60, 120],
        }
    else:
        param_grid = {
            "momentum_lookback": [60, 90, 120, 180],
            "vol_target_annual": [0.12, 0.18, 0.25],
            "vol_estimation_window": [20, 30, 60],
            "rebalance_frequency": [5, 10, 21],
        }

    if args.factory == "vol_target_momentum":
        factory = factory_vol_target_momentum
        df = pd.read_csv(DATA / "sweep_universe_2018_2024.csv")
    else:
        if not args.data:
            print("ERROR: --data <priced_panel.csv> required for batch alphas")
            return 2
        if args.factory == "stat_arb":
            # PIT panel is long/tidy [symbol, timestamp, ...]; do NOT index_col.
            panel = pd.read_csv(args.data, parse_dates=["timestamp"])
        else:
            panel = pd.read_csv(args.data, index_col=0, parse_dates=True)
        factory_builder = (factory_residual_momentum if args.factory == "residual_momentum"
                           else factory_stat_arb)(panel)
        factory = lambda p: factory_builder(p)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_test = df[(df["timestamp"] >= TEST_START) & (df["timestamp"] <= TEST_END)].copy()

    regime = None
    if args.regime:
        from quantcore.alpha.regime_filter import RegimeFilter
        rdf = pd.read_csv(DATA / "sweep_regime_2018_2024.csv", index_col=0, parse_dates=True)
        regime = RegimeFilter.from_yield_curve(rdf["T10Y2Y"], threshold=0.0)

    results = sweep(factory, param_grid, df_test, regime=regime, sample=args.sample)
    out_path = REPORTS / f"sweep_alpha_{args.factory}.csv"
    results.to_csv(out_path, index=False)
    survivors = results[results["survivor"]]
    top = survivors.head(3)
    best = top.iloc[0] if len(top) else None
    print(f"\n[done] survivors={len(survivors)}/{len(results)} -> {out_path}")
    if best is not None:
        params = {k: best[k] for k in param_grid}
        print(f"Top-1 Sharpe={best['oos_sharpe']:.2f} MaxDD={best['oos_max_dd']*100:.1f}% "
              f"params={params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
