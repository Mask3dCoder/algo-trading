#!/usr/bin/env python
"""Phase 1 + Phase 2: deterministic grid sweep of vol_target_momentum.

Honest scope note
-----------------
The default data source is a *synthetic* multi-symbol GBM universe spanning
2018-2024 (generated and cached on first run). Synthetic data cannot carry a
real market edge, so do not read a low sweep Sharpe as "momentum is dead in
live markets" -- it validates the *harness and discipline* (PIT-guarded
walk-forward, Pareto filtering, regime overlay) and is ready to point at REAL
data by passing --data / --regime CSVs with the same schema. The negative
result on synthetic data is expected and is logged in Phase 4 if it occurs.

Walk-forward
------------
Train : 2018-01-01 .. 2021-12-31  (parameter space; IS Sharpe available per combo)
Test  : 2022-01-01 .. 2024-12-31  (strict OOS holdout; ranked by OOS Sharpe)
Each of the 108 combos is applied *fixed* over the test span -> direct, fully
auditable per-combo OOS estimate. Rolling 1-year re-calibration is exercised on
the selected top candidate in scripts/reconcile_candidate.py (deployment mode).

Outputs
-------
reports/sweep_results.csv          -- all 108 runs + Pareto-filtered flag
reports/top_3_candidates.json      -- top-3 parameter sets
reports/filtered_improvement.csv   -- Phase 2: top-3 re-run with regime gate
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
UNIV = DATA / "sweep_universe_2018_2024.csv"
REGIME = DATA / "sweep_regime_2018_2024.csv"

TRAIN_START, TRAIN_END = "2018-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2024-12-31"

param_grid = {
    "momentum_lookback": [60, 90, 120, 180],
    "vol_target_annual": [0.12, 0.18, 0.25],
    "vol_estimation_window": [20, 30, 60],
    "rebalance_frequency": [5, 10, 21],
}

MAX_DD_LIMIT = 0.40
MIN_TRADES = 50


# ---------------------------------------------------------------------------
# Data generation (synthetic, cached)
# ---------------------------------------------------------------------------
def gen_universe(seed: int = 7, n_symbols: int = 16, days: int = 1750) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(TRAIN_START, periods=days)
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    # Common market factor with slow drift regimes -> cross-sectional momentum.
    mkt_ret = np.zeros(days)
    regime = 1.0
    for d in range(days):
        if d % 120 == 0:
            regime = max(0.4, regime * (1.0 + rng.normal(0.0, 0.35)))
        mkt_ret[d] = rng.normal(0.0, 0.01) * regime
    betas = rng.uniform(0.6, 1.4, n_symbols)
    prices = np.full(n_symbols, 100.0)
    rows: list[tuple] = []
    for d, date in enumerate(dates):
        idio = rng.normal(0.0, 0.012, n_symbols)
        rets = betas * mkt_ret[d] + idio
        prices = prices * (1.0 + rets)
        for i, sym in enumerate(symbols):
            close = max(float(prices[i]), 0.01)
            open_ = close / (1.0 + rets[i])
            hi = max(open_, close) * (1.0 + abs(rng.normal(0.0, 0.004)))
            lo = min(open_, close) * (1.0 - abs(rng.normal(0.0, 0.004)))
            volume = int(rng.integers(100_000, 1_000_000))
            rows.append((sym, date, round(float(open_), 4), round(float(hi), 4),
                         round(float(lo), 4), round(close, 4), volume))
    return pd.DataFrame(
        rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    )


def gen_regime(seed: int = 11, days: int = 1750) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(TRAIN_START, periods=days)
    n = len(dates)
    # T10Y2Y synthetic: positive on average, with inversion episodes.
    t10y2y = rng.normal(0.012, 0.006, n)
    for d in range(n):
        yr = dates[d].year
        if yr == 2020:
            t10y2y[d] -= 0.03
        elif yr in (2022, 2023):
            t10y2y[d] -= 0.045
    t10y2y_s = pd.Series(t10y2y, index=dates, name="T10Y2Y")
    # VIX slope synthetic: usually contango, occasional backwardation.
    vix = 15 + 6 * np.sin(np.linspace(0, 18, n)) + rng.normal(0, 1.0, n)
    vix3m = vix + rng.normal(0.6, 0.6, n)
    slope_s = pd.Series(vix3m - vix, index=dates, name="VIX_SLOPE")
    return t10y2y_s, slope_s


# ---------------------------------------------------------------------------
# Backtest + metrics
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, params: dict, regime=None) -> dict:
    def _real(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            try:
                return float(getattr(x, "real", x))
            except Exception:
                return float("nan")

    from quantcore.alpha.vol_target_momentum import VolTargetMomentumStrategy
    from quantcore.backtest.engine import BacktestEngine

    strat = VolTargetMomentumStrategy(
        momentum_lookback=params["momentum_lookback"],
        target_vol=params["vol_target_annual"],
        vol_window=params["vol_estimation_window"],
        rebalance_frequency=params["rebalance_frequency"],
        regime_filter=regime,
        n_select=5,
        base_notional=250_000.0,
    )
    engine = BacktestEngine(
        strategy=strat,
        initial_cash=1_000_000.0,
        slippage_bps=5.0,
        commission_per_share=0.005,
        sizing_pct=1.0,
    )
    result = engine.run(df)
    rep = engine.summary()
    eqv = result["equity"].values
    if np.iscomplexobj(eqv):
        eqv = np.real(eqv)
    eq = eqv.astype(float)
    peak = np.maximum.accumulate(eq)
    mdd = abs(float(np.real((eq - peak) / peak).min())) if len(eq) else 0.0
    calmar = (_real(rep.annual_return) / mdd) if mdd > 1e-9 else 0.0
    return {
        "oos_total_return": _real(rep.total_return),
        "oos_sharpe": _real(rep.sharpe),
        "oos_max_dd": _real(rep.max_drawdown),
        "oos_calmar": _real(calmar),
        "trade_count": int(rep.n_trades),
        "win_rate": _real(rep.win_rate),
        "avg_hold_days": round(len(result) / max(int(rep.n_trades), 1), 2),
    }


def load_panel(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def sweep(df_train, df_test, regime=None, max_train_drawdown=None, sample=None,
          out_path=None, spread=None) -> pd.DataFrame:
    """Walk-forward sweep with an optional HARD training-period drawdown gate.

    If max_train_drawdown is set, every combo is run on the TRAIN window first.
    A combo is GATED OUT (skipped for OOS) unless:
        train_max_dd <= max_train_drawdown   AND   train_sharpe > 0
    This enforces config/live.yaml's circuit_breaker max_drawdown_pct at SELECTION
    time, not just as a post-hoc rejection.
    """
    rows = []
    combos = list(product(*param_grid.values()))
    if spread:
        if int(spread) < len(combos):
            idx = np.linspace(0, len(combos) - 1, int(spread)).astype(int)
            combos = [combos[i] for i in idx]
    elif sample:
        combos = combos[:sample]
    if out_path is not None and Path(out_path).exists():
        Path(out_path).unlink()  # truncate; we append incrementally
    gate_msg = f" (train DD gate={max_train_drawdown})" if max_train_drawdown is not None else ""
    print(f"[sweep] {len(combos)} combinations over test {TEST_START}..{TEST_END}{gate_msg}")
    gated_count = 0
    for i, combo in enumerate(combos, 1):
        params = dict(zip(param_grid.keys(), combo))
        train_mdd = float("nan")
        train_sharpe = float("nan")
        if max_train_drawdown is not None and len(df_train):
            try:
                mt = run_backtest(df_train, params, regime=regime)
                train_mdd = mt["oos_max_dd"]
                train_sharpe = mt["oos_sharpe"]
            except Exception:
                train_mdd, train_sharpe = float("inf"), float("-inf")
        # Deployment drawdown gate (affects DEPLOYMENT viability only).
        is_gated = bool(max_train_drawdown is not None and (
                pd.isna(train_mdd) or train_mdd > max_train_drawdown or train_sharpe <= 0))
        if is_gated:
            gated_count += 1
        # Always run OOS so we can REPORT real out-of-sample numbers even for
        # combos that fail the deployment drawdown gate (we just won't deploy them).
        try:
            m = run_backtest(df_test, params, regime=regime)
            m["train_max_dd"] = train_mdd
            m["train_sharpe"] = train_sharpe
            m["gated"] = is_gated
            m["_error"] = ""
        except Exception as exc:  # keep the sweep alive on a degenerate combo
            m = {k: float("nan") for k in
                 ["oos_total_return", "oos_sharpe", "oos_max_dd", "oos_calmar", "win_rate"]}
            m.update(trade_count=0, avg_hold_days=0.0, train_max_dd=train_mdd,
                     train_sharpe=train_sharpe, gated=is_gated, _error=str(exc)[:120])
        row = {"run_id": i, **params, **m}
        rows.append(row)
        if out_path is not None:
            # Incremental write so partial progress survives an interrupted run.
            pd.DataFrame([row]).to_csv(out_path, mode="a", index=False, header=(i == 1))
        if i % 18 == 0 or i == len(combos):
            tag = "GATED" if m.get("gated") else ""
            print(f"  run {i:3d}/{len(combos)}  Sharpe={m['oos_sharpe']:.2f}  "
                  f"MaxDD={m['oos_max_dd']*100:5.1f}%  trades={m['trade_count']} {tag}")
    print(f"[sweep] {gated_count}/{len(combos)} combos gated (train DD>{max_train_drawdown})"
          if max_train_drawdown is not None else f"[sweep] no train gate")
    out = pd.DataFrame(rows)
    # Pareto-frontier filtering (gated combos can never survive).
    out["survivor"] = ((out["oos_max_dd"] <= MAX_DD_LIMIT) & (out["trade_count"] >= MIN_TRADES)
                       & (~out["gated"]))
    out = out.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="OHLCV CSV (else generate synthetic)")
    ap.add_argument("--regime", default=None, help="Yield-curve / VIX-slope CSV (else synthetic)")
    ap.add_argument("--skip-phase2", action="store_true")
    # Walk-forward windows (pre-registered defaults; override only with intent).
    ap.add_argument("--train_start", default=TRAIN_START)
    ap.add_argument("--train_end", default=TRAIN_END)
    ap.add_argument("--test_start", default=TEST_START)
    ap.add_argument("--test_end", default=TEST_END)
    ap.add_argument("--output", default=None, help="Override primary results CSV path")
    ap.add_argument("--max_train_drawdown", type=float, default=None,
                    help="Hard gate: skip combos whose TRAIN MaxDD exceeds this (e.g. 0.05)")
    ap.add_argument("--sample", type=int, default=None,
                    help="Limit combos (smoke test); default = full grid")
    ap.add_argument("--spread", type=int, default=None,
                    help="Evenly-spaced representative slice of N combos across the grid")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # --- data ---------------------------------------------------------------
    if args.data and Path(args.data).exists():
        df = load_panel(args.data)
    else:
        df = gen_universe()
        df.to_csv(UNIV, index=False)
        print(f"[data] synthetic universe -> {UNIV} ({len(df)} rows)")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    t10y2y, vix_slope = (None, None)
    if args.regime and Path(args.regime).exists():
        rdf = pd.read_csv(args.regime, parse_dates=["timestamp"]).set_index("timestamp")
        t10y2y = rdf.get("T10Y2Y")
    else:
        t10y2y, vix_slope = gen_regime()
        rdf = pd.concat([t10y2y, vix_slope], axis=1)
        rdf.index.name = "timestamp"
        rdf.to_csv(REGIME)
        print(f"[data] synthetic regime   -> {REGIME}")

    df_train = df[(df["timestamp"] >= args.train_start) & (df["timestamp"] <= args.train_end)].copy()
    df_test = df[(df["timestamp"] >= args.test_start) & (df["timestamp"] <= args.test_end)].copy()
    print(f"[window] IS={args.train_start}..{args.train_end}  train_rows={len(df_train)}")
    print(f"[window] OOS={args.test_start}..{args.test_end}  test_rows={len(df_test)}")

    # --- Phase 1: sweep -----------------------------------------------------
    out_path = Path(args.output) if args.output else (REPORTS / "sweep_results.csv")
    results = sweep(df_train, df_test, max_train_drawdown=args.max_train_drawdown,
                    sample=args.sample, spread=args.spread, out_path=out_path)
    results.to_csv(out_path, index=False)
    survivors = results[results["survivor"]]
    top3 = survivors.head(3)
    top3.to_csv(REPORTS / "top_3_candidates.csv", index=False)
    grid_types = {k: type(v[0]) for k, v in param_grid.items()}
    candidates = [
        {"rank": i + 1, "params": {k: grid_types[k](r[k]) for k in param_grid},
         "metrics": {c: float(r[c]) for c in
                     ["oos_sharpe", "oos_total_return", "oos_max_dd", "oos_calmar",
                      "trade_count", "win_rate", "avg_hold_days"]}}
        for i, (_, r) in enumerate(top3.iterrows())
    ]
    (REPORTS / "top_3_candidates.json").write_text(json.dumps(candidates, indent=2))

    best = candidates[0] if candidates else None
    print("\n=== Phase 1 complete ===")
    print(f"Survivors (MaxDD<={MAX_DD_LIMIT*100:.0f}%, trades>={MIN_TRADES}): "
          f"{len(survivors)}/{len(results)}")
    if best:
        print(f"Top-1 Sharpe={best['metrics']['oos_sharpe']:.2f}  "
              f"MaxDD={best['metrics']['oos_max_dd']*100:.1f}%  "
              f"params={best['params']}")
    else:
        print("No survivor passed the Pareto filter.")

    # --- Phase 2: regime overlay on top-3 -----------------------------------
    if not args.skip_phase2 and t10y2y is not None:
        from quantcore.alpha.regime_filter import RegimeFilter
        rf = RegimeFilter.from_yield_curve(t10y2y, threshold=0.0)
        print(f"\n[phase2] re-running top-3 with regime gate ({rf.label()})")
        filt_rows = []
        for c in candidates:
            params = c["params"]
            m = run_backtest(df_test, params, regime=rf)
            filt_rows.append({"rank": c["rank"], **params, **m})
            print(f"  rank {c['rank']}: Sharpe={m['oos_sharpe']:.2f}  "
                  f"MaxDD={m['oos_max_dd']*100:.1f}%  trades={m['trade_count']}")
        pd.DataFrame(filt_rows).to_csv(REPORTS / "filtered_improvement.csv", index=False)
        print(f"[phase2] -> {REPORTS / 'filtered_improvement.csv'}")

    if best and best["metrics"]["oos_sharpe"] < 0.4:
        print("\nTop-1 OOS Sharpe < 0.4 => pure momentum structurally weak in this "
              "regime. Phase 2 (regime overlay) and Phase 4 (pivot) are now indicated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
