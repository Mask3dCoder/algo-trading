#!/usr/bin/env python
"""Run the native vol-targeted momentum strategy through QuantCore's engine.

This is the Phase-3 "native AI research" demo from DECISION_LOG_vibe_trading.md.
It proves the stack already does what Vibe-Trading promises -- NL-assisted
strategy generation -- but better: the strategy reuses QuantCore's own
``VolTarget`` risk module and runs through the deterministic, point-in-time-
guarded ``BacktestEngine``. Zero external LLM/runtime imports.

Usage
-----
    python scripts/run_vol_target_momentum_backtest.py
    python scripts/run_vol_target_momentum_backtest.py --data my_data.csv \
        --html report.html --output equity.csv

If ``--data`` is omitted (or the file is missing) a deterministic synthetic
multi-symbol OHLCV universe is generated and written to
``data/synthetic_universe/vol_target_momentum_sample.csv`` so the run is
fully reproducible with no external dependencies.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("vt_mom_demo")

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = REPO_ROOT / "data" / "synthetic_universe" / "vol_target_momentum_sample.csv"


def make_synthetic_universe(
    seed: int = 42,
    n_symbols: int = 8,
    n_days: int = 504,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Deterministic GBM universe with slow-moving drift regimes (momentum)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    drift = rng.normal(0.0002, 0.0003, size=n_symbols)
    vol = rng.uniform(0.008, 0.018, size=n_symbols)
    prices = np.full(n_symbols, start_price)
    rows: list[tuple] = []
    for d, date in enumerate(dates):
        if d % 60 == 0:  # regime shift -> persistent trends for momentum to capture
            drift = rng.normal(0.0002, 0.0004, size=n_symbols)
        rets = rng.normal(drift, vol)
        prices = prices * (1.0 + rets)
        for i, sym in enumerate(symbols):
            close = max(float(prices[i]), 0.01)
            open_ = close / (1.0 + rets[i])
            hi = max(open_, close) * (1.0 + abs(rng.normal(0.0, 0.004)))
            lo = min(open_, close) * (1.0 - abs(rng.normal(0.0, 0.004)))
            volume = int(rng.integers(100_000, 1_000_000))
            rows.append(
                (
                    sym,
                    date,
                    round(float(open_), 4),
                    round(float(hi), 4),
                    round(float(lo), 4),
                    round(close, 4),
                    volume,
                )
            )
    return pd.DataFrame(
        rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None, help="OHLCV CSV [symbol,timestamp,open,high,low,close,volume]")
    ap.add_argument("--output", default=None, help="Equity curve CSV output path")
    ap.add_argument("--html", default=None, help="HTML report output path")
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--slippage", type=float, default=5.0)
    ap.add_argument("--commission", type=float, default=0.005)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--min-history", type=int, default=120)
    ap.add_argument("--n-select", type=int, default=3)
    ap.add_argument("--base-notional", type=float, default=250_000.0)
    args = ap.parse_args()

    # --- data ---------------------------------------------------------------
    if args.data and Path(args.data).exists():
        df = pd.read_csv(args.data)
        _log.info("Loaded data from %s (%d rows)", args.data, len(df))
    else:
        df = make_synthetic_universe()
        SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SAMPLE_CSV, index=False)
        _log.info("Generated synthetic universe -> %s (%d rows)", SAMPLE_CSV, len(df))

    # --- strategy + engine (native only) -----------------------------------
    from quantcore.alpha.vol_target_momentum import VolTargetMomentumStrategy
    from quantcore.backtest.analytics import generate_html_report
    from quantcore.backtest.engine import BacktestEngine

    strategy = VolTargetMomentumStrategy(
        momentum_lookback=args.lookback,
        min_history=args.min_history,
        n_select=args.n_select,
        base_notional=args.base_notional,
    )

    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=args.cash,
        slippage_bps=args.slippage,
        commission_per_share=args.commission,
        sizing_pct=1.0,
    )

    result = engine.run(df)
    report = engine.summary()

    _log.info("\n=== Vol-Target Momentum Backtest (native QuantCore) ===")
    _log.info("Total Return : %.2f%%", report.total_return * 100)
    _log.info("Annual Return: %.2f%%", report.annual_return * 100)
    _log.info("Sharpe       : %.2f", report.sharpe)
    _log.info("Sortino      : %.2f", report.sortino)
    _log.info("Max Drawdown : %.2f%%", report.max_drawdown * 100)
    _log.info("Win Rate     : %.2f%%", report.win_rate * 100)
    _log.info("N Trades     : %d", report.n_trades)
    if getattr(engine, "_rejected_orders", None):
        _log.info("Risk-rejected orders: %d", len(engine._rejected_orders))

    if args.output:
        result.to_csv(args.output, index=False)
        _log.info("Equity curve -> %s", args.output)
    if args.html:
        Path(args.html).write_text(generate_html_report(report), encoding="utf-8")
        _log.info("HTML report  -> %s", args.html)

    _log.info(
        "\nNOTE: a generated candidate, not a validated edge. Route through "
        "walk-forward (config/live.yaml discipline: 504d train / 126d test, "
        "DSR p<0.1) before any deployment."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
