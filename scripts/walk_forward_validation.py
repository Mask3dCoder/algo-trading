"""Walk-forward validation on Section 2 top candidates.

Tests whether strategy edge persists out-of-sample using rolling
walk-forward with 2-year train / 1-year test windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantcore.backtest.analytics import compute_returns, max_drawdown, trade_analytics
from quantcore.backtest.engine import BacktestEngine
from quantcore.data.yfinance_loader import load_combined
from scripts.strategy_validation import (
    EMACrossoverFast,
    MeanReversionRSI,
    MomentumCrossover,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CANDIDATES: list[tuple[str, str, type]] = [
    ("QQQ", "Momentum (SMA 50/200)", MomentumCrossover),
    ("SPY", "Momentum (SMA 50/200)", MomentumCrossover),
    ("SPY", "EMA Crossover (20/50)", EMACrossoverFast),
    ("GLD", "Mean Reversion (RSI 30)", MeanReversionRSI),
]

FOLDS: list[tuple[str, str, str, str]] = [
    ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
]

CAPITAL = 1_000_000
SLIPPAGE_BPS = 5.0
SIZING_PCT = 0.25


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class WFFoldResult:
    train_period: str
    test_period: str
    oos_return: float = 0.0
    oos_sharpe: float = 0.0
    oos_max_dd: float = 0.0
    oos_win_rate: float = 0.0
    oos_profit_factor: float = 0.0
    oos_n_trades: int = 0


@dataclass
class WFCandidateResult:
    symbol: str
    strategy_name: str
    folds: list[WFFoldResult] = field(default_factory=list)
    aggregate_return: float = 0.0
    aggregate_sharpe: float = 0.0
    aggregate_max_dd: float = 0.0
    n_profitable_folds: int = 0
    n_total_folds: int = 0
    all_oos_returns: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Walk-forward logic
# ---------------------------------------------------------------------------


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def run_wf_fold(
    symbol: str,
    strategy_cls: type,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> WFFoldResult | None:
    """Run a single walk-forward fold.

    Loads data for train+test period, runs backtest with a fresh strategy
    instance, and extracts metrics for the test (OOS) period only.
    """
    data = load_combined([symbol], train_start, test_end, interval="1d")
    if data.empty:
        return None

    strategy = strategy_cls()
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=CAPITAL,
        slippage_bps=SLIPPAGE_BPS,
        sizing_pct=SIZING_PCT,
    )
    result_df = engine.run(data)
    if result_df.empty:
        return None

    test_start_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end)

    # OOS equity curve — normalize timestamps to tz-naive
    equity_df = result_df.set_index("timestamp")
    if equity_df.index.tz is not None:
        equity_df.index = equity_df.index.tz_localize(None)
    test_slice = equity_df[test_start_dt:test_end_dt]
    if len(test_slice) < 2:
        return None

    oos_equity = test_slice["equity"].values.astype(float)
    oos_return = float(oos_equity[-1] / oos_equity[0] - 1.0)
    oos_returns = compute_returns(oos_equity)
    oos_sharpe = float(
        np.mean(oos_returns) / np.std(oos_returns, ddof=1) * np.sqrt(252)
    ) if len(oos_returns) > 1 and np.std(oos_returns, ddof=1) > 1e-12 else 0.0
    oos_max_dd = max_drawdown(oos_equity)

    # OOS trades — strip tz from fill timestamps (they come from tz-aware data)
    test_trades = []
    for t in engine.portfolio.trades:
        fill_ts = pd.Timestamp(t.timestamp)
        if fill_ts.tz is not None:
            fill_ts = fill_ts.tz_convert(None)
        if test_start_dt <= fill_ts <= test_end_dt:
            test_trades.append(t)
    trade_pnls = engine._compute_trade_pnls(test_trades) if test_trades else []
    ta = trade_analytics(np.array(trade_pnls, dtype=float)) if trade_pnls else {
        "win_rate": 0.0, "profit_factor": 0.0, "n_trades": 0,
    }

    return WFFoldResult(
        train_period=f"{train_start} -> {train_end}",
        test_period=f"{test_start} -> {test_end}",
        oos_return=oos_return,
        oos_sharpe=oos_sharpe,
        oos_max_dd=oos_max_dd,
        oos_win_rate=float(ta["win_rate"]),
        oos_profit_factor=float(ta["profit_factor"]),
        oos_n_trades=int(ta["n_trades"]),
    )


def evaluate_candidate(
    symbol: str, strategy_name: str, strategy_cls: type
) -> WFCandidateResult:
    """Run walk-forward across all folds for a single candidate."""
    result = WFCandidateResult(symbol=symbol, strategy_name=strategy_name)

    all_returns: list[float] = []
    profitable = 0
    completed = 0

    for train_start, train_end, test_start, test_end in FOLDS:
        fold = run_wf_fold(symbol, strategy_cls, train_start, train_end, test_start, test_end)
        if fold is None:
            continue
        result.folds.append(fold)
        all_returns.extend(
            [fold.oos_return]  # Each fold contributes one OOS period return
        )
        completed += 1
        if fold.oos_sharpe > 0:
            profitable += 1

    result.n_total_folds = completed
    result.n_profitable_folds = profitable
    result.all_oos_returns = all_returns

    if all_returns:
        rets = np.array(all_returns, dtype=float)
        n_years = completed  # Each fold is 1 year
        result.aggregate_return = float(np.prod(1.0 + rets) - 1.0)
        result.aggregate_sharpe = float(
            np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(n_years)
        ) if n_years > 1 and np.std(rets, ddof=1) > 1e-12 else float(
            np.mean(rets) / 0.01
        ) if n_years == 1 else 0.0
        # Max DD on chained OOS returns
        chain = np.cumprod(1.0 + rets)
        result.aggregate_max_dd = max_drawdown(chain)

    return result


def verdict(results: list[WFCandidateResult]) -> dict[str, Any]:
    """Determine if the walk-forward edge holds."""
    total_folds = sum(r.n_total_folds for r in results)
    profitable_folds = sum(r.n_profitable_folds for r in results)

    passing = [r for r in results if r.n_profitable_folds >= r.n_total_folds * 0.5
               and r.aggregate_sharpe > 0.5]
    weak = [r for r in results if r not in passing
            and (r.n_profitable_folds >= r.n_total_folds * 0.5
                 or r.aggregate_sharpe > 0.5)]

    if len(passing) >= 2:
        return {
            "verdict": "EDGE CONFIRMED OOS",
            "message": (
                f"{len(passing)}/{len(results)} candidates show consistent "
                f"out-of-sample performance (>=50% profitable folds, "
                f"aggregate OOS Sharpe > 0.5)."
            ),
            "passing": passing,
            "weak": weak,
            "total_folds": total_folds,
            "profitable_folds": profitable_folds,
        }
    elif len(passing) >= 1:
        return {
            "verdict": "EDGE PARTIALLY CONFIRMED",
            "message": (
                f"Only {len(passing)}/{len(results)} candidates fully pass. "
                f"{len(weak)} additional show partial OOS strength."
            ),
            "passing": passing,
            "weak": weak,
            "total_folds": total_folds,
            "profitable_folds": profitable_folds,
        }
    else:
        return {
            "verdict": "EDGE NOT CONFIRMED OOS",
            "message": (
                f"0/{len(results)} candidates pass. "
                f"{profitable_folds}/{total_folds} total folds profitable. "
                f"Edge may be period-specific or require longer holding periods."
            ),
            "passing": passing,
            "weak": weak,
            "total_folds": total_folds,
            "profitable_folds": profitable_folds,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 78)
    print("  QUANTCORE WALK-FORWARD VALIDATION")
    print("=" * 78)
    print(f"  Candidates: {len(CANDIDATES)}")
    print(f"  Folds:      {len(FOLDS)} (2yr train / 1yr test, rolling)")
    print(f"  Capital:    ${CAPITAL:,.0f}")
    print(f"  Sizing:     {SIZING_PCT:.0%}")
    print("=" * 78)

    all_results: list[WFCandidateResult] = []

    for symbol, name, cls in CANDIDATES:
        print(f"\n{'-' * 78}")
        print(f"  {symbol} / {name}")
        print(f"{'-' * 78}")

        result = evaluate_candidate(symbol, name, cls)
        all_results.append(result)

        print(f"  {'Fold':<35s} {'Return':>8s} {'Sharpe':>8s} {'MaxDD':>7s} "
              f"{'Win%':>6s} {'PF':>7s} {'Trades':>6s}")
        print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 7} "
              f"{'-' * 6} {'-' * 7} {'-' * 6}")

        for fold in result.folds:
            print(f"  {fold.test_period:<35s} "
                  f"{fold.oos_return:>8.1%} {fold.oos_sharpe:>8.2f} "
                  f"{fold.oos_max_dd:>7.1%} {fold.oos_win_rate:>6.1%} "
                  f"{fold.oos_profit_factor:>7.2f} {fold.oos_n_trades:>6d}")

        print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 7} "
              f"{'-' * 6} {'-' * 7} {'-' * 6}")
        print(f"  {'AGGREGATE OOS':<35s} "
              f"{result.aggregate_return:>8.1%} {result.aggregate_sharpe:>8.2f} "
              f"{result.aggregate_max_dd:>7.1%}")
        print(f"  Profitable folds: {result.n_profitable_folds}/{result.n_total_folds}")

    # Overall verdict
    v = verdict(all_results)
    print(f"\n{'=' * 78}")
    print(f"  VERDICT: {v['verdict']}")
    print(f"  {v['message']}")

    if v["passing"]:
        print(f"\n  >>> Walk-forward confirmed candidates:")
        for r in v["passing"]:
            print(f"      {r.symbol}/{r.strategy_name} — "
                  f"OOS Sharpe={r.aggregate_sharpe:.2f}, "
                  f"{r.n_profitable_folds}/{r.n_total_folds} profitable folds")

    if v["weak"]:
        print(f"\n  >>> Partially confirmed:")
        for r in v["weak"]:
            print(f"      {r.symbol}/{r.strategy_name} — "
                  f"OOS Sharpe={r.aggregate_sharpe:.2f}, "
                  f"{r.n_profitable_folds}/{r.n_total_folds} profitable folds")

    print(f"\n  {v['profitable_folds']}/{v['total_folds']} total folds profitable")
    print("=" * 78)

    return 0 if v["verdict"].startswith("EDGE CONFIRMED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
