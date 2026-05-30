"""Compare risk-budget vs fractional Kelly position sizing.

Runs the VRP portfolio with both sizing methods and compares
Sharpe, MaxDD, and DSR p-value.

Usage:
    python scripts/compare_kelly_sizing.py
    python scripts/compare_kelly_sizing.py --fraction 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _load_spy(start: str, end: str) -> pd.Series:
    """Load SPY close prices."""
    import yfinance as yf

    spy = yf.download("SPY", start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return spy["Close"].dropna()


def _run_vrp_backtest(
    close: pd.Series,
    strategy: str = "iron_condor",
    risk_per_trade_pct: float = 0.02,
) -> tuple[np.ndarray, list[dict]]:
    """Run a VRP strategy backtest with specified sizing."""
    if strategy == "iron_condor":
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(
            close, use_enhanced_iv=True,
            delta_short=0.16, delta_long=0.10,
            dte_entry_min=30, dte_entry_max=45,
            dte_exit=21, profit_target_pct=0.50,
            stop_loss_mult=2.0,
            risk_per_trade_pct=risk_per_trade_pct,
            entry_interval_days=30, ivr_entry=40.0,
        )
    elif strategy == "delta_neutral":
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(
            close, use_enhanced_iv=True,
            structure="straddle", dte_entry_min=30, dte_entry_max=45,
            dte_exit=14, profit_target_pct=0.50,
            risk_per_trade_pct=risk_per_trade_pct,
            max_loss_mult=3.0, entry_interval_days=30, ivr_entry=0.0,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    equity, daily_rets, turnover, logs = strat.run_returns_backtest(
        start=close.index[0], end=close.index[-1],
        training_start=close.index[0], initial_capital=1_000_000.0,
    )

    trades = logs.get("trades", [])
    trade_dicts = []
    for t in trades:
        trade_dicts.append({
            "pnl_abs": t.pnl if hasattr(t, "pnl") else 0.0,
            "pnl_pct": t.pnl / 1_000_000.0 if hasattr(t, "pnl") else 0.0,
        })

    return daily_rets.values, trade_dicts


def _compute_metrics(returns: np.ndarray, label: str) -> dict:
    """Compute performance metrics."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    r = np.clip(r, -0.5, 0.5)

    if len(r) < 20:
        return {"label": label, "error": "insufficient data"}

    mean_r = np.mean(r)
    std_r = np.std(r, ddof=1)
    sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 1e-12 else 0.0

    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(abs(np.min(dd)))

    total_ret = float(eq[-1] / eq[0] - 1) if len(eq) > 0 else 0.0
    n_years = len(r) / 252.0
    annual_ret = float((1 + total_ret) ** (1 / max(n_years, 0.01)) - 1) if total_ret > -1 else -1.0

    # DSR
    from quantcore.research.deflated_sharpe import deflated_sharpe_ratio
    skew = float(np.mean((r - mean_r) ** 3) / (std_r ** 3)) if std_r > 1e-12 else 0.0
    kurt = float(np.mean((r - mean_r) ** 4) / (std_r ** 4) - 3.0) if std_r > 1e-12 else 0.0
    dsr_p, dsr_stat = deflated_sharpe_ratio(
        observed_sharpe=sharpe, num_trials=30,
        sample_length=len(r), skewness=skew, kurtosis_=kurt,
    )

    return {
        "label": label,
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "annual_return": round(annual_ret, 4),
        "total_return": round(total_ret, 4),
        "dsr_pvalue": round(dsr_p, 4),
        "dsr_stat": round(dsr_stat, 4),
        "n_bars": len(r),
    }


def run_comparison(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    kelly_fraction: float = 0.25,
) -> dict:
    """Run the risk-budget vs Kelly comparison."""
    from quantcore.portfolio.kelly_sizing import KellySizer

    print("Loading SPY data...")
    close = _load_spy(start, end)
    print(f"Data: {len(close)} bars from {close.index[0].date()} to {close.index[-1].date()}")

    strategies = ["iron_condor", "delta_neutral"]
    base_weights = [0.58, 0.35]

    # --- Run strategies and collect trades for Kelly fitting ---
    print("\nRunning VRP strategies...")
    strat_returns = {}
    strat_trades = {}

    for strat_name in strategies:
        print(f"  {strat_name}...")
        rets, trades = _run_vrp_backtest(close, strategy=strat_name)
        strat_returns[strat_name] = rets
        strat_trades[strat_name] = trades
        print(f"    {len(rets)} bars, {len(trades)} trades")

    # --- Fit Kelly ---
    print(f"\nFitting Kelly (fraction={kelly_fraction})...")
    kelly_sizer = KellySizer(fraction=kelly_fraction)

    for strat_name in strategies:
        trades = strat_trades[strat_name]
        if len(trades) >= 20:
            stats = kelly_sizer.fit_from_trades(strat_name, trades)
            print(f"  {strat_name}: p={stats.win_rate:.2f}, b={stats.payoff_ratio:.2f}, "
                  f"f*={stats.kelly_fraction:.3f}, frac={stats.fractional_kelly:.3f}")
        else:
            # Use returns instead
            rets = strat_returns[strat_name]
            stats = kelly_sizer.fit_from_returns(strat_name, rets)
            print(f"  {strat_name} (from returns): p={stats.win_rate:.2f}, b={stats.payoff_ratio:.2f}, "
                  f"f*={stats.kelly_fraction:.3f}, frac={stats.fractional_kelly:.3f}")

    # --- Compute Kelly-optimal weights ---
    kelly_stats = kelly_sizer.get_all_stats()
    kelly_fracs = [kelly_stats[s].fractional_kelly for s in strategies]
    total_kelly = sum(kelly_fracs)
    if total_kelly > 0:
        kelly_weights = [f / total_kelly for f in kelly_fracs]
    else:
        kelly_weights = base_weights

    print(f"\n  Risk-budget weights: {base_weights}")
    print(f"  Kelly weights:      {[round(w, 3) for w in kelly_weights]}")

    # --- Build combined returns ---
    min_len = min(len(strat_returns[s]) for s in strategies)

    risk_budget_returns = np.zeros(min_len)
    kelly_returns = np.zeros(min_len)

    for i, strat_name in enumerate(strategies):
        rets = strat_returns[strat_name][:min_len]
        risk_budget_returns += rets * base_weights[i]
        kelly_returns += rets * kelly_weights[i]

    # Scale Kelly returns by the fractional Kelly fraction
    # (Kelly sizing means we risk a fraction of equity, not the full amount)
    avg_kelly_frac = np.mean(kelly_fracs)
    avg_risk_frac = 0.02 * len(strategies)  # current risk budget
    if avg_risk_frac > 0:
        kelly_scale = avg_kelly_frac / avg_risk_frac
        kelly_returns *= kelly_scale

    # --- Compute metrics ---
    rb_metrics = _compute_metrics(risk_budget_returns, "Risk Budget")
    kelly_metrics = _compute_metrics(kelly_returns, "Fractional Kelly")

    results = {
        "period": f"{start} to {end}",
        "kelly_fraction": kelly_fraction,
        "risk_budget_weights": dict(zip(strategies, base_weights)),
        "kelly_weights": dict(zip(strategies, [round(w, 4) for w in kelly_weights])),
        "kelly_stats": {
            s: {
                "win_rate": round(kelly_stats[s].win_rate, 4),
                "payoff_ratio": round(kelly_stats[s].payoff_ratio, 4),
                "kelly_fraction": round(kelly_stats[s].kelly_fraction, 4),
                "fractional_kelly": round(kelly_stats[s].fractional_kelly, 4),
                "edge": round(kelly_stats[s].edge, 6),
            }
            for s in strategies
        },
        "risk_budget": rb_metrics,
        "kelly": kelly_metrics,
    }

    if "sharpe" in rb_metrics and "sharpe" in kelly_metrics:
        results["improvement"] = {
            "sharpe_delta": round(kelly_metrics["sharpe"] - rb_metrics["sharpe"], 4),
            "dd_delta": round(kelly_metrics["max_drawdown"] - rb_metrics["max_drawdown"], 4),
            "dsr_delta": round(kelly_metrics["dsr_pvalue"] - rb_metrics["dsr_pvalue"], 4),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare risk-budget vs Kelly sizing")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--fraction", type=float, default=0.25,
                        help="Fractional Kelly multiplier (0.25 = quarter Kelly)")
    args = parser.parse_args()

    results = run_comparison(args.start, args.end, args.fraction)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    # Print summary
    print("\n" + "=" * 70)
    print("  RISK-BUDGET vs FRACTIONAL KELLY SIZING COMPARISON")
    print("=" * 70)

    rb = results["risk_budget"]
    k = results["kelly"]

    print(f"\n  {'Metric':<25} {'Risk Budget':>15} {'Kelly':>15} {'Delta':>12}")
    print(f"  {'-'*67}")
    print(f"  {'Sharpe':<25} {rb.get('sharpe', 0):>+15.4f} {k.get('sharpe', 0):>+15.4f} {results.get('improvement', {}).get('sharpe_delta', 0):>+12.4f}")
    print(f"  {'Max Drawdown':<25} {rb.get('max_drawdown', 0):>15.4f} {k.get('max_drawdown', 0):>15.4f} {results.get('improvement', {}).get('dd_delta', 0):>+12.4f}")
    print(f"  {'DSR p-value':<25} {rb.get('dsr_pvalue', 0):>15.4f} {k.get('dsr_pvalue', 0):>15.4f} {results.get('improvement', {}).get('dsr_delta', 0):>+12.4f}")
    print(f"  {'Annual Return':<25} {rb.get('annual_return', 0)*100:>14.2f}% {k.get('annual_return', 0)*100:>14.2f}%")

    print(f"\n  Kelly stats:")
    for s, stats in results["kelly_stats"].items():
        print(f"    {s}: p={stats['win_rate']:.2f}, b={stats['payoff_ratio']:.2f}, "
              f"f*={stats['kelly_fraction']:.3f}, frac={stats['fractional_kelly']:.3f}")

    # Save report
    report_path = REPO_ROOT / "reports" / "tier2_kelly_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Risk-Budget vs Fractional Kelly Sizing Comparison",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Setup",
        "",
        f"- **Period**: {results['period']}",
        f"- **Kelly fraction**: {results['kelly_fraction']} ({'quarter' if results['kelly_fraction'] == 0.25 else 'half'} Kelly)",
        f"- **Risk-budget weights**: {results['risk_budget_weights']}",
        f"- **Kelly weights**: {results['kelly_weights']}",
        "",
        "## Kelly Statistics",
        "",
        "| Strategy | Win Rate | Payoff Ratio | Full Kelly | Fractional Kelly | Edge |",
        "|----------|----------|--------------|------------|------------------|------|",
    ]

    for s, stats in results["kelly_stats"].items():
        lines.append(
            f"| {s} | {stats['win_rate']:.2f} | {stats['payoff_ratio']:.2f} | "
            f"{stats['kelly_fraction']:.3f} | {stats['fractional_kelly']:.3f} | "
            f"{stats['edge']:.6f} |"
        )

    lines.extend([
        "",
        "## Results",
        "",
        "| Metric | Risk Budget | Kelly | Delta |",
        "|--------|-------------|-------|-------|",
        f"| Sharpe | {rb.get('sharpe', 0):+.4f} | {k.get('sharpe', 0):+.4f} | {results.get('improvement', {}).get('sharpe_delta', 0):+.4f} |",
        f"| Max Drawdown | {rb.get('max_drawdown', 0):.4f} | {k.get('max_drawdown', 0):.4f} | {results.get('improvement', {}).get('dd_delta', 0):+.4f} |",
        f"| DSR p-value | {rb.get('dsr_pvalue', 0):.4f} | {k.get('dsr_pvalue', 0):.4f} | {results.get('improvement', {}).get('dsr_delta', 0):+.4f} |",
        f"| Annual Return | {rb.get('annual_return', 0)*100:.2f}% | {k.get('annual_return', 0)*100:.2f}% | — |",
        "",
        "## Interpretation",
        "",
        "The fractional Kelly method sizes positions based on each strategy's",
        "actual edge (win rate and payoff ratio) rather than a fixed risk budget.",
        "",
        "If Kelly produces:",
        "- Higher Sharpe: Kelly correctly identified the edge and sized accordingly",
        "- Lower MaxDD: Kelly reduced exposure to strategies with weaker edges",
        "- Lower DSR p-value: The improved performance is more statistically significant",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = REPO_ROOT / "reports" / "kelly_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Report: {report_path}")
    print(f"  JSON:   {json_path}")


if __name__ == "__main__":
    main()
