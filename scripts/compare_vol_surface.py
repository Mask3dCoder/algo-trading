"""Compare flat IV premium vs enhanced vol surface for options strategies.

Runs the Iron Condor and Delta Neutral strategies with both IV models
and compares:
  - Sharpe ratio and max drawdown
  - Win rate and average trade P&L
  - Entry frequency

Usage:
    python scripts/compare_vol_surface.py
    python scripts/compare_vol_surface.py --start 2022-01-01 --end 2024-12-31
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


def _run_iron_condor(
    close: pd.Series,
    use_enhanced: bool,
) -> dict:
    """Run Iron Condor backtest with specified IV model."""
    from quantcore.alpha.iron_condor import IronCondorVRP

    ic = IronCondorVRP(
        close,
        use_enhanced_iv=use_enhanced,
        delta_short=0.16,
        delta_long=0.10,
        dte_entry_min=30,
        dte_entry_max=45,
        dte_exit=21,
        profit_target_pct=0.50,
        stop_loss_mult=2.0,
        risk_per_trade_pct=0.02,
        entry_interval_days=30,
        ivr_entry=40.0,
    )

    equity, daily_rets, turnover, logs = ic.run_returns_backtest(
        start=close.index[0],
        end=close.index[-1],
        training_start=close.index[0],
        initial_capital=1_000_000.0,
    )

    return _compute_metrics(daily_rets, turnover, logs, "Iron Condor")


def _run_delta_neutral(
    close: pd.Series,
    use_enhanced: bool,
) -> dict:
    """Run Delta Neutral backtest with specified IV model."""
    from quantcore.alpha.delta_neutral import DeltaNeutralVRP

    dn = DeltaNeutralVRP(
        close,
        use_enhanced_iv=use_enhanced,
        structure="straddle",
        dte_entry_min=30,
        dte_entry_max=45,
        dte_exit=14,
        profit_target_pct=0.50,
        risk_per_trade_pct=0.02,
        max_loss_mult=3.0,
        entry_interval_days=30,
        ivr_entry=0.0,
    )

    equity, daily_rets, turnover, logs = dn.run_returns_backtest(
        start=close.index[0],
        end=close.index[-1],
        training_start=close.index[0],
        initial_capital=1_000_000.0,
    )

    return _compute_metrics(daily_rets, turnover, logs, "Delta Neutral")


def _compute_metrics(
    daily_rets: pd.Series,
    turnover: pd.Series,
    logs: dict,
    strategy_name: str,
) -> dict:
    """Compute performance metrics from backtest results."""
    if len(daily_rets) == 0 or daily_rets.isna().all():
        return {
            "strategy": strategy_name,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "annual_return": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "avg_trade_pnl": 0.0,
            "avg_turnover": 0.0,
        }

    r = daily_rets.dropna().values
    r = np.clip(r, -0.5, 0.5)  # Cap extreme returns

    # Sharpe
    mean_r = np.mean(r)
    std_r = np.std(r, ddof=1)
    sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 1e-12 else 0.0

    # Max drawdown
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(abs(np.min(dd)))

    # Annual return
    total_ret = float(eq[-1] / eq[0] - 1) if len(eq) > 0 else 0.0
    n_years = len(r) / 252.0
    annual_ret = float((1 + total_ret) ** (1 / max(n_years, 0.01)) - 1) if total_ret > -1 else -1.0

    # Trade stats
    trades = logs.get("trades", [])
    n_trades = len(trades)
    pnls = [t.pnl for t in trades] if trades else []
    win_rate = float(np.mean(np.array(pnls) > 0)) * 100 if pnls else 0.0
    avg_pnl = float(np.mean(pnls)) if pnls else 0.0

    # Turnover
    avg_turn = float(turnover.mean()) if len(turnover) > 0 else 0.0

    return {
        "strategy": strategy_name,
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "annual_return": round(annual_ret, 4),
        "win_rate": round(win_rate, 1),
        "n_trades": n_trades,
        "avg_trade_pnl": round(avg_pnl, 2),
        "avg_turnover": round(avg_turn, 6),
    }


def run_comparison(start: str, end: str) -> dict:
    """Run the full comparison."""
    print("Loading SPY data...")
    close = _load_spy(start, end)
    print(f"Data: {len(close)} bars from {close.index[0].date()} to {close.index[-1].date()}")

    results = {}

    # --- Iron Condor ---
    print("\n--- Iron Condor ---")
    print("  Running with FLAT IV model (rv + 0.15)...")
    t0 = time.time()
    ic_flat = _run_iron_condor(close, use_enhanced=False)
    ic_flat["time_s"] = round(time.time() - t0, 1)
    print(f"  SR={ic_flat['sharpe']:+.4f}  DD={ic_flat['max_drawdown']:.4f}  Trades={ic_flat['n_trades']}")

    print("  Running with ENHANCED IV model (skew-aware)...")
    t0 = time.time()
    ic_enhanced = _run_iron_condor(close, use_enhanced=True)
    ic_enhanced["time_s"] = round(time.time() - t0, 1)
    print(f"  SR={ic_enhanced['sharpe']:+.4f}  DD={ic_enhanced['max_drawdown']:.4f}  Trades={ic_enhanced['n_trades']}")

    results["iron_condor"] = {"flat": ic_flat, "enhanced": ic_enhanced}

    # --- Delta Neutral ---
    print("\n--- Delta Neutral ---")
    print("  Running with FLAT IV model (rv + 0.15)...")
    t0 = time.time()
    dn_flat = _run_delta_neutral(close, use_enhanced=False)
    dn_flat["time_s"] = round(time.time() - t0, 1)
    print(f"  SR={dn_flat['sharpe']:+.4f}  DD={dn_flat['max_drawdown']:.4f}  Trades={dn_flat['n_trades']}")

    print("  Running with ENHANCED IV model (skew-aware)...")
    t0 = time.time()
    dn_enhanced = _run_delta_neutral(close, use_enhanced=True)
    dn_enhanced["time_s"] = round(time.time() - t0, 1)
    print(f"  SR={dn_enhanced['sharpe']:+.4f}  DD={dn_enhanced['max_drawdown']:.4f}  Trades={dn_enhanced['n_trades']}")

    results["delta_neutral"] = {"flat": dn_flat, "enhanced": dn_enhanced}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare flat vs enhanced IV models")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    results = run_comparison(args.start, args.end)

    # Print summary
    print("\n" + "=" * 70)
    print("  VOL SURFACE COMPARISON: FLAT vs ENHANCED")
    print("=" * 70)

    for strat_name, data in results.items():
        flat = data["flat"]
        enhanced = data["enhanced"]
        label = strat_name.replace("_", " ").title()

        print(f"\n  {label}:")
        print(f"    {'Metric':<25} {'Flat (rv+15%)':>15} {'Enhanced':>15} {'Delta':>12}")
        print(f"    {'-'*67}")
        print(f"    {'Sharpe':<25} {flat['sharpe']:>+15.4f} {enhanced['sharpe']:>+15.4f} {enhanced['sharpe'] - flat['sharpe']:>+12.4f}")
        print(f"    {'Max Drawdown':<25} {flat['max_drawdown']:>15.4f} {enhanced['max_drawdown']:>15.4f} {enhanced['max_drawdown'] - flat['max_drawdown']:>+12.4f}")
        print(f"    {'Win Rate %':<25} {flat['win_rate']:>15.1f} {enhanced['win_rate']:>15.1f} {enhanced['win_rate'] - flat['win_rate']:>+12.1f}")
        print(f"    {'Trades':<25} {flat['n_trades']:>15d} {enhanced['n_trades']:>15d} {enhanced['n_trades'] - flat['n_trades']:>+12d}")
        print(f"    {'Avg Trade P&L':<25} {flat['avg_trade_pnl']:>15.2f} {enhanced['avg_trade_pnl']:>15.2f} {enhanced['avg_trade_pnl'] - flat['avg_trade_pnl']:>+12.2f}")

    # Save report
    report_path = REPO_ROOT / "reports" / "tier1_vol_surface_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Volatility Surface Comparison: Flat Premium vs Enhanced",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Models Compared",
        "",
        "### Flat Premium (Legacy)",
        "- IV = realized_vol + 0.15 (constant VRP premium)",
        "- No skew adjustment",
        "- No term structure",
        "",
        "### Enhanced (New)",
        "- VIX-dependent VRP premium (scales with vol level)",
        "- OTM put skew adjustment (based on empirical SPX behavior)",
        "- Term structure adjustment (square-root-of-time rule)",
        "- Mean reversion to long-run IV",
        "",
        "## Setup",
        "",
        f"- **Underlying**: SPY",
        f"- **Period**: {args.start} to {args.end}",
        f"- **Initial capital**: $1,000,000",
        "",
    ]

    for strat_name, data in results.items():
        flat = data["flat"]
        enhanced = data["enhanced"]
        label = strat_name.replace("_", " ").title()

        better_sharpe = "Enhanced" if enhanced["sharpe"] > flat["sharpe"] else "Flat"
        better_dd = "Enhanced" if enhanced["max_drawdown"] < flat["max_drawdown"] else "Flat"

        lines.extend([
            f"## {label}",
            "",
            "| Metric | Flat (rv+15%) | Enhanced | Delta |",
            "|--------|--------------|----------|-------|",
            f"| Sharpe | {flat['sharpe']:+.4f} | {enhanced['sharpe']:+.4f} | {enhanced['sharpe'] - flat['sharpe']:+.4f} |",
            f"| Max Drawdown | {flat['max_drawdown']:.4f} | {enhanced['max_drawdown']:.4f} | {enhanced['max_drawdown'] - flat['max_drawdown']:+.4f} |",
            f"| Annual Return | {flat['annual_return']:.4f} | {enhanced['annual_return']:.4f} | {enhanced['annual_return'] - flat['annual_return']:+.4f} |",
            f"| Win Rate | {flat['win_rate']:.1f}% | {enhanced['win_rate']:.1f}% | {enhanced['win_rate'] - flat['win_rate']:+.1f}% |",
            f"| Trades | {flat['n_trades']} | {enhanced['n_trades']} | {enhanced['n_trades'] - flat['n_trades']:+d} |",
            f"| Avg Trade P&L | ${flat['avg_trade_pnl']:,.2f} | ${enhanced['avg_trade_pnl']:,.2f} | ${enhanced['avg_trade_pnl'] - flat['avg_trade_pnl']:+,.2f} |",
            "",
            f"**Better Sharpe**: {better_sharpe} | **Better Drawdown**: {better_dd}",
            "",
        ])

    # Determine overall recommendation
    ic_better = results["iron_condor"]["enhanced"]["sharpe"] >= results["iron_condor"]["flat"]["sharpe"]
    dn_better = results["delta_neutral"]["enhanced"]["sharpe"] >= results["delta_neutral"]["flat"]["sharpe"]

    if ic_better or dn_better:
        recommendation = "enhanced"
        lines.extend([
            "## Conclusion",
            "",
            "**Recommendation**: Use the **enhanced** vol surface model as default.",
            "",
            "The enhanced model produces more realistic IV estimates by accounting for:",
            "- VIX-dependent VRP premium (higher vol -> larger premium)",
            "- Strike-dependent skew (OTM puts more expensive)",
            "- Term structure (short-dated IV higher than long-dated)",
            "",
            "Even if Sharpe is similar, the enhanced model provides more realistic",
            "backtest results for live trading expectations.",
        ])
    else:
        recommendation = "flat"
        lines.extend([
            "## Conclusion",
            "",
            "**Recommendation**: Keep the **flat** premium model for now.",
            "",
            "The enhanced model did not produce materially better results.",
            "This may be because:",
            "- The strategies are not sensitive enough to skew/term structure",
            "- The VRP premium dominates over skew effects",
            "- More data or different market regimes may show the benefit",
            "",
            "The enhanced model is still available for future use.",
        ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save raw JSON
    json_path = REPO_ROOT / "reports" / "vol_surface_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Report saved: {report_path}")
    print(f"  JSON data:    {json_path}")
    print(f"\n  Recommendation: Use {recommendation.upper()} model")


if __name__ == "__main__":
    main()
