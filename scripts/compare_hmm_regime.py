"""Compare static vs HMM-regime-weighted VRP portfolio allocation.

Runs the VRP portfolio with static weights (58/35/7) vs HMM-dynamic
weights and compares Sharpe, MaxDD, and DSR p-value.

Usage:
    python scripts/compare_hmm_regime.py
    python scripts/compare_hmm_regime.py --train-start 2018-01-01 --train-end 2023-12-31
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


def _load_data(start: str, end: str) -> dict[str, pd.Series]:
    """Load SPY, VIX, HYG, TLT data."""
    import yfinance as yf

    tickers = ["SPY", "^VIX", "HYG", "TLT"]
    data = {}

    for ticker in tickers:
        _log_download(ticker)
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        key = ticker.replace("^", "").lower()
        data[key] = df["Close"].dropna()

    return data


def _log_download(ticker: str) -> None:
    print(f"  Downloading {ticker}...")


def _run_vrp_backtest(
    close: pd.Series,
    use_enhanced_iv: bool = True,
    strategy: str = "iron_condor",
) -> tuple[np.ndarray, dict]:
    """Run a single VRP strategy backtest."""
    if strategy == "iron_condor":
        from quantcore.alpha.iron_condor import IronCondorVRP
        strat = IronCondorVRP(
            close, use_enhanced_iv=use_enhanced_iv,
            delta_short=0.16, delta_long=0.10,
            dte_entry_min=30, dte_entry_max=45,
            dte_exit=21, profit_target_pct=0.50,
            stop_loss_mult=2.0, risk_per_trade_pct=0.02,
            entry_interval_days=30, ivr_entry=40.0,
        )
    elif strategy == "delta_neutral":
        from quantcore.alpha.delta_neutral import DeltaNeutralVRP
        strat = DeltaNeutralVRP(
            close, use_enhanced_iv=use_enhanced_iv,
            structure="straddle", dte_entry_min=30, dte_entry_max=45,
            dte_exit=14, profit_target_pct=0.50,
            risk_per_trade_pct=0.02, max_loss_mult=3.0,
            entry_interval_days=30, ivr_entry=0.0,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    equity, daily_rets, turnover, logs = strat.run_returns_backtest(
        start=close.index[0], end=close.index[-1],
        training_start=close.index[0], initial_capital=1_000_000.0,
    )

    return daily_rets.values, {"trades": logs.get("trades", [])}


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
    train_start: str = "2018-01-01",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    test_end: str = "2024-12-31",
    static_weights: list[float] | None = None,
    hmm_fraction: float = 0.25,
    crisis_scalar: float = 0.0,
) -> dict:
    """Run the HMM vs static comparison."""
    from quantcore.regime.hmm_regime_filter import HMMRegimeFilter, RegimeFilterConfig

    print("Loading market data...")
    data = _load_data(train_start, test_end)

    spy = data["spy"]
    vix = data["vix"]
    hyg = data.get("hyg")
    tlt = data.get("tlt")

    print(f"Data: {len(spy)} bars from {spy.index[0].date()} to {spy.index[-1].date()}")

    # --- Train HMM ---
    print("\nTraining HMM regime filter...")
    t0 = time.time()

    config = RegimeFilterConfig(
        n_states=3,
        covar_type="diag",
        n_restarts=5,
        allocation_scalars=[1.0, 0.5, crisis_scalar],
    )
    hmm_filter = HMMRegimeFilter(config)
    hmm_filter.train(vix, spy, hyg, tlt, train_start=train_start, train_end=train_end)

    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")

    # Print state labels
    for sl in hmm_filter.get_state_labels():
        print(f"  State {sl.state_idx} ({sl.label}): "
              f"VIX={sl.avg_vix:.1f}, RVol={sl.avg_rvol*100:.1f}%, "
              f"scalar={sl.allocation_scalar}, occ={sl.occupancy_pct:.1f}%, "
              f"dur={sl.avg_duration_days:.1f}d")

    # --- Run strategies on test period ---
    test_spy = spy[(spy.index >= test_start) & (spy.index <= test_end)]

    if len(test_spy) < 60:
        return {"error": "Insufficient test data"}

    print(f"\nRunning VRP strategies on test period ({test_start} to {test_end})...")

    # Static portfolio
    static_w = static_weights or [0.58, 0.35, 0.07]
    strategies = ["iron_condor", "delta_neutral"]

    static_returns = np.zeros(len(test_spy) - 1)
    hmm_returns = np.zeros(len(test_spy) - 1)

    for i, strat_name in enumerate(strategies):
        print(f"  Running {strat_name}...")
        strat_rets, _ = _run_vrp_backtest(test_spy, strategy=strat_name)

        if len(strat_rets) > 0:
            # Align lengths
            min_len = min(len(strat_rets), len(static_returns))
            static_returns[:min_len] += strat_rets[:min_len] * static_w[i]

            # HMM-weighted: multiply by daily scalar
            for j in range(min_len):
                date = test_spy.index[j + 1] if j + 1 < len(test_spy) else test_spy.index[-1]
                scalar = hmm_filter.allocation_scalar(date)
                hmm_returns[j] += strat_rets[j] * static_w[i] * scalar

    # Compute metrics
    static_metrics = _compute_metrics(static_returns, "Static (58/35/7)")
    hmm_metrics = _compute_metrics(hmm_returns, "HMM-Weighted")

    results = {
        "train_period": f"{train_start} to {train_end}",
        "test_period": f"{test_start} to {test_end}",
        "static_weights": static_w,
        "hmm_states": [
            {
                "state": sl.state_idx,
                "label": sl.label,
                "allocation_scalar": sl.allocation_scalar,
                "avg_vix": round(sl.avg_vix, 1),
                "avg_rvol_pct": round(sl.avg_rvol * 100, 1),
                "occupancy_pct": round(sl.occupancy_pct, 1),
                "avg_duration_days": round(sl.avg_duration_days, 1),
            }
            for sl in hmm_filter.get_state_labels()
        ],
        "static": static_metrics,
        "hmm_weighted": hmm_metrics,
    }

    # Improvement
    if "sharpe" in static_metrics and "sharpe" in hmm_metrics:
        results["improvement"] = {
            "sharpe_delta": round(hmm_metrics["sharpe"] - static_metrics["sharpe"], 4),
            "dd_delta": round(hmm_metrics["max_drawdown"] - static_metrics["max_drawdown"], 4),
            "dsr_delta": round(hmm_metrics["dsr_pvalue"] - static_metrics["dsr_pvalue"], 4),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare static vs HMM regime allocation")
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument("--crisis-scalar", type=float, default=0.0,
                        help="Allocation scalar for crisis state (0.0 = no VRP, 0.2 = reduced)")
    args = parser.parse_args()

    results = run_comparison(
        args.train_start, args.train_end, args.test_start, args.test_end,
        crisis_scalar=args.crisis_scalar,
    )

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    # Print summary
    print("\n" + "=" * 70)
    print("  HMM REGIME vs STATIC ALLOCATION COMPARISON")
    print("=" * 70)

    s = results["static"]
    h = results["hmm_weighted"]

    print(f"\n  {'Metric':<25} {'Static':>15} {'HMM':>15} {'Delta':>12}")
    print(f"  {'-'*67}")
    print(f"  {'Sharpe':<25} {s.get('sharpe', 0):>+15.4f} {h.get('sharpe', 0):>+15.4f} {results.get('improvement', {}).get('sharpe_delta', 0):>+12.4f}")
    print(f"  {'Max Drawdown':<25} {s.get('max_drawdown', 0):>15.4f} {h.get('max_drawdown', 0):>15.4f} {results.get('improvement', {}).get('dd_delta', 0):>+12.4f}")
    print(f"  {'DSR p-value':<25} {s.get('dsr_pvalue', 0):>15.4f} {h.get('dsr_pvalue', 0):>15.4f} {results.get('improvement', {}).get('dsr_delta', 0):>+12.4f}")
    print(f"  {'Annual Return':<25} {s.get('annual_return', 0)*100:>14.2f}% {h.get('annual_return', 0)*100:>14.2f}%")
    print(f"  {'OOS Bars':<25} {s.get('n_bars', 0):>15d} {h.get('n_bars', 0):>15d}")

    # Save report
    report_path = REPO_ROOT / "reports" / "tier2_hmm_regime_comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# HMM Regime vs Static Allocation Comparison",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Setup",
        "",
        f"- **Train period**: {results['train_period']}",
        f"- **Test period**: {results['test_period']}",
        f"- **Static weights**: {results['static_weights']}",
        f"- **HMM states**: {len(results['hmm_states'])}",
        "",
        "## HMM State Labels",
        "",
        "| State | Label | Avg VIX | Avg RVol% | Scalar | Occupancy% | Avg Duration |",
        "|-------|-------|---------|-----------|--------|------------|--------------|",
    ]

    for st in results["hmm_states"]:
        lines.append(
            f"| {st['state']} | {st['label']} | {st['avg_vix']} | "
            f"{st['avg_rvol_pct']}% | {st['allocation_scalar']} | "
            f"{st['occupancy_pct']}% | {st['avg_duration_days']}d |"
        )

    lines.extend([
        "",
        "## Results",
        "",
        "| Metric | Static | HMM-Weighted | Delta |",
        "|--------|--------|--------------|-------|",
        f"| Sharpe | {s.get('sharpe', 0):+.4f} | {h.get('sharpe', 0):+.4f} | {results.get('improvement', {}).get('sharpe_delta', 0):+.4f} |",
        f"| Max Drawdown | {s.get('max_drawdown', 0):.4f} | {h.get('max_drawdown', 0):.4f} | {results.get('improvement', {}).get('dd_delta', 0):+.4f} |",
        f"| DSR p-value | {s.get('dsr_pvalue', 0):.4f} | {h.get('dsr_pvalue', 0):.4f} | {results.get('improvement', {}).get('dsr_delta', 0):+.4f} |",
        f"| Annual Return | {s.get('annual_return', 0)*100:.2f}% | {h.get('annual_return', 0)*100:.2f}% | — |",
        "",
        "## Interpretation",
        "",
        "The HMM regime filter reduces allocation during high-volatility regimes.",
        "If the HMM-weighted portfolio has:",
        "- Lower MaxDD: The regime filter successfully de-risked during volatile periods",
        "- Similar Sharpe: The filter didn't sacrifice returns for risk reduction",
        "- Lower DSR p-value: The improved risk-adjusted performance is more statistically significant",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = REPO_ROOT / "reports" / "hmm_regime_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Report: {report_path}")
    print(f"  JSON:   {json_path}")


if __name__ == "__main__":
    main()
