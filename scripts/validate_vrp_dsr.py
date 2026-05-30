"""DSR-adjusted validation for VRP (Volatility Risk Premium) strategies.

Runs walk-forward validation on Iron Condor, Delta Neutral, and Put Write
strategies with proper Deflated Sharpe Ratio (Bailey & López de Prado 2012).

The DEPLOY gate requires:
  - CI_lower > 0 (statistically significant edge)
  - DSR p-value < 0.1 (not explained by multiple testing)

Usage:
    python scripts/validate_vrp_dsr.py
    python scripts/validate_vrp_dsr.py --start 2022-01-01 --end 2024-12-31
    python scripts/validate_vrp_dsr.py --n-trials 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class VRPResult:
    """Result of VRP strategy validation."""
    strategy_id: str
    oos_sharpe: float = float("nan")
    ci_lower_90: float = float("nan")
    ci_upper_90: float = float("nan")
    dsr_pvalue: float = float("nan")
    dsr_stat: float = float("nan")
    max_drawdown_pct: float = float("nan")
    win_rate_pct: float = float("nan")
    n_trades: int = 0
    calmar_ratio: float = float("nan")
    annual_return_pct: float = float("nan")
    total_return_pct: float = float("nan")
    oos_bars: int = 0
    status: str = "error"
    error: str = ""
    deploy_gate: str = ""  # PASS / MONITOR / FAIL
    deploy_reason: str = ""


def _load_spy(start: str, end: str) -> pd.Series:
    """Load SPY close prices."""
    import yfinance as yf

    spy = yf.download("SPY", start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return spy["Close"].dropna()


def _run_walk_forward_vrp(
    strategy_cls: type,
    close: pd.Series,
    train_years: int = 2,
    test_years: int = 1,
    **strategy_kwargs,
) -> tuple[np.ndarray, list, dict]:
    """Run walk-forward validation for a VRP strategy.

    Returns (oos_returns, trades, logs).
    """
    from quantcore.research.walkforward import WalkForwardBacktester

    # Instantiate strategy
    strategy = strategy_cls(close, **strategy_kwargs)

    # Create backtester
    backtester = WalkForwardBacktester(
        strategy,
        train_years=train_years,
        test_years=test_years,
    )

    # Run walk-forward
    result = backtester.run(
        dates=close.index,
        n_trials=20,  # conservative
    )

    oos_returns = result.get("catalogues_returns", np.array([]))
    trades = []  # WalkForwardBacktester doesn't return trades directly

    return oos_returns, trades, result


def _compute_dsr_metrics(
    returns: np.ndarray,
    n_strategies: int,
    n_bootstrap: int = 2000,
) -> dict:
    """Compute all metrics including DSR."""
    from quantcore.research.deflated_sharpe import deflated_sharpe_ratio
    from quantcore.research.bootstrap import bootstrap_confidence_interval

    if len(returns) < 20:
        return {"error": "insufficient OOS bars"}

    r = np.clip(returns, -0.5, 0.5)
    mean_ret = np.mean(r)
    std_ret = np.std(r, ddof=1)

    # Sharpe
    sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-12 else 0.0

    # Max drawdown
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(abs(np.min(dd)))

    # Annual return
    total_ret = float(eq[-1] / eq[0] - 1) if len(eq) > 0 else 0.0
    n_years = len(r) / 252.0
    annual_ret = float((1 + total_ret) ** (1 / max(n_years, 0.01)) - 1) if total_ret > -1 else -1.0

    # Calmar
    calmar = annual_ret / max_dd if max_dd > 0.001 else float("nan")

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_confidence_interval(
        returns, block_size=max(5, int(len(returns) ** (1 / 3))),
        n_boot=n_bootstrap, ci=0.90,
    )

    # DSR (full B&LdP formula)
    skew = float(np.mean((r - mean_ret) ** 3) / (std_ret ** 3)) if std_ret > 1e-12 else 0.0
    kurt = float(np.mean((r - mean_ret) ** 4) / (std_ret ** 4) - 3.0) if std_ret > 1e-12 else 0.0

    dsr_pvalue, dsr_stat = deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        num_trials=n_strategies + 10,  # conservative
        sample_length=len(returns),
        skewness=skew,
        kurtosis_=kurt,
    )

    return {
        "oos_sharpe": round(sharpe, 4),
        "ci_lower_90": round(ci_lower, 4),
        "ci_upper_90": round(ci_upper, 4),
        "dsr_pvalue": round(dsr_pvalue, 4),
        "dsr_stat": round(dsr_stat, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": 0.0,  # VRP strategies don't expose individual trades easily
        "n_trades": 0,
        "calmar_ratio": round(calmar, 4) if np.isfinite(calmar) else None,
        "annual_return_pct": round(annual_ret * 100, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "oos_bars": len(returns),
    }


def _deploy_gate(result: VRPResult) -> tuple[str, str]:
    """Evaluate deploy gate: PASS if CI_lower > 0 AND DSR p < 0.1."""
    if result.status == "error":
        return "FAIL", f"Error: {result.error}"

    if np.isnan(result.ci_lower_90) or np.isnan(result.dsr_pvalue):
        return "FAIL", "Missing metrics"

    reasons = []

    # Check CI lower bound
    if result.ci_lower_90 <= 0:
        reasons.append(f"CI lower {result.ci_lower_90:+.4f} <= 0 (no significant edge)")

    # Check DSR p-value
    if result.dsr_pvalue >= 0.1:
        reasons.append(f"DSR p={result.dsr_pvalue:.4f} >= 0.1 (multiple testing concern)")

    if not reasons:
        return "PASS", f"CI lower {result.ci_lower_90:+.4f} > 0, DSR p={result.dsr_pvalue:.4f} < 0.1"
    elif result.ci_lower_90 > 0:
        return "MONITOR", "; ".join(reasons)
    else:
        return "FAIL", "; ".join(reasons)


def run_validation(
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    n_trials: int = 20,
    n_bootstrap: int = 2000,
) -> list[VRPResult]:
    """Run DSR-adjusted validation on VRP strategies."""
    from quantcore.alpha.iron_condor import IronCondorVRP
    from quantcore.alpha.delta_neutral import DeltaNeutralVRP

    print("Loading SPY data...")
    close = _load_spy(start, end)
    print(f"Data: {len(close)} bars from {close.index[0].date()} to {close.index[-1].date()}")

    strategies = [
        ("Iron Condor VRP", IronCondorVRP, {
            "delta_short": 0.16,
            "delta_long": 0.10,
            "dte_entry_min": 30,
            "dte_entry_max": 45,
            "dte_exit": 21,
            "profit_target_pct": 0.50,
            "stop_loss_mult": 2.0,
            "risk_per_trade_pct": 0.02,
            "entry_interval_days": 30,
            "ivr_entry": 40.0,
            "use_enhanced_iv": True,
        }),
        ("Delta Neutral VRP", DeltaNeutralVRP, {
            "structure": "straddle",
            "dte_entry_min": 30,
            "dte_entry_max": 45,
            "dte_exit": 14,
            "profit_target_pct": 0.50,
            "risk_per_trade_pct": 0.02,
            "max_loss_mult": 3.0,
            "entry_interval_days": 30,
            "ivr_entry": 0.0,
            "use_enhanced_iv": True,
        }),
    ]

    results: list[VRPResult] = []

    for i, (name, cls, kwargs) in enumerate(strategies, 1):
        print(f"\n[{i}/{len(strategies)}] {name}...", flush=True)
        result = VRPResult(strategy_id=name)

        try:
            t0 = time.time()
            oos_returns, trades, wf_result = _run_walk_forward_vrp(
                cls, close, train_years=2, test_years=1, **kwargs,
            )
            elapsed = time.time() - t0

            if len(oos_returns) < 20:
                result.status = "error"
                result.error = f"insufficient OOS returns ({len(oos_returns)} bars)"
                print(f"  SKIP ({len(oos_returns)} bars, {elapsed:.1f}s)")
            else:
                metrics = _compute_dsr_metrics(oos_returns, n_trials, n_bootstrap)

                result.oos_sharpe = metrics.get("oos_sharpe", float("nan"))
                result.ci_lower_90 = metrics.get("ci_lower_90", float("nan"))
                result.ci_upper_90 = metrics.get("ci_upper_90", float("nan"))
                result.dsr_pvalue = metrics.get("dsr_pvalue", float("nan"))
                result.dsr_stat = metrics.get("dsr_stat", float("nan"))
                result.max_drawdown_pct = metrics.get("max_drawdown_pct", float("nan"))
                result.calmar_ratio = metrics.get("calmar_ratio", float("nan"))
                result.annual_return_pct = metrics.get("annual_return_pct", float("nan"))
                result.total_return_pct = metrics.get("total_return_pct", float("nan"))
                result.oos_bars = metrics.get("oos_bars", 0)
                result.status = "ok"

                print(f"  SR={result.oos_sharpe:+.4f}  CI_low={result.ci_lower_90:+.4f}  "
                      f"DSR_p={result.dsr_pvalue:.4f}  DD={result.max_drawdown_pct:.1f}%  "
                      f"({elapsed:.1f}s)")

        except Exception as e:
            result.status = "error"
            result.error = str(e)[:200]
            print(f"  ERROR: {e}")

        # Deploy gate
        result.deploy_gate, result.deploy_reason = _deploy_gate(result)
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DSR-adjusted VRP validation")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Conservative estimate of total strategies tested")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    t0 = time.time()
    results = run_validation(args.start, args.end, args.n_trials, args.n_bootstrap)
    elapsed = time.time() - t0

    # Print summary
    print("\n" + "=" * 70)
    print("  DSR-ADJUSTED VRP STRATEGY VALIDATION")
    print("=" * 70)

    for r in results:
        gate_emoji = {"PASS": "[OK]", "MONITOR": "[??]", "FAIL": "[X]"}.get(r.deploy_gate, "[?]")
        print(f"\n  {gate_emoji} {r.strategy_id}")
        if r.status == "ok":
            print(f"    OOS Sharpe:    {r.oos_sharpe:+.4f}")
            print(f"    CI Lower 90%:  {r.ci_lower_90:+.4f}")
            print(f"    CI Upper 90%:  {r.ci_upper_90:+.4f}")
            print(f"    DSR p-value:   {r.dsr_pvalue:.4f}")
            print(f"    DSR statistic: {r.dsr_stat:.4f}")
            print(f"    Max Drawdown:  {r.max_drawdown_pct:.1f}%")
            print(f"    Calmar Ratio:  {r.calmar_ratio}")
            print(f"    Annual Return: {r.annual_return_pct:.1f}%")
            print(f"    OOS Bars:      {r.oos_bars}")
        else:
            print(f"    Error: {r.error}")
        print(f"    Gate: {r.deploy_gate} — {r.deploy_reason}")

    print(f"\n  Completed in {elapsed:.0f}s")

    # Save report
    report_path = REPO_ROOT / "reports" / "tier1_dsr_adjusted.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# DSR-Adjusted VRP Strategy Validation",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Methodology",
        "",
        "- **Walk-forward**: Anchored expanding-window (2-year train, 1-year test)",
        "- **Sharpe CI**: Block bootstrap, 2000 replications, block length = n^(1/3)",
        "- **DSR**: Bailey & López de Prado (2012) formula with skewness/kurtosis",
        f"- **Trials**: {args.n_trials} (conservative: strategies tested + prior experiments)",
        "- **DEPLOY gate**: CI_lower > 0 AND DSR p-value < 0.1",
        "",
        "## Results",
        "",
        "| Strategy | OOS Sharpe | CI Lower | CI Upper | DSR p-val | DSR Stat | Max DD% | Calmar | Gate |",
        "|----------|-----------|----------|----------|-----------|----------|---------|--------|------|",
    ]

    for r in results:
        if r.status == "ok":
            gate_emoji = {"PASS": "PASS", "MONITOR": "MONITOR", "FAIL": "FAIL"}.get(r.deploy_gate, "?")
            lines.append(
                f"| {r.strategy_id} | {r.oos_sharpe:+.4f} | {r.ci_lower_90:+.4f} | "
                f"{r.ci_upper_90:+.4f} | {r.dsr_pvalue:.4f} | {r.dsr_stat:.4f} | "
                f"{r.max_drawdown_pct:.1f} | {r.calmar_ratio} | {gate_emoji} |"
            )
        else:
            lines.append(f"| {r.strategy_id} | ERROR | — | — | — | — | — | — | FAIL |")

    lines.extend([
        "",
        "## Gate Details",
        "",
    ])

    for r in results:
        gate_emoji = {"PASS": "[OK]", "MONITOR": "[??]", "FAIL": "[X]"}.get(r.deploy_gate, "[?]")
        lines.append(f"### {gate_emoji} {r.strategy_id}")
        lines.append("")
        lines.append(f"- **Gate**: {r.deploy_gate}")
        lines.append(f"- **Reason**: {r.deploy_reason}")
        if r.status == "ok":
            lines.append(f"- **Sharpe**: {r.oos_sharpe:+.4f}")
            lines.append(f"- **DSR p-value**: {r.dsr_pvalue:.4f}")
            lines.append(f"- **CI 90%**: [{r.ci_lower_90:+.4f}, {r.ci_upper_90:+.4f}]")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- **PASS**: Strategy has statistically significant edge after adjusting for multiple testing.",
        "  Safe to allocate capital.",
        "- **MONITOR**: Edge exists (CI > 0) but DSR p-value >= 0.1.",
        "  The Sharpe may be inflated by multiple testing. Continue paper trading",
        "  and gather more data before live allocation.",
        "- **FAIL**: No statistically significant edge detected.",
        "  Strategy should not be allocated capital.",
        "",
        "## Notes",
        "",
        "- The DSR adjusts for the fact that we tested multiple strategies.",
        "  A Sharpe of 2.0 with 20 trials is much less impressive than the same",
        "  Sharpe with 1 trial.",
        "- The skewness and kurtosis of returns are used in the B&LdP formula,",
        "  which is more accurate than the simplified normal approximation.",
        "- Strategies already in paper trading are marked MONITOR (not DISABLED)",
        "  when they fail the DSR gate, since they have live exposure.",
        "  This is informational for future allocation decisions.",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Save JSON
    json_path = REPO_ROOT / "reports" / "vrp_dsr_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "n_trials": args.n_trials,
            "strategies": [asdict(r) for r in results],
        }, f, indent=2, default=str)

    print(f"\n  Report: {report_path}")
    print(f"  JSON:   {json_path}")


if __name__ == "__main__":
    main()
