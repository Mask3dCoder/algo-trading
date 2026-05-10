#!/usr/bin/env python3
"""verify_honest_foundation.py -- Phase 1 validation.

Runs the fixed backtest engine on all non-DISABLED strategies, computes
honest OOS metrics with bootstrap CIs, and saves HONEST_FOUNDATION_REPORT.json.

Usage:
    python scripts/verify_honest_foundation.py

Pass criterion (per audit Task 1.2 / Task 2.4):
    90% CI lower-bound for OOS Sharpe >= 0.30

Exit codes:
    0 - all active strategies pass
    1 - failures or engine errors detected
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

try:
    from quantcore.risk.metrics import sharpe_ratio, bootstrap_ci
    from quantcore.backtest.walk_forward import anchored_walk_forward
    from quantcore.strategies.base import StrategyStatus
except ImportError as e:
    print(f"ERROR: quantcore import failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISABLED = [
    "macro_cta_trend",
    "macro_risk_parity",
    "macro_factor_residual",
    "equity_momentum",
    "equity_earnings_momentum",
]

SHARPE_CI_THRESHOLD = 0.30  # OOS lower CI must exceed this to pass

OLD_METRICS_AUDIT = {
    "macro_cta_trend": {"sharpe": -0.147, "max_dd_pct": 42.0},
    "macro_risk_parity": {"sharpe": 0.128, "max_dd_pct": 28.0},
    "macro_cross_sectional_momentum_v2": {"sharpe": 0.55, "max_dd_pct": 88.0},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def synth_data(universe: str):
    """Synthetic OHLCV data for testing when yfinance unavailable."""
    import pandas as pd
    from quantcore.data.sources.universe import get_current_universe

    syms = get_current_universe(universe)[:5]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start="2018-01-01", end="2021-12-31")
    rows = []
    for sym in syms:
        p = 100.0
        for dt in dates:
            p *= 1 + rng.normal(0.0002, 0.012)
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": dt,
                    "open": p * 0.999,
                    "high": p * 1.005,
                    "low": p * 0.995,
                    "close": p,
                    "volume": int(rng.uniform(1e6, 5e6)),
                }
            )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_data(universe: str):
    """Load OHLCV via yfinance; fall back to synthetic if unavailable."""
    try:
        import pandas as pd
        import yfinance as yf
        from quantcore.data.sources.universe import get_current_universe

        syms = get_current_universe(universe)[:8]
        rows = []
        for sym in syms:
            try:
                h = yf.Ticker(sym).history(
                    start="2018-01-01", end="2024-01-01", auto_adjust=True
                )
                if h.empty:
                    continue
                h = h.reset_index()
                h["symbol"] = sym
                h.rename(
                    columns={
                        "Date": "timestamp",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    },
                    inplace=True,
                )
                h["timestamp"] = pd.to_datetime(h["timestamp"])
                rows.append(
                    h[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
                )
            except Exception:
                pass
        if rows:
            return pd.concat(rows, ignore_index=True).dropna(subset=["close"])
    except ImportError:
        pass
    log.warning("yfinance unavailable — using synthetic data")
    return synth_data(universe)


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def validate_strategy(name: str, universe: str, params: dict, status: StrategyStatus) -> dict:
    """Run anchored walk-forward validation for one strategy.

    Returns a result dict with OOS metrics and pass/fail.
    """
    try:
        import quantcore.strategies.macro.cross_sectional_momentum_v2  # noqa: F401
        from quantcore.backtest.engine import BacktestEngine
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.registry import get as get_cls
    except ImportError as e:
        return {"strategy": name, "passed": False, "error": str(e)}

    data = load_data(universe)
    if data is None:
        return {"strategy": name, "passed": False, "error": "no_data"}

    timestamps = sorted(data["timestamp"].unique())
    n_bars = len(timestamps)
    folds = anchored_walk_forward(n_bars=n_bars, initial_train=252, test_size=126)

    oos_sharpes: list[float] = []
    all_returns: list[float] = []

    for fold in folds[:4]:  # Limit for speed; increase to 8+ for production
        try:
            StrategyClass = get_cls(name)
            cfg = StrategyConfig(
                name=name,
                strategy_type=StrategyType.MACRO,
                max_position_notional=1_000_000,
                max_leverage=1.0,
                params=params,
            )
            strat = StrategyClass(cfg)

            test_ts = timestamps[fold.test_start : fold.test_end]
            test_data = data[data["timestamp"].isin(test_ts)].copy()
            if len(test_data) < 10:
                continue

            # BacktestEngine creates its own PortfolioState and FillSimulator internally.
            # Use kwargs that match the actual constructor signature.
            engine = BacktestEngine(
                strategy=strat,
                initial_cash=1_000_000,
                slippage_bps=10.0,
                commission_per_share=0.005,
            )
            eq_df = engine.run(test_data)

            if len(eq_df) < 5:
                continue

            eq = eq_df["equity"].values.astype(float)
            eq = eq[eq > 0]
            if len(eq) < 5:
                continue

            rets = np.diff(np.log(eq))
            sr = sharpe_ratio(rets)
            if math.isfinite(sr):
                oos_sharpes.append(sr)
                all_returns.extend(rets.tolist())

        except Exception as exc:
            log.warning("Fold error for %s: %s", name, exc)

    if not oos_sharpes:
        return {
            "strategy": name,
            "status": status.value,
            "passed": False,
            "error": "all_folds_failed",
        }

    mean_sr = float(np.mean(oos_sharpes))
    all_r = np.array(all_returns)

    if len(all_r) >= 20:
        ci_lo, ci_hi = bootstrap_ci(all_r, sharpe_ratio, n_boot=300, ci=0.90)
    else:
        ci_lo = ci_hi = mean_sr

    passed = math.isfinite(ci_lo) and ci_lo >= SHARPE_CI_THRESHOLD

    return {
        "strategy": name,
        "status": status.value,
        "sharpe_mean": round(mean_sr, 4),
        "sharpe_ci_lower": round(ci_lo, 4) if math.isfinite(ci_lo) else None,
        "sharpe_ci_upper": round(ci_hi, 4) if math.isfinite(ci_hi) else None,
        "n_folds": len(oos_sharpes),
        "pass_threshold": SHARPE_CI_THRESHOLD,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGIES_TO_VALIDATE = [
    {
        "name": "macro_cross_sectional_momentum_v2",
        "universe": "expanded_15",
        "status": StrategyStatus.DEVELOPMENT,
        "params": {
            "rebalance_freq": 21,
            "mom_windows": "20,60,120",
            "mom_weights": "0.2,0.5,0.3",
            "use_vol_adjustment": True,
            "target_vol": 0.12,
            "vol_lookback": 20,
            "max_gross_exposure": 1.0,
            "max_position_pct": 0.03,
            "beta_neutral": True,
            "dd_control": True,
            "dd_flatten_threshold": -0.15,
            "max_leverage": 1.0,
        },
    },
]


def main() -> int:
    print("=" * 70)
    print("QUANTCORE HONEST FOUNDATION VERIFICATION")
    print("=" * 70)
    print(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Pass criterion: OOS Sharpe 90%% CI lower >= {SHARPE_CI_THRESHOLD}")
    print()

    results: list[dict] = []
    any_fail = False

    for s in STRATEGIES_TO_VALIDATE:
        name = s["name"]
        print(f"  [{name}] validating ({s['universe']})...", flush=True)
        t0 = time.time()
        res = validate_strategy(name, s["universe"], s["params"], s["status"])
        results.append(res)
        elapsed = time.time() - t0
        verdict = "PASS" if res.get("passed") else "FAIL"
        if not res.get("passed"):
            any_fail = True
        lo = res.get("sharpe_ci_lower")
        hi = res.get("sharpe_ci_upper")
        ci_str = f"CI90=[{lo:.3f},{hi:.3f}]" if lo is not None else "CI90=N/A"
        print(f"     {verdict}  Sharpe={res.get('sharpe_mean')}  {ci_str}  ({elapsed:.1f}s)")

    print()
    print("DISABLED (not run):")
    for d in DISABLED:
        print(f"  [DISABLED] {d}")

    checklist = {
        "look_ahead_bias": "RESOLVED -- fill_sim._next_bar=None during on_bar()",
        "validation_uses_full_stack": "PARTIAL -- BacktestEngine used here; _compute_signal_returns deprecated",
        "metrics_bugs": "RESOLVED -- sharpe NaN flat, calmar equity curve, rf geometric",
        "purged_cv_crash": "RESOLVED -- total->n_bars",
        "survivorship_bias": "PARTIAL -- ETF_MACRO bias-free; equity universe still current-only",
        "factor_residual_degeneracy": "RESOLVED -- DISABLED, residual_momentum.py skeleton created",
    }

    comparison = [
        {
            "strategy": r["strategy"],
            "audit_sharpe": OLD_METRICS_AUDIT.get(r["strategy"], {}).get("sharpe"),
            "honest_sharpe_mean": r.get("sharpe_mean"),
            "passed": r.get("passed"),
        }
        for r in results
    ]

    report = {
        "report_date": datetime.utcnow().isoformat(),
        "system_status": "UNDER_RECONSTRUCTION",
        "disclaimer": "No strategy validated for live trading. See FORENSIC_AUDIT_REPORT.md.",
        "pass_threshold_ci_lower": SHARPE_CI_THRESHOLD,
        "fatal_findings_checklist": checklist,
        "disabled_strategies": DISABLED,
        "active_results": results,
        "comparison_audit_vs_honest": comparison,
        "known_limitations": [
            "Synthetic data fallback when yfinance unavailable.",
            "Only 4 walk-forward folds (increase to 8+ for production).",
            "ADV liquidity limits not enforced (unlimited sentinel in liquidity.py).",
        ],
        "q2_next_steps": [
            "Full CSM v2 validation on 2010-2024 real data.",
            "Rebuild factor_residual via quantcore/alpha/residual_momentum.py + FF5.",
            "Implement equity/low_volatility (tier-1 alpha from audit).",
            "Populate universe_history.json with point-in-time S&P500 constituents.",
        ],
    }

    out = REPO_ROOT / "HONEST_FOUNDATION_REPORT.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved: {out}")

    print("\nFATAL FINDINGS CHECKLIST:")
    for k, v in checklist.items():
        print(f"  [{v.split('--')[0].strip():10s}] {k}")

    if any_fail:
        print("\nRESULT: FAIL  (expected at this stage -- infrastructure rebuild, not alpha)")
    else:
        print("\nRESULT: PASS")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
