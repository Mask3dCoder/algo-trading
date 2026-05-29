#!/usr/bin/env python3
"""Iron Condor VRP — Real Options Data Validation.

Replaces the synthetic IV model with real options chain data and runs
the same walk-forward validation pipeline.

If real options data is unavailable (no API key / no cached data),
the script documents exactly what subscription is needed and produces
a synthetic-only baseline report.

Outputs:
    reports/iron_condor_real_data_validation.md

Usage:
    python scripts/iron_condor_real_data_validation.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────

TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"
DATA_START = "2021-06-01"

UNDERLYING = "SPY"
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_PER_CONTRACT = 0.65
SLIPPAGE_PCT = 0.05

N_TRIALS_DSR = 13


# ── Data availability check ─────────────────────────────────────────────────


def check_real_data_availability() -> dict:
    """Check whether real options data can be loaded.

    Returns a dict with keys:
        available: bool
        source: str — "polygon", "alpaca", "cached", or "none"
        api_key_set: bool
        cached_dates: int
        message: str
    """
    data_dir = PROJECT_ROOT / "data" / "options" / UNDERLYING

    # Check for cached parquet files first
    cached_count = 0
    if data_dir.exists():
        cached_count = len(list(data_dir.glob("*.parquet")))

    if cached_count >= 100:
        return {
            "available": True,
            "source": "cached",
            "api_key_set": False,
            "cached_dates": cached_count,
            "message": f"Found {cached_count} cached parquet files",
        }

    # Check for API keys
    polygon_key = os.environ.get("POLYGON_API_KEY", "")
    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    alpaca_secret = os.environ.get("ALPACA_API_SECRET", "")

    if polygon_key:
        return {
            "available": True,
            "source": "polygon",
            "api_key_set": True,
            "cached_dates": cached_count,
            "message": "Polygon API key set — will download on demand",
        }

    if alpaca_key and alpaca_secret:
        return {
            "available": True,
            "source": "alpaca",
            "api_key_set": True,
            "cached_dates": cached_count,
            "message": "Alpaca API keys set — will download on demand",
        }

    return {
        "available": False,
        "source": "none",
        "api_key_set": False,
        "cached_dates": cached_count,
        "message": (
            "No API keys set and no cached data found.\n"
            "Set POLYGON_API_KEY or ALPACA_API_KEY + ALPACA_API_SECRET.\n"
            f"Cached files found: {cached_count}"
        ),
    }


def build_options_loader(source: str):
    """Build the appropriate options data loader."""
    if source == "polygon":
        from quantcore.data.options_loader import PolygonOptionsDataLoader
        return PolygonOptionsDataLoader()
    elif source == "alpaca":
        from quantcore.data.options_loader import OptionsDataLoader
        return OptionsDataLoader()
    elif source == "cached":
        # Use whichever loader — both read the same parquet format
        from quantcore.data.options_loader import OptionsDataLoader
        return OptionsDataLoader()
    else:
        return None


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_spy_ohlcv(start: str, end: str) -> pd.Series:
    """Load SPY daily close prices via yfinance."""
    import yfinance as yf

    _log.info("Downloading SPY OHLCV (%s to %s)...", start, end)
    ticker = yf.Ticker(UNDERLYING)
    hist = ticker.history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No data returned for {UNDERLYING}")
    close = hist["Close"]
    close.index = pd.DatetimeIndex([d.date() for d in close.index])
    _log.info("Loaded %d trading days", len(close))
    return close


# ── Run backtest ─────────────────────────────────────────────────────────────


def run_validation(
    close: pd.Series,
    mode: str,
    options_loader=None,
) -> dict:
    """Run walk-forward validation.

    Parameters
    ----------
    mode : str
        "synthetic" or "real"
    options_loader : optional
        OptionsDataLoader instance for real data mode.

    Returns
    -------
    dict with keys: sharpe, ci_lower, max_dd, win_rate, dsr_pvalue, dsr_stat,
                    n_trades, n_days, fold_results, oos_returns
    """
    from quantcore.alpha.iron_condor import IronCondorVRP, generate_synthetic_iv
    from quantcore.research.walkforward import WalkForwardBacktester

    if mode == "synthetic":
        iv = generate_synthetic_iv(close)
        _log.info("Running synthetic validation...")
        strategy = IronCondorVRP(
            close=close,
            iv=iv,
            delta_short=0.16,
            spread_width=5.0,
            dte_entry_min=30,
            dte_entry_max=45,
            dte_exit=21,
            profit_target_pct=0.50,
            stop_loss_mult=2.0,
            risk_per_trade_pct=0.02,
            entry_interval_days=30,
            ivr_entry=40.0,
            risk_free_rate=0.04,
        )
    else:
        _log.info("Running real-data validation...")
        strategy = IronCondorVRP(
            close=close,
            options_loader=options_loader,
            underlying=UNDERLYING,
            delta_short=0.16,
            delta_long=0.10,
            spread_width=5.0,
            dte_entry_min=30,
            dte_entry_max=45,
            dte_exit=21,
            profit_target_pct=0.50,
            stop_loss_mult=2.0,
            risk_per_trade_pct=0.02,
            entry_interval_days=30,
            ivr_entry=40.0,
            risk_free_rate=0.04,
        )

    dates = close.index
    wf = WalkForwardBacktester(strategy, train_years=2, test_years=1)
    result = wf.run(dates=dates, n_trials=N_TRIALS_DSR)

    oos_returns = result["catalogues_returns"]
    fold_results = result["fold_results"]

    # Count total trades from folds
    n_trades = 0
    for fold in fold_results:
        # The fold doesn't directly expose trades, but we can count from returns
        pass

    return {
        "sharpe": result["annualised_sharpe"],
        "ci_lower": result["sharpe_ci_90_lower"],
        "max_dd": result["max_drawdown"],
        "win_rate": float(np.mean(oos_returns > 0)) if len(oos_returns) > 0 else 0.0,
        "dsr_pvalue": result["deflated_sharpe_pvalue"],
        "dsr_stat": result["deflated_sharpe_stat"],
        "n_days": len(oos_returns),
        "fold_results": fold_results,
        "oos_returns": oos_returns,
    }


# ── Report generation ────────────────────────────────────────────────────────


def generate_report(
    synthetic_results: dict,
    real_results: dict | None,
    data_status: dict,
    close: pd.Series,
) -> str:
    """Generate the markdown validation report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    s = synthetic_results
    pass_ci_s = s["ci_lower"] > 0
    pass_dsr_s = s["dsr_pvalue"] < 0.1
    verdict_s = "PASS" if (pass_ci_s and pass_dsr_s) else "FAIL"

    report = f"""# Iron Condor VRP — Real Options Data Validation Report

Generated: {now}

## Strategy

**Iron Condor VRP** — Sells OTM put spreads and OTM call spreads on SPY
to capture the Volatility Risk Premium (VRP).

- **Entry**: Every 30 days when IV Rank > 40%
- **Short strikes**: 16-delta (put and call)
- **Long strikes**: 10-delta (put and call) — real data mode
- **Spread width**: 5 points per wing — synthetic mode
- **Expiration**: 30-45 DTE
- **Profit target**: 50% of max credit
- **Time exit**: 21 DTE
- **Stop-loss**: Underlying touches short strike (2× credit)
- **Position sizing**: 2% of equity per trade

## Data Sources

### Synthetic (Baseline)
- **Underlying**: {UNDERLYING} — yfinance OHLCV
- **IV**: Synthetic BSM (15% VRP premium, mean-reverting to 20%)
- **Period**: {TRAIN_START} to {TEST_END}

### Real Options Data
- **Status**: {data_status['message']}
- **Source**: {data_status['source']}
- **API key set**: {"YES" if data_status['api_key_set'] else "NO"}
- **Cached parquet files**: {data_status['cached_dates']}

## Costs

- **Commission**: ${COMMISSION_PER_CONTRACT:.2f} per contract (4 legs × 2 sides)
- **Slippage**: {"Bid/ask spread (realistic fills)" if real_results else f"{SLIPPAGE_PCT*100:.0f}% of credit (synthetic mode)"}

## Walk-Forward Results

### Synthetic Data (Baseline)

| Metric | Value |
|--------|-------|
| OOS Sharpe (annualized) | {s['sharpe']:.4f} |
| 90% CI Lower Bound | {s['ci_lower']:.4f} |
| Max Drawdown | {s['max_dd']*100:.2f}% |
| Win Rate (positive days) | {s['win_rate']*100:.1f}% |
| DSR p-value (trials={N_TRIALS_DSR}) | {s['dsr_pvalue']:.6f} |
| DSR statistic | {s['dsr_stat']:.4f} |
| OOS trading days | {s['n_days']} |

"""

    if real_results is not None:
        r = real_results
        pass_ci_r = r["ci_lower"] > 0
        pass_dsr_r = r["dsr_pvalue"] < 0.1
        verdict_r = "PASS" if (pass_ci_r and pass_dsr_r) else "FAIL"

        report += f"""### Real Options Data

| Metric | Value |
|--------|-------|
| OOS Sharpe (annualized) | {r['sharpe']:.4f} |
| 90% CI Lower Bound | {r['ci_lower']:.4f} |
| Max Drawdown | {r['max_dd']*100:.2f}% |
| Win Rate (positive days) | {r['win_rate']*100:.1f}% |
| DSR p-value (trials={N_TRIALS_DSR}) | {r['dsr_pvalue']:.6f} |
| DSR statistic | {r['dsr_stat']:.4f} |
| OOS trading days | {r['n_days']} |

### Comparison

| Metric | Synthetic | Real Data | Delta |
|--------|-----------|-----------|-------|
| OOS Sharpe | {s['sharpe']:.4f} | {r['sharpe']:.4f} | {r['sharpe'] - s['sharpe']:+.4f} |
| CI(90%) Lower | {s['ci_lower']:.4f} | {r['ci_lower']:.4f} | {r['ci_lower'] - s['ci_lower']:+.4f} |
| Max Drawdown | {s['max_dd']*100:.2f}% | {r['max_dd']*100:.2f}% | {(r['max_dd'] - s['max_dd'])*100:+.2f}% |
| Win Rate | {s['win_rate']*100:.1f}% | {r['win_rate']*100:.1f}% | {(r['win_rate'] - s['win_rate'])*100:+.1f}% |
| DSR p-value | {s['dsr_pvalue']:.6f} | {r['dsr_pvalue']:.6f} | {r['dsr_pvalue'] - s['dsr_pvalue']:+.6f} |

### Acceptance Criteria — Real Data

| Criterion | Threshold | Actual | Pass? |
|-----------|-----------|--------|-------|
| CI(90%) lower > 0 | > 0 | {r['ci_lower']:.4f} | {"YES" if pass_ci_r else "NO"} |
| DSR p-value < 0.1 | < 0.1 | {r['dsr_pvalue']:.6f} | {"YES" if pass_dsr_r else "NO"} |

### Verdict: **{verdict_r}** (Real Data)

"""
        if verdict_r == "FAIL":
            report += """**Real-data verdict is FAIL.** The VRP edge is real but this Iron Condor
implementation may not capture it with sufficient edge after accounting for:

1. **Bid/ask spread cost**: Real fills cost more than BSM mid-price estimates.
   Short-dated OTM options on SPY often have 5-15 cent wide spreads.
2. **Delta selection noise**: Real chains have discrete strikes; the closest
   available delta may not match the target exactly.
3. **Expiry matching**: Real data has specific expiry dates (usually monthly
   or weekly), not the continuous 30-45 DTE range the synthetic model uses.

**Diagnosis**: If the synthetic result passes but real data fails, the problem
is likely the **Iron Condor implementation** (too much cost drag), not the
VRP itself.  The Volatility Risk Premium is well-documented — the question
is whether this specific options structure captures it after realistic costs.

**Next test**: Delta Neutral strategy (sell single options with delta-hedging)
which has lower structural cost and more flexibility in strike/expiry selection.
"""
    else:
        report += f"""### Real Options Data

**Not available.** {data_status['message']}

## Acceptance Criteria — Synthetic Only

| Criterion | Threshold | Actual | Pass? |
|-----------|-----------|--------|-------|
| CI(90%) lower > 0 | > 0 | {s['ci_lower']:.4f} | {"YES" if pass_ci_s else "NO"} |
| DSR p-value < 0.1 | < 0.1 | {s['dsr_pvalue']:.6f} | {"YES" if pass_dsr_s else "NO"} |

## Verdict: **{verdict_s}** (Synthetic Only — NOT TRUSTWORTHY)

"""

    # Always include the data requirements section when real data is unavailable
    if real_results is None:
        report += """## Required Data Subscription

To complete the real-data validation, **one** of the following is needed:

### Option 1: Polygon.io (Recommended)

- **Plan**: Options Starter ($29/month) or Options Pro ($199/month)
- **API key**: Export `POLYGON_API_KEY` environment variable
- **Coverage**: Full US options chain history with greeks, IV, bid/ask, OI
- **Rate limits**: Starter = 5 req/min, Pro = 100 req/min
- **Historical depth**: Full chain snapshots from 2023-01-01 onwards

```bash
export POLYGON_API_KEY="your_key_here"
python scripts/iron_condor_real_data_validation.py
```

### Option 2: Alpaca Markets

- **Plan**: Free tier available (limited) or paid ($9/month)
- **API keys**: Export `ALPACA_API_KEY` and `ALPACA_API_SECRET`
- **Coverage**: Options chain snapshots (current, not full historical)
- **Limitation**: Free tier returns current chain only, not historical snapshots
  — daily snapshots must be collected over time (not useful for backtesting
  past data unless previously cached)

```bash
export ALPACA_API_KEY="your_key_here"
export ALPACA_API_SECRET="your_secret_here"
python scripts/iron_condor_real_data_validation.py
```

### Option 3: Cached Data

If you have previously downloaded options chain data via either loader,
the parquet files at `data/options/SPY/*.parquet` will be detected automatically.
Each file should contain columns: `date, expiration, contract_type, strike,
bid, ask, implied_volatility, delta, gamma, theta, vega, open_interest`.

## Recommendation

**Polygon.io Options Starter** is the most cost-effective path. At $29/month
it provides the full historical options chain needed for backtesting.  The
free Alpaca tier is useful for live trading but not for historical backtests
of this kind.

Once data is available, this script will automatically run both the synthetic
and real-data comparisons and produce a definitive PASS/FAIL verdict.
"""

    return report


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    _log.info("=" * 60)
    _log.info("Iron Condor VRP — Real Options Data Validation")
    _log.info("=" * 60)

    # 1. Check data availability
    data_status = check_real_data_availability()
    _log.info("Data status: %s", data_status["message"])

    # 2. Load SPY OHLCV
    close = load_spy_ohlcv(DATA_START, TEST_END)

    # 3. Run synthetic baseline
    _log.info("")
    _log.info("─" * 40)
    _log.info("PHASE 1: Synthetic Baseline")
    _log.info("─" * 40)
    synthetic_results = run_validation(close, mode="synthetic")

    _log.info("")
    _log.info("Synthetic Results:")
    _log.info("  OOS Sharpe:    %.4f", synthetic_results["sharpe"])
    _log.info("  CI(90%%) Lower: %.4f", synthetic_results["ci_lower"])
    _log.info("  Max DD:        %.2f%%", synthetic_results["max_dd"] * 100)
    _log.info("  DSR p-value:   %.6f", synthetic_results["dsr_pvalue"])

    # 4. Run real-data validation if available
    real_results = None
    if data_status["available"]:
        _log.info("")
        _log.info("─" * 40)
        _log.info("PHASE 2: Real Options Data")
        _log.info("─" * 40)

        loader = build_options_loader(data_status["source"])

        # If using API (not cached), download first
        if data_status["source"] in ("polygon", "alpaca"):
            _log.info("Downloading options data for %s (%s to %s)...",
                       UNDERLYING, TEST_START, TEST_END)
            try:
                loader.download(UNDERLYING, start=DATA_START, end=TEST_END)
            except Exception as e:
                _log.error("Failed to download options data: %s", e)
                _log.info("Falling back to synthetic-only report")
                data_status["available"] = False
                data_status["message"] = f"Download failed: {e}"

        if data_status["available"]:
            try:
                real_results = run_validation(
                    close, mode="real", options_loader=loader,
                )
                _log.info("")
                _log.info("Real Data Results:")
                _log.info("  OOS Sharpe:    %.4f", real_results["sharpe"])
                _log.info("  CI(90%%) Lower: %.4f", real_results["ci_lower"])
                _log.info("  Max DD:        %.2f%%", real_results["max_dd"] * 100)
                _log.info("  DSR p-value:   %.6f", real_results["dsr_pvalue"])
            except Exception as e:
                _log.error("Real data validation failed: %s", e)
                import traceback
                traceback.print_exc()
                data_status["message"] = f"Validation failed: {e}"

    # 5. Generate report
    report = generate_report(synthetic_results, real_results, data_status, close)
    report_path = REPORTS_DIR / "iron_condor_real_data_validation.md"
    report_path.write_text(report)
    _log.info("")
    _log.info("Report written to %s", report_path)

    # 6. Print verdict
    _log.info("")
    _log.info("=" * 60)

    pass_ci_s = synthetic_results["ci_lower"] > 0
    pass_dsr_s = synthetic_results["dsr_pvalue"] < 0.1
    verdict_s = "PASS" if (pass_ci_s and pass_dsr_s) else "FAIL"

    if real_results is not None:
        pass_ci_r = real_results["ci_lower"] > 0
        pass_dsr_r = real_results["dsr_pvalue"] < 0.1
        verdict_r = "PASS" if (pass_ci_r and pass_dsr_r) else "FAIL"
        _log.info("VERDICT (Synthetic): %s", verdict_s)
        _log.info("VERDICT (Real Data): %s", verdict_r)
        if verdict_r == "FAIL" and verdict_s == "PASS":
            _log.info("")
            _log.info("DIAGNOSIS: The VRP works in theory but the Iron Condor")
            _log.info("implementation fails with real-world costs.")
            _log.info("→ Delta Neutral is the next test.")
    else:
        _log.info("VERDICT (Synthetic): %s — NOT TRUSTWORTHY", verdict_s)
        _log.info("")
        _log.info("Real data not available. Set POLYGON_API_KEY to proceed.")
        _log.info("See report for full details on required data subscription.")

    _log.info("=" * 60)
    _log.info("Done.")


if __name__ == "__main__":
    main()
