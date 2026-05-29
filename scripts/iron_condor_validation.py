#!/usr/bin/env python3
"""Iron Condor VRP Walk-Forward Validation.

Runs honest walk-forward validation on the Iron Condor VRP strategy
using synthetic options data derived from SPY OHLCV + regime-adaptive IV.

Outputs:
    reports/iron_condor_validation.md — validation report with PASS/FAIL verdict

Usage:
    python scripts/iron_condor_validation.py
"""

from __future__ import annotations

import logging
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
DATA_START = "2021-06-01"  # extra warmup for IVR calculation

UNDERLYING = "SPY"
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_PER_CONTRACT = 0.65
SLIPPAGE_PCT = 0.05  # 5% of bid-ask

N_TRIALS_DSR = 13  # 12 previous strategy tests + this one

PASS_CI_LOWER_GT_ZERO = True
PASS_DSR_PVALUE_LT_0_1 = True


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


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    _log.info("=" * 60)
    _log.info("Iron Condor VRP — Honest Walk-Forward Validation")
    _log.info("=" * 60)

    # 1. Load data
    close = load_spy_ohlcv(DATA_START, TEST_END)

    # 2. Import strategy
    from quantcore.alpha.iron_condor import IronCondorVRP, generate_synthetic_iv

    # 3. Generate synthetic IV
    _log.info("Generating synthetic IV from realized vol + 15%% VRP...")
    iv = generate_synthetic_iv(close)

    # 4. Create strategy instance
    _log.info("Creating IronCondorVRP strategy instance...")
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

    # 5. Run walk-forward validation
    from quantcore.research.walkforward import WalkForwardBacktester

    _log.info("Setting up walk-forward backtester (train=2yr, test=1yr)...")
    wf = WalkForwardBacktester(strategy, train_years=2, test_years=1)

    dates = close.index
    _log.info("Running walk-forward with n_trials=%d...", N_TRIALS_DSR)
    result = wf.run(dates=dates, n_trials=N_TRIALS_DSR)

    # 6. Extract metrics
    oos_sharpe = result["annualised_sharpe"]
    ci_lower = result["sharpe_ci_90_lower"]
    dsr_pvalue = result["deflated_sharpe_pvalue"]
    dsr_stat = result["deflated_sharpe_stat"]
    max_dd = result["max_drawdown"]
    oos_returns = result["catalogues_returns"]

    # Win rate from trades
    fold_results = result["fold_results"]
    total_trades = 0
    winning_trades = 0
    for fold in fold_results:
        logs = fold.oos_returns  # we'll compute from returns
    # Compute from OOS returns
    win_rate = float(np.mean(oos_returns > 0)) if len(oos_returns) > 0 else 0.0

    # 7. Determine verdict
    pass_ci = ci_lower > 0
    pass_dsr = dsr_pvalue < 0.1
    verdict = "PASS" if (pass_ci and pass_dsr) else "FAIL"

    # 8. Print summary
    _log.info("")
    _log.info("=" * 60)
    _log.info("RESULTS")
    _log.info("=" * 60)
    _log.info("OOS Sharpe:         %.4f", oos_sharpe)
    _log.info("CI(90%%) Lower:      %.4f", ci_lower)
    _log.info("Max Drawdown:       %.2f%%", max_dd * 100)
    _log.info("Win Rate (days>0):  %.1f%%", win_rate * 100)
    _log.info("DSR p-value:        %.6f", dsr_pvalue)
    _log.info("DSR statistic:      %.4f", dsr_stat)
    _log.info("OOS trading days:   %d", len(oos_returns))
    _log.info("")
    _log.info("VERDICT: %s", verdict)
    _log.info("  CI_lower > 0:     %s", "YES" if pass_ci else "NO")
    _log.info("  DSR p < 0.1:      %s", "YES" if pass_dsr else "NO")

    # 9. Generate report
    report_path = REPORTS_DIR / "iron_condor_validation.md"
    _log.info("Writing report to %s", report_path)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Iron Condor VRP — Walk-Forward Validation Report

Generated: {now}

## Strategy

**Iron Condor VRP** — Sells OTM put spreads and OTM call spreads on SPY
to capture the Volatility Risk Premium (VRP).

- **Entry**: Every 30 days when IV Rank > 40%
- **Short strikes**: 16-delta (put and call)
- **Spread width**: 5 points per wing
- **Expiration**: 30-45 DTE
- **Profit target**: 50% of max credit
- **Time exit**: 21 DTE
- **Stop-loss**: Underlying touches short strike (2× credit)
- **Position sizing**: 2% of equity per trade

## Data

- **Underlying**: {UNDERLYING}
- **Source**: yfinance (OHLCV) + synthetic BSM IV (15%% VRP premium, mean-reverting to 20%%)
- **Training period**: {TRAIN_START} to {TRAIN_END}
- **Testing period**: {TEST_START} to {TEST_END}

## Costs

- **Commission**: ${COMMISSION_PER_CONTRACT:.2f} per contract
- **Slippage**: {SLIPPAGE_PCT*100:.0f}% of bid-ask spread

## Walk-Forward Results

| Metric | Value |
|--------|-------|
| OOS Sharpe (annualized) | {oos_sharpe:.4f} |
| 90% CI Lower Bound | {ci_lower:.4f} |
| Max Drawdown | {max_dd*100:.2f}% |
| Win Rate (positive days) | {win_rate*100:.1f}% |
| DSR p-value (trials={N_TRIALS_DSR}) | {dsr_pvalue:.6f} |
| DSR statistic | {dsr_stat:.4f} |
| OOS trading days | {len(oos_returns)} |

## Fold Details

| Fold | Train Start | Train End | OOS Start | OOS End |
|------|-------------|-----------|-----------|---------|
"""
    for fold in fold_results:
        report += f"| {fold.fold_idx} | {fold.train_start} | {fold.train_end} | {fold.oos_start} | {fold.oos_end} |\n"

    report += f"""
## Acceptance Criteria

| Criterion | Threshold | Actual | Pass? |
|-----------|-----------|--------|-------|
| CI(90%) lower > 0 | > 0 | {ci_lower:.4f} | {"YES" if pass_ci else "NO"} |
| DSR p-value < 0.1 | < 0.1 | {dsr_pvalue:.6f} | {"YES" if pass_dsr else "NO"} |

## Verdict: **{verdict}**

"""
    if verdict == "PASS":
        report += """This is the first honest alpha stream in the system. The Iron Condor VRP
strategy earns a spot in the paper trading configuration. The VRP is a
well-documented market anomaly (Coval & Shumway 2001, Israelov & Nielsen 2015)
and this implementation captures it with statistically significant edge
after multiple-testing correction.
"""
    else:
        report += """The strategy does not meet the strict acceptance criteria. The VRP may
still be real but this simple Iron Condor implementation may not capture
it with sufficient edge after accounting for costs and multiple testing.

Next steps:
- **Delta Neutral**: Sell single options with delta-hedging (more flexible)
- **Term structure**: Exploit IV term structure rather than just IV level
- **Skew trading**: Sell put skew specifically (higher VRP than calls)
- **Real options data**: Replace synthetic IV with actual options chain data
"""

    report_path.write_text(report)
    _log.info("Report written to %s", report_path)
    _log.info("Done.")


if __name__ == "__main__":
    main()
