"""Phase 4: Batch backtest for cross-sectional momentum - FINAL VERSION.

Properly marks portfolio to market.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import numpy as np
import pandas as pd
from scipy.special import softmax

from quantcore.data.yfinance_loader import load_combined

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Config
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
INITIAL_CASH = 1_000_000.0
SLIPPAGE_BPS = 5.0
COMMISSION_PER_SHARE = 0.005
REBALANCE_FREQ = 21  # days
MOM_WINDOWS = [20, 60, 120]
MOM_WEIGHTS = [0.2, 0.5, 0.3]
MIN_WEIGHT = 0.01


def load_data():
    """Load data for Phase 3 universe."""
    from quantcore.data.sources.universe import get_current_universe
    symbols = get_current_universe("PHASE3_UNIVERSE")
    log.info(f"Loading {len(symbols)} symbols...")

    df = load_combined(
        symbols=symbols,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        use_cache=True,
    )

    if df.empty:
        log.error("No data loaded")
        return None

    # Pivot to wide format: (n_periods, n_assets)
    close_pivot = df.pivot_table(index="timestamp", columns="symbol", values="close")
    close_pivot = close_pivot.dropna(axis=1, thresh=len(close_pivot)*0.9)

    log.info(f"Data shape: {close_pivot.shape}")
    return close_pivot.values, close_pivot.columns.tolist()


def compute_momentum_scores(close: np.ndarray) -> np.ndarray:
    """Compute cross-sectional momentum scores."""
    n_periods, n_assets = close.shape
    scores = np.zeros_like(close)

    for window, weight in zip(MOM_WINDOWS, MOM_WEIGHTS):
        if n_periods < window + 2:
            continue

        # Return over window for each asset
        ret = (close[window:] - close[:-window]) / np.maximum(close[:-window], 1e-10)

        # Cross-sectional z-score for each time period
        for t in range(len(ret)):
            mean_r = np.mean(ret[t])
            std_r = np.std(ret[t], ddof=1)
            if std_r > 1e-10:
                scores[t + window, :] += weight * ((ret[t] - mean_r) / std_r)
            else:
                scores[t + window, :] += 0.0

    return scores


def run_backtest(close: np.ndarray, symbols: list) -> tuple[np.ndarray, dict]:
    """Run backtest with proper portfolio simulation."""

    n_periods, n_assets = close.shape
    scores = compute_momentum_scores(close)

    # Initialize: track SHARES, not dollar values
    cash = INITIAL_CASH
    shares = np.zeros(n_assets, dtype=float)  # Number of shares per asset
    portfolio_values = [INITIAL_CASH]

    last_rebalance = 0

    for t in range(120, n_periods):  # Start after longest window
        # Check if we should rebalance
        if t - last_rebalance < REBALANCE_FREQ:
            # Mark to market
            port_val = cash + np.sum(shares * close[t, :])
            portfolio_values.append(port_val)
            continue

        # Close existing positions
        for i in range(n_assets):
            if shares[i] != 0:
                price = close[t, i]
                commission = abs(shares[i]) * COMMISSION_PER_SHARE
                slippage = price * SLIPPAGE_BPS / 10000.0

                if shares[i] > 0:  # Long - selling
                    cash += shares[i] * (price - slippage) - commission
                else:  # Short - buying back
                    cash -= abs(shares[i]) * (price + slippage) + commission
                shares[i] = 0.0

        # Compute new weights
        score_t = scores[t, :]
        vol = np.std(close[max(0, t-20):t+1, :], axis=0, ddof=1)
        vol_rank = np.argsort(np.argsort(vol)).astype(float) / (n_assets - 1)

        # Adjust by volatility
        adjusted = score_t / (vol_rank + 0.5)
        weights = softmax(adjusted) - np.mean(softmax(adjusted))

        # Normalize
        max_abs = np.max(np.abs(weights))
        if max_abs > 1e-10:
            weights = weights / max_abs

        # Filter small weights
        weights[np.abs(weights) < MIN_WEIGHT] = 0.0

        # Check if we have enough signals
        if np.sum(weights > 0) == 0 or np.sum(weights < 0) == 0:
            port_val = cash + np.sum(shares * close[t, :])
            portfolio_values.append(port_val)
            continue

        # Open new positions (dollar-neutral)
        port_val = cash + np.sum(shares * close[t, :])
        target_value = weights * port_val  # Dollar value per asset

        for i in range(n_assets):
            if abs(weights[i]) < MIN_WEIGHT:
                continue
            price = close[t, i]
            new_shares = target_value[i] / price

            commission = abs(new_shares) * COMMISSION_PER_SHARE
            slippage = price * SLIPPAGE_BPS / 10000.0

            if new_shares > 0:  # Long - buying
                cash -= new_shares * (price + slippage) + commission
            else:  # Short - selling
                cash += abs(new_shares) * (price - slippage) - commission

            shares[i] = new_shares

        last_rebalance = t

        # Mark to market
        port_val = cash + np.sum(shares * close[t, :])
        portfolio_values.append(port_val)

    equity = np.array(portfolio_values)
    returns = (equity[1:] / equity[:-1]) - 1.

    # Calculate metrics
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-10 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    max_dd = abs(np.min(dd)) if len(dd) > 0 else 0.0

    total_ret = (equity[-1] / equity[0]) - 1.

    metrics = {
        "sharpe": round(sharpe, 4),
        "max_dd_pct": round(max_dd, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "final_equity": round(equity[-1], 2),
    }

    return equity, metrics


def main():
    log.info("Starting Phase 4 Batch Backtest: Cross-Sectional Momentum (FINAL)")

    # Load data
    result = load_data()
    if result is None:
        return
    close, symbols = result

    # Run backtest
    equity, metrics = run_backtest(close, symbols)

    # Print results
    print("\n" + "="*80)
    print("PHASE 4 BACKTEST RESULTS: CROSS-SECTIONAL MOMENTUM (FINAL)")
    print("="*80)
    print(f"Strategy: Cross-Sectional Momentum (20/60/120d)")
    print(f"Universe: Phase 3 Expanded ({len(symbols)} assets)")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Transaction Costs: {SLIPPAGE_BPS}bps slippage, ${COMMISSION_PER_SHARE}/share commission")
    print("-"*80)
    print(f"Out-of-Sample Sharpe: {metrics['sharpe']:.4f}")
    print(f"Max Drawdown: {metrics['max_dd_pct']:.2f}%")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Final Equity: ${metrics['final_equity']:,.2f}")
    print("-"*80)

    # Acceptance criteria
    if metrics["sharpe"] >= 0.5 and metrics["max_dd_pct"] <= 20.0:
        print("Acceptance Decision: ACCEPT")
        print("\nREASON: Strategy meets all acceptance criteria:")
        print(f"  - OOS Sharpe {metrics['sharpe']:.4f} >= 0.5")
        print(f"  - Max DD {metrics['max_dd_pct']:.2f}% <= 20%")
    else:
        print("Acceptance Decision: REJECT")
        if metrics["sharpe"] < 0.5:
            print(f"  Reason: OOS Sharpe {metrics['sharpe']:.4f} < 0.5")
        if metrics["max_dd_pct"] > 20.0:
            print(f"  Reason: Max DD {metrics['max_dd_pct']:.2f}% > 20%")
    print("="*80)


if __name__ == "__main__":
    main()
