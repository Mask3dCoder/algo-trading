"""Phase 4: Cross-Sectional Momentum Backtest with Fixed Validation.

Uses Phase 3 expanded universe, 2y/6m walk-forward, transaction costs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from datetime import date

from quantcore.backtest.walk_forward import (
    rolling_walk_forward, WFResult,
    check_acceptance_criteria, compute_oos_metrics,
)
from quantcore.backtest.engine import BacktestEngine
from quantcore.backtest.simulation import FixedSlippage, PerShareCommission
from quantcore.strategies.macro.cross_sectional_momentum import CrossSectionalMomentumStrategy
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.data.sources.universe import get_current_universe
from quantcore.data.yfinance_loader import load_combined

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
INITIAL_CASH = 1_000_000.0
SLIPPAGE_BPS = 5.0
COMMISSION_PER_SHARE = 0.005
REBALANCE_FREQ = 21  # Monthly

# Strategy config
STRATEGY_PARAMS = {
    "rebalance_freq": REBALANCE_FREQ,
    "mom_windows": "20,60,120",
    "mom_weights": "0.2,0.5,0.3",
    "use_vol_adjustment": True,
    "long_count": 0,  # Continuous weighting
    "short_count": 0,
    "min_weight": 0.01,
}

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
def load_universe_data(universe_name: str) -> pd.DataFrame:
    """Load adjusted close prices for universe."""
    symbols = get_current_universe(universe_name)
    log.info(f"Loading {len(symbols)} symbols for {universe_name}")
    
    # Load from yfinance (auto_adjust=True gives adjusted close)
    df = load_combined(
        symbols=symbols,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        use_cache=True,
    )
    
    # Filter to require min 252 bars per symbol
    valid_symbols = []
    for sym in symbols:
        sym_data = df[df["symbol"] == sym]
        if len(sym_data) >= 252:
            valid_symbols.append(sym)
    
    log.info(f"Valid symbols after filtering: {len(valid_symbols)}")
    return df[df["symbol"].isin(valid_symbols)]

# ---------------------------------------------------------------------------
# Run Walk-Forward Backtest
# ---------------------------------------------------------------------------
def run_walk_forward_backtest(df: pd.DataFrame) -> list[WFResult]:
    """Run rolling walk-forward validation."""
    # Pivot to get (n_periods, n_assets) close matrix
    close_pivot = df.pivot_table(index="timestamp", columns="symbol", values="close")
    close_pivot = close_pivot.dropna(axis=1, thresh=len(close_pivot)*0.9)  # Drop symbols with >10% NaN
    n_bars, n_assets = close_pivot.shape
    log.info(f"Data shape: {n_bars} bars, {n_assets} assets")
    
    # Generate folds (2y train, 6m test)
    folds = rolling_walk_forward(n_bars=n_bars)
    log.info(f"Generated {len(folds)} walk-forward folds")
    
    results: list[WFResult] = []
    
    for i, fold in enumerate(folds):
        log.info(f"Fold {i+1}/{len(folds)}: Train {fold.train_start}-{fold.train_end}, Test {fold.test_start}-{fold.test_end}")
        
        # Split train/test data
        train_df = df.iloc[fold.train_start:fold.train_end]
        test_df = df.iloc[fold.test_start:fold.test_end]
        
        if len(test_df) < 50:
            log.warning("Test set too small, skipping")
            continue
        
        # Initialize strategy
        config = StrategyConfig(
            name="macro_cross_sectional_momentum",
            strategy_type=StrategyType.MACRO,
            params=STRATEGY_PARAMS,
            max_position_notional=INITIAL_CASH * 0.02,  # 2% per asset
        )
        strategy = CrossSectionalMomentumStrategy(config=config)
        
        # Initialize backtest engine with transaction costs
        engine = BacktestEngine(
            strategy=strategy,
            initial_cash=INITIAL_CASH,
            slippage_bps=SLIPPAGE_BPS,
            commission_per_share=COMMISSION_PER_SHARE,
        )
        
        # Run backtest on test set
        test_result = engine.run(test_df)
        
        # Calculate OOS metrics
        if len(engine.portfolio.equity_curve) < 2:
            log.warning("No equity curve generated, skipping")
            continue
        
        timestamps, values = zip(*engine.portfolio.equity_curve)
        equity = np.array(values)
        returns = (equity[1:] / equity[:-1]) - 1
        
        metrics = compute_oos_metrics(returns, equity)
        
        result = WFResult(
            fold=fold,
            in_sample_sharpe=0.0,  # Not computed for rolling WF
            out_of_sample_sharpe=metrics["sharpe"],
            out_of_sample_returns=returns,
        )
        results.append(result)
        
        log.info(f"Fold {i+1} OOS Sharpe: {metrics['sharpe']:.4f}, Max DD: {metrics['max_dd_pct']:.2f}%")
    
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("Starting Phase 4 Cross-Sectional Momentum Backtest")
    
    # Load data
    df = load_universe_data("PHASE3_UNIVERSE")
    if df.empty:
        log.error("No data loaded, exiting")
        return
    
    # Run walk-forward
    results = run_walk_forward_backtest(df)
    
    if not results:
        log.error("No valid folds, exiting")
        return
    
    # Aggregate OOS results
    all_returns = np.concatenate([r.out_of_sample_returns for r in results if r.out_of_sample_returns is not None])
    all_equity = np.cumprod(1.0 + all_returns)
    
    # Compute aggregate metrics
    agg_metrics = compute_oos_metrics(all_returns, all_equity)
    periods_passed = sum(1 for r in results if r.out_of_sample_sharpe > 0.5)
    
    # Check acceptance criteria
    is_valid, reason = check_acceptance_criteria(
        oos_sharpe=agg_metrics["sharpe"],
        max_dd_pct=agg_metrics["max_dd_pct"],
        periods_passed=periods_passed,
        total_periods=len(results),
    )
    
    # Print results
    print("\n" + "="*80)
    print("PHASE 4 BACKTEST RESULTS: CROSS-SECTIONAL MOMENTUM")
    print("="*80)
    print(f"Strategy: Cross-Sectional Momentum (20/60/120d)")
    print(f"Universe: Phase 3 Expanded ({len(df['symbol'].unique())} assets)")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Transaction Costs: {SLIPPAGE_BPS}bps slippage, ${COMMISSION_PER_SHARE}/share commission")
    print("-"*80)
    print(f"Out-of-Sample Sharpe: {agg_metrics['sharpe']:.4f}")
    print(f"Max Drawdown: {agg_metrics['max_dd_pct']:.2f}%")
    print(f"Annual Return: {agg_metrics['annual_return']:.2f}%")
    print(f"Total Return: {agg_metrics['total_return_pct']:.2f}%")
    print(f"Periods Passed: {periods_passed}/{len(results)}")
    print("-"*80)
    print(f"Acceptance Decision: {'ACCEPT' if is_valid else 'REJECT'}")
    print(f"Reason: {reason}")
    print("="*80)
    
    # Save results
    result_df = pd.DataFrame([{
        "oos_sharpe": agg_metrics["sharpe"],
        "max_dd_pct": agg_metrics["max_dd_pct"],
        "annual_return": agg_metrics["annual_return"],
        "total_return": agg_metrics["total_return_pct"],
        "periods_passed": periods_passed,
        "total_periods": len(results),
        "is_valid": is_valid,
        "reason": reason,
    }])
    result_df.to_csv("phase4_results.csv", index=False)
    log.info("Results saved to phase4_results.csv")

if __name__ == "__main__":
    main()
