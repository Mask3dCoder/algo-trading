"""Simple test: Buy-and-hold SPY to verify backtest engine works."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import numpy as np
import pandas as pd

from quantcore.backtest.engine import BacktestEngine
from quantcore.strategies.macro.simple_test_strategy import SimpleBuyHoldStrategy
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.data.yfinance_loader import load_combined

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"


def main():
    # Load only SPY data
    log.info("Loading SPY data...")
    df = load_combined(
        symbols=["SPY"],
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        use_cache=True,
    )
    
    if df.empty:
        log.error("No data loaded")
        return
    
    log.info(f"Loaded {len(df)} bars for SPY")
    
    # Initialize strategy
    config = StrategyConfig(
        name="simple_buy_hold",
        strategy_type=StrategyType.MACRO,
        max_position_notional=1_000_000.0,
    )
    strategy = SimpleBuyHoldStrategy(config=config)
    
    # Run backtest
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=1_000_000.0,
        slippage_bps=0.0,  # No slippage for this test
        commission_per_share=0.0,  # No commission for this test
    )
    
    result = engine.run(df)
    
    if len(engine.portfolio.equity_curve) < 2:
        log.error("No equity curve generated")
        return
    
    # Calculate Sharpe
    timestamps, values = zip(*engine.portfolio.equity_curve)
    equity = np.array(values)
    returns = (equity[1:] / equity[:-1]) - 1
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-10 else 0.0
    
    total_ret = (equity[-1] / equity[0]) - 1
    
    print("\n" + "="*60)
    print("SIMPLE BUY-AND-HOLD SPY TEST")
    print("="*60)
    print(f"Total Return: {total_ret*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Final Equity: ${equity[-1]:,.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
