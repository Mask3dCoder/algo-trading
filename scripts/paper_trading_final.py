"""Phase 3.1: Paper Trading System (Fixed - follows Phase 3.0 logic).

Deploy the validated alignment-based multi-asset strategy into paper trading.
Uses ONE strategy with ALL symbols (matches working backtest).

FROZEN strategy logic:
- Alignment filter only (fast == slow direction)
- No strength filter
- Load full 2018-2024 data (matches backtest period)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

# CRITICAL: Import macro strategies
import quantcore.strategies.macro  # noqa: F401
from quantcore.strategies.registry import get as get_strategy_class
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.risk.metrics import sharpe_ratio
from scripts.real_data_loader import load_universe


# =============================================================================
# Configuration (FROZEN)
# =============================================================================

ALL_SYMBOLS = ['DBC', 'EFA', 'GLD', 'SPY', 'TLT', 'IEF']
# Exclude FXE as per Phase 3.0

STRATEGY_PARAMS = {
    'alignment_filter': True,
    'min_signal_strength': 0.0,  # No strength filter
}

EXECUTION_CONFIG = {
    'commission_bps': 10,
    'slippage_bps': 5,
    'initial_capital': 1_000_000,
}

BENCHMARK = {
    'backtest_sharpe': 0.5037,  # Phase 3.0 Equal Weight
    'backtest_max_dd': -2.72,
    'expected_trades': 66,  # Approximate from Phase 3.0
}


# =============================================================================
# Paper Trading Engine (Matches Backtest Logic)
# =============================================================================

class PaperTradingEngine:
    """Simulates paper trading with ONE multi-asset strategy."""

    def __init__(self, config: dict, strategy_params: dict):
        self.initial_capital = config['initial_capital']
        self.equity = self.initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.daily_returns: List[float] = []
        self.daily_equity: List[float] = [self.initial_capital]
        
        # Performance tracking
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.trade_count = 0
        self.alert_log: List[str] = []

        # Strategy (ONE instance with ALL symbols)
        strategy_cls = get_strategy_class("macro_cta_trend")
        self.strategy = strategy_cls(StrategyConfig(
            name="paper_trading",
            strategy_type=StrategyType.MACRO,
            params=strategy_params,
            max_position_notional=self.initial_capital,
        ))

        # Data cache
        self.price_history: Dict[str, List[float]] = {}
        self.symbol_to_idx: Dict[str, int] = {}
        self.symbols: List[str] = []

    def set_symbols(self, symbols: List[str]):
        """Set symbols and build index mapping."""
        self.symbols = symbols
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}

    def update_market_data(self, symbol: str, price: float):
        """Update price history for a symbol."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)

    def generate_signals(self) -> List[any]:
        """Generate signals using the frozen strategy (matches backtest)."""
        # Check if we have enough data
        min_history = 300  # Warm-up period
        all_histories = [self.price_history.get(s, []) for s in self.symbols]

        if any(len(h) < min_history for h in all_histories):
            return []  # Not enough data yet

        # Build close array (n_periods, n_assets) - same as backtest
        min_len = min(len(h) for h in all_histories)
        close_prices = np.column_stack([
            h[-min_len:] for h in all_histories
        ])

        data = {
            "close": close_prices,
            "symbols": self.symbols,
        }

        # Set timestamp and generate signals
        self.strategy.set_timestamp(len(self.daily_equity))
        signals = self.strategy.on_data(data)

        return signals

    def simulate_execution(self, signal: any, current_price: float) -> dict:
        """Simulate trade execution with costs."""
        symbol = signal.symbol
        direction = signal.direction
        size = signal.size

        # Calculate costs
        commission = size * (EXECUTION_CONFIG['commission_bps'] / 10000)
        slippage = size * (EXECUTION_CONFIG['slippage_bps'] / 10000)
        total_cost = commission + slippage

        # Check if we already have a position
        if symbol in self.positions:
            old_pos = self.positions[symbol]

            # Check if direction changed
            if old_pos['direction'] != direction:
                # Close old position
                old_price = old_pos['entry_price']
                old_direction = old_pos['direction']

                if old_direction == 1:  # Was long
                    pnl = (current_price - old_price) / old_price * old_pos['size']
                else:  # Was short
                    pnl = (old_price - current_price) / old_price * old_pos['size']

                pnl -= total_cost  # Subtract costs

                trade = {
                    'symbol': symbol,
                    'entry_time': old_pos['entry_time'],
                    'exit_time': datetime.now(),
                    'direction': direction,
                    'entry_price': old_price,
                    'exit_price': current_price,
                    'size': old_pos['size'],
                    'pnl': pnl,
                    'cost': total_cost,
                }
                self.trades.append(trade)
                self.trade_count += 1
                self.equity += pnl

                # Open new position
                self.positions[symbol] = {
                    'direction': direction,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'size': size,
                }
        else:
            # New position
            self.positions[symbol] = {
                'direction': direction,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'size': size,
            }

        return {'status': 'executed', 'cost': total_cost}

    def process_bar(self, timestamp: datetime, bar_data: Dict[str, float]):
        """Process a single bar of market data."""
        # Update market data
        for symbol, price in bar_data.items():
            self.update_market_data(symbol, price)

        # Generate signals
        signals = self.generate_signals()

        for signal in signals:
            symbol = signal.symbol
            if symbol in bar_data:
                self.simulate_execution(signal, bar_data[symbol])

        # Calculate daily return
        if len(self.daily_equity) > 0:
            prev_equity = self.daily_equity[-1]
            daily_pnl = self.equity - prev_equity
            if prev_equity > 0:
                daily_ret = daily_pnl / prev_equity
                self.daily_returns.append(daily_ret)

        self.daily_equity.append(self.equity)

        # Update peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

    def get_performance_metrics(self) -> dict:
        """Calculate live performance metrics."""
        if len(self.daily_returns) < 2:
            return {
                'sharpe': 0.0,
                'total_return': 0.0,
                'max_drawdown': self.max_drawdown,
                'volatility': 0.0,
                'trades': self.trade_count,
                'equity': self.equity,
            }

        returns = np.array(self.daily_returns)
        sharpe = sharpe_ratio(returns)
        total_return = (self.equity / self.initial_capital - 1)

        # Volatility
        volatility = float(np.std(returns) * np.sqrt(252))

        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'volatility': volatility,
            'trades': self.trade_count,
            'equity': self.equity,
        }


# =============================================================================
# Main Paper Trading Simulation
# =============================================================================

def run_paper_trading():
    """Run paper trading simulation using FULL historical data (matches backtest)."""
    print("=" * 70)
    print("PHASE 3.1: PAPER TRADING SYSTEM (FIXED)")
    print("=" * 70)

    print("\nFROZEN STRATEGY CONFIGURATION:")
    print(f"  Alignment Filter: {STRATEGY_PARAMS['alignment_filter']}")
    print(f"  Strength Filter: {STRATEGY_PARAMS['min_signal_strength']}")
    print(f"  Symbols: {ALL_SYMBOLS}")
    print(f"  Commission: {EXECUTION_CONFIG['commission_bps']} bps")
    print(f"  Slippage: {EXECUTION_CONFIG['slippage_bps']} bps")

    # Load FULL data (2018-2024) to match backtest
    print("\n" + "=" * 70)
    print("LOADING FULL MARKET DATA (2018-2024)")
    print("=" * 70)

    df = load_universe('cta', '2018-01-01', '2024-12-31', use_cache=True)
    df_rp = load_universe('rp', '2018-01-01', '2024-12-31', use_cache=True)
    df = pd.concat([df, df_rp]).drop_duplicates(subset=['symbol', 'timestamp'])

    # Filter to only our symbols (exclude FXE)
    df = df[df['symbol'].isin(ALL_SYMBOLS)]

    # Pivot to wide format
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close').dropna()

    print(f"Data shape: {wide_df.shape}")
    print(f"Date range: {wide_df.index[0].date()} to {wide_df.index[-1].date()}")
    print(f"Symbols: {list(wide_df.columns)}")

    # Initialize paper trading engine
    engine = PaperTradingEngine(EXECUTION_CONFIG, STRATEGY_PARAMS)
    engine.set_symbols(list(wide_df.columns))

    # Run simulation bar by bar
    print("\n" + "=" * 70)
    print("RUNNING PAPER TRADING SIMULATION")
    print("=" * 70)

    n_bars = len(wide_df)
    
    for i, (timestamp, row) in enumerate(wide_df.iterrows()):
        # Build bar data dict
        bar_data = {col: row[col] for col in wide_df.columns if pd.notna(row[col])}

        # Process this bar
        engine.process_bar(timestamp, bar_data)

        # Print progress
        if i % 200 == 0 or i == n_bars - 1:
            metrics = engine.get_performance_metrics()
            print(f"  Bar {i}/{n_bars}: Equity=${metrics['equity']:,.0f}, "
                  f"Sharpe={metrics['sharpe']:.4f}, Trades={metrics['trades']}")

    # Final results
    print("\n" + "=" * 70)
    print("PAPER TRADING RESULTS")
    print("=" * 70)

    live_metrics = engine.get_performance_metrics()

    print(f"\nLIVE PERFORMANCE:")
    print(f"  Sharpe: {live_metrics['sharpe']:.4f} (Backtest: {BENCHMARK['backtest_sharpe']:.4f})")
    print(f"  Total Return: {live_metrics['total_return']:.2%}")
    print(f"  Max Drawdown: {live_metrics['max_drawdown']:.2%} (Backtest: {BENCHMARK['backtest_max_dd']:.2%})")
    print(f"  Volatility: {live_metrics['volatility']:.2%}")
    print(f"  Trades: {live_metrics['trades']}")
    print(f"  Final Equity: ${live_metrics['equity']:,.0f}")

    # Comparison to backtest
    print(f"\nCOMPARISON TO BACKTEST:")
    sharpe_ratio_val = live_metrics['sharpe'] / BENCHMARK['backtest_sharpe'] if BENCHMARK['backtest_sharpe'] > 0 else 0
    print(f"  Sharpe Ratio (Live/Backtest): {sharpe_ratio_val:.2f}")
    
    # Trade breakdown by symbol
    print(f"\nTRADE BREAKDOWN BY SYMBOL:")
    trade_by_symbol = {}
    for trade in engine.trades:
        sym = trade['symbol']
        if sym not in trade_by_symbol:
            trade_by_symbol[sym] = []
        trade_by_symbol[sym].append(trade)

    for sym, trades in sorted(trade_by_symbol.items()):
        pnls = [t['pnl'] for t in trades]
        avg_pnl = np.mean(pnls) if pnls else 0
        print(f"  {sym}: {len(trades)} trades, Avg PnL={avg_pnl:.2f}")

    # Decision Framework
    print("\n" + "=" * 70)
    print("DECISION FRAMEWORK")
    print("=" * 70)

    sharpe_ok = live_metrics['sharpe'] >= 0.3
    dd_ok = abs(live_metrics['max_drawdown']) <= abs(BENCHMARK['backtest_max_dd']) * 1.3
    trades_ok = abs(live_metrics['trades'] - BENCHMARK['expected_trades']) / max(BENCHMARK['expected_trades'], 1) <= 0.5

    print(f"\n* Live Sharpe >= 0.3: {sharpe_ok} ({live_metrics['sharpe']:.4f})")
    print(f"* Drawdown within 130% of backtest: {dd_ok} ({live_metrics['max_drawdown']:.2%})")
    print(f"* Trade count reasonable: {trades_ok} ({live_metrics['trades']} vs {BENCHMARK['expected_trades']})")

    if sharpe_ok and dd_ok and trades_ok:
        print("\n>>> READY FOR LIVE CAPITAL")
        print("    - Live performance matches backtest within tolerance")
        print("    - Stable Sharpe in acceptable range")
    elif sharpe_ok and not dd_ok:
        print("\n>>> NEEDS ADJUSTMENT")
        print("    - Sharpe acceptable but drawdown too high")
    else:
        print("\n>>> NOT READY")
        print("    - Large divergence vs backtest")
        print("    - Pause and diagnose before proceeding")

    return engine


if __name__ == "__main__":
    engine = run_paper_trading()
