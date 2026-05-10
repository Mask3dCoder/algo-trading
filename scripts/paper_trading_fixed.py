"""Phase 3.1: Paper Trading System (Fixed).

Deploy the validated alignment-based multi-asset strategy into paper trading.
FROZEN strategy logic - no modifications to signal generation.

Fixes:
- Load longer data period (not just 2024)
- Proper signal generation for each universe
- Working backtest-equivalent logic
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

PAPER_CONFIG = {
    'assets': {
        'CTA': ['DBC', 'EFA', 'GLD', 'SPY', 'TLT'],  # Exclude FXE
    'RiskParity': ['DBC', 'EFA', 'GLD', 'IEF', 'SPY', 'TLT'],
        'Equity': ['SPY', 'EFA'],
        'Bonds': ['TLT', 'IEF'],
        'Commodities': ['GLD', 'DBC'],
    },
    'strategy_params': {
        'alignment_filter': True,
        'min_signal_strength': 0.0,  # No strength filter
    },
    'execution': {
        'commission_bps': 10,
        'slippage_bps': 5,
        'initial_capital': 1_000_000,
    },
    'benchmark': {
        'backtest_sharpe': 0.5037,
        'backtest_max_dd': -2.72,
        'expected_trades': 66,  # From Phase 3.0
    }
}


# =============================================================================
# Paper Trading Engine (Simplified - matches backtest logic)
# =============================================================================

class PaperTradingEngine:
    """Simulates paper trading with frozen strategy logic."""

    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config['execution']['initial_capital']
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

        # Strategy instances (one per asset universe)
        self.strategies: Dict[str, any] = {}
        self._init_strategies()

        # Data cache for each universe
        self.price_history: Dict[str, Dict[str, List[float]]] = {}
        for universe_name in config['assets'].keys():
            self.price_history[universe_name] = {}

    def _init_strategies(self):
        """Initialize strategy instances for each asset universe."""
        strategy_cls = get_strategy_class("macro_cta_trend")

        for universe_name, symbols in self.config['assets'].items():
            config = StrategyConfig(
                name=f"paper_{universe_name}",
                strategy_type=StrategyType.MACRO,
                params=self.config['strategy_params'],
                max_position_notional=self.initial_capital,
            )
            self.strategies[universe_name] = strategy_cls(config)

    def process_bar(self, timestamp: datetime, bar_data: Dict[str, float]):
        """Process a single bar of market data."""
        trade_pnl_by_bar = 0.0

        # Update price history for each universe
        for universe_name, symbols in self.config['assets'].items():
            for symbol in symbols:
                if symbol in bar_data:
                    if symbol not in self.price_history[universe_name]:
                        self.price_history[universe_name][symbol] = []
                    self.price_history[universe_name][symbol].append(bar_data[symbol])

        # Generate signals for each universe
        for universe_name, symbols in self.config['assets'].items():
            signals = self._generate_signals(universe_name)
            
            for signal in signals:
                symbol = signal.symbol
                direction = signal.direction
                price = bar_data.get(symbol, None)
                
                if price is None:
                    continue

                # Simulate execution
                result = self._execute_trade(symbol, direction, price, timestamp, universe_name)
                if result['executed']:
                    trade_pnl_by_bar += result['pnl']

        # Calculate daily return
        if len(self.daily_equity) > 0:
            prev_equity = self.daily_equity[-1]
            daily_pnl = self.equity - prev_equity
            if prev_equity > 0:
                ret = daily_pnl / prev_equity
                self.daily_returns.append(ret)

        self.daily_equity.append(self.equity)

        # Update peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

    def _generate_signals(self, universe: str) -> List[any]:
        """Generate trading signals using the frozen strategy."""
        if universe not in self.strategies:
            return []

        strategy = self.strategies[universe]
        symbols = self.config['assets'][universe]

        # Check if we have enough data
        histories = [self.price_history[universe].get(s, []) for s in symbols]
        if any(len(h) < 300 for h in histories):  # Warm-up
            return []

        # Build close array (n_periods, n_assets)
        min_len = min(len(h) for h in histories)
        close_prices = np.column_stack([
            h[-min_len:] for h in histories
        ])

        data = {
            "close": close_prices,
            "symbols": symbols,
        }

        # Set timestamp and generate signals
        strategy.set_timestamp(len(self.daily_equity))
        signals = strategy.on_data(data)

        return signals

    def _execute_trade(self, symbol: str, direction: int, price: float, 
                       timestamp: datetime, universe: str) -> dict:
        """Simulate trade execution with costs and slippage."""
        executed = False
        pnl = 0.0
        cost = 0.0

        if symbol in self.positions:
            old_pos = self.positions[symbol]

            # Check if direction changed
            if old_pos['direction'] != direction:
                # Close old position
                old_price = old_pos['entry_price']
                old_direction = old_pos['direction']
                size = old_pos['size']

                # Calculate PnL
                if old_direction == 1:  # Was long
                    pnl = (price - old_price) / old_price * size
                else:  # Was short
                    pnl = (old_price - price) / old_price * size

                # Apply costs
                cost = size * (self.config['execution']['commission_bps'] + 
                             self.config['execution']['slippage_bps']) / 10000
                pnl -= cost

                trade = {
                    'symbol': symbol,
                    'entry_time': old_pos['entry_time'],
                    'exit_time': timestamp,
                    'direction': direction,
                    'entry_price': old_price,
                    'exit_price': price,
                    'size': size,
                    'pnl': pnl,
                    'cost': cost,
                    'universe': universe,
                }
                self.trades.append(trade)
                self.trade_count += 1
                executed = True
                self.equity += pnl

                # Open new position
                self.positions[symbol] = {
                    'direction': direction,
                    'entry_price': price,
                    'entry_time': timestamp,
                    'size': size,
                }
        else:
            # New position
            size = self.initial_capital / 100  # Approximate size
            self.positions[symbol] = {
                'direction': direction,
                'entry_price': price,
                'entry_time': timestamp,
                'size': size,
            }

        return {'executed': executed, 'pnl': pnl, 'cost': cost}

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

def run_paper_trading_simulation():
    """Run paper trading simulation using historical data."""
    print("=" * 70)
    print("PHASE 3.1: PAPER TRADING SYSTEM (FIXED)")
    print("=" * 70)

    print("\nFROZEN STRATEGY CONFIGURATION:")
    print(f"  Alignment Filter: {PAPER_CONFIG['strategy_params']['alignment_filter']}")
    print(f"  Strength Filter: {PAPER_CONFIG['strategy_params']['min_signal_strength']}")
    print(f"  Assets: {list(PAPER_CONFIG['assets'].keys())}")

    # Load data - use longer period for warm-up
    print("\n" + "=" * 70)
    print("LOADING MARKET DATA")
    print("=" * 70)

    # Load CTA universe (has most symbols)
    df_cta = load_universe('cta', '2023-01-01', '2024-12-31', use_cache=True)
    
    # Load RP universe for IEF
    df_rp = load_universe('rp', '2023-01-01', '2024-12-31', use_cache=True)
    
    # Combine
    df = pd.concat([df_cta, df_rp]).drop_duplicates(subset=['symbol', 'timestamp'])
    df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

    # Pivot to wide format
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close').dropna()
    
    print(f"Data shape: {wide_df.shape}")
    print(f"Date range: {wide_df.index[0].date()} to {wide_df.index[-1].date()}")
    print(f"Symbols: {list(wide_df.columns)}")

    # Initialize paper trading engine
    engine = PaperTradingEngine(PAPER_CONFIG)

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
        if i % 50 == 0 or i == n_bars - 1:
            metrics = engine.get_performance_metrics()
            print(f"  Bar {i}/{n_bars}: Equity=${metrics['equity']:,.0f}, "
                  f"Sharpe={metrics['sharpe']:.4f}, Trades={metrics['trades']}")

    # Final results
    print("\n" + "=" * 70)
    print("PAPER TRADING RESULTS")
    print("=" * 70)

    live_metrics = engine.get_performance_metrics()

    print(f"\nLIVE PERFORMANCE:")
    print(f"  Sharpe: {live_metrics['sharpe']:.4f} (Backtest: {PAPER_CONFIG['benchmark']['backtest_sharpe']:.4f})")
    print(f"  Total Return: {live_metrics['total_return']:.2%}")
    print(f"  Max Drawdown: {live_metrics['max_drawdown']:.2%} (Backtest: {PAPER_CONFIG['benchmark']['backtest_max_dd']:.2%})")
    print(f"  Volatility: {live_metrics['volatility']:.2%}")
    print(f"  Trades: {live_metrics['trades']}")
    print(f"  Final Equity: ${live_metrics['equity']:,.0f}")

    # Comparison to backtest
    print(f"\nCOMPARISON TO BACKTEST:")
    sharpe_ratio_val = live_metrics['sharpe'] / PAPER_CONFIG['benchmark']['backtest_sharpe'] if PAPER_CONFIG['benchmark']['backtest_sharpe'] > 0 else 0
    print(f"  Sharpe Ratio (Live/Backtest): {sharpe_ratio_val:.2f}")
    
    # Trade breakdown by universe
    print(f"\nTRADE BREAKDOWN BY UNIVERSE:")
    trade_by_universe = {}
    for trade in engine.trades:
        universe = trade.get('universe', 'unknown')
        if universe not in trade_by_universe:
            trade_by_universe[universe] = []
        trade_by_universe[universe].append(trade)

    for universe, trades in trade_by_universe.items():
        pnls = [t['pnl'] for t in trades]
        avg_pnl = np.mean(pnls) if pnls else 0
        print(f"  {universe}: {len(trades)} trades, Avg PnL={avg_pnl:.2f}")

    # Decision Framework
    print("\n" + "=" * 70)
    print("DECISION FRAMEWORK")
    print("=" * 70)

    sharpe_ok = live_metrics['sharpe'] >= 0.3
    dd_ok = live_metrics['max_drawdown'] <= abs(PAPER_CONFIG['benchmark']['backtest_max_dd']) * 1.3
    trades_ok = abs(live_metrics['trades'] - PAPER_CONFIG['benchmark']['expected_trades']) / max(PAPER_CONFIG['benchmark']['expected_trades'], 1) <= 0.5

    print(f"\n* Live Sharpe >= 0.3: {sharpe_ok} ({live_metrics['sharpe']:.4f})")
    print(f"* Drawdown within 130% of backtest: {dd_ok} ({live_metrics['max_drawdown']:.2%})")
    print(f"* Trade count reasonable: {trades_ok} ({live_metrics['trades']} vs {PAPER_CONFIG['benchmark']['expected_trades']})")

    if sharpe_ok and dd_ok and trades_ok:
        print("\n>>> READY FOR LIVE CAPITAL")
        print("    - Live performance matches backtest within tolerance")
        print("    - Stable Sharpe in acceptable range")
        print("    - No structural issues")
    elif sharpe_ok and not dd_ok:
        print("\n>>> NEEDS ADJUSTMENT")
        print("    - Sharpe acceptable but drawdown too high")
    else:
        print("\n>>> NOT READY")
        print("    - Large divergence vs backtest")
        print("    - Pause and diagnose before proceeding")

    return engine


if __name__ == "__main__":
    engine = run_paper_trading_simulation()
