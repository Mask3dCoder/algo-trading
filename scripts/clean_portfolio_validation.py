"""Phase 3.2: Clean Portfolio Re-Validation (Pre-Live Gate).

Validate the robustness of the paper-trading system after removing
the underperforming asset (EFA), ensuring the portfolio is stable,
consistent, and safe for live capital deployment.

CRITICAL:
- Remove EFA entirely (avg PnL was -1135.42 in Phase 3.1)
- Keep ALL other assets unchanged
- Do NOT modify signal logic
- Do NOT tune parameters
- Only evaluate structural stability after EFA removal
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
# Configuration (FROZEN - Do NOT modify signal logic)
# =============================================================================

# Clean portfolio: REMOVE EFA (underperforming)
LEAN_SYMBOLS = ['DBC', 'GLD', 'IEF', 'SPY', 'TLT']

STRATEGY_PARAMS = {
    'alignment_filter': True,
    'min_signal_strength': 0.0,  # No strength filter
}

EXECUTION_CONFIG = {
    'commission_bps': 10,
    'slippage_bps': 5,
    'initial_capital': 1_000_000,
}

# Benchmark from Phase 3.1 (with EFA)
PHASE_3_1_RESULTS = {
    'sharpe': 0.3486,
    'total_return': 7.11,
    'max_dd': 2.14,
    'volatility': 2.94,
    'trades': 56,
    'equity': 1_071_079,
}

# Expected after EFA removal (estimated)
EXPECTED_RESULTS = {
    'sharpe': 0.40,  # Should improve (EFA was losing money)
    'max_dd': 2.0,  # Should improve
    'trades': 45,  # ~15% fewer trades (EFA had 11 trades)
}


# =============================================================================
# Paper Trading Engine (FROZEN - Same as Phase 3.1)
# =============================================================================

class PaperTradingEngine:
    """Simulates paper trading with frozen strategy logic."""

    def __init__(self, config: dict, strategy_params: dict, symbols: List[str]):
        self.config = config
        self.symbols = symbols
        self.initial_capital = config['initial_capital']
        self.equity = self.initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.daily_returns: List[float] = []
        self.daily_equity: List[float] = [self.initial_capital]
        self.trade_count = 0
        
        # Performance tracking
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        
        # Strategy (ONE instance with ALL symbols)
        strategy_cls = get_strategy_class("macro_cta_trend")
        self.strategy = strategy_cls(StrategyConfig(
            name="paper_trading_clean",
            strategy_type=StrategyType.MACRO,
            params=strategy_params,
            max_position_notional=self.initial_capital,
        ))
        
        # Data cache
        self.price_history: Dict[str, List[float]] = {}
        self.symbol_to_idx: Dict[str, int] = {s: i for i, s in enumerate(symbols)}

    def update_market_data(self, symbol: str, price: float):
        """Update price history."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)

    def generate_signals(self) -> List[any]:
        """Generate signals using the frozen strategy."""
        # Check if we have enough data
        histories = [self.price_history.get(s, []) for s in self.symbols]
        
        if any(len(h) < 300 for h in histories):  # Warm-up
            return []

        # Build close array
        min_len = min(len(h) for h in histories)
        close_prices = np.column_stack([
            h[-min_len:] for h in histories
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
        commission = size * (self.config['commission_bps'] / 10000)
        slippage = size * (self.config['slippage_bps'] / 10000)
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
            if symbol in self.symbols:  # Only track our symbols
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
        total_return = (self.equity / self.initial_capital - 1) * 100

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
# Main Paper Trading Simulation (Clean Portfolio)
# =============================================================================

def run_clean_portfolio_simulation():
    """Run paper trading simulation with EFA removed."""
    print("=" * 70)
    print("PHASE 3.2: CLEAN PORTFOLIO RE-VALIDATION")
    print("=" * 70)

    print("\nCLEAN PORTFOLIO CONFIGURATION:")
    print(f"  Symbols (EFA REMOVED): {LEAN_SYMBOLS}")
    print(f"  Alignment Filter: {STRATEGY_PARAMS['alignment_filter']}")
    print(f"  Strength Filter: {STRATEGY_PARAMS['min_signal_strength']}")
    print(f"  Commission: {EXECUTION_CONFIG['commission_bps']} bps")
    print(f"  Slippage: {EXECUTION_CONFIG['slippage_bps']} bps")

    # Load data - use FULL period (2018-2024) to match backtest
    print("\n" + "=" * 70)
    print("LOADING MARKET DATA (2018-2024)")
    print("=" * 70)

    # Load CTA universe (has most of our symbols)
    df_cta = load_universe('cta', '2018-01-01', '2024-12-31', use_cache=True)
    
    # Load RP universe for IEF
    df_rp = load_universe('rp', '2018-01-01', '2024-12-31', use_cache=True)
    
    # Combine and filter to our symbols
    df = pd.concat([df_cta, df_rp]).drop_duplicates(subset=['symbol', 'timestamp'])
    df = df[df['symbol'].isin(LEAN_SYMBOLS)]

    # Pivot to wide format
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close').dropna()

    print(f"Data shape: {wide_df.shape}")
    print(f"Date range: {wide_df.index[0].date()} to {wide_df.index[-1].date()}")
    print(f"Symbols: {list(wide_df.columns)}")

    # Initialize paper trading engine
    engine = PaperTradingEngine(EXECUTION_CONFIG, STRATEGY_PARAMS, LEAN_SYMBOLS)

    # Run simulation bar by bar
    print("\n" + "=" * 70)
    print("RUNNING CLEAN PORTFOLIO SIMULATION")
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
    print("CLEAN PORTFOLIO RESULTS (Phase 3.2)")
    print("=" * 70)

    clean_metrics = engine.get_performance_metrics()

    print(f"\nCLEAN PORTFOLIO PERFORMANCE:")
    print(f"  Sharpe: {clean_metrics['sharpe']:.4f} (Phase 3.1: {PHASE_3_1_RESULTS['sharpe']:.4f})")
    print(f"  Total Return: {clean_metrics['total_return']:.2f}% (Phase 3.1: {PHASE_3_1_RESULTS['total_return']:.2f}%)")
    print(f"  Max Drawdown: {clean_metrics['max_drawdown']:.2%} (Phase 3.1: {PHASE_3_1_RESULTS['max_dd']:.2f}%)")
    print(f"  Volatility: {clean_metrics['volatility']:.2f}% (Phase 3.1: {PHASE_3_1_RESULTS['volatility']:.2f}%)")
    print(f"  Trades: {clean_metrics['trades']} (Phase 3.1: {PHASE_3_1_RESULTS['trades']})")
    print(f"  Final Equity: ${clean_metrics['equity']:,.0f} (Phase 3.1: ${PHASE_3_1_RESULTS['equity']:,.0f})")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE: Phase 3.1 vs Phase 3.2")
    print("=" * 70)

    header = f"{'Metric':<20} {'Phase 3.1':>12} {'Phase 3.2':>12} {'Change':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    metrics_labels = [
        ('Sharpe', 'sharpe', True),  # True = higher is better
        ('Total Return %', 'total_return', True),
        ('Max DD %', 'max_drawdown', False),  # False = lower is better
        ('Volatility %', 'volatility', False),
        ('Trades', 'trades', None),  # None = just show
        ('Final Equity $', 'equity', True),
    ]

    for label, key, higher_better in metrics_labels:
        v1 = PHASE_3_1_RESULTS[key]
        v2 = clean_metrics[key]
        
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if key == 'equity':
                change_pct = (v2 / v1 - 1) * 100 if v1 > 0 else 0
            else:
                change_pct = v2 - v1
            
            if higher_better is True:
                improved = "✅" if v2 > v1 else "⚠️"
            elif higher_better is False:
                improved = "✅" if v2 < v1 else "⚠️"
            else:
                improved = ""
                
            print(f"{label:<20} {v1:>12.2f} {v2:>12.2f} {change_pct:>9.2f} {improved}")
        else:
            print(f"{label:<20} {str(v1):>12} {str(v2):>12} {'':>10}")

    # Trade breakdown by symbol
    print(f"\nTRADE BREAKDOWN BY SYMBOL (Phase 3.2):")
    trade_by_symbol = {}
    for trade in engine.trades:
        sym = trade['symbol']
        if sym not in trade_by_symbol:
            trade_by_symbol[sym] = []
        trade_by_symbol[sym].append(trade)

    for sym in sorted(trade_by_symbol.keys()):
        trades = trade_by_symbol[sym]
        pnls = [t['pnl'] for t in trades]
        avg_pnl = np.mean(pnls) if pnls else 0
        print(f"  {sym}: {len(trades)} trades, Avg PnL={avg_pnl:.2f}")

    # Decision Framework
    print("\n" + "=" * 70)
    print("DECISION FRAMEWORK (Pre-Live Gate)")
    print("=" * 70)

    # Check criteria
    sharpe_ok = clean_metrics['sharpe'] >= 0.30
    sharpe_improved = clean_metrics['sharpe'] > PHASE_3_1_RESULTS['sharpe']
    dd_ok = clean_metrics['max_drawdown'] <= PHASE_3_1_RESULTS['max_dd'] * 1.3
    return_ok = clean_metrics['total_return'] >= PHASE_3_1_RESULTS['total_return'] * 0.8
    trades_reasonable = 30 <= clean_metrics['trades'] <= 80

    print(f"\n* Sharpe >= 0.30: {sharpe_ok} ({clean_metrics['sharpe']:.4f})")
    print(f"* Sharpe improved vs Phase 3.1: {sharpe_improved} ({clean_metrics['sharpe']:.4f} vs {PHASE_3_1_RESULTS['sharpe']:.4f})")
    print(f"* Drawdown within 130% of Phase 3.1: {dd_ok} ({clean_metrics['max_drawdown']:.2%} vs {PHASE_3_1_RESULTS['max_dd']:.2%})")
    print(f"* Return >= 80% of Phase 3.1: {return_ok} ({clean_metrics['total_return']:.2f}% vs {PHASE_3_1_RESULTS['total_return']:.2f}%)")
    print(f"* Trade count reasonable (30-80): {trades_reasonable} ({clean_metrics['trades']})")

    all_ok = sharpe_ok and sharpe_improved and dd_ok and return_ok and trades_reasonable

    if all_ok:
        print("\n>>> ✅ READY FOR LIVE DEPLOYMENT!")
        print("    - Clean portfolio Sharpe improved vs Phase 3.1")
        print("    - Drawdown stable or improved")
        print("    - No structural issues after removing EFA")
        print("    - Portfolio is stable and ready for live capital")
    elif sharpe_ok and not sharpe_improved:
        print("\n>>> ⚠️ MIXED RESULTS")
        print("    - Sharpe acceptable but not improved")
        print("    - Consider live deployment with reduced size")
    else:
        print("\n>>> ❌ NOT READY")
        print("    - Portfolio degraded after removing EFA")
        print("    - Investigate before proceeding to live")

    return clean_metrics, engine


if __name__ == "__main__":
    clean_metrics, engine = run_clean_portfolio_simulation()
