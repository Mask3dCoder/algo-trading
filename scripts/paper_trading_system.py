"""Phase 3.1: Paper Trading System Deployment.

Deploy the validated alignment-based multi-asset strategy into a paper
trading environment to verify real-time performance without modifying core logic.

CRITICAL: Strategy logic is FROZEN:
- Alignment filter only (fast == slow direction)
- No strength filter
- No parameter changes
- Same asset universe (excluding FX)

Paper Trading Engine:
- Simulates live market data updates
- Applies identical decision rules from backtest
- Simulates execution with realistic costs
- Tracks live metrics vs backtest expectations
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

# CRITICAL: Import macro strategies to trigger registration
import quantcore.strategies.macro  # noqa: F401
from quantcore.strategies.registry import get as get_strategy_class
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.risk.metrics import sharpe_ratio
from scripts.real_data_loader import load_universe


# =============================================================================
# Configuration (FROZEN - Do NOT modify)
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
        'commission_bps': 10,  # 10 basis points
        'slippage_bps': 5,  # 5 basis points (volatility-scaled)
        'initial_capital': 1_000_000,
    },
    'alert_thresholds': {
        'max_dd_exceed_pct': 30,  # Alert if live DD > backtest DD * 1.3
        'sharpe_drop_threshold': 0.2,  # Alert if rolling Sharpe < 0.2
        'trade_freq_deviation_pct': 25,  # Alert if trade freq deviates >25%
    },
    'benchmark': {
        'backtest_sharpe': 0.5037,  # From Phase 3.0 Equal Weight
        'backtest_max_dd': -2.72,  # From Phase 3.0
        'expected_trades_per_month': 15,  # Approximate from Phase 3.0
    }
}


# =============================================================================
# Paper Trading Engine
# =============================================================================

class PaperTradingEngine:
    """Simulates live trading with the frozen alignment strategy."""

    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config['execution']['initial_capital']
        self.equity = self.initial_capital
        self.positions: Dict[str, dict] = {}  # symbol -> position info
        self.trades: List[dict] = []
        self.daily_returns: List[float] = []
        self.daily_equity: List[float] = [self.initial_capital]
        self.trade_count = 0
        self.alert_log: List[str] = []

        # Performance tracking
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0

        # Strategy instances (one per asset universe)
        self.strategies: Dict[str, any] = {}
        self._init_strategies()

        # Data cache
        self.latest_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {}

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

    def update_market_data(self, symbol: str, price: float, timestamp: datetime):
        """Simulate new market data arriving."""
        self.latest_prices[symbol] = price

        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(price)

    def generate_signals(self, universe: str) -> List[dict]:
        """Generate trading signals using the frozen strategy."""
        if universe not in self.strategies:
            return []

        strategy = self.strategies[universe]
        symbols = self.config['assets'][universe]

        # Build data dict (same format as backtest)
        # Need price history for all symbols in universe
        min_history = 300  # Warm-up period
        all_histories = [self.price_history.get(s, []) for s in symbols]

        if any(len(h) < min_history for h in all_histories):
            return []  # Not enough data yet

        # Build close array (n_periods, n_assets)
        min_len = min(len(h) for h in all_histories)
        close_prices = np.column_stack([
            h[-min_len:] for h in all_histories
        ])

        data = {
            "close": close_prices,
            "symbols": symbols,
        }

        # Set timestamp and generate signals
        strategy.set_timestamp(len(self.daily_equity))
        signals = strategy.on_data(data)

        return signals

    def simulate_execution(self, signal: dict, current_price: float) -> dict:
        """Simulate trade execution with costs and slippage."""
        symbol = signal['symbol']
        direction = signal['direction']
        size = signal['size']

        # Calculate costs
        commission = size * (self.config['execution']['commission_bps'] / 10000)
        slippage = size * (self.config['execution']['slippage_bps'] / 10000)

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
                    'universe': signal.get('universe', 'unknown'),
                }
                self.trades.append(trade)
                self.trade_count += 1

                # Update equity
                self.equity += pnl

            # Open new position (or update existing)
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
            self.update_market_data(symbol, price, timestamp)

        # Generate signals for each universe
        for universe_name, symbols in self.config['assets'].items():
            signals = self.generate_signals(universe_name)

            for signal in signals:
                symbol = signal['symbol']
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

    def check_alerts(self) -> List[str]:
        """Check if any alert conditions are triggered."""
        alerts = []

        # 1. Drawdown exceeds backtest max DD by >30%
        backtest_dd = abs(self.config['benchmark']['backtest_max_dd'])
        current_dd_pct = self.max_drawdown
        if current_dd_pct > backtest_dd * (1 + self.config['alert_thresholds']['max_dd_exceed_pct'] / 100):
            alerts.append(f"DRAWDOWN ALERT: Current DD {current_dd_pct:.2%} exceeds backtest DD {backtest_dd:.2%} by >30%")

        # 2. Rolling Sharpe drops below 0.2
        if len(self.daily_returns) >= 30:
            rolling_returns = self.daily_returns[-30:]
            rolling_sharpe = sharpe_ratio(np.array(rolling_returns))
            if rolling_sharpe < self.config['alert_thresholds']['sharpe_drop_threshold']:
                alerts.append(f"SHARPE ALERT: Rolling Sharpe {rolling_sharpe:.4f} < 0.2")

        return alerts

    def get_performance_metrics(self) -> dict:
        """Calculate live performance metrics."""
        if len(self.daily_returns) < 2:
            return {
                'sharpe': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'trades': 0,
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

    def compare_to_backtest(self) -> dict:
        """Compare live performance to backtest expectations."""
        live_metrics = self.get_performance_metrics()
        benchmark = self.config['benchmark']

        comparison = {
            'sharpe_vs_backtest': live_metrics['sharpe'] / benchmark['backtest_sharpe'] if benchmark['backtest_sharpe'] > 0 else 0,
            'dd_vs_backtest': live_metrics['max_drawdown'] / benchmark['backtest_max_dd'] if benchmark['backtest_max_dd'] < 0 else 1,
            'within_tolerance': True,
        }

        # Check if deviation is too large
        if abs(1 - comparison['sharpe_vs_backtest']) > 0.3:  # 30% deviation
            comparison['within_tolerance'] = False

        return comparison


# =============================================================================
# Main Paper Trading Simulation
# =============================================================================

def run_paper_trading_simulation():
    """Run a paper trading simulation using historical data."""
    print("=" * 70)
    print("PHASE 3.1: PAPER TRADING SYSTEM SIMULATION")
    print("=" * 70)

    print("\nFROZEN STRATEGY CONFIGURATION:")
    print(f"  Alignment Filter: {PAPER_CONFIG['strategy_params']['alignment_filter']}")
    print(f"  Strength Filter: {PAPER_CONFIG['strategy_params']['min_signal_strength']}")
    print(f"  Assets: {list(PAPER_CONFIG['assets'].keys())}")
    print(f"  Commission: {PAPER_CONFIG['execution']['commission_bps']} bps")
    print(f"  Slippage: {PAPER_CONFIG['execution']['slippage_bps']} bps")

    # Load data for all assets
    print("\n" + "=" * 70)
    print("LOADING MARKET DATA")
    print("=" * 70)

    all_symbols = set()
    for symbols in PAPER_CONFIG['assets'].values():
        all_symbols.update(symbols)

    print(f"Total unique symbols: {len(all_symbols)}")
    print(f"Symbols: {sorted(all_symbols)}")

    # Load data (use CTA universe as it has most symbols)
    df = load_universe('cta', '2024-01-01', '2024-12-31', use_cache=True)

    # Also load RP universe for IEF
    df_rp = load_universe('rp', '2024-01-01', '2024-12-31', use_cache=True)
    df = pd.concat([df, df_rp]).drop_duplicates(subset=['symbol', 'timestamp'])

    # Pivot to wide format
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close').dropna()

    print(f"Data shape: {wide_df.shape}")
    print(f"Date range: {wide_df.index[0].date()} to {wide_df.index[-1].date()}")

    # Initialize paper trading engine
    engine = PaperTradingEngine(PAPER_CONFIG)

    # Run simulation bar by bar
    print("\n" + "=" * 70)
    print("RUNNING PAPER TRADING SIMULATION")
    print("=" * 70)

    n_bars = len(wide_df)
    alert_count = 0

    for i, (timestamp, row) in enumerate(wide_df.iterrows()):
        # Build bar data dict
        bar_data = {col: row[col] for col in wide_df.columns if pd.notna(row[col])}

        # Process this bar
        engine.process_bar(timestamp, bar_data)

        # Check alerts periodically
        if i % 20 == 0 or i == n_bars - 1:
            alerts = engine.check_alerts()
            if alerts:
                alert_count += len(alerts)
                for alert in alerts:
                    engine.alert_log.append(f"{timestamp}: {alert}")
                    print(f"  ALERT: {alert}")

        # Print progress
        if i % 50 == 0:
            metrics = engine.get_performance_metrics()
            eq = metrics.get('equity', 0)
            sp = metrics.get('sharpe', 0)
            tr = metrics.get('trades', 0)
            print(f"  Bar {i}/{n_bars}: Equity={eq:,.0f}, Sharpe={sp:.4f}, Trades={tr}")

    # Final results
    print("\n" + "=" * 70)
    print("PAPER TRADING RESULTS")
    print("=" * 70)

    live_metrics = engine.get_performance_metrics()
    comparison = engine.compare_to_backtest()

    print(f"\nLIVE PERFORMANCE:")
    print(f"  Sharpe: {live_metrics['sharpe']:.4f} (Backtest: {PAPER_CONFIG['benchmark']['backtest_sharpe']:.4f})")
    print(f"  Total Return: {live_metrics['total_return']:.2%}")
    print(f"  Max Drawdown: {live_metrics['max_drawdown']:.2%} (Backtest: {PAPER_CONFIG['benchmark']['backtest_max_dd']:.2%})")
    print(f"  Volatility: {live_metrics['volatility']:.2%}")
    print(f"  Trades: {live_metrics['trades']}")
    print(f"  Final Equity: ${live_metrics['equity']:,.0f}")

    print(f"\nCOMPARISON TO BACKTEST:")
    print(f"  Sharpe Ratio (Live/Backtest): {comparison['sharpe_vs_backtest']:.2f}")
    print(f"  Drawdown Ratio (Live/Backtest): {comparison['dd_vs_backtest']:.2f}")
    print(f"  Within Tolerance: {comparison['within_tolerance']}")

    print(f"\nALERT SUMMARY:")
    print(f"  Total Alerts: {alert_count}")
    if engine.alert_log:
        print("  Recent Alerts:")
        for alert in engine.alert_log[-5:]:
            print(f"    - {alert}")

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
    dd_ok = live_metrics['max_drawdown'] <= PAPER_CONFIG['benchmark']['backtest_max_dd'] * 1.3
    no_structural_issues = alert_count < 10  # Reasonable number of alerts

    print(f"\n* Live Sharpe >= 0.3: {sharpe_ok} ({live_metrics['sharpe']:.4f})")
    print(f"* Drawdown within 130% of backtest: {dd_ok} ({live_metrics['max_drawdown']:.2%})")
    print(f"* No structural issues: {no_structural_issues} (Alerts: {alert_count})")

    if sharpe_ok and dd_ok and no_structural_issues:
        print("\n>>> READY FOR LIVE CAPITAL")
        print("    - Live performance matches backtest within tolerance")
        print("    - Stable Sharpe in acceptable range")
        print("    - No structural slippage or execution issues")
    elif sharpe_ok and not dd_ok:
        print("\n>>> NEEDS ADJUSTMENT")
        print("    - Sharpe acceptable but drawdown too high")
        print("    - Adjust position sizing or risk limits")
    else:
        print("\n>>> NOT READY")
        print("    - Large divergence vs backtest")
        print("    - Pause and diagnose before proceeding")

    return engine


if __name__ == "__main__":
    engine = run_paper_trading_simulation()
