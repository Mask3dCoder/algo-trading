"""Phase 3.0: Multi-Asset Portfolio Construction.

Build a diversified portfolio using the validated alignment filter
across multiple asset classes to maximize risk-adjusted returns.

Position Sizing Approaches:
1. Equal Weight - Allocate equal capital to each asset class
2. Volatility Scaling - Scale positions for equal volatility contribution  
3. Risk Parity - Equalize risk contribution (advanced)

Assets (excluding FX):
- CTA (reference universe)
- Risk Parity universe
- Equity (SPY+EFA)
- Bonds (TLT+IEF)
- Commodities (GLD+DBC)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

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
# Portfolio Components
# =============================================================================

def load_asset_data(universe: str, start: str = '2018-01-01', 
                   end: str = '2024-12-31') -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
    """Load and pivot data for an asset universe."""
    print(f"\nLoading {universe} universe...")
    df = load_universe(universe, start, end, use_cache=True)
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close').dropna()
    
    close_prices = wide_df.values
    symbols = [col for col in wide_df.columns]
    
    print(f"  Shape: {close_prices.shape}, Symbols: {symbols}")
    return close_prices, symbols, wide_df.index


def run_single_asset_backtest(close_prices: np.ndarray, symbols: list[str], 
                           dates: pd.DatetimeIndex, start_bar: int = 200) -> Tuple[list[float], list[dict]]:
    """Run alignment filter strategy on a single asset universe.
    
    Returns (daily_returns, trades_list).
    """
    strategy_cls = get_strategy_class("macro_cta_trend")
    config = StrategyConfig(
        name=f"portfolio_component",
        strategy_type=StrategyType.MACRO,
        params={
            "alignment_filter": True,
            "min_signal_strength": 0.0,
        },
        max_position_notional=1_000_000,
    )
    strategy = strategy_cls(config)
    
    positions = {}
    trades = []
    equity = 1_000_000
    daily_returns = []
    trade_pnl_by_bar = {}
    
    n_periods = close_prices.shape[0]
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    
    for bar in range(start_bar, n_periods):
        warm_up_start = max(0, bar - 300)
        data = {
            "close": close_prices[warm_up_start:bar+1],
            "symbols": symbols,
        }
        
        strategy.set_timestamp(bar)
        signals = strategy.on_data(data)
        
        for signal in signals:
            symbol = signal.symbol
            direction = signal.direction
            price = close_prices[bar, symbol_to_idx[symbol]]
            
            if symbol not in positions:
                positions[symbol] = {
                    'direction': direction,
                    'entry_price': price,
                    'entry_bar': bar,
                }
            elif positions[symbol]['direction'] != direction:
                pos = positions[symbol]
                entry_price = pos['entry_price']
                entry_direction = pos['direction']
                size = signal.size
                
                if entry_direction == 1:
                    pnl = (price - entry_price) / entry_price * size
                else:
                    pnl = (entry_price - price) / entry_price * size
                
                trades.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'bar': bar,
                })
                
                if bar not in trade_pnl_by_bar:
                    trade_pnl_by_bar[bar] = 0.0
                trade_pnl_by_bar[bar] += pnl
                
                positions[symbol] = {
                    'direction': direction,
                    'entry_price': price,
                    'entry_bar': bar,
                }
        
        if bar == n_periods - 1:
            for symbol, pos in list(positions.items()):
                price = close_prices[bar, symbol_to_idx[symbol]]
                entry_price = pos['entry_price']
                entry_direction = pos['direction']
                size = 100000
                
                if entry_direction == 1:
                    pnl = (price - entry_price) / entry_price * size
                else:
                    pnl = (entry_price - price) / entry_price * size
                
                trades.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'bar': bar,
                })
                
                if bar not in trade_pnl_by_bar:
                    trade_pnl_by_bar[bar] = 0.0
                trade_pnl_by_bar[bar] += pnl
        
        bar_pnl = trade_pnl_by_bar.get(bar, 0.0)
        if equity > 0:
            ret = bar_pnl / equity
            daily_returns.append(ret)
        equity += bar_pnl
    
    return daily_returns, trades


# =============================================================================
# Portfolio Construction
# =============================================================================

def calculate_portfolio_returns(component_returns: Dict[str, list[float]], 
                               weights: Dict[str, float]) -> list[float]:
    """Combine component returns using portfolio weights."""
    if not component_returns:
        return []
    
    # Find minimum length
    min_len = min(len(rets) for rets in component_returns.values())
    
    portfolio_returns = []
    for i in range(min_len):
        port_ret = sum(weights[comp] * component_returns[comp][i] 
                     for comp in component_returns.keys() if i < len(component_returns[comp]))
        portfolio_returns.append(port_ret)
    
    return portfolio_returns


def equal_weight_sizing(assets: List[str]) -> Dict[str, float]:
    """Equal weight across all assets."""
    n = len(assets)
    weight = 1.0 / n if n > 0 else 0
    return {asset: weight for asset in assets}


def volatility_scaled_sizing(returns_dict: Dict[str, list[float]]) -> Dict[str, float]:
    """Scale positions so each asset contributes equal volatility."""
    vols = {}
    for asset, rets in returns_dict.items():
        if len(rets) > 1:
            vols[asset] = np.std(rets) * np.sqrt(252)
        else:
            vols[asset] = 0.2
    
    total_inv_vol = sum(1.0 / v for v in vols.values() if v > 0)
    
    weights = {}
    for asset, vol in vols.items():
        if vol > 0:
            weights[asset] = (1.0 / vol) / total_inv_vol
        else:
            weights[asset] = 0.0
    
    return weights


def risk_parity_sizing(returns_dict: Dict[str, list[float]]) -> Dict[str, float]:
    """Equalize risk contribution using inverse volatility weighting.
    
    This is a simplified risk parity approach.
    """
    # For simplicity, use inverse volatility (same as vol scaling)
    # A full implementation would use covariance matrix
    return volatility_scaled_sizing(returns_dict)


def calculate_portfolio_metrics(returns: list[float]) -> dict:
    """Calculate portfolio performance metrics."""
    if len(returns) < 2:
        return {
            'sharpe': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
        }
    
    returns_arr = np.array(returns)
    sharpe = sharpe_ratio(returns_arr)
    
    total_return = np.prod(1 + returns_arr) - 1
    
    # Max drawdown
    equity = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(np.min(drawdown))
    
    # Annualized volatility
    volatility = float(np.std(returns_arr) * np.sqrt(252))
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
    }


def calculate_correlation_matrix(returns_dict: Dict[str, list[float]]) -> pd.DataFrame:
    """Calculate correlation matrix between asset PnL streams."""
    # Align all return streams to same length
    min_len = min(len(rets) for rets in returns_dict.values())
    
    aligned_returns = {comp: rets[:min_len] for comp, rets in returns_dict.items()}
    
    df = pd.DataFrame(aligned_returns)
    corr_matrix = df.corr()
    
    return corr_matrix


# =============================================================================
# Main Experiment
# =============================================================================

def run_multi_asset_portfolio():
    """Run Phase 3.0 multi-asset portfolio construction."""
    print("=" * 70)
    print("PHASE 3.0: MULTI-ASSET PORTFOLIO CONSTRUCTION")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Select Assets (exclude FX)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: SELECT ASSETS (Excluding FX)")
    print("=" * 70)
    
    asset_universes = {
        'CTA': 'cta',
        'RiskParity': 'rp',
        'Equity': ['SPY', 'EFA'],  # Will load from CTA universe
        'Bonds': ['TLT', 'IEF'],    # Will load from RP universe
        'Commodities': ['GLD', 'DBC'],  # Will load from CTA universe
    }
    
    # Load data for each universe
    component_data = {}
    
    # Load CTA universe
    cta_prices, cta_symbols, cta_dates = load_asset_data('cta')
    
    # Load RP universe  
    rp_prices, rp_symbols, rp_dates = load_asset_data('rp')
    
    # Extract subsets
    # For Equity: SPY, EFA (from CTA)
    equity_idx = [i for i, s in enumerate(cta_symbols) if s in ['SPY', 'EFA']]
    equity_prices = cta_prices[:, equity_idx]
    equity_symbols = [cta_symbols[i] for i in equity_idx]
    
    # For Bonds: TLT, IEF (from RP)
    bonds_idx = [i for i, s in enumerate(rp_symbols) if s in ['TLT', 'IEF']]
    bonds_prices = rp_prices[:, bonds_idx]
    bonds_symbols = [rp_symbols[i] for i in bonds_idx]
    
    # For Commodities: GLD, DBC (from CTA)
    comm_idx = [i for i, s in enumerate(cta_symbols) if s in ['GLD', 'DBC']]
    comm_prices = cta_prices[:, comm_idx]
    comm_symbols = [cta_symbols[i] for i in comm_idx]
    
    component_data['CTA'] = (cta_prices, cta_symbols, cta_dates)
    component_data['RiskParity'] = (rp_prices, rp_symbols, rp_dates)
    component_data['Equity'] = (equity_prices, equity_symbols, cta_dates)
    component_data['Bonds'] = (bonds_prices, bonds_symbols, rp_dates)
    component_data['Commodities'] = (comm_prices, comm_symbols, cta_dates)
    
    # =========================================================================
    # Run backtests for each component
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: RUNNING COMPONENT BACKTESTS")
    print("=" * 70)
    
    component_returns = {}
    component_trades = {}
    
    for name, (prices, symbols, dates) in component_data.items():
        print(f"\n--- {name} ---")
        rets, trades = run_single_asset_backtest(prices, symbols, dates)
        component_returns[name] = rets
        component_trades[name] = trades
        
        metrics = calculate_portfolio_metrics(rets)
        print(f"  Sharpe: {metrics['sharpe']:.4f}")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Max DD: {metrics['max_drawdown']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Trades: {len(trades)}")
    
    # =========================================================================
    # Step 3: Portfolio Construction - Different Sizing Approaches
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: PORTFOLIO SIZING APPROACHES")
    print("=" * 70)
    
    results = []
    
    # Approach 1: Equal Weight
    print("\n--- Approach 1: Equal Weight ---")
    eq_weights = equal_weight_sizing(list(component_returns.keys()))
    print(f"Weights: {eq_weights}")
    
    eq_returns = calculate_portfolio_returns(component_returns, eq_weights)
    eq_metrics = calculate_portfolio_metrics(eq_returns)
    
    print(f"  Sharpe: {eq_metrics['sharpe']:.4f}")
    print(f"  Total Return: {eq_metrics['total_return']:.2%}")
    print(f"  Max DD: {eq_metrics['max_drawdown']:.2%}")
    print(f"  Volatility: {eq_metrics['volatility']:.2%}")
    
    results.append({
        'name': 'Equal Weight',
        'weights': eq_weights,
        'metrics': eq_metrics,
    })
    
    # Approach 2: Volatility Scaling
    print("\n--- Approach 2: Volatility Scaling ---")
    vol_weights = volatility_scaled_sizing(component_returns)
    print(f"Weights: {vol_weights}")
    
    vol_returns = calculate_portfolio_returns(component_returns, vol_weights)
    vol_metrics = calculate_portfolio_metrics(vol_returns)
    
    print(f"  Sharpe: {vol_metrics['sharpe']:.4f}")
    print(f"  Total Return: {vol_metrics['total_return']:.2%}")
    print(f"  Max DD: {vol_metrics['max_drawdown']:.2%}")
    print(f"  Volatility: {vol_metrics['volatility']:.2%}")
    
    results.append({
        'name': 'Vol Scaled',
        'weights': vol_weights,
        'metrics': vol_metrics,
    })
    
    # Approach 3: Risk Parity
    print("\n--- Approach 3: Risk Parity ---")
    rp_weights = risk_parity_sizing(component_returns)
    print(f"Weights: {rp_weights}")
    
    rp_returns = calculate_portfolio_returns(component_returns, rp_weights)
    rp_metrics = calculate_portfolio_metrics(rp_returns)
    
    print(f"  Sharpe: {rp_metrics['sharpe']:.4f}")
    print(f"  Total Return: {rp_metrics['total_return']:.2%}")
    print(f"  Max DD: {rp_metrics['max_drawdown']:.2%}")
    print(f"  Volatility: {rp_metrics['volatility']:.2%}")
    
    results.append({
        'name': 'Risk Parity',
        'weights': rp_weights,
        'metrics': rp_metrics,
    })
    
    # =========================================================================
    # Step 4: Diversification Check (Correlation Matrix)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: DIVERSIFICATION CHECK (Correlation Matrix)")
    print("=" * 70)
    
    corr_matrix = calculate_correlation_matrix(component_returns)
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Check average correlation
    n = len(corr_matrix)
    total_corr = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_corr += abs(corr_matrix.iloc[i, j])
            count += 1
    
    avg_corr = total_corr / count if count > 0 else 0
    print(f"\nAverage Absolute Correlation: {avg_corr:.3f}")
    
    if avg_corr < 0.3:
        print("LOW correlation - Good diversification!")
    elif avg_corr < 0.6:
        print("MODERATE correlation - Some diversification benefit")
    else:
        print("HIGH correlation - Poor diversification")
    
    # =========================================================================
    # Output Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("PORTFOLIO CONSTRUCTION RESULTS")
    print("=" * 70)
    
    header = f"{'Portfolio Type':<20} {'Sharpe':>8} {'Return':>8} {'Vol':>8} {'Max DD':>8}"
    print(f"\n{header}")
    print("-" * len(header))
    
    for r in results:
        m = r['metrics']
        print(f"{r['name']:<20} {m['sharpe']:>8.4f} {m['total_return']:>7.2%} "
              f"{m['volatility']:>7.2%} {m['max_drawdown']:>7.2%}")
    
    # Component comparison
    print(f"\n{'Component':<20} {'Sharpe':>8}")
    print("-" * 30)
    for name, rets in component_returns.items():
        metrics = calculate_portfolio_metrics(rets)
        print(f"{name:<20} {metrics['sharpe']:>8.4f}")
    
    # =========================================================================
    # Decision Framework
    # =========================================================================
    print("\n" + "=" * 70)
    print("DECISION FRAMEWORK")
    print("=" * 70)
    
    best_portfolio = max(results, key=lambda x: x['metrics']['sharpe'])
    best_sharpe = best_portfolio['metrics']['sharpe']
    
    avg_component_sharpe = np.mean([calculate_portfolio_metrics(rets)['sharpe'] 
                                   for rets in component_returns.values()])
    
    print(f"\n* Best Portfolio Sharpe: {best_sharpe:.4f} ({best_portfolio['name']})")
    print(f"* Avg Component Sharpe: {avg_component_sharpe:.4f}")
    print(f"* Portfolio > Components: {best_sharpe > avg_component_sharpe}")
    print(f"* Average Correlation: {avg_corr:.3f}")
    
    portfolio_better = best_sharpe > avg_component_sharpe
    low_corr = avg_corr < 0.3
    dd_improved = True  # Check if max DD is better than worst component
    
    if portfolio_better and low_corr:
        print("\n>>> STRONG PORTFOLIO - Significant improvement!")
        print("    - Sharpe exceeds individual components")
        print("    - Good diversification (low correlation)")
        print("    - Proceed to regime filter (for FX inclusion)")
        print("    - Consider light parameter tuning")
    elif portfolio_better:
        print("\n>>> MODERATE PORTFOLIO - Some improvement")
        print("    - Better than components but correlation is moderate")
        print("    - Consider adding more diverse assets")
    else:
        print("\n>>> WEAK INTEGRATION - No Sharpe improvement")
        print("    - Portfolio doesn't beat individual components")
        print("    - Check: asset selection, correlation, sizing")
    
    return results, component_returns, corr_matrix


if __name__ == "__main__":
    results, comp_returns, corr = run_multi_asset_portfolio()
