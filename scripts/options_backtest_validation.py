"""Options strategy backtest validation using synthetic chains from real underlying data.

Downloads underlying OHLCV via yfinance, generates synthetic options chains
using BSM with regime-adaptive IV, and runs the iron_condor strategy through
a simulation engine. Produces reports/options_validation.md.

Usage:
    python scripts/options_backtest_validation.py
    python scripts/options_backtest_validation.py --underlyings SPY QQQ IWM --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = _ROOT / "reports" / "options_validation.md"


# ── BSM helpers ──────────────────────────────────────────────────────────────

def bsm_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-10 or sigma < 1e-10:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bsm_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-10 or sigma < 1e-10:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bsm_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> float:
    if T < 1e-10 or sigma < 1e-10:
        return 1.0 if (is_call and S > K) or (not is_call and S < K) else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if is_call:
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def strike_from_delta(S: float, T: float, r: float, sigma: float, delta_target: float, is_call: bool = True) -> float:
    """Invert BSM to find strike for a given absolute delta."""
    if T < 1e-10 or sigma < 1e-10:
        return S
    target = delta_target / 100.0
    if not is_call:
        target = 1.0 - target
    target = np.clip(target, 0.001, 0.999)
    try:
        d1_target = norm.ppf(target)
        K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
        return float(K)
    except (ValueError, OverflowError):
        return S * (1 + sigma * np.sqrt(T) * (0.5 if is_call else -0.5))


# ── Synthetic IV model ───────────────────────────────────────────────────────

def generate_synthetic_iv(close: pd.Series, lookback: int = 60) -> pd.Series:
    """Generate realistic IV from realized vol with mean-reversion and skew."""
    log_returns = np.log(close / close.shift(1)).dropna()
    realized_vol = log_returns.rolling(lookback).std() * np.sqrt(252)
    # IV trades at premium to realized, with mean-reversion
    iv_premium = 0.15  # vol risk premium
    iv = realized_vol + iv_premium
    # Mean-revert to long-run ~0.20
    long_run_iv = 0.20
    mean_reversion = 0.02
    iv_smooth = iv.copy()
    for i in range(1, len(iv_smooth)):
        if np.isnan(iv_smooth.iloc[i]):
            iv_smooth.iloc[i] = iv_smooth.iloc[i - 1] if not np.isnan(iv_smooth.iloc[i - 1]) else long_run_iv
        else:
            iv_smooth.iloc[i] = iv_smooth.iloc[i - 1] + mean_reversion * (iv_smooth.iloc[i] - iv_smooth.iloc[i - 1])
    iv_smooth = iv_smooth.clip(lower=0.08, upper=0.80)
    iv_smooth = iv_smooth.fillna(long_run_iv)
    # Reindex to match original close index (dropna shifted it by 1)
    iv_smooth = iv_smooth.reindex(close.index, method="ffill").fillna(long_run_iv)
    return iv_smooth


# ── Iron condor simulation ───────────────────────────────────────────────────

@dataclass
class CondorPosition:
    symbol: str
    entry_date: date
    expiry: date
    short_put_strike: float
    long_put_strike: float
    short_call_strike: float
    long_call_strike: float
    net_credit: float  # per share
    contracts: int
    entry_spot: float
    entry_iv: float
    entry_ivr: float
    dte_at_entry: int
    pnl: float = 0.0
    closed: bool = False
    close_date: date | None = None
    close_reason: str = ""


@dataclass
class BacktestResult:
    trades: list[CondorPosition] = field(default_factory=list)
    equity_curve: list[tuple[date, float]] = field(default_factory=list)
    daily_pnl: list[tuple[date, float]] = field(default_factory=list)


def compute_ivr(iv_series: pd.Series, idx: int, lookback: int = 252) -> float:
    start = max(0, idx - lookback)
    window = iv_series.iloc[start : idx + 1].dropna()
    if len(window) < 20:
        return 50.0
    iv_min, iv_max = window.min(), window.max()
    if iv_max - iv_min < 1e-10:
        return 50.0
    return float((window.iloc[-1] - iv_min) / (iv_max - iv_min) * 100)


def run_iron_condor_backtest(
    underlying: str,
    close: pd.Series,
    iv: pd.Series,
    dates: pd.DatetimeIndex,
    config: dict,
    capital: float = 100_000.0,
) -> BacktestResult:
    """Simulate iron condor strategy on historical underlying + synthetic IV data."""
    r = config.get("risk_free_rate", 0.04)
    delta_short = config.get("delta_short", 16)
    delta_wing = config.get("delta_wing_width", 10)
    dte_min = config.get("dte_min", 30)
    dte_max = config.get("dte_max", 45)
    ivr_entry = config.get("ivr_entry", 40)
    ivr_exit = config.get("ivr_exit", 20)
    profit_target = config.get("profit_target_pct", 50) / 100.0
    max_loss_mult = config.get("max_loss_pct", 200) / 100.0
    adjust_threshold = config.get("adjustment_threshold", 3) / 100.0
    position_pct = config.get("position_pct", 0.02)

    result = BacktestResult()
    equity = capital
    active: list[CondorPosition] = []

    n = len(close)
    if n < 60:
        return result

    close_arr = close.values
    iv_arr = iv.values
    date_arr = dates

    for i in range(60, n):
        S = float(close_arr[i])
        sigma = float(iv_arr[i])
        dt = date_arr[i].date() if hasattr(date_arr[i], "date") else date_arr[i]
        ivr = compute_ivr(iv, i)

        daily_pnl = 0.0

        # Update active positions
        for pos in list(active):
            if pos.closed:
                continue

            dte = (pos.expiry - dt).days
            if dte <= 0:
                # Expiration: settle at intrinsic
                put_spread_val = max(pos.short_put_strike - S, 0) - max(pos.long_put_strike - S, 0)
                call_spread_val = max(S - pos.short_call_strike, 0) - max(S - pos.long_call_strike, 0)
                pos.pnl = (pos.net_credit - put_spread_val - call_spread_val) * 100 * pos.contracts
                pos.closed = True
                pos.close_date = dt
                pos.close_reason = "expiration"
                daily_pnl += pos.pnl
                continue

            T = dte / 252.0

            # Mark-to-market
            sp_val = bsm_put_price(S, pos.short_put_strike, T, r, sigma)
            lp_val = bsm_put_price(S, pos.long_put_strike, T, r, sigma)
            sc_val = bsm_call_price(S, pos.short_call_strike, T, r, sigma)
            lc_val = bsm_call_price(S, pos.long_call_strike, T, r, sigma)

            current_cost = (sp_val - lp_val + sc_val - lc_val)
            unrealized = (pos.net_credit - current_cost) * 100 * pos.contracts

            # Profit target
            max_profit = pos.net_credit * 100 * pos.contracts
            if unrealized >= profit_target * max_profit:
                pos.pnl = unrealized
                pos.closed = True
                pos.close_date = dt
                pos.close_reason = "profit_target"
                daily_pnl += pos.pnl
                continue

            # Max loss
            max_loss = max_profit * max_loss_mult
            if unrealized <= -max_loss:
                pos.pnl = unrealized
                pos.closed = True
                pos.close_date = dt
                pos.close_reason = "max_loss"
                daily_pnl += pos.pnl
                continue

            # IV rank exit
            if ivr < ivr_exit:
                pos.pnl = unrealized
                pos.closed = True
                pos.close_date = dt
                pos.close_reason = "iv_rank_exit"
                daily_pnl += pos.pnl
                continue

            # Early exit at 7 DTE
            if dte <= 7:
                pos.pnl = unrealized
                pos.closed = True
                pos.close_date = dt
                pos.close_reason = "near_expiry"
                daily_pnl += pos.pnl

        # Entry: open new condor when IV rank elevated and no active position
        active_open = [p for p in active if not p.closed]
        if ivr > ivr_entry and len(active_open) == 0:
            dte_entry = (dte_min + dte_max) // 2
            T_entry = dte_entry / 252.0
            expiry_date = dt + timedelta(days=dte_entry)

            # For put credit spread: short put at higher strike (closer to ATM), long put at lower strike
            # strike_from_delta with higher delta → lower strike for puts
            lp = strike_from_delta(S, T_entry, r, sigma, delta_short, is_call=False)  # further OTM (lower strike)
            sp = strike_from_delta(S, T_entry, r, sigma, delta_short + delta_wing, is_call=False)  # closer ATM (higher strike)
            # For call credit spread: short call at lower strike (closer to ATM), long call at higher strike
            # strike_from_delta with higher delta → lower strike for calls
            sc = strike_from_delta(S, T_entry, r, sigma, delta_short + delta_wing, is_call=True)  # closer ATM (lower strike)
            lc = strike_from_delta(S, T_entry, r, sigma, delta_short, is_call=True)  # further OTM (higher strike)

            # Net credit from selling the spread
            sp_prem = bsm_put_price(S, sp, T_entry, r, sigma)
            lp_prem = bsm_put_price(S, lp, T_entry, r, sigma)
            sc_prem = bsm_call_price(S, sc, T_entry, r, sigma)
            lc_prem = bsm_call_price(S, lc, T_entry, r, sigma)

            net_credit = (sp_prem - lp_prem + sc_prem - lc_prem)
            if net_credit <= 0:
                continue

            # Position sizing: risk 2% of equity per trade
            risk_amount = equity * position_pct
            max_risk = (min(sp - lp, lc - sc) - net_credit) * 100  # per contract
            if max_risk <= 0:
                max_risk = net_credit * 100  # fallback
            contracts = max(1, int(risk_amount / max_risk))

            pos = CondorPosition(
                symbol=underlying,
                entry_date=dt,
                expiry=expiry_date,
                short_put_strike=sp,
                long_put_strike=lp,
                short_call_strike=sc,
                long_call_strike=lc,
                net_credit=net_credit,
                contracts=contracts,
                entry_spot=S,
                entry_iv=sigma,
                entry_ivr=ivr,
                dte_at_entry=dte_entry,
            )
            active.append(pos)

        # Collect closed trades and clean active list
        newly_closed = [p for p in active if p.closed]
        result.trades.extend(newly_closed)
        active = [p for p in active if not p.closed]

        equity += daily_pnl
        result.equity_curve.append((dt, equity))
        result.daily_pnl.append((dt, daily_pnl))

    # Close any remaining positions at final price
    for pos in active:
        if not pos.closed:
            S_final = float(close_arr[-1])
            dt_final = date_arr[-1].date() if hasattr(date_arr[-1], "date") else date_arr[-1]
            put_spread_val = max(pos.short_put_strike - S_final, 0) - max(pos.long_put_strike - S_final, 0)
            call_spread_val = max(S_final - pos.short_call_strike, 0) - max(S_final - pos.long_call_strike, 0)
            pos.pnl = (pos.net_credit - put_spread_val - call_spread_val) * 100 * pos.contracts
            pos.closed = True
            pos.close_date = dt_final
            pos.close_reason = "end_of_backtest"
            equity += pos.pnl
            result.equity_curve.append((dt_final, equity))

    # Add remaining positions (closed at end_of_backtest)
    result.trades.extend(active)
    return result


# ── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_dte_at_entry: float = 0.0
    avg_holding_days: float = 0.0
    avg_ivr_at_entry: float = 0.0
    total_pnl: float = 0.0
    max_consecutive_losses: int = 0
    close_reasons: dict = field(default_factory=dict)
    regime_performance: dict = field(default_factory=dict)


def compute_metrics(result: BacktestResult, capital: float = 100_000.0) -> PerformanceMetrics:
    m = PerformanceMetrics()

    if not result.equity_curve:
        return m

    eq = pd.Series(
        [e[1] for e in result.equity_curve],
        index=[e[0] for e in result.equity_curve],
    )
    returns = eq.pct_change().dropna()

    m.total_return_pct = float((eq.iloc[-1] / capital - 1) * 100)
    n_days = max(1, (eq.index[-1] - eq.index[0]).days)
    m.annualized_return_pct = float(((eq.iloc[-1] / capital) ** (365.0 / n_days) - 1) * 100)

    if len(returns) > 1 and returns.std() > 0:
        m.sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            m.sortino_ratio = float(returns.mean() / downside.std() * np.sqrt(252))

    # Max drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    m.max_drawdown_pct = float(dd.min() * 100)

    if m.max_drawdown_pct != 0:
        m.calmar_ratio = float(m.annualized_return_pct / abs(m.max_drawdown_pct))

    # Trade-level stats
    closed = [t for t in result.trades if t.closed]
    m.total_trades = len(closed)
    if m.total_trades == 0:
        return m

    pnls = [t.pnl for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate_pct = float(len(wins) / len(pnls) * 100) if pnls else 0
    m.avg_win = float(np.mean(wins)) if wins else 0
    m.avg_loss = float(np.mean(losses)) if losses else 0
    m.total_pnl = float(sum(pnls))

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    m.profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    m.avg_dte_at_entry = float(np.mean([t.dte_at_entry for t in closed]))
    m.avg_ivr_at_entry = float(np.mean([t.entry_ivr for t in closed]))

    holding_days = []
    for t in closed:
        if t.close_date and t.entry_date:
            holding_days.append((t.close_date - t.entry_date).days)
    m.avg_holding_days = float(np.mean(holding_days)) if holding_days else 0

    # Close reasons
    reasons: dict[str, int] = {}
    for t in closed:
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1
    m.close_reasons = reasons

    # Max consecutive losses
    consec = 0
    max_consec = 0
    for p in pnls:
        if p <= 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    m.max_consecutive_losses = max_consec

    # Regime performance (rough: split by year)
    regime_pnl: dict[str, list[float]] = {}
    for t in closed:
        year = str(t.entry_date.year) if t.entry_date else "unknown"
        regime_pnl.setdefault(year, []).append(t.pnl)
    m.regime_performance = {k: float(sum(v)) for k, v in regime_pnl.items()}

    return m


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(
    all_metrics: dict[str, PerformanceMetrics],
    all_results: dict[str, BacktestResult],
    config: dict,
    start: str,
    end: str,
) -> str:
    lines = [
        "# Options Strategy Validation Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Period**: {start} to {end}",
        f"**Strategy**: Iron Condor (16-delta wings, 30-45 DTE)",
        f"**Underlyings**: {', '.join(all_metrics.keys())}",
        f"**Initial Capital**: $100,000 per underlying",
        "",
        "## Methodology",
        "",
        "This validation uses **synthetic options chains** generated from real underlying",
        "price data (via yfinance) using Black-Scholes-Merton pricing with regime-adaptive",
        "implied volatility. IV is derived from realized volatility + vol risk premium,",
        "with mean-reversion to long-run levels. This approach validates the strategy",
        "logic and signal generation without requiring historical options chain data.",
        "",
        "### Strategy Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Short delta | {config['delta_short']} |",
        f"| Wing width | {config['delta_wing_width']} deltas |",
        f"| DTE range | {config['dte_min']}-{config['dte_max']} |",
        f"| IV rank entry | >{config['ivr_entry']} |",
        f"| IV rank exit | <{config['ivr_exit']} |",
        f"| Profit target | {config['profit_target_pct']}% of max profit |",
        f"| Max loss | {config['max_loss_pct']}% of max profit |",
        f"| Adjustment threshold | {config['adjustment_threshold']}% |",
        f"| Position risk | {config.get('position_pct', 0.02)*100:.0f}% of equity |",
        "",
        "## Aggregate Results",
        "",
        "| Metric | " + " | ".join(all_metrics.keys()) + " |",
        "|--------|" + "|".join(["---"] * len(all_metrics)) + "|",
    ]

    def _fmt(m: PerformanceMetrics, attr: str) -> str:
        val = getattr(m, attr)
        if isinstance(val, float):
            if "pct" in attr or "ratio" in attr:
                return f"{val:+.2f}" if "pct" in attr else f"{val:.2f}"
            return f"${val:,.0f}" if "pnl" in attr or "win" in attr.lower() else f"{val:.2f}"
        return str(val)

    metric_rows = [
        ("Total Return", "total_return_pct", "%"),
        ("Annualized Return", "annualized_return_pct", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ""),
        ("Sortino Ratio", "sortino_ratio", ""),
        ("Max Drawdown", "max_drawdown_pct", "%"),
        ("Calmar Ratio", "calmar_ratio", ""),
        ("Win Rate", "win_rate_pct", "%"),
        ("Profit Factor", "profit_factor", ""),
        ("Total Trades", "total_trades", ""),
        ("Winning / Losing", "winning_trades", ""),
        ("Avg Win", "avg_win", "$"),
        ("Avg Loss", "avg_loss", "$"),
        ("Total P&L", "total_pnl", "$"),
        ("Avg DTE at Entry", "avg_dte_at_entry", ""),
        ("Avg Holding Days", "avg_holding_days", ""),
        ("Avg IV Rank at Entry", "avg_ivr_at_entry", ""),
        ("Max Consec Losses", "max_consecutive_losses", ""),
    ]

    for label, attr, suffix in metric_rows:
        vals = []
        for name in all_metrics:
            m = all_metrics[name]
            val = getattr(m, attr)
            if attr == "winning_trades":
                val = f"{m.winning_trades}/{m.losing_trades}"
            elif isinstance(val, float):
                if suffix == "$":
                    val = f"${val:,.0f}"
                elif suffix == "%":
                    val = f"{val:+.1f}%"
                else:
                    val = f"{val:.2f}"
            vals.append(str(val))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Per-Underlying Breakdown")
    lines.append("")

    for name, metrics in all_metrics.items():
        result = all_results.get(name)
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **Total P&L**: ${metrics.total_pnl:,.0f}")
        lines.append(f"- **Trades**: {metrics.total_trades} ({metrics.winning_trades}W / {metrics.losing_trades}L)")
        lines.append(f"- **Win Rate**: {metrics.win_rate_pct:.1f}%")
        lines.append(f"- **Sharpe**: {metrics.sharpe_ratio:.2f}")
        lines.append(f"- **Max DD**: {metrics.max_drawdown_pct:.1f}%")
        lines.append("")

        if metrics.close_reasons:
            lines.append("**Exit Reasons**:")
            for reason, count in sorted(metrics.close_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"- {reason}: {count}")
            lines.append("")

        if metrics.regime_performance:
            lines.append("**Annual P&L**:")
            for year, pnl in sorted(metrics.regime_performance.items()):
                lines.append(f"- {year}: ${pnl:,.0f}")
            lines.append("")

    lines.append("## Key Observations")
    lines.append("")

    # Auto-generate observations
    avg_sharpe = np.mean([m.sharpe_ratio for m in all_metrics.values()])
    avg_win_rate = np.mean([m.win_rate_pct for m in all_metrics.values()])
    avg_dd = np.mean([m.max_drawdown_pct for m in all_metrics.values()])
    total_trades = sum(m.total_trades for m in all_metrics.values())
    total_pnl = sum(m.total_pnl for m in all_metrics.values())

    observations = []

    if avg_sharpe > 1.0:
        observations.append(f"**Strong risk-adjusted returns**: Average Sharpe {avg_sharpe:.2f} across underlyings exceeds the 1.0 threshold for strategy viability.")
    elif avg_sharpe > 0.5:
        observations.append(f"**Moderate risk-adjusted returns**: Average Sharpe {avg_sharpe:.2f}. Signal quality is acceptable but could benefit from improved entry timing.")
    else:
        observations.append(f"**Weak risk-adjusted returns**: Average Sharpe {avg_sharpe:.2f}. Strategy needs refinement before deployment.")

    if avg_win_rate > 70:
        observations.append(f"**High win rate** ({avg_win_rate:.0f}%): Consistent with premium-selling strategies. Tail risk management is critical.")
    elif avg_win_rate > 50:
        observations.append(f"**Moderate win rate** ({avg_win_rate:.0f}%): Acceptable for defined-risk spreads.")

    if abs(avg_dd) > 15:
        observations.append(f"**Significant drawdowns** (avg max DD {avg_dd:.1f}%): Consider tighter loss limits or reduced position sizing.")

    # Close reason analysis
    all_reasons: dict[str, int] = {}
    for m in all_metrics.values():
        for reason, count in m.close_reasons.items():
            all_reasons[reason] = all_reasons.get(reason, 0) + count
    if "expiration" in all_reasons and all_reasons["expiration"] > total_trades * 0.5:
        observations.append(f"**High expiration rate** ({all_reasons['expiration']}/{total_trades}): Many positions held to expiry. Consider more aggressive profit-taking.")

    for obs in observations:
        lines.append(f"- {obs}")

    lines.append("")
    lines.append("## Limitations & Next Steps")
    lines.append("")
    lines.append("### Limitations")
    lines.append("")
    lines.append("1. **Synthetic options data**: IV is derived from realized vol + constant premium, not actual market IV surface. Real IV exhibits term structure, skew, and event-driven spikes not captured here.")
    lines.append("2. **No transaction costs**: Actual iron condors incur 4-leg commissions and slippage. With ~$1-2 per contract per leg, net P&L would be lower.")
    lines.append("3. **No early assignment risk**: American-style options can be assigned early, especially for ITM short puts.")
    lines.append("4. **Simplified adjustment logic**: Real traders may roll, convert to butterfly, or take other creative adjustments not modeled.")
    lines.append("5. **No gap risk**: Overnight gaps can blow through strikes before stop-losses trigger.")
    lines.append("")
    lines.append("### Next Steps")
    lines.append("")
    lines.append("1. **Download real options data** via `OptionsDataLoader` once Alpaca historical chain data is available, or upgrade to Polygon.io Starter ($29/mo) for full historical chains.")
    lines.append("2. **Run walk-forward validation** with real data across GFC, COVID, and Rate Hike regimes.")
    lines.append("3. **Add transaction cost model** (4-leg combo: ~$3.50/contract at IBKR, $0.65/contract at Alpaca).")
    lines.append("4. **Implement term structure overlay**: Use near-term IV for entry timing, longer-term for position sizing.")
    lines.append("5. **Capacity analysis**: Iron condors on SPY/QQQ can handle $500K-$2M before meaningful market impact.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by QuantCore options backtest validation pipeline.*")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Options strategy backtest validation")
    parser.add_argument("--underlyings", nargs="+", default=["SPY", "QQQ", "IWM"])
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=100_000)
    args = parser.parse_args()

    config = {
        "delta_short": 16,
        "delta_wing_width": 10,
        "dte_min": 30,
        "dte_max": 45,
        "ivr_entry": 40,
        "ivr_exit": 20,
        "profit_target_pct": 50,
        "max_loss_pct": 200,
        "adjustment_threshold": 3,
        "risk_free_rate": 0.04,
        "position_pct": 0.02,
    }

    log.info("Options backtest validation: %s (%s -> %s)", args.underlyings, args.start, args.end)

    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    all_metrics: dict[str, PerformanceMetrics] = {}
    all_results: dict[str, BacktestResult] = {}

    for underlying in args.underlyings:
        log.info("Downloading %s...", underlying)
        ticker = yf.Ticker(underlying)
        hist = ticker.history(start=args.start, end=args.end, auto_adjust=True)

        if hist.empty:
            log.warning("No data for %s, skipping", underlying)
            continue

        close = hist["Close"]
        dates = close.index
        iv = generate_synthetic_iv(close)

        # Diagnostic: IVR distribution
        ivr_vals = [compute_ivr(iv, i) for i in range(60, len(iv))]
        log.info("IV range: [%.3f, %.3f], mean=%.3f", iv.min(), iv.max(), iv.mean())
        log.info("IVR range: [%.1f, %.1f], mean=%.1f, >40: %d/%d",
                 min(ivr_vals), max(ivr_vals), sum(ivr_vals)/len(ivr_vals),
                 sum(1 for v in ivr_vals if v > 40), len(ivr_vals))

        log.info("Running iron condor backtest on %s (%d bars)...", underlying, len(close))
        result = run_iron_condor_backtest(underlying, close, iv, dates, config, args.capital)

        closed_trades = [t for t in result.trades if t.closed]
        log.info("%s: %d trades, %d closed", underlying, len(result.trades), len(closed_trades))

        metrics = compute_metrics(result, args.capital)
        all_metrics[underlying] = metrics
        all_results[underlying] = result

        log.info(
            "%s: Return=%+.1f%% Sharpe=%.2f MaxDD=%.1f%% WinRate=%.0f%% PnL=$%.0f",
            underlying,
            metrics.total_return_pct,
            metrics.sharpe_ratio,
            metrics.max_drawdown_pct,
            metrics.win_rate_pct,
            metrics.total_pnl,
        )

    if not all_metrics:
        log.error("No results to report")
        sys.exit(1)

    # Generate report
    report = generate_report(all_metrics, all_results, config, args.start, args.end)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    log.info("Report written to %s", REPORT_PATH)

    # Summary
    print("\n" + "=" * 60)
    print("OPTIONS BACKTEST VALIDATION SUMMARY")
    print("=" * 60)
    for name, m in all_metrics.items():
        print(
            f"  {name}: Return={m.total_return_pct:+.1f}% Sharpe={m.sharpe_ratio:.2f} "
            f"MaxDD={m.max_drawdown_pct:.1f}% WinRate={m.win_rate_pct:.0f}% "
            f"Trades={m.total_trades} PnL=${m.total_pnl:,.0f}"
        )
    avg_sharpe = np.mean([m.sharpe_ratio for m in all_metrics.values()])
    print(f"\n  Average Sharpe: {avg_sharpe:.2f}")
    print(f"  Report: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
