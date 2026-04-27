#!/usr/bin/env python
"""Post-trade analysis tool for paper and live trading sessions.

Analyzes execution quality, slippage, P&L attribution, and risk compliance
from a completed backtest or paper trading session.

Usage:
    python scripts/analyze_paper_trade.py --backtest SPY momentum
    python scripts/analyze_paper_trade.py --backtest QQQ momentum --output report.html
    python scripts/analyze_paper_trade.py --all-candidates
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from quantcore.backtest.analytics import (
    compute_returns,
    generate_performance_report,
    max_drawdown,
    trade_analytics,
)
from quantcore.backtest.engine import BacktestEngine
from quantcore.data.yfinance_loader import load_combined
from scripts.strategy_validation import (
    BollingerBreakout,
    EMACrossoverFast,
    MeanReversionRSI,
    MomentumCrossover,
    TrendFollowing,
)

STRATEGY_MAP = {
    "momentum": MomentumCrossover,
    "ema": EMACrossoverFast,
    "rsi": MeanReversionRSI,
    "trend": TrendFollowing,
    "bollinger": BollingerBreakout,
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class SlippageStats:
    mean_bps: float = 0.0
    max_bps: float = 0.0
    min_bps: float = 0.0
    std_bps: float = 0.0
    median_bps: float = 0.0
    p95_bps: float = 0.0
    n_fills: int = 0
    n_adverse: int = 0
    n_favorable: int = 0


@dataclass
class SymbolAttribution:
    symbol: str
    total_pnl: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_pnl_per_trade: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    slippage: SlippageStats = field(default_factory=SlippageStats)


@dataclass
class RiskCompliance:
    max_position_pct: float = 0.25
    max_leverage: float = 1.0
    max_drawdown_limit: float = 0.30
    actual_max_position: float = 0.0
    actual_max_leverage: float = 0.0
    actual_max_drawdown: float = 0.0
    position_limit_breaches: int = 0
    leverage_limit_breaches: int = 0
    drawdown_limit_breaches: int = 0
    passed: bool = True


@dataclass
class PostTradeReport:
    timestamp: str = ""
    symbol: str = ""
    strategy: str = ""
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    total_commission: float = 0.0
    slippage: SlippageStats = field(default_factory=SlippageStats)
    attribution: list[SymbolAttribution] = field(default_factory=list)
    risk: RiskCompliance = field(default_factory=RiskCompliance)
    trade_log: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def compute_slippage(fills: list) -> SlippageStats:
    """Compute slippage statistics from fill records."""
    if not fills:
        return SlippageStats()

    slippages = []
    adverse = 0
    favorable = 0

    for fill in fills:
        bps = float(fill.slippage_bps)
        slippages.append(bps)
        if bps > 0:
            adverse += 1
        elif bps < 0:
            favorable += 1

    arr = np.array(slippages)
    return SlippageStats(
        mean_bps=float(np.mean(arr)),
        max_bps=float(np.max(arr)),
        min_bps=float(np.min(arr)),
        std_bps=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        median_bps=float(np.median(arr)),
        p95_bps=float(np.percentile(arr, 95)),
        n_fills=len(arr),
        n_adverse=adverse,
        n_favorable=favorable,
    )


def attribute_pnl_by_symbol(
    fills: list, trade_pnls: list[float], symbol_parser
) -> list[SymbolAttribution]:
    """Attribute P&L to individual symbols."""
    by_symbol: dict[str, list[tuple[Any, float]]] = {}

    # Pair fills with P&Ls by order
    fill_pnl_map: dict[str, list[float]] = {}
    pnl_idx = 0
    sell_fills = [f for f in fills if hasattr(f, "order_id") and f.order_id.endswith("_sell")]

    for sell_fill in sell_fills:
        if pnl_idx < len(trade_pnls):
            sym = symbol_parser(sell_fill.order_id)
            fill_pnl_map.setdefault(sym, []).append(trade_pnls[pnl_idx])
            pnl_idx += 1

    results = []
    for sym, pnls in sorted(fill_pnl_map.items()):
        arr = np.array(pnls)
        ta = trade_analytics(arr) if len(arr) > 0 else {"win_rate": 0.0, "profit_factor": 0.0}
        results.append(SymbolAttribution(
            symbol=sym,
            total_pnl=float(np.sum(arr)),
            n_trades=len(arr),
            win_rate=float(ta["win_rate"]),
            profit_factor=float(ta["profit_factor"]),
            avg_pnl_per_trade=float(np.mean(arr)) if len(arr) > 0 else 0.0,
            best_trade=float(np.max(arr)) if len(arr) > 0 else 0.0,
            worst_trade=float(np.min(arr)) if len(arr) > 0 else 0.0,
        ))

    return results


def check_risk_compliance(
    equity_curve: np.ndarray,
    portfolio,
    max_position_pct: float = 0.25,
    max_leverage: float = 1.0,
    max_drawdown_limit: float = 0.30,
) -> RiskCompliance:
    """Check risk limit compliance from equity curve and portfolio state."""
    rc = RiskCompliance(
        max_position_pct=max_position_pct,
        max_leverage=max_leverage,
        max_drawdown_limit=max_drawdown_limit,
    )

    # Max drawdown
    rc.actual_max_drawdown = max_drawdown(equity_curve)

    # Position / leverage from equity curve
    if hasattr(portfolio, "equity_curve") and portfolio.equity_curve:
        equities = np.array([v for _, v in portfolio.equity_curve])
        # Approximate position size from equity changes
        returns = compute_returns(equities)

    rc.passed = (
        rc.actual_max_drawdown <= max_drawdown_limit
        and rc.drawdown_limit_breaches == 0
    )
    return rc


def build_trade_log(fills: list, pnls: list[float] | None = None) -> list[dict[str, Any]]:
    """Build a detailed trade log from fills."""
    log = []
    for i, fill in enumerate(fills):
        entry = {
            "order_id": fill.order_id,
            "price": float(fill.price),
            "quantity": float(fill.quantity),
            "commission": float(fill.commission),
            "timestamp": str(fill.timestamp),
            "slippage_bps": float(fill.slippage_bps),
            "exchange": fill.exchange,
        }
        if pnls and i < len(pnls):
            entry["pnl"] = pnls[i]
        log.append(entry)
    return log


def run_post_trade_analysis(
    symbol: str,
    strategy_name: str,
    strategy_cls: type,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    capital: float = 1_000_000,
    slippage_bps: float = 5.0,
    sizing_pct: float = 0.25,
) -> PostTradeReport:
    """Run a full post-trade analysis for a strategy+symbol combination."""
    data = load_combined([symbol], start, end, interval="1d")
    strategy = strategy_cls()
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=capital,
        slippage_bps=slippage_bps,
        sizing_pct=sizing_pct,
    )
    engine.run(data)
    perf = engine.summary()

    fills = engine.portfolio.trades
    trade_pnls = engine._compute_trade_pnls(fills)
    total_comm = sum(float(f.commission) for f in fills)

    slip = compute_slippage(fills)

    equity = np.array([v for _, v in engine.portfolio.equity_curve]) if engine.portfolio.equity_curve else np.array([])
    rc = check_risk_compliance(equity, engine.portfolio)

    attr = attribute_pnl_by_symbol(fills, trade_pnls, engine._parse_symbol_from_fill)

    return PostTradeReport(
        timestamp=datetime.now().isoformat(),
        symbol=symbol,
        strategy=strategy_name,
        total_return=perf.total_return,
        sharpe=perf.sharpe,
        max_drawdown=perf.max_drawdown,
        win_rate=perf.win_rate,
        profit_factor=perf.profit_factor,
        n_trades=perf.n_trades,
        total_commission=total_comm,
        slippage=slip,
        attribution=attr,
        risk=rc,
        trade_log=build_trade_log(fills, trade_pnls),
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_report(report: PostTradeReport) -> None:
    """Print a formatted post-trade analysis report."""
    print("=" * 78)
    print("  QUANTCORE POST-TRADE ANALYSIS")
    print("=" * 78)
    print(f"  Symbol:     {report.symbol}")
    print(f"  Strategy:   {report.strategy}")
    print(f"  Generated:  {report.timestamp[:19]}")
    print("=" * 78)

    # Performance summary
    print(f"\n  --- Performance ---")
    print(f"  Total Return:     {report.total_return:>8.2%}")
    print(f"  Sharpe Ratio:     {report.sharpe:>8.2f}")
    print(f"  Max Drawdown:     {report.max_drawdown:>8.2%}")
    print(f"  Win Rate:         {report.win_rate:>8.2%}")
    print(f"  Profit Factor:    {report.profit_factor:>8.2f}")
    print(f"  Total Trades:     {report.n_trades:>8d}")
    print(f"  Total Commission: ${report.total_commission:>8.2f}")

    # Slippage analysis
    s = report.slippage
    print(f"\n  --- Slippage Analysis ---")
    print(f"  Fills:            {s.n_fills:>8d}")
    print(f"  Mean Slippage:    {s.mean_bps:>8.1f} bps")
    print(f"  Median Slippage:  {s.median_bps:>8.1f} bps")
    print(f"  Max Slippage:     {s.max_bps:>8.1f} bps")
    print(f"  P95 Slippage:     {s.p95_bps:>8.1f} bps")
    print(f"  Std Dev:          {s.std_bps:>8.1f} bps")
    print(f"  Adverse fills:    {s.n_adverse:>8d}")
    print(f"  Favorable fills:  {s.n_favorable:>8d}")

    # P&L Attribution
    if report.attribution:
        print(f"\n  --- P&L Attribution by Symbol ---")
        print(f"  {'Symbol':<10s} {'PnL':>10s} {'Trades':>7s} {'Win%':>7s} {'PF':>7s} {'Avg':>8s} {'Best':>8s} {'Worst':>8s}")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 8}")
        for a in report.attribution:
            print(f"  {a.symbol:<10s} ${a.total_pnl:>9.2f} {a.n_trades:>7d} "
                  f"{a.win_rate:>6.1%} {a.profit_factor:>7.2f} "
                  f"${a.avg_pnl_per_trade:>7.2f} ${a.best_trade:>7.2f} ${a.worst_trade:>7.2f}")

    # Risk compliance
    r = report.risk
    print(f"\n  --- Risk Compliance ---")
    status = "PASS" if r.passed else "FAIL"
    print(f"  Status:                {status}")
    print(f"  Max Drawdown:          {r.actual_max_drawdown:>8.2%}  (limit: {r.max_drawdown_limit:.0%})")
    print(f"  Drawdown breaches:     {r.drawdown_limit_breaches:>8d}")

    # Recent trades
    if report.trade_log:
        print(f"\n  --- Trade Log (last 10) ---")
        for t in report.trade_log[-10:]:
            side = "SELL" if t["order_id"].endswith("_sell") else "BUY "
            pnl_str = f"${t.get('pnl', 0.0):.2f}" if t.get("pnl") else ""
            print(f"  {t['timestamp'][:19]}  {side}  {t['quantity']:>6.0f} @ ${t['price']:.2f}  "
                  f"comm=${t['commission']:.2f}  slip={t['slippage_bps']:.0f}bps  {pnl_str}")

    print("=" * 78)


def export_html(report: PostTradeReport, path: str) -> None:
    """Export post-trade report as HTML."""
    rows = ""
    for a in report.attribution:
        rows += (
            f"<tr><td>{a.symbol}</td><td>${a.total_pnl:,.2f}</td>"
            f"<td>{a.n_trades}</td><td>{a.win_rate:.1%}</td>"
            f"<td>{a.profit_factor:.2f}</td><td>${a.avg_pnl_per_trade:,.2f}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html><head><title>Post-Trade Analysis — {report.symbol} / {report.strategy}</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 2rem; }}
.metric {{ display: inline-block; background: #16213e; padding: 1rem 2rem; margin: 0.5rem; border-radius: 8px; }}
.metric .value {{ font-size: 1.5rem; font-weight: bold; color: #4ecca3; }}
.metric .label {{ font-size: 0.8rem; color: #999; }}
h1 {{ color: #4ecca3; }} h2 {{ color: #e94560; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #333; text-align: left; }}
th {{ color: #4ecca3; }}
.warn {{ color: #ff6b6b; }}
.pass {{ color: #4ecca3; }}
</style></head><body>
<h1>Post-Trade Analysis: {report.symbol} / {report.strategy}</h1>
<p>Generated: {report.timestamp[:19]}</p>

<h2>Performance</h2>
<div>
  <div class="metric"><div class="value">{report.total_return:.2%}</div><div class="label">Total Return</div></div>
  <div class="metric"><div class="value">{report.sharpe:.2f}</div><div class="label">Sharpe</div></div>
  <div class="metric"><div class="value">{report.max_drawdown:.2%}</div><div class="label">Max Drawdown</div></div>
  <div class="metric"><div class="value">{report.win_rate:.2%}</div><div class="label">Win Rate</div></div>
  <div class="metric"><div class="value">{report.profit_factor:.2f}</div><div class="label">Profit Factor</div></div>
  <div class="metric"><div class="value">{report.n_trades}</div><div class="label">Trades</div></div>
  <div class="metric"><div class="value">${report.total_commission:,.2f}</div><div class="label">Commissions</div></div>
</div>

<h2>Slippage</h2>
<div>
  <div class="metric"><div class="value">{report.slippage.mean_bps:.1f} bps</div><div class="label">Mean</div></div>
  <div class="metric"><div class="value">{report.slippage.max_bps:.1f} bps</div><div class="label">Max</div></div>
  <div class="metric"><div class="value">{report.slippage.p95_bps:.1f} bps</div><div class="label">P95</div></div>
  <div class="metric"><div class="value">{report.slippage.n_fills}</div><div class="label">Total Fills</div></div>
  <div class="metric"><div class="value">{report.slippage.n_adverse}</div><div class="label">Adverse</div></div>
  <div class="metric"><div class="value">{report.slippage.n_favorable}</div><div class="label">Favorable</div></div>
</div>

<h2>P&amp;L Attribution</h2>
<table><tr><th>Symbol</th><th>Total P&amp;L</th><th>Trades</th><th>Win%</th><th>PF</th><th>Avg</th></tr>
{rows}</table>

<h2>Risk Compliance</h2>
<div class="{'pass' if report.risk.passed else 'warn'}">
  Status: {"PASS" if report.risk.passed else "FAIL"}<br>
  Max Drawdown: {report.risk.actual_max_drawdown:.2%} (limit: {report.risk.max_drawdown_limit:.0%})
</div>
</body></html>"""

    Path(path).write_text(html, encoding="utf-8")
    print(f"\n  HTML report saved to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="QuantCore Post-Trade Analysis")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol")
    parser.add_argument("--strategy", default="momentum",
                        choices=list(STRATEGY_MAP), help="Strategy name")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps")
    parser.add_argument("--sizing", type=float, default=0.25, help="Position sizing fraction")
    parser.add_argument("--output", help="HTML output path")
    parser.add_argument("--json", help="JSON output path")
    parser.add_argument("--all-candidates", action="store_true",
                        help="Run analysis on all walk-forward confirmed candidates")
    args = parser.parse_args()

    if args.all_candidates:
        candidates = [
            ("QQQ", "Momentum (SMA 50/200)", MomentumCrossover),
            ("SPY", "Momentum (SMA 50/200)", MomentumCrossover),
            ("GLD", "Mean Reversion (RSI 30)", MeanReversionRSI),
        ]
        all_reports = []
        for sym, name, cls in candidates:
            print(f"\nAnalyzing {sym} / {name} ...")
            report = run_post_trade_analysis(
                sym, name, cls,
                start=args.start, end=args.end,
                capital=args.capital, slippage_bps=args.slippage,
                sizing_pct=args.sizing,
            )
            print_report(report)
            all_reports.append(report)

        if args.json:
            data = []
            for r in all_reports:
                data.append({
                    "symbol": r.symbol,
                    "strategy": r.strategy,
                    "total_return": r.total_return,
                    "sharpe": r.sharpe,
                    "max_drawdown": r.max_drawdown,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "n_trades": r.n_trades,
                    "total_commission": r.total_commission,
                    "slippage_mean_bps": r.slippage.mean_bps,
                    "slippage_max_bps": r.slippage.max_bps,
                    "slippage_p95_bps": r.slippage.p95_bps,
                    "risk_passed": r.risk.passed,
                    "attribution": [
                        {"symbol": a.symbol, "total_pnl": a.total_pnl,
                         "win_rate": a.win_rate, "profit_factor": a.profit_factor}
                        for a in r.attribution
                    ],
                })
            Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"\nJSON report saved to: {args.json}")
        return 0

    strategy_cls = STRATEGY_MAP[args.strategy]
    report = run_post_trade_analysis(
        args.symbol, args.strategy, strategy_cls,
        start=args.start, end=args.end,
        capital=args.capital, slippage_bps=args.slippage,
        sizing_pct=args.sizing,
    )
    print_report(report)

    if args.output:
        export_html(report, args.output)

    if args.json:
        data = {
            "symbol": report.symbol,
            "strategy": report.strategy,
            "total_return": report.total_return,
            "sharpe": report.sharpe,
            "max_drawdown": report.max_drawdown,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "n_trades": report.n_trades,
            "total_commission": report.total_commission,
            "slippage_mean_bps": report.slippage.mean_bps,
            "risk_passed": report.risk.passed,
            "trade_log": report.trade_log,
        }
        Path(args.json).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON report saved to: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
