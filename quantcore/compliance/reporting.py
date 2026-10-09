"""Regulatory compliance reporting: VaR backtesting, best execution, audit trails.

Wired to quantcore.risk.var_backtest for Basel traffic light reports and
quantcore.execution.tca for best execution analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from quantcore.execution.tca import tca_strategy
from quantcore.risk.metrics import compute_all_metrics
from quantcore.risk.var_backtest import (
    var_backtest_report,
)


@dataclass
class RegulatoryReport:
    """Complete regulatory compliance report."""
    report_date: str = ""
    period_start: str = ""
    period_end: str = ""

    # VaR backtesting
    var_method: str = ""
    var_confidence: float = 0.99
    var_violations: int = 0
    var_expected: float = 0.0
    var_traffic_light: str = "green"
    var_kupiec_pvalue: float = 1.0
    var_christoffersen_pvalue: float = 1.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0

    # Exposure
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0

    # Trading activity
    n_trades: int = 0
    turnover: float = 0.0
    avg_trade_size: float = 0.0

    # Best execution
    avg_slippage_bps: float = 0.0
    best_exec_compliant: bool = True

    # Flags
    flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def generate_regulatory_report(
    returns: list[float],
    var_series: list[float],
    var_confidence: float = 0.99,
    positions: dict[str, float] | None = None,
    portfolio_value: float = 0.0,
    trades: list[dict[str, Any]] | None = None,
) -> RegulatoryReport:
    """Generate a complete regulatory compliance report.

    Args:
        returns: Daily portfolio returns.
        var_series: Daily VaR forecasts (same length as returns).
        var_confidence: VaR confidence level (0.99 for Basel).
        positions: Current positions {symbol: market_value}.
        portfolio_value: Total portfolio value.
        trades: Recent trades for best execution analysis.

    Returns:
        RegulatoryReport with all compliance metrics.
    """
    r = np.array(returns, dtype=float)
    v = np.array(var_series, dtype=float)
    now = datetime.utcnow()

    report = RegulatoryReport(
        report_date=now.strftime("%Y-%m-%d"),
        period_start=(now - timedelta(days=len(r))).strftime("%Y-%m-%d"),
        period_end=now.strftime("%Y-%m-%d"),
    )

    # VaR backtesting (Basel traffic light)
    if len(r) >= 20 and len(v) >= 20:
        var_report = var_backtest_report(r, v, confidence=var_confidence)
        report.var_method = "Mixed"
        report.var_confidence = var_confidence
        report.var_violations = var_report.n_violations
        report.var_expected = var_report.expected_rate
        report.var_traffic_light = var_report.traffic_light
        report.var_kupiec_pvalue = var_report.kupiec_pvalue
        report.var_christoffersen_pvalue = var_report.christoffersen_pvalue

        # Basel zone classification
        if var_report.traffic_light == "red":
            report.flags.append("VaR model in RED zone — requires immediate remediation")
            report.recommendations.append("Increase VaR multiplier or recalibrate model")
        elif var_report.traffic_light == "yellow":
            report.flags.append("VaR model in YELLOW zone — enhanced monitoring required")

    # Risk metrics
    metrics = compute_all_metrics(r)
    report.sharpe_ratio = metrics.get("sharpe", 0)
    report.max_drawdown = metrics.get("max_drawdown", 0)
    report.volatility = float(np.std(r) * np.sqrt(252)) if len(r) > 1 else 0

    # Exposures
    if positions:
        gross = sum(abs(v) for v in positions.values())
        net = sum(v for v in positions.values())
        report.gross_exposure = gross / portfolio_value if portfolio_value > 0 else 0
        report.net_exposure = net / portfolio_value if portfolio_value > 0 else 0
        report.leverage = report.gross_exposure

        if report.leverage > 3.0:
            report.flags.append(f"Leverage {report.leverage:.1f}x exceeds 3.0x threshold")
        if abs(report.net_exposure) > 1.5:
            report.flags.append(f"Net exposure {report.net_exposure:.1%} exceeds 150%")

    # Trading activity
    if trades:
        report.n_trades = len(trades)
        trade_values = [
            abs(float(t.get("quantity", 0)) * float(t.get("price", 0)))
            for t in trades
        ]
        total_traded = sum(trade_values)
        report.turnover = total_traded / portfolio_value if portfolio_value > 0 else 0
        report.avg_trade_size = total_traded / len(trades) if trades else 0

        # Best execution from TCA
        try:
            tca_result = tca_strategy(trades, strategy_name="arrival")
            report.avg_slippage_bps = float(tca_result.avg_cost_bps)
            report.best_exec_compliant = abs(report.avg_slippage_bps) <= 50
            if not report.best_exec_compliant:
                report.flags.append(f"Best execution slippage {report.avg_slippage_bps:.1f} bps exceeds 50 bps")
                report.recommendations.append("Review execution algorithms and broker routing")
        except Exception:
            pass

    # Additional risk flags
    if report.max_drawdown < -0.20:
        report.flags.append(f"Maximum drawdown {report.max_drawdown:.1%} exceeds 20%")
    if report.var_violations > 10:
        report.recommendations.append("VaR model recalibration recommended — review parameters")

    return report


def generate_best_execution_report(
    trades: list[dict[str, Any]],
    benchmark: str = "arrival",
) -> dict[str, Any]:
    """Generate a best execution compliance report (SEC Rule 606 / MiFID II).

    Args:
        trades: List of trade dicts with symbol, side, quantity, price, timestamp.
        benchmark: "arrival", "vwap", or "close".

    Returns:
        Dict with execution quality metrics.
    """
    if not trades:
        return {"compliant": True, "message": "No trades to analyze"}

    result: dict[str, Any] = {
        "compliant": True,
        "benchmark": benchmark,
        "n_trades": len(trades),
        "flags": [],
    }

    total_slippage = 0.0
    venue_stats: dict[str, list[float]] = {}

    for trade in trades:
        price = float(trade.get("price", 0))
        benchmark_price = float(trade.get("benchmark_price", price))
        venue = str(trade.get("venue", "unknown"))

        if benchmark_price > 0:
            slippage = (price - benchmark_price) / benchmark_price * 10_000
            total_slippage += abs(slippage)

            if venue not in venue_stats:
                venue_stats[venue] = []
            venue_stats[venue].append(slippage)

    result["avg_slippage_bps"] = total_slippage / len(trades) if trades else 0

    # Per-venue breakdown
    result["venues"] = {}
    for venue, slips in venue_stats.items():
        avg_slip = float(np.mean(slips))
        result["venues"][venue] = {
            "n_trades": len(slips),
            "avg_slippage_bps": avg_slip,
            "compliant": abs(avg_slip) <= 50,
        }

    # Overall compliance
    if result["avg_slippage_bps"] > 50:
        result["compliant"] = False
        result["flags"].append(f"Average slippage {result['avg_slippage_bps']:.1f} bps exceeds 50 bps threshold")

    # Check venue outliers
    for venue, stats in result["venues"].items():
        if not stats["compliant"]:
            result["flags"].append(f"Venue {venue}: {stats['avg_slippage_bps']:.1f} bps avg slippage")

    return result
