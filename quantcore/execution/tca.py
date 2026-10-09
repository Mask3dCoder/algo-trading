"""Transaction Cost Analysis: arrival price, cost decomposition, reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from quantcore.core.types import Fill


class CostCategory(str, Enum):
    SPREAD = "spread"
    IMPACT = "impact"
    TIMING = "timing"
    OPPORTUNITY = "opportunity"


@dataclass
class TCAResult:
    """TCA result for a single trade."""
    symbol: str = ""
    side: str = ""
    shares: float = 0.0
    arrival_price: float = 0.0
    avg_fill_price: float = 0.0
    market_vwap: float = 0.0
    total_cost_bps: float = 0.0
    spread_cost_bps: float = 0.0
    impact_cost_bps: float = 0.0
    timing_cost_bps: float = 0.0
    opportunity_cost_bps: float = 0.0
    total_cost_dollars: float = 0.0
    strategy: str = ""
    venue: str = ""
    time_of_day: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategySummary:
    strategy_name: str
    n_trades: int = 0
    total_notional: float = 0.0
    total_cost_bps: float = 0.0
    avg_cost_bps: float = 0.0
    spread_cost_bps: float = 0.0
    impact_cost_bps: float = 0.0
    timing_cost_bps: float = 0.0
    opportunity_cost_bps: float = 0.0


# ---------------------------------------------------------------------------
# Single-trade TCA
# ---------------------------------------------------------------------------


def tca_trade(
    fills: list[Fill],
    arrival_price: float,
    side: str = "buy",
    market_vwap: float | None = None,
    half_spread_bps: float = 5.0,
    expected_impact_bps: float = 0.0,
) -> TCAResult:
    """Analyze cost of a single trade.

    total_cost_bps = (avg_fill - arrival) / arrival * 10000 × sign(side)

    Cost decomposition:
    - spread_cost = half_spread × sign (paid to cross spread)
    - impact_cost = expected impact from model
    - timing_cost = remainder (unexplained)
    - opportunity_cost = 0 if fully filled, otherwise × unfilled

    Args:
        fills: List of fill records for this trade.
        arrival_price: Price at order decision time.
        side: "buy" or "sell".
        market_vwap: Market VWAP during execution.
        half_spread_bps: Half-spread in bps.
        expected_impact_bps: Predicted impact from impact model.

    Returns:
        TCAResult with cost decomposition.
    """
    if not fills:
        return TCAResult()

    side_mult = 1 if side.lower() == "buy" else -1

    total_qty = sum(float(f.quantity) for f in fills)
    total_value = sum(float(f.price) * float(f.quantity) for f in fills)
    avg_fill = total_value / total_qty if total_qty > 0 else 0.0

    # Total cost in bps
    total_cost_bps = (avg_fill - arrival_price) / arrival_price * 10_000 * side_mult

    # Spread cost
    spread_cost_bps = half_spread_bps * side_mult

    # Impact cost (from model)
    impact_cost_bps = expected_impact_bps * side_mult

    # Timing cost = total - spread - impact
    timing_cost_bps = total_cost_bps - spread_cost_bps - impact_cost_bps

    # Total cost in dollars
    total_cost_dollars = abs(total_cost_bps / 10_000 * total_value)

    symbol = fills[0].order_id if fills else ""

    # Time of day
    ts = fills[0].timestamp if fills else datetime.utcnow()
    hour = ts.hour
    if hour < 11:
        tod = "morning"
    elif hour < 14:
        tod = "midday"
    else:
        tod = "afternoon"

    return TCAResult(
        symbol=symbol,
        side=side,
        shares=total_qty,
        arrival_price=arrival_price,
        avg_fill_price=avg_fill,
        market_vwap=market_vwap or 0.0,
        total_cost_bps=total_cost_bps,
        spread_cost_bps=spread_cost_bps,
        impact_cost_bps=impact_cost_bps,
        timing_cost_bps=timing_cost_bps,
        total_cost_dollars=total_cost_dollars,
        time_of_day=tod,
    )


# ---------------------------------------------------------------------------
# Strategy-level TCA
# ---------------------------------------------------------------------------


def tca_strategy(
    trades: list[TCAResult],
    strategy_name: str = "",
) -> StrategySummary:
    """Aggregate TCA results by strategy."""
    if not trades:
        return StrategySummary(strategy_name=strategy_name)

    summary = StrategySummary(
        strategy_name=strategy_name or trades[0].strategy,
        n_trades=len(trades),
    )

    for t in trades:
        summary.total_notional += t.shares * t.arrival_price
        summary.total_cost_bps += t.total_cost_bps * (t.shares * t.arrival_price)
        summary.spread_cost_bps += t.spread_cost_bps * (t.shares * t.arrival_price)
        summary.impact_cost_bps += t.impact_cost_bps * (t.shares * t.arrival_price)
        summary.timing_cost_bps += t.timing_cost_bps * (t.shares * t.arrival_price)

    if summary.total_notional > 0:
        summary.total_cost_bps /= summary.total_notional
        summary.spread_cost_bps /= summary.total_notional
        summary.impact_cost_bps /= summary.total_notional
        summary.timing_cost_bps /= summary.total_notional

    summary.avg_cost_bps = summary.total_cost_bps

    return summary


# ---------------------------------------------------------------------------
# TCA Report
# ---------------------------------------------------------------------------


def tca_report(
    all_trades: list[TCAResult],
    period: str = "",
) -> dict[str, Any]:
    """Generate a TCA report by strategy, venue, and time-of-day.

    Returns dict suitable for DataFrame conversion or HTML rendering.
    """
    # By strategy
    by_strategy: dict[str, list[TCAResult]] = {}
    by_venue: dict[str, list[TCAResult]] = {}
    by_tod: dict[str, list[TCAResult]] = {}

    for t in all_trades:
        strat = t.strategy or "unclassified"
        venue = t.venue or "unknown"
        tod = t.time_of_day or "unknown"

        by_strategy.setdefault(strat, []).append(t)
        by_venue.setdefault(venue, []).append(t)
        by_tod.setdefault(tod, []).append(t)

    report = {
        "period": period,
        "n_trades": len(all_trades),
        "total_notional": sum(t.shares * t.arrival_price for t in all_trades),
        "total_cost_bps": 0.0,
        "total_cost_dollars": sum(t.total_cost_dollars for t in all_trades),
        "by_strategy": {},
        "by_venue": {},
        "by_time_of_day": {},
    }

    # Aggregate costs
    notional = sum(t.shares * t.arrival_price for t in all_trades)
    if notional > 0:
        report["total_cost_bps"] = sum(
            t.total_cost_bps * (t.shares * t.arrival_price) for t in all_trades
        ) / notional

    for name, trades in by_strategy.items():
        report["by_strategy"][name] = tca_strategy(trades, name)

    for name, trades in by_venue.items():
        report["by_venue"][name] = tca_strategy(trades, name)

    for name, trades in by_tod.items():
        report["by_time_of_day"][name] = tca_strategy(trades, name)

    return report


def tca_report_dataframe(tca_report_dict: dict[str, Any]) -> dict[str, list]:
    """Convert TCA report to DataFrame-compatible dict."""
    rows = {
        "strategy": [],
        "n_trades": [],
        "total_notional": [],
        "total_cost_bps": [],
        "spread_cost_bps": [],
        "impact_cost_bps": [],
        "timing_cost_bps": [],
    }

    for name, summary in tca_report_dict.get("by_strategy", {}).items():
        rows["strategy"].append(name)
        rows["n_trades"].append(summary.n_trades)
        rows["total_notional"].append(summary.total_notional)
        rows["total_cost_bps"].append(summary.total_cost_bps)
        rows["spread_cost_bps"].append(summary.spread_cost_bps)
        rows["impact_cost_bps"].append(summary.impact_cost_bps)
        rows["timing_cost_bps"].append(summary.timing_cost_bps)

    return rows
