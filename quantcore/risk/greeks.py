"""Greeks aggregation: net delta, gamma, vega, theta with tenor bucketing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from quantcore.core.types import PortfolioSnapshot, Position


@dataclass
class GreeksSnapshot:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_delta: float = 0.0
    total_delta_dollar: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0
    total_theta_daily: float = 0.0
    delta_by_underlying: dict[str, float] = field(default_factory=dict)
    gamma_by_tenor: dict[str, float] = field(default_factory=dict)
    vega_by_tenor: dict[str, float] = field(default_factory=dict)
    cross_gamma: dict[str, float] = field(default_factory=dict)
    delta_buckets: dict[str, float] = field(default_factory=dict)  # delta exposure by underlying
    net_exposure: float = 0.0  # Total delta-adjusted exposure


# Default DTE buckets
DTE_BUCKETS = [
    (0, 7, "0-7d"),
    (7, 30, "7-30d"),
    (30, 90, "30-90d"),
    (90, 365, "90+d"),
]

VOL_SHOCK_LEVELS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Option position representation for Greeks calc
# ---------------------------------------------------------------------------


def _parse_option_position(
    symbol: str,
    pos: Position,
) -> dict[str, Any] | None:
    """Attempt to parse an option position from symbol and position metadata.

    Returns dict with keys: underlying, option_type, strike, expiration, qty, etc.
    or None if not an option.
    """
    # Simple heuristic: options have longer symbols or contain specific patterns
    # In production, use a proper options symbol parser
    qty = float(pos.quantity)

    # Check for OCC-style option symbols
    if len(symbol) > 5 or symbol.startswith("O:"):
        # Parsing attempt
        parts = symbol.replace("O:", "").split("_")
        if len(parts) >= 4:
            return {
                "underlying": parts[0],
                "expiration": parts[1],
                "option_type": parts[2][0].upper(),  # C/P
                "strike": float(parts[2][1:]),
                "qty": qty,
                "price": float(pos.market_price),
            }
    return None


# ---------------------------------------------------------------------------
# T-0309: Greeks Aggregation
# ---------------------------------------------------------------------------


def aggregate_greeks(
    portfolio: PortfolioSnapshot,
    stock_delta_adjustments: dict[str, float] | None = None,
    option_greeks: dict[str, dict[str, float]] | None = None,
) -> GreeksSnapshot:
    """Aggregate Greeks across all positions.

    For stock positions: delta = quantity, gamma = vega = theta = 0.
    For options: use provided greeks or approximate from BSM.

    Args:
        portfolio: Current portfolio state.
        stock_delta_adjustments: Optional per-symbol delta override.
        option_greeks: Optional {symbol: {delta, gamma, vega, theta}}.

    Returns:
        GreeksSnapshot with aggregated values.
    """
    snapshot = GreeksSnapshot()
    stock_delta_adj = stock_delta_adjustments or {}
    opt_greeks = option_greeks or {}

    for symbol, pos in portfolio.positions.items():
        qty = float(pos.quantity)
        price = float(pos.market_price) if float(pos.market_price) > 0 else 100.0

        # Check if this is an option
        opt_info = _parse_option_position(symbol, pos)

        if opt_info is not None:
            underlying = opt_info["underlying"]
            otype = opt_info["option_type"]
            opt_info["strike"]
            expiry = opt_info["expiration"]

            # Use provided greeks or default to simple approximation
            if symbol in opt_greeks:
                delta = opt_greeks[symbol].get("delta", 0.0)
                gamma = opt_greeks[symbol].get("gamma", 0.0)
                vega = opt_greeks[symbol].get("vega", 0.0)
                theta = opt_greeks[symbol].get("theta", 0.0)
            else:
                # Rough approximation
                delta = 0.5 if otype == "C" else -0.5
                gamma = 0.01
                vega = 0.1
                theta = -0.02 if otype == "C" else -0.01

            # DTE for tenor bucketing
            try:
                exp_date = date.fromisoformat(expiry[:10])
                dte = max(0, (exp_date - date.today()).days)
            except (ValueError, IndexError):
                dte = 30

            # Contract multiplier (100 for standard equity options)
            multiplier = 100.0

            snapshot.total_delta += delta * qty * multiplier
            snapshot.total_delta_dollar += delta * qty * multiplier * price
            snapshot.total_gamma += gamma * qty * multiplier
            snapshot.total_vega += vega * qty * multiplier
            snapshot.total_theta += theta * qty * multiplier

            # Underlying delta
            current = snapshot.delta_by_underlying.get(underlying, 0.0)
            snapshot.delta_by_underlying[underlying] = current + delta * qty * multiplier

            # Tenor buckets
            for low, high, label in DTE_BUCKETS:
                if low <= dte < high:
                    snapshot.gamma_by_tenor[label] = snapshot.gamma_by_tenor.get(label, 0.0) + gamma * qty * multiplier
                    snapshot.vega_by_tenor[label] = snapshot.vega_by_tenor.get(label, 0.0) + vega * qty * multiplier
                    break
        else:
            # Stock position — delta = 1 per share
            adj = stock_delta_adj.get(symbol, 1.0)
            snapshot.total_delta += adj * qty
            snapshot.total_delta_dollar += adj * qty * price
            current = snapshot.delta_by_underlying.get(symbol, 0.0)
            snapshot.delta_by_underlying[symbol] = current + adj * qty

    snapshot.total_theta_daily = snapshot.total_theta / 252.0 if snapshot.total_theta != 0 else 0.0
    snapshot.net_exposure = snapshot.total_delta_dollar
    return snapshot


def delta_adjusted_exposure(
    stock_positions: list[Position],
    option_positions: list[dict[str, Any]],
) -> dict[str, float]:
    """Calculate total delta per underlying.

    Args:
        stock_positions: List of stock positions.
        option_positions: List of option position dicts with 'underlying', 'qty', 'delta'.

    Returns:
        {underlying: total_delta}
    """
    exposure: dict[str, float] = {}

    for pos in stock_positions:
        val = float(pos.market_value)
        exposure[pos.symbol] = exposure.get(pos.symbol, 0.0) + val

    for opt in option_positions:
        sym = opt["underlying"]
        qty = opt.get("qty", 0)
        delta = opt.get("delta", 0)
        multiplier = 100.0
        exposure[sym] = exposure.get(sym, 0.0) + delta * qty * multiplier

    return exposure


def gamma_by_tenor(
    option_positions: list[dict[str, Any]],
    buckets: list[tuple[int, int, str]] | None = None,
) -> dict[str, float]:
    """Aggregate gamma by DTE bucket."""
    if buckets is None:
        buckets = DTE_BUCKETS

    result: dict[str, float] = {}
    for opt in option_positions:
        dte = opt.get("dte", 30)
        gamma = opt.get("gamma", 0.0)
        qty = opt.get("qty", 0)
        multiplier = 100.0

        for low, high, label in buckets:
            if low <= dte < high:
                result[label] = result.get(label, 0.0) + gamma * qty * multiplier
                break

    return result


def vega_by_tenor(
    option_positions: list[dict[str, Any]],
    buckets: list[tuple[int, int, str]] | None = None,
) -> dict[str, float]:
    """Aggregate vega by DTE bucket."""
    if buckets is None:
        buckets = DTE_BUCKETS

    result: dict[str, float] = {}
    for opt in option_positions:
        dte = opt.get("dte", 30)
        vega = opt.get("vega", 0.0)
        qty = opt.get("qty", 0)
        multiplier = 100.0

        for low, high, label in buckets:
            if low <= dte < high:
                result[label] = result.get(label, 0.0) + vega * qty * multiplier
                break

    return result


def greeks_stress(
    positions: dict[str, Position],
    spot_shocks: list[float] | None = None,
    option_greeks_fn: Any | None = None,
) -> dict[str, list[float]]:
    """Compute Greeks at each spot shock level (±1σ, ±2σ, ±3σ).

    Returns:
        {shock_level: [delta, gamma, vega, theta]} or DataFrame-like dict.
    """
    if spot_shocks is None:
        spot_shocks = VOL_SHOCK_LEVELS

    result: dict[str, list[float]] = {
        "shock_sigma": [],
        "delta": [],
        "gamma": [],
        "vega": [],
        "theta": [],
    }

    for shock in spot_shocks:
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0

        for symbol, pos in positions.items():
            qty = float(pos.quantity)
            price = float(pos.market_price) if float(pos.market_price) > 0 else 100.0
            shocked_price = price * (1.0 + shock * 0.01)

            opt_info = _parse_option_position(symbol, pos)
            if opt_info is not None and option_greeks_fn is not None:
                greeks = option_greeks_fn(symbol, shocked_price)
                total_delta += greeks.get("delta", 0) * qty * 100
                total_gamma += greeks.get("gamma", 0) * qty * 100
                total_vega += greeks.get("vega", 0) * qty * 100
                total_theta += greeks.get("theta", 0) * qty * 100
            elif opt_info is not None:
                total_delta += 0.5 * qty * 100
                total_gamma += 0.01 * qty * 100
                total_vega += 0.1 * qty * 100
            else:
                total_delta += qty

        result["shock_sigma"].append(shock)
        result["delta"].append(total_delta)
        result["gamma"].append(total_gamma)
        result["vega"].append(total_vega)
        result["theta"].append(total_theta)

    return result
