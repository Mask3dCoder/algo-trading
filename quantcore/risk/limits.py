"""Position and portfolio limits with circuit breakers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from quantcore.core.types import Order, Position, PortfolioSnapshot
from quantcore.core.exceptions import RiskLimitBreached


class LimitStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class LimitCheckResult:
    status: LimitStatus
    reason: str = ""
    limit_name: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskLimits:
    """All risk limits, loadable from config."""

    # Position-level limits
    max_position_pct_adv: float = 0.05  # Max 5% of ADV per position
    max_dollar_exposure_per_name: float = 1_000_000.0  # Max $1M per name
    max_loss_per_position: float = -50_000.0  # Max loss per position

    # Portfolio-level limits
    max_net_delta: float = 500_000.0  # Max net delta dollars
    max_gross_gamma: float = 100_000.0  # Max absolute gamma
    max_vega: float = 500_000.0  # Max vega
    max_sector_exposure_pct: float = 0.40  # Max 40% in any sector
    max_concentration_pct: float = 0.25  # Max 25% in single name

    # VaR limits
    max_var_utilization_pct: float = 0.80  # Max 80% of VaR budget

    # Margin
    max_margin_utilization_pct: float = 0.85  # Max 85% of buying power

    # Circuit breaker
    circuit_breaker_threshold_pct: float = 0.20  # 20% over limit → breaker
    circuit_breaker_action: str = "flatten"  # "flatten" or "reduce_50"


# ---------------------------------------------------------------------------
# Pre-trade limit check
# ---------------------------------------------------------------------------


def check_pre_trade(
    order: Order,
    portfolio: PortfolioSnapshot,
    limits: RiskLimits,
    adv_data: dict[str, float] | None = None,
    current_greeks: dict[str, float] | None = None,
) -> LimitCheckResult:
    """Check all limits before submitting an order.

    Args:
        order: The proposed order.
        portfolio: Current portfolio snapshot.
        limits: Risk limits configuration.
        adv_data: {symbol: average_daily_volume_dollars}.
        current_greeks: Aggregated Greeks for the portfolio.

    Returns:
        LimitCheckResult: APPROVED or REJECTED with reason.
    """
    qty = float(order.quantity)
    order_value = qty * float(order.limit_price or 100.0)

    # 1. Concentration check
    total_value = float(portfolio.total_value)
    if total_value > 0:
        existing_val = 0.0
        if order.symbol in portfolio.positions:
            existing_val = float(abs(portfolio.positions[order.symbol].market_value))
        new_concentration = (existing_val + order_value) / total_value
        if new_concentration > limits.max_concentration_pct:
            return LimitCheckResult(
                status=LimitStatus.REJECTED,
                reason=f"Concentration limit: {new_concentration:.1%} > {limits.max_concentration_pct:.1%} for {order.symbol}",
                limit_name="max_concentration",
                current_value=new_concentration,
                limit_value=limits.max_concentration_pct,
            )

    # 2. Dollar exposure per name
    if order_value > limits.max_dollar_exposure_per_name:
        return LimitCheckResult(
            status=LimitStatus.REJECTED,
            reason=f"Max dollar exposure: ${order_value:,.0f} > ${limits.max_dollar_exposure_per_name:,.0f}",
            limit_name="max_dollar_exposure",
            current_value=order_value,
            limit_value=limits.max_dollar_exposure_per_name,
        )

    # 3. ADV-based position limit
    if adv_data and order.symbol in adv_data:
        adv = adv_data[order.symbol]
        pct_adv = order_value / adv if adv > 0 else 999
        if pct_adv > limits.max_position_pct_adv:
            return LimitCheckResult(
                status=LimitStatus.REJECTED,
                reason=f"ADV limit: {pct_adv:.1%} of ADV > {limits.max_position_pct_adv:.1%}",
                limit_name="max_position_pct_adv",
                current_value=pct_adv,
                limit_value=limits.max_position_pct_adv,
            )

    # 4. Greeks limits (if available)
    if current_greeks:
        net_delta = current_greeks.get("total_delta_dollar", 0.0)
        new_delta = net_delta + (qty * float(order.limit_price or 100.0))
        if abs(new_delta) > limits.max_net_delta:
            return LimitCheckResult(
                status=LimitStatus.REJECTED,
                reason=f"Net delta limit: ${abs(new_delta):,.0f} > ${limits.max_net_delta:,.0f}",
                limit_name="max_net_delta",
                current_value=abs(new_delta),
                limit_value=limits.max_net_delta,
            )

    return LimitCheckResult(status=LimitStatus.APPROVED)


# ---------------------------------------------------------------------------
# Post-trade limit recheck
# ---------------------------------------------------------------------------


def check_post_trade(
    portfolio: PortfolioSnapshot,
    limits: RiskLimits,
    current_greeks: dict[str, float] | None = None,
) -> list[LimitCheckResult]:
    """Recheck all limits after a fill. Returns list of breaches."""
    breaches: list[LimitCheckResult] = []
    total_value = float(portfolio.total_value)

    if total_value <= 0:
        return breaches

    # Concentration
    for symbol, pos in portfolio.positions.items():
        concentration = abs(float(pos.market_value)) / total_value
        if concentration > limits.max_concentration_pct:
            breaches.append(LimitCheckResult(
                status=LimitStatus.REJECTED,
                reason=f"Concentration breach: {symbol} at {concentration:.1%}",
                limit_name="max_concentration",
                current_value=concentration,
                limit_value=limits.max_concentration_pct,
            ))

    # Greeks
    if current_greeks:
        for key, limit_val, limit_name in [
            ("total_delta_dollar", limits.max_net_delta, "max_net_delta"),
            ("total_gamma", limits.max_gross_gamma, "max_gross_gamma"),
            ("total_vega", limits.max_vega, "max_vega"),
        ]:
            current = abs(current_greeks.get(key, 0.0))
            if current > limit_val:
                breaches.append(LimitCheckResult(
                    status=LimitStatus.REJECTED,
                    reason=f"Greeks limit breach: {key}={current:.0f} > {limit_val:.0f}",
                    limit_name=limit_name,
                    current_value=current,
                    limit_value=limit_val,
                ))

    return breaches


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreaker:
    """Auto-flatten on severe limit breaches."""

    limits: RiskLimits = field(default_factory=RiskLimits)
    activated: bool = False
    activation_reason: str = ""
    activation_time: datetime | None = None
    flattened_positions: list[str] = field(default_factory=list)
    on_activate: Callable[[str, list[str]], None] | None = None

    def check(self, breaches: list[LimitCheckResult]) -> bool:
        """Check if any breach triggers the circuit breaker.

        A breach > 20% over the limit triggers auto-flatten.
        """
        if self.activated:
            return True

        for breach in breaches:
            if breach.status == LimitStatus.REJECTED and breach.limit_value > 0:
                overage = (breach.current_value - breach.limit_value) / breach.limit_value
                if overage > self.limits.circuit_breaker_threshold_pct:
                    self.activated = True
                    self.activation_time = datetime.utcnow()
                    self.activation_reason = (
                        f"{breach.limit_name}: {breach.current_value:.2f} > "
                        f"{breach.limit_value:.2f} (+{overage:.1%})"
                    )
                    if self.on_activate:
                        self.on_activate(self.activation_reason, self.flattened_positions)
                    return True
        return False

    def reset(self) -> None:
        """Reset the circuit breaker (requires manual intervention)."""
        self.activated = False
        self.activation_reason = ""
        self.activation_time = None
        self.flattened_positions = []


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def check_order(
    order: Order,
    portfolio: PortfolioSnapshot,
    limits: RiskLimits | None = None,
    adv_data: dict[str, float] | None = None,
    current_greeks: dict[str, float] | None = None,
) -> LimitCheckResult:
    """Pre-trade limit check with defaults."""
    if limits is None:
        limits = RiskLimits()
    return check_pre_trade(order, portfolio, limits, adv_data, current_greeks)
