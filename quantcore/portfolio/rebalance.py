"""Rebalance engine: calendar/threshold triggers, cost-aware, turnover budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class RebalanceTrigger(str, Enum):
    CALENDAR = "calendar"
    THRESHOLD = "threshold"
    MANUAL = "manual"


@dataclass
class RebalanceConfig:
    """Rebalance engine configuration."""
    trigger: RebalanceTrigger = RebalanceTrigger.THRESHOLD
    calendar_frequency: str = "monthly"  # daily, weekly, monthly
    drift_threshold: float = 0.05  # 5% absolute weight drift triggers rebalance
    min_trade_value: float = 500.0  # Don't trade if value < $500
    cost_benefit_threshold: float = 2.0  # Benefit must be > 2× estimated cost
    turnover_limit: float = 0.30  # Max 30% turnover per rebalance
    tolerance: float = 0.01  # Allow 1% deviation from target


@dataclass
class RebalanceResult:
    rebalanced: bool = False
    trigger: str = ""
    trades: list[dict] = field(default_factory=list)
    turnover: float = 0.0
    estimated_cost: float = 0.0
    reason: str = ""


class RebalanceEngine:
    """Determines when to rebalance and computes the required trades."""

    def __init__(self, config: RebalanceConfig | None = None) -> None:
        self.config = config or RebalanceConfig()
        self._last_rebalance: datetime | None = None

    def should_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        current_date: datetime | None = None,
    ) -> tuple[bool, str]:
        """Check if rebalance should be triggered.

        Returns:
            (should_rebalance, reason)
        """
        cfg = self.config
        date = current_date or datetime.utcnow()

        # Calendar trigger
        if cfg.trigger == RebalanceTrigger.CALENDAR or cfg.trigger == "both":
            if self._last_rebalance is None:
                return True, "initial_rebalance"

            if cfg.calendar_frequency == "daily":
                next_date = self._last_rebalance + timedelta(days=1)
            elif cfg.calendar_frequency == "weekly":
                next_date = self._last_rebalance + timedelta(days=7)
            elif cfg.calendar_frequency == "monthly":
                next_date = self._last_rebalance + timedelta(days=30)
            else:
                next_date = self._last_rebalance + timedelta(days=30)

            if date >= next_date:
                return True, "calendar_trigger"

        # Threshold trigger
        if cfg.trigger in (RebalanceTrigger.THRESHOLD, "both"):
            all_symbols = set(current_weights.keys()) | set(target_weights.keys())
            max_drift = 0.0

            for sym in all_symbols:
                cw = current_weights.get(sym, 0.0)
                tw = target_weights.get(sym, 0.0)
                drift = abs(cw - tw)
                max_drift = max(max_drift, drift)

            if max_drift > cfg.drift_threshold:
                return True, f"threshold_trigger (max drift={max_drift:.3f})"

            # Also check if any weight outside tolerance band
            for sym in all_symbols:
                cw = current_weights.get(sym, 0.0)
                tw = target_weights.get(sym, 0.0)
                if abs(cw - tw) > cfg.tolerance:
                    pass  # Already caught by max_drift check

        return False, "within_limits"

    def compute_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        current_prices: dict[str, float] | None = None,
        cost_estimates: dict[str, float] | None = None,  # Estimated cost per trade in $
    ) -> list[dict]:
        """Compute the trades needed to rebalance.

        Returns:
            List of trade dicts: {symbol, side, shares, value, weight_delta, cost}
        """
        cfg = self.config
        all_symbols = sorted(set(current_weights.keys()) | set(target_weights.keys()))

        trades = []
        total_turnover_value = 0.0

        for sym in all_symbols:
            cw = current_weights.get(sym, 0.0)
            tw = target_weights.get(sym, 0.0)
            delta = tw - cw

            # Skip small deviations
            if abs(delta) <= cfg.tolerance:
                continue

            trade_value = abs(delta) * portfolio_value

            # Skip small trades
            if trade_value < cfg.min_trade_value:
                continue

            side = "buy" if delta > 0 else "sell"
            shares = int(trade_value / (current_prices.get(sym, 100.0) or 1.0)) if current_prices else 0
            estimated_cost = cost_estimates.get(sym, 5.0) if cost_estimates else 5.0  # Default $5

            # Cost-benefit check: only trade if benefit > cost × threshold
            benefit = abs(delta) * portfolio_value
            if benefit < estimated_cost * cfg.cost_benefit_threshold:
                continue

            total_turnover_value += trade_value

            trades.append({
                "symbol": sym,
                "side": side,
                "shares": shares,
                "value": trade_value,
                "weight_delta": delta,
                "cost": estimated_cost,
            })

        # Turnover budget
        if total_turnover_value > portfolio_value * cfg.turnover_limit:
            # Scale down trades proportionally
            scale = (portfolio_value * cfg.turnover_limit) / total_turnover_value
            for trade in trades:
                trade["value"] *= scale
                trade["shares"] = int(trade["shares"] * scale)
                trade["weight_delta"] *= scale

        return trades

    def execute_rebalance(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        portfolio_value: float,
        current_prices: dict[str, float] | None = None,
        current_date: datetime | None = None,
    ) -> RebalanceResult:
        """Full rebalance cycle: check, compute, execute.

        Returns:
            RebalanceResult with trades and metrics.
        """
        should, reason = self.should_rebalance(current_weights, target_weights, current_date)

        if not should:
            return RebalanceResult(rebalanced=False, reason=reason)

        trades = self.compute_trades(current_weights, target_weights, portfolio_value, current_prices)

        if not trades:
            return RebalanceResult(rebalanced=False, reason="no_trades_needed")

        turnover = sum(t["value"] for t in trades) / portfolio_value if portfolio_value > 0 else 0.0
        total_cost = sum(t.get("cost", 5.0) for t in trades)

        self._last_rebalance = current_date or datetime.utcnow()

        return RebalanceResult(
            rebalanced=True,
            trigger=reason,
            trades=trades,
            turnover=turnover,
            estimated_cost=total_cost,
        )
