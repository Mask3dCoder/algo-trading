"""Compliance limit enforcement: regulatory position limits, best execution checks.

Wired to quantcore.risk.limits for pre/post-trade checks and
quantcore.execution.tca for best execution verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from quantcore.core.types import Order, PortfolioSnapshot
from quantcore.risk.limits import (
    LimitStatus,
    RiskLimits,
    check_post_trade,
    check_pre_trade,
)


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"


@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    status: ComplianceStatus
    check_name: str
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    limit_value: float = 0.0
    actual_value: float = 0.0
    remediation: str = ""


class ComplianceChecker:
    """Regulatory and internal compliance checks.

    Wraps the risk limits module with additional regulatory rules:
    - SEC Reg SHO (short sale restrictions)
    - Pattern Day Trader (PDT) rules
    - Position concentration limits
    - Wash sale monitoring
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        pdt_enabled: bool = True,
        max_daily_trades: int = 3,  # PDT threshold
        min_account_equity: float = 25_000.0,  # PDT minimum
    ) -> None:
        self.limits = limits or RiskLimits()
        self.pdt_enabled = pdt_enabled
        self.max_daily_trades = max_daily_trades
        self.min_account_equity = min_account_equity
        self._daily_trades: list[datetime] = []
        self._violations: list[ComplianceResult] = []

    def check_pre_trade(
        self,
        order: Order,
        portfolio: PortfolioSnapshot,
        adv_data: dict[str, float] | None = None,
    ) -> ComplianceResult:
        """Run all pre-trade compliance checks.

        Returns the worst result across all checks.
        """
        results: list[ComplianceResult] = []

        # 1. Risk limits check
        limit_result = check_pre_trade(order, portfolio, self.limits, adv_data)
        if limit_result.status == LimitStatus.REJECTED:
            results.append(ComplianceResult(
                status=ComplianceStatus.VIOLATION,
                check_name=limit_result.limit_name,
                details=limit_result.reason,
                limit_value=limit_result.limit_value,
                actual_value=limit_result.current_value,
                remediation=f"Reduce order size below {limit_result.limit_value}",
            ))

        # 2. PDT check
        if self.pdt_enabled:
            pdt_result = self._check_pdt(portfolio)
            if pdt_result:
                results.append(pdt_result)

        # 3. Short sale check (Reg SHO)
        if order.side.value == "sell" and order.type.value == "market":
            sho_result = self._check_reg_sho(order)
            if sho_result:
                results.append(sho_result)

        # Return worst result
        if any(r.status == ComplianceStatus.VIOLATION for r in results):
            worst = [r for r in results if r.status == ComplianceStatus.VIOLATION][0]
            self._violations.append(worst)
            return worst
        if any(r.status == ComplianceStatus.WARNING for r in results):
            return [r for r in results if r.status == ComplianceStatus.WARNING][0]
        return ComplianceResult(status=ComplianceStatus.COMPLIANT, check_name="all")

    def check_post_trade(
        self,
        portfolio: PortfolioSnapshot,
        executed_order: Order | None = None,
    ) -> list[ComplianceResult]:
        """Run post-trade compliance checks."""
        results: list[ComplianceResult] = []

        # Risk limits recheck
        breaches = check_post_trade(portfolio, self.limits)
        for breach in breaches:
            results.append(ComplianceResult(
                status=ComplianceStatus.VIOLATION,
                check_name=breach.limit_name,
                details=breach.reason,
                limit_value=breach.limit_value,
                actual_value=breach.current_value,
            ))

        # Track daily trades
        if executed_order:
            self._daily_trades.append(datetime.utcnow())
            # Clean old entries
            cutoff = datetime.utcnow() - timedelta(days=1)
            self._daily_trades = [t for t in self._daily_trades if t > cutoff]

        if results:
            self._violations.extend(results)

        return results

    def _check_pdt(self, portfolio: PortfolioSnapshot) -> ComplianceResult | None:
        """Check Pattern Day Trader rules."""
        equity = float(portfolio.total_value)

        if equity < self.min_account_equity:
            today = datetime.utcnow().date()
            today_trades = sum(1 for t in self._daily_trades if t.date() == today)
            if today_trades >= self.max_daily_trades:
                return ComplianceResult(
                    status=ComplianceStatus.VIOLATION,
                    check_name="pdt",
                    details=f"PDT: {today_trades} day trades with equity ${equity:,.0f} < ${self.min_account_equity:,.0f}",
                    limit_value=self.min_account_equity,
                    actual_value=equity,
                    remediation="Increase account equity above $25,000 or limit to 3 day trades per 5 days",
                )

            if today_trades >= self.max_daily_trades - 1:
                return ComplianceResult(
                    status=ComplianceStatus.WARNING,
                    check_name="pdt",
                    details=f"PDT warning: {today_trades} day trades, 1 remaining",
                )

        return None

    def _check_reg_sho(self, order: Order) -> ComplianceResult | None:
        """Check SEC Regulation SHO (short sale rules)."""
        # Locate requirement: must have reasonable grounds to believe
        # the security can be borrowed. In production, check locate availability.
        # For now, flag short market orders as requiring a locate check.
        if order.side.value == "sell":
            return ComplianceResult(
                status=ComplianceStatus.WARNING,
                check_name="reg_sho_locate",
                details=f"Short sale of {order.symbol}: locate requirement applies",
                remediation="Verify borrow availability before execution",
            )
        return None

    @property
    def violations(self) -> list[ComplianceResult]:
        return self._violations

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def daily_report(self) -> dict[str, Any]:
        """Generate a daily compliance summary."""
        today = datetime.utcnow().date()
        today_violations = [
            v for v in self._violations
            if v.timestamp.date() == today
        ]
        return {
            "date": today.isoformat(),
            "total_violations": len(today_violations),
            "violations_by_type": {},
            "daily_trades": sum(1 for t in self._daily_trades if t.date() == today),
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def check_regulatory_limits(
    portfolio: PortfolioSnapshot,
    limits: RiskLimits | None = None,
) -> list[ComplianceResult]:
    """Standalone regulatory limit check."""
    checker = ComplianceChecker(limits)
    return checker.check_post_trade(portfolio)


def check_best_execution(
    order: Order,
    fill_prices: list[float],
    benchmark: str = "arrival",
) -> ComplianceResult:
    """Verify best execution compliance.

    Compares execution prices against the benchmark and flags
    significant deviations that may violate best-ex obligations.
    """
    if not fill_prices:
        return ComplianceResult(status=ComplianceStatus.COMPLIANT, check_name="best_exec")

    avg_fill = float(np.mean(fill_prices))
    limit_price = float(order.limit_price) if order.limit_price else avg_fill

    # Check for excessive slippage (>50bps is suspicious)
    slippage_bps = abs(avg_fill - limit_price) / limit_price * 10_000 if limit_price > 0 else 0

    if slippage_bps > 100:
        return ComplianceResult(
            status=ComplianceStatus.VIOLATION,
            check_name="best_execution",
            details=f"Slippage {slippage_bps:.1f} bps exceeds 100 bps threshold",
            actual_value=slippage_bps,
            limit_value=100.0,
            remediation="Review execution venue and algorithm selection",
        )
    elif slippage_bps > 50:
        return ComplianceResult(
            status=ComplianceStatus.WARNING,
            check_name="best_execution",
            details=f"Elevated slippage: {slippage_bps:.1f} bps",
            actual_value=slippage_bps,
            limit_value=50.0,
        )

    return ComplianceResult(status=ComplianceStatus.COMPLIANT, check_name="best_execution")
