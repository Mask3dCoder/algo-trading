"""Intraday risk loop: periodic repricing, exposure checks, auto-flatten.

Runs every 1-5 minutes in the live trading loop to:
  1. Reprice all positions using latest market data.
  2. Recompute gross/net exposure, leverage, margin usage.
  3. Enforce hard limits (concentration, leverage, margin, VaR).
  4. Auto-reduce or flatten on breach.

This is SEPARATE from the per-tick risk checks in LiveRunner.
It runs on a timer to catch silent risk accumulation between ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Protocol

from quantcore.core.types import Position
from quantcore.risk.limits import (
    LimitCheckResult,
    RiskLimits,
    check_post_trade,
)


class RiskAction(str, Enum):
    NONE = "none"
    WARN = "warn"
    REDUCE = "reduce"       # Reduce position sizes
    FLATTEN = "flatten"     # Close all positions
    HALT = "halt"           # Halt all trading


@dataclass
class IntradayRiskSnapshot:
    """Result of a single intraday risk check."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    action: RiskAction = RiskAction.NONE
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    margin_used: float = 0.0
    margin_pct: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    drawdown_pct: float = 0.0
    breaches: list[LimitCheckResult] = field(default_factory=list)
    details: str = ""


@dataclass
class IntradayRiskConfig:
    """Configuration for the intraday risk loop."""
    check_interval_sec: float = 60.0          # Run every 60 seconds
    max_leverage: float = 2.0                  # Reg T max
    max_margin_pct: float = 0.85               # 85% margin utilization
    max_drawdown_pct: float = 0.10             # 10% DD → reduce
    max_drawdown_flatten_pct: float = 0.20     # 20% DD → flatten
    max_concentration_pct: float = 0.25        # 25% single name
    min_equity: float = 100_000.0              # Min equity before halt
    auto_reduce_factor: float = 0.50           # Reduce by 50% on breach
    reduce_on_warn: bool = True                # Auto-reduce on warning
    flatten_on_critical: bool = True           # Auto-flatten on critical


class MarketDataProvider(Protocol):
    """Protocol for getting current market prices."""

    def get_current_price(self, symbol: str) -> float | None: ...


class IntradayRiskLoop:
    """Periodic intraday risk monitor with auto-escalation.

    Usage:
        risk_loop = IntradayRiskLoop(config)
        snapshot = risk_loop.check(
            positions=current_positions,
            cash=ledger.settled_cash,
            equity=total_equity,
            drawdown_pct=0.05,
            prices={"AAPL": 150.0},
        )
        if snapshot.action in (RiskAction.FLATTEN, RiskAction.HALT):
            trader.emergency_flatten()
    """

    def __init__(self, config: IntradayRiskConfig | None = None) -> None:
        self.config = config or IntradayRiskConfig()
        self._last_check: datetime | None = None
        self._breach_history: list[IntradayRiskSnapshot] = []
        self._consecutive_breaches: int = 0
        self._max_consecutive_breaches: int = 3

    def check(
        self,
        positions: dict[str, Position],
        cash: float,
        equity: float,
        drawdown_pct: float,
        prices: dict[str, float] | None = None,
        limits: RiskLimits | None = None,
    ) -> IntradayRiskSnapshot:
        """Run a single intraday risk check.

        Args:
            positions: Current {symbol: Position} dict.
            cash: Current settled cash.
            equity: Current total equity (cash + position market value).
            drawdown_pct: Current drawdown as decimal (0.05 = 5%).
            prices: Current market prices {symbol: price}.
            limits: Risk limits to enforce. Uses defaults if None.

        Returns:
            IntradayRiskSnapshot with action recommendation.
        """
        now = datetime.now(UTC)
        prices = prices or {}

        # 1. Reprice positions using latest market data
        gross_exposure = 0.0
        net_exposure = 0.0
        for sym, pos in positions.items():
            qty = float(pos.quantity)
            price = prices.get(sym, float(pos.market_price))
            mv = qty * price
            gross_exposure += abs(mv)
            net_exposure += mv

        # 2. Compute metrics
        leverage = gross_exposure / equity if equity > 0 else 999.0
        margin_used = max(0.0, gross_exposure - cash)
        margin_pct = margin_used / gross_exposure if gross_exposure > 0 else 0.0

        # 3. Build snapshot
        snapshot = IntradayRiskSnapshot(
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            leverage=leverage,
            margin_used=margin_used,
            margin_pct=margin_pct,
            cash=cash,
            equity=equity,
            drawdown_pct=drawdown_pct,
        )

        # 4. Enforce hard limits
        breaches: list[str] = []

        # Leverage
        if leverage > self.config.max_leverage:
            breaches.append(
                f"Leverage {leverage:.2f}x > {self.config.max_leverage:.2f}x"
            )

        # Margin
        if margin_pct > self.config.max_margin_pct:
            breaches.append(
                f"Margin {margin_pct:.1%} > {self.config.max_margin_pct:.1%}"
            )

        # Drawdown
        dd = abs(drawdown_pct)
        if dd >= self.config.max_drawdown_flatten_pct:
            breaches.append(
                f"Drawdown {dd:.1%} >= flatten threshold "
                f"{self.config.max_drawdown_flatten_pct:.1%}"
            )
            snapshot.action = RiskAction.FLATTEN
        elif dd >= self.config.max_drawdown_pct:
            breaches.append(
                f"Drawdown {dd:.1%} >= reduce threshold "
                f"{self.config.max_drawdown_pct:.1%}"
            )
            snapshot.action = RiskAction.REDUCE
            snapshot.details = "; ".join(breaches)

        # Equity floor
        if equity < self.config.min_equity:
            breaches.append(
                f"Equity ${equity:.2f} < min ${self.config.min_equity:.2f}"
            )
            snapshot.action = RiskAction.HALT

        # Post-trade limit checks (concentration, position limits)
        if limits is not None:
            from datetime import datetime as dt

            from quantcore.core.types import Decimal, PortfolioSnapshot
            snap = PortfolioSnapshot(
                timestamp=dt.utcnow(),
                positions=positions,
                total_value=Decimal(str(equity)),
                cash=Decimal(str(cash)),
                returns=0.0,
            )
            limit_breaches = check_post_trade(snap, limits)
            snapshot.breaches = limit_breaches
            for b in limit_breaches:
                breaches.append(f"{b.limit_name}: {b.reason}")

        # Determine action from breaches
        if snapshot.action == RiskAction.NONE and breaches:
            if self.config.reduce_on_warn:
                snapshot.action = RiskAction.REDUCE
            else:
                snapshot.action = RiskAction.WARN
            snapshot.details = "; ".join(breaches)

        # Track consecutive breaches
        if breaches:
            self._consecutive_breaches += 1
            if self._consecutive_breaches >= self._max_consecutive_breaches:
                if snapshot.action in (RiskAction.NONE, RiskAction.WARN, RiskAction.REDUCE):
                    snapshot.action = RiskAction.FLATTEN
                    snapshot.details = (
                        f"{self._consecutive_breaches} consecutive breaches: "
                        f"{'; '.join(breaches)}"
                    )
        else:
            self._consecutive_breaches = 0

        self._last_check = now
        self._breach_history.append(snapshot)

        return snapshot

    @property
    def last_check(self) -> datetime | None:
        return self._last_check

    @property
    def breach_history(self) -> list[IntradayRiskSnapshot]:
        return list(self._breach_history)

    def reset(self) -> None:
        """Reset breach tracking."""
        self._consecutive_breaches = 0
        self._breach_history.clear()
