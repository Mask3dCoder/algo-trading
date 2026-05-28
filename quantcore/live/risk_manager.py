"""Live risk manager with concrete pre-trade checks and daily P&L tracking.

Wired into the live orchestrator so every order is checked before it
reaches the broker.  Limits are read from environment variables (prefix
``QUANTCORE_RISK_``) or defaulted.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from quantcore.core.types import Order, Side

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Approval:
    """Result of a pre-trade risk check."""

    approved: bool
    reason: str = ""


@dataclass
class RiskLimitsConfig:
    """All configurable risk limits for live trading.

    Values can be overridden via environment variables::

        QUANTCORE_RISK_MAX_POSITION_PCT=0.05
        QUANTCORE_RISK_MAX_GROSS_EXPOSURE=2.0
        QUANTCORE_RISK_MAX_DAILY_LOSS_PCT=0.02
        QUANTCORE_RISK_MAX_DRAWDOWN_PCT=0.10
        QUANTCORE_RISK_MAX_SINGLE_ORDER_PCT=0.05
        QUANTCORE_RISK_MAX_LEVERAGE=2.0
    """

    max_position_per_ticker_pct: float = 0.05   # 5% of equity per ticker
    max_gross_exposure_pct: float = 2.0           # 200% (long + short)
    max_daily_loss_pct: float = 0.02              # 2% max loss per day
    max_drawdown_pct: float = 0.10                # 10% peak-to-trough
    max_single_order_pct: float = 0.05            # 5% of equity per order
    max_leverage: float = 2.0                     # gross / equity

    @classmethod
    def from_env(cls) -> "RiskLimitsConfig":
        """Load limits from environment variables."""
        return cls(
            max_position_per_ticker_pct=float(
                os.environ.get("QUANTCORE_RISK_MAX_POSITION_PCT", "0.05")
            ),
            max_gross_exposure_pct=float(
                os.environ.get("QUANTCORE_RISK_MAX_GROSS_EXPOSURE", "2.0")
            ),
            max_daily_loss_pct=float(
                os.environ.get("QUANTCORE_RISK_MAX_DAILY_LOSS_PCT", "0.02")
            ),
            max_drawdown_pct=float(
                os.environ.get("QUANTCORE_RISK_MAX_DRAWDOWN_PCT", "0.10")
            ),
            max_single_order_pct=float(
                os.environ.get("QUANTCORE_RISK_MAX_SINGLE_ORDER_PCT", "0.05")
            ),
            max_leverage=float(
                os.environ.get("QUANTCORE_RISK_MAX_LEVERAGE", "2.0")
            ),
        )


# ---------------------------------------------------------------------------
# RiskManagerLive
# ---------------------------------------------------------------------------

class RiskManagerLive:
    """Concrete risk manager with pre-trade checks and daily/equity tracking.

    All methods are stateful (track peak equity and daily P&L).
    Thread-safe for single-threaded orchestrator use only.
    """

    def __init__(self, config: RiskLimitsConfig | None = None) -> None:
        self.config = config or RiskLimitsConfig.from_env()
        self._peak_equity: float = 0.0
        self._starting_equity_for_day: float = 0.0
        self._current_daily_pnl: float = 0.0
        self._total_equity: float = 0.0
        self._current_date: date | None = None

    # ------------------------------------------------------------------
    # Pre-trade check
    # ------------------------------------------------------------------

    def pre_trade_check(
        self,
        order: Order,
        equity: float,
        positions: dict[str, object] | None = None,
    ) -> Approval:
        """Run all pre-trade risk checks.  Returns Approval with reason if rejected."""
        if equity <= 0:
            return Approval(approved=False, reason="Equity <= 0 — cannot trade")

        qty = float(order.quantity)
        # Use a crude price estimate for notional sizing
        price = self._order_price(order)
        notional = qty * price

        # 1. Single order size limit
        if notional / equity > self.config.max_single_order_pct:
            return Approval(
                approved=False,
                reason=f"Single order size exceeds {self.config.max_single_order_pct:.1%} of equity "
                       f"(${notional:,.2f} / ${equity:,.2f})",
            )

        # 2. Position per-ticker limit
        current_notional = 0.0
        if positions and order.symbol in positions:
            pos = positions[order.symbol]
            current_notional = abs(float(pos.quantity)) * price
        long_short = 1 if order.side == Side.BUY else -1
        total_notional = current_notional + notional
        if total_notional / equity > self.config.max_position_per_ticker_pct:
            return Approval(
                approved=False,
                reason=f"Position per-ticker exceeds {self.config.max_position_per_ticker_pct:.1%} of equity "
                       f"(${total_notional:,.2f} / ${equity:,.2f})",
            )

        # 3. Leverage / gross exposure limit (post this order + existing)
        gross = self._gross_exposure(positions or {}, price)
        if order.side == Side.BUY:
            gross += notional
        else:
            gross += notional  # SELL also adds to gross (existing reduced)
        leverage = gross / equity
        if leverage > self.config.max_leverage:
            return Approval(
                approved=False,
                reason=f"Leverage {leverage:.2f}x exceeds limit {self.config.max_leverage:.2f}x",
            )

        # 4. Daily loss limit
        if self._current_daily_pnl < -equity * self.config.max_daily_loss_pct:
            return Approval(
                approved=False,
                reason=f"Daily loss limit hit: PnL {self._current_daily_pnl:,.2f} < -{self.config.max_daily_loss_pct:.1%} of equity",
            )

        # 5. Drawdown limit
        if self._peak_equity > 0:
            dd = (equity - self._peak_equity) / self._peak_equity
            if dd < -self.config.max_drawdown_pct:
                return Approval(
                    approved=False,
                    reason=f"Drawdown {dd:.2%} exceeds max {self.config.max_drawdown_pct:.1%}",
                )

        return Approval(approved=True)

    # ------------------------------------------------------------------
    # Portfolio / equity tracking
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Call after every trade or reconciliation to track peaks and P&L."""
        self._total_equity = equity
        if self._peak_equity == 0.0:
            self._peak_equity = equity
        else:
            self._peak_equity = max(self._peak_equity, equity)
        # Update daily P&L based on starting equity for the day
        if self._starting_equity_for_day > 0:
            self._current_daily_pnl = equity - self._starting_equity_for_day

    def start_day(self, equity: float) -> None:
        """Call at market open to reset daily P&L tracking."""
        self._starting_equity_for_day = equity
        self._current_daily_pnl = 0.0
        if self._peak_equity == 0.0:
            self._peak_equity = equity

    def eod_check(self, current_date: date, equity_curve: list[float]) -> dict:
        """End-of-day check: update peak, daily P&L, and return a status dict.

        Parameters
        ----------
        current_date : date
            The trading day that just finished.
        equity_curve : list[float]
            Closing equity values for the day (usually one point).

        Returns
        -------
        dict
            Status with keys: date, equity, peak, drawdown, daily_pnl.
        """
        if not equity_curve:
            return {}
        equity = equity_curve[-1]
        self._total_equity = equity
        if self._peak_equity == 0.0:
            self._peak_equity = equity
        else:
            self._peak_equity = max(self._peak_equity, equity)

        # Reset daily P&L for next day
        self._current_daily_pnl = 0.0

        dd = (equity - self._peak_equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        status = {
            "date": str(current_date),
            "equity": equity,
            "peak_equity": self._peak_equity,
            "drawdown": dd,
            "daily_pnl": self._current_daily_pnl,
        }
        log.info("EOD risk check: %s", status)
        return status

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _order_price(self, order: Order) -> float:
        """Best-effort price for notional calculation."""
        if order.limit_price is not None:
            return float(order.limit_price)
        # Try to get latest known price from any external source if available;
        # for MVP we use a simple fallback.
        return 100.0

    def _gross_exposure(
        self, positions: dict[str, object], market_price: float
    ) -> float:
        """Sum absolute notional of all positions."""
        total = 0.0
        for pos in positions.values():
            try:
                total += abs(float(pos.quantity)) * float(getattr(pos, "market_price", market_price))
            except Exception:
                total += abs(float(pos.quantity)) * market_price
        return total

    @property
    def total_equity(self) -> float:
        return self._total_equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def current_drawdown(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._total_equity - self._peak_equity) / self._peak_equity

    @property
    def current_daily_pnl(self) -> float:
        return self._current_daily_pnl
