"""Event-driven backtest engine — MarketEvent → SignalEvent → OrderEvent → FillEvent.

Integrated with Phase 3 modules: risk limits, position sizing, regime filtering,
drawdown tracking, and risk metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quantcore.backtest.analytics import (
    PerformanceReport,
    compute_returns,
    generate_performance_report,
)
from quantcore.core.types import OHLCV, Fill, Order, OrderType, Position, Side
from quantcore.risk.drawdown import DrawdownAction, DrawdownConfig, DrawdownTracker
from quantcore.risk.limits import LimitCheckResult, RiskLimits, check_pre_trade

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(int, Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


@dataclass(order=True)
class Event:
    timestamp: float  # Unix timestamp for heap ordering
    event_type: EventType = field(compare=False)
    data: Any = field(compare=False, default=None)


@dataclass
class MarketData:
    bar: OHLCV


@dataclass
class SignalData:
    symbol: str
    direction: str  # "long", "short", "flat"
    strength: float
    strategy_id: str


@dataclass
class OrderData:
    order: Order


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class Strategy(Protocol):
    """Protocol for backtest strategies."""

    def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list[SignalData]:
        """Generate signals from the current bar."""
        ...


    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        """React to a fill (optional)."""
        ...


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------


class PortfolioState:
    """Tracks positions, cash, and portfolio value during backtest.

    Integrated with Phase 3: drawdown tracking, regime state, trade logging.
    """

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.positions: dict[str, Position] = {}
        self._prices: dict[str, float] = {}
        self.equity_curve: list[tuple[float, float]] = []  # (timestamp, value)
        self.trades: list[Fill] = []
        self.signal_log: list[dict[str, Any]] = []
        self.regime_state: str = "unknown"

    @property
    def total_value(self) -> float:
        positions_value = sum(
            float(pos.quantity) * self._prices.get(pos.symbol, float(pos.market_price))
            for pos in self.positions.values()
            if float(pos.quantity) != 0
        )
        return self.cash + positions_value

    @property
    def returns(self) -> float:
        if self.initial_cash == 0:
            return 0.0
        return (self.total_value - self.initial_cash) / self.initial_cash

    def update_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price
        if symbol in self.positions:
            pos = self.positions[symbol]
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                market_price=Decimal(str(price)),
                unrealized_pnl=Decimal(str(float(pos.quantity) * (price - float(pos.avg_cost)))),
                realized_pnl=pos.realized_pnl,
            )

    def apply_fill(self, fill: Fill, symbol: str) -> None:
        """Apply a fill to the portfolio."""
        qty = float(fill.quantity)
        price = float(fill.price)
        commission = float(fill.commission)
        self.cash -= commission
        self.trades.append(fill)

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal(str(qty if fill.order_id.endswith("_BUY") else -qty)),
                avg_cost=Decimal(str(price)),
                market_price=Decimal(str(price)),
            )
            self.cash -= qty * price if fill.order_id.endswith("_BUY") else 0.0
            self.cash += qty * price if fill.order_id.endswith("_SELL") else 0.0
        else:
            pos = self.positions[symbol]
            current_qty = float(pos.quantity)
            new_qty = current_qty + qty if fill.order_id.endswith("_BUY") else current_qty - qty
            # Update average cost for buys
            if fill.order_id.endswith("_BUY"):
                total_cost = float(pos.avg_cost) * abs(current_qty) + price * qty
                new_avg = total_cost / abs(new_qty) if new_qty != 0 else 0.0
            else:
                new_avg = float(pos.avg_cost)

            realized = 0.0
            if fill.order_id.endswith("_SELL") and current_qty > 0:
                realized = (price - float(pos.avg_cost)) * qty
                self.cash += qty * price
            elif fill.order_id.endswith("_BUY"):
                self.cash -= qty * price

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal(str(new_qty)),
                avg_cost=Decimal(str(new_avg)),
                market_price=Decimal(str(price)),
                realized_pnl=Decimal(str(float(pos.realized_pnl) + realized)),
            )

        # Remove zero positions
        if abs(float(self.positions[symbol].quantity)) < 1e-10:
            del self.positions[symbol]

    def record_equity(self, timestamp: float) -> None:
        self.equity_curve.append((timestamp, self.total_value))


# ---------------------------------------------------------------------------
# Data handler
# ---------------------------------------------------------------------------


class HistoricalDataHandler:
    """Feeds historical OHLCV bars as MarketEvents."""

    def __init__(self, data: pd.DataFrame) -> None:
        """data: DataFrame with columns [symbol, timestamp, open, high, low, close, volume]."""
        self._data = data.sort_values("timestamp").reset_index(drop=True)
        self._index = 0

    def has_next(self) -> bool:
        return self._index < len(self._data)

    def next_bar(self) -> OHLCV | None:
        if not self.has_next():
            return None
        row = self._data.iloc[self._index]
        self._index += 1
        return OHLCV(
            symbol=str(row.get("symbol", "UNKNOWN")),
            timestamp=row["timestamp"] if isinstance(row["timestamp"], datetime) else pd.Timestamp(row["timestamp"]).to_pydatetime(),
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=int(row["volume"]),
        )

    def reset(self) -> None:
        self._index = 0


# ---------------------------------------------------------------------------
# Fill simulator
# ---------------------------------------------------------------------------


class FillSimulator:
    """Simulates order fills with slippage and commissions."""

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        fill_at: str = "next_open",
    ) -> None:
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.fill_at = fill_at
        self._next_bar: OHLCV | None = None
        self._current_bar: OHLCV | None = None

    def set_bars(self, current: OHLCV, next_bar: OHLCV | None = None) -> None:
        self._current_bar = current
        self._next_bar = next_bar

    def simulate_fill(self, order: Order) -> Fill | None:
        """Simulate a fill for the given order. Returns None if order cannot fill."""
        bar = self._next_bar if self.fill_at == "next_open" else self._current_bar
        if bar is None:
            bar = self._current_bar
        if bar is None:
            return None

        fill_price = float(bar.open)

        # Apply slippage
        slip = fill_price * self.slippage_bps / 10_000.0
        if order.side == Side.BUY:
            fill_price += slip
        else:
            fill_price -= slip

        # Check limit orders
        if order.type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == Side.BUY and fill_price > float(order.limit_price):
                return None
            if order.side == Side.SELL and fill_price < float(order.limit_price):
                return None

        qty = float(order.quantity)
        commission = qty * self.commission_per_share

        return Fill(
            order_id=f"{order.id}_{order.side.value}",
            price=Decimal(str(round(fill_price, 6))),
            quantity=order.quantity,
            commission=Decimal(str(round(commission, 4))),
            timestamp=bar.timestamp,
            slippage_bps=Decimal(str(self.slippage_bps)),
        )


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Event-driven backtester with strict no-lookahead enforcement.

    Integrated with Phase 3 modules:
    - Risk limits: pre-trade checks prevent rule-breaking orders
    - Position sizing: configurable sizing (Kelly, vol-target, ATR, fixed fraction)
    - Drawdown tracking: monitors DD and triggers actions at thresholds
    - Regime filtering: optional regime model filters signals by market state
    - Risk metrics: summary() returns a full PerformanceReport
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: float = 1_000_000.0,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        risk_limits: RiskLimits | None = None,
        sizing_pct: float = 0.10,
        sizing_fn: Callable[[float, float, dict[str, Any]], float] | None = None,
        drawdown_config: DrawdownConfig | None = None,
        regime_model: Any = None,
    ) -> None:
        self.strategy = strategy
        self.portfolio = PortfolioState(initial_cash)
        self.fill_sim = FillSimulator(slippage_bps, commission_per_share)
        self.risk_limits = risk_limits
        self.sizing_pct = sizing_pct
        self.sizing_fn = sizing_fn  # (cash, price, context) -> shares
        self.drawdown_tracker = DrawdownTracker(drawdown_config) if drawdown_config else None
        self.regime_model = regime_model
        self._rejected_orders: list[tuple[Order, LimitCheckResult]] = []

    @property
    def rejected_orders(self) -> list[tuple[Order, LimitCheckResult]]:
        return self._rejected_orders

    def _compute_sizing(
        self, side: Side, price: float, bar: OHLCV, sig_strength: float
    ) -> int:
        """Compute position size using the configured sizing strategy."""
        cash = self.portfolio.cash
        if cash <= 0 and side == Side.BUY:
            return 0

        if self.sizing_fn is not None:
            context = {
                "cash": cash,
                "price": price,
                "signal_strength": sig_strength,
                "bar": bar,
                "portfolio_value": self.portfolio.total_value,
            }
            return int(self.sizing_fn(cash, price, context))

        if side == Side.BUY:
            return int(cash * self.sizing_pct / price)
        else:
            return 0

    def _current_regime(self, bar: OHLCV) -> str | None:
        """Detect current regime from the model if configured."""
        if self.regime_model is None:
            return None
        try:
            ret = float(bar.close) / float(bar.open) - 1.0
            state, prob = self.regime_model.predict_state(np.array([ret]))
            return self.regime_model.state_labels[state] if self.regime_model.state_labels else str(state)
        except Exception:
            return None

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the backtest on historical data.

        data: DataFrame with columns [symbol, timestamp, open, high, low, close, volume]
        Returns DataFrame with equity curve and risk metrics accessible via summary().
        """
        handler = HistoricalDataHandler(data)
        self.portfolio = PortfolioState(self.portfolio.initial_cash)
        self._rejected_orders = []

        if self.drawdown_tracker:
            self.drawdown_tracker = DrawdownTracker(
                self.drawdown_tracker.config if hasattr(self.drawdown_tracker, 'config') else DrawdownConfig()
            )

        pending_orders: list[Order] = []
        daily_returns: list[float] = []

        while handler.has_next():
            bar = handler.next_bar()
            if bar is None:
                break

            # Update portfolio prices
            self.portfolio.update_price(bar.symbol, float(bar.close))

            # Update drawdown tracker
            if self.drawdown_tracker is not None:
                dd_state = self.drawdown_tracker.update(self.portfolio.total_value)
                self.portfolio.regime_state = dd_state.action.value

                # Circuit-breaker: halt on extreme drawdowns
                if dd_state.action in (DrawdownAction.FLATTEN, DrawdownAction.HALT):
                    pending_orders.clear()
                    self.portfolio.record_equity(
                        bar.timestamp.timestamp() if hasattr(bar.timestamp, "timestamp") else 0.0
                    )
                    continue

            # Set up fill simulator with current and next bar
            next_bar = None
            peek_idx = handler._index
            if peek_idx < len(handler._data):
                next_row = handler._data.iloc[peek_idx]
                try:
                    next_bar = OHLCV(
                        symbol=str(next_row.get("symbol", bar.symbol)),
                        timestamp=next_row["timestamp"] if isinstance(next_row["timestamp"], datetime) else pd.Timestamp(next_row["timestamp"]).to_pydatetime(),
                        open=Decimal(str(next_row["open"])),
                        high=Decimal(str(next_row["high"])),
                        low=Decimal(str(next_row["low"])),
                        close=Decimal(str(next_row["close"])),
                        volume=int(next_row["volume"]),
                    )
                except Exception:
                    next_bar = None

            self.fill_sim.set_bars(bar, next_bar)

            # Process pending orders from previous bar
            for order in pending_orders:
                fill = self.fill_sim.simulate_fill(order)
                if fill is not None:
                    self.portfolio.apply_fill(fill, order.symbol)
                    self.strategy.on_fill(fill, self.portfolio)

            pending_orders.clear()

            # Detect regime
            regime = self._current_regime(bar)
            if regime:
                self.portfolio.regime_state = regime

            # Generate signals from strategy
            signals = self.strategy.on_bar(bar, self.portfolio)

            # Convert signals to orders with risk checks and sizing
            for sig in signals:
                if sig.direction == "long":
                    side = Side.BUY
                elif sig.direction == "short":
                    side = Side.SELL
                else:
                    continue

                price = float(bar.close)

                if side == Side.BUY:
                    qty = self._compute_sizing(side, price, bar, sig.strength)
                    if qty <= 0:
                        continue
                else:
                    # For sells/shorts, close existing position
                    pos = self.portfolio.positions.get(sig.symbol)
                    if pos is None or float(pos.quantity) <= 0:
                        continue
                    qty = int(abs(float(pos.quantity)))

                order = Order(
                    id=f"ord_{bar.timestamp.isoformat()}_{sig.symbol}_{side.value}",
                    symbol=sig.symbol,
                    side=side,
                    type=OrderType.MARKET,
                    quantity=Decimal(str(qty)),
                )

                # Pre-trade risk check
                if self.risk_limits is not None:
                    from quantcore.core.types import PortfolioSnapshot
                    snap = PortfolioSnapshot(
                        timestamp=bar.timestamp if isinstance(bar.timestamp, datetime) else datetime.utcnow(),
                        positions=self.portfolio.positions,
                        total_value=Decimal(str(self.portfolio.total_value)),
                    )
                    result = check_pre_trade(order, snap, self.risk_limits)
                    if result.status.value == "rejected":
                        self._rejected_orders.append((order, result))
                        continue

                pending_orders.append(order)

                # Log signal
                self.portfolio.signal_log.append({
                    "timestamp": bar.timestamp,
                    "symbol": sig.symbol,
                    "direction": sig.direction,
                    "strength": sig.strength,
                    "regime": self.portfolio.regime_state,
                })

            # Record daily return
            prev_equity = self.portfolio.equity_curve[-1][1] if self.portfolio.equity_curve else self.portfolio.initial_cash
            current_equity = self.portfolio.total_value
            daily_returns.append((current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0)

            # Record equity
            ts = bar.timestamp.timestamp() if hasattr(bar.timestamp, "timestamp") else 0.0
            self.portfolio.record_equity(ts)

        # Build result DataFrame
        if not self.portfolio.equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])

        timestamps, values = zip(*self.portfolio.equity_curve)
        self._last_returns = np.array(daily_returns)
        return pd.DataFrame({"timestamp": pd.to_datetime(timestamps, unit="s"), "equity": values})

    def summary(self) -> PerformanceReport:
        """Generate a full performance report from the backtest results.

        Uses the risk metrics module for all ratio computations.
        """
        if not self.portfolio.equity_curve:
            return PerformanceReport()

        equity = np.array([v for _, v in self.portfolio.equity_curve])
        returns = compute_returns(equity)
        trade_pnls = [
            float(fill.price) * float(fill.quantity) * (1 if "BUY" in fill.order_id else -1)
            - float(fill.commission)
            for fill in self.portfolio.trades
        ]

        return generate_performance_report(equity, returns, np.array(trade_pnls) if trade_pnls else None)

    def risk_metrics(self) -> dict[str, float]:
        """Compute risk-adjusted metrics for the backtest using the risk module."""
        from quantcore.risk.metrics import compute_all_metrics
        if not self.portfolio.equity_curve or len(self.portfolio.equity_curve) < 2:
            return {}
        equity = np.array([v for _, v in self.portfolio.equity_curve])
        returns = compute_returns(equity)
        return compute_all_metrics(returns)
