"""Paper trading engine: simulated order fills with realistic market mechanics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np

from decimal import Decimal

from quantcore.core.types import Order, Position, Side, OrderStatus, OrderType
from quantcore.execution.oms import OrderManager, OrderState
from quantcore.risk.limits import RiskLimits, check_pre_trade, LimitStatus
from quantcore.risk.drawdown import DrawdownTracker, DrawdownConfig, DrawdownAction


@dataclass
class PaperConfig:
    """Configuration for paper trading simulation."""
    initial_capital: float = 100_000.0
    slippage_bps: float = 2.0  # base slippage in basis points
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    max_commission_pct: float = 0.01
    latency_ms: float = 5.0
    fill_probability: float = 0.98
    partial_fill_probability: float = 0.05
    reject_bad_prices: bool = True
    max_spread_pct: float = 0.05
    position_sizing_enabled: bool = True
    max_position_pct: float = 0.20
    daily_vol_scale: float = 0.02  # assumption for vol-based slippage


@dataclass
class PaperFill:
    """Paper trading fill record."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: int = 0
    fill_price: float = 0.0
    slippage_bps: float = 0.0
    commission: float = 0.0
    is_partial: bool = False


@dataclass
class PaperAccount:
    """Paper trading account state."""
    cash: float = 100_000.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[PaperFill] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    daily_pnl: float = 0.0
    total_commission: float = 0.0


class PaperBroker:
    """Simulated broker with realistic fills, commissions, and market mechanics.

    Used for paper trading validation before live deployment.
    """

    def __init__(self, config: PaperConfig | None = None) -> None:
        self.config = config or PaperConfig()
        self.account = PaperAccount(cash=self.config.initial_capital)
        self.oms = OrderManager()
        self.drawdown_tracker = DrawdownTracker()
        self._current_prices: dict[str, float] = {}
        self._price_history: dict[str, list[float]] = {}

    def update_market_data(self, prices: dict[str, float], timestamp: datetime | None = None) -> None:
        """Update current prices used for fills."""
        self._current_prices.update(prices)
        for sym, price in prices.items():
            self._price_history.setdefault(sym, []).append(price)
            if len(self._price_history[sym]) > 252 * 6.5 * 60:
                self._price_history[sym] = self._price_history[sym][-252 * 6.5 * 60:]

    def submit_order(self, order: Order) -> str:
        """Submit an order to the paper broker. Returns order ID."""
        self.oms.submit(order)
        self.account.orders.append(order)
        return str(order.id)

    def cancel_order(self, order_id: str) -> bool:
        return self.oms.cancel(order_id)

    def process_fills(self) -> list[PaperFill]:
        """Process pending orders and generate fills at current market prices."""
        fills: list[PaperFill] = []
        pending = [
            o for o in self.oms.orders.values()
            if o.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL, OrderStatus.PENDING)
        ]
        for order in pending:
            price = self._current_prices.get(str(order.symbol))
            if price is None:
                continue
            fill = self._simulate_fill(order, price)
            if fill:
                fills.append(fill)
                self.account.fills.append(fill)
                self._apply_fill(fill)
                self.oms.fill(order.id, Decimal(str(fill.quantity)), Decimal(str(fill.fill_price)))
        return fills

    def _simulate_fill(self, order: Order, market_price: float) -> PaperFill | None:
        symbol = str(order.symbol)
        qty = int(float(order.quantity))

        if self.config.reject_bad_prices:
            limit = float(order.limit_price) if order.limit_price else None
            if limit is not None and order.type == OrderType.LIMIT:
                if order.side == Side.BUY and limit < market_price:
                    return None
                if order.side == Side.SELL and limit > market_price:
                    return None

        if np.random.random() > self.config.fill_probability:
            return None

        fill_qty = qty
        if np.random.random() < self.config.partial_fill_probability:
            fill_qty = max(1, int(qty * np.random.uniform(0.3, 0.9)))

        spread = market_price * np.random.uniform(0.0001, 0.001)
        slippage_bps = self.config.slippage_bps + np.random.exponential(self.config.slippage_bps)
        slippage = market_price * slippage_bps / 10000.0

        if order.side == Side.BUY:
            fill_price = market_price + spread + slippage
        else:
            fill_price = market_price - spread - slippage

        gross = fill_qty * fill_price
        commission = max(self.config.min_commission, fill_qty * self.config.commission_per_share)
        commission = min(commission, gross * self.config.max_commission_pct)

        fill = PaperFill(
            timestamp=datetime.now(timezone.utc),
            order_id=str(order.id), symbol=symbol,
            side="buy" if order.side == Side.BUY else "sell",
            quantity=fill_qty, fill_price=float(fill_price),
            slippage_bps=float(slippage_bps), commission=float(commission),
            is_partial=(fill_qty < qty),
        )
        return fill

    def _apply_fill(self, fill: PaperFill) -> None:
        from decimal import Decimal as D
        symbol = fill.symbol
        qty = fill.quantity
        cost = qty * fill.fill_price + fill.commission

        if fill.side == "sell":
            qty = -qty
            cost = -(abs(qty) * fill.fill_price) + fill.commission

        if symbol not in self.account.positions:
            self.account.positions[symbol] = Position(
                symbol=symbol, quantity=D("0"), avg_cost=D(str(fill.fill_price)),
                market_price=D(str(fill.fill_price)),
            )

        pos = self.account.positions[symbol]
        old_qty = float(pos.quantity)
        old_cost = float(pos.avg_cost) * old_qty
        new_qty = old_qty + qty
        new_cost = old_cost + abs(cost)
        pos.quantity = D(str(new_qty))
        pos.avg_cost = D(str(new_cost / new_qty)) if new_qty != 0 else D(str(fill.fill_price))
        pos.market_price = D(str(fill.fill_price))

        self.account.cash -= cost
        self.account.total_commission += fill.commission

    def get_positions(self) -> dict[str, Position]:
        return dict(self.account.positions)

    def get_account(self) -> dict[str, float]:
        total_mv = sum(
            float(abs(p.quantity)) * float(p.market_price)
            for p in self.account.positions.values()
        )
        return {
            "cash": self.account.cash,
            "equity": self.account.cash + total_mv,
            "portfolio_value": self.account.cash + total_mv,
            "total_commission": self.account.total_commission,
        }

    def update_equity(self) -> None:
        acct = self.get_account()
        self.account.equity_curve.append((datetime.now(timezone.utc), float(acct["portfolio_value"])))
        self.drawdown_tracker.update(float(acct["portfolio_value"]))

    def get_equity_curve(self) -> tuple[list[datetime], list[float]]:
        if not self.account.equity_curve:
            return [], []
        dates, values = zip(*self.account.equity_curve)
        return list(dates), list(values)

    @property
    def portfolio_value(self) -> float:
        return float(self.get_account()["portfolio_value"])

    @property
    def daily_pnl(self) -> float:
        if len(self.account.equity_curve) < 2:
            return 0.0
        return float(self.account.equity_curve[-1][1] - self.account.equity_curve[0][1])

    def reset(self) -> None:
        self.account = PaperAccount(cash=self.config.initial_capital)
        self.oms = OrderManager()
        self.drawdown_tracker = DrawdownTracker()


def compute_graduation_metrics(paper_broker: PaperBroker) -> dict[str, float]:
    """Compute metrics used to determine if a strategy is ready for live trading."""
    dates, values = paper_broker.get_equity_curve()
    if len(values) < 5:
        return {"ready": 0.0, "sharpe": 0.0, "max_dd": 0.0, "win_rate": 0.0}
    returns = np.diff(values) / np.array(values[:-1])
    sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
    peak = np.maximum.accumulate(values)
    dd = (np.array(values) - peak) / peak
    max_dd = float(np.min(dd))
    wins = sum(1 for r in returns if r > 0)
    win_rate = float(wins / len(returns))
    ready_score = min(1.0, max(0.0, (sharpe / 2.0 + (1.0 + max_dd) + win_rate) / 3.0))
    return {
        "ready": ready_score, "sharpe": sharpe, "max_dd": max_dd,
        "win_rate": win_rate, "n_days": len(returns),
    }
