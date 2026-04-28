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
from quantcore.backtest.impact_config import (
    MarketImpactConfig,
    MarketImpactEngine,
    VolumeFilterConfig,
)
from quantcore.backtest.financing import (
    BorrowCostRegistry,
    FinancingConfig,
    compute_daily_borrow_cost,
    compute_daily_margin_interest,
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
    notional_size: float | None = None  # strategy-computed position size, overrides sizing_pct


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
        self.total_market_impact_cost: float = 0.0
        # Financing tracking (P2)
        self.total_borrow_cost: float = 0.0
        self.total_margin_interest: float = 0.0
        self._financing_config: FinancingConfig | None = None

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

        # Track market impact cost as a separate line item (already reflected in fill price)
        self.total_market_impact_cost += float(fill.market_impact_cost)

        self.trades.append(fill)

        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal(str(qty if fill.order_id.endswith("_buy") else -qty)),
                avg_cost=Decimal(str(price)),
                market_price=Decimal(str(price)),
            )
            self.cash -= qty * price if fill.order_id.endswith("_buy") else 0.0
            self.cash += qty * price if fill.order_id.endswith("_sell") else 0.0
        else:
            pos = self.positions[symbol]
            current_qty = float(pos.quantity)
            new_qty = current_qty + qty if fill.order_id.endswith("_buy") else current_qty - qty
            # Update average cost for buys
            if fill.order_id.endswith("_buy"):
                total_cost = float(pos.avg_cost) * abs(current_qty) + price * qty
                new_avg = total_cost / abs(new_qty) if abs(new_qty) > 1e-10 else float(pos.avg_cost)
            else:
                new_avg = float(pos.avg_cost)

            realized = 0.0
            if fill.order_id.endswith("_sell"):
                if current_qty > 0:
                    realized = (price - float(pos.avg_cost)) * qty
                # current_qty <= 0: adding to short, no realized P&L yet
                self.cash += qty * price
            elif fill.order_id.endswith("_buy"):
                if current_qty < 0:
                    realized = (float(pos.avg_cost) - price) * qty
                # current_qty >= 0: adding to long, no realized P&L yet
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

    def set_financing_config(self, config: FinancingConfig | None) -> None:
        """Set the financing config for borrow/margin cost calculations."""
        self._financing_config = config

    def daily_financing_update(self) -> None:
        """Accrue borrowing costs and margin interest for the current bar.

        Called once per bar in the engine loop after price updates.
        Deducts costs from cash and updates tracking totals.
        If margin limit is breached, sets a flag for forced liquidation.
        """
        config = self._financing_config
        if config is None:
            return

        registry = config.borrow_cost_registry
        total_short_value = 0.0
        total_long_value = 0.0

        for symbol, pos in self.positions.items():
            qty = float(pos.quantity)
            price = self._prices.get(symbol, float(pos.market_price))
            market_value = qty * price

            if qty < 0:  # Short position
                abs_value = abs(market_value)
                total_short_value += abs_value
                borrow_rate = registry.get_rate(symbol) if registry else 0.005
                daily_cost = abs_value * borrow_rate / 360.0
                if daily_cost > 0:
                    self.cash -= daily_cost
                    self.total_borrow_cost += daily_cost
            elif qty > 0:  # Long position
                total_long_value += market_value

        # Margin interest on leveraged long positions
        if config.margin_rate > 0 and total_long_value > self.cash:
            margin_used = total_long_value - self.cash
            daily_interest = margin_used * config.margin_rate / 360.0
            if daily_interest > 0:
                self.cash -= daily_interest
                self.total_margin_interest += daily_interest

    @property
    def is_margin_breach(self) -> bool:
        """Check if margin usage exceeds the configured limit."""
        config = self._financing_config
        if config is None:
            return False
        total_value = self.total_value
        if total_value <= 0:
            return False
        # Margin used = total position value / equity (1.0 = no leverage)
        long_value = sum(
            max(0, float(p.quantity)) * self._prices.get(s, float(p.market_price))
            for s, p in self.positions.items()
        )
        short_value = sum(
            abs(min(0, float(p.quantity))) * self._prices.get(s, float(p.market_price))
            for s, p in self.positions.items()
        )
        total_exposure = long_value + short_value
        leverage = total_exposure / total_value
        return leverage > config.margin_limit


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
    """Simulates order fills with slippage, commissions, and market impact."""

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        fill_at: str = "next_open",
        impact_config: MarketImpactConfig | None = None,
        volume_filter_config: VolumeFilterConfig | None = None,
    ) -> None:
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.fill_at = fill_at
        self._next_bar: OHLCV | None = None
        self._current_bar: OHLCV | None = None
        self._impact_engine: MarketImpactEngine | None = None
        self._permanent_impact: dict[str, float] = {}  # symbol → cumulative perm impact in price units

        if impact_config is not None:
            self._impact_engine = MarketImpactEngine(impact_config, volume_filter_config)

    def set_data(self, data: pd.DataFrame) -> None:
        """Provide historical data for ADV/sigma estimation."""
        if self._impact_engine is not None:
            self._impact_engine.set_data(data)

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

        # Apply base slippage
        slip = fill_price * self.slippage_bps / 10_000.0
        if order.side == Side.BUY:
            fill_price += slip
        else:
            fill_price -= slip

        qty = float(order.quantity)
        bar_volume = int(bar.volume)
        market_impact_cost = 0.0
        impact_bps = 0.0

        # Volume filter and market impact
        if self._impact_engine is not None:
            # Check volume filter (may reduce fill quantity)
            adjusted_qty, is_clipped = self._impact_engine.check_volume_filter(
                qty, bar_volume
            )
            if adjusted_qty <= 0:
                return None  # Fill rejected by volume filter

            if is_clipped:
                qty = adjusted_qty

            # Compute market impact
            result = self._impact_engine.estimate_impact(
                shares=qty,
                price=float(bar.open),
                bar_volume=bar_volume,
            )
            impact_bps = result.total_cost_bps

            # Apply temporary impact to fill price
            temp_impact_price = float(bar.open) * result.temp_impact_bps / 10_000.0
            if order.side == Side.BUY:
                fill_price += temp_impact_price
            else:
                fill_price -= temp_impact_price

            # Apply accumulated permanent impact from prior trades
            sym_perm = self._permanent_impact.get(order.symbol, 0.0)
            if sym_perm != 0.0:
                if order.side == Side.BUY:
                    fill_price += sym_perm
                else:
                    fill_price -= sym_perm

            # Update cumulative permanent impact
            perm_impact_price = float(bar.open) * result.perm_impact_bps / 10_000.0
            self._permanent_impact[order.symbol] = (
                self._permanent_impact.get(order.symbol, 0.0) + perm_impact_price
            )

            market_impact_cost = result.total_cost_dollars

        # Check limit orders
        if order.type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == Side.BUY and fill_price > float(order.limit_price):
                return None
            if order.side == Side.SELL and fill_price < float(order.limit_price):
                return None

        commission = qty * self.commission_per_share

        return Fill(
            order_id=f"{order.id}_{order.side.value}",
            price=Decimal(str(round(fill_price, 6))),
            quantity=Decimal(str(qty)),
            commission=Decimal(str(round(commission, 4))),
            timestamp=bar.timestamp,
            slippage_bps=Decimal(str(self.slippage_bps)),
            market_impact_cost=Decimal(str(round(market_impact_cost, 4))),
            impact_bps=Decimal(str(round(impact_bps, 4))),
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
        impact_config: MarketImpactConfig | None = None,
        volume_filter_config: VolumeFilterConfig | None = None,
        financing_config: FinancingConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.portfolio = PortfolioState(initial_cash)
        self.portfolio.set_financing_config(financing_config)
        self._financing_config = financing_config
        self.fill_sim = FillSimulator(
            slippage_bps,
            commission_per_share,
            impact_config=impact_config,
            volume_filter_config=volume_filter_config,
        )
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
        self, side: Side, price: float, bar: OHLCV, sig: SignalData
    ) -> int:
        """Compute position size using the configured sizing strategy."""
        cash = self.portfolio.cash
        pv = self.portfolio.total_value

        # Strategy-computed notional_size takes priority
        if sig.notional_size is not None and sig.notional_size > 0:
            qty = int(sig.notional_size / price)
            if side == Side.BUY:
                return min(qty, int(cash / price)) if cash > 0 else 0
            else:
                # Short: limit to 2x portfolio value (Reg T margin)
                return min(qty, int(pv * 2.0 / price))

        if cash <= 0 and side == Side.BUY:
            return 0

        if self.sizing_fn is not None:
            context = {
                "cash": cash,
                "price": price,
                "signal_strength": sig.strength,
                "bar": bar,
                "portfolio_value": pv,
            }
            return int(self.sizing_fn(cash, price, context))

        qty = int(pv * self.sizing_pct / price)
        if side == Side.BUY:
            return min(qty, int(cash / price)) if cash > 0 else 0
        else:
            return qty

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
        self.portfolio.set_financing_config(self._financing_config)
        self._rejected_orders = []

        # Set up market impact engine with historical data
        self.fill_sim.set_data(data)

        if self.drawdown_tracker:
            self.drawdown_tracker = DrawdownTracker(
                self.drawdown_tracker.config if hasattr(self.drawdown_tracker, 'config') else DrawdownConfig()
            )

        pending_orders: list[Order] = []
        daily_returns: list[float] = []
        bar: OHLCV | None = None

        while handler.has_next():
            bar = handler.next_bar()
            if bar is None:
                break

            # Update portfolio prices
            self.portfolio.update_price(bar.symbol, float(bar.close))

            # Daily financing cost accrual (borrow costs + margin interest)
            self.portfolio.daily_financing_update()

            # Forced liquidation on margin breach
            if self.portfolio.is_margin_breach:
                for sym in list(self.portfolio.positions.keys()):
                    pos = self.portfolio.positions[sym]
                    close_qty = int(abs(float(pos.quantity)))
                    if close_qty == 0:
                        continue
                    close_side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
                    oid = f"ord_margin_{sym}_{'sell' if close_side == Side.SELL else 'buy_cover'}"
                    order = Order(
                        id=oid, symbol=sym, side=close_side,
                        type=OrderType.MARKET, quantity=Decimal(str(close_qty)),
                    )
                    self.fill_sim.set_bars(bar, None)
                    fill = self.fill_sim.simulate_fill(order)
                    if fill is not None:
                        self.portfolio.apply_fill(fill, sym)
                        self.strategy.on_fill(fill, self.portfolio)
                # Skip signal processing for this bar (all positions closed)
                self.portfolio.record_equity(
                    bar.timestamp.timestamp() if hasattr(bar.timestamp, "timestamp") else 0.0
                )
                continue

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
                price = float(bar.close)
                pos = self.portfolio.positions.get(sig.symbol)
                current_qty = float(pos.quantity) if pos else 0.0

                # "flat" signal: close any existing position
                if sig.direction == "flat":
                    if abs(current_qty) > 1e-8:
                        close_side = Side.SELL if current_qty > 0 else Side.BUY
                        close_order = Order(
                            id=f"ord_{bar.timestamp.isoformat()}_{sig.symbol}_exit",
                            symbol=sig.symbol, side=close_side, type=OrderType.MARKET,
                            quantity=Decimal(str(int(abs(current_qty)))),
                        )
                        pending_orders.append(close_order)
                    continue

                if sig.direction == "long":
                    side = Side.BUY
                elif sig.direction == "short":
                    side = Side.SELL
                else:
                    continue

                # Close opposing position before opening new one
                if side == Side.BUY and current_qty < 0:
                    cover_qty = int(abs(current_qty))
                    cover_order = Order(
                        id=f"ord_{bar.timestamp.isoformat()}_{sig.symbol}_buy_cover",
                        symbol=sig.symbol, side=Side.BUY, type=OrderType.MARKET,
                        quantity=Decimal(str(cover_qty)),
                    )
                    pending_orders.append(cover_order)
                elif side == Side.SELL and current_qty > 0:
                    close_qty = int(abs(current_qty))
                    close_order = Order(
                        id=f"ord_{bar.timestamp.isoformat()}_{sig.symbol}_sell_close",
                        symbol=sig.symbol, side=Side.SELL, type=OrderType.MARKET,
                        quantity=Decimal(str(close_qty)),
                    )
                    pending_orders.append(close_order)

                qty = self._compute_sizing(side, price, bar, sig)
                if qty <= 0:
                    continue

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

        # --- End-of-period cleanup: close remaining positions ---
        if bar is not None:
            # Process any pending orders from the final bar
            for order in pending_orders:
                self.fill_sim.set_bars(bar, None)
                fill = self.fill_sim.simulate_fill(order)
                if fill is not None:
                    self.portfolio.apply_fill(fill, order.symbol)
                    self.strategy.on_fill(fill, self.portfolio)
            pending_orders.clear()

            # Close any remaining open positions at the last available price
            for sym in list(self.portfolio.positions.keys()):
                pos = self.portfolio.positions[sym]
                close_qty = int(abs(float(pos.quantity)))
                if close_qty == 0:
                    continue
                close_side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
                oid = f"ord_eop_{sym}_{'sell' if close_side == Side.SELL else 'buy_cover'}"
                order = Order(
                    id=oid,
                    symbol=sym,
                    side=close_side,
                    type=OrderType.MARKET,
                    quantity=Decimal(str(close_qty)),
                )
                self.fill_sim.set_bars(bar, None)
                fill = self.fill_sim.simulate_fill(order)
                if fill is not None:
                    self.portfolio.apply_fill(fill, sym)
                    self.strategy.on_fill(fill, self.portfolio)
                    self.portfolio.signal_log.append({
                        "timestamp": bar.timestamp,
                        "symbol": sym,
                        "direction": "eop_close",
                        "strength": 1.0,
                        "regime": self.portfolio.regime_state,
                    })

            # Record final equity after EOP closes
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
        trade_pnls = self._compute_trade_pnls(self.portfolio.trades)

        # Detect multi-symbol: count unique symbols across all trades
        # Symbol is embedded in order_id (not a direct Fill field)
        unique_symbols = set()
        for t in self.portfolio.trades:
            unique_symbols.add(self._parse_symbol_from_fill(t.order_id))
        n_symbols = len(unique_symbols)

        return generate_performance_report(
            equity,
            returns,
            np.array(trade_pnls) if trade_pnls else None,
            total_market_impact_cost=self.portfolio.total_market_impact_cost,
            total_borrow_cost=self.portfolio.total_borrow_cost,
            total_margin_interest=self.portfolio.total_margin_interest,
            is_multi_symbol=(n_symbols > 1),
            constituent_symbol_count=n_symbols,
        )

    @staticmethod
    def _parse_symbol_from_fill(order_id: str) -> str:
        """Extract symbol from a fill's order_id.

        order_id format: ord_{iso_timestamp}_{symbol}_{side}_{side}
        e.g. 'ord_2020-01-02T14:30:00_AAPL_buy_buy'
        """
        base = order_id.rsplit("_", 2)[0]
        return base.split("_", 2)[2]

    @staticmethod
    def _compute_trade_pnls(trades: list) -> list[float]:
        """Compute realized P&Ls via FIFO pairing of buys and sells per symbol."""
        from collections import defaultdict, deque

        by_symbol: dict[str, list] = defaultdict(list)
        for fill in trades:
            symbol = BacktestEngine._parse_symbol_from_fill(fill.order_id)
            by_symbol[symbol].append(fill)

        realized: list[float] = []
        for fills in by_symbol.values():
            buy_queue: deque = deque()
            for fill in fills:
                qty = float(fill.quantity)
                price = float(fill.price)
                comm = float(fill.commission)

                if fill.order_id.endswith("_buy"):
                    buy_queue.append((price, qty, comm))
                elif fill.order_id.endswith("_sell"):
                    sell_qty = qty
                    sell_price = price
                    total_sell_qty = sell_qty

                    while sell_qty > 0 and buy_queue:
                        buy_price, buy_qty, buy_comm = buy_queue[0]
                        matched = min(sell_qty, buy_qty)

                        buy_comm_alloc = buy_comm * (matched / buy_qty)
                        sell_comm_alloc = comm * (matched / total_sell_qty)

                        pnl = (sell_price - buy_price) * matched - buy_comm_alloc - sell_comm_alloc
                        realized.append(pnl)

                        sell_qty -= matched
                        if matched >= buy_qty:
                            buy_queue.popleft()
                        else:
                            buy_queue[0] = (buy_price, buy_qty - matched, buy_comm - buy_comm_alloc)

        return realized

    def risk_metrics(self) -> dict[str, float]:
        """Compute risk-adjusted metrics for the backtest using the risk module."""
        from quantcore.risk.metrics import compute_all_metrics
        if not self.portfolio.equity_curve or len(self.portfolio.equity_curve) < 2:
            return {}
        equity = np.array([v for _, v in self.portfolio.equity_curve])
        returns = compute_returns(equity)
        return compute_all_metrics(returns)
