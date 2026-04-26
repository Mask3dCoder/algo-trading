"""Live trading event loop: strategy orchestrator with full Phase 3 integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from quantcore.core.types import PortfolioSnapshot, Position, Side
from quantcore.execution.oms import OrderManager, market_order
from quantcore.live.runner import Broker, DataProvider, LiveConfig, LiveState
from quantcore.risk.drawdown import DrawdownAction, DrawdownState, DrawdownTracker
from quantcore.risk.limits import (
    CircuitBreaker,
    LimitStatus,
    RiskLimits,
    check_post_trade,
    check_pre_trade,
)


class TraderState(str, Enum):
    INIT = "init"
    PRE_MARKET = "pre_market"
    TRADING = "trading"
    POST_MARKET = "post_market"
    RECONCILIATION = "reconciliation"
    SHUTDOWN = "shutdown"
    EMERGENCY = "emergency"


@dataclass
class TraderConfig:
    """Configuration for the live trading engine."""
    live_config: LiveConfig = field(default_factory=LiveConfig)
    startup_checks: bool = True
    reconciliation_enabled: bool = True
    watchdog_enabled: bool = True
    max_startup_retries: int = 3
    reconnect_delay_sec: float = 5.0
    emergency_flatten_timeout_sec: float = 30.0


@dataclass
class TraderSnapshot:
    """Full state snapshot of the trading engine."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    trader_state: TraderState = TraderState.INIT
    live_state: LiveState = LiveState.IDLE
    portfolio_value: float = 0.0
    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    pending_orders: int = 0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    drawdown: DrawdownState | None = None
    circuit_breaker_active: bool = False
    active_alerts: list[str] = field(default_factory=list)
    last_signal_time: datetime | None = None
    uptime_seconds: float = 0.0


class LiveTrader:
    """Live trading event loop orchestrator.

    Ties together data feeds, strategies, risk checks, OMS, execution algos,
    and monitoring into a single event-driven loop.
    """

    def __init__(
        self,
        config: TraderConfig | None = None,
        data_provider: DataProvider | None = None,
        broker: Broker | None = None,
        strategy: Any = None,
    ) -> None:
        self.config = config or TraderConfig()
        self.data_provider = data_provider
        self.broker = broker
        self.strategy = strategy

        self.oms = OrderManager()
        self.state = TraderState.INIT
        self.live_state = LiveState.IDLE
        self.drawdown_tracker = DrawdownTracker(self.config.live_config.drawdown)
        self.circuit_breaker = CircuitBreaker(limits=self.config.live_config.risk_limits or RiskLimits())
        self._start_time = datetime.now(UTC)
        self._last_signal: datetime | None = None
        self._daily_start_value: float = 0.0
        self._weekly_start_value: float = 0.0
        self._alerts: list[str] = []
        self._positions: dict[str, Position] = {}
        self._cash: float = 0.0

    def startup(self) -> bool:
        """Run startup sequence: connect, sync, validate, enter pre-market."""
        self.state = TraderState.INIT
        if self.config.startup_checks:
            if not self._run_startup_checks():
                return False
        self._sync_portfolio()
        self._daily_start_value = self._current_value()
        self._weekly_start_value = self._daily_start_value
        self._start_time = datetime.now(UTC)
        self.state = TraderState.PRE_MARKET
        self.live_state = LiveState.RUNNING
        return True

    def _run_startup_checks(self) -> bool:
        for attempt in range(self.config.max_startup_retries):
            try:
                if self.broker:
                    acct = self.broker.get_account()
                    if not acct:
                        raise ConnectionError("Empty account response")
                if self.data_provider:
                    if not self.data_provider.is_market_open():
                        pass
                return True
            except Exception:
                if attempt < self.config.max_startup_retries - 1:
                    import time
                    time.sleep(self.config.reconnect_delay_sec)
        return False

    def tick(self) -> TraderSnapshot:
        """Single iteration of the trading loop."""
        now = datetime.now(UTC)

        if self.state == TraderState.EMERGENCY:
            return self._snapshot()

        if self.state in (TraderState.PRE_MARKET,):
            if self.data_provider and self.data_provider.is_market_open():
                self.state = TraderState.TRADING

        if self.state == TraderState.TRADING:
            self._sync_portfolio()
            self._process_signals()
            dd = self.drawdown_tracker.update(self._current_value())
            if dd.action in (DrawdownAction.HALT, DrawdownAction.FLATTEN):
                self._emergency_flatten()

            daily_pnl = self._current_value() - self._daily_start_value
            loss_pct = abs(daily_pnl) / self._daily_start_value if self._daily_start_value > 0 and daily_pnl < 0 else 0.0
            if loss_pct > self.config.live_config.max_daily_loss_pct:
                self._emergency_flatten()

            post_breaches = check_post_trade(
                PortfolioSnapshot(
                    timestamp=now, positions=self._positions,
                    total_value=self._current_value(),
                ),
                self.config.live_config.risk_limits or RiskLimits(),
            )
            if self.circuit_breaker.check(post_breaches):
                self._emergency_flatten()

            if self.data_provider and not self.data_provider.is_market_open():
                self.state = TraderState.POST_MARKET

        if self.state == TraderState.POST_MARKET and self.config.reconciliation_enabled:
            self.state = TraderState.RECONCILIATION

        return self._snapshot()

    def _process_signals(self) -> None:
        if not self.strategy or not self.data_provider:
            return
        try:
            signals = self.strategy.generate_signals(self.data_provider)
        except Exception:
            return
        self._last_signal = datetime.now(UTC)
        for sig in signals:
            symbol = str(sig.get("symbol", ""))
            price = float(sig.get("price", 0))
            strength = float(sig.get("strength", 0.5))
            direction = str(sig.get("direction", "long"))
            if price <= 0 or not symbol:
                continue
            qty = int(self._cash * self.config.live_config.max_position_pct / price) if self._cash > 0 else 0
            if qty <= 0:
                continue
            side = Side.BUY if direction == "long" else Side.SELL
            order = market_order(symbol=symbol, side=side, quantity=qty)
            snap = PortfolioSnapshot(
                timestamp=datetime.now(UTC), positions=self._positions,
                total_value=self._current_value(),
            )
            result = check_pre_trade(order, snap, self.config.live_config.risk_limits or RiskLimits())
            if result.status == LimitStatus.REJECTED:
                continue
            self.oms.submit(order)
            if self.broker:
                try:
                    self.broker.submit_order(order)
                except Exception:
                    self.oms.reject(order.id, "Broker submission failed")

    def shutdown(self) -> None:
        """Graceful shutdown: cancel orders, final sync, exit."""
        self.state = TraderState.SHUTDOWN
        self.live_state = LiveState.STOPPED
        for oid in list(self.oms.orders.keys()):
            try:
                self.oms.cancel(oid)
            except Exception:
                pass

    def _emergency_flatten(self) -> None:
        """Emergency flatten: cancel all, liquidate everything."""
        from quantcore.core.health_state import set_kill_switch_active
        set_kill_switch_active(True)
        self.state = TraderState.EMERGENCY
        self.live_state = LiveState.HALTED
        for oid in list(self.oms.orders.keys()):
            try:
                self.oms.cancel(oid)
            except Exception:
                pass
        if self.broker:
            for sym, pos in self._positions.items():
                if abs(float(pos.quantity)) < 1e-10:
                    continue
                side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
                order = market_order(sym, side, abs(int(float(pos.quantity))))
                try:
                    self.broker.submit_order(order)
                except Exception:
                    pass

    def _sync_portfolio(self) -> None:
        if self.broker:
            try:
                acct = self.broker.get_account()
                self._cash = float(acct.get("cash", 0))
                self._positions = self.broker.get_positions()
            except Exception:
                pass
        from quantcore.core.health_state import set_open_positions_count
        set_open_positions_count(len(self._positions))

    def _current_value(self) -> float:
        if self.broker:
            try:
                acct = self.broker.get_account()
                return float(acct.get("portfolio_value", acct.get("equity", 0)))
            except Exception:
                pass
        return self._cash + sum(
            float(abs(p.quantity)) * float(p.market_price)
            for p in self._positions.values()
        )

    def _snapshot(self) -> TraderSnapshot:
        dd = None
        if self.drawdown_tracker:
            dd = DrawdownState(
                peak=self.drawdown_tracker.peak,
                current_nav=self._current_value(),
                drawdown=self.drawdown_tracker.current_drawdown,
                drawdown_pct=self.drawdown_tracker.current_drawdown * 100,
                duration_days=0, max_drawdown_pct=self.drawdown_tracker.max_drawdown * 100,
                action=self.drawdown_tracker._determine_action(self.drawdown_tracker.current_drawdown),
            )
        uptime = (datetime.now(UTC) - self._start_time).total_seconds()
        return TraderSnapshot(
            trader_state=self.state, live_state=self.live_state,
            portfolio_value=self._current_value(), cash=self._cash,
            positions=self._positions, pending_orders=len(self.oms.orders),
            daily_pnl=self._current_value() - self._daily_start_value,
            weekly_pnl=self._current_value() - self._weekly_start_value,
            drawdown=dd, circuit_breaker_active=self.circuit_breaker.activated,
            active_alerts=list(self._alerts),
            last_signal_time=self._last_signal, uptime_seconds=uptime,
        )

    @property
    def is_live(self) -> bool:
        return self.state == TraderState.TRADING

    @property
    def is_healthy(self) -> bool:
        return self.state not in (TraderState.EMERGENCY, TraderState.SHUTDOWN)
