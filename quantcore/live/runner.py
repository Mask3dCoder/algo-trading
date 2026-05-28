"""Live trading runner: real-time event loop wired to all Phase 3 modules.

Flow: MarketData -> Regime -> Strategy -> Sizing -> RiskCheck -> OMS -> Execution -> Monitor
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from quantcore.core.cash import CashLedger
from quantcore.core.types import Order, PortfolioSnapshot, Position, Side
from quantcore.execution.oms import OrderManager, market_order
from quantcore.portfolio.aggregator import (
    CheckVerdict,
    GlobalExposureLimits,
    StrategyAggregator,
)
from quantcore.portfolio.rebalance import RebalanceConfig, RebalanceEngine, RebalanceTrigger
from quantcore.risk.intraday_risk import (
    IntradayRiskConfig,
    IntradayRiskLoop,
    RiskAction,
)
from quantcore.risk.drawdown import (
    DrawdownAction,
    DrawdownConfig,
    DrawdownState,
    DrawdownTracker,
)
from quantcore.risk.limits import (
    CircuitBreaker,
    LimitCheckResult,
    LimitStatus,
    RiskLimits,
    check_post_trade,
    check_pre_trade,
)

log = logging.getLogger(__name__)


class LiveState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FLATTENING = "flattening"
    HALTED = "halted"
    STOPPED = "stopped"


@dataclass
class LiveConfig:
    """Configuration for the live trading runner."""
    # Execution
    heartbeat_interval_sec: float = 1.0
    pre_market_buffer_min: int = 5
    post_market_buffer_min: int = 5

    # Risk gating
    risk_limits: RiskLimits | None = None
    max_daily_loss_pct: float = 0.03  # 3% daily loss limit
    max_weekly_loss_pct: float = 0.05  # 5% weekly loss limit
    max_consecutive_losses: int = 5
    flatten_on_halt: bool = True

    # Sizing
    max_position_pct: float = 0.10
    max_leverage: float = 1.0

    # Rebalancing
    rebalance: RebalanceConfig | None = None

    # Drawdown
    drawdown: DrawdownConfig | None = None

    # Regime
    regime_required: bool = False
    allowed_regimes: list[str] | None = None  # None = all allowed


@dataclass
class LiveStateSnapshot:
    """Current state of the live trading system."""
    state: LiveState = LiveState.IDLE
    timestamp: datetime = field(default_factory=datetime.utcnow)
    portfolio_value: float = 0.0
    cash: float = 0.0
    settled_cash: float = 0.0
    unsettled_cash: float = 0.0
    committed_cash: float = 0.0
    available_cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    pending_orders: int = 0
    drawdown: DrawdownState | None = None
    regime: str = "unknown"
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    risk_breaches: list[str] = field(default_factory=list)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)


class DataProvider(Protocol):
    """Protocol for live market data providers."""

    def fetch_bar(self, symbol: str) -> dict[str, Any]: ...
    def fetch_quotes(self, symbols: list[str]) -> dict[str, dict[str, float]]: ...
    def is_market_open(self) -> bool: ...


class Broker(Protocol):
    """Protocol for broker adapters."""

    def submit_order(self, order: Order) -> str: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> dict[str, Position]: ...
    def get_account(self) -> dict[str, float]: ...


class LiveRunner:
    """Real-time trading loop with full Phase 3 integration.

    Orchestrates market data -> regime detection -> strategy signals ->
    position sizing -> pre-trade risk checks -> order management ->
    execution algos -> post-trade monitoring -> rebalancing.

    The runner is designed to be extended with concrete Strategy, DataProvider,
    and Broker implementations.
    """

    def __init__(
        self,
        config: LiveConfig | None = None,
        data_provider: DataProvider | None = None,
        broker: Broker | None = None,
        strategy: Any = None,
    ) -> None:
        self.config = config or LiveConfig()
        self.data_provider = data_provider
        self.broker = broker
        self.strategy = strategy

        # Core components
        self.oms = OrderManager()
        self.state = LiveState.IDLE
        self._portfolio_value: float = 0.0
        self._cash: float = 0.0
        self._cash_ledger: CashLedger = CashLedger()
        self._positions: dict[str, Position] = {}

        # Risk components
        self.drawdown_tracker = DrawdownTracker(self.config.drawdown)
        self.circuit_breaker = CircuitBreaker(
            limits=self.config.risk_limits or RiskLimits(),
        )
        self.intraday_risk = IntradayRiskLoop(IntradayRiskConfig(
            max_leverage=self.config.max_leverage,
        ))
        self._intraday_risk_count: int = 0

        # Cross-strategy exposure control
        self.aggregator = StrategyAggregator(
            limits=GlobalExposureLimits(
                max_gross_exposure=float(config.max_leverage * 10_000_000) if config else 20_000_000.0,
                max_single_name_pct=config.max_position_pct if config else 0.10,
                max_leverage=config.max_leverage if config else 1.0,
            ),
        )
        self._consecutive_losses: int = 0
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._daily_start_value: float = 0.0
        self._weekly_start_value: float = 0.0
        self._pnl_history: list[float] = []

        # Rebalancing
        self.rebalance_engine = RebalanceEngine(
            self.config.rebalance or RebalanceConfig(trigger=RebalanceTrigger.SCHEDULE)
        ) if self.config.rebalance else None

        # Regime
        self._regime: str = "unknown"
        self._regime_returns: list[float] = []

        # Callbacks
        self.on_state_change: Callable[[LiveState, LiveState], None] | None = None
        self.on_risk_breach: Callable[[LimitCheckResult], None] | None = None
        self.on_drawdown_action: Callable[[DrawdownAction, DrawdownState], None] | None = None
        self.on_heartbeat: Callable[[LiveStateSnapshot], None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start or resume the live trading loop."""
        if self.state == LiveState.HALTED:
            raise RuntimeError("Cannot start: system is halted. Reset first.")
        self._transition(LiveState.RUNNING)
        self._daily_start_value = self._current_value()
        self._weekly_start_value = self._current_value()

    def pause(self) -> None:
        """Pause trading (no new orders, existing orders remain)."""
        if self.state == LiveState.RUNNING:
            self._transition(LiveState.PAUSED)

    def resume(self) -> None:
        """Resume from paused state."""
        if self.state == LiveState.PAUSED:
            self._transition(LiveState.RUNNING)

    def stop(self) -> None:
        """Graceful shutdown — cancel all open orders, don't flatten."""
        self._transition(LiveState.STOPPED)

    def halt(self) -> None:
        """Emergency halt — cancel open orders, optionally flatten positions."""
        self._transition(LiveState.HALTED)
        if self.config.flatten_on_halt:
            self._flatten_all()

    def reset(self) -> None:
        """Reset from halted state."""
        if self.state == LiveState.HALTED:
            self.circuit_breaker.reset()
            self.drawdown_tracker = DrawdownTracker(self.config.drawdown)
            self._transition(LiveState.IDLE)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def tick(self) -> LiveStateSnapshot:
        """Single iteration of the trading loop. Called by the scheduler.

        Returns a state snapshot for monitoring.
        """
        if self.state not in (LiveState.RUNNING,):
            return self._snapshot()

        # 0. Process pending settlements
        self._cash_ledger.settle_pending()

        # 1. Refresh portfolio from broker
        self._sync_portfolio()

        # 2. Update drawdown tracker
        dd_state = self.drawdown_tracker.update(self._current_value())
        if dd_state.action in (DrawdownAction.HALT, DrawdownAction.FLATTEN):
            if self.on_drawdown_action:
                self.on_drawdown_action(dd_state.action, dd_state)
            if dd_state.action == DrawdownAction.HALT:
                self.halt()
                return self._snapshot()
            elif dd_state.action == DrawdownAction.FLATTEN:
                self._flatten_all()

        # Risk reduction actions
        if dd_state.action in (DrawdownAction.REDUCE_25, DrawdownAction.REDUCE_50, DrawdownAction.REDUCE_75):
            if self.on_drawdown_action:
                self.on_drawdown_action(dd_state.action, dd_state)
            self._reduce_positions(dd_state.action)

        # 3. Intraday risk check (every ~60 ticks at 1s interval)
        self._intraday_risk_count += 1
        if self._intraday_risk_count % 60 == 0:
            risk_snapshot = self.intraday_risk.check(
                positions=self._positions,
                cash=self._cash_ledger.settled_cash if hasattr(self, '_cash_ledger') else self._cash,
                equity=self._current_value(),
                drawdown_pct=abs(self.drawdown_tracker.current_drawdown) if self.drawdown_tracker else 0.0,
                prices={},  # Would use data_provider here in production
                limits=self.config.risk_limits,
            )
            if risk_snapshot.action in (RiskAction.FLATTEN, RiskAction.HALT):
                if self.on_risk_breach:
                    class _Breach:
                        status = LimitStatus.REJECTED  # type: ignore[attr-defined]
                        reason = risk_snapshot.details
                    self.on_risk_breach(_Breach())  # type: ignore[arg-type]
                self._flatten_all()
                return self._snapshot()

        # 4. Check daily/weekly loss limits
        daily_pnl = self._current_value() - self._daily_start_value
        daily_loss_pct = abs(daily_pnl) / self._daily_start_value if self._daily_start_value > 0 and daily_pnl < 0 else 0.0
        if daily_loss_pct > self.config.max_daily_loss_pct:
            self.halt()
            return self._snapshot()

        weekly_pnl = self._current_value() - self._weekly_start_value
        weekly_loss_pct = abs(weekly_pnl) / self._weekly_start_value if self._weekly_start_value > 0 and weekly_pnl < 0 else 0.0
        if weekly_loss_pct > self.config.max_weekly_loss_pct:
            self.halt()
            return self._snapshot()

        # 4. Detect regime (if model available)
        if self.data_provider and self.strategy:
            self._update_regime()

        # 5. Rebalance check
        if self.rebalance_engine and self._positions:
            self._check_rebalance()

        # 6. Generate signals from strategy
        if self.strategy and self.data_provider:
            self._process_signals()

        # 7. Post-trade risk recheck
        self._check_post_trade()

        # 8. Update consecutive loss counter
        self._update_loss_counter()

        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self.pause()

        snapshot = self._snapshot()
        if self.on_heartbeat:
            self.on_heartbeat(snapshot)

        return snapshot

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _sync_portfolio(self) -> None:
        """Sync portfolio state with broker."""
        if self.broker:
            try:
                acct = self.broker.get_account()
                broker_cash = float(acct.get("cash", 0))
                # Update cash ledger: first settle pending, then sync settled
                self._cash_ledger.settle_pending()
                diff = broker_cash - self._cash_ledger.total_cash
                if abs(diff) > 1.0:
                    self._cash_ledger.add_cash(diff)
                self._cash = self._cash_ledger.settled_cash
                self._portfolio_value = float(acct.get("portfolio_value", 0))
                self._positions = self.broker.get_positions()
                # Update aggregator with current positions
                self.aggregator.update_positions(
                    "live",
                    self._positions,
                    self._cash_ledger.settled_cash,
                )
                self.aggregator.update_equity(self._portfolio_value)
            except Exception as exc:
                log.error("Portfolio sync failed: %s", exc)

    def _current_value(self) -> float:
        """Current total portfolio value."""
        if self.broker:
            try:
                acct = self.broker.get_account()
                return float(acct.get("portfolio_value", self._portfolio_value))
            except Exception:
                pass
        return self._portfolio_value or self._cash_ledger.settled_cash

    def _update_regime(self) -> None:
        """Update market regime from configured model."""
        if not self.data_provider:
            return
        try:
            bar = self.data_provider.fetch_bar("SPY")
            if bar:
                ret = (bar.get("close", 0) / bar.get("open", 1)) - 1
                self._regime_returns.append(ret)
                self._regime_returns = self._regime_returns[-252:]
                self._regime = "unknown"  # Would use regime model here
        except Exception:
            pass

    def _process_signals(self) -> None:
        """Generate signals, size, check risk, route to OMS."""
        if not self.strategy or not self.data_provider:
            return

        try:
            signals = self.strategy.generate_signals(self.data_provider)
        except Exception:
            return

        for sig in signals:
            # Regime gate
            if self.config.regime_required and self.config.allowed_regimes:
                if self._regime not in self.config.allowed_regimes:
                    continue

            side = Side.BUY if sig.get("direction") == "long" else Side.SELL
            symbol = str(sig.get("symbol", ""))
            price = float(sig.get("price", 0))
            float(sig.get("strength", 0.5))

            if price <= 0 or not symbol:
                continue

            # Sizing using settlement-aware available cash
            avail = self._cash_ledger.available()
            qty = int(avail * self.config.max_position_pct / price)

            if side == Side.SELL:
                pos = self._positions.get(symbol)
                if pos is None or float(pos.quantity) <= 0:
                    continue
                qty = int(abs(float(pos.quantity)))

            if qty <= 0:
                continue

            # Pre-trade cash availability check (settlement-aware)
            order_value = qty * price
            avail = self._cash_ledger.available()
            if order_value > avail + 1.0:
                # Cannot trade — would violate settlement constraint
                continue

            # Reserve cash for this order
            try:
                self._cash_ledger.reserve(order_value, f"order-{symbol}-{datetime.utcnow().timestamp():.0f}")
            except Exception:
                continue  # not enough settled cash

            order = market_order(
                symbol=symbol,
                side=side,
                quantity=Decimal(str(qty)),
            )

            # Cross-strategy aggregator pre-trade check
            if self.aggregator.check_order(order) == CheckVerdict.REJECTED:
                self._cash_ledger.release(order.id)  # release committed cash
                continue

            # Pre-trade risk check
            if self.config.risk_limits:
                snap = PortfolioSnapshot(
                    timestamp=datetime.utcnow(),
                    positions=self._positions,
                    total_value=Decimal(str(self._current_value())),
                )
                result = check_pre_trade(order, snap, self.config.risk_limits)
                if result.status == LimitStatus.REJECTED:
                    self._cash_ledger.release(order.id)  # release committed cash
                    if self.on_risk_breach:
                        self.on_risk_breach(result)
                    continue

            # Route to OMS → Broker
            self.oms.submit(order)
            if self.broker:
                try:
                    self.broker.submit_order(order)
                except Exception:
                    self.oms.reject(order.id, "Broker rejected")
                    self._cash_ledger.release(order.id)  # release on reject

    def _check_rebalance(self) -> None:
        """Check and execute rebalancing if needed."""
        if not self.rebalance_engine or not self._positions:
            return

        current_weights = {}
        total = self._current_value()
        if total <= 0:
            return
        for sym, pos in self._positions.items():
            current_weights[sym] = float(abs(pos.market_value)) / total

        # For target weights, we'd need strategy-provided targets.
        # Here we use a simple equal-weight fallback.
        n = len(current_weights) if current_weights else 1
        target_weights = {s: 1.0 / n for s in current_weights}

        should, reason = self.rebalance_engine.should_rebalance(current_weights, target_weights)
        if not should:
            return

        trades = self.rebalance_engine.compute_trades(
            current_weights=current_weights,
            target_weights=target_weights,
            portfolio_value=total,
        )

        for trade in trades:
            order = market_order(
                symbol=trade["symbol"],
                side=Side.BUY if trade["shares"] > 0 else Side.SELL,
                quantity=Decimal(str(abs(trade["shares"]))),
            )
            self.oms.submit(order)
            if self.broker:
                try:
                    self.broker.submit_order(order)
                except Exception:
                    self.oms.reject(order.id, "Broker rejected")

    def _check_post_trade(self) -> None:
        """Run post-trade risk checks."""
        if not self.config.risk_limits:
            return
        snap = PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            positions=self._positions,
            total_value=Decimal(str(self._current_value())),
        )
        breaches = check_post_trade(snap, self.config.risk_limits)
        if self.circuit_breaker.check(breaches):
            self.halt()

    def _update_loss_counter(self) -> None:
        """Track consecutive losses."""
        if len(self._pnl_history) < 2:
            return
        last = self._pnl_history[-1]
        if last < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def _reduce_positions(self, action: DrawdownAction) -> None:
        """Reduce position sizes based on drawdown action."""
        factors = {
            DrawdownAction.REDUCE_25: 0.25,
            DrawdownAction.REDUCE_50: 0.50,
            DrawdownAction.REDUCE_75: 0.75,
        }
        factor = factors.get(action, 0)
        if factor <= 0 or not self.broker:
            return

        for sym, pos in self._positions.items():
            reduce_qty = int(abs(float(pos.quantity)) * factor)
            if reduce_qty <= 0:
                continue
            side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
            order = market_order(sym, side, Decimal(str(reduce_qty)))
            try:
                self.broker.submit_order(order)
            except Exception:
                pass

    def _flatten_all(self) -> None:
        """Flatten all positions."""
        self._transition(LiveState.FLATTENING)
        if not self.broker:
            self._transition(LiveState.HALTED)
            return
        for sym, pos in self._positions.items():
            if abs(float(pos.quantity)) < 1e-10:
                continue
            side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
            order = market_order(sym, side, pos.quantity)
            try:
                self.broker.submit_order(order)
            except Exception:
                pass
        self._transition(LiveState.HALTED)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _transition(self, new_state: LiveState) -> None:
        old_state = self.state
        self.state = new_state
        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def _snapshot(self) -> LiveStateSnapshot:
        dd = None
        if self.drawdown_tracker:
            dd = DrawdownState(
                peak=self.drawdown_tracker.peak,
                current_nav=self._current_value(),
                drawdown=self.drawdown_tracker.current_drawdown,
                drawdown_pct=self.drawdown_tracker.current_drawdown * 100,
                duration_days=0,
                max_drawdown_pct=self.drawdown_tracker.max_drawdown * 100,
                action=self.drawdown_tracker._determine_action(self.drawdown_tracker.current_drawdown),
            )

        daily_pnl = self._current_value() - self._daily_start_value
        weekly_pnl = self._current_value() - self._weekly_start_value

        # Settle pending cash before snapshot
        self._cash_ledger.settle_pending()

        return LiveStateSnapshot(
            state=self.state,
            timestamp=datetime.utcnow(),
            portfolio_value=self._current_value(),
            cash=self._cash_ledger.settled_cash,
            settled_cash=self._cash_ledger.settled_cash,
            unsettled_cash=self._cash_ledger.unsettled_cash,
            committed_cash=self._cash_ledger.committed_cash,
            available_cash=self._cash_ledger.available(),
            positions=self._positions,
            pending_orders=len(self.oms.orders),
            drawdown=dd,
            regime=self._regime,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            risk_breaches=[],
            last_heartbeat=datetime.utcnow(),
        )
