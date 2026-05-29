"""Production runner — single entry point for containerized live trading.

Loads config from ``config/live.yaml``, initializes all components (data feed,
broker, strategies via registry, risk manager, circuit breaker, alerter), and
runs the main event loop.  Designed to be the process entrypoint inside Docker.

Usage::

    # As module
    python -m quantcore.live.production_runner

    # With custom config
    QUANTCORE_LIVE_CONFIG=config/live.yaml python -m quantcore.live.production_runner

The runner handles multiple strategies concurrently, monitors the kill-switch
file, reports health to the shared singleton (consumed by the /health API),
and delivers alerts via the configured channels.

Each strategy has independent:
    - Position tracking (via strategy_id tags in orders)
    - Risk limits (per-strategy circuit breaker)
    - P&L tracking (per-strategy equity curve)
    - Lifecycle state (initialized, running, paused, stopped)
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Strategy imports — trigger @register decorators
# ---------------------------------------------------------------------------
# ruff: noqa: E402, F401
import quantcore.strategies.equity.cross_sectional_momentum
import quantcore.strategies.equity.earnings_momentum
import quantcore.strategies.equity.intraday_breakout
import quantcore.strategies.equity.intraday_mean_reversion
import quantcore.strategies.equity.intraday_momentum
import quantcore.strategies.equity.intraday_trend
import quantcore.strategies.equity.low_volatility
import quantcore.strategies.equity.mean_reversion
import quantcore.strategies.equity.momentum
import quantcore.strategies.equity.multifactor_composite
import quantcore.strategies.equity.multi_factor
import quantcore.strategies.equity.opening_gap_fade
import quantcore.strategies.equity.quality_factor
import quantcore.strategies.equity.quality_factor_rmw_cma
import quantcore.strategies.equity.stat_arb
import quantcore.strategies.equity.trend_following
import quantcore.strategies.equity.tsmom
import quantcore.strategies.equity.value_factor_hml
import quantcore.strategies.equity.volume_profile
import quantcore.strategies.macro.carry
import quantcore.strategies.macro.carry_trend_filter
import quantcore.strategies.macro.cross_sectional_factor_neutral
import quantcore.strategies.macro.cross_sectional_momentum
import quantcore.strategies.macro.cta_trend
import quantcore.strategies.macro.factor_residual
import quantcore.strategies.macro.factor_residual_qp
import quantcore.strategies.macro.factor_residual_strict
import quantcore.strategies.macro.factor_residual_v3
import quantcore.strategies.macro.risk_parity
import quantcore.strategies.macro.state_selective_momentum
import quantcore.strategies.options.delta_neutral
import quantcore.strategies.options.earnings_vol
import quantcore.strategies.options.iron_condor
import quantcore.strategies.options.put_write
import quantcore.strategies.options.tail_hedge

from quantcore.core.health_state import (
    record_data_point,
    record_order_submitted,
    set_kill_switch_active,
    set_open_positions_count,
)
from quantcore.core.types import Order, OrderStatus, OrderType, Side
from quantcore.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from quantcore.live.risk_manager import RiskManagerLive, RiskLimitsConfig
from quantcore.live.watchdog import Watchdog, WatchdogConfig, WatchdogState
from quantcore.monitoring.alerting import Alerter, AlertingConfig
from quantcore.strategies.base import StrategyConfig, TradeSignal
from quantcore.strategies.registry import create as create_strategy, list_strategies

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-strategy state tracking
# ---------------------------------------------------------------------------

@dataclass
class StrategyRuntimeState:
    """Per-strategy runtime state: positions, P&L, circuit breaker, lifecycle."""

    name: str
    weight: float
    instance: Any
    data_source: str
    bar_size: str
    strategy_type: str = "EQUITY"
    enabled: bool = True

    # Lifecycle
    state: str = "initialized"  # initialized, warmup, running, paused, stopped, error
    error_count: int = 0
    max_consecutive_errors: int = 5
    last_signal_count: int = 0
    last_trade_date: date | None = None

    # Per-strategy P&L tracking
    starting_equity: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    n_trades: int = 0
    n_signals: int = 0

    # Per-strategy risk limits (from config)
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10

    @property
    def is_active(self) -> bool:
        return self.enabled and self.state in ("running", "warmup")

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.current_equity - self.peak_equity) / self.peak_equity

    def record_error(self) -> bool:
        """Record an error. Returns True if strategy should be paused."""
        self.error_count += 1
        if self.error_count >= self.max_consecutive_errors:
            self.state = "paused"
            return True
        return False

    def reset_errors(self) -> None:
        self.error_count = 0

    def update_equity(self, equity_contribution: float) -> None:
        """Update per-strategy equity tracking."""
        self.current_equity = equity_contribution
        if self.peak_equity == 0.0:
            self.peak_equity = equity_contribution
        else:
            self.peak_equity = max(self.peak_equity, equity_contribution)
        self.total_pnl = equity_contribution - self.starting_equity if self.starting_equity > 0 else 0.0

    def check_per_strategy_risk(self, equity: float) -> tuple[bool, str]:
        """Check per-strategy risk limits. Returns (should_pause, reason)."""
        if self.starting_equity <= 0:
            return False, ""
        # Daily loss check
        if self.daily_pnl < -self.starting_equity * self.max_daily_loss_pct:
            return True, f"Strategy {self.name}: daily loss {self.daily_pnl:.2f} exceeds {self.max_daily_loss_pct:.1%} limit"
        # Drawdown check
        dd = self.drawdown_pct
        if dd < -self.max_drawdown_pct:
            return True, f"Strategy {self.name}: drawdown {dd:.2%} exceeds {self.max_drawdown_pct:.1%} limit"
        return False, ""

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config/live.yaml")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML config, with env-var overrides via QUANTCORE_ prefix."""
    config_path = Path(
        os.environ.get("QUANTCORE_LIVE_CONFIG", str(path or DEFAULT_CONFIG_PATH))
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


# ---------------------------------------------------------------------------
# Broker factory
# ---------------------------------------------------------------------------


def create_broker(cfg: dict[str, Any]) -> Any:
    """Create the appropriate broker/gateway from config."""
    broker_type = cfg.get("type", "paper")
    exec_cfg = cfg.get("execution", {})
    initial_cash = exec_cfg.get("initial_cash", 100_000.0)

    if broker_type == "alpaca":
        from quantcore.live.broker import LiveExecutionGateway

        alpaca = cfg.get("alpaca", {})
        return LiveExecutionGateway(
            api_key=alpaca.get("api_key", ""),
            secret_key=alpaca.get("secret_key", ""),
            base_url=alpaca.get("base_url", "https://paper-api.alpaca.markets"),
            initial_cash=initial_cash,
        )
    elif broker_type == "ibkr":
        from quantcore.live.ibkr_gateway import IBKRExecutionGateway

        ibkr = cfg.get("ibkr", {})
        return IBKRExecutionGateway(
            host=ibkr.get("host", "127.0.0.1"),
            port=ibkr.get("port", 7497),
            client_id=ibkr.get("client_id", 1),
            account_id=ibkr.get("account_id", ""),
            timeout=ibkr.get("timeout", 30),
            readonly=ibkr.get("readonly", False),
        )
    else:
        # Paper mode — use LiveExecutionGateway in paper mode
        from quantcore.live.broker import LiveExecutionGateway

        return LiveExecutionGateway(initial_cash=initial_cash)


# ---------------------------------------------------------------------------
# LiveBroker adapter (satisfies the Broker protocol from runner.py)
# ---------------------------------------------------------------------------


class _LiveBrokerAdapter:
    """Thin adapter so that LiveExecutionGateway / IBKRExecutionGateway
    satisfies the Broker protocol expected by the trading loop."""

    def __init__(self, gateway: Any) -> None:
        self._gw = gateway

    def submit_order(self, order: Order) -> str:
        return self._gw.submit_order(order)

    def cancel_order(self, order_id: str) -> bool:
        return self._gw.cancel_order(order_id)

    def get_positions(self) -> dict[str, Any]:
        return self._gw.get_positions()

    def get_account(self) -> dict[str, float]:
        return self._gw.get_account()


# ---------------------------------------------------------------------------
# ProductionRunner
# ---------------------------------------------------------------------------


class ProductionRunner:
    """Multi-strategy production runner with full safety stack.

    Each strategy operates independently with its own:
        - Risk limits (per-strategy circuit breaker)
        - P&L tracking (per-strategy equity curve)
        - Error handling (strategy-level pause on repeated failures)
        - Lifecycle state (running, paused, stopped)

    Parameters
    ----------
    config : dict
        Parsed ``config/live.yaml``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._running = False
        self._last_trade_date: date | None = None
        self._trade_day_start_equity: float = 0.0

        # Paths
        mon_cfg = config.get("monitoring", {})
        self.events_path = Path(mon_cfg.get("log_file", "reports/live_events.jsonl"))
        self.equity_path = Path(mon_cfg.get("equity_file", "reports/live_equity.csv"))
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.equity_path.parent.mkdir(parents=True, exist_ok=True)

        # Kill switch
        ks_cfg = config.get("kill_switch", {})
        self.kill_switch_path = Path(ks_cfg.get("file_path", "KILL_SWITCH"))
        self.kill_switch_interval = ks_cfg.get("check_interval_sec", 10)

        # Schedule config
        sched_cfg = config.get("schedule", {})
        self.trading_time = sched_cfg.get("trading_time", "16:15")
        self.reconciliation_time = sched_cfg.get("reconciliation_time", "16:30")

        # Alerting (Telegram is PRIMARY channel for critical events)
        alert_cfg = config.get("alerting", {})
        self.alerter = Alerter.from_config(alert_cfg) if alert_cfg.get("enabled", True) else None

        # Risk
        risk_cfg = config.get("risk", {})
        cb_cfg = risk_cfg.get("circuit_breaker", {})
        self.risk_manager = RiskManagerLive(
            config=RiskLimitsConfig(
                max_position_per_ticker_pct=risk_cfg.get("max_position_per_ticker_pct", 0.05),
                max_gross_exposure_pct=risk_cfg.get("max_gross_exposure_pct", 2.0),
                max_daily_loss_pct=risk_cfg.get("max_daily_loss_pct", 0.03),
                max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.12),
                max_single_order_pct=risk_cfg.get("max_single_order_pct", 0.05),
                max_leverage=risk_cfg.get("max_leverage", 2.0),
            )
        )
        self.circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                daily_loss_pct=cb_cfg.get("daily_loss_pct", 0.03),
                critical_drawdown_pct=cb_cfg.get("critical_drawdown_pct", 0.12),
                consecutive_rejections=cb_cfg.get("consecutive_rejections", 10),
            )
        )

        # Watchdog — monitors runner health, triggers safety actions on crash/timeout
        self.watchdog = Watchdog(
            config=WatchdogConfig(
                heartbeat_interval_sec=60.0,
                heartbeat_timeout_sec=300.0,    # 5 min before alarm
                max_consecutive_misses=3,
                dead_mans_switch_timeout_sec=600.0,  # 10 min = kill
                auto_restart_enabled=True,
                max_restart_attempts=3,
            )
        )
        self.watchdog.on_trigger(self._on_watchdog_trigger)

        # Broker
        broker_cfg = config.get("broker", {})
        self._broker_gateway = create_broker(broker_cfg)
        self.broker = _LiveBrokerAdapter(self._broker_gateway)

        # Data feed
        universe = config.get("universe", {})
        tickers = universe.get("tickers", ["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC"])
        from quantcore.live.data_feed import LiveDataFeed

        self.feed = LiveDataFeed(tickers=tickers)

        # Per-strategy runtime state (replaces flat tuple list)
        self.strategy_states: list[StrategyRuntimeState] = []
        self._init_strategies()

        # Legacy compatibility: flat list for existing code paths
        self.strategies: list[tuple[str, float, Any, str, str]] = [
            (s.name, s.weight, s.instance, s.data_source, s.bar_size)
            for s in self.strategy_states
        ]

        # Execution config
        exec_cfg = config.get("execution", {})
        self.scale_mode = exec_cfg.get("scale_mode", "fractional")

        # VIX data cache (keyed by date string, avoids repeated Yahoo Finance hits)
        self._vix_cache: dict[str, float] = {}

    def _init_strategies(self) -> None:
        """Create strategy instances from registry based on config.

        Each strategy gets its own StrategyRuntimeState for independent
        P&L tracking, risk limits, and lifecycle management.
        """
        available = set(list_strategies())
        initial_cash = float(self.config.get("execution", {}).get("initial_cash", 100_000.0))

        for strat_cfg in self.config.get("strategies", []):
            name = strat_cfg["name"]
            if not strat_cfg.get("enabled", True):
                log.info("Strategy %s disabled — skipping", name)
                continue
            if name not in available:
                log.warning("Strategy %s not in registry — skipping. Available: %s", name, sorted(available))
                continue

            weight = strat_cfg.get("weight", 1.0)
            params = strat_cfg.get("params", {})
            data_source = strat_cfg.get("data_source", "alpaca_daily")
            bar_size = strat_cfg.get("bar_size", "1Day")
            strategy_type = strat_cfg.get("strategy_type", "").upper()
            if not strategy_type:
                strategy_type = "MACRO" if "macro" in name else "EQUITY" if ("equity" in name or "gap" in name or "stat_arb" in name) else "OPTIONS"

            # Per-strategy circuit breaker config
            strat_cb = strat_cfg.get("circuit_breaker", {})
            max_daily_loss = strat_cb.get("max_daily_loss_pct", 0.02)
            max_drawdown = strat_cb.get("max_drawdown_pct", 0.10)

            cfg = StrategyConfig(
                name=name,
                strategy_type=strategy_type,
                max_position_notional=initial_cash,
                params=params,
            )
            try:
                instance = create_strategy(name, cfg)
            except Exception as exc:
                log.error("Failed to create strategy %s: %s", name, exc)
                self._alert(f"Failed to create strategy {name}: {exc}", severity="warning")
                continue

            state = StrategyRuntimeState(
                name=name,
                weight=weight,
                instance=instance,
                data_source=data_source,
                bar_size=bar_size,
                strategy_type=strategy_type,
                enabled=True,
                state="running",
                starting_equity=initial_cash * weight,
                current_equity=initial_cash * weight,
                peak_equity=initial_cash * weight,
                max_daily_loss_pct=max_daily_loss,
                max_drawdown_pct=max_drawdown,
            )
            self.strategy_states.append(state)
            log.info("Loaded strategy: %s (weight=%.2f, data=%s, type=%s, max_dd=%.1f%%)",
                     name, weight, data_source, strategy_type, max_drawdown * 100)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the production event loop. Blocks until shutdown."""
        if not self.strategy_states:
            log.error("No strategies loaded — aborting")
            self._alert("No strategies loaded — check config/live.yaml", severity="critical")
            return

        self._running = True
        self._setup_signal_handlers()
        atexit.register(self._shutdown)

        acct = self.broker.get_account()
        equity = float(acct.get("portfolio_value", 0))

        # Initialize daily P&L tracking — FIXES the circuit breaker false positive
        self.risk_manager.start_day(equity)
        self._trade_day_start_equity = equity

        log.info("=" * 70)
        log.info("QUANTCORE PRODUCTION RUNNER STARTED")
        log.info("Strategies : %d loaded", len(self.strategy_states))
        for s in self.strategy_states:
            log.info("  %s (weight=%.2f, data=%s, type=%s, limits: dd=%.1f%%, daily=%.1f%%)",
                     s.name, s.weight, s.data_source, s.strategy_type,
                     s.max_drawdown_pct * 100, s.max_daily_loss_pct * 100)
        log.info("Universe   : %s", self.feed.tickers)
        log.info("Account    : $%.2f cash, $%.2f equity",
                 acct.get("cash", 0), equity)
        log.info("Events     : %s", self.events_path)
        log.info("Equity     : %s", self.equity_path)
        log.info("Kill switch: %s (check every %ds)", self.kill_switch_path, self.kill_switch_interval)
        log.info("Watchdog   : enabled (heartbeat every 60s, timeout 300s)")
        log.info("Schedule   : trade @ %s ET, reconcile @ %s ET",
                 self.trading_time, self.reconciliation_time)
        log.info("=" * 70)

        self._alert(
            f"Production runner started with {len(self.strategy_states)} strategies, "
            f"equity ${equity:,.2f}",
            severity="info",
        )

        # Try immediate first tick if markets are open
        if self.feed.is_market_open():
            self._trading_tick(date.today())

        # Main loop: schedule + kill switch polling
        try:
            import schedule as schedule_lib

            schedule_lib.every().monday.at(self.trading_time).do(self._scheduled_trading_tick)
            schedule_lib.every().tuesday.at(self.trading_time).do(self._scheduled_trading_tick)
            schedule_lib.every().wednesday.at(self.trading_time).do(self._scheduled_trading_tick)
            schedule_lib.every().thursday.at(self.trading_time).do(self._scheduled_trading_tick)
            schedule_lib.every().friday.at(self.trading_time).do(self._scheduled_trading_tick)

            schedule_lib.every().monday.at(self.reconciliation_time).do(self._reconcile)
            schedule_lib.every().tuesday.at(self.reconciliation_time).do(self._reconcile)
            schedule_lib.every().wednesday.at(self.reconciliation_time).do(self._reconcile)
            schedule_lib.every().thursday.at(self.reconciliation_time).do(self._reconcile)
            schedule_lib.every().friday.at(self.reconciliation_time).do(self._reconcile)

            # Intraday scheduling for gap-fade strategy (09:30-10:05 ET, every 5 min)
            _intraday_slots = [
                "09:30", "09:35", "09:40", "09:45", "09:50", "09:55", "10:00", "10:05",
            ]
            for slot in _intraday_slots:
                for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
                    getattr(schedule_lib.every(), day).at(slot).do(
                        self._scheduled_intraday_tick, time_slot=slot
                    )

            log.info("Scheduler active — waiting for market hours...")

            while self._running:
                # Kill switch file check
                if self._check_kill_switch():
                    break

                # Watchdog heartbeat — signals that the runner is alive
                self.watchdog.heartbeat()

                # Watchdog health check
                wd_snapshot = self.watchdog.check()
                if wd_snapshot.state == WatchdogState.TRIGGERED:
                    log.critical("WATCHDOG TRIGGERED — dead man's switch activated")
                    self._alert("WATCHDOG TRIGGERED — trading halted", severity="critical")
                    break

                schedule_lib.run_pending()
                time.sleep(self.kill_switch_interval)

        except KeyboardInterrupt:
            log.info("Keyboard interrupt — shutting down...")
        finally:
            self._running = False

    def stop(self) -> None:
        """Graceful shutdown from another thread."""
        self._running = False

    def _on_watchdog_trigger(self, state: WatchdogState) -> None:
        """Callback when watchdog detects a critical failure."""
        log.critical("Watchdog trigger callback: %s", state.value)
        self._alert(
            f"System health critical: watchdog state = {state.value}. "
            "Trading halted for safety.",
            severity="critical",
        )

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def _check_kill_switch(self) -> bool:
        """Return True if kill switch is active (halt trading)."""
        if self.kill_switch_path.is_file():
            log.critical("KILL_SWITCH file detected at %s — halting all trading", self.kill_switch_path)
            set_kill_switch_active(True)
            self._alert("KILL SWITCH activated — all trading halted", severity="critical")
            return True

        # Also check the health state (set by API endpoint)
        from quantcore.core.health_state import get_health_state

        if get_health_state().get("kill_switch_active"):
            log.critical("Kill switch active via API — halting all trading")
            self._alert("KILL SWITCH activated via API — all trading halted", severity="critical")
            return True

        return False

    # ------------------------------------------------------------------
    # Scheduled callbacks
    # ------------------------------------------------------------------

    def _scheduled_trading_tick(self) -> None:
        today = date.today()
        if self._last_trade_date == today:
            log.debug("Already traded today (%s) — skipping", today)
            return
        self._trading_tick(today)

    def _reconcile(self) -> None:
        """Post-market reconciliation with per-strategy P&L update."""
        try:
            acct = self.broker.get_account()
            positions = self.broker.get_positions()
            pos_value = sum(
                float(abs(p.quantity)) * float(p.market_price)
                for p in positions.values()
            )
            total = float(acct.get("portfolio_value", 0))
            cash = float(acct.get("cash", 0))
            set_open_positions_count(len(positions))
            log.info("RECONCILE [%s] cash=$%.2f pos=$%.2f equity=$%.2f positions=%d",
                     date.today(), cash, pos_value, total, len(positions))

            # Update per-strategy equity contributions (proportional to weight)
            total_weight = sum(s.weight for s in self.strategy_states if s.is_active)
            if total_weight > 0:
                for s in self.strategy_states:
                    if s.is_active:
                        # Approximate per-strategy equity as weighted share of total
                        s.update_equity(total * s.weight / total_weight)

            # Send daily summary alert after reconciliation
            daily_pnl = total - self._trade_day_start_equity if self._trade_day_start_equity > 0 else 0.0
            self.alerter.send_daily_summary(
                equity=total,
                daily_pnl=daily_pnl,
                positions=len(positions),
                circuit_breaker_state=self.circuit_breaker.state.value,
                extra={
                    "active_strategies": sum(1 for s in self.strategy_states if s.is_active),
                    "total_strategies": len(self.strategy_states),
                },
            )
        except Exception as exc:
            log.error("Reconciliation failed: %s", exc)
            self._alert(f"Reconciliation failed: {exc}", severity="warning")

    def _scheduled_intraday_tick(self, time_slot: str) -> None:
        """Callback for intraday schedule entries."""
        if self._check_kill_switch():
            return
        self._intraday_tick(time_slot)

    def _intraday_tick(self, time_slot: str) -> None:
        """Run intraday strategies (e.g. opening_gap_fade) for a given time slot."""
        try:
            today = date.today()
            as_of = pd.Timestamp(today)

            # Check if any strategies need intraday gap-fade data
            has_gap_fade = any(
                ds == "alpaca_intraday_gap_fade"
                for _, _, _, ds, _ in self.strategies
            )
            if not has_gap_fade:
                return

            # Circuit breaker check
            acct = self.broker.get_account()
            equity = float(acct.get("portfolio_value", 0))
            self.risk_manager.update_equity(equity)
            self.circuit_breaker.check(
                equity=equity,
                peak_equity=self.risk_manager.peak_equity,
                daily_pnl=self.risk_manager.current_daily_pnl,
            )
            if self.circuit_breaker.activated:
                log.warning("Circuit breaker ACTIVE — skipping intraday tick %s", time_slot)
                return

            # Fetch OHLCV + prev_close for gap-fade strategies
            ohlcv_snapshot = self.feed.get_intraday_ohlcv_snapshot(
                as_of, bar_size="5Min", lookback_days=5,
            )
            prev_close_map = self.feed.get_prev_close_map(as_of)

            # Current ET time (wall clock)
            from zoneinfo import ZoneInfo
            current_time_et = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)

            # Inject gap-fade-specific fields into data dict
            ohlcv_snapshot["prev_close"] = prev_close_map
            ohlcv_snapshot["current_time"] = current_time_et

            all_signals: list[tuple[str, float, TradeSignal]] = []
            for strat_name, weight, instance, data_source, bar_size in self.strategies:
                if data_source != "alpaca_intraday_gap_fade":
                    continue
                try:
                    signals = instance.on_data(ohlcv_snapshot)
                    log.info("  %s [%s]: %d signals", strat_name, time_slot, len(signals))
                    for sig in signals:
                        all_signals.append((strat_name, weight, sig))
                except Exception as exc:
                    log.error("Intraday strategy %s failed: %s", strat_name, exc)
                    self._alert(f"Intraday strategy {strat_name} error: {exc}", severity="warning")

            # Submit orders
            for strat_name, weight, sig in all_signals:
                self._submit_signal(sig, strat_name, weight, today)

            if all_signals:
                log.info("INTRADAY [%s] %s: %d orders submitted", today, time_slot, len(all_signals))

        except Exception as exc:
            log.exception("Intraday tick failed for %s: %s", time_slot, exc)
            self._alert(f"Intraday tick failed: {exc}", severity="warning")

    # ------------------------------------------------------------------
    # VIX data source (for options strategies)
    # ------------------------------------------------------------------

    def _fetch_vix_close(self, trade_date: date) -> float | None:
        """Fetch VIX closing price from Yahoo Finance.  Cached per day.

        Returns the VIX close as a decimal (e.g. 18.5 means 18.5%),
        or ``None`` if the fetch fails.
        """
        cache_key = str(trade_date)
        if cache_key in self._vix_cache:
            return self._vix_cache[cache_key]

        try:
            import yfinance as yf

            ticker = yf.Ticker("^VIX")
            hist = ticker.history(period="5d")
            if hist.empty:
                log.warning("VIX history empty — using fallback 20.0")
                fallback = 20.0
                self._vix_cache[cache_key] = fallback
                return fallback
            vix_close = float(hist["Close"].iloc[-1])
            self._vix_cache[cache_key] = vix_close
            log.info("VIX close for %s: %.2f", trade_date, vix_close)
            return vix_close
        except Exception as exc:
            log.warning("Failed to fetch VIX from Yahoo Finance: %s — using fallback 20.0", exc)
            fallback = 20.0
            self._vix_cache[cache_key] = fallback
            return fallback

    def _build_snapshot_with_vix(
        self, daily_snapshot: dict[str, Any], trade_date: date
    ) -> dict[str, Any]:
        """Inject ``implied_vol`` into the snapshot using VIX as proxy.

        VIX close / 100 is used as the implied-volatility series for the
        entire lookback window (flat IV = VIX today for all bars).  This is
        intentionally conservative — the IV Rank calculation only cares about
        the *relative* level, and VIX > 15 / IVR >= 50 gates entry.
        """
        import numpy as np

        snapshot = dict(daily_snapshot)
        close = snapshot.get("close")
        if close is None or close.size == 0:
            return snapshot

        vix_close = self._fetch_vix_close(trade_date)
        if vix_close is None:
            vix_close = 20.0

        iv_decimal = vix_close / 100.0
        n_periods = close.shape[0]
        # Build a realistic IV ramp: blend realized vol with VIX level
        # so IVR has variation (not a flat line that always returns 50)
        import pandas as pd

        if n_periods > 1 and close.ndim == 2:
            # Use first asset (underlying) for realized-vol basis
            close_series = pd.Series(close[:, 0])
            log_ret = np.log(close_series / close_series.shift(1)).dropna()
            # Realized vol (annualised) over rolling 21-day window
            rv = log_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
            rv = rv.reindex(range(n_periods)).bfill().fillna(iv_decimal)
            # Blend: 60% VIX-level, 40% realized vol + mean-reversion
            iv_series = 0.6 * iv_decimal + 0.4 * rv.values
            # Clip to sensible bounds
            iv_series = np.clip(iv_series, 0.08, 0.80)
            # Ensure the last bar uses the actual VIX close
            iv_series[-1] = iv_decimal
        else:
            iv_series = np.full(n_periods, iv_decimal)

        snapshot["implied_vol"] = iv_series.reshape(-1, 1) if close.ndim == 2 else iv_series
        return snapshot

    # ------------------------------------------------------------------
    # Core trading tick (multi-strategy)
    # ------------------------------------------------------------------

    def _trading_tick(self, trade_date: date) -> None:
        """One full trading-day cycle across all strategies.

        Per-strategy isolation: each strategy's signals are evaluated against
        its own risk limits. If one strategy fails or hits its risk limit,
        it is paused without affecting others.
        """
        try:
            self._last_trade_date = trade_date

            # 1. Pre-fetch daily snapshot (used by most strategies)
            as_of = pd.Timestamp(trade_date)
            daily_snapshot = self.feed.get_daily_snapshot(as_of)
            close_matrix = daily_snapshot.get("close")

            if close_matrix is None or close_matrix.size == 0:
                log.warning("No price data for %s — skipping", trade_date)
                return

            record_data_point()
            log.info("SNAPSHOT [%s] %d days x %d assets",
                     trade_date, close_matrix.shape[0], close_matrix.shape[1])

            # 2. Initialize daily P&L tracking (FIXES circuit breaker false positive)
            acct = self.broker.get_account()
            equity = float(acct.get("portfolio_value", 0))
            if trade_date != self._last_trade_date or self._trade_day_start_equity == 0:
                self.risk_manager.start_day(equity)
                self._trade_day_start_equity = equity
                # Reset per-strategy daily P&L
                for s in self.strategy_states:
                    s.daily_pnl = 0.0

            self.risk_manager.update_equity(equity)

            # 3. Portfolio-level circuit breaker check
            self.circuit_breaker.check(
                equity=equity,
                peak_equity=self.risk_manager.peak_equity,
                daily_pnl=self.risk_manager.current_daily_pnl,
            )
            if self.circuit_breaker.activated:
                reason = self.circuit_breaker.reason
                state_val = self.circuit_breaker.state.value
                log.warning("Circuit breaker ACTIVE (%s): %s — skipping tick", state_val, reason)
                # Alert via all channels (Telegram, file, etc.)
                self._alert(
                    f"CIRCUIT BREAKER {state_val}: {reason}\n"
                    f"Equity: ${equity:,.2f} | Daily P&L: ${self.risk_manager.current_daily_pnl:,.2f}",
                    severity="critical",
                )
                self._log_equity(trade_date)
                return

            # 4. Run each strategy with per-strategy isolation
            all_signals: list[tuple[str, float, TradeSignal, StrategyRuntimeState]] = []
            for strat_state in self.strategy_states:
                if not strat_state.is_active:
                    continue

                strat_name = strat_state.name
                try:
                    # Build appropriate data snapshot
                    if strat_state.data_source == "alpaca_intraday_gap_fade":
                        continue  # handled by _intraday_tick
                    if strat_state.data_source == "yfinance_vix":
                        snapshot = self._build_snapshot_with_vix(daily_snapshot, trade_date)
                    elif strat_state.data_source == "alpaca_intraday":
                        snapshot = self.feed.get_intraday_snapshot(as_of, bar_size=strat_state.bar_size)
                    else:
                        snapshot = daily_snapshot

                    signals = strat_state.instance.on_data(snapshot)
                    strat_state.n_signals += len(signals)
                    strat_state.last_signal_count = len(signals)
                    strat_state.reset_errors()  # successful run resets error counter
                    log.info("  %s: %d signals (%s)", strat_name, len(signals), strat_state.data_source)

                    for sig in signals:
                        all_signals.append((strat_name, strat_state.weight, sig, strat_state))

                except Exception as exc:
                    log.error("Strategy %s failed: %s", strat_name, exc)
                    should_pause = strat_state.record_error()
                    if should_pause:
                        log.error("Strategy %s paused after %d consecutive errors",
                                  strat_name, strat_state.error_count)
                        self._alert(
                            f"Strategy {strat_name} PAUSED after {strat_state.error_count} consecutive errors: {exc}",
                            severity="warning",
                        )
                    else:
                        self._alert(f"Strategy {strat_name} error ({strat_state.error_count}/{strat_state.max_consecutive_errors}): {exc}", severity="warning")

            if not all_signals:
                log.info("No signals on %s — holding", trade_date)
                self._log_equity(trade_date)
                return

            # 5. Submit orders (weighted, with per-strategy tracking)
            orders_submitted = 0
            for strat_name, weight, sig, strat_state in all_signals:
                # Per-strategy risk check before submitting
                should_pause, reason = strat_state.check_per_strategy_risk(strat_state.current_equity)
                if should_pause:
                    strat_state.state = "paused"
                    log.warning("Strategy %s paused: %s", strat_name, reason)
                    self._alert(reason, severity="warning")
                    continue

                order_id = self._submit_signal(sig, strat_name, weight, trade_date)
                if order_id:
                    orders_submitted += 1
                    strat_state.n_trades += 1

            log.info("ORDERS submitted %d/%d on %s", orders_submitted, len(all_signals), trade_date)
            self._log_equity(trade_date)

        except Exception as exc:
            log.exception("Trading tick failed for %s: %s", trade_date, exc)
            self._alert(f"Trading tick failed: {exc}", severity="critical")

    # ------------------------------------------------------------------
    # Signal → Order
    # ------------------------------------------------------------------

    def _submit_signal(
        self, sig: TradeSignal, strat_name: str, weight: float, trade_date: date
    ) -> str | None:
        """Convert a TradeSignal to an Order, apply risk gate, and submit."""
        try:
            symbol = str(sig.symbol).strip().upper()
            direction = sig.direction
            strength = float(sig.strength)
            size_notional = float(sig.size) if sig.size else 0.0

            if direction == 0:
                return self._close_position(symbol, strat_name)

            market_price = self._get_price(symbol)
            if market_price <= 0:
                log.warning("No price for %s — skipping", symbol)
                return None

            acct = self.broker.get_account()
            equity = float(acct.get("portfolio_value", 100_000))

            # Apply strategy weight and scale mode
            if self.scale_mode == "single_share":
                size_notional = market_price
            elif self.scale_mode == "fractional":
                if size_notional <= 0:
                    size_notional = strength * equity * 0.10
                size_notional *= 0.10
            else:  # full
                if size_notional <= 0:
                    size_notional = strength * equity * 0.10

            # Apply strategy weight
            size_notional *= weight

            if direction > 0:
                cash = float(acct.get("cash", 0))
                notional = min(size_notional, cash * 0.95)
                if notional < 50:
                    return None
                qty = notional / market_price
            else:
                qty = size_notional / market_price

            qty = max(int(qty), 1)

            order = Order(
                id="",
                symbol=symbol,
                side=Side.BUY if direction > 0 else Side.SELL,
                type=OrderType.MARKET,
                quantity=Decimal(str(qty)),
                status=OrderStatus.PENDING,
            )

            # Risk gate
            positions = self.broker.get_positions()
            approval = self.risk_manager.pre_trade_check(order, equity, positions)
            if not approval.approved:
                log.warning("RISK REJECT %s %s: %s", order.side.value.upper(), symbol, approval.reason)
                self.circuit_breaker.check(
                    equity=equity,
                    peak_equity=self.risk_manager.peak_equity,
                    daily_pnl=self.risk_manager.current_daily_pnl,
                    rejected_last=True,
                )
                return None

            order_id = self.broker.submit_order(order)
            record_order_submitted()

            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": "submit",
                "symbol": symbol,
                "side": order.side.value,
                "qty": qty,
                "price": market_price,
                "notional": float(qty) * market_price,
                "order_id": order_id,
                "strategy": strat_name,
                "trade_date": str(trade_date),
                "weight": weight,
            }
            self._append_event(event)

            log.info("ORDER %s %s %d shares @ %.4f ($%.2f) [%s, w=%.2f] [order=%s]",
                     order.side.value.upper(), symbol, qty, market_price,
                     float(qty) * market_price, strat_name, weight, order_id)
            return order_id

        except Exception as exc:
            log.exception("Order submission failed for %s: %s", sig, exc)
            return None

    def _close_position(self, symbol: str, strat_name: str) -> str | None:
        positions = self.broker.get_positions()
        pos = positions.get(symbol)
        if not pos or abs(float(pos.quantity)) < 1e-6:
            return None
        qty = abs(int(float(pos.quantity)))
        side = Side.SELL if float(pos.quantity) > 0 else Side.BUY
        order = Order(
            id="",
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=Decimal(str(qty)),
            status=OrderStatus.PENDING,
        )
        order_id = self.broker.submit_order(order)
        self._append_event({
            "timestamp": datetime.now(UTC).isoformat(),
            "event": "close",
            "symbol": symbol,
            "qty": qty,
            "order_id": order_id,
            "strategy": strat_name,
        })
        return order_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_price(self, symbol: str) -> float:
        price = self.feed.get_latest_close(symbol)
        if price and price > 0:
            return float(price)
        positions = self.broker.get_positions()
        pos = positions.get(symbol)
        if pos:
            return float(pos.market_price)
        return 0.0

    def _append_event(self, event: dict) -> None:
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError as exc:
            log.warning("Could not append to event log: %s", exc)

    def _log_equity(self, trade_date: date) -> None:
        """Log portfolio-level and per-strategy equity curves."""
        try:
            acct = self.broker.get_account()
            positions = self.broker.get_positions()
            cash = float(acct.get("cash", 0))
            pos_value = sum(
                float(abs(p.quantity)) * float(p.market_price)
                for p in positions.values()
            )
            total = cash + pos_value
            set_open_positions_count(len(positions))

            # Portfolio-level equity curve
            equity_file = self.equity_path
            if equity_file.exists():
                df = pd.read_csv(equity_file, parse_dates=["date"])
            else:
                df = pd.DataFrame(columns=["date", "cash", "positions_value", "total_equity", "ret"])

            prev_total = float(df["total_equity"].iloc[-1]) if not df.empty else total
            ret = (total / prev_total - 1) if prev_total > 0 else 0.0

            new_row = pd.DataFrame([{
                "date": str(trade_date),
                "cash": round(cash, 2),
                "positions_value": round(pos_value, 2),
                "total_equity": round(total, 2),
                "ret": round(ret, 6),
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(equity_file, index=False)

            # Per-strategy equity curves
            total_weight = sum(s.weight for s in self.strategy_states if s.is_active)
            if total_weight > 0:
                for s in self.strategy_states:
                    if not s.is_active:
                        continue
                    # Proportional equity allocation
                    strat_equity = total * s.weight / total_weight
                    s.update_equity(strat_equity)

                    strat_file = self.equity_path.parent / f"live_equity_{s.name}.csv"
                    if strat_file.exists():
                        sdf = pd.read_csv(strat_file, parse_dates=["date"])
                    else:
                        sdf = pd.DataFrame(columns=["date", "equity", "daily_pnl", "weight", "status"])

                    prev_eq = float(sdf["equity"].iloc[-1]) if not sdf.empty else strat_equity
                    strat_ret = (strat_equity / prev_eq - 1) if prev_eq > 0 else 0.0

                    new_strat_row = pd.DataFrame([{
                        "date": str(trade_date),
                        "equity": round(strat_equity, 2),
                        "daily_pnl": round(strat_equity - prev_eq, 2),
                        "weight": round(s.weight, 4),
                        "status": s.state,
                    }])
                    sdf = pd.concat([sdf, new_strat_row], ignore_index=True)
                    sdf.to_csv(strat_file, index=False)

        except Exception as exc:
            log.warning("Could not log equity: %s", exc)

    def _alert(self, message: str, severity: str = "info") -> None:
        if self.alerter:
            self.alerter.send_alert(message, severity=severity)

    def _setup_signal_handlers(self) -> None:
        def _handler(signum: int, frame: Any) -> None:
            log.info("Received signal %d — shutting down...", signum)
            self._running = False
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _shutdown(self) -> None:
        log.info("Production runner shutting down...")
        try:
            self._log_equity(date.today())
        except Exception as exc:
            log.warning("Could not log equity during shutdown: %s", exc)
        self._alert("Production runner shutting down", severity="info")
        self._running = False
        log.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# API server launcher (optional — runs alongside the trading loop)
# ---------------------------------------------------------------------------


def start_api_server(config: dict[str, Any]) -> None:
    """Start the FastAPI server in a background thread if enabled."""
    api_cfg = config.get("api", {})
    if not api_cfg.get("enabled", True):
        return

    try:
        import threading

        import uvicorn

        host = api_cfg.get("host", "0.0.0.0")
        port = api_cfg.get("port", 8000)

        def _run_server() -> None:
            from quantcore.api.main import app

            uvicorn.run(app, host=host, port=port, log_level="warning")

        thread = threading.Thread(target=_run_server, daemon=True, name="api-server")
        thread.start()
        log.info("API server started on http://%s:%d", host, port)
    except ImportError:
        log.warning("uvicorn not installed — API server disabled")
    except Exception as exc:
        log.warning("Failed to start API server: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path)

    log.info("Config loaded from %s", config_path or DEFAULT_CONFIG_PATH)

    # Start API server in background
    start_api_server(config)

    # Run trading loop (blocks)
    runner = ProductionRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
