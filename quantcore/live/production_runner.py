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
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import time
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
import quantcore.strategies.macro.factor_residual_strict
import quantcore.strategies.macro.risk_parity
import quantcore.strategies.macro.state_selective_momentum
import quantcore.strategies.options.delta_neutral
import quantcore.strategies.options.earnings_vol
import quantcore.strategies.options.iron_condor
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
from quantcore.monitoring.alerting import Alerter, AlertingConfig
from quantcore.strategies.base import StrategyConfig, TradeSignal
from quantcore.strategies.registry import create as create_strategy, list_strategies

log = logging.getLogger(__name__)

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

    Parameters
    ----------
    config : dict
        Parsed ``config/live.yaml``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._running = False
        self._last_trade_date: date | None = None

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

        # Alerting
        alert_cfg = config.get("alerting", {})
        self.alerter = Alerter.from_config(alert_cfg) if alert_cfg.get("enabled", True) else None

        # Risk
        risk_cfg = config.get("risk", {})
        cb_cfg = risk_cfg.get("circuit_breaker", {})
        self.risk_manager = RiskManagerLive(
            config=RiskLimitsConfig(
                max_position_per_ticker_pct=risk_cfg.get("max_position_per_ticker_pct", 0.05),
                max_gross_exposure_pct=risk_cfg.get("max_gross_exposure_pct", 2.0),
                max_daily_loss_pct=risk_cfg.get("max_daily_loss_pct", 0.02),
                max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.10),
                max_single_order_pct=risk_cfg.get("max_single_order_pct", 0.05),
                max_leverage=risk_cfg.get("max_leverage", 2.0),
            )
        )
        self.circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                daily_loss_pct=cb_cfg.get("daily_loss_pct", 0.02),
                critical_drawdown_pct=cb_cfg.get("critical_drawdown_pct", 0.10),
                consecutive_rejections=cb_cfg.get("consecutive_rejections", 5),
            )
        )

        # Broker
        broker_cfg = config.get("broker", {})
        self._broker_gateway = create_broker(broker_cfg)
        self.broker = _LiveBrokerAdapter(self._broker_gateway)

        # Data feed
        universe = config.get("universe", {})
        tickers = universe.get("tickers", ["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC"])
        from quantcore.live.data_feed import LiveDataFeed

        self.feed = LiveDataFeed(tickers=tickers)

        # Strategies
        self.strategies: list[tuple[str, float, Any, str, str]] = []  # (name, weight, instance, data_source, bar_size)
        self._init_strategies()

        # Execution config
        exec_cfg = config.get("execution", {})
        self.scale_mode = exec_cfg.get("scale_mode", "fractional")

    def _init_strategies(self) -> None:
        """Create strategy instances from registry based on config."""
        available = set(list_strategies())
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
            strategy_type = "MACRO" if "macro" in name else "EQUITY" if "equity" in name else "OPTIONS"
            cfg = StrategyConfig(
                name=name,
                strategy_type=strategy_type,
                max_position_notional=float(self.config.get("execution", {}).get("initial_cash", 100_000.0)),
                params=params,
            )
            instance = create_strategy(name, cfg)
            data_source = strat_cfg.get("data_source", "alpaca_daily")
            bar_size = strat_cfg.get("bar_size", "1Day")
            self.strategies.append((name, weight, instance, data_source, bar_size))
            log.info("Loaded strategy: %s (weight=%.2f, data=%s, bar=%s)", name, weight, data_source, bar_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the production event loop. Blocks until shutdown."""
        if not self.strategies:
            log.error("No strategies loaded — aborting")
            self._alert("No strategies loaded — check config/live.yaml", severity="critical")
            return

        self._running = True
        self._setup_signal_handlers()
        atexit.register(self._shutdown)

        acct = self.broker.get_account()
        log.info("=" * 70)
        log.info("QUANTCORE PRODUCTION RUNNER STARTED")
        log.info("Strategies : %s", [(s[0], s[3]) for s in self.strategies])
        log.info("Universe   : %s", self.feed.tickers)
        log.info("Account    : $%.2f cash, $%.2f equity",
                 acct.get("cash", 0), acct.get("portfolio_value", 0))
        log.info("Events     : %s", self.events_path)
        log.info("Equity     : %s", self.equity_path)
        log.info("Kill switch: %s (check every %ds)", self.kill_switch_path, self.kill_switch_interval)
        log.info("Schedule   : trade @ %s ET, reconcile @ %s ET",
                 self.trading_time, self.reconciliation_time)
        log.info("=" * 70)

        self._alert("Production runner started", severity="info")

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

            log.info("Scheduler active — waiting for market hours...")

            while self._running:
                # Kill switch file check
                if self._check_kill_switch():
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
        """Post-market reconciliation."""
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
        except Exception as exc:
            log.error("Reconciliation failed: %s", exc)
            self._alert(f"Reconciliation failed: {exc}", severity="warning")

    # ------------------------------------------------------------------
    # Core trading tick (multi-strategy)
    # ------------------------------------------------------------------

    def _trading_tick(self, trade_date: date) -> None:
        """One full trading-day cycle across all strategies."""
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

            # 2. Circuit breaker check
            acct = self.broker.get_account()
            equity = float(acct.get("portfolio_value", 0))
            self.risk_manager.update_equity(equity)
            self.circuit_breaker.check(
                equity=equity,
                peak_equity=self.risk_manager.peak_equity,
                daily_pnl=self.risk_manager.current_daily_pnl,
            )
            if self.circuit_breaker.activated:
                log.warning("Circuit breaker ACTIVE (%s): %s — skipping tick",
                            self.circuit_breaker.state.value, self.circuit_breaker.reason)
                self._alert(
                    f"Circuit breaker: {self.circuit_breaker.state.value} — {self.circuit_breaker.reason}",
                    severity="critical",
                )
                self._log_equity(trade_date)
                return

            # 3. Run each strategy with the appropriate data source
            all_signals: list[tuple[str, float, TradeSignal]] = []
            for strat_name, weight, instance, data_source, bar_size in self.strategies:
                try:
                    if data_source == "alpaca_intraday":
                        snapshot = self.feed.get_intraday_snapshot(as_of, bar_size=bar_size)
                    else:
                        snapshot = daily_snapshot
                    signals = instance.on_data(snapshot)
                    log.info("  %s: %d signals (%s)", strat_name, len(signals), data_source)
                    for sig in signals:
                        all_signals.append((strat_name, weight, sig))
                except Exception as exc:
                    log.error("Strategy %s failed: %s", strat_name, exc)
                    self._alert(f"Strategy {strat_name} error: {exc}", severity="warning")

            if not all_signals:
                log.info("No signals on %s — holding", trade_date)
                self._log_equity(trade_date)
                return

            # 4. Submit orders (weighted)
            orders_submitted = 0
            for strat_name, weight, sig in all_signals:
                order_id = self._submit_signal(sig, strat_name, weight, trade_date)
                if order_id:
                    orders_submitted += 1

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
        except Exception:
            pass
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
