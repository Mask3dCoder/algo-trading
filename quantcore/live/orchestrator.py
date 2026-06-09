"""Paper trading orchestrator — daily loop for live validation.

One code path: the same ``strategy.on_data(data)`` that the walk-forward
backtest calls is called here with live data, producing the same signals.

Components wired on construction (all via constructor — no singletons):
    feed     : LiveDataFeed   — market OHLCV in on_data() format
    broker   : LiveExecutionGateway  — paper order execution + position tracking
    strategy : Any Strategy subclass registered via @register()

Daily loop (schedule library, cross-platform):
    1. At 16:15 ET Mon-Fri: fetch daily snapshot → on_data() → submit orders
    2. Reconciliation pass at 16:30 ET (sync positions from broker)
    3. Equity log written after each trading-day tick

Output files (relative to CWD / reports/):
    live_events.jsonl        — every order event as newline JSON
    live_equity.csv          — date, cash, positions_value, total_equity, ret

Usage::

    from quantcore.live.orchestrator import make_paper_trader

    trader = make_paper_trader(
        strategy_name="my_validated_strategy",
        tickers=["SPY", "QQQ"],
    )
    trader.run()   # blocks — Ctrl+C to exit
"""

from __future__ import annotations

import atexit
import json
import logging
import signal
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quantcore.core.types import Position

# schedule is a pure stdlib-compatible scheduling library (pip install schedule)
try:
    import schedule
except ImportError as exc:
    raise ImportError("schedule module required: pip install schedule") from exc

from quantcore.core.types import Order, Side
from quantcore.live.broker import LiveExecutionGateway
from quantcore.live.circuit_breaker import CircuitBreaker
from quantcore.live.data_feed import LiveDataFeed
from quantcore.live.risk_manager import RiskLimitsConfig, RiskManagerLive
from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import create as registry_create
from quantcore.strategies.registry import list_strategies

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
EVENTS_PATH  = REPORTS_DIR / "live_events.jsonl"
EQUITY_PATH  = REPORTS_DIR / "live_equity.csv"

KILL_SWITCH_PATH = Path("KILL_SWITCH")

# ---------------------------------------------------------------------------
# Broker adapter that LiveExecutionGateway conforms to
# LiveRunner.Broker protocol expects get_positions() -> dict[str, Position]
# Our LiveExecutionGateway.get_positions() returns exactly that.
# ---------------------------------------------------------------------------


class LiveBroker:
    """Thin adapter so LiveExecutionGateway satisfies live/runner.py Broker.

    LiveExecutionGateway already implements the required sync methods;
    this class is only here to document the Protocol conformance and
    provide a clean type anchor.
    """

    def __init__(self, gateway: LiveExecutionGateway) -> None:
        self._gw = gateway

    # Required by live.runner.Broker protocol
    def submit_order(self, order: Order) -> str:
        return self._gw.submit_order(order)

    def cancel_order(self, order_id: str) -> bool:
        return self._gw.cancel_order(order_id)

    def get_positions(self) -> dict[str, Position]:
        return self._gw.get_positions()

    def get_account(self) -> dict[str, float]:
        return self._gw.get_account()

    # Paper-specific extras
    def is_paper(self) -> bool:
        return self._gw.is_paper


class ScaleMode(str, Enum):
    """Graduated scale of live deployment."""

    SINGLE_SHARE = "single_share"   # 1 share per signal (minimal)
    FRACTIONAL = "fractional"       # 10% notional sizing
    FULL = "full"                   # full strategy notional


# ---------------------------------------------------------------------------
# PaperTrader
# ---------------------------------------------------------------------------

class PaperTrader:
    """Daily paper-trading loop with risk, circuit breaker, and kill switch.

    Parameters
    ----------
    feed     : LiveDataFeed instance with tickers pre-configured.
    broker   : LiveExecutionGateway instance.
    strategy : Strategy implementing on_data(data: dict) → list[TradeSignal].
    risk_manager : RiskManagerLive, optional
        Pre-trade risk gate.  If None, a default one is created.
    circuit_breaker : CircuitBreaker, optional
        Circuit breaker for graduated halt states.
    scale_mode : ScaleMode or str
        single_share | fractional | full
    equity_log_path : Path to equity CSV (created automatically).
    events_log_path : Path to JSONL event log (created automatically).
    """

    def __init__(
        self,
        feed: LiveDataFeed,
        broker: LiveExecutionGateway,
        strategy: Strategy,
        risk_manager: RiskManagerLive | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        scale_mode: ScaleMode | str = ScaleMode.FULL,
        equity_log_path: Path = EQUITY_PATH,
        events_log_path: Path = EVENTS_PATH,
    ) -> None:
        self.feed = feed
        self.broker = broker
        self.strategy = strategy
        self.events_path = str(events_log_path)
        self.equity_path = str(equity_log_path)

        # Risk / safety
        self.risk = risk_manager or RiskManagerLive()
        self.circuit = circuit_breaker or CircuitBreaker()
        self.scale_mode = ScaleMode(scale_mode) if isinstance(scale_mode, str) else scale_mode

        # Internal state
        self._running = False
        self._last_trade_date: date | None = None
        self._equity_df: pd.DataFrame = pd.DataFrame(
            columns=["date", "cash", "positions_value", "total_equity", "ret"]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the daily scheduling loop. Blocks the calling thread."""
        self._running = True
        self._setup_handlers()

        log.info("=" * 60)
        log.info("PAPER TRADING STARTED — no real money is at risk.")
        log.info("Strategy  : %s", self.strategy.__class__.__name__)
        log.info("Assets    : %s", self.feed.tickers)
        log.info("Account   : %s mode, initial cash $%.2f",
                 "PAPER" if self.broker.is_paper else "LIVE",
                 self.broker.get_account()["cash"])
        log.info("Events    : %s", self.events_path)
        log.info("Equity    : %s", self.equity_path)
        log.info("Schedule  : trading tick at 16:15 ET, reconcile at 16:30 ET")
        log.info("=" * 60)

        # Immediate first tick if markets are open today
        today = date.today()
        if self.feed.is_market_open():
            self._trading_tick(today)
        else:
            log.info("Markets closed today (%s) — skipping first tick", today)

        # Schedule daily jobs (runs at the scheduled time every weekday)
        schedule.every().monday.at("16:15").do(self._scheduled_trading_tick)
        schedule.every().tuesday.at("16:15").do(self._scheduled_trading_tick)
        schedule.every().wednesday.at("16:15").do(self._scheduled_trading_tick)
        schedule.every().thursday.at("16:15").do(self._scheduled_trading_tick)
        schedule.every().friday.at("16:15").do(self._scheduled_trading_tick)

        # 5-minute reconciliation pass after close
        schedule.every().monday.at("16:30").do(self._reconcile)
        schedule.every().tuesday.at("16:30").do(self._reconcile)
        schedule.every().wednesday.at("16:30").do(self._reconcile)
        schedule.every().thursday.at("16:30").do(self._reconcile)
        schedule.every().friday.at("16:30").do(self._reconcile)

        # Register cleanup
        atexit.register(self._shutdown)
        log.info("Scheduler active — press Ctrl+C to stop.")

        while self._running:
            schedule.run_pending()
            import time
            time.sleep(30)   # check every 30 s

    def stop(self) -> None:
        """Graceful shutdown from another thread."""
        self._running = False

    # ------------------------------------------------------------------
    # Scheduled callbacks
    # ------------------------------------------------------------------

    def _scheduled_trading_tick(self) -> None:
        today = date.today()
        if self._last_trade_date == today:
            log.debug("Already traded today (%s) — skipping duplicate tick", today)
            return
        self._trading_tick(today)

    def _reconcile(self) -> None:
        """Post-market reconciliation: sync broker positions to our state."""
        try:
            acct = self.broker.get_account()
            positions = self.broker.get_positions()
            pos_value = sum(
                float(abs(p.quantity)) * float(p.market_price)
                for p in positions.values()
            )
            total_equity = float(acct.get("portfolio_value", 0))
            cash = float(acct.get("cash", 0))
            log.info(
                "RECONCILE [%s] cash=$%.2f pos=$%.2f equity=$%.2f",
                date.today(), cash, pos_value, total_equity,
            )
        except Exception as exc:
            log.error("Reconciliation failed: %s", exc)

    # ------------------------------------------------------------------
    # Core daily tick
    # ------------------------------------------------------------------

    def _trading_tick(self, trade_date: date) -> None:
        """One full trading-day cycle: snapshot → signal → risk → submit → log."""
        try:
            self._last_trade_date = trade_date

            # 1. Fetch price data for today's snapshot (lookback 252 days)
            as_of = pd.Timestamp(trade_date)
            snapshot = self.feed.get_daily_snapshot(as_of)
            close_matrix = snapshot.get("close")

            if close_matrix is None or close_matrix.size == 0:
                log.warning("No price data for %s — skipping tick", trade_date)
                return

            log.info("SNAPSHOT [%s] %d days × %d assets",
                     trade_date, close_matrix.shape[0], close_matrix.shape[1])

            # 2. Kill switch check
            if KILL_SWITCH_PATH.exists():
                log.critical("KILL_SWITCH detected — halting trading immediately.")
                self._running = False
                return

            # 3. Circuit breaker check
            acct = self.broker.get_account()
            equity = float(acct.get("portfolio_value", 0))
            self.risk.update_equity(equity)
            self.circuit.check(
                equity=equity,
                peak_equity=self.risk.peak_equity,
                daily_pnl=self.risk.current_daily_pnl,
            )
            if self.circuit.activated:
                log.warning("Circuit breaker ACTIVE (%s): %s — skipping trading tick",
                            self.circuit.state.value, self.circuit.reason)
                self._log_equity(trade_date)
                return

            # 4. Generate signals via the SAME on_data() call used in backtest
            signals = self.strategy.on_data(snapshot)
            log.info("SIGNALS generated %d trade signals", len(signals))

            if not signals:
                log.info("No signals on %s — holding current positions", trade_date)
                self._log_equity(trade_date)
                return

            # 5. Map signals → orders and submit (with risk gate)
            orders_submitted = 0
            for sig in signals:
                order_id = self._submit_signal(sig, trade_date)
                if order_id:
                    orders_submitted += 1

            log.info("ORDERS submitted %d/%d on %s",
                     orders_submitted, len(signals), trade_date)

            # 6. Log equity state after trading
            self._log_equity(trade_date)

        except Exception as exc:
            log.exception("Trading tick failed for %s: %s", trade_date, exc)

    # ------------------------------------------------------------------
    # Signal → Order conversion
    # ------------------------------------------------------------------

    def _submit_signal(self, sig: TradeSignal, trade_date: date) -> str | None:
        """Convert a TradeSignal to an Order and submit via broker.

        Size logic:
          signal.size is the notional weight (fraction of portfolio).
          Clamp to available cash for long signals; for shorts, the
          broker handles margin.
        """
        try:
            symbol = str(sig.symbol).strip().upper()
            direction = sig.direction   # +1 long, -1 short, 0 flat
            strength  = float(sig.strength)
            size_notional = float(sig.size) if sig.size else 0.0

            if direction == 0:
                log.debug("Flat signal for %s — closing if open", symbol)
                return self._close_position(symbol)

            # Get current market price from our data feed
            market_price = self._get_price(symbol)
            if market_price <= 0:
                log.warning("No price for %s — skipping signal", symbol)
                return None

            # Broker account for sizing
            acct = self.broker.get_account()
            equity = float(acct.get("portfolio_value", 100_000))
            cash  = float(acct.get("cash", 0))

            # Scale mode adjustment
            if self.scale_mode == ScaleMode.SINGLE_SHARE:
                size_notional = market_price
            elif self.scale_mode == ScaleMode.FRACTIONAL:
                if size_notional <= 0:
                    size_notional = strength * equity * 0.10
                size_notional *= 0.10  # 10% of normal notional
            else:
                if size_notional <= 0:
                    size_notional = strength * equity * 0.10

            # Long signals: limited by available cash
            if direction > 0:
                max_notional = cash * 0.95  # 5% cash buffer
                notional = min(size_notional, max_notional)
                if notional < 50:
                    log.debug("Notional $%.2f < $50 minimum — skipping %s", notional, symbol)
                    return None
                qty = notional / market_price
            else:
                # Short signals: use full signal size (margin)
                notional = size_notional
                qty = notional / market_price

            qty = max(int(qty), 1)   # at least 1 share

            # Build order
            side = Side.BUY if direction > 0 else Side.SELL
            order = Order(
                id="",          # filled by broker
                symbol=symbol,
                side=side,
                type=__import__("quantcore.core.types", fromlist=["OrderType"]).OrderType.MARKET,
                quantity=Decimal(str(qty)),
                status=__import__("quantcore.core.types", fromlist=["OrderStatus"]).OrderStatus.PENDING,
            )

            # Risk pre-trade check
            positions = self.broker.get_positions()
            approval = self.risk.pre_trade_check(order, equity, positions, market_prices={symbol: market_price})
            if not approval.approved:
                log.warning("RISK REJECT order %s %s: %s", side.value.upper(), symbol, approval.reason)
                return None

            order_id = self.broker.submit_order(order)

            # Log event
            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "event":     "submit",
                "symbol":    symbol,
                "side":      side.value,
                "qty":       qty,
                "price":     market_price,
                "notional":  float(qty) * market_price,
                "order_id":  order_id,
                "strategy":  self.strategy.__class__.__name__,
                "trade_date": str(trade_date),
                "strength":  strength,
                "signal_size": size_notional,
            }
            self._append_event(event)

            log.info(
                "ORDER %s %s %d shares %s @ %.4f (notional $%.2f) [order=%s]",
                "BUY" if direction > 0 else "SELL", symbol, qty,
                trade_date, market_price, float(qty) * market_price, order_id,
            )
            return order_id

        except Exception as exc:
            log.exception("Order submission failed for signal %s: %s", sig, exc)
            return None

    def _close_position(self, symbol: str) -> str | None:
        """Market-close any open position in symbol. Returns order_id or None."""
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
            type=__import__("quantcore.core.types", fromlist=["OrderType"]).OrderType.MARKET,
            quantity=Decimal(str(qty)),
            status=__import__("quantcore.core.types", fromlist=["OrderStatus"]).OrderStatus.PENDING,
        )
        order_id = self.broker.submit_order(order)
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event":     "close",
            "symbol":    symbol,
            "qty":       qty,
            "order_id":  order_id,
            "strategy":  self.strategy.__class__.__name__,
        }
        self._append_event(event)
        return order_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_price(self, symbol: str) -> float:
        """Latest close price for symbol from data feed."""
        price = self.feed.get_latest_close(symbol)
        if price and price > 0:
            return price
        # Fallback to broker market price
        positions = self.broker.get_positions()
        pos = positions.get(symbol)
        if pos:
            return float(pos.market_price)
        return 0.0

    def _append_event(self, event: dict) -> None:
        """Append a JSON line to the event log."""
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:
            log.warning("Could not append to event log: %s", exc)

    def _log_equity(self, trade_date: date) -> None:
        """Append a row to the equity CSV."""
        try:
            acct  = self.broker.get_account()
            positions = self.broker.get_positions()
            cash = float(acct.get("cash", 0))
            pos_value = sum(
                float(abs(p.quantity)) * float(p.market_price)
                for p in positions.values()
            )
            total = cash + pos_value

            # Read existing equity df
            equity_file = Path(self.equity_path)
            if equity_file.exists():
                df = pd.read_csv(equity_file, parse_dates=["date"])
            else:
                df = self._equity_df.copy()

            # Compute daily return
            prev_total = float(df["total_equity"].iloc[-1]) if not df.empty else total
            ret = (total / prev_total - 1) if prev_total > 0 else 0.0

            new_row = pd.DataFrame([{
                "date":            str(trade_date),
                "cash":            round(cash, 2),
                "positions_value": round(pos_value, 2),
                "total_equity":    round(total, 2),
                "ret":             round(ret, 6),
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(equity_file, index=False)
            log.debug("Equity logged [%s]: $%.2f (ret %.2f%%)", trade_date, total, ret * 100)
        except Exception as exc:
            log.warning("Could not log equity: %s", exc)

    def _setup_handlers(self) -> None:
        """Attach signal handlers for graceful shutdown."""
        def _sig_handler(signum, frame):
            log.info("Received signal %d — shutting down...", signum)
            self._running = False
        signal.signal(signal.SIGINT,  _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

    def _shutdown(self) -> None:
        """Log final snapshot and stop the loop."""
        log.info("Paper trader shutting down...")
        try:
            self._log_equity(date.today())
        except Exception:
            pass
        self._running = False
        log.info("Shutdown complete. Events: %s  Equity: %s",
                 self.events_path, self.equity_path)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_paper_trader(
    strategy_name: str,
    tickers: list[str] | None = None,
    initial_cash: float = 100_000.0,
    scale_mode: ScaleMode | str = ScaleMode.FULL,
    risk_config: RiskLimitsConfig | None = None,
    strategy_type: str = "MACRO",
    params: dict | None = None,
) -> PaperTrader:
    """Factory: build a fully-wired PaperTrader for a named registered strategy.

    Parameters
    ----------
    strategy_name : str
        Name of a strategy registered via @register().
        Use list_strategies() to see available names.
    tickers : list[str], optional
        Assets to trade.  Defaults to the 6-asset macro universe.
    initial_cash : float
        Paper account starting cash.  Default $100 000.
    scale_mode : ScaleMode or str
        single_share | fractional | full
    risk_config : RiskLimitsConfig, optional
        Pre-built risk configuration.  If None, loads from env vars.
    strategy_type : str
        Strategy type for config. Default "MACRO".
    params : dict, optional
        Strategy parameters forwarded to StrategyConfig.

    Raises
    ------
    KeyError
        If strategy_name is not found in the registry.
    """
    available = list_strategies()
    if strategy_name not in available:
        raise KeyError(
            f"Strategy '{strategy_name}' not found in registry. "
            f"Available: {available}"
        )

    feed = LiveDataFeed(tickers=tickers)
    broker = LiveExecutionGateway(initial_cash=initial_cash)

    cfg = StrategyConfig(
        name=strategy_name,
        strategy_type=strategy_type,  # type: ignore[arg-type]
        max_position_notional=None,
        params=params or {},
    )
    strategy = registry_create(strategy_name, cfg)
    risk = RiskManagerLive(config=risk_config or RiskLimitsConfig.from_env())
    circuit = CircuitBreaker()

    return PaperTrader(
        feed=feed,
        broker=broker,
        strategy=strategy,
        risk_manager=risk,
        circuit_breaker=circuit,
        scale_mode=scale_mode,
    )
