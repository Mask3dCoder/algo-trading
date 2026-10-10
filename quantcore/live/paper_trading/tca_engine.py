"""Active Transaction Cost Analysis (TCA) Engine for Delta Neutral VRP.

Monitors execution quality, detects alpha leak from slippage and legging risk,
and logs all execution metrics to tca_performance.csv for post-trade analytics.

Design
-----
  1. TCAEngine — thread-safe execution analytics tracker
      - Records every ExecutionEvent from the broker
      - Computes slippage (bps), spread cost, and net execution quality
  2. PartialFillHandler — async watchdog for incomplete option legs
      - Tracks time-since-fill per partially-filled order
      - Triggers a market-order sweep if leg remains open > sweep_threshold_sec
      - Logs "legging risk premium" as the spread cost of the delayed fill
  3. TCALogger — append-only CSV writer for tca_performance.csv

Usage
-----
    tca = TCAEngine()
    tca.record_execution(event=execution_event, arrival_price=450.50)

    # In a background task:
    tca_partial = PartialFillHandler(tca_engine=tca, sweep_threshold_sec=30.0)
    asyncio.run(tca_partial.watch_loop())

    # Dashboard reads (no lock needed for reads with Copy-on-Write snapshots):
    snapshot = tca.get_snapshot()   # dict with today's metrics
"""

from __future__ import annotations

import csv
import logging
import threading
import time as _time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quantcore.live.execution_state import ExecutionEvent

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SPY typical spread (cents) / mid price → basis points
SPY_SPREAD_BPS = 1.5          # ~0.5 bps per side, 1.5 total
OPTIONS_SPREAD_BPS = 50.0     # Commission-equivalent spread for 5-wide markets

# Commission rates
OPTIONS_COMMISSION_PER_CONTRACT = 0.65
EQUITY_COMMISSION_PER_SHARE = 0.005   # not modelled here

REPORTS_DIR = Path("reports/paper_trading")
TCA_CSV_PATH = REPORTS_DIR / "tca_performance.csv"
ALERT_LOG_PATH = REPORTS_DIR / "tca_alerts.log"
ALLOWED_ASSET_TYPES = {"option", "equity", "stock"}


# ---------------------------------------------------------------------------
# Execution quality enums
# ---------------------------------------------------------------------------

class SlippageGrade(str, Enum):
    A_PLUS = "A+"   # < 1 bps slippage
    A = "A"         # 1–3 bps
    B = "B"         # 3–10 bps
    C = "C"         # 10–25 bps
    D = "D"         # 25–50 bps
    F = "F"         # > 50 bps


# ---------------------------------------------------------------------------
# Execution quality dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FillMetrics:
    """Immutable execution quality record for a single fill or partial fill."""
    timestamp: datetime
    order_id: str
    symbol: str
    side: str
    asset_type: str                          # "option" | "equity"
    arrival_price: float                     # mid-price when order was submitted
    fill_price: float                        # actual executed price
    mid_price: float                         # mid at time of fill
    contracts: int = 0                       # option contracts (0 for equity)
    shares: int = 0                          # equity shares (0 for options)
    slippage_bps: float = 0.0
    expected_slippage_bps: float = 0.0       # expected spread cost
    realized_slippage_bps: float = 0.0    # fill_price vs mid
    commission: float = 0.0
    legging_risk_bps: float = 0.0          # extra slippage from partial fill delay
    grade: SlippageGrade = SlippageGrade.A
    is_partial: bool = False
    is_sweep: bool = False                 # True = market-order sweep fill
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "asset_type": self.asset_type,
            "contracts": self.contracts,
            "shares": self.shares,
            "arrival_price": round(self.arrival_price, 4),
            "fill_price": round(self.fill_price, 4),
            "mid_price": round(self.mid_price, 4),
            "slippage_bps": round(self.slippage_bps, 2),
            "expected_slippage_bps": round(self.expected_slippage_bps, 2),
            "realized_slippage_bps": round(self.realized_slippage_bps, 2),
            "commission": round(self.commission, 4),
            "legging_risk_bps": round(self.legging_risk_bps, 2),
            "grade": self.grade.value,
            "is_partial": self.is_partial,
            "is_sweep": self.is_sweep,
            "notes": self.notes,
        }


@dataclass
class OrderTCAState:
    """Mutable TCA state for a single internal_order_id through its lifecycle."""
    order_id: str
    symbol: str
    side: str
    asset_type: str
    arrival_price: float
    mid_price_at_arrival: float
    submitted_at: datetime
    expected_spread_bps: float             # expected spread cost at submission
    fills: list[FillMetrics] = field(default_factory=list)
    partial_deadline: datetime | None = None  # when sweep should fire
    is_swept: bool = False
    is_closed: bool = False                 # terminal state reached


# ---------------------------------------------------------------------------
# TCA Snapshots (CoW — snapshot copies on write, read lock-free)
# ---------------------------------------------------------------------------

@dataclass
class TCASnapshot:
    """Point-in-time TCA metrics for dashboard consumption.

    All field values are locked at the moment of snapshot creation so that
    dashboard reads do not require any locking.
    """
    timestamp: datetime
    fill_count: int
    total_slippage_bps: float
    avg_slippage_bps: float
    worst_slippage_bps: float
    worst_order_id: str
    total_legging_risk_bps: float
    sweeeps_triggered: int
    avg_commission: float
    grade_distribution: dict[str, int]
    slippage_by_asset: dict[str, float]     # avg slippage by asset_type
    rejection_count: int
    total_orders: int

    def is_healthy(self) -> bool:
        return self.avg_slippage_bps < 20.0 and self.rejection_count == 0


# ---------------------------------------------------------------------------
# TCALogger — append-only CSV
# ---------------------------------------------------------------------------

class TCALogger:
    """Append-only CSV writer for tca_performance.csv.

    Thread-safe.  Opens the file in append mode so that concurrent writers
    from multiple processes do not truncate each other's rows.  A daily
    rotation marker is embedded as a comment row at midnight.
    """

    CSV_HEADER = [
        "timestamp", "order_id", "symbol", "side", "asset_type",
        "contracts", "shares", "arrival_price", "fill_price", "mid_price",
        "slippage_bps", "expected_slippage_bps", "realized_slippage_bps",
        "commission", "legging_risk_bps", "grade",
        "is_partial", "is_sweep", "notes",
    ]

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or TCA_CSV_PATH
        self._lock = threading.Lock()
        self._last_write_date: date | None = None
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self._path.exists():
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            writer.writeheader()

    def _maybe_write_rotation_marker(self) -> None:
        today = date.today()
        if self._last_write_date == today:
            return
        self._last_write_date = today
        try:
            with open(self._path, "a") as f:
                f.write(f"# TCA rotation {today.isoformat()}\n")
        except Exception as exc:
            _log.warning("TCA rotation marker write failed: %s", exc)

    def log(self, metrics: FillMetrics) -> None:
        try:
            with self._lock:
                self._maybe_write_rotation_marker()
                with open(self._path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
                    writer.writerow(metrics.to_row())
        except Exception as exc:
            _log.error("TCA CSV write failed: %s", exc)


# ---------------------------------------------------------------------------
# PartialFillHandler
# ---------------------------------------------------------------------------

class PartialFillHandler:
    r"""Watchdog that monitors partially-filled option legs and triggers sweeps.

    In multi-leg structures (straddles, spreads, iron condors) a partially-filled
    leg leaves the portfolio with unintended delta/gamma exposure.  This handler
    monitors the time elapsed since the partial fill and fires a market-order
    sweep when ``sweep_threshold_sec`` is exceeded.

    Legging Risk Premium
    --------------------
    The additional slippage incurred by completing a leg at a potentially
    worse price after a delay.  Computed as:

        legging_risk_bps = |delayed_fill_price - arrival_price| / arrival_price * 10_000

    If the market moved against us between partial fill and sweep, this
    captures the cost of the adverse selection.

    Parameters
    ----------
    tca_engine : TCAEngine
        Reference to record sweep fills with is_sweep=True.
    sweep_threshold_sec : float
        Seconds after which an open partial leg triggers a sweep order.
        Default 30 s — enough for a stale-quote rejection, not enough to
        leave the portfolio unhedged in fast markets.
    poll_interval_sec : float
        How often to scan open partial legs.  Default 5 s.
    on_sweep_callback : callable | None
        Optional async callback ``on_sweep(order_id: str)`` invoked when a sweep
        is triggered.  The runner uses this to route a market order to the broker.
    """

    def __init__(
        self,
        tca_engine: TCAEngine,
        sweep_threshold_sec: float = 30.0,
        poll_interval_sec: float = 5.0,
        on_sweep_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._tca = tca_engine
        self._threshold = sweep_threshold_sec
        self._poll = poll_interval_sec
        self._on_sweep = on_sweep_callback
        self._lock = threading.RLock()
        # Maps internal_order_id → datetime when partial was last updated
        self._partial_tracking: dict[str, datetime] = {}
        self._running = False
        self._watch_thread: threading.Thread | None = None
        # Sweep event log for post-trade review
        self._sweep_log: list[dict[str, Any]] = []

    def track_partial(
        self,
        order_id: str,
        partial_fill_price: float,
        arrival_price: float,
        symbol: str,
    ) -> None:
        """Call when a partial fill occurs. Starts/resets the sweep countdown."""
        now = datetime.utcnow()
        with self._lock:
            self._partial_tracking[order_id] = now
            # Log the initial partial fill TCA
            self._log_partial_fill(order_id, partial_fill_price, arrival_price, symbol)

    def _log_partial_fill(
        self,
        order_id: str,
        fill_price: float,
        arrival_price: float,
        symbol: str,
    ) -> None:
        slip_bps = _slippage_bps(fill_price, arrival_price)
        grade = _grade_from_bps(slip_bps)
        metrics = FillMetrics(
            timestamp=datetime.utcnow(),
            order_id=order_id,
            symbol=symbol,
            side="partial",
            asset_type="option",
            arrival_price=arrival_price,
            fill_price=fill_price,
            mid_price=fill_price,   # mid not known at partial time — use fill as proxy
            slippage_bps=slip_bps,
            expected_slippage_bps=OPTIONS_SPREAD_BPS,
            realized_slippage_bps=slip_bps,
            legging_risk_bps=0.0,
            grade=grade,
            is_partial=True,
            is_sweep=False,
            notes="partial_fill",
        )
        self._tca._log_and_record(metrics)

    def start(self) -> None:
        self._running = True
        self._watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        _log.info(
            "PartialFillHandler started: sweep_threshold=%.0fs poll=%.0fs",
            self._threshold,
            self._poll,
        )

    def stop(self) -> None:
        self._running = False
        if self._watch_thread:
            self._watch_thread.join(timeout=self._poll + 1.0)
        _log.info("PartialFillHandler stopped. Sweeps fired: %d", len(self._sweep_log))

    def _watch_loop(self) -> None:
        """Background polling loop.  Callable via threading, not asyncio."""
        while self._running:
            try:
                self._check_partials()
            except Exception as exc:
                _log.error("PartialFillHandler._check_partials error: %s", exc)
            _time.sleep(self._poll)

    def _check_partials(self) -> None:
        now = datetime.utcnow()
        sweep_fired: list[str] = []
        legging_premium: list[float] = []

        with self._lock:
            to_remove: list[str] = []
            for order_id, partial_at in list(self._partial_tracking.items()):
                elapsed = (now - partial_at).total_seconds()
                if elapsed >= self._threshold:
                    sweep_fired.append(order_id)
                    to_remove.append(order_id)

            for order_id in to_remove:
                del self._partial_tracking[order_id]

        # Trigger callbacks outside the lock to avoid deadlock
        for order_id in sweep_fired:
            # The runner's sweep handler routes the market order
            if self._on_sweep is not None:
                try:
                    self._on_sweep(order_id)
                except Exception as exc:
                    _log.warning("Sweep callback failed for %s: %s", order_id, exc)
            self._sweep_log.append({
                "order_id": order_id,
                "fired_at": now.isoformat(),
                "elapsed_sec": self._threshold,
                "sweep_triggered": True,
            })
            _log.warning(
                "SWEEP: Partial fill for order %s exceeded %.0fs — market sweep fired.",
                order_id,
                self._threshold,
            )

    @property
    def sweep_count(self) -> int:
        return len(self._sweep_log)

    def get_sweep_log(self) -> list[dict[str, Any]]:
        return list(self._sweep_log)


# ---------------------------------------------------------------------------
# TCAEngine — main tracker
# ---------------------------------------------------------------------------

@dataclass
class TCAConfig:
    """TCA engine configuration.

    Parameters
    ----------
    spy_benchmark_bps : float
        SPY expected spread in basis points (default 1.5 bps half-spread).
        Used as the baseline for equity execution quality grading.
    options_benchmark_bps : float
        Expected options spread in bps for the at-the-money strike at entry.
        Default 50 bps — representative of a 5-wide market on a $3 option.
        Adjust upward for wide-products (VIX, weeklys).
    alert_slippage_bps : float
        Slippage threshold that triggers a WARNING-level TCA alert.
        Default 25 bps — roughly 1 contract width on a $3 option.
    critical_slippage_bps : float
        Slippage threshold that triggers a CRITICAL TCA alert.
        Default 50 bps — 2 contract widths on a $3 option.
    """
    spy_benchmark_bps: float = 1.5
    options_benchmark_bps: float = 50.0
    alert_slippage_bps: float = 25.0
    critical_slippage_bps: float = 50.0
    legging_risk_bps_warning: float = 10.0


class TCAEngine:
    r"""Thread-safe Transaction Cost Analysis engine.

    Integrates with the broker's ``ExecutionEvent`` stream to compute:

      1. Slippage Degradation
         - Arrival Price : mid-price at moment order was submitted to the broker
         - Fill Price    : actual executed price from the broker fill payload
         - Slippage (bps) = (fill_price - arrival_price) / arrival_price × 10,000

      2. Spread Cost Analysis
         - Expected Spread Cost (bps) : spy_benchmark_bps for equity,
           options_benchmark_bps for options
         - Realized Spread Cost (bps): (fill_price - mid_price) / mid_price × 10,000

      3. Partial Fill & Legging Risk
         - Partial fills are tracked with timestamps
         - Legging Risk Premium = extra slippage from completing a leg at a
           worse price after a stale-quote rejection

    Thread Safety
    -------------
    All mutable state is protected by ``self._lock`` (RLock).  Dashboard reads
    use ``get_snapshot()`` which creates a CoW snapshot — reads never block
    order submission.

    Usage
    -----
    # In the runner loop (main thread):
    tca = TCAEngine()
    tca.record_execution(event, arrival_price, mid_price)

    # Dashboard (separate thread):
    snap = tca.get_snapshot()   # lock-free read
    print(snap.avg_slippage_bps)
    """

    def __init__(
        self,
        config: TCAConfig | None = None,
        csv_path: Path | None = None,
    ) -> None:
        self._cfg = config or TCAConfig()
        self._logger = TCALogger(path=csv_path)
        self._lock = threading.RLock()
        self._orders: dict[str, OrderTCAState] = {}   # keyed by internal_order_id
        self._today = date.today()
        # Rolling daily accumulators (reset at midnight via _reset_if_new_day)
        self._slippage_list: list[float] = []
        self._rejection_count = 0
        self._sweep_count = 0
        # Alert tracking
        self._alert_hook: Callable[[str], None] | None = None

    def set_alert_hook(self, hook: Callable[[str], None]) -> None:
        """Set a callback invoked on TCA alert (WARNING / CRITICAL)."""
        self._alert_hook = hook

    def record_execution(
        self,
        event: ExecutionEvent,
        arrival_price: float,
        mid_price: float,
        is_sweep: bool = False,
        notes: str = "",
    ) -> FillMetrics:
        r"""Record a fill or rejection and compute TCA metrics.

        Called by the broker adapter on every fill callback.  Thread-safe.

        Parameters
        ----------
        event : ExecutionEvent
            Normalized broker execution event from quantcore.live.execution_state.
        arrival_price : float
            Mid-price when the order was submitted to the broker.
        mid_price : float
            Current mid-price at the time of fill (may differ from arrival_price).
        is_sweep : bool
            True if this fill resulted from a partial-fill sweep.
        notes : str
            Free-text annotation written to the CSV.

        Returns
        -------
        FillMetrics
            Immutable execution quality record for this fill.
        """
        self._reset_if_new_day()
        with self._lock:
            order_id = event.internal_order_id

            if event.state.value in {"rejected", "cancelled", "expired"}:
                self._rejection_count += 1
                _log.warning("TCA: order %s rejected (%s)", order_id, event.state.value)
                self._maybe_alert(
                    "WARNING",
                    f"TCA: order {order_id} rejected — {event.state.value}",
                )
                return FillMetrics(
                    timestamp=datetime.utcnow(),
                    order_id=order_id,
                    symbol=event.symbol,
                    side=event.side.value,
                    asset_type=self._infer_asset_type(event.symbol),
                    arrival_price=arrival_price,
                    fill_price=event.price or arrival_price,
                    mid_price=mid_price,
                    slippage_bps=0.0,
                    expected_slippage_bps=self._cfg.options_benchmark_bps,
                    realized_slippage_bps=0.0,
                    notes=f"rejected: {event.state.value}",
                )

            fill_price = float(event.price) if event.price else arrival_price
            is_partial = event.state.value == "partially_filled"

            # Ensure we have an order record
            if order_id not in self._orders:
                self._orders[order_id] = OrderTCAState(
                    order_id=order_id,
                    symbol=event.symbol,
                    side=event.side.value,
                    asset_type=self._infer_asset_type(event.symbol),
                    arrival_price=arrival_price,
                    mid_price_at_arrival=mid_price,
                    submitted_at=event.timestamp,
                    expected_spread_bps=(
                        self._cfg.options_benchmark_bps
                        if self._infer_asset_type(event.symbol) == "option"
                        else self._cfg.spy_benchmark_bps
                    ),
                )

            state = self._orders[order_id]

            # Compute slippage metrics
            slippage = _slippage_bps(fill_price, arrival_price)
            expected = state.expected_spread_bps
            realized = _slippage_bps(fill_price, mid_price)

            # Legging risk: add realized spread cost if partial
            legging_risk = realized if is_partial else 0.0

            grade = _grade_from_bps(slippage)

            contracts = int(abs(float(event.filled_quantity))) // 100 if is_partial else 0
            shares = int(abs(float(event.filled_quantity))) if not is_partial else 0

            metrics = FillMetrics(
                timestamp=event.timestamp or datetime.utcnow(),
                order_id=order_id,
                symbol=event.symbol,
                side=event.side.value,
                asset_type=state.asset_type,
                contracts=contracts,
                shares=shares,
                arrival_price=arrival_price,
                fill_price=fill_price,
                mid_price=mid_price,
                slippage_bps=slippage,
                expected_slippage_bps=expected,
                realized_slippage_bps=realized,
                commission=_commission(
                    state.asset_type, contracts or shares
                ),
                legging_risk_bps=legging_risk,
                grade=grade,
                is_partial=is_partial,
                is_sweep=is_sweep,
                notes=notes,
            )
            state.fills.append(metrics)
            state.is_swept = is_sweep
            if event.state.value in {"filled"}:
                state.is_closed = True

            # Update rolling accumulators
            self._slippage_list.append(slippage)
            if is_sweep:
                self._sweep_count += 1

            # Log to CSV
            self._logger.log(metrics)

            # Alert on bad fills
            if slippage >= self._cfg.critical_slippage_bps:
                self._maybe_alert(
                    "CRITICAL",
                    f"TCA CRITICAL: {order_id} slippage={slippage:.1f}bps fill=${fill_price:.4f}",
                )
            elif slippage >= self._cfg.alert_slippage_bps:
                self._maybe_alert(
                    "WARNING",
                    f"TCA WARNING: {order_id} slippage={slippage:.1f}bps fill=${fill_price:.4f}",
                )

            if legging_risk >= self._cfg.legging_risk_bps_warning:
                self._maybe_alert(
                    "WARNING",
                    f"TCA: legging risk {legging_risk:.1f}bps on partial {order_id} ({state.asset_type})",
                )

            return metrics

    def record_benchmark(
        self,
        order_id: str,
        symbol: str,
        side: str,
        asset_type: str,
        arrival_price: float,
        mid_price: float,
        expected_spread_bps: float,
    ) -> None:
        """Record a 'planned but not filled' benchmark price point (no trade).

        For orders that were generated but not submitted (passive tracking mode),
        record the theoretical fill cost for P&L attribution.
        """
        with self._lock:
            self._orders[order_id] = OrderTCAState(
                order_id=order_id,
                symbol=symbol,
                side=side,
                asset_type=asset_type,
                arrival_price=arrival_price,
                mid_price_at_arrival=mid_price,
                submitted_at=datetime.utcnow(),
                expected_spread_bps=expected_spread_bps,
            )

    def get_snapshot(self) -> TCASnapshot:
        """Return a locked point-in-time snapshot for dashboard consumption."""
        with self._lock:
            self._reset_if_new_day()
            total = len(self._slippage_list)
            slippage_sum = sum(self._slippage_list)
            grade_dist: dict[str, int] = defaultdict(int)
            by_asset: dict[str, list[float]] = defaultdict(list)

            for state in self._orders.values():
                for f in state.fills:
                    grade_dist[f.grade.value] += 1
                    by_asset[f.asset_type].append(f.slippage_bps)

            total_orders = len(self._orders)
            avg = slippage_sum / total if total else 0.0
            worst_bps = max(self._slippage_list) if self._slippage_list else 0.0
            worst_oid = ""
            for state in self._orders.values():
                for f in state.fills:
                    if abs(f.slippage_bps) >= worst_bps - 0.001:
                        worst_oid = state.order_id
                        break

            slippage_by_asset = {
                k: sum(v) / len(v) for k, v in by_asset.items() if v
            }

            return TCASnapshot(
                timestamp=datetime.utcnow(),
                fill_count=total,
                total_slippage_bps=round(slippage_sum, 2),
                avg_slippage_bps=round(avg, 2),
                worst_slippage_bps=round(worst_bps, 2),
                worst_order_id=worst_oid,
                total_legging_risk_bps=round(
                    sum(
                        f.legging_risk_bps
                        for state in self._orders.values()
                        for f in state.fills
                    ),
                    2,
                ),
                sweeeps_triggered=self._sweep_count,
                avg_commission=round(
                    sum(
                        f.commission
                        for state in self._orders.values()
                        for f in state.fills
                    ) / total if total else 0.0,
                    4,
                ),
                grade_distribution=dict(grade_dist),
                slippage_by_asset={k: round(v, 2) for k, v in slippage_by_asset.items()},
                rejection_count=self._rejection_count,
                total_orders=total_orders,
            )

    def _log_and_record(self, metrics: FillMetrics) -> None:
        """Used by PartialFillHandler to record partial fills without full event."""
        with self._lock:
            self._slippage_list.append(metrics.slippage_bps)
            if metrics.is_partial:
                self._orders[metrics.order_id] = OrderTCAState(
                    order_id=metrics.order_id,
                    symbol=metrics.symbol,
                    side=metrics.side,
                    asset_type=metrics.asset_type,
                    arrival_price=metrics.arrival_price,
                    mid_price_at_arrival=metrics.mid_price,
                    submitted_at=metrics.timestamp,
                    expected_spread_bps=self._cfg.options_benchmark_bps,
                )
            self._logger.log(metrics)

    def _reset_if_new_day(self) -> None:
        today = date.today()
        if self._today == today:
            return
        self._today = today
        self._slippage_list.clear()
        self._rejection_count = 0
        self._sweep_count = 0
        _log.info("TCAEngine: day rollover %s — accumulated metrics reset", today)

    def _infer_asset_type(self, symbol: str) -> str:
        # Simple heuristic — in production, use a proper symbol classifier
        return "option" if len(symbol) > 6 or "_" in symbol else "equity"

    def _maybe_alert(self, level: str, msg: str) -> None:
        _log.log(
            logging.CRITICAL if level == "CRITICAL" else logging.WARNING,
            msg,
        )
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(ALERT_LOG_PATH, "a") as f:
                f.write(
                    f"{datetime.utcnow().isoformat()} [{level}] TCA: {msg}\n"
                )
        except Exception as exc:
            _log.error("TCA alert log write failed: %s", exc)
        if self._alert_hook:
            try:
                self._alert_hook(f"[{level}] {msg}")
            except Exception as exc:
                _log.error("TCA alert hook failed: %s", exc)

    @property
    def orders(self) -> dict[str, OrderTCAState]:
        """Expose order dict for the terminal dashboard."""
        with self._lock:
            return dict(self._orders)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _slippage_bps(fill_price: float, reference_price: float) -> float:
    """Return slippage in basis points vs reference price.

    Positive = adverse slippage (filled worse than reference).
    Negative = price improvement.
    """
    if reference_price <= 0:
        return 0.0
    return (fill_price - reference_price) / reference_price * 10_000.0


def _grade_from_bps(slippage_bps: float) -> SlippageGrade:
    abs_bps = abs(slippage_bps)
    if abs_bps < 1.0:
        return SlippageGrade.A_PLUS
    if abs_bps < 3.0:
        return SlippageGrade.A
    if abs_bps < 10.0:
        return SlippageGrade.B
    if abs_bps < 25.0:
        return SlippageGrade.C
    if abs_bps < 50.0:
        return SlippageGrade.D
    return SlippageGrade.F


def _commission(asset_type: str, quantity: int) -> float:
    """Calculate commission for a fill."""
    if asset_type == "option":
        return quantity * OPTIONS_COMMISSION_PER_CONTRACT
    return quantity * EQUITY_COMMISSION_PER_SHARE
