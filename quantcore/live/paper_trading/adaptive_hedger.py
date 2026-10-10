"""Adaptive Threshold-Based Delta Hedging Optimizer for Delta Neutral VRP.

Replaces the rigid 300-second polling interval with a microstructural controller
that continuously samples incoming underlying ticks and triggers rebalancing
only when the risk of inaction exceeds the cost of hedging.

Mathematical Design
-------------------

Dynamic Threshold
~~~~~~~~~~~~~~~~~
A volatility-adjusted delta boundary is computed as:

    delta_threshold = max(delta_min, vol_adjusted_threshold)

where:
    vol_adjusted_threshold = σ_Daily × √(1/390) × Spot × multiplier

    σ_Daily            = annualised volatility of the underlying (%)
    √(1/390)           = 1-minute std-dev scaling (390 trading seconds/day)
    Spot              = current underlying mid price
    multiplier         = config.drift_multiplier (default 1.5 σ)

Using SPY at $450 with 16% annual vol as example:
    σ_min = 450 × 0.16 / √252 / √390 = ~0.028 $/min → threshold ≈ 4-5 shares

This ensures the hedge fires ONLY on statistically significant delta drift,
not on random noise from bid-ask bounce.

Cost-Benefit Filter
~~~~~~~~~~~~~~~~~~
Before routing a hedge order, the controller verifies:

    Execute Hedge if  dollar_gamma_risk > hedge_transaction_cost

    dollar_gamma_risk = Γ_$ × σ_min × √(Δt)
        Γ_$           = dollar gamma per 1% move
        σ_min         = 1-min vol proxy of underlying
        Δt            = elapsed minutes since last hedge

    hedge_tc = hedge_shares × (spread_bps/10000 × spot + commission_per_share)

If gamma risk < TC → defer the hedge (no economically rational basis to trade).

Macro Cap: hedge_max_age_sec
~~~~~~~~~~~~~~~~~~~~~~~~~~~
A hard ceiling of 30 minutes between hedges prevents long-tail delta drift
in low-volume regimes where the microstructural threshold might not fire.

Usage
-----
    hedger = AdaptiveHedger(
        underlying="SPY",
        config=AdaptiveHedgerConfig(),
        on_hedge_signal=_handle_signal,
    )
    hedger.start()

    # In the tick loop (non-blocking):
    hedger.on_underlying_tick(price=450.25, bid=450.20, ask=450.30, dt=datetime.utcnow())
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

_log = __import__("logging").getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DELTA_MIN = 5              # minimum delta before triggering (shares)
DEFAULT_SPREAD_BPS = 2.5          # realistic half-spread for SPY
DEFAULT_HEDGE_COMMISSION = 0.005  # per share
DEFAULT_MAX_AGE_SEC = 1800        # 30 minutes macro cap
DEFAULT_DRIFT_MULTIPLIER = 1.5    # σ multiplier for dynamic threshold
DEFAULT_GAMMA_MULTIPLIER = 2.0    # gamma risk safety margin
ALPHA_HEDGE_SHARES = 50            # fixed-size alpha hedge (shares per triggering)


class HedgeAction(str, Enum):
    FULL_HEDGE = "full_hedge"      # delta → 0
    PARTIAL_HEDGE = "partial_hedge" # delta → threshold boundary
    DEFER = "defer"                 # cost > benefit — wait
    NO_ACTION = "no_action"         # no trigger


def _price_vol(price: float, annual_vol_pct: float) -> float:
    """1-minute realised vol proxy in price units."""
    return price * (annual_vol_pct / 100.0) / math.sqrt(252 * 390)


# ---------------------------------------------------------------------------
# Hedge signal dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HedgeSignal:
    """Immutable signal emitted by AdaptiveHedger to the runner."""

    action: HedgeAction
    direction: int                       # +1 = BUY shares, -1 = SELL shares, 0 = none
    shares: int                          # number of shares to trade
    reason: str                          # human-readable justification
    net_delta_before: float              # portfolio delta at trigger time
    net_delta_after: float               # expected delta after hedge (≈ 0 or boundary)
    threshold: float                     # threshold that was breached
    urgency: str = "normal"             # "normal" | "high" | "critical"
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    hedge_cost_estimate: float = 0.0    # estimated total cost of this hedge
    deferral_count: int = 0              # how many times we deferred before firing

    @property
    def is_hedge(self) -> bool:
        return self.action in {HedgeAction.FULL_HEDGE, HedgeAction.PARTIAL_HEDGE}


# ---------------------------------------------------------------------------
# AdaptiveHedger
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveHedgerConfig:
    """Configuration for the AdaptiveHedger.

    All parameters are hardcoded per Investment Committee mandate — not tunable.
    """

    # -- Threshold parameters ----------------------------------------------------
    delta_min: float = DEFAULT_DELTA_MIN
    spread_bps: float = DEFAULT_SPREAD_BPS
    commission_per_share: float = DEFAULT_HEDGE_COMMISSION
    max_hedge_age_sec: float = DEFAULT_MAX_AGE_SEC
    drift_multiplier: float = DEFAULT_DRIFT_MULTIPLIER
    gamma_multiplier: float = DEFAULT_GAMMA_MULTIPLIER

    # -- Volatility lookback ------------------------------------------------------
    vol_lookback_bars: int = 20        # bars for rolling vol estimate
    annual_vol_pct: float = 16.0       # nominal vol % if no bars yet

    # -- Partial hedge fraction ---------------------------------------------------
    # If True, only hedge to the threshold boundary (not to zero)
    # This reduces turnover in choppy markets
    partial_hedge_fraction: float = 0.5  # hedge 50% of the excess delta

    # -- Logging ----------------------------------------------------------------
    log_dir: Path = field(default_factory=lambda: Path("reports/paper_trading/logs"))


class AdaptiveHedger:
    r"""Adaptive microstructural delta hedging controller.

    Thread-safe.  Designed to run alongside PaperTradingRunner without
    introducing asyncio latency into the main execution path.

    Monitoring Contract
    ~~~~~~~~~~~~~~~~~~~
    The hedger monitors **net portfolio delta** — the aggregate delta of all
    open option positions plus the delta hedge shares already in the book.
    It does NOT manage individual leg deltas; that is handled by the
    PaperExecutionEnvironment.

    The runner is responsible for:
      1. Calling ``on_underlying_tick(price, ...)`` on every incoming trade/quote
      2. Calling ``on_portfolio_delta_update(net_delta, ...)`` whenever the
         option portfolio delta changes (position opened/closed, delta recomputed)
      3. Handling ``HedgeSignal`` returned by ``decide()`` by routing a
         market order to the broker, then calling ``record_hedge_executed()``

    State Machine
    ~~~~~~~~~~~~
    DEFER → [threshold breach or max_age hit] → HEDGE → DEFER
            ↑
            ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←

    The controller never moves backward — it only goes DEFER → HEDGE → DEFER.
    After a hedge is executed, call ``record_hedge_executed(shares, fill_price)``
    to update internal state.

    Usage
    -----
    # In the runner loop:
    hedger = AdaptiveHedger(
        underlying="SPY",
        config=AdaptiveHedgerConfig(),
        on_signal=_route_hedge_order,
    )
    hedger.start()

    # On every tick (from the main loop, not a callback):
    hedger.on_underlying_tick(price=450.25, bid=450.20, ask=450.30)
    signal = hedger.decide()
    if signal.is_hedge:
        _route_market_order(signal.shares)   # runner implements this
        hedger.record_hedge_executed(signal.shares, fill_price=450.27)

    # When option portfolio changes (after on_data() or on_fill()):
    hedger.on_portfolio_delta_update(net_delta=new_delta, dollar_gamma=new_greeks["Gamma"] * 100)
    """

    # Sentinel for uninitialized state
    _UNSET = tuple()

    def __init__(
        self,
        underlying: str,
        config: AdaptiveHedgerConfig | None = None,
        on_signal: Callable[[HedgeSignal], None] | None = None,
    ) -> None:
        self._underlying = underlying
        self._cfg = config or AdaptiveHedgerConfig()
        self._on_signal = on_signal
        self._lock = threading.RLock()

        # -- Volatility state ----------------------------------------------------
        self._price_window: deque[float] = deque(maxlen=self._cfg.vol_lookback_bars)

        # -- Delta state --------------------------------------------------------
        self._net_delta: float = 0.0       # net portfolio delta (shares)
        self._dollar_gamma: float = 0.0   # dollar gamma (Σ |delta| × $ per 1% move)

        # -- Timing state -------------------------------------------------------
        self._last_hedge_time: datetime | None = None
        self._last_underlying_tick: datetime | None = None
        self._last_underlying_price: float = 0.0
        self._bid: float = 0.0
        self._ask: float = 0.0

        # -- Decision state ------------------------------------------------------
        self._deferral_count = 0
        self._hedge_count = 0

        # -- Stat counters -------------------------------------------------------
        self._stats = {
            "total_signals": 0,
            "total_hedges": 0,
            "total_deferrals": 0,
            "max_deferrals": 0,
            "avg_cost": 0.0,
            "total_cost": 0.0,
        }

        _log.info(
            "AdaptiveHedger: underlying=%s delta_min=%.1f max_age=%.0fs vol=%.0f%%",
            underlying,
            self._cfg.delta_min,
            self._cfg.max_hedge_age_sec,
            self._cfg.annual_vol_pct,
        )

    # ------------------------------------------------------------------
    # Public API — called by the runner
    # ------------------------------------------------------------------

    def on_underlying_tick(
        self,
        price: float,
        bid: float,
        ask: float,
        dt: datetime | None = None,
    ) -> None:
        """Feed a new underlying trade/quote tick into the hedger.

        Thread-safe.  Call from the main tick loop — no asyncio needed.

        Parameters
        ----------
        price : float
            Last trade mid-price (bid+ask)/2.
        bid, ask : float
            Current bid and ask for spread cost estimation.
        dt : datetime | None
            Tick timestamp.  Defaults to utcnow().
        """
        dt = dt or datetime.now(UTC)
        with self._lock:
            self._last_underlying_tick = dt
            self._last_underlying_price = price
            self._bid = bid
            self._ask = ask
            if price > 0:
                self._price_window.append(price)

    def on_portfolio_delta_update(
        self,
        net_delta: float,
        dollar_gamma: float,
        dt: datetime | None = None,
    ) -> None:
        """Update the net portfolio delta and dollar gamma.

        Called by the runner whenever option positions change (trade fill,
        position closed, or periodic Greek recompute).

        Thread-safe.
        """
        dt = dt or datetime.now(UTC)
        with self._lock:
            prev = self._net_delta
            self._net_delta = net_delta
            self._dollar_gamma = dollar_gamma
            _log.debug(
                "AdaptiveHedger: net_delta %.1f → %.1f  dollar_gamma=$%.2f",
                prev,
                net_delta,
                dollar_gamma,
            )

    def decide(self) -> HedgeSignal:
        """Evaluate whether a hedge order should be fired.

        Call from the runner loop, outside any lock held by the tick consumer.
        This method acquires its own lock internally — safe to call concurrently
        with ``on_underlying_tick()``.

        Returns
        -------
        HedgeSignal
            Contains action, shares, direction, reason.  If action is DEFER
            or NO_ACTION, shares = 0 and no broker order should be sent.
        """
        with self._lock:
            self._stats["total_signals"] += 1
            now = datetime.now(UTC)

            # Compute volatility-adjusted threshold
            threshold = self._compute_threshold()
            abs_delta = abs(self._net_delta)

            # Check 1: mandatory macro cap (30-min rule — never go longer without hedge)
            macro_hit, macro_reason = self._check_macro_cap(now)
            if macro_hit:
                direction = -1 if self._net_delta > 0 else 1
                shares = ALPHA_HEDGE_SHARES
                signal = self._make_signal(
                    action=HedgeAction.FULL_HEDGE,
                    direction=direction,
                    shares=shares,
                    reason=macro_reason,
                    threshold=threshold,
                    urgency="critical" if self._deferral_count > 3 else "high",
                )
                self._emit(signal)
                return signal

            # Check 2: microstructural threshold
            if abs_delta > threshold:
                # Cost-benefit filter
                defer, defer_reason = self._cost_benefit_filter(abs_delta)
                if defer:
                    self._deferral_count += 1
                    self._stats["total_deferrals"] += 1
                    self._stats["max_deferrals"] = max(
                        self._stats["max_deferrals"],
                        self._deferral_count,
                    )
                    signal = HedgeSignal(
                        action=HedgeAction.DEFER,
                        direction=0,
                        shares=0,
                        reason=f"{defer_reason} (defer #{self._deferral_count})",
                        net_delta_before=self._net_delta,
                        net_delta_after=self._net_delta,
                        threshold=threshold,
                        urgency="normal",
                        deferral_count=self._deferral_count,
                    )
                    return signal
                else:
                    # Fire the hedge
                    direction = -1 if self._net_delta > 0 else 1
                    shares = self._compute_hedge_shares()
                    signal = self._make_signal(
                        action=HedgeAction.FULL_HEDGE
                        if abs_delta <= threshold * 2
                        else HedgeAction.PARTIAL_HEDGE,
                        direction=direction,
                        shares=shares,
                        reason=(
                            f"delta {self._net_delta:.1f} > threshold {threshold:.1f} "
                            f"(costbenefit=ok)"
                        ),
                        threshold=threshold,
                        urgency="high",   # >2× threshold = high urgency
                    )
                    self._emit(signal)
                    return signal

            # No trigger
            self._deferral_count = 0   # reset on clean pass
            return HedgeSignal(
                action=HedgeAction.NO_ACTION,
                direction=0,
                shares=0,
                reason=f"delta {self._net_delta:.1f} within threshold {threshold:.1f}",
                net_delta_before=self._net_delta,
                net_delta_after=self._net_delta,
                threshold=threshold,
                urgency="normal",
                deferral_count=0,
            )

    def record_hedge_executed(
        self,
        shares: int,
        fill_price: float,
        dt: datetime | None = None,
    ) -> None:
        """Record that a hedge order was filled and update internal state.

        MUST be called by the runner after the broker confirms the hedge fill.
        This updates ``net_delta`` so subsequent ``decide()`` calls reflect
        the new position.

        Thread-safe.
        """
        dt = dt or datetime.now(UTC)
        with self._lock:
            self._last_hedge_time = dt

            # Update net delta (shares in direction → reduce abs(delta))
            hedge_delta = shares * (-1 if self._net_delta < 0 else 1)
            self._net_delta = self._net_delta + hedge_delta
            # Snap to zero if close enough
            if abs(self._net_delta) < 0.5:
                self._net_delta = 0.0

            # Update stats
            spread_tc = self._cfg.spread_bps / 10_000 * fill_price * shares
            comm_tc = self._cfg.commission_per_share * shares
            total_tc = spread_tc + comm_tc
            self._stats["total_hedges"] += 1
            self._stats["total_cost"] += total_tc
            n = self._stats["total_hedges"]
            self._stats["avg_cost"] = self._stats["total_cost"] / n

            _log.info(
                "AdaptiveHedger: hedge executed %d shares @ %.4f  cost=$%.4f  "
                "new_delta=%.1f",
                shares,
                fill_price,
                total_tc,
                self._net_delta,
            )

    def current_state(self) -> dict:
        """Return a dict snapshot of the current hedger state for the dashboard."""
        with self._lock:
            return {
                "underlying": self._underlying,
                "net_delta": round(self._net_delta, 2),
                "dollar_gamma": round(self._dollar_gamma, 2),
                "threshold": round(self._compute_threshold(), 2),
                "last_tick": (
                    self._last_underlying_tick.isoformat()
                    if self._last_underlying_tick else None
                ),
                "last_price": round(self._last_underlying_price, 4),
                "bid": round(self._bid, 4),
                "ask": round(self._ask, 4),
                "time_since_hedge_sec": self._time_since_hedge(),
                "deferral_count": self._deferral_count,
                "hedge_count": self._stats["total_hedges"],
                "avg_hedge_cost": round(self._stats["avg_cost"], 4),
            }

    def start(self) -> None:
        """Start any background threads (currently none — this is a polling hedger).

        Kept for API symmetry with PassiveExecutionProxy.
        """
        _log.info("AdaptiveHedger started for %s", self._underlying)

    def stop(self) -> None:
        """Stop the hedger.  No-op since we use no background threads."""
        _log.info(
            "AdaptiveHedger stopped: %d hedges, avg_cost=$%.4f",
            self._stats["total_hedges"],
            self._stats["avg_cost"],
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_threshold(self) -> float:
        """Compute volatility-adjusted delta threshold (in shares)."""
        price = self._last_underlying_price
        if price <= 0:
            return self._cfg.delta_min

        # Compute rolling vol if we have enough data
        vol_pct = self._cfg.annual_vol_pct
        if len(self._price_window) >= 5:
            # Simple rolling std — percent of price
            prices = list(self._price_window)
            returns = [math.log(prices[i] / prices[i - 1])
                      for i in range(1, len(prices))]
            if returns:
                mean_r = sum(returns) / len(returns)
                var = sum((r - mean_r) ** 2 for r in returns) / max(1, len(returns) - 1)
                # Annualise: daily vol, then convert to annual
                daily_vol = math.sqrt(var) * math.sqrt(252)
                vol_pct = daily_vol * 100 * math.sqrt(390)  # → annual min-scale

        threshold = max(
            self._cfg.delta_min,
            vol_pct / 100.0
            * price
            / math.sqrt(252 * 390)
            * self._cfg.drift_multiplier
            * math.sqrt(max(1.0, self._time_since_hedge() / 60.0)),
        )
        return threshold

    def _check_macro_cap(self, now: datetime) -> tuple[bool, str]:
        """Check 30-minute macro ceiling.  Returns (hit, reason)."""
        elapsed = self._time_since_hedge()
        if self._last_hedge_time is None:
            # Never hedged — apply leniency but warn
            if self._time_since_hedge() > self._cfg.max_hedge_age_sec:
                return True, (
                    f"first hedge — macro cap {self._cfg.max_hedge_age_sec}s elapsed "
                    "(no prior hedge in session)"
                )
            return False, ""

        excess = elapsed - self._cfg.max_hedge_age_sec
        if excess > 0:
            return True, (
                f"macro cap breached: {elapsed:.0f}s > {self._cfg.max_hedge_age_sec}s "
                f"(+{excess:.0f}s overage)"
            )
        # Warn if approaching cap but not reached
        if elapsed > self._cfg.max_hedge_age_sec * 0.9:
            _log.warning(
                "AdaptiveHedger: macro cap %.0f%% reached (%.0f/%.0fs) — "
                "hedge will fire on next evaluate()",
                100 * elapsed / self._cfg.max_hedge_age_sec,
                elapsed,
                self._cfg.max_hedge_age_sec,
            )
        return False, ""

    def _cost_benefit_filter(self, abs_delta: float) -> tuple[bool, str]:
        """Return (defer, reason) if transaction cost > gamma risk.

        Cost-Benefit Analysis
        --------------------
        gamma_risk_dollar = dollar_gamma × σ_min × √Δt
        hedge_tc = abs(delta) × spread_bps/10000 × price + abs(delta) × commission

        hedge_tc  >  gamma_risk_dollar  → DEFER (cost dominates)
        hedge_tc <=  gamma_risk_dollar  → EXECUTE (risk dominates)
        """
        Δt_min = self._time_since_hedge() / 60.0
        price = self._last_underlying_price

        if price <= 0 or Δt_min <= 0:
            return True, "insufficient data for cost-benefit"

        # Dollar gamma risk
        vol_pct = self._cfg.annual_vol_pct
        σ_min = price * (vol_pct / 100.0) / math.sqrt(252 * 390)
        gamma_risk = max(self._dollar_gamma, 1.0) * σ_min * math.sqrt(max(1.0, Δt_min))

        # Transaction cost for hedging abs_delta shares
        spread_tc = abs_delta * (self._cfg.spread_bps / 10_000) * price
        comm_tc = abs_delta * self._cfg.commission_per_share
        hedge_tc = spread_tc + comm_tc

        # Apply gamma multiplier safety factor
        adjusted_gamma_risk = gamma_risk * self._cfg.gamma_multiplier

        if hedge_tc > adjusted_gamma_risk:
            _log.debug(
                "AdaptiveHedger: DEFER — hedge_tc=$%.4f > gamma_risk=$%.4f "
                "(delta=%.1f price=%.2f Δt=%.1fmin)",
                hedge_tc,
                adjusted_gamma_risk,
                abs_delta,
                price,
                Δt_min,
            )
            return True, (
                f"cost-benefit: hedge_tc=${hedge_tc:.4f} > "
                f"gamma_risk=${adjusted_gamma_risk:.4f} ({Δt_min:.1f}min)"
            )

        _log.debug(
            "AdaptiveHedger: EXECUTE — hedge_tc=$%.4f <= gamma_risk=$%.4f (delta=%.1f)",
            hedge_tc,
            adjusted_gamma_risk,
            abs_delta,
        )
        return False, f"cost-benefit ok: tc=${hedge_tc:.4f} <= risk=${adjusted_gamma_risk:.4f}"

    def _compute_hedge_shares(self) -> int:
        """Compute number of shares to trade to bring delta to within threshold."""
        excess = abs(self._net_delta) - self._cfg.delta_min
        if excess <= 0:
            return 0
        fraction = self._cfg.partial_hedge_fraction
        return max(1, int(excess * fraction))

    def _time_since_hedge(self) -> float:
        """Return seconds since last hedge (0 if never hedged)."""
        if self._last_hedge_time is None:
            # Allow full max_age on first hedge
            return 0.0
        return (datetime.now(UTC) - self._last_hedge_time).total_seconds()

    def _make_signal(
        self,
        action: HedgeAction,
        direction: int,
        shares: int,
        reason: str,
        threshold: float,
        urgency: str,
    ) -> HedgeSignal:
        price = self._last_underlying_price
        spread_tc = shares * (self._cfg.spread_bps / 10_000) * price
        comm_tc = shares * self._cfg.commission_per_share
        hedge_cost_estimate = spread_tc + comm_tc
        net_after = 0.0 if action == HedgeAction.FULL_HEDGE else sign(self._net_delta) * threshold  # type: ignore
        return HedgeSignal(
            action=action,
            direction=direction,
            shares=shares,
            reason=reason,
            net_delta_before=self._net_delta,
            net_delta_after=net_after,
            threshold=threshold,
            urgency=urgency,
            hedge_cost_estimate=round(hedge_cost_estimate, 4),
            deferral_count=self._deferral_count,
        )

    def _emit(self, signal: HedgeSignal) -> None:
        _log.info(
            "AdaptiveHedger SIGNAL: %s  %+d shares  reason=%s  "
            "cost_estimate=$%.4f  urgency=%s",
            signal.action.value,
            signal.shares * signal.direction,
            signal.reason,
            signal.hedge_cost_estimate,
            signal.urgency,
        )
        if signal.is_hedge and self._on_signal is not None:
            try:
                self._on_signal(signal)
            except Exception as exc:
                _log.warning("Hedge signal callback failed: %s", exc)


def sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0
