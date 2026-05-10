"""Event Chaos Generator (ECG) + Failure Injection Engine + Replay Engine.

Three layers:
  1. EventChaosGenerator — produces deterministic/randomized event streams
  2. FailureInjector — applies scripted or probabilistic failures
  3. ReplayEngine — orchestrates replay, fuzz, and adversarial modes

Output is a time-ordered or intentionally corrupted event stream consumed
by the test harness orchestrator.
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable

from quantcore.core.types import Fill, Order, OrderType, Side


# ===========================================================================
# Event types
# ===========================================================================


class EventType(str, Enum):
    # Market events
    MARKET_DATA = "market_data"
    PRICE_SPIKE = "price_spike"
    LIQUIDITY_DROP = "liquidity_drop"

    # Broker events
    BROKER_DISCONNECT = "broker_disconnect"
    BROKER_RECONNECT = "broker_reconnect"

    # Fill events
    FILL_GENERATED = "fill_generated"
    FILL_DELIVERED = "fill_delivered"
    FILL_DROPPED = "fill_dropped"
    FILL_DUPLICATED = "fill_duplicated"
    FILL_REORDERED = "fill_reordered"
    FILL_DELAYED = "fill_delayed"

    # Order events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Settlement events
    SETTLEMENT_CYCLE = "settlement_cycle"
    SETTLEMENT_SHIFT = "settlement_shift"
    SETTLEMENT_PROCESSED = "settlement_processed"

    # System events
    TICK = "tick"
    RESTART = "restart"
    RECONCILIATION = "reconciliation"
    ASSERT = "assert"  # validation check point


@dataclass(order=True)
class ChaosEvent:
    """A single event in the chaos stream. Sortable by timestamp."""
    timestamp: float = field(compare=True)
    event_type: EventType = field(compare=False)
    seq: int = field(compare=False, default=0)
    parameters: dict[str, Any] = field(compare=False, default_factory=dict)

    def __repr__(self) -> str:
        return f"[T={self.timestamp:.3f}] #{self.seq} {self.event_type.value} {self.parameters}"


@dataclass
class ChaosConfig:
    """Configuration for chaos generation."""
    # General
    seed: int = 42
    duration_seconds: float = 60.0
    tick_interval: float = 0.1  # 100ms ticks

    # Market chaos
    price_spike_prob: float = 0.01
    price_spike_magnitude: tuple[float, float] = (0.05, 0.50)  # 5-50% spikes
    liquidity_drop_prob: float = 0.005

    # Broker chaos
    disconnect_prob: float = 0.002
    disconnect_duration: tuple[float, float] = (2.0, 15.0)
    reconnect_prob: float = 0.01

    # Fill chaos
    fill_drop_prob: float = 0.05       # 5% of fills dropped
    fill_dup_prob: float = 0.03        # 3% of fills duplicated
    fill_delay_prob: float = 0.10      # 10% of fills delayed
    fill_delay_range: tuple[float, float] = (0.5, 5.0)
    fill_reorder_prob: float = 0.02
    partial_fill_prob: float = 0.10
    partial_fill_range: tuple[float, float] = (0.1, 0.9)

    # Settlement chaos
    settlement_shift_prob: float = 0.001
    settlement_shift_cycles: list[int] = field(default_factory=lambda: [0, 1, 3, 5])

    # Order chaos
    order_reject_prob: float = 0.02

    # Adversarial mode overrides
    adversarial_mode: bool = False

    # Replay
    deterministic: bool = True


# ===========================================================================
# Event Chaos Generator
# ===========================================================================


class EventChaosGenerator:
    """Generates deterministic or randomized chaos event streams.

    Supports:
      - Scripted: explicit event lists
      - Probabilistic: random events based on config probabilities
      - Burst: high-frequency chaos windows
      - Adversarial: maximize divergence attempts
    """

    def __init__(self, config: ChaosConfig | None = None) -> None:
        self.config = config or ChaosConfig()
        self._rng = random.Random(self.config.seed)
        self._seq: int = 0
        self._event_log: list[ChaosEvent] = []
        self._disconnected: bool = False
        self._burst_active: bool = False
        self._symbols: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        self._base_prices: dict[str, float] = {
            "AAPL": 150.0, "MSFT": 350.0, "GOOGL": 140.0,
            "AMZN": 130.0, "TSLA": 200.0,
        }
        self._open_orders: list[str] = []

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _make_event(self, event_type: EventType, **params: Any) -> ChaosEvent:
        return ChaosEvent(
            timestamp=0.0,  # will be set by script builder
            event_type=event_type,
            seq=self._next_seq(),
            parameters=params,
        )

    # ------------------------------------------------------------------
    # Scripted event stream builders
    # ------------------------------------------------------------------

    def build_market_stream(
        self,
        symbols: list[str] | None = None,
        n_ticks: int = 100,
        tick_interval: float = 0.1,
    ) -> list[ChaosEvent]:
        """Build a clean market data event stream (no failures)."""
        syms = symbols or self._symbols
        events: list[ChaosEvent] = []
        t = 0.0
        for i in range(n_ticks):
            t = i * tick_interval
            for sym in syms:
                price = self._base_prices[sym] * (1.0 + self._rng.uniform(-0.005, 0.005))
                events.append(ChaosEvent(
                    timestamp=t,
                    event_type=EventType.MARKET_DATA,
                    seq=self._next_seq(),
                    parameters={"symbol": sym, "price": round(price, 2)},
                ))
        return events

    def build_fill_stream(
        self,
        order_ids: list[str],
        symbols: list[str] | None = None,
        base_time: float = 0.0,
    ) -> list[ChaosEvent]:
        """Build a clean fill event stream."""
        syms = symbols or self._symbols
        events: list[ChaosEvent] = []
        for i, oid in enumerate(order_ids):
            sym = syms[i % len(syms)]
            events.append(ChaosEvent(
                timestamp=base_time + i * 0.5,
                event_type=EventType.FILL_GENERATED,
                seq=self._next_seq(),
                parameters={
                    "order_id": oid,
                    "symbol": sym,
                    "price": round(self._base_prices[sym], 2),
                    "quantity": 100,
                },
            ))
        return events

    def build_scripted(self, script: list[dict[str, Any]]) -> list[ChaosEvent]:
        """Build a scripted event stream from a list of dicts.

        Each dict: {"type": EventType, "timestamp": float, "params": {...}}
        """
        events: list[ChaosEvent] = []
        for entry in script:
            events.append(ChaosEvent(
                timestamp=entry["timestamp"],
                event_type=entry["type"],
                seq=self._next_seq(),
                parameters=entry.get("params", {}),
            ))
        return events

    # ------------------------------------------------------------------
    # Probabilistic event injection
    # ------------------------------------------------------------------

    def inject_failures(
        self,
        base_events: list[ChaosEvent],
    ) -> list[ChaosEvent]:
        """Inject probabilistic failures into a clean event stream.

        Returns a new event list with failure events injected.
        """
        result: list[ChaosEvent] = []
        self._rng = random.Random(self.config.seed)  # deterministic

        pending_fills: list[ChaosEvent] = []

        for event in base_events:
            result.append(event)

            # Inject failures based on event type
            if event.event_type == EventType.MARKET_DATA:
                # Price spikes
                if self._rng.random() < self.config.price_spike_prob:
                    sym = event.parameters.get("symbol", "AAPL")
                    base = event.parameters.get("price", 150.0)
                    magnitude = self._rng.uniform(*self.config.price_spike_magnitude)
                    if self._rng.random() < 0.5:
                        spiked = base * (1.0 + magnitude)
                    else:
                        spiked = base * (1.0 - magnitude)
                    result.append(ChaosEvent(
                        timestamp=event.timestamp + 0.001,
                        event_type=EventType.PRICE_SPIKE,
                        seq=self._next_seq(),
                        parameters={"symbol": sym, "price": round(spiked, 2), "magnitude": round(magnitude, 3)},
                    ))

                # Liquidity drops
                if self._rng.random() < self.config.liquidity_drop_prob:
                    result.append(ChaosEvent(
                        timestamp=event.timestamp + 0.001,
                        event_type=EventType.LIQUIDITY_DROP,
                        seq=self._next_seq(),
                        parameters={"symbol": event.parameters.get("symbol", "AAPL"), "factor": round(self._rng.uniform(0.01, 0.5), 3)},
                    ))

                # Adversarial: extra chaos on every event
                if self.config.adversarial_mode:
                    if self._rng.random() < 0.3:  # 30% extra chaos in adversarial
                        result.append(ChaosEvent(
                            timestamp=event.timestamp + 0.0005,
                            event_type=EventType.PRICE_SPIKE,
                            seq=self._next_seq(),
                            parameters={"symbol": event.parameters.get("symbol", "AAPL"), "price": round(event.parameters.get("price", 150.0) * self._rng.uniform(0.5, 1.5), 2), "magnitude": 0.5},
                        ))

            elif event.event_type == EventType.FILL_GENERATED:
                # Drop fills
                if self._rng.random() < self.config.fill_drop_prob:
                    dropped = copy.deepcopy(event)
                    dropped.event_type = EventType.FILL_DROPPED
                    dropped.seq = self._next_seq()
                    result.append(dropped)

                # Delay fills
                if self._rng.random() < self.config.fill_delay_prob:
                    delay = self._rng.uniform(*self.config.fill_delay_range)
                    delayed = copy.deepcopy(event)
                    delayed.event_type = EventType.FILL_DELAYED
                    delayed.seq = self._next_seq()
                    delayed.parameters["delay"] = round(delay, 3)
                    result.append(delayed)

                # Duplicate fills
                if self._rng.random() < self.config.fill_dup_prob:
                    dup = copy.deepcopy(event)
                    dup.event_type = EventType.FILL_DUPLICATED
                    dup.seq = self._next_seq()
                    dup.timestamp += 0.1  # slight delay on duplicate
                    result.append(dup)

                # Reorder: queue and emit later
                if self._rng.random() < self.config.fill_reorder_prob:
                    reorder = copy.deepcopy(event)
                    reorder.event_type = EventType.FILL_REORDERED
                    reorder.seq = self._next_seq()
                    reorder.timestamp += self._rng.uniform(0.5, 2.0)
                    result.append(reorder)

                # Adversarial: drop more fills
                if self.config.adversarial_mode and self._rng.random() < 0.2:
                    dropped = copy.deepcopy(event)
                    dropped.event_type = EventType.FILL_DROPPED
                    dropped.seq = self._next_seq()
                    result.append(dropped)

            elif event.event_type == EventType.BROKER_DISCONNECT:
                self._disconnected = True

            elif event.event_type == EventType.BROKER_RECONNECT:
                self._disconnected = False

            # Inject disconnect/reconnect
            if event.event_type in (EventType.MARKET_DATA, EventType.TICK):
                if not self._disconnected and self._rng.random() < self.config.disconnect_prob:
                    duration = self._rng.uniform(*self.config.disconnect_duration)
                    result.append(ChaosEvent(
                        timestamp=event.timestamp,
                        event_type=EventType.BROKER_DISCONNECT,
                        seq=self._next_seq(),
                        parameters={"duration": round(duration, 2)},
                    ))
                    self._disconnected = True
                    # Schedule reconnect
                    result.append(ChaosEvent(
                        timestamp=event.timestamp + duration,
                        event_type=EventType.BROKER_RECONNECT,
                        seq=self._next_seq(),
                        parameters={},
                    ))

                # Adversarial: extra disconnects
                if self.config.adversarial_mode and self._rng.random() < 0.05:
                    duration = self._rng.uniform(1.0, 10.0)
                    result.append(ChaosEvent(
                        timestamp=event.timestamp,
                        event_type=EventType.BROKER_DISCONNECT,
                        seq=self._next_seq(),
                        parameters={"duration": round(duration, 2)},
                    ))
                    result.append(ChaosEvent(
                        timestamp=event.timestamp + duration,
                        event_type=EventType.BROKER_RECONNECT,
                        seq=self._next_seq(),
                        parameters={},
                    ))

            # Settlement shifts
            if event.event_type in (EventType.FILL_GENERATED, EventType.TICK):
                if self._rng.random() < self.config.settlement_shift_prob:
                    cycle = self._rng.choice(self.config.settlement_shift_cycles)
                    result.append(ChaosEvent(
                        timestamp=event.timestamp,
                        event_type=EventType.SETTLEMENT_SHIFT,
                        seq=self._next_seq(),
                        parameters={"asset_class": "equity", "new_cycle": cycle},
                    ))

        return result

    # ------------------------------------------------------------------
    # Burst mode
    # ------------------------------------------------------------------

    def build_burst(
        self,
        burst_start: float,
        burst_duration: float,
        failure_rate: float = 0.5,
    ) -> list[ChaosEvent]:
        """Build a burst of failure events (high-frequency chaos window).

        During a burst, normal probabilities are overridden with
        elevated failure rates to test system stability under duress.
        """
        events: list[ChaosEvent] = []
        t = burst_start
        while t < burst_start + burst_duration:
            if self._rng.random() < failure_rate:
                # Random failure type
                ftype = self._rng.choice([
                    EventType.FILL_DROPPED,
                    EventType.BROKER_DISCONNECT,
                    EventType.PRICE_SPIKE,
                    EventType.FILL_DUPLICATED,
                ])
                params: dict[str, Any] = {}
                if ftype == EventType.PRICE_SPIKE:
                    sym = self._rng.choice(self._symbols)
                    params = {"symbol": sym, "price": round(self._base_prices[sym] * self._rng.uniform(0.5, 1.5), 2), "magnitude": 0.5}
                elif ftype == EventType.BROKER_DISCONNECT:
                    params = {"duration": round(self._rng.uniform(1.0, 5.0), 2)}
                    # Schedule reconnect
                    events.append(ChaosEvent(
                        timestamp=t + params["duration"],
                        event_type=EventType.BROKER_RECONNECT,
                        seq=self._next_seq(),
                        parameters={},
                    ))
                elif ftype == EventType.FILL_DROPPED:
                    params = {"order_id": f"burst-{self._next_seq()}"}
                elif ftype == EventType.FILL_DUPLICATED:
                    params = {"order_id": f"burst-{self._next_seq()}"}

                events.append(ChaosEvent(
                    timestamp=t,
                    event_type=ftype,
                    seq=self._next_seq(),
                    parameters=params,
                ))
            t += self._rng.uniform(0.05, 0.5)
        return events

    # ------------------------------------------------------------------
    # Determinism support
    # ------------------------------------------------------------------

    def deterministic_hash(self, events: list[ChaosEvent]) -> str:
        """Compute a deterministic hash of an event sequence."""
        h = hashlib.sha256()
        for e in sorted(events, key=lambda x: (x.timestamp, x.seq)):
            h.update(f"{e.timestamp}:{e.seq}:{e.event_type.value}:{e.parameters}\n".encode())
        return h.hexdigest()


# ===========================================================================
# Failure Injection Engine
# ===========================================================================


@dataclass
class FailureEvent:
    """A single injected failure."""
    type: str
    timestamp: float
    parameters: dict[str, Any] = field(default_factory=dict)


class FailureInjector:
    """Deterministic failure injector.

    Supports:
      - Probabilistic injection (randomized but seed-controlled)
      - Scripted injection (explicit failure list)
      - Burst modes (high-frequency chaos windows)
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._injected: list[FailureEvent] = []

    def inject_probabilistic(
        self,
        events: list[ChaosEvent],
        probabilities: dict[EventType, float] | None = None,
    ) -> list[ChaosEvent]:
        """Inject failures into an event stream based on probabilities."""
        ecg = EventChaosGenerator(ChaosConfig(seed=self._rng.randint(0, 2**31)))
        if probabilities:
            for et, prob in probabilities.items():
                if et == EventType.FILL_DROPPED:
                    ecg.config.fill_drop_prob = prob
                elif et == EventType.FILL_DUPLICATED:
                    ecg.config.fill_dup_prob = prob
        return ecg.inject_failures(events)

    def inject_scripted(
        self,
        events: list[ChaosEvent],
        failures: list[FailureEvent],
    ) -> list[ChaosEvent]:
        """Inject specific scripted failures into an event stream."""
        result = list(events)
        for failure in failures:
            result.append(ChaosEvent(
                timestamp=failure.timestamp,
                event_type=EventType(failure.type),
                seq=0,
                parameters=failure.parameters,
            ))
        result.sort(key=lambda e: (e.timestamp, e.seq))
        for i, e in enumerate(result):
            e.seq = i + 1
        self._injected.extend(failures)
        return result

    def inject_burst(
        self,
        events: list[ChaosEvent],
        burst_start: float,
        burst_duration: float,
        failure_rate: float = 0.5,
    ) -> list[ChaosEvent]:
        """Inject a burst of failures during a time window."""
        ecg = EventChaosGenerator(ChaosConfig(seed=self._rng.randint(0, 2**31)))
        burst_events = ecg.build_burst(burst_start, burst_duration, failure_rate)
        result = list(events) + burst_events
        result.sort(key=lambda e: (e.timestamp, e.seq))
        for i, e in enumerate(result):
            e.seq = i + 1
        for be in burst_events:
            self._injected.append(FailureEvent(
                type=be.event_type.value,
                timestamp=be.timestamp,
                parameters=be.parameters,
            ))
        return result


# ===========================================================================
# Replay Engine
# ===========================================================================


class ReplayMode(str, Enum):
    DETERMINISTIC = "deterministic"  # fixed seed, identical every run
    FUZZ = "fuzz"                    # randomized, different each run
    ADVERSARIAL = "adversarial"      # maximize divergence attempts


@dataclass
class RunResult:
    """Result of a single replay run."""
    mode: ReplayMode
    seed: int
    n_events: int
    n_failures_injected: int
    sis_score: float
    sis_components: dict[str, float]
    divergences: list[dict[str, Any]]
    first_failure: dict[str, Any] | None
    state_drift_timeline: list[dict[str, Any]]
    risk_breach_timeline: list[dict[str, Any]]
    reconciliation_summary: str
    duration_seconds: float
    deterministic_hash: str = ""


class ReplayEngine:
    """Deterministic replay engine for adversarial testing.

    Three modes:
      - DETERMINISTIC: fixed seed, byte-for-byte identical every run
      - FUZZ: randomized failure injections, different each run
      - ADVERSARIAL: maximizes divergence attempts to break reconciliation

    Usage:
        engine = ReplayEngine(sut, broker_emulator, oracle)
        result = engine.run_deterministic(seed=42, n_ticks=1000)
        result = engine.run_fuzz(n_runs=100)
        result = engine.run_adversarial(n_ticks=5000)
    """

    def __init__(self, config: ChaosConfig | None = None) -> None:
        self.config = config or ChaosConfig()

    def build_event_stream(
        self,
        mode: ReplayMode,
        seed: int = 42,
        n_ticks: int = 100,
        symbols: list[str] | None = None,
        script: list[dict[str, Any]] | None = None,
    ) -> list[ChaosEvent]:
        """Build event stream for a given mode."""
        cfg = copy.deepcopy(self.config)
        cfg.seed = seed
        cfg.adversarial_mode = (mode == ReplayMode.ADVERSARIAL)

        if mode == ReplayMode.DETERMINISTIC:
            cfg.deterministic = True

        ecg = EventChaosGenerator(cfg)

        if script:
            events = ecg.build_scripted(script)
        else:
            events = ecg.build_market_stream(symbols, n_ticks, cfg.tick_interval)

        # Inject failures
        events = ecg.inject_failures(events)
        events.sort(key=lambda e: (e.timestamp, e.seq))
        for i, e in enumerate(events):
            e.seq = i + 1

        return events

    def compute_run_hash(self, events: list[ChaosEvent]) -> str:
        ecg = EventChaosGenerator()
        return ecg.deterministic_hash(events)

    def compute_sis(
        self,
        total_events: int,
        divergences: list[dict[str, Any]],
        component_counts: dict[str, tuple[int, int]],
    ) -> tuple[float, dict[str, float]]:
        """Compute System Integrity Score.

        SIS = correct_events / total_events

        Tracked per component: cash, position, order, risk.
        """
        n_divergences = len(divergences)
        correct = max(0, total_events - n_divergences)
        overall = correct / total_events if total_events > 0 else 1.0

        components = {}
        for component, (correct_count, total_count) in component_counts.items():
            components[component] = correct_count / total_count if total_count > 0 else 1.0

        return overall, components
