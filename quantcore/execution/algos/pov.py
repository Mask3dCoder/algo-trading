"""Percentage-of-Volume (POV) execution algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class POVConfig:
    participation_rate: float = 0.05  # Trade 5% of real-time market volume
    urgency: float = 1.0  # Scale participation (0.5 = passive, 2.0 = aggressive)
    window_minutes: int = 60
    max_participation: float = 0.25  # Cap at 25%
    min_interval_seconds: int = 10  # Minimum time between slices


@dataclass
class POVSlice:
    slice_id: int
    target_time: datetime
    target_shares: float
    market_volume: float  # Volume observed in this interval
    executed_shares: float = 0.0
    fill_price: float | None = None
    participation_actual: float = 0.0


@dataclass
class POVResult:
    slices: list[POVSlice]
    total_shares: float
    total_executed: float
    target_participation: float
    actual_participation: float
    remaining: float = 0.0
    elapsed_minutes: float = 0.0


class POVAlgo:
    """POV: trade at a fixed percentage of real-time market volume.

    Monitors market volume and dynamically sizes child orders to maintain
    a constant participation rate.
    """

    def __init__(self, config: POVConfig | None = None) -> None:
        self.config = config or POVConfig()

    def compute_target(
        self,
        market_volume: float,
        remaining_shares: float,
        elapsed_fraction: float,
    ) -> float:
        """Compute target shares for the current interval.

        Args:
            market_volume: Volume traded in the market during this interval.
            remaining_shares: Shares still to be executed.
            elapsed_fraction: Fraction of time window elapsed.

        Returns:
            Target shares for this interval.
        """
        cfg = self.config
        rate = min(cfg.participation_rate * cfg.urgency, cfg.max_participation)

        # Primary: participation rate × market volume
        target = market_volume * rate

        # Also ensure we finish within the window
        if elapsed_fraction < 0.99:
            min_to_finish = remaining_shares / (1.0 - elapsed_fraction) * (cfg.min_interval_seconds / (cfg.window_minutes * 60))
            target = max(target, min_to_finish)
        else:
            target = remaining_shares

        return target

    def run(
        self,
        total_shares: float,
        market_volumes: list[float],  # Historical or real-time volume per interval
        start_time: datetime | None = None,
    ) -> POVResult:
        """Simulate a POV execution.

        Args:
            total_shares: Total order quantity.
            market_volumes: Per-interval market volumes.
            start_time: Start time.

        Returns:
            POVResult with execution details.
        """
        cfg = self.config
        t0 = start_time or datetime.utcnow()
        total_volume = sum(market_volumes)

        slices: list[POVSlice] = []
        remaining = total_shares
        total_executed = 0.0

        for i, mv in enumerate(market_volumes):
            elapsed = (i + 1) / len(market_volumes)
            target = self.compute_target(mv, remaining, elapsed)

            # Execute (in simulation, fill = target)
            filled = min(target, remaining)
            total_executed += filled
            remaining -= filled

            slice_time = t0 + timedelta(seconds=cfg.min_interval_seconds * i)
            actual_rate = filled / mv if mv > 0 else 0.0

            slices.append(POVSlice(
                slice_id=i,
                target_time=slice_time,
                target_shares=target,
                market_volume=mv,
                executed_shares=filled,
                participation_actual=actual_rate,
            ))

            if remaining <= 0:
                break

        return POVResult(
            slices=slices,
            total_shares=total_shares,
            total_executed=total_executed,
            target_participation=cfg.participation_rate,
            actual_participation=total_executed / total_volume if total_volume > 0 else 0.0,
            remaining=remaining,
            elapsed_minutes=cfg.window_minutes * (len(slices) / max(len(market_volumes), 1)),
        )
