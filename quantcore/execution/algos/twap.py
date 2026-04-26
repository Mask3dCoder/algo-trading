"""Time-Weighted Average Price execution algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np


@dataclass
class TWAPConfig:
    """TWAP algorithm configuration."""
    window_minutes: int = 60
    num_slices: int = 10
    jitter_pct: float = 0.10  # ±10% random timing jitter
    min_slice_size: float = 1.0
    max_slice_size: float = 100_000.0


@dataclass
class TWAPSlice:
    slice_id: int
    target_time: datetime
    shares: float
    executed: bool = False
    fill_price: float | None = None
    fill_shares: float = 0.0


@dataclass
class TWAPResult:
    slices: list[TWAPSlice]
    total_shares: float
    total_executed: float
    avg_price: float | None = None
    benchmark_price: float | None = None
    performance_bps: float | None = None
    cancelled_remaining: float = 0.0


class TWAPAlgo:
    """TWAP execution: split parent order into equal time slices.

    Adds ±10% random jitter to each slice timing to avoid detection.
    """

    def __init__(self, config: TWAPConfig | None = None, seed: int = 42) -> None:
        self.config = config or TWAPConfig()
        self._rng = np.random.default_rng(seed)

    def compute_schedule(
        self,
        total_shares: float,
        start_time: datetime | None = None,
    ) -> list[TWAPSlice]:
        """Compute the TWAP slice schedule.

        Args:
            total_shares: Total shares to execute.
            start_time: When execution begins (default: now).

        Returns:
            List of TWAPSlice objects with target times and sizes.
        """
        cfg = self.config
        t0 = start_time or datetime.utcnow()
        window_ms = int(cfg.window_minutes * 60 * 1000)
        base_interval = window_ms / cfg.num_slices

        # Base slice size
        base_size = total_shares / cfg.num_slices

        slices = []
        for i in range(cfg.num_slices):
            # Random jitter on timing
            jitter = self._rng.uniform(-cfg.jitter_pct, cfg.jitter_pct)
            offset = base_interval * i + base_interval * jitter
            target_time = t0 + timedelta(milliseconds=max(0, offset))

            # Small randomization on size too
            size_jitter = self._rng.uniform(0.95, 1.05)
            size = base_size * size_jitter
            size = max(cfg.min_slice_size, min(cfg.max_slice_size, size))

            slices.append(TWAPSlice(
                slice_id=i,
                target_time=target_time,
                shares=round(size, 0),
            ))

        # Adjust last slice to hit exact total
        allocated = sum(s.shares for s in slices)
        if allocated != total_shares:
            slices[-1].shares += total_shares - allocated

        return slices

    def evaluate(
        self,
        slices: list[TWAPSlice],
        start_price: float,
        end_price: float,
    ) -> TWAPResult:
        """Evaluate TWAP performance.

        Args:
            slices: Completed slices with fills.
            start_price: Price at start of execution.
            end_price: Price at end of execution.

        Returns:
            TWAPResult with performance metrics.
        """
        total_shares = sum(s.shares for s in slices)
        total_exec = sum(s.fill_shares for s in slices)

        # Calculate average execution price
        if total_exec > 0:
            avg_price = sum(
                (s.fill_price or 0) * s.fill_shares for s in slices if s.fill_price is not None
            ) / total_exec
        else:
            avg_price = None

        # TWAP benchmark = simple average of start/end
        benchmark = (start_price + end_price) / 2.0

        # Performance vs benchmark in bps
        if avg_price is not None and benchmark > 0:
            perf_bps = (avg_price - benchmark) / benchmark * 10_000
        else:
            perf_bps = None

        return TWAPResult(
            slices=slices,
            total_shares=total_shares,
            total_executed=total_exec,
            avg_price=avg_price,
            benchmark_price=benchmark,
            performance_bps=perf_bps,
            cancelled_remaining=total_shares - total_exec,
        )
