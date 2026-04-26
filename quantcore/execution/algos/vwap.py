"""Volume-Weighted Average Price execution algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass
class VWAPConfig:
    window_minutes: int = 60  # for full-day: 390
    num_slices: int = 12
    min_slice_pct: float = 0.01
    max_slice_pct: float = 0.50


# Standard US equity volume profile (30-min buckets, 13 buckets = 6.5h)
DEFAULT_VOLUME_PROFILE: dict[str, float] = {
    "09:30": 0.09, "10:00": 0.065, "10:30": 0.055,
    "11:00": 0.05, "11:30": 0.045, "12:00": 0.04,
    "12:30": 0.04, "13:00": 0.04, "13:30": 0.04,
    "14:00": 0.05, "14:30": 0.06, "15:00": 0.085,
    "15:30": 0.15, "16:00": 0.14,
}

# Standard crypto volume profile (24h, 2h buckets)
CRYPTO_VOLUME_PROFILE: dict[str, float] = {
    "00:00": 0.06, "02:00": 0.05, "04:00": 0.04,
    "06:00": 0.05, "08:00": 0.08, "10:00": 0.11,
    "12:00": 0.12, "14:00": 0.12, "16:00": 0.11,
    "18:00": 0.09, "20:00": 0.08, "22:00": 0.07,
}


@dataclass
class VWAPSlice:
    slice_id: int
    target_time: datetime
    shares: float
    volume_pct: float  # Expected % of period volume
    executed: bool = False
    fill_price: float | None = None
    fill_shares: float = 0.0


@dataclass
class VWAPResult:
    slices: list[VWAPSlice]
    total_shares: float
    total_executed: float
    vwap_fill: float | None = None
    vwap_market: float | None = None
    performance_bps: float | None = None
    remaining: float = 0.0


class VWAPAlgo:
    """VWAP execution: size child orders proportional to historical volume profile."""

    def __init__(
        self,
        config: VWAPConfig | None = None,
        volume_profile: dict[str, float] | None = None,
    ) -> None:
        self.config = config or VWAPConfig()
        self.volume_profile = volume_profile or DEFAULT_VOLUME_PROFILE

    def load_volume_profile(self, symbol: str) -> dict[str, float]:
        """Load volume profile for a symbol."""
        # In production: load from DB/TimescaleDB
        return self.volume_profile

    def compute_schedule(
        self,
        total_shares: float,
        symbol: str = "",
        start_time: datetime | None = None,
    ) -> list[VWAPSlice]:
        """Compute VWAP slice schedule based on volume profile.

        Args:
            total_shares: Total shares to execute.
            symbol: Trading symbol.
            start_time: Execution start time.

        Returns:
            List of VWAPSlice objects.
        """
        cfg = self.config
        profile = self.load_volume_profile(symbol)
        t0 = start_time or datetime.utcnow()
        window_ms = cfg.window_minutes * 60 * 1000
        interval_ms = window_ms / cfg.num_slices

        # Normalize profile bucket keys to match our window
        buckets = list(profile.keys())
        bucket_values = list(profile.values())

        slices = []
        total_allocated = 0.0

        for i in range(cfg.num_slices):
            # Map slice to volume profile bucket
            bucket_idx = i % len(buckets)
            vol_pct = bucket_values[bucket_idx]

            # Clamp
            vol_pct = max(cfg.min_slice_pct, min(cfg.max_slice_pct, vol_pct))

            target_time = t0 + timedelta(milliseconds=interval_ms * i)

            slices.append(VWAPSlice(
                slice_id=i,
                target_time=target_time,
                shares=0.0,  # Will size based on remaining
                volume_pct=vol_pct,
            ))

        # Size slices proportionally
        remaining = total_shares
        for i, slc in enumerate(slices):
            remaining_slices = len(slices) - i
            if remaining_slices == 1:
                slc.shares = remaining
            else:
                slc.shares = round(remaining * slc.volume_pct, 0)
                remaining -= slc.shares

        return slices

    def adjust_profile(
        self,
        slices: list[VWAPSlice],
        actual_volume: float,
        expected_volume: float,
        current_idx: int,
    ) -> list[VWAPSlice]:
        """Adjust remaining slices based on observed volume deviations."""
        if expected_volume <= 0 or current_idx >= len(slices) - 1:
            return slices

        vol_ratio = actual_volume / expected_volume
        remaining = sum(s.shares - s.fill_shares for s in slices[current_idx + 1:])

        for i in range(current_idx + 1, len(slices)):
            if i == len(slices) - 1:
                slices[i].shares = remaining
            else:
                adj = slices[i].shares * vol_ratio
                slices[i].shares = max(1, round(adj, 0))
                remaining -= slices[i].shares

        return slices

    def evaluate(
        self,
        slices: list[VWAPSlice],
        market_vwap: float,
    ) -> VWAPResult:
        """Evaluate VWAP execution performance."""
        total_shares = sum(s.shares for s in slices)
        total_exec = sum(s.fill_shares for s in slices)

        if total_exec > 0:
            vwap_fill = sum(
                (s.fill_price or 0) * s.fill_shares for s in slices if s.fill_price is not None
            ) / total_exec
        else:
            vwap_fill = None

        if vwap_fill is not None and market_vwap > 0:
            perf_bps = (vwap_fill - market_vwap) / market_vwap * 10_000
        else:
            perf_bps = None

        return VWAPResult(
            slices=slices,
            total_shares=total_shares,
            total_executed=total_exec,
            vwap_fill=vwap_fill,
            vwap_market=market_vwap,
            performance_bps=perf_bps,
            remaining=total_shares - total_exec,
        )
