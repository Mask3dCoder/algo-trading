"""Iceberg (reserve) order execution algorithm.

Shows only a small visible portion of the total order in the order book to
conceal the true order size. As visible slices get filled, the algorithm
replenishes from the hidden reserve. Used to minimize market impact and
avoid signaling large positions to other market participants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass
class IcebergConfig:
    """Iceberg algorithm configuration."""
    total_shares: float = 0.0
    visible_shares: float = 100.0  # Shares shown per slice
    max_visible_pct: float = 0.10  # Max visible as % of total
    min_visible: float = 1.0  # Minimum visible slice size
    replenish_delay_ms: int = 500  # Delay before refreshing visible
    price_limit: float | None = None  # Optional limit price cap
    randomize_visible: bool = True  # Randomize visible size within range
    random_range_pct: float = 0.20  # ±20% randomization on visible size


@dataclass
class IcebergSlice:
    """A single visible slice of the iceberg order."""
    slice_id: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    visible_shares: float = 0.0
    hidden_reserve: float = 0.0  # Remaining hidden after this slice
    filled_shares: float = 0.0
    fill_price: float | None = None
    is_active: bool = True


@dataclass
class IcebergResult:
    """Iceberg execution result."""
    slices: list[IcebergSlice]
    total_shares: float
    total_filled: float
    avg_fill_price: float | None = None
    visible_ratio: float = 0.0  # Avg visible / total per refresh
    hidden_ratio: float = 0.0  # Hidden / total
    total_elapsed_ms: float = 0.0
    cancelled_remaining: float = 0.0


class IcebergAlgo:
    """Iceberg execution: conceal large orders by showing small visible slices.

    The order book only shows `visible_shares` at a time. When those are filled,
    the algorithm replenishes from the hidden reserve after a configurable delay.
    This conceals the true order size and reduces predatory trading / front-running.

    Usage:
        algo = IcebergAlgo(IcebergConfig(total_shares=10000, visible_shares=200))
        schedule = algo.compute_schedule()
        for slice in schedule.slices:
            # Place visible slice, when filled call algo.replenish(slice_id)
    """

    def __init__(self, config: IcebergConfig | None = None, seed: int = 42) -> None:
        self.config = config or IcebergConfig()
        self._rng = np.random.default_rng(seed)
        self._order_book_visible: float = 0.0
        self._cumulative_hidden: float = 0.0
        self._slice_counter: int = 0

    def compute_schedule(self) -> IcebergResult:
        """Compute full iceberg schedule.

        Returns an IcebergResult with pre-computed slices. In practice,
        slices are replenished dynamically as fills arrive.

        Returns:
            IcebergResult with planned slices.
        """
        cfg = self.config
        remaining = cfg.total_shares
        slices: list[IcebergSlice] = []
        slice_id = 0

        while remaining > 0:
            visible = self._compute_visible_size(remaining)
            visible = min(visible, remaining)

            slices.append(IcebergSlice(
                slice_id=slice_id,
                visible_shares=visible,
                hidden_reserve=remaining - visible,
            ))

            remaining -= visible
            slice_id += 1

        return IcebergResult(
            slices=slices,
            total_shares=cfg.total_shares,
            total_filled=0.0,
        )

    def replenish(
        self,
        filled_slice_id: int,
        fill_price: float,
        filled_shares: float | None = None,
        market_conditions: dict[str, Any] | None = None,
    ) -> IcebergSlice | None:
        """Replenish a visible slice after it has been filled.

        Call this when a visible slice has been fully or partially executed.
        Returns the next visible slice, or None if the order is complete.

        Args:
            filled_slice_id: The slice that was filled.
            fill_price: Average fill price of the filled slice.
            filled_shares: Shares filled (defaults to visible size).
            market_conditions: Optional dict with market data for adaptive sizing.

        Returns:
            Next IcebergSlice, or None if exhausted.
        """
        cfg = self.config
        remaining = max(0.0, cfg.total_shares - self._cumulative_hidden - (filled_shares or 0))

        actual_filled = filled_shares or self._compute_visible_size(remaining)
        self._cumulative_hidden = min(cfg.total_shares, self._cumulative_hidden + actual_filled)

        remaining = max(0.0, cfg.total_shares - self._cumulative_hidden)

        if remaining < cfg.min_visible:
            return None

        # Adaptive sizing based on market conditions
        visible = self._compute_visible_size(remaining, market_conditions)

        self._slice_counter += 1
        self._order_book_visible = visible

        return IcebergSlice(
            slice_id=self._slice_counter,
            visible_shares=visible,
            hidden_reserve=remaining - visible,
        )

    def _compute_visible_size(
        self,
        remaining: float,
        market_conditions: dict[str, Any] | None = None,
    ) -> float:
        """Compute the visible slice size.

        Adjusts based on:
        - Base visible shares from config
        - Remaining shares
        - Market volatility (increase visible in calm, decrease in volatile)
        - Spread width (decrease visible when wide)
        """
        cfg = self.config
        base = min(cfg.visible_shares, remaining * cfg.max_visible_pct)
        base = max(cfg.min_visible, base)

        if not cfg.randomize_visible or self._rng is None:
            return base

        scale = 1.0

        # Adaptive: tighten visible when volatility is high
        if market_conditions:
            spread_bps = float(market_conditions.get("spread_bps", 0))
            volatility = float(market_conditions.get("volatility", 0))
            volume_ratio = float(market_conditions.get("volume_ratio", 1.0))

            if spread_bps > 50:
                scale *= 0.7  # Wide spread → show less
            elif spread_bps < 5:
                scale *= 1.1

            if volatility > 0.03:  # 3%+ vol
                scale *= 0.8
            elif volatility < 0.01:
                scale *= 1.05

            if volume_ratio < 0.5:
                scale *= 0.8  # Low volume → show less to avoid impact
            elif volume_ratio > 2.0:
                scale *= 1.15

        # Randomize within range
        jitter = self._rng.uniform(1.0 - cfg.random_range_pct, 1.0 + cfg.random_range_pct)
        visible = base * scale * jitter

        return max(cfg.min_visible, min(visible, remaining, cfg.visible_shares * 2.0))

    def evaluate(
        self,
        filled_slices: list[IcebergSlice],
    ) -> IcebergResult:
        """Evaluate iceberg execution performance.

        Args:
            filled_slices: Slices that have been filled.

        Returns:
            IcebergResult with performance metrics.
        """
        cfg = self.config
        total_filled = sum(s.filled_shares for s in filled_slices)
        total_shares = cfg.total_shares

        if total_filled > 0:
            prices = [(s.fill_price or 0) * s.filled_shares for s in filled_slices if s.fill_price is not None]
            weights = [s.filled_shares for s in filled_slices if s.fill_price is not None]
            avg_price = float(np.average(prices, weights=weights)) if sum(weights) > 0 else None
        else:
            avg_price = None

        visible_ratio = sum(s.visible_shares for s in filled_slices) / total_shares if total_shares > 0 else 0.0

        return IcebergResult(
            slices=filled_slices,
            total_shares=total_shares,
            total_filled=total_filled,
            avg_fill_price=avg_price,
            visible_ratio=visible_ratio,
            hidden_ratio=1.0 - visible_ratio if total_shares > 0 else 0.0,
            cancelled_remaining=total_shares - total_filled,
        )

    @property
    def remaining_hidden(self) -> float:
        return max(0.0, self.config.total_shares - self._cumulative_hidden - self._order_book_visible)

    def cancel_hidden(self) -> float:
        """Cancel all hidden shares. Returns the number of shares cancelled."""
        hidden = self.remaining_hidden
        self._cumulative_hidden = self.config.total_shares
        self._order_book_visible = 0.0
        return hidden
