"""Market impact configuration and computation for backtest fills.

Wires execution/impact.py models into the backtest engine's FillSimulator.
All configs have defaults so existing code is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from quantcore.execution.impact import predict_cost as _predict_cost


@dataclass
class MarketImpactConfig:
    """Configuration for market impact cost modeling in backtest fills.

    When None (default), the FillSimulator applies only fixed slippage.
    """

    model_type: Literal["sqrt", "linear"] = "sqrt"
    """Impact model: 'sqrt' (Almgren et al.) or 'linear' (Kyle's lambda)."""

    eta: float = 0.0625
    """Temporary impact coefficient (Almgren-Chriss parameterization)."""

    gamma: float = 0.314
    """Permanent impact coefficient (Almgren-Chriss parameterization)."""

    sigma: float | None = None
    """Daily volatility. If None, estimated from data."""

    adv: float | None = None
    """Average daily volume in shares. If None, estimated from data."""

    kyle_lambda: float | None = None
    """Kyle's lambda for linear model. If None, falls back to sqrt."""

    half_spread_bps: float = 5.0
    """Assumed half-spread in basis points (default 5 bps)."""


@dataclass
class VolumeFilterConfig:
    """Volume-based fill constraints to prevent unrealistic executions.

    When None (default), no volume filtering is applied.
    """

    max_participation_rate: float = 0.20
    """Maximum fraction of bar volume an order can represent (default 20%)."""

    min_volume: int = 100
    """Minimum bar volume required to consider any fill (default 100 shares)."""

    partial_fill_enabled: bool = True
    """If True, fills exceeding max_participation_rate are clipped; if False, rejected."""


@dataclass
class ImpactResult:
    """Result of a market impact calculation for a single fill."""

    temp_impact_bps: float = 0.0
    """Temporary (transitory) price impact in basis points."""

    perm_impact_bps: float = 0.0
    """Permanent price impact in basis points (persists for subsequent fills)."""

    total_cost_dollars: float = 0.0
    """Total dollar cost of spread + impact for this fill."""

    total_cost_bps: float = 0.0
    """Total cost in basis points of order value."""

    participation_rate: float = 0.0
    """Order quantity / bar volume ratio."""

    is_clipped: bool = False
    """Whether the fill quantity was reduced due to volume filter."""


class MarketImpactEngine:
    """Computes market impact for backtest fills using execution/impact.py models.

    Handles:
    - ADV and sigma estimation from data
    - Volume-based fill filtering
    - Temporary and permanent price impact
    - Almgren-Chriss and sqrt model computations
    """

    def __init__(
        self,
        config: MarketImpactConfig,
        volume_filter: VolumeFilterConfig | None = None,
    ) -> None:
        self.config = config
        self.volume_filter = volume_filter
        self._adv: float | None = config.adv
        self._sigma: float | None = config.sigma

    def set_data(self, data: pd.DataFrame) -> None:
        """Estimate ADV and sigma from historical data using ONLY the initial window.

        Uses the first 60 trading days to estimate ADV and sigma, avoiding
        look-ahead bias.  These are static estimates used throughout the backtest.
        For a production system, these should be updated with an expanding window
        at each bar, but the static estimate is sufficient for impact modeling
        and avoids the complexity of per-bar recalculation.

        data: DataFrame with columns [symbol, timestamp, open, high, low, close, volume].
        """
        lookback = 60  # Use first 60 bars only (no future data)

        if self._adv is None and "volume" in data.columns:
            sym = data["symbol"].iloc[0] if "symbol" in data.columns else None
            vol_series = (
                data.loc[data["symbol"] == sym, "volume"]
                if sym
                else data["volume"]
            )
            # Use only the first `lookback` bars
            initial_vol = vol_series.iloc[:lookback]
            self._adv = float(np.mean(initial_vol)) if len(initial_vol) > 0 else 1_000_000.0

        if self._sigma is None and "close" in data.columns:
            sym = data["symbol"].iloc[0] if "symbol" in data.columns else None
            close_series = (
                data.loc[data["symbol"] == sym, "close"].values.astype(float)
                if sym
                else data["close"].values.astype(float)
            )
            # Use only the first `lookback` bars
            initial_close = close_series[:lookback]
            if len(initial_close) > 1:
                returns = np.diff(initial_close) / initial_close[:-1]
                self._sigma = float(np.std(returns, ddof=1))

    def check_volume_filter(
        self,
        order_qty: float,
        bar_volume: int,
    ) -> tuple[float, bool]:
        """Check volume filter and return (adjusted_qty, is_clipped).

        Returns the order quantity (possibly reduced) and whether it was clipped.
        """
        if self.volume_filter is None:
            return order_qty, False

        if bar_volume < self.volume_filter.min_volume:
            return 0.0, False  # Cannot fill at all

        participation = order_qty / bar_volume
        if participation <= self.volume_filter.max_participation_rate:
            return order_qty, False

        if self.volume_filter.partial_fill_enabled:
            max_qty = bar_volume * self.volume_filter.max_participation_rate
            return float(max_qty), True

        return 0.0, False  # Rejected

    def estimate_impact(
        self,
        shares: float,
        price: float,
        bar_volume: int,
    ) -> ImpactResult:
        """Compute market impact for a single fill.

        Args:
            shares: Number of shares in the fill (possibly clipped by volume filter).
            price: Current market price (bar open).
            bar_volume: Volume of the current bar.

        Returns:
            ImpactResult with temporary impact, permanent impact, and cost breakdown.
        """
        adv = self._adv or 1_000_000.0
        sigma = self._sigma or 0.02

        participation = shares / bar_volume if bar_volume > 0 else 0.0

        if shares <= 0 or price <= 0:
            return ImpactResult(participation_rate=participation)

        # Use predict_cost from execution/impact.py
        cost = _predict_cost(
            shares=shares,
            price=price,
            adv=adv,
            sigma=sigma,
            model=self.config.model_type,
            kyle_lambda=self.config.kyle_lambda,
            half_spread_bps=self.config.half_spread_bps,
        )

        # Split into temporary and permanent components
        # For sqrt model: total impact = cost.impact_cost_bps
        # Temporary impact is ~half for the fill, permanent affects later fills
        temp_impact_bps = cost.impact_cost_bps * 0.5
        perm_impact_bps = cost.impact_cost_bps * 0.5

        return ImpactResult(
            temp_impact_bps=temp_impact_bps,
            perm_impact_bps=perm_impact_bps,
            total_cost_dollars=cost.total_dollars,
            total_cost_bps=cost.total_bps,
            participation_rate=participation,
            is_clipped=False,
        )

    @property
    def adv(self) -> float | None:
        return self._adv

    @property
    def sigma(self) -> float | None:
        return self._sigma
