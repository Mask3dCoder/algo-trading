"""Pydantic schema for strategy factory YAML configuration."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SignalSpec(BaseModel):
    """Signal generation specification.

    Attributes:
        formula: Signal function call, e.g. "momentum(12)", "zscore(20)",
                 "moving_average_cross(50,200)", "factor_loading(HML)".
        rank: Cross-sectional ranking method, e.g.
              "top_minus_bottom_quintile", "zscore_long_short(2.0)".
        parameters: Additional parameters passed to the signal function.
    """

    formula: str
    rank: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class FilterSpec(BaseModel):
    """Pre-signal filters.

    Attributes:
        min_volume: Minimum average daily volume (USD).
        min_price: Minimum price per share.
        max_price: Maximum price per share.
        exclude_symbols: Symbols to always exclude.
        include_symbols: If set, only trade these symbols.
    """

    min_volume: float = 0.0
    min_price: float = 0.0
    max_price: float = float("inf")
    exclude_symbols: list[str] = Field(default_factory=list)
    include_symbols: Optional[list[str]] = None


class ExecutionSpec(BaseModel):
    """Execution parameters.

    Attributes:
        tc_bps: Transaction cost in basis points per unit turnover.
        slippage_bps: Slippage in basis points.
        max_participation: Maximum participation rate (fraction of ADV).
    """

    tc_bps: float = 5.0
    slippage_bps: float = 2.0
    max_participation: float = 0.05


class RegimeFilterSpec(BaseModel):
    """FRED-based regime filter specification.

    Attributes:
        enabled: Whether to apply the regime filter.
        signal: Regime signal name, e.g. "yield_curve_regime",
                "financial_conditions", "economic_momentum".
        threshold: Signal threshold; only trade when signal > threshold.
        inverse: If True, only trade when signal < threshold.
    """

    enabled: bool = True
    signal: str = "yield_curve_regime"
    threshold: float = 0.0
    inverse: bool = False


class StrategyFactoryConfig(BaseModel):
    """Top-level strategy configuration loaded from YAML.

    This is the schema for files in quantcore/strategy_factory/configs/.
    """

    name: str
    type: Literal[
        "time_series",
        "cross_sectional",
        "mean_reversion",
        "pairs",
        "factor",
        "arbitrage",
        "event_driven",
        "hft",
    ]
    universe: str
    universe_tickers: Optional[list[str]] = None
    signal: SignalSpec
    rebalance: Literal["daily", "weekly", "monthly", "continuous"] = "daily"
    market_neutral: bool = True
    position_limit: float = 0.05
    leverage_cap: float = 2.0
    filters: Optional[FilterSpec] = None
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    regime_filter: Optional[RegimeFilterSpec] = None
    status: Literal["active", "requires_data", "requires_infra"] = "active"
    requires: Optional[str] = None  # human-readable description of missing data
    description: str = ""

    # Backtest overrides
    warmup_bars: int = 252
    risk_free_rate: float = 0.04
