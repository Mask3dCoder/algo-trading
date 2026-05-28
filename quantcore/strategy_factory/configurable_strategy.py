"""Configurable strategy — reads logic from YAML, no code required.

Implements the Strategy ABC from quantcore.strategies.base. Uses SignalEngine
to compute raw signals, applies cross-sectional ranking, enforces position
limits and market neutrality, and optionally applies FRED regime filters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, StrategyType, TradeSignal
from quantcore.strategy_factory.config_schema import StrategyFactoryConfig
from quantcore.strategy_factory.signal_engine import SignalEngine

log = logging.getLogger(__name__)

# Map factory strategy types to base StrategyType
_TYPE_MAP: dict[str, StrategyType] = {
    "time_series": StrategyType.MACRO,
    "cross_sectional": StrategyType.MACRO,
    "mean_reversion": StrategyType.MACRO,
    "pairs": StrategyType.MACRO,
    "factor": StrategyType.MACRO,
    "arbitrage": StrategyType.MACRO,
    "event_driven": StrategyType.MACRO,
    "hft": StrategyType.MACRO,
}


def load_factory_config(path: str | Path) -> StrategyFactoryConfig:
    """Load a strategy factory config from a YAML file."""
    import yaml

    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return StrategyFactoryConfig(**raw)


def load_factory_config_by_name(name: str, configs_dir: str | Path | None = None) -> StrategyFactoryConfig:
    """Load a config by name from the configs directory.

    Args:
        name: Config name (without .yaml extension), e.g. "time_series_momentum".
        configs_dir: Override configs directory. Defaults to quantcore/strategy_factory/configs/.
    """
    if configs_dir is None:
        configs_dir = Path(__file__).parent / "configs"
    else:
        configs_dir = Path(configs_dir)

    path = configs_dir / f"{name}.yaml"
    if not path.exists():
        # Try with .yml extension
        path = configs_dir / f"{name}.yml"
    if not path.exists():
        available = [f.stem for f in configs_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Config '{name}' not found at {configs_dir}. Available: {sorted(available)}"
        )
    return load_factory_config(path)


class ConfigurableStrategy(Strategy):
    """Universal strategy driven by YAML configuration.

    Reads a StrategyFactoryConfig, uses SignalEngine to compute signals,
    applies cross-sectional ranking, enforces position limits and market
    neutrality, and returns TradeSignal objects.

    Parameters
    ----------
    factory_config:
        Strategy factory configuration (loaded from YAML).
    """

    def __init__(self, factory_config: StrategyFactoryConfig) -> None:
        # Build the base StrategyConfig
        base_config = StrategyConfig(
            name=factory_config.name,
            strategy_type=_TYPE_MAP.get(factory_config.type, StrategyType.MACRO),
            max_leverage=factory_config.leverage_cap,
            params={
                "formula": factory_config.signal.formula,
                "rank": factory_config.signal.rank or "",
                "rebalance": factory_config.rebalance,
                "market_neutral": factory_config.market_neutral,
                "position_limit": factory_config.position_limit,
            },
        )
        super().__init__(base_config)

        self._factory_config = factory_config
        self._engine = SignalEngine()
        self._last_rebalance_bar: int = -9999
        self._rebalance_freq: int = self._rebalance_interval()
        self._warmup_bars: int = factory_config.warmup_bars
        self._current_bar: int = 0

    @property
    def factory_config(self) -> StrategyFactoryConfig:
        return self._factory_config

    def _rebalance_interval(self) -> int:
        mapping = {"daily": 1, "weekly": 5, "monthly": 21, "continuous": 1}
        return mapping.get(self._factory_config.rebalance, 1)

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Process market data and emit signals.

        The data dict contains:
            - "close": 2D array (n_periods, n_assets) of close prices
            - "open", "high", "low", "volume": 2D arrays
            - "symbols": list of symbol names
            - Optional: "factors", "sentiment", "fred_signals", "pairs"
        """
        symbols = data.get("symbols", [])
        if not symbols:
            return []

        close = data.get("close")
        if close is None or close.ndim < 2 or close.shape[0] < 2:
            return []

        n_periods = close.shape[0]
        n_assets = close.shape[1]
        self._current_bar = n_periods - 1

        # Wait for warmup
        if n_periods < self._warmup_bars:
            return []

        # Check rebalance schedule
        bars_since_rebalance = n_periods - self._last_rebalance_bar - 1
        if bars_since_rebalance < self._rebalance_freq and self._last_rebalance_bar > 0:
            return []

        self._last_rebalance_bar = n_periods

        # Compute raw signal
        formula = self._factory_config.signal.formula
        try:
            raw_signal = self._engine.compute(formula, data)
        except Exception as e:
            log.warning("Signal computation failed for %s: %s", formula, e)
            return []

        if raw_signal is None or raw_signal.shape != close.shape:
            return []

        # Apply cross-sectional ranking if specified
        rank_method = self._factory_config.signal.rank
        if rank_method and n_assets > 1:
            weights = self._engine.rank_cross_sectional(
                raw_signal, method=rank_method,
            )
        else:
            # Time-series: use signal sign as direction, magnitude as strength
            weights = np.sign(raw_signal) * np.minimum(np.abs(raw_signal), 1.0)

        # Get the latest bar's weights
        latest_weights = weights[-1]

        # Apply regime filter if configured
        if self._factory_config.regime_filter and self._factory_config.regime_filter.enabled:
            latest_weights = self._apply_regime_filter(data, latest_weights)

        # Apply filters (volume, price, symbol inclusion/exclusion)
        latest_weights = self._apply_filters(data, latest_weights, symbols)

        # Enforce market neutrality
        if self._factory_config.market_neutral:
            latest_weights = self._enforce_market_neutral(latest_weights)

        # Enforce position limits
        latest_weights = self._enforce_position_limits(latest_weights)

        # Convert to TradeSignal list
        return self._to_trade_signals(latest_weights, symbols, data)

    def _apply_regime_filter(
        self, data: dict, weights: NDArray,
    ) -> NDArray:
        """Apply FRED regime filter — zero out weights when regime is unfavorable."""
        regime_cfg = self._factory_config.regime_filter
        if regime_cfg is None:
            return weights

        try:
            regime_signal = self._engine.compute(
                f"fred_regime_score({regime_cfg.signal})", data,
            )
            if regime_signal is not None and regime_signal.shape[0] > 0:
                latest_regime = regime_signal[-1, 0]  # same for all assets
                if regime_cfg.inverse:
                    if latest_regime > regime_cfg.threshold:
                        return np.zeros_like(weights)
                else:
                    if latest_regime < regime_cfg.threshold:
                        return np.zeros_like(weights)
        except Exception:
            pass  # If regime data unavailable, trade normally

        return weights

    def _apply_filters(
        self, data: dict, weights: NDArray, symbols: list[str],
    ) -> NDArray:
        """Apply volume, price, and symbol filters."""
        filters = self._factory_config.filters
        if filters is None:
            return weights

        close = data.get("close")
        volume = data.get("volume")

        # Symbol inclusion/exclusion
        if filters.include_symbols:
            for i, sym in enumerate(symbols):
                if sym not in filters.include_symbols and i < len(weights):
                    weights[i] = 0.0

        if filters.exclude_symbols:
            for i, sym in enumerate(symbols):
                if sym in filters.exclude_symbols and i < len(weights):
                    weights[i] = 0.0

        # Price filters
        if close is not None and close.shape[0] > 0:
            latest_close = close[-1]
            for i in range(min(len(weights), len(latest_close))):
                price = latest_close[i]
                if price < filters.min_price or price > filters.max_price:
                    weights[i] = 0.0

        # Volume filter
        if volume is not None and volume.shape[0] > 0 and filters.min_volume > 0:
            latest_vol = volume[-1]
            for i in range(min(len(weights), len(latest_vol))):
                if latest_vol[i] < filters.min_volume:
                    weights[i] = 0.0

        return weights

    def _enforce_market_neutral(self, weights: NDArray) -> NDArray:
        """Ensure long and short exposures are balanced."""
        long_mask = weights > 0
        short_mask = weights < 0

        long_sum = np.sum(weights[long_mask])
        short_sum = np.abs(np.sum(weights[short_mask]))

        if long_sum > 1e-10 and short_sum > 1e-10:
            # Scale to balance
            target = (long_sum + short_sum) / 2.0
            if long_sum > 0:
                weights[long_mask] *= target / long_sum
            if short_sum > 0:
                weights[short_mask] *= target / short_sum
        elif long_sum > 1e-10 and short_sum < 1e-10:
            # All long — can't make neutral without short capability
            pass
        return weights

    def _enforce_position_limits(self, weights: NDArray) -> NDArray:
        """Cap individual position weights at position_limit."""
        limit = self._factory_config.position_limit
        if limit <= 0:
            return weights
        return np.clip(weights, -limit, limit)

    def _to_trade_signals(
        self, weights: NDArray, symbols: list[str], data: dict,
    ) -> list[TradeSignal]:
        """Convert weight array to TradeSignal objects."""
        signals: list[TradeSignal] = []
        close = data.get("close")

        for i, sym in enumerate(symbols):
            if i >= len(weights):
                break
            w = float(weights[i])
            if abs(w) < 1e-8:
                continue

            direction = 1 if w > 0 else -1
            strength = min(abs(w), 1.0)

            # Compute stop loss / take profit from recent volatility
            stop_loss = None
            take_profit = None
            if close is not None and close.shape[0] > 20 and i < close.shape[1]:
                recent = close[-20:, i]
                vol = float(np.std(np.diff(recent) / np.maximum(recent[:-1], 1e-10)))
                current_price = float(close[-1, i])
                if vol > 0 and current_price > 0:
                    stop_loss = current_price * (1 - direction * 2 * vol)
                    take_profit = current_price * (1 + direction * 3 * vol)

            signals.append(TradeSignal(
                symbol=sym,
                direction=direction,
                strength=strength,
                size=abs(w),
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "formula": self._factory_config.signal.formula,
                    "strategy_type": self._factory_config.type,
                    "weight": w,
                },
            ))

        return signals
