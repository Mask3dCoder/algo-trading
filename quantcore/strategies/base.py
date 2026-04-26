"""Abstract strategy base class, lifecycle, and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from numpy.typing import NDArray


class StrategyState(str, Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class StrategyType(str, Enum):
    EQUITY = "equity"
    OPTIONS = "options"
    MACRO = "macro"


@dataclass
class StrategyConfig:
    """Base configuration for all strategies."""
    name: str
    strategy_type: StrategyType
    max_position_notional: float = 1_000_000.0
    max_leverage: float = 1.0
    risk_budget_pct: float = 0.02
    params: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    """Runtime performance metrics for a strategy."""
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    n_trades: int = 0
    n_winning: int = 0
    n_losing: int = 0
    gross_leverage: float = 0.0
    net_leverage: float = 0.0
    portfolio_value: float = 0.0


@dataclass
class TradeSignal:
    """Signal emitted by a strategy."""
    symbol: str
    direction: int  # -1, 0, +1
    strength: float  # [0, 1]
    size: float  # suggested position size (notional)
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Lifecycle: initialize → on_data → on_fill → teardown
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.state = StrategyState.UNINITIALIZED
        self.metrics = StrategyMetrics()
        self._positions: dict[str, float] = {}
        self._pending_signals: list[TradeSignal] = []
        self._history: list[dict] = []

    def initialize(self) -> None:
        """Initialize strategy resources and validate config."""
        self._validate_config()
        self._init_state()
        self.state = StrategyState.INITIALIZED

    def _validate_config(self) -> None:
        if self.config.max_leverage <= 0:
            raise ValueError(f"max_leverage must be positive, got {self.config.max_leverage}")
        if self.config.risk_budget_pct <= 0 or self.config.risk_budget_pct > 1:
            raise ValueError(f"risk_budget_pct must be in (0, 1], got {self.config.risk_budget_pct}")

    def _init_state(self) -> None:
        self.metrics = StrategyMetrics()
        self._positions = {}
        self._pending_signals = []

    @abstractmethod
    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Process new market data and emit signals.

        Args:
            data: dict mapping field names to arrays (e.g. 'close', 'high', etc.)

        Returns:
            List of trade signals.
        """

    def on_fill(self, symbol: str, fill_price: float, fill_qty: float, side: str) -> None:
        """Handle a fill update."""
        if side == "buy":
            self._positions[symbol] = self._positions.get(symbol, 0.0) + fill_qty
        else:
            self._positions[symbol] = self._positions.get(symbol, 0.0) - fill_qty
        self.metrics.n_trades += 1

    def update_pnl(self, realized: float, unrealized: float) -> None:
        """Update P&L tracking."""
        self.metrics.realized_pnl += realized
        self.metrics.unrealized_pnl = unrealized
        self.metrics.total_pnl = self.metrics.realized_pnl + unrealized
        if realized > 0:
            self.metrics.n_winning += 1
        elif realized < 0:
            self.metrics.n_losing += 1

    def teardown(self) -> None:
        """Clean up resources."""
        self.state = StrategyState.STOPPED

    @property
    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def strategy_type(self) -> StrategyType:
        return self.config.strategy_type
