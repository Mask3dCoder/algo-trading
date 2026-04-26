"""Fill simulation models: slippage, commissions, borrowing costs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from quantcore.core.types import Fill, Order, Side


class SlippageModel(ABC):
    """Abstract slippage model."""

    @abstractmethod
    def apply(self, price: float, side: Side, quantity: float, **kwargs: float) -> float:
        """Return fill price after slippage."""
        ...


@dataclass
class FixedSlippage(SlippageModel):
    """Fixed basis-point slippage."""
    bps: float = 5.0

    def apply(self, price: float, side: Side, quantity: float, **kwargs: float) -> float:
        slip = price * self.bps / 10_000.0
        return price + slip if side == Side.BUY else price - slip


@dataclass
class VolumeSlippage(SlippageModel):
    """Volume-proportional slippage: participation_rate * sigma."""
    participation_rate: float = 0.05
    daily_volume: float = 1_000_000.0
    sigma: float = 0.02

    def apply(self, price: float, side: Side, quantity: float, **kwargs: float) -> float:
        vol = kwargs.get("volume", self.daily_volume)
        participation = quantity / vol if vol > 0 else self.participation_rate
        slip_bps = self.sigma * np.sqrt(participation) * 10_000
        slip = price * slip_bps / 10_000.0
        return price + slip if side == Side.BUY else price - slip


@dataclass
class VolatilitySlippage(SlippageModel):
    """Volatility-adjusted slippage."""
    vol_mult: float = 0.5
    annual_vol: float = 0.20

    def apply(self, price: float, side: Side, quantity: float, **kwargs: float) -> float:
        daily_vol = self.annual_vol / np.sqrt(252)
        slip = price * self.vol_mult * daily_vol
        return price + slip if side == Side.BUY else price - slip


class CommissionModel(ABC):
    """Abstract commission model."""

    @abstractmethod
    def calculate(self, quantity: float, price: float) -> float:
        """Return commission amount."""
        ...


@dataclass
class PerShareCommission(CommissionModel):
    """Commission per share traded."""
    rate: float = 0.005

    def calculate(self, quantity: float, price: float) -> float:
        return abs(quantity) * self.rate


@dataclass
class PerTradeCommission(CommissionModel):
    """Flat commission per trade."""
    flat: float = 1.0

    def calculate(self, quantity: float, price: float) -> float:
        return self.flat


@dataclass
class TieredCommission(CommissionModel):
    """Tiered commission: different rates at different quantity breakpoints."""
    brackets: list[tuple[float, float]] = None  # [(max_qty, rate), ...]

    def __post_init__(self) -> None:
        if self.brackets is None:
            self.brackets = [(500, 0.005), (1000, 0.003), (float("inf"), 0.001)]

    def calculate(self, quantity: float, price: float) -> float:
        remaining = abs(quantity)
        total = 0.0
        prev_max = 0.0
        for bracket_max, rate in self.brackets:
            bracket_qty = min(remaining, bracket_max - prev_max)
            if bracket_qty > 0:
                total += bracket_qty * rate
                remaining -= bracket_qty
            prev_max = bracket_max
            if remaining <= 0:
                break
        return total


@dataclass
class ShortBorrowCost:
    """Annualized short borrow cost."""
    annual_rate: float = 0.02

    def calculate(self, quantity: float, price: float, days_held: int = 1) -> float:
        daily_rate = self.annual_rate / 252.0
        return abs(quantity) * price * daily_rate * days_held


@dataclass
class MarginCost:
    """Margin interest cost."""
    annual_rate: float = 0.06  # broker margin rate

    def calculate(self, notional: float, days_held: int = 1) -> float:
        daily_rate = self.annual_rate / 252.0
        return notional * daily_rate * days_held
