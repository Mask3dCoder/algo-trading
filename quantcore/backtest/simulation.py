"""Fill simulation models: slippage, commissions, borrowing costs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from quantcore.core.types import Side


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


class OptionFillSimulator:
    """Simulates option fills with theoretical pricing + spread haircuts.

    Unlike equities, options have no OHLCV bars.  Fills are computed from
    the Black-Scholes model using the underlying's price + implied vol.
    """

    def simulate_option_fill(
        self,
        underlying_price: float,
        strike: float,
        iv: float,
        dte: float,
        r: float = 0.045,
        is_call: bool = False,
        side: Side = Side.SELL,
        contracts: int = 1,
        spread_pct: float = 0.05,
        commission_per_contract: float = 0.65,
    ) -> dict:
        """Return fill details: fill_price, notional, commission."""
        from quantcore.math_models.options.bsm import bsm_call, bsm_put
        T = max(dte / 365.0, 1e-6)
        if is_call:
            theo_mid = float(bsm_call(underlying_price, strike, T, r, iv))
        else:
            theo_mid = float(bsm_put(underlying_price, strike, T, r, iv))

        if side == Side.SELL:
            fill_price = theo_mid * (1.0 - spread_pct)
        else:
            fill_price = theo_mid * (1.0 + spread_pct)

        commission = commission_per_contract * contracts
        notional = fill_price * contracts * 100.0
        return {
            "fill_price": fill_price,
            "notional": notional,
            "commission": commission,
            "theoretical_mid": theo_mid,
            "multiplier": 100,
        }


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
    annual_rate: float = 0.06

    def calculate(self, notional: float, days_held: int = 1) -> float:
        daily_rate = self.annual_rate / 252.0
        return notional * daily_rate * days_held
