"""Financing costs for backtest: short borrow costs, margin interest, forced liquidation.

Wires ShortBorrowCost and MarginCost models from simulation.py into the
backtest engine's PortfolioState. All configs are optional (None = disabled)
for backward compatibility.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field


@dataclass
class ShortPosition:
    """Tracks a short position's borrow cost accrual."""

    market_value: float = 0.0
    """Current absolute market value of the short position."""

    borrow_rate: float = 0.005
    """Blended annual borrow rate in decimal (e.g., 0.005 = 50 bps)."""

    accrued_cost: float = 0.0
    """Total borrow cost accrued so far for this position."""


@dataclass
class FinancingConfig:
    """Configuration for backtest financing costs.

    All fields are optional — set to None to disable the corresponding cost.
    """

    borrow_cost_registry: BorrowCostRegistry | None = None
    """Registry of annual borrow rates by symbol. If None, no borrow costs."""

    margin_rate: float = 0.065
    """Annual margin interest rate (default: Fed Funds ~5% + 150 bps = 6.5%)."""

    margin_limit: float = 2.0
    """Maximum leverage (e.g., 2.0 = 200% of equity). Triggers forced liquidation."""

    short_rebate_rate: float = 0.0
    """Rebate rate paid on short sale cash proceeds (default: 0%)."""


class BorrowCostRegistry:
    """Manages annual borrow rates by symbol for short positions.

    Rates are stored as annualized decimals (e.g., 0.05 = 500 bps = 5%).
    """

    def __init__(self, default_rate: float = 0.005) -> None:
        self._rates: dict[str, float] = {}
        self.default_rate: float = default_rate

    def set_rate(self, symbol: str, rate: float) -> None:
        """Set the annual borrow rate for a symbol."""
        self._rates[symbol] = rate

    def get_rate(self, symbol: str) -> float:
        """Look up a symbol's borrow rate, falling back to the default."""
        return self._rates.get(symbol, self.default_rate)

    def classify_borrow(self, rate: float) -> str:
        """Classify a borrow rate into a difficulty category.

        Categories (annual rate):
        - easy: < 0.50%
        - medium: 0.50% — 1.50%
        - hard: 1.50% — 5.00%
        - special: >= 5.00%
        """
        if rate < 0.005:
            return "easy"
        if rate < 0.015:
            return "medium"
        if rate < 0.05:
            return "hard"
        return "special"

    def load_from_csv(self, path: str) -> None:
        """Load borrow rates from a CSV file.

        Expected CSV columns: symbol, rate (annual decimal), effective_date (optional).

        Example:
            symbol,rate,effective_date
            GME,0.50,2024-01-01
            AAPL,0.0025,2024-01-01
        """
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row["symbol"].strip()
                rate = float(row["rate"])
                self._rates[symbol] = rate

    @property
    def symbols(self) -> set[str]:
        return set(self._rates.keys())

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._rates


def compute_daily_borrow_cost(
    short_market_value: float,
    borrow_rate: float,
    days: int = 1,
) -> float:
    """Compute daily borrow cost for a short position.

    Uses 360-day convention standard in securities lending.
    """
    return abs(short_market_value) * borrow_rate / 360.0 * days


def compute_daily_margin_interest(
    margin_balance: float,
    margin_rate: float,
    days: int = 1,
) -> float:
    """Compute daily margin interest.

    margin_balance: the amount of borrowed money (positive = owed).
    margin_rate: annual rate in decimal.
    Uses 360-day convention.
    """
    return max(margin_balance, 0.0) * margin_rate / 360.0 * days
