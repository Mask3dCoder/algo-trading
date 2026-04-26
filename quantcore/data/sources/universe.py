"""Universe definitions: static, point-in-time, ADV/price filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


# Static universe definitions (current constituents)
_SP500 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO",
    "AVGO", "COST", "CSCO", "WMT", "MCD", "TMO", "NKE", "ABT", "DHR", "CRM",
    "ORCL", "ACN", "TXN", "LLY", "ADBE", "LIN", "BMY", "UPS", "NEE", "PM",
    "QCOM", "RTX", "AMD", "T", "GS", "INTC", "HON", "VZ", "COP", "CAT",
]

_SP100 = _SP500[:100]

_NASDAQ100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO", "COST", "AMD",
    "NFLX", "ADBE", "CRM", "ORCL", "INTC", "QCOM", "CSCO", "TXN", "PYPL", "BKNG",
    "SBUX", "MDLZ", "GILD", "INTU", "ISRG", "VRTX", "REGN", "ADI", "FISV", "CSX",
    "MU", "ATVI", "PEP", "CHTR", "AMGN", "MRVL", "LRCX", "SNPS", "PANW", "KLAC",
]

_SECTOR_ETFS = {
    "XLF": "Financials", "XLK": "Technology", "XLV": "Healthcare",
    "XLE": "Energy", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLI": "Industrials", "XLU": "Utilities", "XLRE": "Real Estate", "XLB": "Materials",
}


@dataclass
class UniverseDef:
    name: str
    symbols: list[str]
    is_static: bool = True


UNIVERSES = {
    "SP500": UniverseDef("S&P 500", _SP500),
    "SP100": UniverseDef("S&P 100", _SP100),
    "NASDAQ100": UniverseDef("Nasdaq 100", _NASDAQ100),
    "RUSSELL2000": UniverseDef("Russell 2000", []),  # Populated dynamically
    "SECTOR_ETF": UniverseDef("Sector ETFs", list(_SECTOR_ETFS.keys())),
}


def get_current_universe(name: str) -> list[str]:
    """Get current constituents of a named universe."""
    if name in UNIVERSES:
        return UNIVERSES[name].symbols.copy()
    return []


def get_universe_at_date(name: str, as_of: date) -> list[str]:
    """Point-in-time universe reconstruction.

    For accurate historical analysis, returns the constituents as of `as_of`.
    This avoids survivorship bias.
    """
    # For static universes, just return current (this should be replaced with
    # historical membership data for production use)
    universe = get_current_universe(name)

    # TODO: Fetch historical constituent data from a proper source
    # (e.g., S&P official records, Wikipedia historical snapshots)
    return universe


def filter_by_adv(
    symbols: list[str],
    daily_volumes: dict[str, float],
    min_adv: float = 1_000_000.0,
) -> list[str]:
    """Filter symbols by average daily volume (in dollars)."""
    return [s for s in symbols if daily_volumes.get(s, 0.0) >= min_adv]


def filter_by_price(
    symbols: list[str],
    prices: dict[str, float],
    min_price: float = 5.0,
    max_price: float = float("inf"),
) -> list[str]:
    """Filter symbols by price range."""
    return [s for s in symbols if min_price <= prices.get(s, 0.0) <= max_price]
