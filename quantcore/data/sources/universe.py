"""Universe definitions: static, point-in-time, ADV/price filters.

AUDIT STATUS (2026-05-09):
  - Equity universes (SP500, NASDAQ100) use CURRENT constituents only.
    Survivorship bias is present for all equity backtests.
  - ETF_MACRO universe (55 highly liquid ETFs) is survivorship-bias-free.
  - Use ETF_MACRO for ANY strategy requiring a valid historical backtest
    until historical_universe.py is populated with point-in-time equity data.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date

_log = logging.getLogger(__name__)

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

# 55 liquid ETFs available via yfinance from 2010 (some from 2006)
_ETF_MACRO = [
    # US Broad Market
    "SPY", "QQQ", "IWM", "DIA", "VTI", "MDY",
    # International Equities
    "EFA", "EEM", "VEU", "FEZ", "EWJ", "EWZ",
    # Fixed Income -- Rates
    "TLT", "IEF", "SHY", "TIP", "IEI", "BIL",
    # Fixed Income -- Credit
    "LQD", "HYG", "EMB", "BKLN", "MBB",
    # Commodities
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "CPER",
    # Real Assets
    "VNQ", "VNQI", "IAU",
    # Currencies (ETF proxies)
    "UUP", "FXE", "FXY", "FXB", "FXA",
    # US Sectors
    "XLF", "XLK", "XLV", "XLE", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLB",
    # Volatility (regime only -- do NOT use for momentum)
    "VXX",
    # Factor ETFs
    "MTUM", "VLUE", "QUAL", "USMV", "SIZE",
]
_ETF_MACRO = list(dict.fromkeys(_ETF_MACRO))

_ETF_MACRO_CORE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "DBC", "UUP",
    "FXE", "VNQ", "XLE", "XLK", "XLV",
]


@dataclass
class UniverseDef:
    name: str
    symbols: list[str]
    is_static: bool = True
    survivorship_bias_free: bool = False
    data_start: str = "2010-01-01"


UNIVERSES = {
    "SP500": UniverseDef("S&P 500 (survivorship-biased)", _SP500,
                         survivorship_bias_free=False),
    "SP100": UniverseDef("S&P 100 (survivorship-biased)", _SP100,
                         survivorship_bias_free=False),
    "NASDAQ100": UniverseDef("Nasdaq 100 (survivorship-biased)", _NASDAQ100,
                             survivorship_bias_free=False),
    "RUSSELL2000": UniverseDef("Russell 2000", [], survivorship_bias_free=False),
    "SECTOR_ETF": UniverseDef("Sector ETFs", list(_SECTOR_ETFS.keys()),
                              survivorship_bias_free=True, data_start="2005-01-01"),
    "ETF_MACRO": UniverseDef(
        "Macro ETF Universe -- 55 liquid ETFs (survivorship-bias minimal)",
        _ETF_MACRO, is_static=True, survivorship_bias_free=True, data_start="2010-01-01",
    ),
    "ETF_MACRO_CORE": UniverseDef(
        "Core Macro ETF Universe -- 20 liquid ETFs (available from 2006)",
        _ETF_MACRO_CORE, is_static=True, survivorship_bias_free=True, data_start="2006-01-01",
    ),
    "PHASE3_UNIVERSE": UniverseDef(
        "Phase 3 Expanded Universe (legacy alias for ETF_MACRO_CORE)",
        _ETF_MACRO_CORE, is_static=False, survivorship_bias_free=True,
    ),
    "cta": UniverseDef(
        "CTA trend universe",
        ["SPY", "TLT", "GLD", "DBC", "IEF"],
        is_static=True, survivorship_bias_free=True, data_start="2005-01-01",
    ),
    "rp": UniverseDef(
        "Risk parity universe",
        ["SPY", "TLT", "GLD", "DBC", "IEF"],
        is_static=True, survivorship_bias_free=True, data_start="2005-01-01",
    ),
    "expanded_15": UniverseDef(
        "Expanded 15-asset macro universe",
        ["SPY", "QQQ", "IWM", "EFA", "EEM",
         "TLT", "IEF", "SHY", "LQD", "HYG",
         "GLD", "SLV", "DBC", "UUP", "VNQ"],
        is_static=True, survivorship_bias_free=True, data_start="2008-01-01",
    ),
}


def get_current_universe(name: str) -> list[str]:
    """Get current constituents of a named universe."""
    if name in UNIVERSES:
        return UNIVERSES[name].symbols.copy()
    return []


def get_universe_at_date(name: str, as_of: date) -> list[str]:
    """Point-in-time universe reconstruction.

    ETF universes: static, survivorship-bias minimal. Returns list after
    checking data_start.

    Equity universes: tries universe_history.json, falls back to current
    constituents with a SURVIVORSHIP BIAS audit warning.
    """
    import json
    from pathlib import Path

    udef = UNIVERSES.get(name)

    if udef is not None and udef.survivorship_bias_free:
        if as_of.isoformat() < udef.data_start:
            _log.warning(
                "universe_date_before_start",
                extra={"universe": name, "as_of": str(as_of),
                       "data_start": udef.data_start},
            )
        return udef.symbols.copy()

    history_file = Path(__file__).parent / "universe_history.json"
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
            if name in history:
                date_str = as_of.isoformat()
                available_dates = sorted(history[name].keys())
                closest = None
                for d in available_dates:
                    if d <= date_str:
                        closest = d
                    else:
                        break
                if closest:
                    _log.info("universe_pit_loaded", universe=name, date=closest)
                    return history[name][closest]
        except Exception as e:
            _log.error("universe_history_load_failed", error=str(e))

    _log.warning(
        "SURVIVORSHIP_BIAS_FALLBACK",
        extra={
            "universe": name, "as_of": str(as_of),
            "warning": (
                "Returning CURRENT constituents. SURVIVORSHIP BIAS IS PRESENT. "
                "Backtest results are NOT institutionally valid. "
                "Populate universe_history.json to fix."
            ),
        },
    )
    return get_current_universe(name)


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
