"""Liquidity-aware position sizing.

Provides the max_position_size() interface so strategies can be written
against a realistic ADV-based capacity constraint.  The actual limit
currently returns a large sentinel value because ETF ADV data has not yet
been integrated (Q2 deliverable).  The interface is stable.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

# Sentinel: returned when we have no ADV data.
_UNLIMITED = 1_000_000_000.0

_adv_cache: dict[str, float] = {}  # symbol -> 30-day ADV in dollars


def update_adv_cache(adv_data: dict[str, float]) -> None:
    """Update the ADV cache with fresh data.

    Call this daily after market close with fresh 30-day ADV estimates.
    Format: {"SPY": 15_000_000_000.0, "QQQ": 8_000_000_000.0, ...}
    """
    _adv_cache.update(adv_data)
    _log.info("adv_cache_updated", extra={"n_symbols": len(adv_data)})


def max_position_size(
    symbol: str,
    capital: float,
    participation_rate: float = 0.05,
    adv_override: float | None = None,
) -> float:
    """Maximum notional position size based on ADV participation rate.

    The standard institutional limit is to avoid exceeding 5-10%% of the
    30-day average daily volume.  Exceeding this risks significant market
    impact and delayed execution.

    Args:
        symbol:             Ticker symbol.
        capital:            Total portfolio capital in dollars.
        participation_rate: Maximum fraction of daily volume allowed.
                            Default 0.05 (5%% of ADV).
        adv_override:       Override ADV with a specific value (for testing).

    Returns:
        Maximum notional position in dollars.  Returns _UNLIMITED sentinel
        (1e9) when ADV data is unavailable so strategies do not break.
    """
    adv = adv_override
    if adv is None:
        adv = _adv_cache.get(symbol)
    if adv is None:
        _log.debug(
            "adv_not_available",
            extra={
                "symbol": symbol,
                "note": "Returning unlimited sentinel. Integrate ADV data in Q2.",
            },
        )
        return _UNLIMITED

    limit = adv * participation_rate
    if capital > 0:
        # Also enforce a portfolio-level cap (e.g., max 20%% in one name)
        limit = min(limit, capital * 0.20)
    return float(limit)


def fraction_of_adv(
    notional: float,
    symbol: str,
    adv_override: float | None = None,
) -> float | None:
    """Return the fraction of ADV that a given notional represents.

    Returns None when ADV data is unavailable.
    """
    adv = adv_override or _adv_cache.get(symbol)
    if adv is None or adv <= 0:
        return None
    return notional / adv
