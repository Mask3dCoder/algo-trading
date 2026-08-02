"""Corporate-action handling for backtest fills: dividends and splits.

Wires into ``PortfolioState`` via ``CorporateActionsConfig``. All fields are
optional (empty = disabled) for backward compatibility.

Design
------
Backtests run on **unadjusted** bars (the PIT convention — see
``quantcore.data.pit.market``). A dividend or split therefore shows up as a
cash flow / price discontinuity that the portfolio must handle explicitly:

* **Dividends** — a long position held on the ex-date receives ``amount`` cash
  per share; a short position *pays* it. Credited after mark-to-market for the
  bar, so the equity curve reflects the income on the day it is earned.
* **Splits** — on the split date a held position's quantity is multiplied by
  the ratio and its average cost divided by it, preserving position value
  across the unadjusted-price jump (e.g. 2:1 split halves the price, doubles
  the shares).

Both are keyed by ISO ex-date (``"YYYY-MM-DD"``), the same convention the PIT
corp-action table uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CorporateActionsConfig:
    """Dividend/split schedule applied during a backtest.

    dividends: symbol -> {ex_date "YYYY-MM-DD": cash amount per share}
    splits:    symbol -> {split_date "YYYY-MM-DD": ratio (to/from)}
    """

    dividends: dict[str, dict[str, float]] = field(default_factory=dict)
    splits: dict[str, dict[str, float]] = field(default_factory=dict)


def ex_date_key(timestamp: object) -> str:
    """Return the ISO date key (``YYYY-MM-DD``) for a snapshot timestamp."""
    return pd.Timestamp(timestamp).date().isoformat()
