"""Canonical column schemas for the point-in-time market-data parquet store.

Design notes
------------
* Equity bars are stored **unadjusted** (Polygon ``adjusted=false``). Corporate
  actions live in their own table and are applied at research time with
  announcement-awareness, so a backtest at ``T`` never sees a split/dividend
  that was not yet known at ``T`` (back-adjusted prices are a lookahead leak).
* Every table carries an explicit ``*_asof``-style timestamp. For market data
  the observation *is* the availability instant (a daily bar is known at its
  close; an NBBO quote is known when printed), so the reader gates purely on
  ``ts <= asof``. Corporate actions are the exception: they carry both an
  ``effective_date`` and a nullable ``announced_at``.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

# --- Equity end-of-day bars (Polygon /v2/aggs, adjusted=false) --------------
# ts is the bar close instant in UTC. Rows are append-only; a bar is only
# written once the session is finalised (never intraday).
EQUITY_BAR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ts",  # datetime64[ns, UTC] — bar close
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
)

# --- Options contract reference (Polygon /v3/reference/options/contracts) ----
OPTION_CONTRACT_COLUMNS: tuple[str, ...] = (
    "contract_symbol",  # OCC symbol, e.g. SPY250116C00600000
    "underlying_symbol",
    "expiration_date",  # date
    "strike_price",     # float
    "right",            # "C" | "P"
    "exercise_style",   # "A"merican | "E"uropean
    "multiplier",       # contract multiplier (100)
    "cfi",
    "primary_exchange",
    "created_at",       # datetime64[ns, UTC] — contract creation
    "updated_at",       # datetime64[ns, UTC] — last reference update
)

# --- Options NBBO quote snapshots (Polygon /v3/quotes, Advanced tier) --------
# One row per quote print. ts is the exchange timestamp. High volume: partition
# by contract symbol + YYYYMM.
OPTION_QUOTE_COLUMNS: tuple[str, ...] = (
    "contract_symbol",  # OCC symbol
    "underlying_symbol",
    "ts",               # datetime64[ns, UTC] — quote exchange timestamp
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "midpoint",         # (bid + ask) / 2
    "spread",           # ask - bid
    "iv",               # implied vol (nullable — only if server supplies greeks)
    "delta",            # greek (nullable)
)

# --- Corporate actions (splits / dividends / delistings / ticker changes) ----
# effective_date: the market date the action bites (ex-date, split date).
# announced_at:   when the market first knew (nullable — PIT gate when present).
CORP_ACTION_COLUMNS: tuple[str, ...] = (
    "action_type",      # split | dividend | delisting | merger | spinoff
    "symbol",           # symbol the action applies to (old symbol for changes)
    "effective_date",   # date
    "announced_at",     # datetime64[ns, UTC] or NaT
    "value",            # split ratio / dividend-per-share
    "ticker_old",       # for delisting/merger remaps
    "ticker_new",
    "description",
    "source",           # "polygon" | "manual"
)

# --- Universe / tickers reference (Polygon /v3/reference/tickers) ------------
# delisted_utc is the survivorship-bias killer: active=false tickers are kept,
# so a backtest at T never assumes a ticker exists that had already died.
UNIVERSE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "name",
    "market_cap",           # float or NaN
    "shares_outstanding",   # float or NaN
    "type",                 # "CS" etc.
    "delisted_utc",         # datetime64[ns, UTC] or NaT
    "last_updated",         # datetime64[ns, UTC]
)

DATASET_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "equity_bars": EQUITY_BAR_COLUMNS,
    "options_contracts": OPTION_CONTRACT_COLUMNS,
    "options_quotes": OPTION_QUOTE_COLUMNS,
    "corp_actions": CORP_ACTION_COLUMNS,
    "universe": UNIVERSE_COLUMNS,
}

# Period granularity for partitioning:
#   equity_bars      -> <shard>/<YYYY>.parquet      (one file per symbol-year)
#   options_contracts-> <shard>/<YYYY>.parquet
#   options_quotes   -> <shard>/<YYYYMM>.parquet    (monthly — high volume)
#   corp_actions     -> <shard>/<YYYY>.parquet
#   universe         -> <shard=all>/<YYYY>.parquet
PERIOD_GRANULARITY: Mapping[str, str] = {
    "equity_bars": "year",
    "options_contracts": "year",
    "options_quotes": "month",
    "corp_actions": "year",
    "universe": "year",
}


def period_shard(ts: pd.Timestamp, dataset: str) -> str:
    """Return the partition key for a timestamp in a dataset."""
    t = pd.Timestamp(ts)
    return t.strftime("%Y%m") if PERIOD_GRANULARITY.get(dataset) == "month" else t.strftime("%Y")
