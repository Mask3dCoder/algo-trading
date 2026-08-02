"""Point-in-time market-data store.

This package is the **market-data** counterpart to the SEC XBRL PIT database in
``quantcore.data.pit.blueprint``. It stores unadjusted equity bars, options
contracts / NBBO quotes, and corporate actions as append-only parquet files,
and exposes as-of readers that return exactly what the market knew at ``T``.

The generic in-memory feature store (``quantcore.data.point_in_time``) remains
the right tool for feature vectors; this package is the durable raw-data layer
that feeds it.
"""

from quantcore.data.pit.market.schema import (
    CORP_ACTION_COLUMNS,
    EQUITY_BAR_COLUMNS,
    OPTION_CONTRACT_COLUMNS,
    OPTION_QUOTE_COLUMNS,
    UNIVERSE_COLUMNS,
)
from quantcore.data.pit.market.store import ParquetPitStore, PitStoreError

__all__ = [
    "CORP_ACTION_COLUMNS",
    "EQUITY_BAR_COLUMNS",
    "OPTION_CONTRACT_COLUMNS",
    "OPTION_QUOTE_COLUMNS",
    "UNIVERSE_COLUMNS",
    "ParquetPitStore",
    "PitStoreError",
]


def default_market_data_dir() -> str:
    """Resolve the default PIT market-data directory.

    Override with ``QUANTCORE_PIT_MARKET_DIR`` (the project-wide dynaconf env
    convention). Defaults to ``<repo>/data/pit/market`` (gitignored).
    """
    import os
    from pathlib import Path

    override = os.environ.get("QUANTCORE_PIT_MARKET_DIR")
    if override:
        return override
    repo_root = Path(__file__).resolve().parents[4]
    return str(repo_root / "data" / "pit" / "market")
