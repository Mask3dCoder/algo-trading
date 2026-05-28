"""Fama-French 5-factor daily data loader.

Wraps the existing FamaFrenchLoader in a clean interface suitable for
alpha model consumption.
"""

from pathlib import Path

import pandas as pd


def load_5_factors(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Load Fama-French 5-factor daily data.

    Returns a DataFrame with columns: ``Mkt-RF, SMB, HML, RMW, CMA, RF``
    indexed by date (sorted ascending). If ``cache_dir`` is provided, raw
    CSVs are persisted there for fast re-use.
    """
    from quantcore.data.sources.fama_french import FamaFrenchLoader

    loader = FamaFrenchLoader(cache_dir=cache_dir)
    df = loader.load_5_factors(freq="daily")
    return df
