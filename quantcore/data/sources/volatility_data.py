"""Volatility index data loader with Parquet caching.

Downloads VIX, VVIX, MOVE, OVX, GVZ indices and VIX futures term structure
from Yahoo Finance via ``yfinance``.  Caches each series as a Parquet file at
``{cache_dir}/{symbol}.parquet`` to avoid redundant API calls.

Usage::

    loader = VolatilityDataLoader()
    data = loader.load_all(start="2015-01-01")
    slope = loader.compute_term_structure_slope()

Environment:
    No API key required — uses public yfinance endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_log = logging.getLogger(__name__)

# Yahoo Finance tickers for vol indices
_TICKERS = {
    "vix": "^VIX",
    "vvix": "^VVIX",
    "move": "^MOVE",
    "ovx": "^OVX",
    "gvz": "^GVZ",
}

# VIX futures front and back month (approximated via ^VIX and ^VIX3M)
_VIX_FUTURES = {
    "vix_front": "^VIX",
    "vix_back": "^VIX3M",
}


@dataclass
class VolIndexData:
    """Aligned volatility index series.

    All series share the same DatetimeIndex (business days, forward-filled).
    """
    vix: pd.Series
    vvix: pd.Series
    move: pd.Series
    ovx: pd.Series
    gvz: pd.Series
    vix_front: pd.Series
    vix_back: pd.Series

    @property
    def term_structure_slope(self) -> pd.Series:
        """(back - front) / front. Positive = contango, negative = backwardation."""
        safe_front = self.vix_front.replace(0, np.nan)
        return ((self.vix_back - self.vix_front) / safe_front).rename("ts_slope")

    def to_dict(self, bar_index: int | None = None) -> dict[str, np.ndarray]:
        """Convert to dict of numpy arrays for strategy consumption.

        Args:
            bar_index: If provided, return arrays up to this index (inclusive).
                       If None, return full arrays.
        """
        fields = {
            "vix": self.vix,
            "vvix": self.vvix,
            "move": self.move,
            "ovx": self.ovx,
            "gvz": self.gvz,
            "vix_front": self.vix_front,
            "vix_back": self.vix_back,
            "ts_slope": self.term_structure_slope,
        }
        result = {}
        for name, s in fields.items():
            arr = s.values.astype(np.float64)
            if bar_index is not None:
                arr = arr[: bar_index + 1]
            result[name] = arr
        return result


class VolatilityDataLoader:
    """Download, cache, and serve volatility index data.

    Args:
        cache_dir: Directory for Parquet cache files.  Defaults to
            ``data/vol/`` relative to the project root.

    Example::

        loader = VolatilityDataLoader()
        data = loader.load_all(start="2015-01-01")
        print(data.vix.tail())
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[2] / "data" / "vol"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load_all(
        self,
        start: str = "2015-01-01",
        end: str | None = None,
    ) -> VolIndexData:
        """Load all volatility indices, aligned on a common business-day index.

        Missing values are forward-filled then backward-filled.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD). None = latest available.

        Returns:
            VolIndexData with all series aligned.
        """
        series: dict[str, pd.Series] = {}

        for name, ticker in {**_TICKERS, **_VIX_FUTURES}.items():
            try:
                series[name] = self._load_series(ticker, name, start, end)
            except Exception as e:
                _log.warning("failed to load %s (%s): %s", name, ticker, e)
                series[name] = pd.Series(dtype=float, name=name)

        # Align on common index
        df = pd.DataFrame(series)
        df = df.sort_index().ffill().bfill()

        return VolIndexData(
            vix=df["vix"],
            vvix=df["vvix"],
            move=df["move"],
            ovx=df["ovx"],
            gvz=df["gvz"],
            vix_front=df["vix_front"],
            vix_back=df["vix_back"],
        )

    def compute_term_structure_slope(
        self,
        start: str = "2015-01-01",
        end: str | None = None,
    ) -> pd.Series:
        """Compute VIX term structure slope: (back - front) / front.

        Positive = contango (normal), negative = backwardation (stress).
        """
        data = self.load_all(start, end)
        return data.term_structure_slope

    def refresh(self, name: str) -> None:
        """Clear cache for a specific series."""
        path = self._cache_path(name)
        if path.exists():
            path.unlink()
            _log.info("cache_cleared: %s", path)

    def refresh_all(self) -> None:
        """Clear all cached data."""
        for f in self._cache_dir.glob("*.parquet"):
            f.unlink()
            _log.info("cache_cleared: %s", f)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_series(
        self,
        ticker: str,
        name: str,
        start: str,
        end: str | None,
    ) -> pd.Series:
        path = self._cache_path(name)
        if path.exists():
            _log.debug("cache_hit: %s", path)
            return pd.read_parquet(path).squeeze().sort_index()

        _log.info("cache_miss: fetching %s (%s) from yfinance", name, ticker)
        kwargs = {"start": start}
        if end:
            kwargs["end"] = end

        data = yf.download(ticker, progress=False, **kwargs)
        if data.empty:
            _log.warning("no data returned for %s", ticker)
            return pd.Series(dtype=float, name=name)

        # yfinance returns Close column
        if isinstance(data.columns, pd.MultiIndex):
            s = data[("Close", ticker)]
        elif "Close" in data.columns:
            s = data["Close"]
        else:
            s = data.iloc[:, 0]

        s = pd.to_numeric(s, errors="coerce").dropna()
        s.name = name
        s.index = pd.to_datetime(s.index).tz_localize(None)

        self._cache(name, s)
        return s

    def _cache(self, name: str, series: pd.Series) -> None:
        path = self._cache_path(name)
        df = series.to_frame()
        df.to_parquet(path, engine="pyarrow")
        _log.debug("cached: %s (%d rows)", path, len(df))

    def _cache_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.parquet"
