"""FRED data loader with Parquet caching and ALFRED first-release support.

Downloads macroeconomic series from the Federal Reserve Economic Data (FRED)
API via the ``fredapi`` package.  Caches each series as a Parquet file at
``{cache_dir}/{series_id}.parquet`` to avoid redundant API calls.

Look-ahead bias prevention:
  - Daily market data (yields, spreads, VIX): same-day values are acceptable.
  - Revised economic series (GDP, CPI, employment): use
    ``get_series_first_release()`` which queries ALFRED for the value as
    originally published, NOT the latest revision.

Usage::

    loader = FREDDataLoader()               # reads FRED_API_KEY from env
    s = loader.get_series("DGS10")          # daily, cached
    gdp = loader.get_series_first_release("GDP")  # first-release, no look-ahead
    df = loader.build_macro_dataframe()     # all configured series

Environment:
    FRED_API_KEY — required.  Get one at https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantcore.data.point_in_time import feature_frame_from_records

_log = logging.getLogger(__name__)

_FRED_API_KEY_VAR = "FRED_API_KEY"

# Series that are revised after initial publication — MUST use first-release
# to avoid look-ahead bias in backtests.
_REVISED_SERIES = frozenset({
    "GDP", "CPIAUCSL", "PCEPI", "UNRATE", "PAYEMS", "INDPRO", "RSAFS",
    "M2SL", "TOTALSL", "HOUST", "MSPUS",
})


class FREDDataLoader:
    """Download, cache, and serve FRED macroeconomic series.

    Args:
        api_key: FRED API key.  If None, reads from ``FRED_API_KEY`` env var.
        cache_dir: Directory for Parquet cache files.  Defaults to
            ``data/fred/`` relative to the project root.

    Raises:
        ValueError: If no API key is provided and ``FRED_API_KEY`` is not set.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get(_FRED_API_KEY_VAR, "")
        if not self._api_key:
            raise ValueError(
                f"FRED API key required.  Set {_FRED_API_KEY_VAR} env var "
                "or pass api_key=.  Get one at "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[2] / "data" / "fred"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._fred: Any | None = None
        self._series_config: dict[str, dict[str, Any]] = {}
        self._load_series_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_series(
        self,
        series_id: str,
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> pd.Series:
        """Get a FRED series as a pandas Series (datetime index, float values).

        Checks Parquet cache first.  On miss, downloads from FRED API and
        caches the result.

        Args:
            series_id: FRED series ID (e.g. "DGS10", "UNRATE").
            start: Observation start date (YYYY-MM-DD).
            end: Observation end date (YYYY-MM-DD).  None = latest available.

        Returns:
            pd.Series with DatetimeIndex and float64 values.
        """
        path = self._cache_path(series_id)
        if path.exists():
            _log.debug("cache_hit: %s", path)
            return pd.read_parquet(path).squeeze().sort_index()

        _log.info("cache_miss: fetching %s from FRED API", series_id)
        s = self._fetch_series(series_id, start, end)
        self._cache(series_id, s)
        return s

    def get_series_first_release(
        self,
        series_id: str,
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> pd.Series:
        """Get a series using ALFRED's first-release values.

        Critical for revised series (GDP, CPI, employment) to prevent
        look-ahead bias.  Returns the value as it was ORIGINALLY published,
        not the latest revision.

        For non-revised daily series (yields, spreads), use ``get_series()``
        instead — same-day values are acceptable for those.

        Args:
            series_id: FRED series ID.
            start: Observation start date.
            end: Observation end date.

        Returns:
            pd.Series with DatetimeIndex and float64 values.
        """
        cache_key = f"{series_id}_first_release"
        path = self._cache_path(cache_key)
        if path.exists():
            _log.debug("cache_hit: %s (first_release)", path)
            return pd.read_parquet(path).squeeze().sort_index()

        _log.info("cache_miss: fetching %s first_release from ALFRED", series_id)
        fred = self._get_fred()
        df = fred.get_series_first_release(series_id)
        if isinstance(df, pd.DataFrame):
            if "value" in df.columns:
                s = df["value"]
            else:
                s = df.iloc[:, 0]
        else:
            s = df

        s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        s.index = pd.to_datetime(s.index)
        s.name = series_id

        # Filter by date range (get_series_first_release has no date params)
        if start:
            s = s[s.index >= pd.Timestamp(start)]
        if end:
            s = s[s.index <= pd.Timestamp(end)]

        self._cache(cache_key, s)
        return s

    def get_spread(
        self,
        long_id: str,
        short_id: str,
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> pd.Series:
        """Compute a yield spread between two FRED series (long minus short).

        Both series are fetched via ``get_series()`` (daily data, no ALFRED
        needed) and aligned on their date index.

        Returns:
            pd.Series named ``"{long_id}-{short_id}"``.
        """
        long = self.get_series(long_id, start, end)
        short = self.get_series(short_id, start, end)
        spread = long - short
        spread.name = f"{long_id}-{short_id}"
        return spread.dropna()

    def get_yoy_change(
        self,
        series_id: str,
        start: str = "2000-01-01",
        end: str | None = None,
        use_first_release: bool = True,
    ) -> pd.Series:
        """Compute year-over-year percentage change for a monthly/quarterly series.

        Args:
            series_id: FRED series ID.
            start: Observation start date.
            end: Observation end date.
            use_first_release: If True and the series is revised, use ALFRED
                first-release data.  Default True for backtest safety.

        Returns:
            pd.Series of YoY percentage changes.
        """
        use_alfred = use_first_release and series_id in _REVISED_SERIES
        if use_alfred:
            s = self.get_series_first_release(series_id, start, end)
        else:
            s = self.get_series(series_id, start, end)

        return s.pct_change(periods=12).dropna()

    def get_feature_frame(
        self,
        series_id: str,
        start: str = "2000-01-01",
        end: str | None = None,
        use_first_release: bool = False,
    ) -> pd.DataFrame:
        """Return FRED observations in the unified feature schema.

        The returned frame has ``source``, ``symbol``, ``as_of``, ``valid_from``,
        ``feature_name`` and ``value`` columns. For ordinary market-observable
        daily series, ``valid_from`` equals ``as_of``. For ALFRED first-release
        observations, ``valid_from`` is the realtime start when the value was
        first publishable.
        """

        if use_first_release:
            return self._first_release_feature_frame(series_id, start, end)

        s = self.get_series(series_id, start, end)
        if s.empty:
            return pd.DataFrame()

        df = s.rename(series_id).reset_index()
        df = df.rename(columns={"index": "as_of", series_id: "value"})
        df["source"] = "fred"
        df["symbol"] = series_id
        df["valid_from"] = df["as_of"]
        df["feature_name"] = series_id
        return feature_frame_from_records(df)

    def build_macro_feature_frame(
        self,
        start: str = "2000-01-01",
        end: str | None = None,
        categories: list[str] | None = None,
        use_first_release: bool = True,
    ) -> pd.DataFrame:
        """Return all configured FRED series as one long point-in-time table."""

        frames: list[pd.DataFrame] = []
        for entry in self._series_config.values():
            cat = entry.get("category", "")
            if categories and cat not in categories:
                continue

            sid = entry["id"]
            name = entry.get("name", sid)
            try:
                use_alfred = use_first_release and sid in _REVISED_SERIES
                frames.append(self.get_feature_frame(sid, start, end, use_alfred))
            except Exception as e:
                _log.warning("failed to load feature frame for %s (%s): %s", sid, name, e)
                continue

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def build_macro_dataframe(
        self,
        start: str = "2000-01-01",
        end: str | None = None,
        categories: list[str] | None = None,
        use_first_release: bool = True,
    ) -> pd.DataFrame:
        """Build a wide DataFrame of all configured FRED macro series.

        Monthly/quarterly series are forward-filled to daily frequency.
        Column names are the FRED series IDs.

        Args:
            start: Observation start date.
            end: Observation end date.
            categories: Limit to specific categories (e.g. ["yields", "credit"]).
                None = all categories.
            use_first_release: Use ALFRED first-release for revised series.

        Returns:
            pd.DataFrame with DatetimeIndex and one column per series.
        """
        frames: list[pd.Series] = []

        for entry in self._series_config.values():
            cat = entry.get("category", "")
            if categories and cat not in categories:
                continue

            sid = entry["id"]
            freq = entry.get("frequency", "daily")
            name = entry.get("name", sid)

            try:
                use_alfred = use_first_release and sid in _REVISED_SERIES
                if use_alfred:
                    s = self.get_series_first_release(sid, start, end)
                else:
                    s = self.get_series(sid, start, end)
            except Exception as e:
                _log.warning("failed to load %s (%s): %s", sid, name, e)
                continue

            s.name = sid

            # Resample non-daily series to daily via forward-fill
            if freq in ("monthly", "quarterly"):
                s = s.resample("D").ffill()

            frames.append(s)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, axis=1).sort_index()
        return df

    def refresh_series(self, series_id: str) -> pd.Series:
        """Force re-download of a series (clears cache).

        Returns:
            Freshly downloaded pd.Series.
        """
        path = self._cache_path(series_id)
        if path.exists():
            path.unlink()
            _log.info("cache_cleared: %s", path)

        # Also clear first-release cache if it exists
        fr_path = self._cache_path(f"{series_id}_first_release")
        if fr_path.exists():
            fr_path.unlink()
            _log.info("cache_cleared: %s", fr_path)

        return self.get_series(series_id)

    def build_cache(self) -> int:
        """Pre-populate the cache with all configured series.

        Returns:
            Number of series successfully cached.
        """
        count = 0
        for entry in self._series_config.values():
            sid = entry["id"]
            try:
                self.get_series(sid)
                count += 1
            except Exception as e:
                _log.warning("build_cache failed for %s: %s", sid, e)

        _log.info("build_cache complete: %d/%d series cached", count, len(self._series_config))
        return count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_fred(self) -> Any:
        if self._fred is None:
            try:
                from fredapi import Fred
            except ImportError as err:
                raise ImportError(
                    "fredapi is required.  Install with: pip install fredapi"
                ) from err
            self._fred = Fred(api_key=self._api_key)
        return self._fred

    def _first_release_feature_frame(
        self,
        series_id: str,
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return ALFRED first-release observations in feature-frame form."""

        fred = self._get_fred()
        df = fred.get_series_first_release(series_id)
        if not isinstance(df, pd.DataFrame):
            s = pd.to_numeric(df, errors="coerce").dropna().sort_index()
            s.name = series_id
            frame = s.reset_index().rename(columns={"index": "as_of", series_id: "value"})
            frame["valid_from"] = frame["as_of"]
            frame["source"] = "alfred_first_release"
            frame["symbol"] = series_id
            frame["feature_name"] = series_id
            return feature_frame_from_records(frame)

        if df.empty:
            return pd.DataFrame()

        if "date" in df.columns:
            as_of_col = "date"
        elif "observation_date" in df.columns:
            as_of_col = "observation_date"
        else:
            raise ValueError("ALFRED response missing observation date column")

        valid_from_col = "realtime_start" if "realtime_start" in df.columns else as_of_col
        value_col = "value" if "value" in df.columns else df.columns[0]

        frame = df[[as_of_col, valid_from_col, value_col]].copy()
        frame = frame.rename(columns={as_of_col: "as_of", valid_from_col: "valid_from", value_col: "value"})
        frame["source"] = "alfred_first_release"
        frame["symbol"] = series_id
        frame["feature_name"] = series_id
        frame["as_of"] = pd.to_datetime(frame["as_of"], utc=True)
        frame["valid_from"] = pd.to_datetime(frame["valid_from"], utc=True)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["as_of", "valid_from", "value"])

        if start:
            frame = frame.loc[frame["as_of"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            frame = frame.loc[frame["as_of"] <= pd.Timestamp(end, tz="UTC")]

        return feature_frame_from_records(frame)

    def _fetch_series(
        self,
        series_id: str,
        start: str = "2000-01-01",
        end: str | None = None,
    ) -> pd.Series:
        fred = self._get_fred()
        kwargs: dict[str, Any] = {"observation_start": start}
        if end:
            kwargs["observation_end"] = end

        s = fred.get_series(series_id, **kwargs)
        if isinstance(s, pd.DataFrame):
            if "value" in s.columns:
                s = s["value"]
            else:
                s = s.iloc[:, 0]

        s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
        s.name = series_id
        return s

    def _cache(self, key: str, series: pd.Series) -> None:
        path = self._cache_path(key)
        df = series.to_frame()
        df.to_parquet(path, engine="pyarrow")
        _log.debug("cached: %s (%d rows)", path, len(df))

    def _cache_path(self, series_id: str) -> Path:
        return self._cache_dir / f"{series_id}.parquet"

    def _load_series_config(self) -> None:
        config_path = Path(__file__).resolve().parent / "fred_series_config.yaml"
        if not config_path.exists():
            _log.warning("fred_series_config.yaml not found at %s", config_path)
            return

        with open(config_path) as f:
            data = yaml.safe_load(f)

        for entry in data.get("series", []):
            self._series_config[entry["id"]] = entry
