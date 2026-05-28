"""Live data feed for paper trading — Alpaca IEX REST adapter.

Provides daily OHLCV snapshots in the same data format used by the
walk-forward backtest, so `strategy.on_data()` is called identically live
vs backtest (cardinal rule: one code path).

Assets are configured at construction time. Default: the 6 macro assets
used in the Phase 2 cross-sectional momentum validation.

Data format returned by get_daily_snapshot():
    {
        "close":  np.ndarray(shape=(n_periods, n_assets)),  # price matrix
        "symbols": list[str],                                # asset identifiers
    }

Usage:
    feed = LiveDataFeed(tickers=["SVXY", "SPXL", "TMF", "TLT", "SPY", "QQQ"])
    snapshot = feed.get_daily_snapshot(pd.Timestamp("2024-12-31"))
    signals  = strategy.on_data(snapshot)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default macro universe — matches Phase 2 validation
# ---------------------------------------------------------------------------
DEFAULT_MACRO_ASSETS: list[str] = [
    "SPY",    # S&P 500
    "QQQ",    # Nasdaq 100
    "IWM",    # Russell 2000
    "GLD",    # Gold
    "TLT",    # 20+ Year Treasury
    "DBC",    # Commodities
]

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
_CACHE_DIR = Path("quantcore/data/live_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _alpaca_headers(api_key: str | None, secret_key: str | None) -> dict[str, str]:
    key = api_key or os.environ.get("ALPACA_API_KEY", "")
    secret = secret_key or os.environ.get("ALPACA_API_SECRET", "")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _fetch_bars_raw(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1Day",
    api_key: str | None = None,
    secret_key: str | None = None,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """Fetch OHLCV bars from Alpaca REST API with retry logic.

    Uses the free IEX data tier (no paid subscription required).
    """
    base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    data_url = base.replace("paper-api.", "data.") + "/v2/stocks/bars"

    headers = _alpaca_headers(api_key, secret_key)
    # Alpaca Data API v2 requires RFC3339 format with timezone
    start_rfc = start + "Z" if "Z" not in start.upper() else start
    end_rfc = end + "Z" if "Z" not in end.upper() else end
    params: dict[str, str] = {
        "symbols": symbol,
        "start": start_rfc,
        "end": end_rfc,
        "timeframe": timeframe,
        "limit": "10000",
        "adjustment": "all",   # split/dividend adjusted
    }

    for attempt in range(1, max_retries + 1):
        try:
            import urllib.request

            # Build URL
            from urllib.parse import urlencode
            url = f"{data_url}?{urlencode(params)}"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                import json
                obj = json.loads(data.decode())
                bars = obj.get("bars", {}).get(symbol, [])
                log.debug("Fetched %d bars for %s", len(bars), symbol)
                return bars
        except Exception as exc:
            log.warning(
                "Alpaca bar fetch attempt %s/%s failed for %s: %s",
                attempt, max_retries, symbol, exc,
            )
            if attempt < max_retries:
                wait = 2 ** attempt
                log.info("Retrying in %ds...", wait)
                time.sleep(wait)
            else:
                log.error("All %d attempts failed for %s — returning empty", max_retries, symbol)
                return []

    return []


def _parse_bars(bars: list[dict[str, Any]]) -> pd.DataFrame:
    """Parse raw Alpaca bar list into a tz-naive DataFrame."""
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    records = []
    for b in bars:
        # Alpaca returns ISO8601 timestamps in UTC
        ts_raw = b.get("t", "") or b.get("timestamp", "")
        try:
            ts = pd.Timestamp(ts_raw).tz_localize(None)
        except Exception:
            ts = pd.Timestamp(ts_raw)

        records.append({
            "timestamp": ts,
            "open":  float(b.get("o", 0) or 0),
            "high":  float(b.get("h", 0) or 0),
            "low":   float(b.get("l", 0) or 0),
            "close": float(b.get("c", 0) or 0),
            "volume": int(b.get("v", 0) or 0),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.set_index("timestamp").sort_index()
    return df


class LiveDataFeed:
    """Live OHLCV data feed using Alpaca IEX REST.

    Fetches daily bars on demand and caches them locally to avoid
    redundant API calls.  The cache is a per-symbol parquet file keyed
    by date range.

    Parameters
    ----------
    tickers : list[str]
        Assets to track.  Must be tradeable on Alpaca.
    cache_dir : Path, optional
        Local cache directory.  Defaults to quantcore/data/live_cache/.
    api_key, secret_key : str, optional
        Override ALPACA_API_KEY / ALPACA_API_SECRET env vars.
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        cache_dir: Path | str | None = None,
        api_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._tickers: list[str] = list(tickers or DEFAULT_MACRO_ASSETS)
        self._cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = api_key
        self._secret_key = secret_key

        # In-memory cache for the current session: symbol → DataFrame
        self._bar_cache: dict[str, pd.DataFrame] = {}
        self._cache_meta: dict[str, tuple[date, date]] = {}  # symbol → (earliest, latest)

        log.info(
            "LiveDataFeed initialised for %d assets: %s",
            len(self._tickers), self._tickers,
        )

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """Return True if US equity markets are currently open.

        Uses Alpaca's calendar endpoint (free, no data subscription needed).
        """
        try:
            base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            headers = _alpaca_headers(self._api_key, self._secret_key)
            import urllib.request
            from urllib.parse import urlencode

            calendar_url = base.replace("paper-api.", "").rstrip("/") + "/v2/calendar"
            params = urlencode({"start": date.today().isoformat(), "end": date.today().isoformat()})
            req = urllib.request.Request(f"{calendar_url}?{params}", headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                import json
                calendars = json.loads(resp.read().decode()).get("calendars", [])
                if calendars:
                    return bool(calendars[0].get("is_open"))
        except Exception as exc:
            log.warning("is_market_open() check failed, assuming market open: %s", exc)
        return True  # degrade open — let strategies decide based on data availability

    def get_daily_snapshot(
        self,
        as_of_date: pd.Timestamp | date | str,
        lookback_days: int = 252,
    ) -> dict[str, NDArray]:
        """Fetch the trailing price matrix up to and including as_of_date.

        Returns the same dict structure expected by `strategy.on_data()`:
            {
                "close":   np.ndarray (n_periods, n_assets),
                "symbols": list[str],
            }

        Live: fetches from Alpaca REST and caches.
        Parity test: pass the same lookback window as the backtest so the
        signal computation covers the identical period.

        Parameters
        ----------
        as_of_date : pd.Timestamp | date | str
            The trading day to compute signals for.  Fetches up to this date.
        lookback_days : int
            Calendar days of history to include before as_of_date.
            Default 252 ≈ 1 trading year.
        """
        # Normalise date
        if isinstance(as_of_date, str):
            as_of_date = pd.Timestamp(as_of_date)
        elif isinstance(as_of_date, date):
            as_of_date = pd.Timestamp(as_of_date)
        as_of_ts = as_of_date

        start_ts = as_of_ts - pd.Timedelta(days=lookback_days)
        start_str = start_ts.isoformat()
        end_str   = as_of_ts.isoformat()

        close_frames: list[pd.DataFrame] = []

        for ticker in self._tickers:
            df = self._fetch_or_read_cache(ticker, start_str, end_str)
            if df.empty:
                # Fall back to synthetic / forward-fill missing bars
                df = self._fill_missing(ticker, df, start_ts, as_of_ts)
            close_frames.append(df[["close"]])

        if not close_frames:
            return {"close": np.empty((0, 0)), "symbols": []}

        # Align all tickers on a common date index (inner join)
        aligned = pd.concat(close_frames, axis=1, names=["symbol"])
        aligned.columns = self._tickers
        aligned = aligned.dropna()
        # Ensure sorted oldest → newest
        aligned = aligned.sort_index()

        n_periods, n_assets = aligned.shape
        if n_periods == 0 or n_assets == 0:
            return {"close": np.empty((0, len(self._tickers))), "symbols": self._tickers}

        close_matrix: NDArray = aligned.values.astype(np.float64)

        return {
            "close":   close_matrix,
            "symbols": self._tickers,
        }

    def get_intraday_snapshot(
        self,
        as_of_date: pd.Timestamp | date | str,
        bar_size: str = "5Min",
        lookback_days: int = 5,
    ) -> dict[str, NDArray]:
        """Fetch intraday bars up to and including as_of_date.

        Parameters
        ----------
        as_of_date : pd.Timestamp | date | str
            The trading day to fetch intraday data for.
        bar_size : str
            Alpaca timeframe string: "1Min", "5Min", "15Min", "1Hour".
        lookback_days : int
            Calendar days of intraday history before as_of_date. Default 5.

        Returns the same dict structure as get_daily_snapshot():
            {"close": np.ndarray (n_periods, n_assets), "symbols": list[str]}
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.Timestamp(as_of_date)
        elif isinstance(as_of_date, date):
            as_of_date = pd.Timestamp(as_of_date)
        as_of_ts = as_of_date

        start_ts = as_of_ts - pd.Timedelta(days=lookback_days)
        start_str = start_ts.isoformat()
        end_str = as_of_ts.isoformat()

        close_frames: list[pd.DataFrame] = []

        for ticker in self._tickers:
            bars = _fetch_bars_raw(
                symbol=ticker,
                start=start_str,
                end=end_str,
                timeframe=bar_size,
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            df = _parse_bars(bars)
            if df.empty:
                dates = pd.date_range(start_ts, as_of_ts, freq="5min")
                df = pd.DataFrame({"close": 0.0}, index=dates)
            close_frames.append(df[["close"]])

        if not close_frames:
            return {"close": np.empty((0, 0)), "symbols": []}

        aligned = pd.concat(close_frames, axis=1, names=["symbol"])
        aligned.columns = self._tickers
        aligned = aligned.dropna()
        aligned = aligned.sort_index()

        n_periods, n_assets = aligned.shape
        if n_periods == 0 or n_assets == 0:
            return {"close": np.empty((0, len(self._tickers))), "symbols": self._tickers}

        return {
            "close": aligned.values.astype(np.float64),
            "symbols": self._tickers,
        }

    def get_intraday_ohlcv_snapshot(
        self,
        as_of_date: pd.Timestamp | date | str,
        bar_size: str = "5Min",
        lookback_days: int = 5,
    ) -> dict[str, NDArray | list[str]]:
        """Fetch intraday OHLCV bars (not just close) up to and including as_of_date.

        Returns:
            {
                "open":    np.ndarray (n_periods, n_assets),
                "high":    np.ndarray (n_periods, n_assets),
                "low":     np.ndarray (n_periods, n_assets),
                "close":   np.ndarray (n_periods, n_assets),
                "volume":  np.ndarray (n_periods, n_assets),
                "symbols": list[str],
            }
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.Timestamp(as_of_date)
        elif isinstance(as_of_date, date):
            as_of_date = pd.Timestamp(as_of_date)
        as_of_ts = as_of_date

        start_ts = as_of_ts - pd.Timedelta(days=lookback_days)
        start_str = start_ts.isoformat()
        end_str = as_of_ts.isoformat()

        field_frames: dict[str, list[pd.DataFrame]] = {
            "open": [], "high": [], "low": [], "close": [], "volume": [],
        }

        for ticker in self._tickers:
            bars = _fetch_bars_raw(
                symbol=ticker,
                start=start_str,
                end=end_str,
                timeframe=bar_size,
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
            df = _parse_bars(bars)
            if df.empty:
                dates = pd.date_range(start_ts, as_of_ts, freq="5min")
                df = pd.DataFrame(
                    {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0},
                    index=dates,
                )
            for field in field_frames:
                field_frames[field].append(df[[field]])

        if not field_frames["close"]:
            empty = np.empty((0, 0))
            return {
                "open": empty, "high": empty, "low": empty,
                "close": empty, "volume": empty, "symbols": [],
            }

        result: dict[str, NDArray | list[str]] = {"symbols": self._tickers}
        for field, frames in field_frames.items():
            aligned = pd.concat(frames, axis=1, names=["symbol"])
            aligned.columns = self._tickers
            aligned = aligned.dropna().sort_index()
            if field == "volume":
                result[field] = aligned.values.astype(np.int64)
            else:
                result[field] = aligned.values.astype(np.float64)

        return result

    def get_prev_close_map(
        self,
        as_of_date: pd.Timestamp | date | str,
    ) -> dict[str, float]:
        """Return {ticker: previous_trading_day_close} for each tracked ticker.

        Fetches daily bars with 10-day lookback and returns the second-to-last
        close (i.e. yesterday's close relative to as_of_date).
        """
        if isinstance(as_of_date, str):
            as_of_date = pd.Timestamp(as_of_date)
        elif isinstance(as_of_date, date):
            as_of_date = pd.Timestamp(as_of_date)

        start_str = (as_of_date - pd.Timedelta(days=10)).isoformat()
        end_str = as_of_date.isoformat()

        prev_close: dict[str, float] = {}
        for ticker in self._tickers:
            df = self._fetch_or_read_cache(ticker, start_str, end_str)
            if df.empty or len(df) < 2:
                continue
            # Second-to-last row is the previous trading day
            prev_close[ticker] = float(df["close"].iloc[-2])

        return prev_close

    def get_latest_close(self, ticker: str) -> float | None:
        """Return today's latest close price for ticker, or None if not loaded."""
        df = self._bar_cache.get(ticker)
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_or_read_cache(
        self,
        ticker: str,
        start_str: str,
        end_str: str,
    ) -> pd.DataFrame:
        """Fetch bars from Alpaca or serve from local parquet cache."""
        cache_path = self._cache_dir / f"{ticker}_{start_str[:10]}_{end_str[:10]}.parquet"

        # Check in-memory cache first
        if ticker in self._bar_cache:
            df_cached = self._bar_cache[ticker]
            cached_range = self._cache_meta.get(ticker)
            if cached_range is not None:
                start_dt = pd.Timestamp(start_str)
                end_dt   = pd.Timestamp(end_str)
                cs, ce = cached_range
                cached_start = pd.Timestamp(cs) if isinstance(cs, date) else cs
                cached_end   = pd.Timestamp(ce) if isinstance(ce, date) else ce
                if cached_start <= start_dt and cached_end >= end_dt:
                    return df_cached.loc[start_dt:end_dt]

        # Check disk cache
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                    self._bar_cache[ticker] = df
                    cached_start = df.index.min()
                    cached_end   = df.index.max()
                    if cached_start and cached_end:
                        self._cache_meta[ticker] = (cached_start.date(), cached_end.date())
                    log.debug("Served %s from disk cache (%s bar count: %d)", ticker, cache_path, len(df))
                    return df
            except Exception as exc:
                log.warning("Cache read failed for %s: %s", ticker, exc)

        # Fetch from Alpaca
        log.info("Fetching %s bars from Alpaca (%s → %s)", ticker, start_str[:10], end_str[:10])
        raw_bars = _fetch_bars_raw(
            symbol=ticker,
            start=start_str,
            end=end_str,
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        df = _parse_bars(raw_bars)

        if not df.empty:
            # Persist to disk
            try:
                df.to_parquet(cache_path)
                log.debug("Cached %d bars for %s at %s", len(df), ticker, cache_path)
            except Exception as exc:
                log.warning("Cache write failed for %s: %s", ticker, exc)

        # Update in-memory
        if not df.empty:
            self._bar_cache[ticker] = df
            self._cache_meta[ticker] = (df.index.min().date(), df.index.max().date())

        return df

    def _fill_missing(
        self,
        ticker: str,
        df: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame:
        """Forward-fill gaps when a ticker has partial history in the cache."""
        if df.empty:
            # No data at all — return zero-filled series (strategy will skip)
            dates = pd.date_range(start_ts, end_ts, freq="B")
            return pd.DataFrame({"close": 0.0}, index=dates)

        # Reindex to business days, forward-filling the last known close
        dates = pd.date_range(start_ts, end_ts, freq="B")
        df = df.reindex(dates)
        df["close"] = df["close"].ffill()
        # Only keep rows that have at least been observed
        df = df.dropna(subset=["close"])
        return df