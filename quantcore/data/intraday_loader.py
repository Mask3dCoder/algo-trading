"""Intraday 1-minute bar data loader with Alpaca Markets integration.

Downloads and caches 1-minute OHLCV bars from Alpaca's IEX data feed.
Stores as Parquet at data/intraday/{ticker}/{year}/{month:02d}/{day:02d}.parquet (snappy).

Requires ALPACA_API_KEY and ALPACA_API_SECRET environment variables.
Free tier (IEX): 1000 bars per request, 200 requests/minute.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = _ROOT / "data" / "intraday"


class AlpacaIntradayLoader:
    """Download, cache, and load 1-minute bars from Alpaca Markets.

    Uses alpaca-py StockHistoricalDataClient with IEX data feed.
    Parquet partitioning: data/intraday/{ticker}/{year}/{month:02d}/{day:02d}.parquet
    Supports incremental updates — only downloads missing trading days.
    """

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        api_key: str = "",
        secret_key: str = "",
        rate_limit_sleep: float = 0.3,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_API_SECRET", "")
        self.rate_limit_sleep = rate_limit_sleep
        self._last_request_time = 0.0
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Alpaca historical data client."""
        if self._client is None:
            if not self.api_key or not self.secret_key:
                raise RuntimeError(
                    "Alpaca API keys not set. Export ALPACA_API_KEY and "
                    "ALPACA_API_SECRET environment variables."
                )
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
        return self._client

    # ── Parquet storage ─────────────────────────────────────────────────

    def _parquet_path(self, ticker: str, dt: date) -> Path:
        return self.data_dir / ticker / str(dt.year) / f"{dt.month:02d}" / f"{dt.day:02d}.parquet"

    def _save_day(self, ticker: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["_date"] = df["timestamp"].dt.date
        for dt, group in df.groupby("_date"):
            path = self._parquet_path(ticker, dt)
            path.parent.mkdir(parents=True, exist_ok=True)
            group.drop(columns=["_date"]).to_parquet(path, index=False, compression="snappy")

    def _load_cached(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Load cached parquet files for a date range.

        Reads by month directory (not per-day) for performance:
        ~16 tickers × 12 months = ~192 reads vs ~5,840 per-day reads.
        """
        ticker_dir = self.data_dir / ticker
        if not ticker_dir.exists():
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        # Collect unique (year, month) pairs in the range
        months_seen: set[tuple[int, int]] = set()
        current = start
        while current <= end:
            months_seen.add((current.year, current.month))
            # Advance to first day of next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        for year, month in sorted(months_seen):
            month_dir = ticker_dir / str(year) / f"{month:02d}"
            if not month_dir.exists():
                continue
            for f in sorted(month_dir.iterdir()):
                if f.suffix != ".parquet":
                    continue
                try:
                    day = int(f.stem)
                    file_date = date(year, month, day)
                    if start <= file_date <= end:
                        frames.append(pd.read_parquet(f))
                except (ValueError, OSError):
                    continue

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def _get_cached_dates(self, ticker: str, start: date, end: date) -> set[date]:
        """Get set of cached dates by scanning month directories."""
        ticker_dir = self.data_dir / ticker
        if not ticker_dir.exists():
            return set()

        cached: set[date] = set()
        months_seen: set[tuple[int, int]] = set()
        current = start
        while current <= end:
            months_seen.add((current.year, current.month))
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        for year, month in sorted(months_seen):
            month_dir = ticker_dir / str(year) / f"{month:02d}"
            if not month_dir.exists():
                continue
            for f in month_dir.iterdir():
                if f.suffix != ".parquet":
                    continue
                try:
                    d = date(year, month, int(f.stem))
                    if start <= d <= end:
                        cached.add(d)
                except (ValueError, OSError):
                    continue
        return cached

    def _find_last_cached_date(self, ticker: str, before: date) -> date | None:
        ticker_dir = self.data_dir / ticker
        if not ticker_dir.exists():
            return None
        latest: date | None = None
        for year_dir in sorted(ticker_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue
                for f in month_dir.iterdir():
                    if f.suffix == ".parquet":
                        try:
                            d = date(int(year_dir.name), int(month_dir.name), int(f.stem))
                            if d <= before and (latest is None or d > latest):
                                latest = d
                        except ValueError:
                            continue
        return latest

    # ── Alpaca API ──────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_sleep:
            time.sleep(self.rate_limit_sleep - elapsed)
        self._last_request_time = time.time()

    def _fetch_alpaca_bars(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch 1-minute bars from Alpaca with chunked pagination.

        Alpaca free tier returns max 1000 bars per request.
        We paginate by advancing the start date after each chunk.
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = self._get_client()
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        # Ensure timezone-aware (UTC) for comparison with Alpaca bar timestamps
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)

        all_bars: list[dict] = []
        current_start = start_dt

        while current_start < end_dt:
            self._rate_limit()
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=current_start,
                end=end_dt,
                limit=1000,
            )
            try:
                response = client.get_stock_bars(request)
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "limit" in error_str:
                    log.warning("Rate limited on %s, sleeping 5s", ticker)
                    time.sleep(5)
                    continue
                raise

            bars = response.data.get(ticker, [])
            if not bars:
                break

            for bar in bars:
                all_bars.append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                })

            # Advance start past the last bar we received
            last_ts = bars[-1].timestamp
            if isinstance(last_ts, datetime):
                current_start = last_ts + timedelta(minutes=1)
            else:
                current_start = pd.Timestamp(last_ts).to_pydatetime() + timedelta(minutes=1)

            # If we got fewer than 1000 bars, we've reached the end
            if len(bars) < 1000:
                break

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[cols]
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ── Public API ──────────────────────────────────────────────────────

    def download(
        self,
        ticker: str,
        start: str = "2023-01-01",
        end: str = "2024-12-31",
    ) -> pd.DataFrame:
        """Download 1-minute bars from Alpaca and cache to parquet."""
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        cached = self._load_cached(ticker, start_date, end_date)

        if not cached.empty:
            cached_start = cached["timestamp"].dt.date.min()
            cached_end = cached["timestamp"].dt.date.max()
            if cached_start <= start_date and cached_end >= end_date:
                log.info("Fully cached: %s (%s -> %s)", ticker, start, end)
                return cached

        log.info("Downloading %s 1min bars: %s -> %s", ticker, start, end)
        df = self._fetch_alpaca_bars(ticker, start, end)

        if not df.empty:
            self._save_day(ticker, df)
            log.info("Saved %d bars for %s", len(df), ticker)
        else:
            log.warning("No data returned for %s (%s -> %s)", ticker, start, end)

        return self._load_cached(ticker, start_date, end_date)

    def update_ticker(
        self,
        ticker: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Incremental update: download from last cached date to `end`.

        If no cached data exists, does a full download from 2023-01-01.
        """
        end_date = date.fromisoformat(end) if end else date.today()
        last_cached = self._find_last_cached_date(ticker, end_date)

        if last_cached is None:
            return self.download(ticker, "2023-01-01", end_date.isoformat())

        start = (last_cached + timedelta(days=1)).isoformat()
        if start > end_date.isoformat():
            log.info("Already up to date: %s (last cached %s)", ticker, last_cached)
            return self._load_cached(ticker, date(2023, 1, 1), end_date)

        df = self._fetch_alpaca_bars(ticker, start, end_date.isoformat())
        if not df.empty:
            self._save_day(ticker, df)
            log.info("Updated %s: %d new bars (%s -> %s)", ticker, len(df), start, end_date)
        else:
            log.info("No new data for %s (%s -> %s)", ticker, start, end_date)

        return self._load_cached(ticker, date(2023, 1, 1), end_date)

    def download_universe(
        self,
        tickers: list[str],
        start: str = "2023-01-01",
        end: str = "2024-12-31",
    ) -> dict[str, pd.DataFrame]:
        """Download bars for multiple tickers with progress logging."""
        results: dict[str, pd.DataFrame] = {}
        for i, ticker in enumerate(tickers):
            log.info("[%d/%d] %s", i + 1, len(tickers), ticker)
            try:
                df = self.download(ticker, start, end)
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                log.error("Failed %s: %s", ticker, e)
        log.info("Downloaded %d / %d tickers", len(results), len(tickers))
        return results

    def update_universe(
        self,
        tickers: list[str],
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Incremental update for multiple tickers."""
        results: dict[str, pd.DataFrame] = {}
        for i, ticker in enumerate(tickers):
            log.info("[%d/%d] %s", i + 1, len(tickers), ticker)
            try:
                df = self.update_ticker(ticker, end)
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                log.error("Failed %s: %s", ticker, e)
        log.info("Updated %d / %d tickers", len(results), len(tickers))
        return results

    def load_range(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Load minute bars from cache for a date range."""
        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if isinstance(end, str):
            end = datetime.fromisoformat(end.replace("Z", "+00:00"))

        start_date = start.date() if hasattr(start, "date") else start
        end_date = end.date() if hasattr(end, "date") else end

        df = self._load_cached(ticker, start_date, end_date)
        if df.empty:
            return df

        start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)

        mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
        return df.loc[mask].reset_index(drop=True)

    def load_snapshot(
        self,
        tickers: list[str],
        timestamp: str | datetime,
    ) -> pd.DataFrame:
        """Load the latest bar for each ticker at or before the given minute."""
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        ts = pd.Timestamp(timestamp, tz="UTC") if pd.Timestamp(timestamp).tzinfo is None else pd.Timestamp(timestamp)

        rows: dict[str, dict] = {}
        for ticker in tickers:
            try:
                start_date = ts.date()
                df = self._load_cached(ticker, start_date, start_date)
                if df.empty:
                    continue
                before = df[df["timestamp"] <= ts]
                if before.empty:
                    continue
                last = before.iloc[-1]
                rows[ticker] = {
                    "open": last["open"], "high": last["high"], "low": last["low"],
                    "close": last["close"], "volume": last["volume"],
                }
            except Exception:
                continue

        return pd.DataFrame.from_dict(rows, orient="index")

    def load_universe_range(
        self,
        tickers: list[str],
        start: str | datetime,
        end: str | datetime,
    ) -> pd.DataFrame:
        """Load minute bars for multiple tickers as long-format DataFrame.

        Returns columns: [symbol, timestamp, open, high, low, close, volume].
        Compatible with _df_to_data() in run_backtest_focused.py.
        """
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            df = self.load_range(ticker, start, end)
            if not df.empty:
                df = df.copy()
                df["symbol"] = ticker
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return result

    # ── Universe building ───────────────────────────────────────────────

    def build_universe(
        self,
        min_adv: float = 1_000_000,
        min_price: float = 5.0,
        as_of: date | None = None,
    ) -> list[str]:
        """Filter S&P 500 by ADV and price using yfinance daily data.

        Does NOT require intraday data — bootstraps from yfinance.
        """
        try:
            from quantcore.data.sources.historical_universe import HistoricalUniverseLoader
            loader = HistoricalUniverseLoader()
            tickers = loader._sp500()
        except Exception:
            tickers = [
                "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK.B", "UNH",
                "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
                "LLY", "PEP", "COST", "KO", "AVGO", "WMT", "MCD", "CSCO", "TMO",
                "ACN", "ABT", "DHR", "NEE", "LIN", "PM", "TXN", "UNP", "RTX",
                "LOW", "HON", "AMGN", "IBM", "CAT", "GE", "SPGI", "BA", "GILD",
                "DE", "INTC", "AMD", "QCOM", "AMAT",
            ]

        if as_of is None:
            as_of = date.today()

        try:
            import yfinance as yf
            end_str = as_of.isoformat()
            start_str = (as_of - timedelta(days=30)).isoformat()
            data = yf.download(tickers, start=start_str, end=end_str, progress=False)
            if data.empty:
                log.warning("yfinance returned empty data, returning raw ticker list")
                return sorted(tickers)

            close = data["Close"] if "Close" in data.columns.get_level_values(0) else data[["Close"]]
            volume = data["Volume"] if "Volume" in data.columns.get_level_values(0) else data[["Volume"]]

            if isinstance(close.columns, pd.MultiIndex):
                close = close.droplevel(0, axis=1)
                volume = volume.droplevel(0, axis=1)

            adv = (close * volume).tail(20).mean()
            last_price = close.iloc[-1]

            mask = (adv >= min_adv) & (last_price >= min_price)
            result = sorted(adv[mask].dropna().index.tolist())
            log.info("Universe filter: %d / %d tickers pass (ADV >= %.0f, price >= %.2f)",
                     len(result), len(tickers), min_adv, min_price)
            return result
        except Exception as e:
            log.warning("Universe filter failed (%s), returning raw list", e)
            return sorted(tickers)


# Backward compatibility aliases
IntradayDataLoader = AlpacaIntradayLoader
PolygonIntradayLoader = AlpacaIntradayLoader
