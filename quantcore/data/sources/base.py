"""Abstract DataSource base class and Yahoo Finance adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DataSourceConfig:
    api_key: str = ""
    rate_limit: int = 5  # requests per minute
    retry_attempts: int = 3
    retry_delay: float = 1.0


class DataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, config: DataSourceConfig | None = None) -> None:
        self.config = config or DataSourceConfig()
        self._last_request_time: float = 0.0

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol and date range."""
        ...


    @abstractmethod
    async def fetch_options(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain data for a symbol."""
        ...


    @abstractmethod
    def supported_assets(self) -> list[str]:
        """Return list of supported asset symbols/identifiers."""
        ...


    async def close(self) -> None:
        """Release any held resources (connections, pools, etc.)."""

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import asyncio
        import time
        now = time.monotonic()
        min_interval = 60.0 / self.config.rate_limit
        elapsed = now - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    async def _retry(self, fn, *args: Any, **kwargs: Any) -> Any:
        """Execute a function with retry and exponential backoff."""
        import asyncio
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                delay = self.config.retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
        raise last_error  # type: ignore


class YahooDataSource(DataSource):
    """Yahoo Finance data source using yfinance."""

    def __init__(self, config: DataSourceConfig | None = None) -> None:
        super().__init__(config)
        self.config.rate_limit = 60  # yfinance is generous

    async def fetch_ohlcv(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance."""
        import yfinance as yf

        await self._rate_limit()

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["symbol"] = symbol
        df = df.rename(columns={
            "date": "timestamp",
            "datetimes_etc": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "stock_splits": "split_factor",
        })

        # Ensure timestamp column
        if "timestamp" not in df.columns:
            for col in ["date", "datetime"]:
                if col in df.columns:
                    df = df.rename(columns={col: "timestamp"})
                    break

        return df

    async def fetch_options(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain from Yahoo Finance."""
        import yfinance as yf

        await self._rate_limit()

        ticker = yf.Ticker(symbol)
        expirations = ticker.options

        all_chains = []
        for exp in expirations[:3]:  # Limit to first 3 expirations
            chain = ticker.option_chain(exp)
            calls = chain.calls.assign(expiry=exp, right="C")
            puts = chain.puts.assign(expiry=exp, right="P")
            all_chains.append(calls)
            all_chains.append(puts)

        if all_chains:
            return pd.concat(all_chains, ignore_index=True)
        return pd.DataFrame()

    def supported_assets(self) -> list[str]:
        return ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
                "IWM", "QQQ", "DIA", "TLT", "GLD", "SLV", "VIX"]
