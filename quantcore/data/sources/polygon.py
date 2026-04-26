"""Polygon.io data source adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import pandas as pd

from quantcore.data.sources.base import DataSource, DataSourceConfig


class PolygonDataSource(DataSource):
    """Polygon.io REST API data source."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str = "", rate_limit: int = 5) -> None:
        config = DataSourceConfig(api_key=api_key, rate_limit=rate_limit)
        super().__init__(config)
        self.api_key = api_key or self.config.api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        await super().close()

    async def fetch_ohlcv(
        self,
        symbol: str,
        start: str | datetime,
        end: str | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV aggregates from Polygon."""
        await self._rate_limit()

        multiplier, timespan = self._parse_interval(interval)
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"

        client = await self._get_client()
        resp = await client.get(url, params={"apiKey": self.api_key, "limit": 50000})
        resp.raise_for_status()
        data = resp.json()

        if "results" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        df = df.rename(columns={
            "t": "timestamp", "o": "open", "h": "high",
            "l": "low", "c": "close", "v": "volume",
            "n": "transactions", "vw": "vwap",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol

        return df

    async def fetch_options(self, symbol: str) -> pd.DataFrame:
        """Fetch options chain from Polygon."""
        await self._rate_limit()

        url = f"{self.BASE_URL}/v3/reference/options/contracts"
        client = await self._get_client()
        resp = await client.get(url, params={
            "underlying_ticker": symbol, "apiKey": self.api_key, "limit": 1000
        })
        resp.raise_for_status()
        data = resp.json()

        if "results" not in data:
            return pd.DataFrame()

        return pd.DataFrame(data["results"])

    async def fetch_trades(self, symbol: str, date: str) -> pd.DataFrame:
        """Fetch tick-level trade data."""
        url = f"{self.BASE_URL}/v3/trades/{symbol}"
        client = await self._get_client()
        resp = await client.get(url, params={
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "apiKey": self.api_key, "limit": 50000,
        })
        resp.raise_for_status()
        data = resp.json()

        if "results" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        return df

    def supported_assets(self) -> list[str]:
        return []  # Polygon supports all US equities

    @staticmethod
    def _parse_interval(interval: str) -> tuple[int, str]:
        """Parse interval string like '1d', '5m', '1h' into (multiplier, timespan)."""
        import re
        match = re.match(r"(\d+)([mhd])", interval)
        if not match:
            return 1, "day"
        mult = int(match.group(1))
        unit = {"m": "minute", "h": "hour", "d": "day"}[match.group(2)]
        return mult, unit
