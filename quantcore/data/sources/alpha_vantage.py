"""Alpha Vantage and FRED data source adapters."""

from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd

from quantcore.data.sources.base import DataSource, DataSourceConfig


class AlphaVantageDataSource(DataSource):
    """Alpha Vantage data source for fundamentals and earnings."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str = "", rate_limit: int = 5) -> None:
        config = DataSourceConfig(api_key=api_key, rate_limit=rate_limit)
        super().__init__(config)
        self.api_key = api_key
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
        self, symbol: str, start: str | datetime, end: str | datetime, interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch daily OHLCV from Alpha Vantage."""
        await self._rate_limit()

        func = "TIME_SERIES_DAILY" if interval == "1d" else "TIME_SERIES_INTRADAY"
        params = {"function": func, "symbol": symbol, "apikey": self.api_key, "outputsize": "full"}
        if interval != "1d":
            params["interval"] = interval

        client = await self._get_client()
        resp = await client.get(self.BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        key = [k for k in data if "Time Series" in k]
        if not key:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[key[0]], orient="index")
        df.index = pd.to_datetime(df.index)
        df.columns = [c.split(". ")[-1].lower().replace(" ", "_") for c in df.columns]
        df = df.reset_index().rename(columns={"index": "timestamp"})
        df["symbol"] = symbol
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    async def fetch_fundamentals(self, symbol: str, statement: str = "income") -> pd.DataFrame:
        """Fetch fundamental data (income, balance, cashflow)."""
        func_map = {"income": "INCOME_STATEMENT", "balance": "BALANCE_SHEET", "cashflow": "CASH_FLOW"}
        params = {"function": func_map[statement], "symbol": symbol, "apikey": self.api_key}

        client = await self._get_client()
        resp = await client.get(self.BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        f"annual{statement.capitalize()}" if "annualReports" in data else f"quarterly{statement.capitalize()}"
        reports = data.get("annualReports", []) or data.get("quarterlyReports", [])
        return pd.DataFrame(reports)

    async def fetch_options(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame()  # Alpha Vantage doesn't support options natively

    def supported_assets(self) -> list[str]:
        return []


class FREDDataSource(DataSource):
    """Federal Reserve Economic Data source."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str = "", rate_limit: int = 120) -> None:
        config = DataSourceConfig(api_key=api_key, rate_limit=rate_limit)
        super().__init__(config)
        self.api_key = api_key
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

    async def fetch_series(self, series_id: str, start: str = "2000-01-01", end: str = "2024-12-31") -> pd.DataFrame:
        """Fetch a time series from FRED by series ID."""
        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id, "api_key": self.api_key,
            "observation_start": start, "observation_end": end,
            "file_type": "json",
        }

        client = await self._get_client()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        df = pd.DataFrame(observations)
        if "date" in df.columns and "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"date": "timestamp"})

        return df

    async def fetch_yield_curve(self, start: str = "2000-01-01") -> pd.DataFrame:
        """Fetch US Treasury yield curve data."""
        series = {
            "DGS2": "2Y", "DGS5": "5Y", "DGS10": "10Y", "DGS30": "30Y",
            "DFF": "FedFunds", "CPIAUCSL": "CPI", "UNRATE": "Unemployment",
            "GDP": "GDP",
        }
        dfs = {}
        for sid, name in series.items():
            df = await self.fetch_series(sid, start=start)
            if not df.empty:
                dfs[name] = df.set_index("timestamp")["value"]

        if dfs:
            return pd.DataFrame(dfs)
        return pd.DataFrame()

    async def fetch_ohlcv(self, symbol: str, start: str | datetime, end: str | datetime, interval: str = "1d") -> pd.DataFrame:
        return await self.fetch_series(symbol, str(start), str(end))

    async def fetch_options(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame()

    def supported_assets(self) -> list[str]:
        return ["DGS2", "DGS5", "DGS10", "DGS30", "DFF", "CPIAUCSL", "UNRATE", "GDP"]
