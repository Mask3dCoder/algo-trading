"""Options chain data loader with Alpaca Markets integration.

Downloads and caches options chain snapshots from Alpaca's Options API.
Stores as Parquet at data/options/{underlying}/{date}.parquet (snappy).

Requires ALPACA_API_KEY and ALPACA_API_SECRET environment variables.
Alpaca free tier: Options Chain endpoint (IEX feed), rate-limited.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = _ROOT / "data" / "options"


class OptionsDataLoader:
    """Download, cache, and load options chain data from Alpaca Markets.

    Uses alpaca-py OptionHistoricalDataClient.
    Parquet partitioning: data/options/{underlying}/{date}.parquet
    Each parquet file contains the full options chain snapshot for that day.
    """

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        api_key: str = "",
        secret_key: str = "",
        rate_limit_sleep: float = 0.5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_API_SECRET", "")
        self.rate_limit_sleep = rate_limit_sleep
        self._last_request_time = 0.0
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Alpaca options historical data client."""
        if self._client is None:
            if not self.api_key or not self.secret_key:
                raise RuntimeError(
                    "Alpaca API keys not set. Export ALPACA_API_KEY and "
                    "ALPACA_API_SECRET environment variables."
                )
            from alpaca.data.historical import OptionHistoricalDataClient
            self._client = OptionHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
        return self._client

    # ── Parquet storage ─────────────────────────────────────────────────

    def _parquet_path(self, underlying: str, dt: date) -> Path:
        return self.data_dir / underlying / f"{dt.isoformat()}.parquet"

    def _save_chain(self, underlying: str, dt: date, df: pd.DataFrame) -> None:
        if df.empty:
            return
        path = self._parquet_path(underlying, dt)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")

    def _load_cached(self, underlying: str, start: date, end: date) -> pd.DataFrame:
        """Load cached parquet files for a date range."""
        underlying_dir = self.data_dir / underlying
        if not underlying_dir.exists():
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for f in sorted(underlying_dir.iterdir()):
            if f.suffix != ".parquet":
                continue
            try:
                file_date = date.fromisoformat(f.stem)
                if start <= file_date <= end:
                    frames.append(pd.read_parquet(f))
            except (ValueError, OSError):
                continue

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return df

    def _get_cached_dates(self, underlying: str, start: date, end: date) -> set[date]:
        """Get set of cached dates."""
        underlying_dir = self.data_dir / underlying
        if not underlying_dir.exists():
            return set()

        cached: set[date] = set()
        for f in underlying_dir.iterdir():
            if f.suffix != ".parquet":
                continue
            try:
                d = date.fromisoformat(f.stem)
                if start <= d <= end:
                    cached.add(d)
            except (ValueError, OSError):
                continue
        return cached

    def _find_last_cached_date(self, underlying: str, before: date) -> date | None:
        underlying_dir = self.data_dir / underlying
        if not underlying_dir.exists():
            return None
        latest: date | None = None
        for f in underlying_dir.iterdir():
            if f.suffix != ".parquet":
                continue
            try:
                d = date.fromisoformat(f.stem)
                if d <= before and (latest is None or d > latest):
                    latest = d
            except (ValueError, OSError):
                continue
        return latest

    # ── Trading day generation ──────────────────────────────────────────

    @staticmethod
    def _trading_days(start: date, end: date) -> list[date]:
        """Generate weekday dates in [start, end] (conservative — no holiday calendar)."""
        days: list[date] = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Mon-Fri
                days.append(current)
            current += timedelta(days=1)
        return days

    # ── Alpaca API ──────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_sleep:
            time.sleep(self.rate_limit_sleep - elapsed)
        self._last_request_time = time.time()

    def _fetch_chain_snapshot(
        self,
        underlying: str,
        trade_date: date,
    ) -> pd.DataFrame:
        """Fetch options chain snapshot for a single date via Alpaca.

        Uses get_option_chain() which returns the full chain with greeks,
        IV, bid/ask, volume, open_interest for each contract.
        """
        from alpaca.data.requests import OptionChainRequest

        client = self._get_client()
        self._rate_limit()

        try:
            request = OptionChainRequest(
                underlying_symbol=underlying,
                feed="indicative",  # Free tier
            )
            response = client.get_option_chain(request)

            if not response:
                return pd.DataFrame()

            rows: list[dict] = []
            for symbol, snapshot in response.items():
                if snapshot is None:
                    continue

                # Parse option symbol: e.g. SPY240119C00450000
                row: dict = {
                    "option_symbol": symbol,
                    "underlying": underlying,
                    "date": trade_date,
                }

                # Parse contract type from symbol
                if "C" in symbol.split(underlying)[1] if underlying in symbol else "":
                    # crude parse — Alpaca symbols follow OCC format
                    pass

                # Snapshot data
                if hasattr(snapshot, "latest_trade") and snapshot.latest_trade:
                    row["last_price"] = float(snapshot.latest_trade.price) if snapshot.latest_trade.price else None
                    row["last_trade_size"] = int(snapshot.latest_trade.size) if snapshot.latest_trade.size else None

                if hasattr(snapshot, "latest_quote") and snapshot.latest_quote:
                    row["bid"] = float(snapshot.latest_quote.bid_price) if snapshot.latest_quote.bid_price else None
                    row["ask"] = float(snapshot.latest_quote.ask_price) if snapshot.latest_quote.ask_price else None
                    row["bid_size"] = int(snapshot.latest_quote.bid_size) if snapshot.latest_quote.bid_size else None
                    row["ask_size"] = int(snapshot.latest_quote.ask_size) if snapshot.latest_quote.ask_size else None

                if hasattr(snapshot, "implied_volatility") and snapshot.implied_volatility:
                    row["implied_volatility"] = float(snapshot.implied_volatility)

                if hasattr(snapshot, "greeks") and snapshot.greeks:
                    g = snapshot.greeks
                    row["delta"] = float(g.delta) if g.delta else None
                    row["gamma"] = float(g.gamma) if g.gamma else None
                    row["theta"] = float(g.theta) if g.theta else None
                    row["vega"] = float(g.vega) if g.vega else None
                    row["rho"] = float(g.rho) if g.rho else None

                if hasattr(snapshot, "open_interest") and snapshot.open_interest:
                    row["open_interest"] = int(snapshot.open_interest)

                if hasattr(snapshot, "volume") and snapshot.volume:
                    row["volume"] = int(snapshot.volume)

                # Parse strike, expiry, type from OCC symbol
                try:
                    parts = symbol.replace(underlying, "")
                    # OCC format: YYMMDD + C/P + strike (8 digits, 5 implied decimals)
                    expiry_str = parts[:6]
                    contract_type = parts[6]
                    strike_str = parts[7:]
                    row["expiration"] = datetime.strptime(expiry_str, "%y%m%d").date()
                    row["contract_type"] = "call" if contract_type == "C" else "put"
                    row["strike"] = int(strike_str) / 1000.0
                except (ValueError, IndexError):
                    pass

                rows.append(row)

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            return df

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str:
                log.warning("Rate limited on %s chain, sleeping 5s", underlying)
                time.sleep(5)
                return self._fetch_chain_snapshot(underlying, trade_date)
            log.error("Failed to fetch chain for %s on %s: %s", underlying, trade_date, e)
            return pd.DataFrame()

    # ── Public API ──────────────────────────────────────────────────────

    def download(
        self,
        underlying: str,
        start: str = "2023-01-01",
        end: str = "2024-12-31",
    ) -> pd.DataFrame:
        """Download options chain snapshots and cache to parquet.

        Note: Alpaca's free tier get_option_chain() returns the CURRENT chain,
        not historical. For historical chains, we download daily and cache.
        For backtesting, we use the latest available chain per date.
        """
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        cached = self._load_cached(underlying, start_date, end_date)

        if not cached.empty:
            cached_dates = set(cached["date"].unique()) if "date" in cached.columns else set()
            needed = set(self._trading_days(start_date, end_date))
            if needed.issubset(cached_dates):
                log.info("Fully cached: %s (%s -> %s)", underlying, start, end)
                return cached

        log.info("Downloading %s options chain: %s -> %s", underlying, start, end)
        trading_days = self._trading_days(start_date, end_date)
        cached_dates = self._get_cached_dates(underlying, start_date, end_date)
        missing = [d for d in trading_days if d not in cached_dates]

        if not missing:
            log.info("All dates cached for %s", underlying)
            return self._load_cached(underlying, start_date, end_date)

        log.info("Need to fetch %d dates for %s", len(missing), underlying)
        fetched = 0
        for dt in missing:
            df = self._fetch_chain_snapshot(underlying, dt)
            if not df.empty:
                self._save_chain(underlying, dt, df)
                fetched += 1
                if fetched % 10 == 0:
                    log.info("Fetched %d / %d dates", fetched, len(missing))

        log.info("Downloaded %d dates for %s", fetched, underlying)
        return self._load_cached(underlying, start_date, end_date)

    def update_ticker(
        self,
        underlying: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Incremental update: download from last cached date to `end`."""
        end_date = date.fromisoformat(end) if end else date.today()
        last_cached = self._find_last_cached_date(underlying, end_date)

        if last_cached is None:
            return self.download(underlying, "2023-01-01", end_date.isoformat())

        start = (last_cached + timedelta(days=1)).isoformat()
        if start > end_date.isoformat():
            log.info("Already up to date: %s (last cached %s)", underlying, last_cached)
            return self._load_cached(underlying, date(2023, 1, 1), end_date)

        df = self.download(underlying, start, end_date.isoformat())
        return df

    def load_chain(
        self,
        underlying: str,
        trade_date: str | date,
    ) -> pd.DataFrame:
        """Load the options chain for a specific underlying and date.

        Returns DataFrame with columns:
            option_symbol, underlying, date, expiration, contract_type,
            strike, bid, ask, last_price, implied_volatility,
            delta, gamma, theta, vega, open_interest, volume
        """
        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)

        df = self._load_cached(underlying, trade_date, trade_date)
        if df.empty:
            log.warning("No cached chain for %s on %s", underlying, trade_date)
        return df

    def get_atm_options(
        self,
        underlying: str,
        trade_date: str | date,
        spot_price: float | None = None,
        dte_range: tuple[int, int] = (20, 60),
    ) -> pd.DataFrame:
        """Get at-the-money options closest to the given spot price.

        Filters by DTE range and selects options closest to ATM.
        Returns both calls and puts near the spot price.
        """
        chain = self.load_chain(underlying, trade_date)
        if chain.empty:
            return chain

        # Filter by expiration DTE
        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)

        chain = chain.copy()
        chain["dte"] = pd.to_datetime(chain["expiration"]).dt.date.apply(
            lambda x: (x - trade_date).days
        )
        chain = chain[(chain["dte"] >= dte_range[0]) & (chain["dte"] <= dte_range[1])]

        if chain.empty:
            return chain

        # If no spot price provided, estimate from ATM options
        if spot_price is None:
            # Use midpoint of bid/ask for ATM detection
            if "bid" in chain.columns and "ask" in chain.columns:
                chain["mid"] = (chain["bid"].fillna(0) + chain["ask"].fillna(0)) / 2
            spot_price = chain["strike"].median()

        # Select options closest to spot price
        chain["dist_from_atm"] = abs(chain["strike"] - spot_price)
        chain = chain.sort_values("dist_from_atm")

        # Return top 5 calls and top 5 puts per expiration
        calls = chain[chain["contract_type"] == "call"].groupby("expiration").head(5)
        puts = chain[chain["contract_type"] == "put"].groupby("expiration").head(5)

        result = pd.concat([calls, puts], ignore_index=True)
        result = result.drop(columns=["dist_from_atm"], errors="ignore")
        return result.sort_values(["expiration", "contract_type", "strike"]).reset_index(drop=True)

    def get_implied_volatility_surface(
        self,
        underlying: str,
        trade_date: str | date,
    ) -> pd.DataFrame:
        """Build an implied volatility surface: strike × expiration.

        Returns DataFrame indexed by (expiration, strike) with columns:
            call_iv, put_iv, call_delta, put_delta, call_volume, put_volume
        """
        chain = self.load_chain(underlying, trade_date)
        if chain.empty:
            return chain

        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date)

        chain = chain.copy()
        chain["dte"] = pd.to_datetime(chain["expiration"]).dt.date.apply(
            lambda x: (x - trade_date).days
        )

        # Pivot into surface format
        calls = chain[chain["contract_type"] == "call"][[
            "expiration", "strike", "implied_volatility", "delta", "volume",
        ]].rename(columns={
            "implied_volatility": "call_iv",
            "delta": "call_delta",
            "volume": "call_volume",
        })

        puts = chain[chain["contract_type"] == "put"][[
            "expiration", "strike", "implied_volatility", "delta", "volume",
        ]].rename(columns={
            "implied_volatility": "put_iv",
            "delta": "put_delta",
            "volume": "put_volume",
        })

        surface = pd.merge(calls, puts, on=["expiration", "strike"], how="outer")
        surface = surface.sort_values(["expiration", "strike"]).reset_index(drop=True)
        return surface

    def download_universe(
        self,
        underlyings: list[str],
        start: str = "2023-01-01",
        end: str = "2024-12-31",
    ) -> dict[str, pd.DataFrame]:
        """Download options chains for multiple underlyings."""
        results: dict[str, pd.DataFrame] = {}
        for i, underlying in enumerate(underlyings):
            log.info("[%d/%d] %s", i + 1, len(underlyings), underlying)
            try:
                df = self.download(underlying, start, end)
                if not df.empty:
                    results[underlying] = df
            except Exception as e:
                log.error("Failed %s: %s", underlying, e)
        log.info("Downloaded %d / %d underlyings", len(results), len(underlyings))
        return results

    def update_universe(
        self,
        underlyings: list[str],
        end: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Incremental update for multiple underlyings."""
        results: dict[str, pd.DataFrame] = {}
        for i, underlying in enumerate(underlyings):
            log.info("[%d/%d] %s", i + 1, len(underlyings), underlying)
            try:
                df = self.update_ticker(underlying, end)
                if not df.empty:
                    results[underlying] = df
            except Exception as e:
                log.error("Failed %s: %s", underlying, e)
        log.info("Updated %d / %d underlyings", len(results), len(underlyings))
        return results


# Backward compatibility alias
AlpacaOptionsLoader = OptionsDataLoader
