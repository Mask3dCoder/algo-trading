"""Rate-limited, paginating Polygon.io REST client for PIT ingestion.

Tier notes
----------
* EOD aggregates and tickers reference are available on the Starter tier.
* Historical options **contracts** are available on Starter/Developer; historical
  options **NBBO quotes** require the Advanced tier (~$199/mo).
* Default rate limit is 5 calls/minute (Starter). Raise it via
  ``rate_limit_per_minute`` when on a higher tier.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

BASE_URL = "https://api.polygon.io"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class PolygonClientError(RuntimeError):
    """Raised when Polygon rejects or the client exhausts retries."""


class PolygonPitClient:
    """Synchronous Polygon client with throttling, retry, and pagination."""

    def __init__(
        self,
        api_key: str,
        rate_limit_per_minute: int = 5,
        retries: int = 4,
    ) -> None:
        if not api_key:
            raise PolygonClientError("POLYGON_API_KEY is required (currently a placeholder in .env)")
        self.api_key = api_key
        self._min_interval = 60.0 / max(rate_limit_per_minute, 1)
        self._retries = max(retries, 1)
        self._last_request = 0.0
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=60.0,
            headers={"Authorization": f"Bearer {api_key}"},
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    # -- transport ------------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str) -> dict[str, Any]:
        for attempt in range(self._retries):
            self._throttle()
            try:
                resp = self._client.get(url)
                if resp.status_code in _RETRYABLE_STATUS and attempt < self._retries - 1:
                    wait = 30.0 if resp.status_code == 429 else float(2 ** attempt)
                    if resp.status_code == 429:
                        try:
                            wait = float(resp.headers.get("Retry-After", 30))
                        except ValueError:
                            pass
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in _RETRYABLE_STATUS and attempt < self._retries - 1:
                    time.sleep(float(2 ** attempt))
                    continue
                raise PolygonClientError(f"Polygon HTTP {exc.response.status_code} for {url}") from exc
            except httpx.TimeoutException as exc:
                if attempt < self._retries - 1:
                    time.sleep(float(2 ** attempt))
                    continue
                raise PolygonClientError(f"Polygon timeout for {url}") from exc
        raise PolygonClientError(f"Polygon request failed for {url}")

    def _paginate(self, url: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        next_url: str | None = url
        while next_url:
            payload = self._get(next_url)
            results.extend(payload.get("results") or [])
            next_url = payload.get("next_url")
        return results

    # -- EOD bars (Starter) ---------------------------------------------------
    def eod_bars(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> list[dict[str, Any]]:
        """Return unadjusted daily aggregates for ``[start, end]`` (YYYY-MM-DD)."""
        url = (
            f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
            "?adjusted=false&sort=asc&limit=50000"
        )
        return self._paginate(url)

    # -- options --------------------------------------------------------------
    def options_contracts(
        self,
        underlying: str,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        url = f"/v3/reference/options/contracts?underlying_ticker={underlying}&limit=1000"
        if as_of:
            url += f"&as_of={as_of}"
        return self._paginate(url)

    def options_quotes(
        self,
        contract: str,
        start: str,
        end: str,
    ) -> list[dict[str, Any]]:
        """Return NBBO quotes for one contract over a date range (Advanced tier)."""
        url = (
            f"/v3/quotes/{contract}"
            f"?timestamp.gte={start}T00:00:00Z&timestamp.lt={end}T23:59:59.999Z&limit=1000"
        )
        return self._paginate(url)

    # -- corporate actions ----------------------------------------------------
    def splits(self, symbol: str) -> list[dict[str, Any]]:
        return self._paginate(f"/v3/reference/splits?ticker={symbol}&limit=1000")

    def dividends(self, symbol: str) -> list[dict[str, Any]]:
        return self._paginate(f"/v3/reference/dividends?ticker={symbol}&limit=1000")

    def ticker(self, symbol: str) -> dict[str, Any]:
        payload = self._get(f"/v3/reference/tickers/{symbol}")
        return payload.get("results") or {}

    # -- universe -------------------------------------------------------------
    def all_tickers(
        self,
        *,
        active: bool = True,
        market: str = "stocks",
        type_: str = "CS",
    ) -> list[dict[str, Any]]:
        """Return all tickers matching the filter (paginated, 1000/page)."""
        url = (
            f"/v3/reference/tickers?market={market}&type={type_}"
            f"&active={str(active).lower()}&sort=ticker&order=asc&limit=1000"
        )
        return self._paginate(url)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PolygonPitClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
