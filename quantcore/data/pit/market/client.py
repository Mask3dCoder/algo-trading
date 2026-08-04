"""Rate-limited, paginating Massive (Polygon.io) REST client for PIT ingestion.

Hosts
-----
Polygon.io rebranded to Massive (massive.com). The REST base URL is now
``https://api.massive.com`` (legacy ``https://api.polygon.io`` still works for an
extended period). Flat-file bulk export is served from an S3-compatible endpoint
(``https://files.massive.com``) under ``flatfiles/us_stocks_sip/day_aggs_v1/``.
All hosts are overridable via env vars (see ``quantcore/data/polygon_hosts.py``).

Tier notes
----------
* EOD aggregates and tickers reference are available on the Starter tier.
* Historical options **contracts** are available on Starter/Developer; historical
  options **NBBO quotes** require the Advanced tier (~$199/mo).
* Default rate limit is 5 calls/minute (Starter). Raise it via
  ``rate_limit_per_minute`` when on a higher tier.
* Day-aggregate flat files are included in Starter/Developer/Advanced/Business,
  NOT in Basic (see massive.com/docs/flat-files/stocks/day-aggregates).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from quantcore.data.polygon_hosts import (
    DAY_AGG_PREFIX,
    FLAT_FILE_BUCKET,
    FLAT_FILE_ENDPOINT,
    REST_BASE_URL,
)

BASE_URL = REST_BASE_URL

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def resolve_api_key(explicit: str = "") -> str:
    """Return the first non-empty API key from the arg / env.

    Massive accepts the existing ``POLYGON_API_KEY`` and recommends
    ``MASSIVE_API_KEY`` (same keys, both hosts). Raises if neither is set.
    """
    key = (explicit or os.environ.get("MASSIVE_API_KEY", "") or os.environ.get("POLYGON_API_KEY", "")).strip()
    if not key:
        raise PolygonClientError("MASSIVE_API_KEY / POLYGON_API_KEY is required")
    return key


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
        self.api_key = resolve_api_key(api_key)
        self._min_interval = 60.0 / max(rate_limit_per_minute, 1)
        self._retries = max(retries, 1)
        self._last_request = 0.0
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=60.0,
            headers={"Authorization": f"Bearer {self.api_key}"},
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


class PolygonFlatFileClient:
    """Minimal S3 client for Massive/Polygon flat-file bulk export.

    Day aggregates are served from the S3-compatible ``FLAT_FILE_ENDPOINT`` under
    the ``flatfiles`` bucket at ``us_stocks_sip/day_aggs_v1/{year}/{month}/``
    (``*.csv.gz`` per ticker/day). This is the bulk path for the 500-ticker
    backfill — orders of magnitude cheaper than per-symbol REST calls.

    **Authentication uses S3 credentials (access key id + secret), NOT the REST
    API key.** Get them from the Massive dashboard alongside the REST key. Env
    vars accepted (new name first, legacy second):

        MASSIVE_S3_ACCESS_ID        / MASSIVE_S3_SECRET_ACCESS_KEY
        POLYGON_S3_Access_ID        / POLYGON_Secret_Access_Key

    Requires ``boto3``/``botocore`` (present in the project deps).
    """

    def __init__(
        self,
        access_key_id: str = "",
        secret_access_key: str = "",
        endpoint: str = "",
    ) -> None:
        self.access_key_id = access_key_id or os.environ.get("MASSIVE_S3_ACCESS_ID", "") or os.environ.get("POLYGON_S3_Access_ID", "")
        self.secret_access_key = secret_access_key or os.environ.get("MASSIVE_S3_SECRET_ACCESS_KEY", "") or os.environ.get("POLYGON_Secret_Access_Key", "")
        self.endpoint = (endpoint or FLAT_FILE_ENDPOINT).rstrip("/")
        if not (self.access_key_id and self.secret_access_key):
            raise PolygonClientError(
                "Flat-file S3 credentials missing. Set MASSIVE_S3_ACCESS_ID and "
                "MASSIVE_S3_SECRET_ACCESS_KEY (or legacy POLYGON_S3_Access_ID / "
                "POLYGON_Secret_Access_Key)."
            )
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - env dependent
            raise PolygonClientError("boto3/botocore are required for flat-file download") from exc
        self._s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="us-east-1",
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        )

    def day_agg_keys(self, year: int, month: int | None = None) -> list[str]:
        """List flat-file object keys under ``us_stocks_sip/day_aggs_v1``."""
        prefix = f"{DAY_AGG_PREFIX}/{year}"
        if month is not None:
            prefix += f"/{int(month):02d}"
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=FLAT_FILE_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return sorted(keys)

    def download(self, key: str, dest_dir: str | Path) -> Path:
        """Download one flat-file object to ``dest_dir`` (basename preserved)."""
        dest = Path(dest_dir) / Path(key).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(FLAT_FILE_BUCKET, key, str(dest))
        return dest
