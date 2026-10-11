"""Alpaca market-data REST client for the PIT store.

Scope
-----
Alpaca is the primary source for the decisive point-in-time panel (see the
2026-10-11 amendment in ``docs/cross_sectional_factor_protocol.md``). It is
free at the IEX tier, which is the tier this project can afford; the trade-off
is that IEX is one exchange, so volume is a fraction of consolidated volume.

What Alpaca does NOT provide, and what this module therefore does not pretend
to provide:

* **No corporate-actions endpoint.** Splits and dividends arrive only as a
  pre-adjusted price series (``adjustment=all``). The ingestor derives an
  adjustment factor from the raw/adjusted pair and labels it ``derived``; it is
  not an authoritative dividend ledger.
* **No index-membership history.** PIT membership is reconstructed from each
  name's first and last available bar, which is permissive (it can admit a
  name that traded but was not an index member) rather than survivorship-
  optimistic. See the amendment log for the bias direction.

Delisted securities are a supported case, not an error: their history is
retained and is fetched like any other symbol, because retaining the dead tail
is the entire point of a survivorship-free panel. A terminal zero-volume bar at
the delisting date is passed through unchanged - do not filter it, it is the
observation that proves the name died.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

DATA_BASE_URL = "https://data.alpaca.markets"
TRADING_BASE_URL = "https://api.alpaca.markets"

#: Substrings that mean the URL was never filled in. The repo's .env carries a
#: placeholder ALPACA_BASE_URL pointing at the *paper trading* host, which is a
#: different service from the market-data API - reusing it would silently query
#: the wrong API and 404 on every bars call.
_PLACEHOLDER_MARKERS = ("placeholder", "your-", "your_", "changeme", "example.com")

#: Alpaca's free tier allows 200 requests/minute. 0.42s spacing keeps a
#: single-threaded sweep at ~143 req/min with headroom for retries.
DEFAULT_MIN_INTERVAL = 0.42

#: Alpaca caps a bars page at 10,000 rows; a 7-year daily series is ~1,760,
#: so one page per symbol-window is the normal case.
MAX_PAGE_LIMIT = 10_000

#: Fields the store's normaliser expects from Polygon-style rows.
_POLYGON_FIELD_MAP = {
    "t": "t",
    "o": "o",
    "h": "h",
    "l": "l",
    "c": "c",
    "v": "v",
    "vw": "vw",
    "n": "n",
}


class AlpacaClientError(RuntimeError):
    """Raised when Alpaca credentials or requests are unusable."""


def _to_epoch_ms(ts: str) -> int:
    """Convert an Alpaca RFC3339 timestamp to epoch milliseconds."""
    from datetime import datetime

    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


class AlpacaDataClient:
    """Minimal, rate-limited Alpaca REST client.

    Synchronous on purpose: the ingestion sweep is I/O-bound and already
    paced by ``min_interval``; a thread pool would only multiply the chance of
    tripping the rate limiter.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = DATA_BASE_URL,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        if not api_key or not api_secret:
            raise AlpacaClientError("ALPACA_API_KEY / ALPACA_API_SECRET are required.")
        if base_url.rstrip("/").lower() in TRADING_BASE_URL or "paper-api" in base_url:
            raise AlpacaClientError(
                f"{base_url!r} is the Alpaca TRADING host, not the market-data host. "
                f"Market data lives at {DATA_BASE_URL}."
            )
        if any(m in base_url.lower() for m in _PLACEHOLDER_MARKERS):
            raise AlpacaClientError(
                f"Alpaca base URL still contains placeholder text ({base_url!r}); "
                f"expected {DATA_BASE_URL}."
            )
        self._key = api_key
        self._secret = api_secret
        self._base = base_url.rstrip("/")
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._timeout = timeout
        self._last_request = 0.0
        self._client = httpx.Client(timeout=timeout)

    # -- plumbing -------------------------------------------------------------
    def _throttle(self) -> None:
        wait = self._min_interval - (time.perf_counter() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._base}{path}"
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            self._throttle()
            self._last_request = time.perf_counter()
            try:
                resp = self._client.get(url, params=params, headers=self._headers())
            except httpx.HTTPError as exc:  # network hiccup / timeout
                last_error = exc
                time.sleep(2**attempt)
                continue

            if resp.status_code == 200:
                return resp.json()
            # Alpaca signals throttling with 429; 5xx is transient. Both are
            # worth another try. A 404/422 is a real answer (no such symbol).
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = AlpacaClientError(f"HTTP {resp.status_code} from Alpaca for {path}")
                time.sleep(2**attempt)
                continue
            raise AlpacaClientError(f"HTTP {resp.status_code} from Alpaca for {path}: {resp.text[:200]}")

        raise AlpacaClientError(f"Alpaca request failed after {self._max_retries} attempts: {last_error}")

    # -- public API -----------------------------------------------------------
    def daily_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        adjustment: str = "raw",
        limit: int = MAX_PAGE_LIMIT,
    ) -> list[dict[str, Any]]:
        """Return daily bars for ``symbol`` as Polygon-shaped rows.

        Alpaca keys the payload by symbol; both that and the bare-list shape
        are handled. Timestamps are converted to epoch milliseconds so the
        store's existing normaliser can be reused unchanged.

        A ``bars`` key of ``None`` means "no data for this window" (which is
        different from an empty list and must not be written as zero bars);
        it is reported as an empty result and logged by the caller.
        """
        params = {
            "start": start,
            "end": end,
            "timeframe": "1Day",
            "limit": limit,
            "adjustment": adjustment,
            "sort": "asc",
        }
        payload = self._get(f"/v2/stocks/{symbol}/bars", params)
        raw = payload.get("bars")
        if raw is None:
            return []
        rows = raw if isinstance(raw, list) else next((v for v in raw.values() if isinstance(v, list)), [])

        out: list[dict[str, Any]] = []
        for row in rows:
            mapped: dict[str, Any] = {}
            for src, dst in _POLYGON_FIELD_MAP.items():
                if src in row:
                    mapped[dst] = row[src]
            if "t" not in mapped:
                continue
            mapped["t"] = _to_epoch_ms(mapped["t"])
            out.append(mapped)
        return out

    def asset(self, symbol: str) -> dict[str, Any] | None:
        """Return Alpaca's asset reference for ``symbol``, or None if unknown.

        Uses the trading API host, which is a different service from the data
        API used for bars.
        """
        try:
            client = httpx.Client(timeout=self._timeout)
            resp = client.get(
                f"{TRADING_BASE_URL}/v2/assets/{symbol}",
                headers=self._headers(),
            )
            if resp.status_code != 200:
                return None
            return resp.json()
        except httpx.HTTPError:
            return None
        finally:
            try:
                client.close()
            except Exception:
                pass

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AlpacaDataClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
