"""Massive (formerly Polygon.io) host configuration.

Polygon.io rebranded to **Massive** (massive.com) effective 2025-10-30. The
REST, WebSocket, and flat-file hosts moved from ``*.polygon.io`` to
``*.massive.com``. The legacy hosts are still served for an extended period
(existing keys and integrations keep working), but new code should target the
massive.com hosts.

All hosts are overridable via env vars so a future host phase-out is a
configuration change, not a code change:

* ``MASSIVE_REST_BASE_URL``        (default ``https://api.massive.com``)
* ``MASSIVE_WS_BASE_URL``          (default ``wss://socket.massive.com``)
* ``MASSIVE_FLAT_FILE_ENDPOINT``   (default ``https://files.massive.com``)

REST authentication is ``Authorization: Bearer <API_KEY>`` — the same key
works on both hosts. Flat-file access uses **separate S3 credentials** (access
key id + secret), see ``PolygonFlatFileClient`` in
``quantcore/data/pit/market/client.py``.
"""

from __future__ import annotations

import os


def _host(name: str, default: str) -> str:
    """Return the env override for ``name`` or ``default``."""
    return os.environ.get(name, "").strip() or default


# REST API base (v2/v3 endpoints). Legacy: https://api.polygon.io
REST_BASE_URL = _host("MASSIVE_REST_BASE_URL", "https://api.massive.com")

# WebSocket streaming. Legacy: wss://socket.polygon.io
WS_BASE_URL = _host("MASSIVE_WS_BASE_URL", "wss://socket.massive.com")

# S3-compatible flat-file endpoint (bulk export). Legacy: https://files.polygon.io
FLAT_FILE_ENDPOINT = _host("MASSIVE_FLAT_FILE_ENDPOINT", "https://files.massive.com")

# Flat files live in the "flatfiles" bucket under this key prefix:
#   us_stocks_sip/day_aggs_v1/{year}/{month}[/{day}/{ticker}.csv.gz]
FLAT_FILE_BUCKET = "flatfiles"
DAY_AGG_PREFIX = "us_stocks_sip/day_aggs_v1"
