"""Ken French Fama-French data loader.

Downloads and caches daily factor returns from the Ken French Data Library:
<link to Ken French Data Library>

Supported datasets:
  - F-F_5_Factor_2x3: Fama-French 5 factors (MKT-RF, SMB, HML, RMW, CMA)
  - Fama-French 3 Factor (legacy): MKT-RF, SMB, HML

Usage::
    loader = FamaFrenchLoader(cache_dir=".cache/fama_french")
    df = loader.load_5_factors(freq="daily")  # or "monthly"
    factor_returns = df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].to_numpy()

The loader persists a local CSV cache to avoid repeated network calls.
"""

from __future__ import annotations

import io
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np

_log = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

_START_1926: date = date(1926, 7, 1)

# URLs for the daily and monthly CSV archives (zip).
# These are the canonical Ken French Data Library links as of 2025-12.
_FAMAFRENCH_URLS = {
    "5_factors_2x3": {
        "daily": (
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        ),
        "monthly": (
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_5_Factors_2x3_CSV.zip"
        ),
    },
    "momentum": {
        "daily": (
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Momentum_Factor_daily_CSV.zip"
        ),
        "monthly": (
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Momentum_Factor_CSV.zip"
        ),
    },
}


class FamaFrenchLoader:
    """Download, cache, and serve Ken French factor data.

    Args:
        cache_dir: Directory for cached CSV files.  If None, uses a transient
            temp directory (no persistence).
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self._is_temp = False
        if cache_dir is None:
            import tempfile as _tf
            self._cache_dir = Path(_tf.mkdtemp(prefix="ff_"))  # nosec: B306 (random is fine for temp dir)
            self._is_temp = True
        else:
            self._cache_dir = Path(cache_dir).expanduser()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._is_temp = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_5_factors(
        self,
        freq: Literal["daily", "monthly"] = "daily",
    ) -> pd.DataFrame:
        """Load the 5-factor (2x3) dataset as a DataFrame.

        Columns returned: [Mkt-RF, SMB, HML, RMW, CMA, RF]
        All values are **percentage returns** (e.g. 0.05 = 5%).
        """
        if pd is None:
            raise ImportError("pandas is required for FamaFrenchLoader")  # type: ignore[unreachable]
        return self._load("5_factors_2x3", freq)

    def load_3_factors(
        self,
        freq: Literal["daily", "monthly"] = "daily",
    ) -> pd.DataFrame:
        """Load the classic 3-factor dataset as a DataFrame.

        Columns returned: [Mkt-RF, SMB, HML, RF]
        All values are **percentage returns** (e.g. 0.05 = 5%).
        """
        if pd is None:
            raise ImportError("pandas is required for FamaFrenchLoader")  # type: ignore[unreachable]
        return self._load("3_factors", freq)

    def load_momentum(
        self,
        freq: Literal["daily", "monthly"] = "daily",
    ) -> pd.DataFrame:
        """Load the momentum factor dataset as a DataFrame.

        Columns returned: [Mom]
        All values are **fraction returns**.
        """
        if pd is None:
            raise ImportError("pandas is required for FamaFrenchLoader")  # type: ignore[unreachable]
        return self._load("momentum", freq)

    def get_factor_returns(
        self,
        start: str | None = None,
        end: str | None = None,
        freq: Literal["daily", "monthly"] = "daily",
    ) -> np.ndarray:
        """Return a NumPy array of 5-factor returns suitable for alpha models.

        Returns an (N, 5) array with columns [Mkt-RF, SMB, HML, RMW, CMA].
        The risk-free rate (RF) is **excluded** so the data is ready for
        cross-asset regression.
        """
        df = self.load_5_factors(freq=freq)
        df = df.sort_index()
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        return df[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, dataset: str, freq: str) -> pd.DataFrame:
        cache_file = self._cache_dir / f"{dataset}_{freq}.csv"
        if cache_file.exists():
            _log.debug("cache_hit: %s", cache_file)
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        _log.info("cache_miss: downloading %s %s...", dataset, freq)
        df = self._download(dataset, freq)
        df.to_csv(cache_file)
        _log.info("cached: %s", cache_file)
        return df

    def _download(self, dataset: str, freq: str) -> pd.DataFrame:
        if requests is None:
            raise ImportError("requests is required to download Fama-French data")
        url = _FAMAFRENCH_URLS[dataset][freq]
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            # Fallback to direct CSV (sometimes the zip link changes)
            _log.warning("download_failed %s: %s", r.status_code, url)
            raise ConnectionError(f"Failed to download {url}: {r.status_code}")
        return self._parse_zip(r.content, dataset, freq)

    def _parse_zip(self, content: bytes, dataset: str, freq: str) -> pd.DataFrame:
        import zipfile

        z = zipfile.ZipFile(io.BytesIO(content))
        names = z.namelist()
        _log.debug("zip_members: %s", names)

        # Find the CSV inside the zip (there is usually exactly one)
        csv_name = next((n for n in names if n.lower().endswith(".csv")), None)
        if csv_name is None:
            raise ValueError(f"No CSV found inside downloaded zip: {names}")

        with z.open(csv_name) as f:
            df = self._parse_csv(f, dataset, freq)
            return df

    def _parse_csv(self, f, dataset: str, freq: str) -> pd.DataFrame:
        if pd is None:
            raise ImportError("pandas is required")  # type: ignore[unreachable]

        raw = f.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = raw

        lines = text.splitlines()

        # Ken French CSVs have comment lines followed by a header row that
        # starts with a comma (empty first column -> date index).  Find it.
        header_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Header line starts with comma and contains a known column name
            if stripped.startswith(",") and any(
                name in stripped for name in ("Mkt-RF", "Mom", "SMB", "HML")
            ):
                header_idx = i

        # Build CSV from the header line onwards
        from io import StringIO
        # Strip trailing junk (copyright line, empty lines, etc.)
        footer_idx = len(lines)
        for j in range(header_idx + 1, len(lines)):
            if "Copyright" in lines[j] or "Fama and" in lines[j] or "Ken French" in lines[j]:
                footer_idx = j
                break

        csv_text = "\n".join(lines[header_idx:footer_idx])

        df = pd.read_csv(
            StringIO(csv_text),
            index_col=0,
            parse_dates=False,
        )
        df.index.name = "Date"

        # Convert index to datetime. Daily: YYYYMMDD, Monthly: YYYYMM
        if freq == "daily":
            df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
        else:
            # Monthly: use first day of month
            df.index = pd.to_datetime(df.index, format="%Y%m", errors="coerce")

        # Rename columns to canonical names (Ken French files already use canon)
        col_map = {
            "Mkt-RF": "Mkt-RF", "SMB": "SMB", "HML": "HML",
            "RMW": "RMW", "CMA": "CMA", "RF": "RF", "Mom": "Mom",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        # Drop any empty columns that may appear
        df = df.loc[:, df.columns.notna()]
        df = df.loc[:, df.columns != ""]

        # Ken French returns are in **percent** (e.g. 0.05). We store as fraction.
        df = df / 100.0
        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        # Only cleanup if we created a temp directory automatically
        if getattr(self, "_is_temp", False):
            try:
                shutil.rmtree(self._cache_dir, ignore_errors=True)  # type: ignore[arg-type]
            except Exception:
                pass
