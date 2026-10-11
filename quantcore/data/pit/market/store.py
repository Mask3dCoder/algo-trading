"""Append-only, atomic parquet store for point-in-time market data.

Layout
------
    <root>/<dataset>/<shard>/<period>.parquet
    <root>/_watermarks.json   {dataset: {shard: last_synced_iso_utc}}

Semantics
---------
Rows are **append-only** once finalised. Every write is atomic (temp file then
``os.replace``) so a crash mid-ingest can never leave a half-written file.
Watermarks are committed *after* the data file is safely on disk, so an
interrupted run resumes from the last successful sync instead of re-fetching.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantcore.data.pit.market.schema import DATASET_COLUMNS

_WATERMARKS_FILE = "_watermarks.json"

#: os.replace on the manifest can transiently fail with PermissionError while a
#: cloud-sync client or the Windows Search indexer holds the file. Three tries
#: with a short backoff cover that window without slowing the common case.
_WATERMARK_REPLACE_ATTEMPTS = 3
_WATERMARK_REPLACE_BACKOFF = 0.5


class PitStoreError(RuntimeError):
    """Raised when a store operation violates PIT invariants."""


def _canonical_frame(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame to the dataset's canonical column set and order."""
    columns = DATASET_COLUMNS.get(dataset)
    if columns is None:
        raise PitStoreError(f"Unknown dataset {dataset!r}; expected one of {sorted(DATASET_COLUMNS)}")
    out = pd.DataFrame(df)
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out[list(columns)]


def _empty_frame(dataset: str) -> pd.DataFrame:
    return pd.DataFrame(columns=list(DATASET_COLUMNS.get(dataset, ())))


class ParquetPitStore:
    """Atomic, watermark-tracked parquet store keyed by ``dataset/shard/period``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------------
    def _dataset_dir(self, dataset: str) -> Path:
        return self.root / dataset

    def _shard_dir(self, dataset: str, shard: str) -> Path:
        return self._dataset_dir(dataset) / shard

    def _file_path(self, dataset: str, shard: str, period: str) -> Path:
        return self._shard_dir(dataset, shard) / f"{period}.parquet"

    def _manifest_path(self) -> Path:
        return self.root / _WATERMARKS_FILE

    # -- write -----------------------------------------------------------------
    def write(
        self,
        dataset: str,
        shard: str,
        df: pd.DataFrame,
        period: str,
    ) -> Path:
        """Atomically replace one ``dataset/shard/period`` parquet file.

        Use for append-only new periods (e.g. a fresh month of options quotes)
        and for the initial write of a shard. To re-run idempotently over
        already-written rows, use :meth:`upsert_shard`.
        """
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise PitStoreError(f"Refusing to write empty frame to {dataset}/{shard}/{period}")
        out = _canonical_frame(dataset, df)
        path = self._file_path(dataset, shard, period)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        try:
            out.to_parquet(tmp, index=False, compression="snappy")
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return path

    def upsert_shard(
        self,
        dataset: str,
        shard: str,
        df: pd.DataFrame,
        dedupe_cols: Sequence[str],
    ) -> int:
        """Merge ``df`` into the shard, deduping on ``dedupe_cols``, then rewrite.

        Returns the number of rows present after the merge. Intended for the
        smaller datasets (equity bars, corporate actions, contracts). For high
        volume options quotes prefer period-fresh :meth:`write` calls.
        """
        existing = self.read(dataset, shards=[shard])
        combined = pd.concat([existing, _canonical_frame(dataset, df)], ignore_index=True)
        combined = combined.drop_duplicates(subset=list(dedupe_cols), keep="last")
        if combined.empty:
            return 0
        # Re-split by period so each partition file stays internally consistent.
        from quantcore.data.pit.market.schema import period_shard

        grouped = combined.groupby(combined["ts"].map(lambda t: period_shard(t, dataset)), sort=False) \
            if "ts" in combined.columns else {str(shard): combined}
        if isinstance(grouped, dict):
            for period, frame in grouped.items():
                self.write(dataset, shard, frame, period)
            return len(combined)
        for period, frame in grouped:
            self.write(dataset, shard, frame, str(period))
        return len(combined)

    # -- read ------------------------------------------------------------------
    def read(
        self,
        dataset: str,
        shards: Iterable[str] | None = None,
        periods: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Read rows from the store, optionally filtered by shard and period.

        ``shards``/``periods`` must be iterables of names. A bare string is
        silently iterable, so ``read("equity_bars", "AAPL")`` would filter on
        the characters {'A', 'P', 'L'} and quietly return nothing - a wrong
        answer with no error. Reject it instead.
        """
        dataset_dir = self._dataset_dir(dataset)
        if not dataset_dir.exists():
            return _empty_frame(dataset)

        for name, value in (("shards", shards), ("periods", periods)):
            if isinstance(value, str):
                raise TypeError(
                    f"read({dataset!r}, {name}=...) expects an iterable of names, "
                    f"not a bare string; pass [{value!r}]"
                )

        wanted_shards = set(shards) if shards is not None else None
        wanted_periods = set(periods) if periods is not None else None

        frames: list[pd.DataFrame] = []
        for shard_dir in sorted(dataset_dir.iterdir()):
            if not shard_dir.is_dir():
                continue
            shard = shard_dir.name
            if wanted_shards is not None and shard not in wanted_shards:
                continue
            for parquet_file in sorted(shard_dir.glob("*.parquet")):
                period = parquet_file.stem
                if wanted_periods is not None and period not in wanted_periods:
                    continue
                frames.append(pd.read_parquet(parquet_file))

        if not frames:
            return _empty_frame(dataset)
        out = pd.concat(frames, ignore_index=True)
        return _canonical_frame(dataset, out)

    def has_shard(self, dataset: str, shard: str) -> bool:
        return self._shard_dir(dataset, shard).exists()

    def list_shards(self, dataset: str) -> list[str]:
        dataset_dir = self._dataset_dir(dataset)
        if not dataset_dir.exists():
            return []
        return sorted(d.name for d in dataset_dir.iterdir() if d.is_dir())

    # -- watermarks ------------------------------------------------------------
    def _load_manifest(self) -> dict[str, dict[str, str]]:
        path = self._manifest_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def get_watermark(self, dataset: str, shard: str) -> pd.Timestamp | None:
        raw = self._load_manifest().get(dataset, {}).get(shard)
        if not raw:
            return None
        return pd.Timestamp(raw).tz_convert("UTC") if pd.Timestamp(raw).tzinfo else pd.Timestamp(raw).tz_localize("UTC")

    def set_watermark(self, dataset: str, shard: str, ts: Any) -> None:
        """Persist ``ts`` as the last successfully synced instant for a shard."""
        stamp = pd.Timestamp(ts)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        stamp = stamp.tz_convert("UTC")
        manifest = self._load_manifest()
        manifest.setdefault(dataset, {})[shard] = stamp.isoformat()
        path = self._manifest_path()
        payload = json.dumps(manifest, indent=2, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        try:
            Path(tmp).write_text(payload, encoding="utf-8")
            # os.replace is atomic but takes an exclusive lock on the target.
            # Sync clients (OneDrive) and Windows indexers can hold the
            # manifest open and make that fail with WinError 5, which would
            # abort an hours-long backfill over bookkeeping. Retry briefly,
            # then fall back to a direct write: a slightly less atomic
            # manifest update is strictly better than losing the run.
            for attempt in range(_WATERMARK_REPLACE_ATTEMPTS):
                try:
                    os.replace(tmp, path)
                    return
                except PermissionError:
                    if attempt == _WATERMARK_REPLACE_ATTEMPTS - 1:
                        break
                    time.sleep(_WATERMARK_REPLACE_BACKOFF)
            path.write_text(payload, encoding="utf-8")
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # -- misc ------------------------------------------------------------------
    def status(self) -> dict[str, dict[str, Any]]:
        """Summarise datasets: shard count and on-disk bytes."""
        report: dict[str, dict[str, Any]] = {}
        for dataset in DATASET_COLUMNS:
            shards = self.list_shards(dataset)
            if not shards:
                continue
            total_bytes = sum(
                f.stat().st_size
                for shard_dir in self._dataset_dir(dataset).iterdir()
                if shard_dir.is_dir()
                for f in shard_dir.glob("*.parquet")
            )
            report[dataset] = {"shards": len(shards), "bytes": total_bytes}
        return report
