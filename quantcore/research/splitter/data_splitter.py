"""DataSplitter — hard IS/Val/Vault boundary enforcement for hypothesis testing.

This class enforces the Research Governance Layer's strictest rule:
OUT-OF-SAMPLE DATA IS QUARANTINED UNTIL ALL IN-SAMPLE CALIBRATION IS COMPLETE.

No analyst, no script, and no backtester may access Vault data before the
TriageGate promotion check passes. Violations raise ``RuntimeError`` immediately.

Usage
-----
    # Phase 1: In-Sample calibration — only IS data is accessible
    splitter = DataSplitter(current_phase="is")
    is_data = splitter.get(raw_df, "is")   # ✅ returns IS slice
    splitter.get(raw_df, "val")            # ❌ raises RuntimeError

    # Phase 2: Validation — Val data becomes accessible
    splitter = DataSplitter(current_phase="val")
    is_data  = splitter.get(raw_df, "is")  # ✅
    val_data = splitter.get(raw_df, "val") # ✅
    splitter.get(raw_df, "vault")          # ❌ raises RuntimeError

    # Phase 3: Vault — ONLY after TriageGate promotion
    splitter = DataSplitter(current_phase="vault")
    splitter.get(raw_df, "vault")         # ✅ — one shot only
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame, DatetimeIndex


# ── Canonical date boundaries for HYPOTHESIS_001 ──────────────────────────────
IS_START  = "2016-01-01"
IS_END    = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END   = "2023-12-31"
VAULT_START = "2024-01-01"
VAULT_END   = "2026-06-01"

SPLIT_RANGES: dict[str, tuple[str, str]] = {
    "is":    (IS_START,    IS_END),
    "val":   (VAL_START,   VAL_END),
    "vault": (VAULT_START, VAULT_END),
}

VALID_PHASES = list(SPLIT_RANGES)


@dataclass
class SplitMask:
    """Reusable boolean mask for a given date range."""
    start: str
    end: str

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows of *df* whose index falls within [start, end]."""
        idx = df.index
        # Normalise index to datetime for comparison
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        return df[(idx >= self.start) & (idx <= self.end)]

    def contains(self, dt: date | str) -> bool:
        """Return True if *dt* falls within the split range."""
        d = pd.Timestamp(dt)
        return (d >= self.start) and (d <= self.end)


# ── Cached masks (created once per-phase) ───────────────────────────────────────
_MASK_CACHE: dict[str, SplitMask] = {
    phase: SplitMask(*bounds)
    for phase, bounds in SPLIT_RANGES.items()
}


class DataSplitter:
    """Enforce IS/Val/Vault isolation with hard runtime raises.

    Parameters
    ----------
    current_phase : {"is", "val", "vault"}
        The current research phase. Data for any phase strictly after this
        one raises ``RuntimeError`` on access.  Must be advanced manually
        to prevent accidental vault exposure during IS/Val calibration.

    Attributes
    ----------
    phase : str
        The active research phase.

    Example
    -------
    >>> splitter = DataSplitter(current_phase="is")
    >>> spy_is = splitter.get(raw_spy, "is")
    >>> splitter.get(raw_spy, "val")   # raises RuntimeError
    Traceback (most recent call):
        RuntimeError: VAULT BREACH ATTEMPT: vault data is quarantined until ...
    """

    VALID_PHASES = VALID_PHASES
    SPLIT_RANGES = SPLIT_RANGES

    def __init__(self, current_phase: Literal["is", "val", "vault"] = "is") -> None:
        if current_phase not in VALID_PHASES:
            raise ValueError(
                f"Unknown phase {current_phase!r}. Must be one of {VALID_PHASES}."
            )
        self._phase = current_phase
        self._phase_order = VALID_PHASES  # chronological order
        self._current_idx = self._phase_order.index(current_phase)

    @property
    def phase(self) -> str:
        """Current research phase."""
        return self._phase

    def advance(self) -> "DataSplitter":
        """Return a new DataSplitter advanced one phase.

        Use this when IS calibration is complete and you are ready to work
        on Validation — not when you need to peek at Val data during IS.

        Raises
        ------
        RuntimeError
            When called in "vault" phase (the vault is a one-shot — there is
            no phase after it).
        """
        if self._phase == "vault":
            raise RuntimeError(
                "Cannot advance from 'vault' phase. The vault is a one-shot "
                "evaluation with no subsequent phase."
            )
        next_idx = self._current_idx + 1
        next_phase = self._phase_order[next_idx]
        return DataSplitter(current_phase=next_phase)

    def get(self, df: pd.DataFrame, for_phase: str) -> pd.DataFrame:
        """Return the slice of *df* accessible in *for_phase*.

        Parameters
        ----------
        df : pd.DataFrame
            Full dataset (all dates). Must have a datetime index.
        for_phase : {"is", "val", "vault"}
            Phase whose data should be returned.

        Returns
        -------
        pd.DataFrame
            Only the rows whose index falls within *for_phase* boundaries.

        Raises
        ------
        RuntimeError
            If *for_phase* is chronologically after the current phase
            (i.e., data is not yet accessible).
        ValueError
            If *for_phase* is not a known phase.
        """
        if for_phase not in VALID_PHASES:
            raise ValueError(
                f"Unknown phase {for_phase!r}. Must be one of {VALID_PHASES}."
            )

        req_idx = self._phase_order.index(for_phase)

        if req_idx > self._current_idx:
            # Strictly future phase — hard quarantine breach
            raise RuntimeError(
                f"VAULT BREACH ATTEMPT: {for_phase.upper()} data is quarantined "
                f"until all prior phase gates pass.\n"
                f"  Current phase : {self._phase.upper()}\n"
                f"  Attempted     : {for_phase.upper()}\n"
                f"  Fix           : Complete {self._phase.upper()} work before "
                f"advancing to {for_phase.upper()}."
            )

        # Past or current phase — return the slice
        return _MASK_CACHE[for_phase].apply(df)

    def is_accessible(self, for_phase: str) -> bool:
        """Return True if *for_phase* data is accessible right now."""
        if for_phase not in VALID_PHASES:
            return False
        return self._phase_order.index(for_phase) <= self._current_idx

    def report(self) -> dict[str, str | list[str]]:
        """Human-readable phase status report (useful for CI logs)."""
        lines = []
        for phase in self._phase_order:
            marker = ">>>" if phase == self._phase else "   "
            accessible = self.is_accessible(phase)
            start, end = SPLIT_RANGES[phase]
            status = "accessible" if accessible else "LOCKED"
            lines.append(f"{marker} {phase.upper():6s}  [{start} → {end}]  {status}")
        return {
            "current_phase": self._phase,
            "phases": lines,
        }