"""TickContext — session-aware context for intraday bar processing.

Tracks pre-market volume, first-5-min VWAP, and overnight gap % to support
session-aware strategy logic (entry on first tick of RTH, hard exit by 10:00 ET).

This is the PRIMARY context object for opening_gap_fade and any other
intraday strategy that needs to know the "session origin" of a bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantcore.core.types import OHLCV


# ── Trading session boundaries ────────────────────────────────────────────────

# US equity regular trading hours
RTH_OPEN = time(9, 30)     # 9:30 AM ET — first RTH bar is after open auction
RTH_CLOSE = time(16, 0)   # 4:00 PM ET — last RTH bar

# Pre-market: 4:00 AM – 9:30 AM ET
PRE_MARKET_OPEN = time(4, 0)

# Hard exit time for intraday strategies
# Any position open at EXIT_TIME must be closed
EXIT_TIME = time(10, 0)    # 10:00 AM ET — no overnight gap risk

# First-5-minute window after open
FIRST_5_MIN_OPEN = time(9, 30)
FIRST_5_MIN_CLOSE = time(9, 35)  # 5 minutes


def is_rth_bar(ts: datetime) -> bool:
    """Return True if datetime is within regular trading hours (US/Eastern)."""
    t = ts.time()
    return RTH_OPEN <= t < RTH_CLOSE


def is_premarket_bar(ts: datetime) -> bool:
    """Return True if datetime is in pre-market hours."""
    t = ts.time()
    return PRE_MARKET_OPEN <= t < RTH_OPEN


def is_first_5_min_bar(ts: datetime) -> bool:
    """Return True if datetime is in the first 5 minutes of RTH."""
    t = ts.time()
    return FIRST_5_MIN_OPEN <= t <= FIRST_5_MIN_CLOSE


def is_exit_time(ts: datetime) -> bool:
    """Return True if datetime is at or after the hard exit time."""
    return ts.time() >= EXIT_TIME


# ── Aggregated bar types ───────────────────────────────────────────────────────

@dataclass
class FiveMinBar:
    """A 5-minute OHLCV bar with aggregated session context."""
    bar: OHLCV
    session_open: float          # RTH session open price (from 9:30 baseline)
    prev_close: float            # Previous day close
    overnight_gap_pct: float     # (session_open - prev_close) / prev_close * 100
    pre_market_volume: int       # Aggregated pre-market volume
    first_5_min_vwap: float      # VWAP over first 5 minutes of RTH
    first_5_min_volume: int      # Volume over first 5 minutes of RTH
    rth_volume_running: int      # Cumulative RTH volume
    vix_level: float = np.nan    # VIX level at bar time (forward-filled)
    vix_1d_change: float = np.nan # VIX 1-day % change


@dataclass
class TickContext:
    """Session-level context accumulated across all 5-min bars for a trading day.

    TickContext is built up bar-by-bar as 5-minute bars arrive and provides
    the strategy with the "session origin" information needed for gap-fade
    and other intraday signals.

    Attributes:
        date: The trading date (datetime.date)
        prev_close: Previous trading day's close (index reference level)
        session_open: The RTH open price (first 5-min bar's open after auction)
        overnight_gap_pct: (session_open - prev_close) / prev_close * 100
        pre_market_volume: Total pre-market volume
        first_5_min_vwap: Volume-weighted average price in first 5 minutes
        first_5_min_volume: Volume in first 5 minutes
        first_5_min_high: High in first 5 minutes
        first_5_min_low: Low in first 5 minutes
        rth_cumulative_volume: Cumulative RTH volume at current bar
        rth_cumulative_trades: Cumulative number of trades in RTH
        vix: Current VIX level (forward-filled from prior day)
        vix_1d_pct_change: VIX 1-day % change
        vix_spike_detected: True if VIX > 30 or VIX 1d change > 20%
        bars_processed: Number of 5-min bars received so far today
        is_first_bar: True if this is the first RTH bar of the day
        is_last_bar: True if this is the last bar before exit time
        vol_regime: "low" (<15 VIX), "normal" (15-25), "elevated" (25-30), "high" (>30)
    """
    date: datetime
    prev_close: float = np.nan
    session_open: float = np.nan
    overnight_gap_pct: float = np.nan
    pre_market_volume: int = 0
    first_5_min_vwap: float = np.nan
    first_5_min_volume: int = 0
    first_5_min_high: float = -np.inf
    first_5_min_low: float = np.inf
    first_5_min_done: bool = False   # True once at least 1 bar AFTER 9:35
    rth_cumulative_volume: int = 0
    rth_cumulative_trades: int = 0
    vix: float = np.nan
    vix_1d_pct_change: float = np.nan
    vix_spike_detected: bool = False
    bars_processed: int = 0
    is_first_bar: bool = True
    is_last_bar: bool = False
    vol_regime: str = "unknown"  # low | normal | elevated | high
    bars: list[OHLCV] = field(default_factory=list)  # 5-min bars for this session

    def update(self, bar: OHLCV, vix: float = np.nan, vix_1d_change: float = np.nan) -> None:
        """Update context with a new 5-minute bar.

        Accumulates VWAP, volume, and detects session boundaries.
        This method should be called ONCE per 5-min bar in the engine loop.
        """

        self.bars.append(bar)
        self.bars_processed += 1

        t = bar.timestamp
        t_time = t.time() if hasattr(t, 'time') else None

        # ── Detect first RTH bar (open auction) ───────────────────────────────
        if self.is_first_bar and is_rth_bar(t):
            self.session_open = float(bar.open)
            self.is_first_bar = False

        # ── Accumulate 5-min window metrics ─────────────────────────────────
        if self.first_5_min_done:
            pass  # Already accumulated
        elif is_first_5_min_bar(t):
            # We're inside the first 5 minutes — accumulate using bar high/low
            H = float(bar.high)
            L = float(bar.low)
            C = float(bar.close)
            V = int(bar.volume)
            self.first_5_min_volume += V
            if self.first_5_min_high < H:
                self.first_5_min_high = H
            if self.first_5_min_low > L:
                self.first_5_min_low = L
            # Running VWAP for first 5 min
            prev_vwap = self.first_5_min_vwap if np.isfinite(self.first_5_min_vwap) else 0.0
            if self.first_5_min_volume > 0:
                self.first_5_min_vwap = (
                    (prev_vwap * (self.first_5_min_volume - V) + C * V)
                    / self.first_5_min_volume
                )
        else:
            # First bar outside 9:30–9:35 — first-5-min window is closed
            self.first_5_min_done = True

        # ── Cumulative RTH volume ────────────────────────────────────────────
        self.rth_cumulative_volume += int(bar.volume)

        # ── VIX update ───────────────────────────────────────────────────────
        if np.isfinite(vix):
            prev_vix = self.vix
            self.vix = vix
            if np.isfinite(vix_1d_change):
                self.vix_1d_pct_change = vix_1d_change
            elif np.isfinite(prev_vix) and prev_vix > 0:
                self.vix_1d_pct_change = (vix - prev_vix) / prev_vix * 100.0

        # ── Vol regime ───────────────────────────────────────────────────────
        self._update_vol_regime()

        # ── Spike detection ──────────────────────────────────────────────────
        self.vix_spike_detected = (
            (np.isfinite(self.vix) and self.vix > 30) or
            (np.isfinite(self.vix_1d_pct_change) and abs(self.vix_1d_pct_change) > 20)
        )

        # ── Last bar before exit (10:00 AM ET) ─────────────────────────────
        self.is_last_bar = is_exit_time(t)

    def _update_vol_regime(self) -> None:
        v = self.vix
        if not np.isfinite(v):
            self.vol_regime = "unknown"
        elif v < 15:
            self.vol_regime = "low"
        elif v < 25:
            self.vol_regime = "normal"
        elif v < 30:
            self.vol_regime = "elevated"
        else:
            self.vol_regime = "high"

    def should_take_trade(self) -> bool:
        """Return True if we should allow new entries at current context.

        For opening_gap_fade:
        - Only enter on first bar of RTH if gap > 2%
        - Never enter after 9:35 AM ET
        - Skip pre-market volume > 150% of 20-day avg (news-driven gap)
        """
        return True  # Hook point for strategy-specific rules

    def gap_filter_passed(self, gap_threshold_pct: float, avg_pre_mkt_vol: float | None = None) -> bool:
        """Return True if the overnight gap exceeds threshold and pre-mkt volume is not elevated.

        Args:
            gap_threshold_pct: Minimum abs(gap%) to qualify (e.g. 2.0 for 2%)
            avg_pre_mkt_vol: 20-day average pre-market volume. If pre-mkt vol > 1.5× this, skip.

        The logic: institutional news-driven gaps (high pre-mkt volume) tend to
        continue rather than reverse. We only fade平淡 gaps where pre-mkt volume
        is near baseline (< 150% of 20-day avg).
        """
        if not np.isfinite(self.overnight_gap_pct):
            return False
        if abs(self.overnight_gap_pct) < gap_threshold_pct:
            return False
        if avg_pre_mkt_vol is not None and avg_pre_mkt_vol > 0:
            if self.pre_market_volume > 1.5 * avg_pre_mkt_vol:
                return False  # Likely a news gap, not a faded fade
        return self.is_first_bar  # Only open on first RTH bar


class TickContextHistory:
    """Builds and stores TickContext for each trading day.

    Usage:
        history = TickContextHistory(prev_close_map={date: close})
        for bar_5min in data_5min_bars:
            ctx = history.update(bar_5min, vix=vix)
            strategy.on_bar_intraday(bar_5min, ctx)
    """

    def __init__(
        self,
        prev_close_map: dict[datetime, float] | None = None,
        vix_map: dict[datetime, float] | None = None,
    ) -> None:
        self._prev_close_map: dict[datetime, float] = prev_close_map or {}
        self._vix_map: dict[datetime, float] = vix_map or {}
        self._current_ctx: TickContext | None = None
        self._all_contexts: list[TickContext] = []

    def update(self, bar: OHLCV, vix: float = np.nan) -> TickContext:
        """Process a 5-min bar and return the updated context.

        Creates a new TickContext when a new trading day begins (based on date).
        """
        ts = bar.timestamp
        date = ts.date() if hasattr(ts, 'date') else ts

        # New trading day → reset context
        if self._current_ctx is None or self._current_ctx.date.date() != date:
            # Get previous close from prior day
            prev = self._find_prev_close(ts)
            ctx = TickContext(
                date=ts,
                prev_close=prev,
                session_open=np.nan,
            )
            self._current_ctx = ctx
            self._all_contexts.append(ctx)

        # Compute VIX 1-day change
        vix_1d = self._vix_map.get(ts, np.nan) if self._vix_map else np.nan
        self._current_ctx.update(bar, vix=vix, vix_1d_change=vix_1d)
        return self._current_ctx

    def _find_prev_close(self, ts: datetime) -> float:
        """Find the most recent prior day's close as reference for gap calculation."""
        ts_date = ts.date() if hasattr(ts, 'date') else ts
        # Search backwards from current date
        for delta in range(1, 10):
            prev_date = datetime(*(ts_date.year, ts_date.month, ts_date.day)).timestamp()
            from datetime import timedelta
            check = ts_date - timedelta(days=delta)
            if check in self._prev_close_map:
                return self._prev_close_map[check]
        # Default: use close from first bar as fallback (cold start protection)
        return np.nan

    @property
    def current(self) -> TickContext | None:
        """The current (in-progress) session context."""
        return self._current_ctx

    @property
    def all_sessions(self) -> list[TickContext]:
        """All completed session contexts."""
        return [c for c in self._all_contexts if not c.is_first_bar]


# ── Slippage model for intraday bars ─────────────────────────────────────────

def compute_intraday_slippage(
    bar_timestamp: datetime,
    is_first_5_min: bool,
    iv_rank: float,
    instrument: str = "equity",
    base_bps: float = 1.0,
) -> float:
    """Compute realistic slippage in basis points for intraday execution.

    Opening auction (first 5 minutes) has elevated volatility → wider spreads.
    Options have wider spreads than equities.

    Models:
    - SPY/QQQ equity: 1 bp base
    - Options (equity options): 2 bp + 0.5bp per 10% IV Rank

    Args:
        bar_timestamp: When the fill occurs
        is_first_5_min: If True, double slippage (opening auction volatility)
        iv_rank: IV rank 0–100 (only used for options)
        instrument: "equity" or "option"
        base_bps: Base slippage in bps

    Returns:
        Slippage in basis points (e.g., 2.5 → 2.5 bps)
    """
    slip = float(base_bps)

    # Opening auction multiplier — first 5 min has 2× slippage due to auction vol
    if is_first_5_min_bar(bar_timestamp):
        slip *= 2.0

    # Options: IV-rank-adjusted spread
    if instrument == "option":
        slip += 0.5 * (iv_rank / 10.0)  # +0.5 bp per 10% IV Rank

    return float(slip)


# ── Intraday bar data handler ─────────────────────────────────────────────────

def build_5min_bar_df(
    df: pd.DataFrame,      # noqa: F821
    timeframe: str = "5min",
) -> pd.DataFrame:         # noqa: F821
    """Resample a tick/quote DataFrame to N-minute OHLCV bars.

    This exists because the main backtest engine uses daily bars by default.
    opening_gap_fade needs 5-min bars.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
           OR Tick-level data with [timestamp, price, volume]
        timeframe: Pandas offset alias (e.g., "5min", "1min", "15min")

    Returns:
        DataFrame resampled to the target timeframe with OHLCV columns.
    """
    import pandas as pd

    if df.empty:
        return df

    # Ensure timestamp index
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Check if we have OHLCV columns already
    has_ohlcv = all(c in df.columns for c in ["open", "high", "low", "close", "volume"])

    if has_ohlcv:
        # Resample OHLCV to larger timeframe
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        # Only include cols that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        resampled = df.resample(timeframe).agg(agg_dict)
        resampled = resampled.dropna(subset=["close"])
        resampled = resampled.reset_index()
        return resampled
    else:
        # Tick data — build OHLCV from price/volume
        df = df.sort_index()
        resampled = df.resample(timeframe).agg({
            "price": ["first", "max", "min", "last"],
            "volume": "sum",
        })
        resampled.columns = ["open", "high", "low", "close", "volume"]
        resampled = resampled.dropna(subset=["close"])
        resampled = resampled.reset_index()
        # Add symbol if missing
        if "symbol" not in resampled.columns:
            resampled["symbol"] = "UNKNOWN"
        return resampled
