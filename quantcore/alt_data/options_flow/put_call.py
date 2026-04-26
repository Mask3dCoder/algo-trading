"""Advanced put/call analysis: PCR term structure, equity-only filtering, sentiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


@dataclass
class PCRConfig:
    equity_only: bool = True
    min_volume: int = 50
    lookback_days: int = 20


@dataclass
class PCRSnapshot:
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    symbol: str = ""
    total_put_volume: float = 0.0
    total_call_volume: float = 0.0
    total_put_oi: float = 0.0
    total_call_oi: float = 0.0
    volume_pcr: float = 0.0
    oi_pcr: float = 0.0
    pcr_1d_change: float = 0.0
    pcr_5d_avg: float = 0.0
    pcr_zscore: float = 0.0
    pcr_percentile: float = 50.0
    sentiment: str = "neutral"


class PutCallAnalyzer:
    """Compute put/call ratios with historical context and sentiment classification."""

    def __init__(self, config: PCRConfig | None = None) -> None:
        self.config = config or PCRConfig()
        self._history: list[PCRSnapshot] = []

    def compute(self, trades: list[dict]) -> PCRSnapshot:
        """Compute PCR snapshot from a list of option trades.

        Each trade dict needs: option_type (call/put), volume, open_interest.
        """
        total_call_vol = 0.0
        total_put_vol = 0.0
        total_call_oi = 0.0
        total_put_oi = 0.0

        for t in trades:
            otype = str(t.get("option_type", "")).lower()
            vol = float(t.get("volume", t.get("quantity", 0)))
            oi = float(t.get("open_interest", 0))
            if vol < self.config.min_volume:
                continue
            if otype == "call":
                total_call_vol += vol
                total_call_oi += oi
            elif otype == "put":
                total_put_vol += vol
                total_put_oi += oi

        vol_pcr = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0
        oi_pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

        snapshot = PCRSnapshot(
            total_put_volume=total_put_vol, total_call_volume=total_call_vol,
            total_put_oi=total_put_oi, total_call_oi=total_call_oi,
            volume_pcr=float(vol_pcr), oi_pcr=float(oi_pcr),
        )

        if self._history:
            prev = self._history[-1]
            snapshot.pcr_1d_change = float(snapshot.volume_pcr - prev.volume_pcr)

        recent = self._history[-5:] + [snapshot] if self._history else [snapshot]
        snapshot.pcr_5d_avg = float(np.mean([s.volume_pcr for s in recent]))

        self._history.append(snapshot)
        if len(self._history) > 252:
            self._history = self._history[-252:]

        return snapshot

    def compute_rolling(self, history: list[PCRSnapshot]) -> PCRSnapshot:
        """Compute PCR with historical context (z-score and percentile)."""
        if len(history) < self.config.lookback_days:
            return history[-1] if history else PCRSnapshot()
        current = history[-1]
        recent = history[-self.config.lookback_days:]
        pcr_values = np.array([s.volume_pcr for s in recent])
        mean, std = float(np.mean(pcr_values)), float(np.std(pcr_values, ddof=1))
        if std > 0:
            current.pcr_zscore = float((current.volume_pcr - mean) / std)
        current.pcr_percentile = float(np.mean(pcr_values <= current.volume_pcr) * 100.0)
        current.sentiment = self.classify_sentiment(current.volume_pcr, current.pcr_5d_avg)
        # Update the last entry in history
        self._history.append(current)
        return current

    def classify_sentiment(self, pcr: float, pcr_5d_avg: float) -> str:
        """Classify market sentiment based on PCR levels."""
        if pcr > 1.2:
            return "bearish"
        elif pcr > 0.9:
            return "neutral_bearish"
        elif pcr < 0.6:
            return "bullish"
        elif pcr < 0.8:
            return "neutral_bullish"
        return "neutral"


def compute_equity_only_pcr(trades: list[dict]) -> float:
    """Compute PCR excluding index and ETF options."""
    etf_symbols = {"SPY", "QQQ", "IWM", "DIA", "EEM", "XLF", "XLE", "XLK", "XLV", "XLI",
                   "XLP", "XLY", "XLU", "XLB", "XLRE", "TLT", "GLD", "USO", "VXX"}
    total_call_vol = 0.0
    total_put_vol = 0.0
    for t in trades:
        symbol = str(t.get("symbol", "")).upper()
        if symbol in etf_symbols:
            continue
        otype = str(t.get("option_type", "")).lower()
        vol = float(t.get("volume", t.get("quantity", 0)))
        if otype == "call":
            total_call_vol += vol
        elif otype == "put":
            total_put_vol += vol
    return total_put_vol / total_call_vol if total_call_vol > 0 else 1.0


def pcr_term_structure(pcr_by_expiry: dict[str, float]) -> dict[str, Any]:
    """Analyze PCR across expirations — slope and shape."""
    if len(pcr_by_expiry) < 2:
        return {"slope": 0.0, "shape": "flat", "values": pcr_by_expiry}
    expiries = sorted(pcr_by_expiry.keys())
    values = np.array([pcr_by_expiry[e] for e in expiries])
    x = np.arange(len(values))
    slope = float(np.polyfit(x, values, 1)[0]) if len(values) >= 2 else 0.0
    if slope > 0.05:
        shape = "steepening"
    elif slope < -0.05:
        shape = "flattening"
    else:
        shape = "flat"
    return {"slope": slope, "shape": shape, "values": pcr_by_expiry}
