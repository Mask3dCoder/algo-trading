"""Unusual options activity detection: sweep/block classification and premium analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantcore.data.point_in_time import feature_frame_from_records


class UOAType(str, Enum):
    SWEEP = "sweep"
    BLOCK = "block"
    SPLIT = "split"
    SPREAD = "spread"
    BUTTERFLY = "butterfly"
    CONDOR = "condor"
    UNKNOWN = "unknown"


@dataclass
class UOAConfig:
    """Detection thresholds for unusual options activity."""
    min_premium_threshold: float = 100_000.0
    min_volume_ratio: float = 3.0
    min_oi_ratio: float = 2.0
    lookback_minutes: int = 30
    exclude_market_makers: bool = True
    sweep_min_volume: int = 50
    block_min_volume: int = 500


@dataclass
class UOATrade:
    """A single unusual options trade."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    valid_from: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    option_type: str = ""
    strike: float = 0.0
    expiration: str = ""
    price: float = 0.0
    quantity: int = 0
    premium: float = 0.0
    side: str = ""
    trade_type: UOAType = UOAType.UNKNOWN
    exchange: str = ""
    conditions: list[str] = field(default_factory=list)
    uoa_score: float = 0.0


@dataclass
class OptionsFlowConfig:
    """Configuration for OPRA options-flow feature generation."""

    min_large_order_notional: float = 50_000.0
    default_avg_daily_volume: float = 1.0
    source: str = "alpaca_options"


class OptionsFlowAnalyzer:
    """Compute minimum OPRA flow features from trade-stream records.

    Required feature set:
    - unusual_volume_ratio: contract volume / 30-day average volume
    - put_call_ratio_by_expiry: put volume / call volume by expiration
    - large_order_flag: any single trade > configured notional threshold
    - skew: OTM put IV minus OTM call IV
    - term_structure: front-month IV minus back-month IV

    Every returned row follows the unified feature schema and uses
    ``valid_from = trade_timestamp``.
    """

    def __init__(self, config: OptionsFlowConfig | None = None) -> None:
        self.config = config or OptionsFlowConfig()

    def compute_feature_frame(
        self,
        trades: list[dict],
        historical_avg_volume: dict[str, float] | None = None,
        as_of: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return OPRA flow features in the canonical feature-frame schema."""

        if not trades:
            return pd.DataFrame()

        historical_avg_volume = historical_avg_volume or {}
        df = pd.DataFrame(trades)
        if df.empty:
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["valid_from"] = df["timestamp"]
        df["as_of"] = pd.to_datetime(as_of or df["timestamp"].dt.normalize(), utc=True).dt.normalize()
        df["underlying"] = df.get("underlying", df.get("symbol", "")).astype(str).str.upper()
        df["option_type"] = df["option_type"].astype(str).str.lower()
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
        df["iv"] = pd.to_numeric(df.get("iv", 0.0), errors="coerce").fillna(0.0)
        df["volume"] = pd.to_numeric(df.get("volume", df.get("quantity", 0)), errors="coerce").fillna(0.0)
        df["price"] = pd.to_numeric(df.get("price", 0.0), errors="coerce").fillna(0.0)
        df["strike"] = pd.to_numeric(df.get("strike", 0.0), errors="coerce").fillna(0.0)
        df["spot"] = pd.to_numeric(df.get("spot", 0.0), errors="coerce").fillna(0.0)
        df["notional"] = df["price"] * df["volume"] * 100.0

        records: list[dict] = []
        for underlying, group in df.groupby("underlying", sort=True):
            symbol = str(underlying).upper()
            as_of_ts = group["as_of"].min()
            valid_from = group["valid_from"].max()
            total_volume = float(group["volume"].sum())
            avg_volume = float(historical_avg_volume.get(symbol, self.config.default_avg_daily_volume))
            unusual_ratio = total_volume / avg_volume if avg_volume > 0 else 0.0

            put_call = self._put_call_ratio_by_expiry(group)
            large_order = bool((group["notional"] > self.config.min_large_order_notional).any())
            skew = self._otm_iv_skew(group)
            term_structure = self._term_structure(group)

            signal_values = {
                "unusual_volume_ratio": unusual_ratio,
                "put_call_ratio_by_expiry": put_call,
                "large_order_flag": float(large_order),
                "skew": skew,
                "term_structure": term_structure,
            }
            for feature_name, value in signal_values.items():
                records.append({
                    "source": self.config.source,
                    "symbol": symbol,
                    "as_of": as_of_ts,
                    "valid_from": valid_from,
                    "feature_name": feature_name,
                    "value": float(value),
                })

        return feature_frame_from_records(records)

    @staticmethod
    def _put_call_ratio_by_expiry(group: pd.DataFrame) -> float:
        ratios: list[float] = []
        for _, expiry_group in group.groupby("expiration", dropna=True):
            puts = float(expiry_group.loc[expiry_group["option_type"].eq("put"), "volume"].sum())
            calls = float(expiry_group.loc[expiry_group["option_type"].eq("call"), "volume"].sum())
            ratios.append(puts / calls if calls > 0 else (float("inf") if puts > 0 else 1.0))
        return float(np.nanmean(ratios)) if ratios else 1.0

    @staticmethod
    def _otm_iv_skew(group: pd.DataFrame) -> float:
        spot = float(group["spot"].replace(0, np.nan).median())
        if not np.isfinite(spot) or spot <= 0:
            return 0.0

        put_otm = group[(group["option_type"].eq("put")) & (group["strike"] < spot)]["iv"]
        call_otm = group[(group["option_type"].eq("call")) & (group["strike"] > spot)]["iv"]
        put_iv = float(put_otm.mean()) if not put_otm.empty else 0.0
        call_iv = float(call_otm.mean()) if not call_otm.empty else 0.0
        return put_iv - call_iv

    @staticmethod
    def _term_structure(group: pd.DataFrame) -> float:
        valid = group.dropna(subset=["expiration", "iv"]).copy()
        if valid.empty:
            return 0.0
        as_of_naive = pd.Timestamp(valid["as_of"].iloc[0]).tz_localize(None)
        valid["dte"] = (valid["expiration"].dt.tz_localize(None) - as_of_naive).dt.days
        valid = valid.loc[valid["dte"] > 0].sort_values("dte")
        if valid.empty:
            return 0.0
        front = float(valid.iloc[0]["iv"])
        back = float(valid.iloc[-1]["iv"])
        return front - back


class UOADetector:
    """Detect unusual options activity from trade stream data."""

    def __init__(self, config: UOAConfig | None = None) -> None:
        self.config = config or UOAConfig()

    def classify_trade(self, trade: dict) -> UOAType:
        qty = int(trade.get("quantity", 0))
        conditions = trade.get("conditions", [])
        cond_str = " ".join(str(c) for c in conditions).lower()
        if qty >= self.config.block_min_volume:
            if "sweep" in cond_str or "intermarket" in cond_str:
                return UOAType.SWEEP
            return UOAType.BLOCK
        if qty >= self.config.sweep_min_volume:
            if "sweep" in cond_str or "iso" in cond_str:
                return UOAType.SWEEP
        if "split" in cond_str:
            return UOAType.SPLIT
        if "spread" in cond_str:
            return UOAType.SPREAD
        return UOAType.UNKNOWN

    def is_unusual(
        self,
        trade: UOATrade,
        historical_volumes: NDArray[np.floating],
        open_interest: float,
    ) -> bool:
        premium = trade.premium
        volume = float(trade.quantity)
        if premium < self.config.min_premium_threshold:
            return False
        if len(historical_volumes) >= 5:
            vol_mean = float(np.mean(historical_volumes))
            vol_std = float(np.std(historical_volumes))
            if vol_std > 0:
                vol_ratio = (volume - vol_mean) / vol_std
                if vol_ratio < self.config.min_volume_ratio:
                    return False
        if open_interest > 0 and volume / open_interest < self.config.min_oi_ratio:
            return False
        return True

    def scan(
        self,
        trades: list[dict],
        historical_data: dict[str, NDArray[np.floating]] | None = None,
        open_interest: dict[str, float] | None = None,
    ) -> list[UOATrade]:
        historical_data = historical_data or {}
        open_interest = open_interest or {}
        unusual_trades: list[UOATrade] = []
        for t in trades:
            trade_type = self.classify_trade(t)
            symbol = str(t.get("symbol", ""))
            qty = int(t.get("quantity", 0))
            price = float(t.get("price", 0))
            timestamp = t.get("timestamp", datetime.utcnow())
            uoa = UOATrade(
                timestamp=timestamp,
                valid_from=t.get("valid_from", timestamp),
                symbol=symbol,
                option_type=str(t.get("option_type", "")),
                strike=float(t.get("strike", 0)),
                expiration=str(t.get("expiration", "")),
                price=price,
                quantity=qty,
                premium=price * qty * 100.0,
                side=str(t.get("side", "")),
                trade_type=trade_type,
                exchange=str(t.get("exchange", "")),
                conditions=list(t.get("conditions", [])),
            )
            hist_vol = historical_data.get(symbol, np.array([], dtype=float))
            oi = open_interest.get(symbol, 0.0)
            if self.is_unusual(uoa, hist_vol, oi):
                unusual_trades.append(uoa)
        return unusual_trades

    def filter_sweeps(self, trades: list[UOATrade]) -> list[UOATrade]:
        return [t for t in trades if t.trade_type == UOAType.SWEEP]

    def aggregate_by_symbol(self, trades: list[UOATrade]) -> dict[str, list[UOATrade]]:
        result: dict[str, list[UOATrade]] = {}
        for t in trades:
            result.setdefault(t.symbol, []).append(t)
        return result

    def compute_uoa_score(self, trades: list[UOATrade], symbol: str) -> float:
        sym_trades = [t for t in trades if t.symbol == symbol]
        if not sym_trades:
            return 0.0
        total_premium = sum(t.premium for t in sym_trades)
        premium_score = min(total_premium / (self.config.min_premium_threshold * 5), 1.0)
        trade_count_score = min(len(sym_trades) / 10.0, 1.0)
        sweep_ratio = len([t for t in sym_trades if t.trade_type == UOAType.SWEEP]) / max(len(sym_trades), 1)
        return float(premium_score * 0.5 + trade_count_score * 0.2 + sweep_ratio * 0.3)


def detect_sweep_cascade(
    trades: list[UOATrade], time_window_sec: float = 60.0
) -> list[list[UOATrade]]:
    """Group sweeps executed in rapid succession (cascade detection)."""
    sweeps = sorted(
        [t for t in trades if t.trade_type == UOAType.SWEEP],
        key=lambda x: x.timestamp,
    )
    cascades: list[list[UOATrade]] = []
    current: list[UOATrade] = []
    for s in sweeps:
        if not current:
            current = [s]
        else:
            dt = (s.timestamp - current[-1].timestamp).total_seconds()
            if dt <= time_window_sec and s.symbol == current[0].symbol:
                current.append(s)
            else:
                if len(current) >= 3:
                    cascades.append(current)
                current = [s]
    if len(current) >= 3:
        cascades.append(current)
    return cascades


def compute_premium_sentiment(trades: list[UOATrade]) -> dict[str, float]:
    """Call premium / total premium per symbol. > 0.5 = bullish, < 0.5 = bearish."""
    by_sym: dict[str, tuple[float, float]] = {}
    for t in trades:
        call_prem, total_prem = by_sym.get(t.symbol, (0.0, 0.0))
        if t.option_type.lower() == "call":
            call_prem += t.premium
        total_prem += t.premium
        by_sym[t.symbol] = (call_prem, total_prem)
    return {
        sym: (call_prem / total_prem if total_prem > 0 else 0.5)
        for sym, (call_prem, total_prem) in by_sym.items()
    }
