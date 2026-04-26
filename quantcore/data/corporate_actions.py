"""Corporate actions: split/dividend/merger adjustments, delisting handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

import pandas as pd


class ActionType(str, Enum):
    SPLIT = "split"
    DIVIDEND = "dividend"
    MERGER = "merger"
    SPINOFF = "spinoff"
    DELISTING = "delisting"


@dataclass
class AdjustmentEvent:
    symbol: str
    date: date
    action_type: ActionType
    value: float  # split ratio, dividend amount, merger ratio, etc.
    description: str = ""


def adjust_for_split(
    prices: pd.DataFrame,
    split_ratio: float,
    split_date: date,
) -> pd.DataFrame:
    """Adjust OHLCV prices for a stock split.

    For a 4:1 split with ratio=4: prices before split_date are divided by 4,
    volumes before split_date are multiplied by 4.
    """
    df = prices.copy()
    mask = pd.to_datetime(df["timestamp"]) < pd.Timestamp(split_date)

    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            df.loc[mask, col] = df.loc[mask, col] / split_ratio

    if "volume" in df.columns:
        df.loc[mask, "volume"] = df.loc[mask, "volume"] * split_ratio

    return df


def adjust_for_dividend(
    prices: pd.DataFrame,
    div_amount: float,
    ex_date: date,
) -> pd.DataFrame:
    """Adjust prices for dividend (proportional price adjustment).

    On ex-date, price drops by approximately div_amount * (1 - tax_rate).
    We adjust pre-ex-date prices: adj_close = close * (1 - div/close) for each prior day.
    """
    df = prices.copy()
    ex_ts = pd.Timestamp(ex_date)
    mask = pd.to_datetime(df["timestamp"]) < ex_ts

    # Proportional adjustment: multiply pre-ex-date adj_close by (1 - div/close)
    if "adj_close" in df.columns:
        for idx in df.index[mask]:
            close = df.loc[idx, "close"]
            if close > 0:
                df.loc[idx, "adj_close"] = df.loc[idx, "adj_close"] * (1.0 - div_amount / close)

    return df


def handle_merger(
    old_symbol: str,
    new_symbol: str,
    ratio: float,
    effective_date: date,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Handle merger: remap old symbol to new, adjust quantities by ratio."""
    df = prices.copy()
    mask = pd.to_datetime(df["timestamp"]) >= pd.Timestamp(effective_date)

    if "symbol" in df.columns:
        df.loc[mask & (df["symbol"] == old_symbol), "symbol"] = new_symbol

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df.loc[mask, col] = df.loc[mask, col] * ratio

    return df


def handle_delisting(
    symbol: str,
    last_date: date,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Mark delisted stock. Retain historical data with a delisted flag."""
    df = prices.copy()
    df["is_delisted"] = False
    mask = (df["symbol"] == symbol) & (pd.to_datetime(df["timestamp"]) > pd.Timestamp(last_date))
    df.loc[mask, "is_delisted"] = True
    return df


def get_adjustment_factors(
    actions: list[AdjustmentEvent],
    date_range: pd.DatetimeIndex,
) -> pd.Series:
    """Compute cumulative adjustment factor for each date in the range."""
    factors = pd.Series(1.0, index=date_range)

    for action in sorted(actions, key=lambda a: a.date, reverse=True):
        action_ts = pd.Timestamp(action.date)
        if action.action_type == ActionType.SPLIT:
            # Pre-split prices need to be divided by ratio
            factors[factors.index < action_ts] /= action.value
        elif action.action_type == ActionType.DIVIDEND:
            # Apply proportional adjustment
            # This is simplified; exact adjustment depends on close price
            factors[factors.index < action_ts] *= (1.0 - action.value * 0.01)  # Approx

    return factors
