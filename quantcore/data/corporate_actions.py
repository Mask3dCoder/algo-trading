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


def _split_on_date(
    actions: list[AdjustmentEvent], date: date
) -> float | None:
    """Return split ratio for any split on the given date, or None."""
    for a in actions:
        if a.action_type == ActionType.SPLIT and a.date == date:
            return a.value
    return None


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
    split_ratio: float | None = None,
) -> pd.DataFrame:
    """Adjust prices for dividend (proportional price adjustment).

    On the ex-date the price drops by the dividend amount (modulo taxes).
    We back-adjust all pre-ex-date `adj_close` by multiplying by the
    proportional factor (1 - div_amount / cum_close), where cum_close
    is the closing price on the last cum-dividend day.

    If `split_ratio` is provided, the dividend is first divided by the split
    ratio (dividends are declared on a per-share basis and therefore shrink
    after a stock split).
    """
    df = prices.copy()
    ex_ts = pd.Timestamp(ex_date)
    mask = pd.to_datetime(df["timestamp"]) < ex_ts

    # Normalize dividend for any same-date split
    adj_div = div_amount / split_ratio if split_ratio and split_ratio > 0 else div_amount

    # Proportional adjustment using the cum-dividend close
    if "adj_close" in df.columns:
        pre_ex = df[mask]
        if not pre_ex.empty:
            cum_close = float(pre_ex["close"].iloc[-1])
            if cum_close > 0:
                factor = 1.0 - adj_div / cum_close
                df.loc[mask, "adj_close"] *= factor

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
    price_data: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute cumulative adjustment factor for each date in the range.

    For dividends, the exact proportional factor is:
        factor = 1.0 - (div_amount / cum_close)
    where cum_close is the closing price on the last cum-dividend day.
    If price_data is not provided, falls back to a heuristic.

    When a dividend and split occur on the same ex-date, the dividend is
    applied *post-split*: the dividend amount is divided by the split ratio
    before computing the proportional factor.
    """
    factors = pd.Series(1.0, index=date_range)

    for action in sorted(actions, key=lambda a: a.date, reverse=True):
        action_ts = pd.Timestamp(action.date)
        if action.action_type == ActionType.SPLIT:
            # Pre-split prices need to be divided by ratio
            factors[factors.index < action_ts] /= action.value
        elif action.action_type == ActionType.DIVIDEND:
            # Adjust dividend for any split on the same ex-date
            split_ratio = _split_on_date(actions, action.date)
            div_value = action.value
            if split_ratio is not None and split_ratio > 0:
                div_value /= split_ratio

            if price_data is not None and "close" in price_data.columns:
                pre = price_data[pd.to_datetime(price_data["timestamp"]) < action_ts]
                if not pre.empty:
                    cum_close = float(pre["close"].iloc[-1])
                    factor = 1.0 - div_value / cum_close if cum_close > 0 else 1.0
                else:
                    factor = 1.0 - div_value * 0.01
            else:
                factor = 1.0 - div_value * 0.01
            factors[factors.index < action_ts] *= factor

    return factors
