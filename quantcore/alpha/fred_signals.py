"""Composite regime signal constructors from FRED macroeconomic data.

Each function takes a wide DataFrame of FRED series (as built by
``FREDDataLoader.build_macro_dataframe()``) and returns a pd.Series of
z-scored composite signals at daily frequency.

Signals are designed for regime classification, NOT for direct trading.
They feed into ``FREDRegimeStrategy`` which maps regime combinations to
position sizing and horizon selection.

Signals:
  - ``yield_curve_regime()`` — curve shape + Fed stance
  - ``inflation_pressure()`` — price-level + market-implied expectations
  - ``financial_conditions()`` — credit stress + volatility + uncertainty
  - ``economic_momentum()`` — employment + production + GDP growth

All signals are z-scored over a rolling window to ensure stationarity
and comparability across regimes.
"""

from __future__ import annotations

import pandas as pd


def yield_curve_regime(macro_df: pd.DataFrame) -> pd.Series:
    """Yield curve regime signal: steepening vs. flattening/inversion.

    Components (equal-weighted z-scores):
      - T10Y2Y: 10Y-2Y spread (primary)
      - T10Y3M: 10Y-3M spread (NY Fed recession model input)
      - DGS10:  level (high = restrictive)
      - DFF:    Fed funds rate (tightening proxy)

    Returns pd.Series of z-scored composite.  Positive = steepening/easing,
    negative = flattening/tightening.
    """
    cols = ["T10Y2Y", "T10Y3M", "DGS10", "DFF"]
    available = [c for c in cols if c in macro_df.columns]
    if not available:
        return pd.Series(dtype=float, name="yield_curve_regime")

    z = _zscore(macro_df[available])
    return z.mean(axis=1).rename("yield_curve_regime")


def inflation_pressure(macro_df: pd.DataFrame) -> pd.Series:
    """Inflation pressure signal: rising vs. falling price pressures.

    Components:
      - T5YIE:  5Y breakeven (market-implied, daily)
      - T10YIE: 10Y breakeven (market-implied, daily)
      - CPIAUCSL YoY: headline CPI (monthly, forward-filled)
      - PCEPI YoY:     PCE price index (monthly, forward-filled)

    Returns pd.Series of z-scored composite.  Positive = rising inflation,
    negative = disinflation/deflation.
    """
    components: list[pd.Series] = []

    # Market-implied expectations (daily, no ALFRED needed)
    for col in ("T5YIE", "T10YIE"):
        if col in macro_df.columns:
            components.append(macro_df[col])

    # Realized inflation (YoY change of monthly series)
    for col in ("CPIAUCSL", "PCEPI"):
        if col in macro_df.columns:
            yoy = macro_df[col].pct_change(periods=252)  # ~1yr in daily data
            yoy.name = f"{col}_yoy"
            components.append(yoy)

    if not components:
        return pd.Series(dtype=float, name="inflation_pressure")

    combined = pd.concat(components, axis=1).dropna()
    if combined.empty:
        return pd.Series(dtype=float, name="inflation_pressure")

    z = _zscore(combined)
    return z.mean(axis=1).rename("inflation_pressure")


def financial_conditions(macro_df: pd.DataFrame) -> pd.Series:
    """Financial conditions signal: stress vs. accommodation.

    Components (higher = tighter/stressed):
      - BAMLH0A0HYM2: HY OAS (credit stress)
      - BAMLC0A0CM:   IG OAS (investment-grade stress)
      - VIXCLS:       VIX closing value
      - USEPUINDXD:   Economic Policy Uncertainty Index

    Returns pd.Series of z-scored composite.  Positive = tighter/stressed
    conditions, negative = accommodative conditions.
    """
    cols = ["BAMLH0A0HYM2", "BAMLC0A0CM", "VIXCLS", "USEPUINDXD"]
    available = [c for c in cols if c in macro_df.columns]
    if not available:
        return pd.Series(dtype=float, name="financial_conditions")

    z = _zscore(macro_df[available])
    return z.mean(axis=1).rename("financial_conditions")


def economic_momentum(macro_df: pd.DataFrame) -> pd.Series:
    """Economic momentum signal: accelerating vs. decelerating activity.

    Components:
      - UNRATE:   unemployment rate (level — inverted: lower = stronger)
      - PAYEMS:   nonfarm payrolls (YoY change)
      - INDPRO:   industrial production (YoY change)
      - RSAFS:    retail sales (YoY change)
      - GDP:      GDP (YoY change, quarterly, forward-filled)

    Returns pd.Series of z-scored composite.  Positive = accelerating
    activity, negative = decelerating/contraction.
    """
    components: list[pd.Series] = []

    # Unemployment — invert so positive = strong labor market
    if "UNRATE" in macro_df.columns:
        inv_unrate = -macro_df["UNRATE"]
        inv_unrate.name = "inv_UNRATE"
        components.append(inv_unrate)

    # YoY changes for level series
    for col in ("PAYEMS", "INDPRO", "RSAFS", "GDP"):
        if col in macro_df.columns:
            yoy = macro_df[col].pct_change(periods=252)
            yoy.name = f"{col}_yoy"
            components.append(yoy)

    if not components:
        return pd.Series(dtype=float, name="economic_momentum")

    combined = pd.concat(components, axis=1).dropna()
    if combined.empty:
        return pd.Series(dtype=float, name="economic_momentum")

    z = _zscore(combined)
    return z.mean(axis=1).rename("economic_momentum")


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _zscore(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Rolling z-score normalization.

    Uses a 252-trading-day (1-year) rolling window so that the signal
    adapts to structural level shifts while remaining stationary.
    """
    rolling_mean = df.rolling(window, min_periods=max(window // 2, 20)).mean()
    rolling_std = df.rolling(window, min_periods=max(window // 2, 20)).std()
    return (df - rolling_mean) / rolling_std.replace(0, 1e-10)
