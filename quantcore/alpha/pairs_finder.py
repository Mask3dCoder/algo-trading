"""Pairs finder using Engle-Granger cointegration and OU half-life.

For a given universe of stocks, computes all pair cointegration tests
on daily or intraday close prices and returns the top N pairs ranked
by fastest mean reversion (smallest half-life).

Supports intraday bar sizes via ``bar_size`` parameter.  When running
on minute bars, use ``reestimate_interval`` to trigger periodic
re-estimation of cointegration relationships.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

log = logging.getLogger(__name__)

# Trading bars per day for common bar sizes.
BARS_PER_DAY: dict[str, int] = {
    "1d": 1,
    "1h": 7,       # ~6.5h US session rounded up
    "30min": 13,
    "15min": 26,
    "5min": 78,
    "1min": 390,
}


def _parse_interval(interval: str) -> timedelta:
    """Parse a human interval string like '1h', '30min', '1d' into a timedelta."""
    interval = interval.strip().lower()
    if interval.endswith("min"):
        return timedelta(minutes=int(interval[:-3]))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return timedelta(days=int(interval[:-1]))
    raise ValueError(f"Unknown interval format: {interval!r} (expected '1h', '30min', '1d')")


@dataclass
class CointegratedPair:
    """Result of a cointegration test on a stock pair."""
    stock_a: str
    stock_b: str
    hedge_ratio: float       # beta from regressing B on A
    intercept: float         # intercept from regression
    adf_stat: float          # ADF test statistic
    adf_pvalue: float        # ADF p-value
    half_life: float         # OU half-life in bars (see half_life_unit)
    spread_mean: float       # mean of the spread (residuals)
    spread_std: float        # std of the spread
    half_life_unit: str = "bars"  # "bars" or "days"


class PairsFinder:
    """Find cointegrated stock pairs using Engle-Granger method.

    Parameters:
        lookback_days: number of trading days for the test window (default 252)
        max_pvalue: ADF p-value threshold for cointegration (default 0.05)
        max_pairs: max number of pairs to return (default 50)
        max_half_life: discard pairs with half-life above this (default 60 bars)
        min_half_life: discard pairs with half-life below this (default 1 bar)
        bar_size: bar granularity — "1d", "1h", "30min", "15min", "5min", "1min"
        reestimate_interval: how often to re-estimate pairs (e.g. "1h", "1d").
            None means estimate once and hold.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        max_pvalue: float = 0.05,
        max_pairs: int = 50,
        max_half_life: float = 60.0,
        min_half_life: float = 1.0,
        bar_size: str = "1d",
        reestimate_interval: str | None = None,
    ) -> None:
        self.lookback_days = lookback_days
        self.max_pvalue = max_pvalue
        self.max_pairs = max_pairs
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life
        self.bar_size = bar_size
        self.reestimate_interval = reestimate_interval
        self._bars_per_day = BARS_PER_DAY.get(bar_size, 1)
        self.lookback_bars = lookback_days * self._bars_per_day
        self._last_estimate_ts: datetime | None = None

    def _engle_granger(
        self,
        y: NDArray,
        x: NDArray,
    ) -> tuple[float, float, float, float, NDArray]:
        """Run Engle-Granger cointegration test.

        Returns (hedge_ratio, intercept, adf_stat, adf_pvalue, spread).
        """
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        from statsmodels.tsa.stattools import adfuller

        # Regress y on x (with intercept)
        X = add_constant(x)
        model = OLS(y, X).fit()
        intercept, hedge_ratio = model.params
        spread = model.resid

        # ADF test on residuals — use AIC lag selection and constant
        # regression="c" is more robust than "nc" for finite samples
        adf_result = adfuller(spread, maxlag=12, autolag="AIC", regression="c")
        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])

        return hedge_ratio, intercept, adf_stat, adf_pvalue, spread

    def _ou_half_life(self, spread: NDArray) -> float:
        """Estimate Ornstein-Uhlenbeck half-life of mean reversion.

        Fits: dS(t) = theta * (mu - S(t-1)) * dt + epsilon
        Equivalent to AR(1) on spread differences:
            spread_diff(t) = a + b * spread(t-1) + e
        where theta = -b, half_life = ln(2) / theta
        """
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        spread = np.asarray(spread, dtype=float)
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)

        X = add_constant(spread_lag)
        model = OLS(spread_diff, X).fit()
        b = model.params[1]

        if b >= 0:
            return 999.0  # no mean reversion

        theta = -b
        half_life = np.log(2) / theta
        return float(half_life)

    def find_pairs(
        self,
        prices: pd.DataFrame,
        max_test_pairs: int | None = None,
    ) -> list[CointegratedPair]:
        """Find cointegrated pairs from a DataFrame of close prices.

        Args:
            prices: DataFrame with DatetimeIndex and ticker columns.
                    Last ``lookback_bars`` rows are used (computed from
                    ``lookback_days * bars_per_day``).
            max_test_pairs: if set, randomly sample this many pairs to test
                           (for speed on large universes). None = test all.

        Returns:
            List of CointegratedPair sorted by half-life (fastest first).
        """
        # Use last lookback_bars of data
        prices = prices.iloc[-self.lookback_bars:].dropna(axis=1, how="any")

        if prices.shape[1] < 2:
            log.warning("Need at least 2 stocks with complete data, got %d", prices.shape[1])
            return []

        tickers = list(prices.columns)
        all_pairs = list(itertools.combinations(tickers, 2))
        log.info("Testing %d pairs on %d days of data", len(all_pairs), len(prices))

        if max_test_pairs and len(all_pairs) > max_test_pairs:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(all_pairs), size=max_test_pairs, replace=False)
            all_pairs = [all_pairs[i] for i in sorted(indices)]
            log.info("Subsampled to %d pairs", max_test_pairs)

        results: list[CointegratedPair] = []

        for i, (a, b) in enumerate(all_pairs):
            if (i + 1) % 100 == 0:
                log.info("  Tested %d / %d pairs, %d cointegrated", i + 1, len(all_pairs), len(results))

            # Use log prices for cointegration — normalizes scale differences
            y = np.log(prices[b].values.astype(float))
            x = np.log(prices[a].values.astype(float))

            try:
                hedge_ratio, intercept, adf_stat, adf_pvalue, spread = self._engle_granger(y, x)
            except Exception:
                continue

            if adf_pvalue > self.max_pvalue:
                continue

            half_life = self._ou_half_life(spread)

            if half_life < self.min_half_life or half_life > self.max_half_life:
                continue

            results.append(CointegratedPair(
                stock_a=a,
                stock_b=b,
                hedge_ratio=round(hedge_ratio, 6),
                intercept=round(intercept, 6),
                adf_stat=round(adf_stat, 4),
                adf_pvalue=round(adf_pvalue, 6),
                half_life=round(half_life, 2),
                spread_mean=round(float(np.mean(spread)), 6),
                spread_std=round(float(np.std(spread)), 6),
                half_life_unit="bars",
            ))

        results.sort(key=lambda p: p.half_life)
        top = results[:self.max_pairs]

        log.info("Found %d cointegrated pairs (returned top %d)", len(results), len(top))
        self.mark_estimated(datetime.utcnow())
        return top

    def should_reestimate(self, current_ts: datetime) -> bool:
        """Check whether pairs should be re-estimated at *current_ts*.

        Returns True when ``reestimate_interval`` has elapsed since the
        last estimation, or when no estimation has been performed yet.
        """
        if self.reestimate_interval is None:
            return self._last_estimate_ts is None
        if self._last_estimate_ts is None:
            return True
        delta = current_ts - self._last_estimate_ts
        interval_td = _parse_interval(self.reestimate_interval)
        return delta >= interval_td

    def mark_estimated(self, ts: datetime) -> None:
        """Record that pairs were (re-)estimated at *ts*."""
        self._last_estimate_ts = ts

    @staticmethod
    def pairs_to_dataframe(pairs: list[CointegratedPair]) -> pd.DataFrame:
        """Convert pairs list to a DataFrame for storage."""
        if not pairs:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "stock_a": p.stock_a,
                "stock_b": p.stock_b,
                "hedge_ratio": p.hedge_ratio,
                "intercept": p.intercept,
                "adf_stat": p.adf_stat,
                "adf_pvalue": p.adf_pvalue,
                "half_life": p.half_life,
                "spread_mean": p.spread_mean,
                "spread_std": p.spread_std,
                "half_life_unit": p.half_life_unit,
            }
            for p in pairs
        ])

    @staticmethod
    def dataframe_to_pairs(df: pd.DataFrame) -> list[CointegratedPair]:
        """Load pairs from a DataFrame."""
        pairs = []
        for _, row in df.iterrows():
            pairs.append(CointegratedPair(
                stock_a=row["stock_a"],
                stock_b=row["stock_b"],
                hedge_ratio=row["hedge_ratio"],
                intercept=row["intercept"],
                adf_stat=row["adf_stat"],
                adf_pvalue=row["adf_pvalue"],
                half_life=row["half_life"],
                spread_mean=row["spread_mean"],
                spread_std=row["spread_std"],
                half_life_unit=row.get("half_life_unit", "bars"),
            ))
        return pairs
