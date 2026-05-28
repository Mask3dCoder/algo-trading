"""Classic Jegadeesh-Titman cross-sectional momentum strategy.

Long top decile / short bottom decile of 12-1 month momentum,
equal-weighted, dollar-neutral, monthly rebalance.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def _last_trading_day_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on the last trading day of each month."""
    months = dates.to_period("M")
    is_last = months.shift(-1) != months
    return pd.Series(is_last, index=dates)


class EquityMomentumStrategy:
    """Jegadeesh-Titman cross-sectional momentum.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        Daily close prices. Index = timestamps, columns = tickers.
    stock_volumes : pd.DataFrame | None
        Daily dollar volume (close * volume). Same shape as stock_prices.
        If None, the volume filter is skipped.
    formation_months : int
        Lookback period for momentum signal (default: 12).
    skip_month : int
        Skip the most recent N months to avoid reversal (default: 1).
    top_quantile : float
        Fraction of stocks to go long (default: 0.1).
    bottom_quantile : float
        Fraction of stocks to go short (default: 0.1).
    min_price : float
        Minimum stock price filter (default: 5.0).
    min_volume : float
        Minimum average 20-day dollar volume (default: 5e6).
    min_history_days : int
        Minimum trading days in formation window (default: 200).
    """

    def __init__(
        self,
        stock_prices: pd.DataFrame,
        stock_volumes: pd.DataFrame | None = None,
        *,
        formation_months: int = 12,
        skip_month: int = 1,
        top_quantile: float = 0.1,
        bottom_quantile: float = 0.1,
        min_price: float = 5.0,
        min_volume: float = 5e6,
        min_history_days: int = 200,
    ) -> None:
        self._prices = stock_prices.sort_index()
        self._volumes = stock_volumes.sort_index() if stock_volumes is not None else None
        self._formation_months = formation_months
        self._skip_month = skip_month
        self._top_quantile = top_quantile
        self._bottom_quantile = bottom_quantile
        self._min_price = min_price
        self._min_volume = min_volume
        self._min_history_days = min_history_days

        # Simple returns for the portfolio
        self._simple_returns = self._prices.pct_change().dropna(how="all")
        self._dates = self._prices.index

        # Rebalance on last trading day of each month
        self._rebalance_dates = self._dates[_last_trading_day_mask(self._dates)]

        _log.info(
            "equity_momentum_init: tickers=%d, dates=%d, rebal_dates=%d",
            len(self._prices.columns),
            len(self._dates),
            len(self._rebalance_dates),
        )

    def rebalance_dates(self) -> pd.DatetimeIndex:
        return self._rebalance_dates

    def compute_target_weights(
        self,
        as_of: pd.Timestamp,
        training_start: pd.Timestamp,
    ) -> dict[str, float]:
        """Compute momentum-sorted portfolio weights as of ``as_of``.

        Returns dict[ticker, weight] with sum(weights) == 0 (dollar-neutral).
        """
        if as_of not in self._dates:
            as_of = self._dates[self._dates <= as_of][-1]

        # Formation window: t-12 to t-1 (skip most recent month)
        dates_before = self._dates[self._dates < as_of]
        if len(dates_before) < 22 * self._skip_month:
            return {}

        # Skip month boundary
        skip_end = dates_before[-1]
        skip_start_idx = max(0, len(dates_before) - 22 * self._skip_month)
        formation_end = dates_before[skip_start_idx - 1] if skip_start_idx > 0 else dates_before[0]

        # Formation start: 12 months before as_of
        formation_start = as_of - pd.DateOffset(months=self._formation_months)

        # Get formation window prices
        mask = (self._dates >= formation_start) & (self._dates <= formation_end)
        window_prices = self._prices.loc[mask]

        if len(window_prices) < self._min_history_days:
            _log.debug("insufficient_history: %d < %d", len(window_prices), self._min_history_days)
            return {}

        # Compute momentum: return over formation window
        first_valid = window_prices.iloc[0]
        last_valid = window_prices.iloc[-1]
        momentum = (last_valid / first_valid - 1.0).dropna()

        # Apply filters
        eligible = momentum.index.tolist()

        # Price filter: current price >= min_price
        current_prices = self._prices.loc[:as_of].iloc[-1]
        eligible = [t for t in eligible if current_prices.get(t, 0) >= self._min_price]

        # Volume filter
        if self._volumes is not None:
            vol_window = self._volumes.loc[:as_of].tail(20)
            avg_vol = vol_window.mean()
            eligible = [t for t in eligible if avg_vol.get(t, 0) >= self._min_volume]

        # History filter: must have enough data in formation window
        eligible = [t for t in eligible if not window_prices[t].isna().any() and len(window_prices[t].dropna()) >= self._min_history_days]

        if len(eligible) < 20:
            _log.debug("insufficient_eligible: %d < 20", len(eligible))
            return {}

        # Rank by momentum
        mom_eligible = momentum[eligible].sort_values(ascending=False)
        n = len(mom_eligible)

        # Top and bottom deciles
        n_top = max(1, int(n * self._top_quantile))
        n_bot = max(1, int(n * self._bottom_quantile))

        winners = mom_eligible.index[:n_top].tolist()
        losers = mom_eligible.index[-n_bot:].tolist()

        # Dollar-neutral equal weight
        weights: dict[str, float] = {}
        for t in winners:
            weights[t] = +0.5 / len(winners)
        for t in losers:
            weights[t] = -0.5 / len(losers)

        return weights

    def run_returns_backtest(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        training_start: pd.Timestamp,
        initial_capital: float = 1_000_000.0,
        spread_bps: float = 5.0,
        impact_bps: float = 10.0,
        commission_per_share: float = 0.005,
    ) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, list[int]]]:
        """Run a returns-based backtest over [start, end].

        Returns (equity, daily_returns, turnover_series, layer_logs).
        """
        mask = (self._dates >= start) & (self._dates <= end)
        test_dates = self._dates[mask]
        rebal_dates = self._rebalance_dates.intersection(test_dates)

        tickers = self._prices.columns.tolist()
        weights = pd.Series(0.0, index=tickers)
        equity_values: list[float] = [initial_capital]
        daily_rets_list: list[float] = [0.0]
        turnover_list: list[float] = [0.0]
        layer_logs: dict[str, list[int]] = {}

        for date in test_dates:
            # Daily portfolio return
            today_rets = self._simple_returns.loc[date] if date in self._simple_returns.index else pd.Series(0.0, index=tickers)
            today_rets = today_rets.reindex(tickers).fillna(0.0)
            port_ret = float(np.nansum(weights.values * today_rets.values))
            gross_equity = equity_values[-1] * (1.0 + port_ret)

            if date in rebal_dates:
                # Compute new target weights
                new_weights = self.compute_target_weights(date, training_start)
                new_w = pd.Series(0.0, index=tickers)
                for t, w in new_weights.items():
                    if t in new_w.index:
                        new_w[t] = w

                # Turnover
                delta = (new_w - weights).abs().sum()
                turnover_list.append(float(delta))

                # Transaction costs
                cost_bps = spread_bps / 2.0 + impact_bps
                notional_traded = gross_equity * delta
                spread_cost = notional_traded * (cost_bps / 10_000.0)

                # Commission: ~$100 avg price assumption
                shares = notional_traded / 100.0
                commission = shares * commission_per_share

                total_cost = spread_cost + commission
                after_cost_equity = gross_equity - total_cost

                equity_values.append(after_cost_equity)
                weights = new_w.copy()

                n_long = int((weights > 0).sum())
                n_short = int((weights < 0).sum())
                layer_logs[str(date.date())] = [n_long + n_short, n_long, n_short]
            else:
                equity_values.append(gross_equity)
                turnover_list.append(0.0)

            daily_rets_list.append((equity_values[-1] - equity_values[-2]) / equity_values[-2])

        equity_series = pd.Series(equity_values[1:], index=test_dates)
        daily_series = pd.Series(daily_rets_list[1:], index=test_dates)
        turnover_series = pd.Series(turnover_list[1:], index=test_dates)

        return equity_series, daily_series, turnover_series, layer_logs
