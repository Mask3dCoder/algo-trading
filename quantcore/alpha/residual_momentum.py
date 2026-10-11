"""Factor-neutral residual momentum — production alpha module.

This module implements a cross-sectional residual momentum strategy that
estimates stock-specific factor exposures via expanding-window OLS, and
forms a dollar-neutral long/short portfolio based on cumulative
idiosyncratic returns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def _last_trading_day_mask(dates: pd.DatetimeIndex) -> pd.Series:  # type: ignore[type-arg]
    """Boolean mask: True on the last trading day of each month."""
    months = dates.to_period("M")
    is_last = months.shift(-1) != months  # type: ignore[no-untyped-call]
    return pd.Series(is_last, index=dates)


class ResidualMomentumStrategy:
    """Factor-neutral residual momentum strategy.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        Daily close prices. Index = timestamps, columns = tickers.
    factor_returns : pd.DataFrame
        Daily Fama-French 5-factor returns. Index = timestamps,
        columns = ``[Mkt-RF, SMB, HML, RMW, CMA]``.
    lookback_months : int
        Number of months over which to accumulate idiosyncratic returns
        for the ranking signal (default: 1).
    min_obs : int
        Minimum number of trading days required before the first OLS
        coefficient set is accepted (default: 252).
    max_weight : float
        Maximum |weight| for any single stock (default: 0.05).
    min_tickers_for_signal : int
        Minimum number of tickers with valid idiosyncratic scores required
        to produce a non-empty signal.  Must be at least 2 * floor(N/5)
        for a meaningful quintile portfolio (default: 6).
    """

    def __init__(
        self,
        stock_prices: pd.DataFrame,
        factor_returns: pd.DataFrame,
        *,
        lookback_months: int = 1,
        min_obs: int = 252,
        max_weight: float = 0.05,
        min_tickers_for_signal: int = 6,
    ) -> None:
        self._prices = stock_prices.sort_index()
        self._factors = factor_returns.reindex(stock_prices.index).sort_index()
        self._lookback_months = lookback_months
        self._min_obs = min_obs
        self._max_weight = max_weight
        self._min_tickers_for_signal = min_tickers_for_signal

        # -- derived quantities ------------------------------------------------
        # Log returns, aligned with factor dates
        common = self._prices.index.intersection(self._factors.index)
        self._prices = self._prices.loc[common]
        self._factors = self._factors.loc[common]
        self._log_returns = np.log(self._prices / self._prices.shift(1)).dropna(how="all")
        self._dates = self._log_returns.index
        self._factors = self._factors.loc[self._dates]

        # -- rebalance schedule (last trading day of month) -------------------
        self._rebalance_dates: pd.DatetimeIndex = self._dates[_last_trading_day_mask(self._dates)]  # type: ignore[assignment]
        _log.info("strategy_initialised: tickers=%d, dates=%d", len(self._prices.columns), len(self._dates))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def rebalance_dates(self) -> pd.DatetimeIndex:  # type: ignore[type-arg]
        """Return all rebalance dates defined by the strategy."""
        return self._rebalance_dates

    def compute_target_weights(
        self,
        as_of: pd.Timestamp,  # type: ignore[type-arg]
        training_start: pd.Timestamp,  # type: ignore[type-arg]
        training_end: pd.Timestamp | None = None,  # type: ignore[type-arg]
    ) -> dict[str, float]:
        """Compute target portfolio weights as of ``as_of``.

        The OLS betas are estimated on data from ``training_start`` up to (and
        including) ``as_of``.  ``training_end`` optionally caps that estimation
        window, which is how a purged walk-forward keeps training from reaching
        across the fold boundary: a beta estimated on data that overlaps the OOS
        window is fitted on the answer. When ``training_end`` is None the window
        is unbounded above, exactly as before.

        The cumulative idiosyncratic return is summed over the past
        ``lookback_months`` months from ``as_of``.

        Returns
        -------
        dict
            Mapping ``ticker -> weight`` with ``sum(weights) == 0``
            (dollar-neutral). Top and bottom quintiles are equal-weighted
            within each leg.
        """
        if as_of not in self._dates:
            as_of = self._dates[self._dates <= as_of][-1]

        # -- expanding window up to `as_of`, optionally capped --------------
        mask = (self._dates >= training_start) & (self._dates <= as_of)
        if training_end is not None:
            mask &= self._dates <= training_end
        window_rets = self._log_returns.loc[mask]
        window_factors = self._factors.loc[mask]

        tickers = [c for c in window_rets.columns if not window_rets[c].isna().all()]

        idiosyncratic_momentum: dict[str, float] = {}
        factor_cols = [c for c in self._factors.columns if c != "RF"]

        for ticker in tickers:
            try:
                mom = self._compute_idiosyncratic_momentum(
                    ticker, window_rets, window_factors, factor_cols
                )
                if mom is not None:
                    idiosyncratic_momentum[ticker] = mom
            except Exception as exc:
                _log.debug("signal_error: ticker=%s, date=%s, exc=%s", ticker, as_of, exc)

        if len(idiosyncratic_momentum) < 20:
            _log.warning("insufficient_tickers: n=%d, date=%s", len(idiosyncratic_momentum), as_of)
            return {}

        # -- rank, z-score, quintile selection --------------------------------
        series = pd.Series(idiosyncratic_momentum)
        z = (series - series.mean()) / series.std(ddof=0)

        n = len(z)
        q = max(1, n // 5)
        top = z.nlargest(q)
        bot = z.nsmallest(q)

        weights: dict[str, float] = {}

        # Equal weight within each leg; dollar-neutral => long + short = 0
        if len(top):
            for t in top.index:
                weights[t] = +0.5 / len(top)
        if len(bot):
            for t in bot.index:
                weights[t] = -0.5 / len(bot)

        # Enforce per-stock max-weight constraint
        weights = {k: v for k, v in weights.items() if abs(v) > 0}
        weights = self._enforce_max_weight(weights)
        weights = self._enforce_dollar_neutral(weights)

        return weights

    # ------------------------------------------------------------------ #
    # Core backtest entry point (returns-based)
    # ------------------------------------------------------------------ #

    def run_returns_backtest(
        self,
        *,
        start: pd.Timestamp,  # type: ignore[type-arg]
        end: pd.Timestamp,  # type: ignore[type-arg]
        training_start: pd.Timestamp,  # type: ignore[type-arg],
        training_end: pd.Timestamp | None = None,  # type: ignore[type-arg]
        initial_capital: float = 1_000_000.0,
        spread_bps: float = 5.0,
        impact_bps: float = 10.0,
        commission_per_share: float = 0.005,
        trading_days_per_month: int = 21,
    ) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, list[int]]]:  # type: ignore[type-arg]
        """Run a direct returns-based backtest over [start, end].

        Returns
        -------
        equity : pd.Series
            Daily equity curve (index = dates).
        daily_returns : pd.Series
            Daily simple returns.
        turnover_series : pd.Series
            Daily turnover (0 except on rebalance days).
        layer_logs : dict
            ``{date: [n_active, n_long, n_short]}`` per rebalance.
        """
        mask = (self._dates >= start) & (self._dates <= end)
        test_dates = self._dates[mask]
        rebal_dates = self._rebalance_dates.intersection(test_dates)

        n_assets = len(self._prices.columns)
        weights = pd.Series(0.0, index=self._prices.columns)
        equity_values: list[float] = [initial_capital]
        daily_rets_list: list[float] = [0.0]
        turnover_list: list[float] = [0.0]
        layer_logs: dict[str, list[int]] = {}
        leverage = 1.0  # dollar-neutral, ~2x gross (long + short)

        for date in test_dates:
            # Compute daily portfolio return = sum(weight_i * ret_i)
            today_rets = self._log_returns.loc[date]
            port_ret = float(np.nansum(weights.values * today_rets.values))
            # Gross daily return (before costs)
            gross_equity = equity_values[-1] * (1.0 + port_ret)

            # Transaction costs applied on rebalance dates
            if date in rebal_dates:
                # Compute new target weights
                new_weights = self.compute_target_weights(date, training_start, training_end)
                new_w = pd.Series(0.0, index=weights.index)
                new_w.update(new_weights)
                new_w = new_w.reindex(weights.index).fillna(0.0)

                # Turnover = sum(|delta weights|)
                delta = (new_w - weights).abs().sum()
                turnover_list.append(float(delta))

                # Cost = delta * half-spread + delta * impact + commission
                # Simplified: total cost ≈ delta * (spread/2 + impact)  [per $ traded]
                cost_bps = spread_bps / 2.0 + impact_bps
                cost = gross_equity * delta * (cost_bps / 10_000.0)

                # Commission per share: assume avg price ~$100 for simplicity
                # (more accurate would use actual prices, but for a portfolio
                #  backtest the impact is dominated by spread + impact)
                avg_price = 100.0
                notional_traded = gross_equity * delta
                shares = notional_traded / avg_price
                commission = shares * commission_per_share

                total_cost = cost + commission
                after_cost_equity = gross_equity - total_cost

                # Apply: equity after rebalancing
                equity_values.append(after_cost_equity)
                weights = new_w.copy()

                # Log layer sizes
                n_long = (weights > 0).sum()
                n_short = (weights < 0).sum()
                layer_logs[str(date.date())] = [int(n_long + n_short), int(n_long), int(n_short)]
            else:
                equity_values.append(gross_equity)
                turnover_list.append(0.0)

                # Drift: weights are held constant (assume no rebalancing)
                # While prices change, so weights drift. For simplicity,
                # we'll keep nominal weights fixed. This is standard for a
                # monthly-rebalanced portfolio.

            # Compute daily return relative to previous day
            daily_rets_list.append((equity_values[-1] - equity_values[-2]) / equity_values[-2])

        equity_series = pd.Series(equity_values[1:], index=test_dates)
        daily_series = pd.Series(daily_rets_list[1:], index=test_dates)
        turnover_series = pd.Series(turnover_list[1:], index=test_dates)

        return equity_series, daily_series, turnover_series, layer_logs

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_idiosyncratic_momentum(
        self,
        ticker: str,
        window_rets: pd.DataFrame,
        window_factors: pd.DataFrame,
        factor_cols: list[str],
    ) -> float | None:
        """Return cumulative idiosyncratic log return over lookback_months."""
        y = window_rets[ticker].dropna()
        if len(y) < self._min_obs:
            return None

        # Align with factors
        X_raw = window_factors.loc[y.index, factor_cols]
        valid = X_raw.notna().all(axis=1)
        if not valid.sum() >= self._min_obs:
            return None

        y = y[valid].values
        X = X_raw[valid].values

        # OLS with intercept
        X_design = np.column_stack([np.ones(len(X)), X])
        beta_hat, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        # Predicted returns
        y_pred = X_design @ beta_hat

        # Residuals = observed - predicted
        residuals = y - y_pred  # shape (T,)

        # Cumulative idiosyncratic return over lookback period
        start_idx = max(0, len(residuals) - self._lookback_months * 21)
        recent_residuals = residuals[start_idx:]
        return float(recent_residuals.sum())

    @staticmethod
    def _enforce_max_weight(weights: dict[str, float]) -> dict[str, float]:
        max_w = max(abs(w) for w in weights.values()) if weights else 0.0
        if max_w <= 0:
            return weights
        scale = min(1.0, 0.05 / max_w)
        if scale < 1.0:
            return {k: v * scale for k, v in weights.items()}
        return weights

    @staticmethod
    def _enforce_dollar_neutral(weights: dict[str, float]) -> dict[str, float]:
        if not weights:
            return weights
        total = sum(weights.values())
        if abs(total) > 1e-10:
            n = len(weights)
            adj = total / n
            return {k: v - adj for k, v in weights.items()}
        return weights
