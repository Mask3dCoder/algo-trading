"""Walk-forward backtester for institutional-grade alpha validation.

The ``WalkForwardBacktester`` honours the principle that *out-of-sample
data is never touched by the training process*.  It generates a
collection of anchored folds (expanding training, fixed test window),
running the strategy end-to-end on each, and concatenating only the
strictly OOS returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantcore.research.bootstrap import bootstrap_confidence_interval
from quantcore.research.dsr_published import deflated_sharpe_ratio_published

_log = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Result of a single walk-forward fold."""
    fold_idx: int
    train_start: str
    train_end: str
    oos_start: str
    oos_end: str
    oos_returns: NDArray[np.floating]  # daily simple returns
    oos_equity: NDArray[np.floating]


class WalkForwardBacktester:
    """Orchestrates an anchored walk-forward backtest.

    Parameters
    ----------
    strategy : ResidualMomentumStrategy
        Instance already seeded with full data.  The backtester only uses
        its ``run_returns_backtest`` method.
    train_years : int
        Initial training window length in years (default: 10).
    test_years : int
        OOS window length in years (default: 1).
    """

    def __init__(
        self,
        strategy: Any,
        *,
        train_years: int = 10,
        test_years: int = 1,
    ) -> None:
        self.strategy = strategy
        self.train_years = train_years
        self.test_years = test_years

    def generate_folds(self, dates: pd.DatetimeIndex) -> list[dict[str, Any]]:  # type: ignore[type-arg]
        """Generate anchored walk-forward folds.

        Returns a list of dicts with keys ``train_start``, ``train_end``,
        ``oos_start``, ``oos_end``.
        """
        start_date = dates[0]
        end_date = dates[-1]
        folds: list[dict[str, Any]] = []

        # Expand the test window year by year
        oos_start = start_date + pd.DateOffset(years=self.train_years)  # type: ignore[operator]
        while oos_start <= end_date:
            oos_end = min(oos_start + pd.DateOffset(years=self.test_years) - pd.Timedelta(days=1), end_date)  # type: ignore[operator]
            train_end = oos_start - pd.Timedelta(days=1)

            # Clamp to available dates
            _train_end = min(train_end, dates[dates <= train_end][-1] if any(dates <= train_end) else dates[0])
            _oos_end = min(oos_end, dates[dates <= oos_end][-1] if any(dates <= oos_end) else dates[-1])

            folds.append({
                "train_start": start_date,
                "train_end": _train_end,
                "oos_start": oos_start,
                "oos_end": _oos_end,
            })
            oos_start += pd.DateOffset(years=1)  # type: ignore[operator]

        return folds

    def run(
        self,
        *,
        dates: pd.DatetimeIndex,  # type: ignore[type-arg]
        n_trials: int = 10,
        cost_params: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run the walk-forward backtest.

        Returns a dict with keys:
        - ``catalogues_returns``: concatenated OOS returns
        - ``catalogues_equity``: concatenated OOS equity
        - ``fold_results``: list of :class:`WalkForwardResult`
        - ``annualised_sharpe``: OOS annualised Sharpe ratio
        - ``sharpe_ci_90_lower``: 90% CI lower bound from block bootstrap
        - ``deflated_sharpe_pvalue``: p-value from DSR test
        - ``deflated_sharpe_stat``: DSR statistic
        - ``max_drawdown``: maximum drawdown fraction
        - ``avg_turnover``: average monthly turnover
        """
        folds = self.generate_folds(dates)
        _log.info("generated_folds: n_folds=%d", len(folds))

        fold_results: list[WalkForwardResult] = []
        all_oos_returns: list[NDArray[np.floating]] = []

        cparams = cost_params or {}

        for i, fold_def in enumerate(folds, start=1):
            _log.info(
                "running_fold: fold=%d, train_end=%s, oos_start=%s, oos_end=%s",
                i,
                fold_def["train_end"],
                fold_def["oos_start"],
                fold_def["oos_end"],
            )

            # Run the returns-based backtest for this fold
            equity, daily_rets, turnover, _ = self.strategy.run_returns_backtest(
                start=fold_def["oos_start"],
                end=fold_def["oos_end"],
                training_start=fold_def["train_start"],
                **cparams,
            )

            if len(daily_rets) == 0 or daily_rets.isna().all():
                _log.warning("fold_empty_returns: fold=%d", i)
                continue

            result = WalkForwardResult(
                fold_idx=i,
                train_start=str(fold_def["train_start"].date()),
                train_end=str(fold_def["train_end"].date()),
                oos_start=str(fold_def["oos_start"].date()),
                oos_end=str(fold_def["oos_end"].date()),
                oos_returns=daily_rets.values,
                oos_equity=equity.values if len(equity) > 0 else np.array([]),
            )
            fold_results.append(result)
            all_oos_returns.append(daily_rets.values)

            _log.info(
                "fold_complete: fold=%d, oos_days=%d, avg_turnover=%.4f",
                i,
                len(daily_rets),
                float(turnover.mean()) if len(turnover) else 0.0,
            )

        # Concatenate all OOS returns (chronological, across folds)
        if not all_oos_returns:
            return {
                "catalogues_returns": np.array([]),
                "catalogues_equity": np.array([]),
                "fold_results": fold_results,
                "annualised_sharpe": float("nan"),
                "sharpe_ci_90_lower": float("nan"),
                "deflated_sharpe_pvalue": 1.0,
                "deflated_sharpe_stat": 0.0,
                "max_drawdown": 0.0,
                "avg_turnover": 0.0,
            }

        catalogues_returns = np.concatenate(all_oos_returns)
        catalogues_equity = np.exp(np.cumsum(np.log(1.0 + catalogues_returns)))

        # -- metrics on the full OOS series ----------------------------------
        mean_ret = float(np.mean(catalogues_returns))
        std_ret = float(np.std(catalogues_returns, ddof=1))
        annual_sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-12 else 0.0

        # Block bootstrap 90% CI on Sharpe
        ci_lower, _ = bootstrap_confidence_interval(
            catalogues_returns, block_size=max(5, int(len(catalogues_returns) ** (1 / 3))), n_boot=2000, ci=0.90
        )

        # DSR — published Bailey & Lopez de Prado (2014) form.
        # Contract: per-observation Sharpe + RAW kurtosis (3.0 = normal).
        # The annualized Sharpe (mean/std * sqrt(252)) must NOT be fed here;
        # see quantcore/research/dsr_published.py docstring + docs/dsr_audit.md.
        per_obs_sharpe = mean_ret / std_ret if std_ret > 1e-12 else 0.0
        skew = float(np.mean((catalogues_returns - mean_ret) ** 3) / (std_ret ** 3)) if std_ret > 1e-12 else 0.0
        kurt_raw = float(np.mean((catalogues_returns - mean_ret) ** 4) / (std_ret ** 4)) if std_ret > 1e-12 else 3.0
        dski = deflated_sharpe_ratio_published(
            observed_sharpe=per_obs_sharpe,
            num_trials=n_trials,
            sample_length=len(catalogues_returns),
            skewness=skew,
            kurtosis=kurt_raw,
        )

        # Max drawdown on OOS equity
        peak = np.maximum.accumulate(catalogues_equity)
        dd = (catalogues_equity - peak) / np.where(peak > 0, peak, 1.0)
        max_dd = float(np.abs(dd.min()))

        return {
            "catalogues_returns": catalogues_returns,
            "catalogues_equity": catalogues_equity,
            "fold_results": fold_results,
            "annualised_sharpe": float(annual_sharpe),
            "sharpe_ci_90_lower": float(ci_lower),
            "deflated_sharpe_pvalue": float(dski[0]),
            "deflated_sharpe_stat": float(dski[1]),
            "max_drawdown": float(max_dd),
        }
