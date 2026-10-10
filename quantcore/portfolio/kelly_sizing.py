"""Fractional Kelly position sizing for multi-strategy portfolios.

Replaces ad-hoc risk-budget sizing with optimal Kelly criterion:
  f* = p - q / b
where:
  p = win rate
  q = 1 - p
  b = avg_win / |avg_loss|

Fractional Kelly (0.25x to 0.5x) accounts for estimation error and
provides more conservative sizing than full Kelly.

The module computes Kelly fractions from OOS backtest data and provides
a sizing function that can replace the current risk-budget approach.

Usage:
    from quantcore.portfolio.kelly_sizing import KellySizer

    sizer = KellySizer(fraction=0.25)
    sizer.fit_from_returns(strategy_returns)
    size = sizer.position_size(equity=1_000_000, edge_scalar=1.0)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Kelly computation
# ---------------------------------------------------------------------------


@dataclass
class KellyStats:
    """Kelly criterion statistics for a strategy."""
    win_rate: float          # p: probability of winning trade
    avg_win: float           # average winning return (positive)
    avg_loss: float          # average losing return (positive, absolute)
    payoff_ratio: float      # b = avg_win / avg_loss
    kelly_fraction: float    # f* = (bp - q) / b
    fractional_kelly: float  # fraction * f*
    n_trades: int
    edge: float              # p * avg_win - q * avg_loss (expected value)
    kelly_used: float        # fractional_kelly with edge_scalar applied


def compute_kelly_from_trades(
    trades: list[dict],
    fraction: float = 0.25,
    min_trades: int = 20,
) -> KellyStats:
    """Compute Kelly fraction from a list of trade records.

    Parameters
    ----------
    trades : list[dict]
        Trade records with 'pnl_abs' or 'pnl_pct' key.
    fraction : float
        Fractional Kelly multiplier (0.25 = quarter Kelly).
    min_trades : int
        Minimum number of trades for reliable estimation.

    Returns
    -------
    KellyStats
        Kelly criterion statistics.
    """
    # Extract P&L values
    pnls = []
    for t in trades:
        pnl = t.get("pnl_pct", t.get("pnl_abs"))
        if pnl is not None and np.isfinite(pnl):
            pnls.append(float(pnl))

    if len(pnls) < min_trades:
        _log.warning("Only %d trades (need %d). Using conservative defaults.", len(pnls), min_trades)
        return KellyStats(
            win_rate=0.5, avg_win=0.01, avg_loss=0.01,
            payoff_ratio=1.0, kelly_fraction=0.0,
            fractional_kelly=0.0, n_trades=len(pnls),
            edge=0.0, kelly_used=0.0,
        )

    pnls_arr = np.array(pnls)
    winners = pnls_arr[pnls_arr > 0]
    losers = pnls_arr[pnls_arr <= 0]

    p = len(winners) / len(pnls_arr)  # win rate
    q = 1.0 - p

    avg_win = float(np.mean(winners)) if len(winners) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losers))) if len(losers) > 0 else 1e-10

    b = avg_win / max(avg_loss, 1e-10)  # payoff ratio

    # Kelly fraction: f* = (bp - q) / b
    if b > 0:
        f_star = (b * p - q) / b
    else:
        f_star = 0.0

    f_star = max(0.0, min(f_star, 1.0))

    # Edge: expected value per trade
    edge = p * avg_win - q * avg_loss

    fractional = f_star * fraction

    return KellyStats(
        win_rate=p,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=b,
        kelly_fraction=f_star,
        fractional_kelly=fractional,
        n_trades=len(pnls),
        edge=edge,
        kelly_used=fractional,
    )


def compute_kelly_from_returns(
    returns: NDArray,
    fraction: float = 0.25,
    min_observations: int = 60,
) -> KellyStats:
    """Compute Kelly fraction from a return series (no explicit trades).

    Treats each bar as a "trade" and computes win rate, avg win/loss
    from the return distribution.

    Parameters
    ----------
    returns : NDArray
        Daily return series.
    fraction : float
        Fractional Kelly multiplier.
    min_observations : int
        Minimum observations for reliable estimation.

    Returns
    -------
    KellyStats
        Kelly criterion statistics.
    """
    r = np.asarray(returns, dtype=float).flatten()
    r = r[np.isfinite(r)]

    if len(r) < min_observations:
        _log.warning("Only %d observations (need %d).", len(r), min_observations)
        return KellyStats(
            win_rate=0.5, avg_win=0.001, avg_loss=0.001,
            payoff_ratio=1.0, kelly_fraction=0.0,
            fractional_kelly=0.0, n_trades=len(r),
            edge=0.0, kelly_used=0.0,
        )

    winners = r[r > 0]
    losers = r[r <= 0]

    p = len(winners) / len(r)
    q = 1.0 - p

    avg_win = float(np.mean(winners)) if len(winners) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losers))) if len(losers) > 0 else 1e-10

    b = avg_win / max(avg_loss, 1e-10)

    if b > 0:
        f_star = (b * p - q) / b
    else:
        f_star = 0.0

    f_star = max(0.0, min(f_star, 1.0))
    edge = p * avg_win - q * avg_loss
    fractional = f_star * fraction

    return KellyStats(
        win_rate=p,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff_ratio=b,
        kelly_fraction=f_star,
        fractional_kelly=fractional,
        n_trades=len(r),
        edge=edge,
        kelly_used=fractional,
    )


# ---------------------------------------------------------------------------
# Kelly Sizer (multi-strategy)
# ---------------------------------------------------------------------------


class KellySizer:
    """Fractional Kelly position sizer for multi-strategy portfolios.

    Computes Kelly fractions for each strategy from OOS backtest data,
    then provides position sizes that replace the ad-hoc risk-budget
    approach.

    Parameters
    ----------
    fraction : float
        Fractional Kelly multiplier (default 0.25 = quarter Kelly).
        Lower = more conservative. Typical range: 0.25-0.50.
    edge_scalar : float
        Additional scaling factor applied after Kelly (default 1.0).
        Can be used to further reduce size during uncertain periods.
    max_kelly : float
        Maximum Kelly fraction allowed (default 0.20 = 20% of equity).
    """

    def __init__(
        self,
        fraction: float = 0.25,
        edge_scalar: float = 1.0,
        max_kelly: float = 0.20,
    ) -> None:
        self._fraction = fraction
        self._edge_scalar = edge_scalar
        self._max_kelly = max_kelly
        self._strategy_stats: dict[str, KellyStats] = {}
        self._fitted = False

    def fit_from_trades(
        self,
        strategy_id: str,
        trades: list[dict],
    ) -> KellyStats:
        """Fit Kelly stats for a strategy from trade records.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier.
        trades : list[dict]
            Trade records with 'pnl_abs' or 'pnl_pct' key.

        Returns
        -------
        KellyStats
            Computed Kelly statistics.
        """
        stats = compute_kelly_from_trades(trades, fraction=self._fraction)
        self._strategy_stats[strategy_id] = stats
        self._fitted = True

        _log.info(
            "Kelly fit %s: p=%.2f, b=%.2f, f*=%.3f, frac=%.3f, edge=%.4f",
            strategy_id, stats.win_rate, stats.payoff_ratio,
            stats.kelly_fraction, stats.fractional_kelly, stats.edge,
        )

        return stats

    def fit_from_returns(
        self,
        strategy_id: str,
        returns: NDArray,
    ) -> KellyStats:
        """Fit Kelly stats for a strategy from return series.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier.
        returns : NDArray
            Daily return series.

        Returns
        -------
        KellyStats
            Computed Kelly statistics.
        """
        stats = compute_kelly_from_returns(returns, fraction=self._fraction)
        self._strategy_stats[strategy_id] = stats
        self._fitted = True

        _log.info(
            "Kelly fit %s: p=%.2f, b=%.2f, f*=%.3f, frac=%.3f, edge=%.4f",
            strategy_id, stats.win_rate, stats.payoff_ratio,
            stats.kelly_fraction, stats.fractional_kelly, stats.edge,
        )

        return stats

    def position_size(
        self,
        strategy_id: str,
        equity: float,
        edge_scalar: float | None = None,
    ) -> float:
        """Compute position size for a strategy.

        Parameters
        ----------
        strategy_id : str
            Strategy identifier.
        equity : float
            Current equity.
        edge_scalar : float | None
            Override the default edge scalar.

        Returns
        -------
        float
            Position size in dollars.
        """
        if strategy_id not in self._strategy_stats:
            return equity * 0.02  # 2% default if not fitted

        stats = self._strategy_stats[strategy_id]
        scalar = edge_scalar if edge_scalar is not None else self._edge_scalar

        # Kelly fraction with edge scalar
        kelly = min(stats.fractional_kelly * scalar, self._max_kelly)

        return equity * kelly

    def portfolio_weights(
        self,
        equity: float,
        strategy_ids: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute Kelly-optimal weights for all strategies.

        Parameters
        ----------
        equity : float
            Current equity.
        strategy_ids : list[str] | None
            Specific strategies to include. If None, uses all fitted.

        Returns
        -------
        dict[str, float]
            Mapping from strategy_id to position size in dollars.
        """
        ids = strategy_ids or list(self._strategy_stats.keys())
        weights = {}

        for sid in ids:
            weights[sid] = self.position_size(sid, equity)

        # Normalize so total doesn't exceed equity
        total = sum(weights.values())
        if total > equity:
            scale = equity / total
            weights = {k: v * scale for k, v in weights.items()}

        return weights

    def get_stats(self, strategy_id: str) -> KellyStats | None:
        """Get Kelly stats for a strategy."""
        return self._strategy_stats.get(strategy_id)

    def get_all_stats(self) -> dict[str, KellyStats]:
        """Get Kelly stats for all fitted strategies."""
        return dict(self._strategy_stats)

    @property
    def fraction(self) -> float:
        return self._fraction

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# Rolling Kelly (adaptive)
# ---------------------------------------------------------------------------


class RollingKellySizer:
    """Rolling Kelly sizer that adapts to changing market conditions.

    Instead of using a fixed Kelly fraction from the full backtest,
    this computes Kelly on a rolling window to adapt to recent
    strategy performance.

    Parameters
    ----------
    lookback : int
        Rolling window size in bars (default 252 = ~1 year).
    fraction : float
        Fractional Kelly multiplier.
    min_observations : int
        Minimum observations before Kelly is computed.
    """

    def __init__(
        self,
        lookback: int = 252,
        fraction: float = 0.25,
        min_observations: int = 60,
    ) -> None:
        self._lookback = lookback
        self._fraction = fraction
        self._min_obs = min_observations
        self._return_history: dict[str, list[float]] = {}

    def update(self, strategy_id: str, daily_return: float) -> None:
        """Add a daily return observation for a strategy."""
        if strategy_id not in self._return_history:
            self._return_history[strategy_id] = []
        self._return_history[strategy_id].append(daily_return)

        # Trim to lookback
        if len(self._return_history[strategy_id]) > self._lookback * 2:
            self._return_history[strategy_id] = self._return_history[strategy_id][-self._lookback:]

    def current_kelly(self, strategy_id: str) -> KellyStats:
        """Compute current Kelly fraction from rolling window."""
        if strategy_id not in self._return_history:
            return KellyStats(
                win_rate=0.5, avg_win=0.001, avg_loss=0.001,
                payoff_ratio=1.0, kelly_fraction=0.0,
                fractional_kelly=0.0, n_trades=0,
                edge=0.0, kelly_used=0.0,
            )

        recent = self._return_history[strategy_id][-self._lookback:]
        returns = np.array(recent)

        return compute_kelly_from_returns(
            returns,
            fraction=self._fraction,
            min_observations=self._min_obs,
        )

    def position_size(
        self,
        strategy_id: str,
        equity: float,
    ) -> float:
        """Compute current Kelly-optimal position size."""
        stats = self.current_kelly(strategy_id)
        kelly = min(stats.fractional_kelly, 0.20)
        return equity * kelly
