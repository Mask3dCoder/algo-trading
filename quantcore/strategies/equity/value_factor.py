"""Cross-sectional value factor: long cheap, short expensive, inv-vol weighted.

Ranks assets by composite value score from price-derived proxies when
fundamental data is unavailable: earnings yield (inverse PE via return
stability), book value proxy (tangibility via low-return stability), and
cash flow yield (price normalised by rolling realized return consistency).

When fundamental data (earnings, book_value, cash_flow arrays) is provided
in the data dict, uses actual yields instead of proxies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    TradeSignal,
)
from quantcore.strategies.registry import register


@register("equity_value_factor")
class ValueFactorStrategy(Strategy):
    """Cross-sectional value: long cheap, short expensive, inv-vol weighted.

    Config params:
        lookback: window for proxy computation (default 252)
        n_long: number of long positions (default 5)
        n_short: number of short positions (default 5)
        vol_lookback: window for volatility estimation (default 63)
        rebalance_freq: rebalance frequency in trading days (default 21)
        value_weight: weight of value signal (default 0.4)
        yield_weight: weight of earnings yield signal (default 0.3)
        cf_weight: weight of cash flow yield signal (default 0.3)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback = int(config.params.get("lookback", 252))
        self.n_long = int(config.params.get("n_long", 5))
        self.n_short = int(config.params.get("n_short", 5))
        self.vol_lookback = int(config.params.get("vol_lookback", 63))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.value_weight = float(config.params.get("value_weight", 0.4))
        self.yield_weight = float(config.params.get("yield_weight", 0.3))
        self.cf_weight = float(config.params.get("cf_weight", 0.3))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None

    def _earnings_yield_proxy(self, close: NDArray) -> NDArray:
        """Earnings yield proxy: inverse of normalised price volatility.

        Low price volatility relative to returns → stable earnings → high yield.
        Computed as (roll_mean_ret / roll_std_price) for each asset.
        """
        n_periods, n_assets = close.shape
        yield_score = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 20:
                continue
            rets = np.diff(valid) / valid[:-1]
            mean_ret = np.mean(rets)
            std_price = np.std(valid) / np.mean(valid) if np.mean(valid) > 0 else 1e10
            yield_score[a] = mean_ret / max(std_price, 1e-10)

        return yield_score

    def _book_value_proxy(self, close: NDArray) -> NDArray:
        """Book value proxy: rolling max drawup from recent lows.

        Assets trading closer to their recent lows are "cheaper" on a
        price-to-tangible-value basis. Proxied by -max_deviation_from_mean.
        """
        n_periods, n_assets = close.shape
        bv_score = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 20:
                continue
            mean_p = np.mean(valid)
            # How far below the mean is current price? Higher = cheaper
            bv_score[a] = -(valid[-1] - mean_p) / mean_p if mean_p > 0 else 0.0

        return bv_score

    def _cf_yield_proxy(self, close: NDArray) -> NDArray:
        """Cash flow yield proxy: realised return consistency / price level.

        Assets with high, consistent returns relative to price have high
        cash-flow-like yields. Proxied by (sharpe_ratio * recent_return).
        """
        n_periods, n_assets = close.shape
        cf_score = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 20:
                continue
            rets = np.diff(valid) / valid[:-1]
            mean_r = np.mean(rets)
            std_r = np.std(rets, ddof=1)
            if std_r > 1e-10:
                cf_score[a] = mean_r / std_r  # Sharpe-like
            else:
                cf_score[a] = 0.0

        return cf_score

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        signals: list[TradeSignal] = []

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq and self._current_weights is not None:
            return signals

        if n_periods < self.lookback:
            return signals

        # Use fundamental data when available, otherwise use proxies
        earnings = data.get("earnings")
        book_value = data.get("book_value")
        cash_flow = data.get("cash_flow")

        if earnings is not None and earnings.ndim == 2 and earnings.shape[0] == n_periods:
            yield_score = earnings[-1] / np.maximum(close[-1], 1e-10)
        else:
            yield_score = self._earnings_yield_proxy(close)

        if book_value is not None and book_value.ndim == 2 and book_value.shape[0] == n_periods:
            bv_score = book_value[-1] / np.maximum(close[-1], 1e-10)
        else:
            bv_score = self._book_value_proxy(close)

        if cash_flow is not None and cash_flow.ndim == 2 and cash_flow.shape[0] == n_periods:
            cf_score = cash_flow[-1] / np.maximum(close[-1], 1e-10)
        else:
            cf_score = self._cf_yield_proxy(close)

        # Composite value score
        composite = (
            self.value_weight * _zscore(bv_score)
            + self.yield_weight * _zscore(yield_score)
            + self.cf_weight * _zscore(cf_score)
        )

        # Volatility for inverse-vol weighting
        vol_start = max(0, n_periods - self.vol_lookback)
        daily_rets = np.zeros((n_periods - vol_start, n_assets))
        for i in range(vol_start + 1, n_periods):
            daily_rets[i - vol_start] = close[i] / close[i - 1] - 1.0
        vols = np.std(daily_rets, axis=0, ddof=1)
        vols[vols < 1e-10] = 1e-10

        # Rank by composite value score
        ranks = np.argsort(np.argsort(composite))
        n_long_actual = min(self.n_long, n_assets)
        n_short_actual = min(self.n_short, n_assets - n_long_actual)
        long_mask = ranks >= (n_assets - n_long_actual)
        short_mask = ranks < n_short_actual

        # Inverse-vol weights
        inv_vol = 1.0 / vols
        inv_vol[~long_mask & ~short_mask] = 0.0
        total_inv_vol = np.sum(inv_vol[long_mask]) + np.sum(inv_vol[short_mask])
        if total_inv_vol < 1e-10:
            return signals

        weights = np.zeros(n_assets)
        if np.any(long_mask):
            weights[long_mask] = inv_vol[long_mask] / np.sum(inv_vol[long_mask]) / 2.0
        if np.any(short_mask):
            weights[short_mask] = -inv_vol[short_mask] / np.sum(inv_vol[short_mask]) / 2.0

        self._current_weights = weights
        self._days_since_rebalance = 0

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])
        for i in range(n_assets):
            if abs(weights[i]) > 1e-8:
                direction = 1 if weights[i] > 0 else -1
                signals.append(TradeSignal(
                    symbol=symbols[i],
                    direction=direction,
                    strength=min(abs(weights[i]) * 2, 1.0),
                    size=abs(weights[i]) * float(self.config.max_position_notional),
                    metadata={"value_rank": int(ranks[i]), "composite": float(composite[i])},
                ))

        return signals


def _zscore(x: NDArray) -> NDArray:
    s = np.std(x, ddof=1)
    if s < 1e-10:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s
