"""Atlas Regime-Adaptive Multi-Factor Alpha strategy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, StrategyStatus, TradeSignal
from quantcore.strategies.registry import register


@register("atlas_ramfa")
class AtlasRamfaStrategy(Strategy):
    """Regime-adaptive multi-factor long/short equity alpha strategy."""

    STRATEGY_STATUS: StrategyStatus = StrategyStatus.RESEARCH

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.momentum_weight = float(config.params.get("momentum_weight", 0.20))
        self.tsmom_weight = float(config.params.get("tsmom_weight", 0.15))
        self.quality_weight = float(config.params.get("quality_weight", 0.15))
        self.value_weight = float(config.params.get("value_weight", 0.15))
        self.lowvol_weight = float(config.params.get("lowvol_weight", 0.10))
        self.reversal_weight = float(config.params.get("reversal_weight", 0.10))
        self.liquidity_weight = float(config.params.get("liquidity_weight", 0.05))
        self.vol_carry_weight = float(config.params.get("vol_carry_weight", 0.10))
        self.ic_weighting = bool(config.params.get("ic_weighting", False))
        self.ic_window = int(config.params.get("ic_window", 63))
        self.ic_blend = float(config.params.get("ic_blend", 0.35))
        self.lookback_days = int(config.params.get("lookback_days", 252))
        self.skip_days = int(config.params.get("skip_days", 21))
        self.vol_lookback = int(config.params.get("vol_lookback", 63))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.long_threshold = float(config.params.get("long_threshold", 0.65))
        self.short_threshold = float(config.params.get("short_threshold", 0.65))
        self.n_long = int(config.params.get("n_long", 20))
        self.n_short = int(config.params.get("n_short", 20))
        self.min_liquidity_score = float(config.params.get("min_liquidity_score", 75.0))
        self.max_gross_exposure = float(
            min(config.params.get("max_gross_exposure", 1.5), config.max_leverage, 2.25)
        )
        self.min_net_exposure = float(config.params.get("min_net_exposure", -0.25))
        self.max_net_exposure = float(config.params.get("max_net_exposure", 0.50))
        self.single_name_max_pct = float(config.params.get("single_name_max_pct", 0.02))
        self.stop_loss_mult = float(config.params.get("stop_loss_mult", 2.5))
        self.take_profit_mult = float(config.params.get("take_profit_mult", 2.0))
        self.time_stop_days = int(config.params.get("time_stop_days", 45))
        self.exit_threshold = float(config.params.get("exit_threshold", 0.20))
        self.use_regime_filter = bool(config.params.get("use_regime_filter", True))
        self.min_regime_score = float(config.params.get("min_regime_score", 55.0))
        self.regime_exposure_multiplier = bool(
            config.params.get("regime_exposure_multiplier", True)
        )
        self.max_regime_multiplier = float(config.params.get("max_regime_multiplier", 1.0))
        self.min_regime_multiplier = float(config.params.get("min_regime_multiplier", 0.15))
        self.portfolio_value = float(
            config.params.get("portfolio_value", config.max_position_notional)
        )
        self._days_since_rebalance = 0
        self._current_weights: NDArray[np.float64] | None = None
        self._current_scores: NDArray[np.float64] | None = None
        self._current_vols: NDArray[np.float64] | None = None
        self._current_symbols: list[str] = []
        self._factor_ics: dict[str, float] = {}
        self._entry_prices: dict[str, float] = {}
        self._entry_days: dict[str, int] = {}
        self._entry_scores: dict[str, float] = {}
        self._last_regime_score = 100.0
        self._last_gross_multiplier = 1.0

    def on_data(self, data: dict[str, NDArray[np.float64]]) -> list[TradeSignal]:
        """Process market data and emit rebalance or exit signals."""
        close = data.get("close")
        if close is None:
            return []

        prices = _as_2d(close)
        n_periods, n_assets = prices.shape
        symbols = _symbols(data, n_assets)
        self._days_since_rebalance += 1

        exit_signals = self._exit_signals(symbols, prices, data)
        rebalance_due = (
            self._days_since_rebalance >= self.rebalance_freq
            or self._current_weights is None
        )
        if not rebalance_due:
            return exit_signals

        min_required = self.lookback_days + self.skip_days + self.vol_lookback + 5
        if n_periods < min_required:
            return exit_signals

        returns = _returns(prices)
        if returns.shape[0] < 2:
            return exit_signals

        factors = self._compute_factors(prices, returns, data)
        composite = self._combine_factors(factors)
        vols = _realized_vol(prices, self.vol_lookback)
        liquidity = _liquidity_score(prices, data)
        regime_score = _regime_score(data, prices)
        self._last_regime_score = regime_score
        gross_multiplier = self._regime_multiplier(regime_score)
        self._last_gross_multiplier = gross_multiplier

        if self.use_regime_filter and regime_score < self.min_regime_score:
            self._current_weights = np.zeros(n_assets, dtype=float)
            self._current_scores = np.zeros(n_assets, dtype=float)
            self._current_vols = vols
            self._current_symbols = symbols
            return exit_signals

        eligible = (liquidity >= self.min_liquidity_score) & np.isfinite(composite)
        if int(np.sum(eligible)) < max(4, self.n_long + self.n_short):
            return exit_signals

        composite[~eligible] = np.nan
        composite = _sector_neutralize(composite, data)
        composite = _safe_zscore(_winsorize(composite, 0.025, 0.975))

        if self.ic_weighting:
            weights = self._ic_adjusted_weights(composite, factors, returns)
        else:
            weights = composite

        scores = _safe_zscore(weights)
        raw_weights = self._raw_weights(scores, vols)
        raw_weights[~eligible] = 0.0
        raw_weights = self._apply_tail_selection(raw_weights, scores)
        raw_weights = self._apply_exposure_limits(raw_weights)
        raw_weights *= gross_multiplier
        raw_weights = self._apply_single_name_limit(raw_weights)

        if np.sum(np.abs(raw_weights)) < 1e-10:
            self._current_weights = raw_weights
            self._current_scores = scores
            self._current_vols = vols
            self._current_symbols = symbols
            return exit_signals

        signals = _signals_from_weights(
            weights=raw_weights,
            symbols=symbols,
            prices=prices[-1],
            vols=vols,
            scores=scores,
            max_position_notional=self.portfolio_value,
            single_name_max_pct=self.single_name_max_pct,
            stop_loss_mult=self.stop_loss_mult,
            take_profit_mult=self.take_profit_mult,
            metadata={
                "regime_score": regime_score,
                "gross_multiplier": gross_multiplier,
                "factor_ics": dict(self._factor_ics),
            },
        )

        self._current_weights = raw_weights
        self._current_scores = scores
        self._current_vols = vols
        self._current_symbols = symbols
        self._days_since_rebalance = 0
        return signals + exit_signals

    def on_fill(self, symbol: str, fill_price: float, fill_qty: float, side: str) -> None:
        """Track fills for stop, target, and time-stop logic."""
        super().on_fill(symbol, fill_price, fill_qty, side)
        if fill_qty == 0:
            self._entry_prices.pop(symbol, None)
            self._entry_days.pop(symbol, None)
            self._entry_scores.pop(symbol, None)
            return

        signed_qty = fill_qty if side == "buy" else -fill_qty
        current = self._positions.get(symbol, 0.0)
        if abs(current) < 1e-10:
            self._entry_prices.pop(symbol, None)
            self._entry_days.pop(symbol, None)
            self._entry_scores.pop(symbol, None)
            return

        if np.sign(current) == np.sign(signed_qty):
            self._entry_prices[symbol] = fill_price
            self._entry_days[symbol] = self._days_since_rebalance
            self._entry_scores[symbol] = float(self._score_for_symbol(symbol))

    def _compute_factors(
        self,
        prices: NDArray[np.float64],
        returns: NDArray[np.float64],
        data: dict[str, NDArray[np.float64]],
    ) -> dict[str, NDArray[np.float64]]:
        """Compute all alpha factors from point-in-time data."""
        return {
            "momentum": self._momentum_signal(prices),
            "tsmom": self._tsmom_signal(prices),
            "quality": self._quality_signal(prices, returns),
            "value": self._value_signal(prices, data),
            "lowvol": self._lowvol_signal(prices, returns),
            "reversal": self._reversal_signal(prices),
            "liquidity": self._liquidity_factor(prices, data),
            "vol_carry": self._vol_carry_signal(prices, returns, data),
        }

    def _momentum_signal(self, prices: NDArray[np.float64]) -> NDArray[np.float64]:
        """12-1 cross-sectional momentum."""
        n_periods, n_assets = prices.shape
        score = np.zeros(n_assets, dtype=float)
        end_idx = max(0, n_periods - self.skip_days)
        start_idx = max(0, end_idx - self.lookback_days)
        for asset in range(n_assets):
            valid = prices[start_idx:end_idx, asset]
            valid = valid[np.isfinite(valid) & (valid > 0)]
            if len(valid) >= 20:
                score[asset] = valid[-1] / valid[0] - 1.0
        return _safe_zscore(_sector_neutralize(score, {}))

    def _tsmom_signal(self, prices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Time-series momentum score."""
        n_periods, n_assets = prices.shape
        score = np.zeros(n_assets, dtype=float)
        end_idx = max(0, n_periods - self.skip_days)
        start_idx = max(0, end_idx - self.lookback_days)
        for asset in range(n_assets):
            valid = prices[start_idx:end_idx, asset]
            valid = valid[np.isfinite(valid) & (valid > 0)]
            if len(valid) >= 20:
                score[asset] = valid[-1] / valid[0] - 1.0
        return _safe_zscore(score)

    def _quality_signal(
        self,
        prices: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Quality: profitability, stability, and accruals proxy."""
        n_assets = prices.shape[1]
        hit_rate = np.zeros(n_assets, dtype=float)
        stability = np.zeros(n_assets, dtype=float)
        accrual_proxy = np.zeros(n_assets, dtype=float)

        for asset in range(n_assets):
            rets = returns[:, asset]
            rets = rets[np.isfinite(rets)]
            if len(rets) < 30:
                continue
            hit_rate[asset] = np.mean(rets > 0.0)
            mean_ret = np.mean(rets)
            std_ret = np.std(rets, ddof=1)
            stability[asset] = -std_ret / max(abs(mean_ret), 1e-10) if std_ret > 1e-10 else 0.0
            if np.std(rets[:-1], ddof=1) > 1e-10 and np.std(rets[1:], ddof=1) > 1e-10:
                corr = np.corrcoef(rets[:-1], rets[1:])[0, 1]
                accrual_proxy[asset] = 0.0 if np.isnan(corr) else corr

        quality = (
            0.40 * _safe_zscore(hit_rate)
            + 0.35 * _safe_zscore(stability)
            + 0.25 * _safe_zscore(accrual_proxy)
        )
        return _safe_zscore(quality)

    def _value_signal(
        self,
        prices: NDArray[np.float64],
        data: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Value: optional fundamentals, otherwise price-derived cheapness."""
        n_periods, n_assets = prices.shape
        earnings = _optional_2d(data.get("earnings"), n_periods, n_assets)
        book_value = _optional_2d(data.get("book_value"), n_periods, n_assets)
        cash_flow = _optional_2d(data.get("cash_flow"), n_periods, n_assets)
        enterprise_value = _optional_2d(data.get("enterprise_value"), n_periods, n_assets)

        if earnings is not None and enterprise_value is not None:
            earnings_yield = earnings[-1] / np.maximum(enterprise_value[-1], 1e-10)
        else:
            earnings_yield = _earnings_yield_proxy(prices)

        if book_value is not None and enterprise_value is not None:
            book_yield = book_value[-1] / np.maximum(enterprise_value[-1], 1e-10)
        else:
            book_yield = _book_yield_proxy(prices)

        if cash_flow is not None and enterprise_value is not None:
            cf_yield = cash_flow[-1] / np.maximum(enterprise_value[-1], 1e-10)
        else:
            cf_yield = _cashflow_yield_proxy(prices)

        value = (
            0.35 * _safe_zscore(book_yield)
            + 0.35 * _safe_zscore(earnings_yield)
            + 0.30 * _safe_zscore(cf_yield)
        )
        return _safe_zscore(value)

    def _lowvol_signal(
        self,
        prices: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Low-volatility / risk-efficiency signal."""
        n_assets = prices.shape[1]
        vol = _realized_vol(prices, self.vol_lookback)
        beta = self._beta_signal(prices)
        sharpe = np.zeros(n_assets, dtype=float)
        for asset in range(n_assets):
            rets = returns[:, asset]
            rets = rets[np.isfinite(rets)]
            if len(rets) > 20 and np.std(rets, ddof=1) > 1e-10:
                sharpe[asset] = np.mean(rets) / np.std(rets, ddof=1)
        return _safe_zscore(
            -0.60 * _safe_zscore(vol) - 0.25 * _safe_zscore(beta) + 0.15 * _safe_zscore(sharpe)
        )

    def _reversal_signal(self, prices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Short-term reversal: negative 5-day return."""
        n_periods, n_assets = prices.shape
        score = np.zeros(n_assets, dtype=float)
        window = min(5, n_periods)
        for asset in range(n_assets):
            valid = prices[-window:, asset]
            valid = valid[np.isfinite(valid) & (valid > 0)]
            if len(valid) >= 3:
                score[asset] = -(valid[-1] / valid[0] - 1.0)
        return _safe_zscore(score)

    def _liquidity_factor(
        self,
        prices: NDArray[np.float64],
        data: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Liquidity/flow factor."""
        liquidity_score = _liquidity_score(prices, data)
        volume = _optional_2d(data.get("volume"), prices.shape[0], prices.shape[1])
        if volume is None:
            return _safe_zscore(liquidity_score / 100.0)

        adv = np.nanmean(volume[-21:] * prices[-21:], axis=0)
        recent_adv = np.nanmean(volume[-5:] * prices[-5:], axis=0)
        volume_surprise = (recent_adv - adv) / np.maximum(adv, 1e-10)
        return _safe_zscore(0.70 * (liquidity_score / 100.0) + 0.30 * _safe_zscore(volume_surprise))

    def _vol_carry_signal(
        self,
        prices: NDArray[np.float64],
        returns: NDArray[np.float64],
        data: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Implied-realized volatility carry."""
        _ = returns
        implied = _optional_2d(data.get("implied_vol"), prices.shape[0], prices.shape[1])
        if implied is None:
            return np.zeros(prices.shape[1], dtype=float)

        rv = _realized_vol(prices, min(30, self.vol_lookback)) / math.sqrt(252)
        iv = np.nan_to_num(implied[-1], nan=0.0, posinf=0.0, neginf=0.0)
        carry = (iv - rv) / np.maximum(iv, 1e-10)
        return _safe_zscore(carry)

    def _beta_signal(self, prices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Beta to cross-sectional market proxy."""
        n_periods, n_assets = prices.shape
        betas = np.ones(n_assets, dtype=float)
        if n_periods < 30:
            return betas

        market = np.nanmean(prices, axis=1)
        market_returns = _returns(market.reshape(-1, 1)).ravel()
        asset_returns = _returns(prices)
        window = min(self.lookback_days, market_returns.shape[0])
        mkt = market_returns[-window:]
        mkt = mkt[np.isfinite(mkt)]
        if len(mkt) < 20 or np.var(mkt) <= 1e-10:
            return betas

        for asset in range(n_assets):
            asset_ret = asset_returns[-window:, asset]
            asset_ret = asset_ret[np.isfinite(asset_ret)]
            length = min(len(mkt), len(asset_ret))
            if length < 20:
                continue
            cov = np.cov(mkt[-length:], asset_ret[-length:])[0, 1]
            betas[asset] = cov / np.var(mkt[-length:])
        return betas

    def _combine_factors(self, factors: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Combine factors into a raw composite alpha score."""
        weights = {
            "momentum": self.momentum_weight,
            "tsmom": self.tsmom_weight,
            "quality": self.quality_weight,
            "value": self.value_weight,
            "lowvol": self.lowvol_weight,
            "reversal": self.reversal_weight,
            "liquidity": self.liquidity_weight,
            "vol_carry": self.vol_carry_weight,
        }
        composite = np.zeros_like(factors["momentum"], dtype=float)
        for name, factor in factors.items():
            composite += weights.get(name, 0.0) * _safe_zscore(factor)
        return _safe_zscore(composite)

    def _ic_adjusted_weights(
        self,
        composite: NDArray[np.float64],
        factors: dict[str, NDArray[np.float64]],
        returns: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Blend fixed composite with trailing IC-adjusted composite."""
        ics = self._estimate_trailing_ics(factors, returns)
        self._factor_ics = ics
        names = [
            "momentum",
            "tsmom",
            "quality",
            "value",
            "lowvol",
            "reversal",
            "liquidity",
            "vol_carry",
        ]
        positive = {name: max(ics.get(name, 0.0), 0.0) for name in names}
        total = sum(positive.values())
        if total <= 1e-10:
            return composite

        ic_composite = np.zeros_like(composite)
        for name in names:
            ic_composite += (positive[name] / total) * _safe_zscore(factors[name])
        return (1.0 - self.ic_blend) * composite + self.ic_blend * _safe_zscore(ic_composite)

    def _estimate_trailing_ics(
        self,
        factors: dict[str, NDArray[np.float64]],
        returns: NDArray[np.float64],
    ) -> dict[str, float]:
        """Estimate trailing Spearman IC using past returns as a stability proxy."""
        n_assets = returns.shape[1]
        if n_assets < 5 or returns.shape[0] < self.ic_window:
            return {name: 0.10 for name in factors}

        trailing = np.zeros(n_assets, dtype=float)
        start = max(0, returns.shape[0] - self.ic_window)
        for asset in range(n_assets):
            rets = returns[start:, asset]
            rets = rets[np.isfinite(rets)]
            trailing[asset] = np.prod(1.0 + rets[-min(len(rets), self.ic_window):]) - 1.0

        trailing_rank = _rank(trailing)
        ics: dict[str, float] = {}
        for name, factor in factors.items():
            corr = np.corrcoef(_rank(factor), trailing_rank)[0, 1]
            ics[name] = 0.0 if np.isnan(corr) else float(corr)
        return ics

    def _raw_weights(
        self,
        scores: NDArray[np.float64],
        vols: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Conviction/variance raw weights."""
        safe_vols = np.maximum(vols, 1e-10)
        weights = scores / (safe_vols**2)
        weights[~np.isfinite(weights)] = 0.0
        return weights

    def _apply_tail_selection(
        self,
        weights: NDArray[np.float64],
        scores: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Keep top/bottom tails and enforce threshold gates."""
        selected = np.zeros_like(weights)
        long_mask = (scores >= self.long_threshold) | (
            _rank(scores) >= max(1, len(scores) - self.n_long)
        )
        short_mask = (scores <= -self.short_threshold) | (_rank(scores) < self.n_short)
        long_mask &= scores > 0.0
        short_mask &= scores < 0.0
        short_mask &= ~long_mask
        selected[long_mask] = weights[long_mask]
        selected[short_mask] = weights[short_mask]
        return selected

    def _apply_exposure_limits(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale gross and net exposure to configured limits."""
        gross = float(np.sum(np.abs(weights)))
        if gross > 1e-10:
            weights = weights * min(self.max_gross_exposure, gross) / gross

        net = float(np.sum(weights))
        if net > self.max_net_exposure and net > 1e-10:
            weights = weights * self.max_net_exposure / net
        elif net < self.min_net_exposure and net < -1e-10:
            weights = weights * self.min_net_exposure / net
        return weights

    def _apply_single_name_limit(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cap single-name exposure."""
        cap = min(self.single_name_max_pct, 1.0)
        clipped = np.clip(weights, -cap, cap)
        gross = float(np.sum(np.abs(clipped)))
        if gross > self.max_gross_exposure > 1e-10:
            clipped *= self.max_gross_exposure / gross
        return clipped

    def _regime_multiplier(self, regime_score: float) -> float:
        """Convert regime score to gross exposure multiplier."""
        if not self.regime_exposure_multiplier:
            return 1.0
        if regime_score >= 70.0:
            multiplier = 1.0
        elif regime_score >= 55.0:
            multiplier = 0.85
        elif regime_score >= 40.0:
            multiplier = 0.60
        elif regime_score >= 25.0:
            multiplier = 0.35
        else:
            multiplier = self.min_regime_multiplier
        return float(np.clip(multiplier, self.min_regime_multiplier, self.max_regime_multiplier))

    def _exit_signals(
        self,
        symbols: list[str],
        prices: NDArray[np.float64],
        data: dict[str, NDArray[np.float64]],
    ) -> list[TradeSignal]:
        """Generate stop, target, time-stop, and signal-decay exits."""
        signals: list[TradeSignal] = []
        if not self._positions:
            return signals

        current_prices = prices[-1]
        vols = _realized_vol(prices, self.vol_lookback) if prices.shape[0] > 2 else None
        regime_score = _regime_score(data, prices)
        self._last_regime_score = regime_score

        for symbol, qty in list(self._positions.items()):
            if symbol not in self._entry_prices:
                continue
            try:
                idx = self._symbol_index(symbol)
            except ValueError:
                continue

            entry_price = self._entry_prices[symbol]
            current_price = float(current_prices[idx])
            if current_price <= 0.0 or entry_price <= 0.0:
                continue

            ret = (current_price - entry_price) / entry_price
            direction = 1 if qty > 0 else -1
            stop_pct = 0.08
            if vols is not None:
                stop_pct = float(np.clip(self.stop_loss_mult * vols[idx] * math.sqrt(20), 0.03, 0.12))
            take_pct = self.take_profit_mult * stop_pct

            should_exit = False
            action = "signal_decay"
            if (direction > 0 and ret <= -stop_pct) or (direction < 0 and ret >= stop_pct):
                should_exit = True
                action = "stop_loss"
            elif (direction > 0 and ret >= take_pct) or (direction < 0 and ret <= -take_pct):
                should_exit = True
                action = "take_profit"

            entry_day = self._entry_days.get(symbol, 0)
            if self._days_since_rebalance - entry_day >= self.time_stop_days:
                should_exit = True
                action = "time_stop"

            score = self._entry_scores.get(symbol, 0.0)
            current_score = float(self._current_scores[idx]) if self._current_scores is not None else score
            if (direction > 0 and current_score < self.exit_threshold) or (direction < 0 and current_score > -self.exit_threshold):
                should_exit = True
                action = "signal_decay"

            if regime_score < 25.0:
                should_exit = True
                action = "regime_halt"

            if should_exit:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        direction=0,
                        strength=1.0,
                        size=0.0,
                        metadata={"exit_action": action, "entry_return": ret},
                    )
                )

        return signals

    def _score_for_symbol(self, symbol: str) -> float:
        """Return current composite score for a symbol."""
        if self._current_scores is None or symbol not in self._current_symbols:
            return 0.0
        return float(self._current_scores[self._current_symbols.index(symbol)])

    def _symbol_index(self, symbol: str) -> int:
        """Return current symbol index."""
        if symbol in self._current_symbols:
            return self._current_symbols.index(symbol)
        raise ValueError(f"Symbol {symbol} not found in current universe")


def _as_2d(values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _optional_2d(
    value: Any,
    n_periods: int,
    n_assets: int,
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and arr.shape[0] in (n_periods, n_assets):
        if arr.shape[0] == n_assets:
            return np.tile(arr.reshape(1, -1), (n_periods, 1))
        return arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape == (n_periods, n_assets):
        return arr
    return None


def _optional_1d(value: Any, expected_len: int) -> NDArray[np.float64] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1 and arr.shape[0] == expected_len:
        return arr
    if arr.ndim == 2 and arr.shape[1] == expected_len:
        return arr[-1]
    return None


def _symbols(data: dict[str, NDArray[np.float64]], n_assets: int) -> list[str]:
    raw = data.get("symbols")
    if raw is None:
        return [f"asset_{idx}" for idx in range(n_assets)]
    return [str(symbol) for symbol in list(raw)[:n_assets]]


def _returns(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    safe = np.where(prices > 0, prices, np.nan)
    returns = np.diff(safe, axis=0) / np.maximum(safe[:-1], 1e-10)
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def _realized_vol(prices: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    n_periods = prices.shape[0]
    n_assets = prices.shape[1]
    vols = np.zeros(n_assets, dtype=float)
    start = max(0, n_periods - max(window, 2))
    for asset in range(n_assets):
        valid = prices[start:, asset]
        valid = valid[np.isfinite(valid) & (valid > 0)]
        if len(valid) >= 3:
            rets = np.diff(valid) / valid[:-1]
            vols[asset] = np.std(rets, ddof=1) * math.sqrt(252)
    return np.maximum(vols, 1e-10)


def _safe_zscore(values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    median = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - median))
    denom = 1.4826 * mad if mad > 1e-10 else np.nanstd(arr, ddof=1)
    if denom <= 1e-10:
        return np.zeros_like(arr)
    z = (arr - median) / denom
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _winsorize(values: NDArray[np.float64], lower: float, upper: float) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float).copy()
    lo = np.nanquantile(arr, lower)
    hi = np.nanquantile(arr, upper)
    return np.clip(arr, lo, hi)


def _rank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    valid = ~np.isnan(arr)
    ranks = np.zeros_like(arr, dtype=float)
    if np.sum(valid) == 0:
        return ranks
    order = np.argsort(np.argsort(arr[valid])).astype(float)
    n = int(np.sum(valid))
    ranks[valid] = order / max(n - 1, 1)
    return ranks


def _sector_neutralize(
    values: NDArray[np.float64],
    data: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    sectors = data.get("sector") if data else None
    if sectors is None:
        return values
    arr = np.asarray(values, dtype=float).copy()
    sec = np.asarray(sectors)
    if sec.shape[0] != arr.shape[0]:
        return arr
    neutral = arr.copy()
    for sector in np.unique(sec):
        mask = sec == sector
        if np.sum(mask) > 1:
            neutral[mask] = arr[mask] - np.nanmedian(arr[mask])
    return neutral


def _liquidity_score(
    prices: NDArray[np.float64],
    data: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    n_periods, n_assets = prices.shape
    volume = _optional_2d(data.get("volume"), n_periods, n_assets)
    spread = _optional_2d(data.get("spread_bps"), n_periods, n_assets)
    adv_dollar = _optional_2d(data.get("adv_dollar"), n_periods, n_assets)
    float_value = _optional_2d(data.get("float_value"), n_periods, n_assets)
    inst_ownership = _optional_2d(data.get("institutional_ownership"), n_periods, n_assets)
    days_to_liquidate = _optional_2d(data.get("days_to_liquidate"), n_periods, n_assets)

    if adv_dollar is not None:
        adv = np.nanmean(adv_dollar[-21:], axis=0)
    elif volume is None:
        adv = np.nanmean(prices[-21:] * 1.0, axis=0)
    else:
        adv = np.nanmean(volume[-21:] * prices[-21:], axis=0)

    spread_bps = np.nanmean(spread[-21:], axis=0) if spread is not None else np.full(n_assets, 10.0)
    float_mv = (
        np.nanmean(float_value[-1], axis=0)
        if float_value is not None
        else np.log1p(adv) * 1e6
    )
    ownership = (
        np.nanmean(inst_ownership[-1], axis=0)
        if inst_ownership is not None
        else np.full(n_assets, 50.0)
    )
    dtl = (
        np.nanmean(days_to_liquidate[-1], axis=0)
        if days_to_liquidate is not None
        else np.full(n_assets, 1.0)
    )

    score = (
        40.0 * _percentile_score(np.log1p(np.maximum(adv, 0.0)))
        + 20.0 * _percentile_score(np.log1p(np.maximum(float_mv, 0.0)))
        + 20.0 * np.clip(1.0 - spread_bps / 50.0, 0.0, 1.0)
        + 10.0 * _percentile_score(1.0 / np.maximum(dtl, 1e-10))
        + 10.0 * np.clip(ownership / 100.0, 0.0, 1.0)
    )
    return np.clip(score, 0.0, 100.0)


def _percentile_score(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return _rank(values) * 100.0


def _regime_score(data: dict[str, NDArray[np.float64]], prices: NDArray[np.float64]) -> float:
    explicit = data.get("regime_score")
    if explicit is not None:
        arr = np.asarray(explicit, dtype=float)
        value = float(arr[-1] if arr.ndim == 1 else arr[-1, -1])
        return float(np.clip(value, 0.0, 100.0))

    market = _optional_1d(data.get("market_close"), prices.shape[0])
    if market is None or len(market) < 22:
        return 55.0

    ret = market[-1] / market[-21] - 1.0 if market[-21] > 0 else 0.0
    vol = _realized_vol(market.reshape(-1, 1), 21)[0] / math.sqrt(252)
    score = 50.0 + 100.0 * ret - 150.0 * vol
    return float(np.clip(score, 0.0, 100.0))


def _earnings_yield_proxy(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    n_assets = prices.shape[1]
    score = np.zeros(n_assets, dtype=float)
    for asset in range(n_assets):
        valid = prices[-252:, asset]
        valid = valid[np.isfinite(valid) & (valid > 0)]
        if len(valid) >= 30:
            rets = np.diff(valid) / valid[:-1]
            mean_ret = np.mean(rets)
            price_cv = np.std(valid) / max(np.mean(valid), 1e-10)
            score[asset] = mean_ret / max(price_cv, 1e-10)
    return score


def _book_yield_proxy(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    n_assets = prices.shape[1]
    score = np.zeros(n_assets, dtype=float)
    for asset in range(n_assets):
        valid = prices[-252:, asset]
        valid = valid[np.isfinite(valid) & (valid > 0)]
        if len(valid) >= 30:
            mean_price = np.mean(valid)
            score[asset] = -(valid[-1] - mean_price) / max(mean_price, 1e-10)
    return score


def _cashflow_yield_proxy(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    n_assets = prices.shape[1]
    score = np.zeros(n_assets, dtype=float)
    for asset in range(n_assets):
        valid = prices[-252:, asset]
        valid = valid[np.isfinite(valid) & (valid > 0)]
        if len(valid) >= 30:
            rets = np.diff(valid) / valid[:-1]
            std_ret = np.std(rets, ddof=1)
            score[asset] = 0.0 if std_ret <= 1e-10 else np.mean(rets) / std_ret
    return score


def _signals_from_weights(
    weights: NDArray[np.float64],
    symbols: list[str],
    prices: NDArray[np.float64],
    vols: NDArray[np.float64],
    scores: NDArray[np.float64],
    max_position_notional: float,
    single_name_max_pct: float,
    stop_loss_mult: float,
    take_profit_mult: float,
    metadata: dict[str, Any] | None = None,
) -> list[TradeSignal]:
    """Convert target weights into TradeSignal objects."""
    signals: list[TradeSignal] = []
    metadata = metadata or {}
    cap = min(single_name_max_pct, 1.0)
    for idx, weight in enumerate(weights):
        if abs(weight) <= 1e-10 or idx >= len(symbols):
            continue
        direction = 1 if weight > 0 else -1
        capped_weight = float(np.clip(weight, -cap, cap))
        stop_pct = float(np.clip(stop_loss_mult * vols[idx] * math.sqrt(20), 0.03, 0.12))
        take_pct = float(np.clip(take_profit_mult * stop_pct, 0.05, 0.20))
        current_price = float(prices[idx])
        if direction > 0:
            stop_loss = current_price * (1.0 - stop_pct)
            take_profit = current_price * (1.0 + take_pct)
        else:
            stop_loss = current_price * (1.0 + stop_pct)
            take_profit = current_price * (1.0 - take_pct)

        item_metadata = dict(metadata)
        item_metadata.update(
            {
                "composite_alpha": float(scores[idx]),
                "weight": capped_weight,
                "ann_vol": float(vols[idx]),
                "stop_loss_pct": stop_pct,
                "take_profit_pct": take_pct,
            }
        )
        signals.append(
            TradeSignal(
                symbol=symbols[idx],
                direction=direction,
                strength=float(np.clip(abs(scores[idx]) / 2.0, 0.0, 1.0)),
                size=abs(capped_weight) * max_position_notional,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=item_metadata,
            )
        )
    return signals
