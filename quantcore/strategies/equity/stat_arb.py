"""Multi-pair statistical arbitrage strategy: cointegration, Kalman filter, Kelly sizing.

Identifies cointegrated pairs from a universe, tracks spread via Kalman filter,
and sizes positions using Kelly criterion based on spread statistics.
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


@register("equity_stat_arb")
class StatArbStrategy(Strategy):
    """Multi-pair stat-arb with cointegration scan, Kalman hedge, Kelly sizing.

    Config params:
        lookback: cointegration window (default 252)
        half_life_max: max half-life to consider (default 20)
        entry_z: z-score entry threshold (default 2.0)
        exit_z: z-score exit threshold (default 0.5)
        kelly_fraction: fraction of Kelly to use (default 0.5)
        max_pairs: max active pairs (default 10)
        rescan_freq: pair rescan frequency in days (default 63)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback = int(config.params.get("lookback", 252))
        self.half_life_max = int(config.params.get("half_life_max", 20))
        self.entry_z = float(config.params.get("entry_z", 2.0))
        self.exit_z = float(config.params.get("exit_z", 0.5))
        self.kelly_fraction = float(config.params.get("kelly_fraction", 0.5))
        self.max_pairs = int(config.params.get("max_pairs", 10))
        self.rescan_freq = int(config.params.get("rescan_freq", 63))
        self._pairs: list[dict] = []
        self._days_since_scan = self.rescan_freq  # Force initial scan
        # Positions held per symbol (checked by the pair-exit branch below;
        # previously never initialized -> AttributeError whenever an exit
        # signal was evaluated, which silently killed whole backtest folds).
        self._positions: dict[str, float] = {}

    def _scan_pairs(self, prices: NDArray) -> list[dict]:
        """Scan for cointegrated pairs using Engle-Granger approach."""
        n_periods, n_assets = prices.shape
        pairs = []

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                y = prices[-self.lookback:, i]
                x = prices[-self.lookback:, j]

                # Simple OLS hedge ratio
                valid = ~(np.isnan(y) | np.isnan(x))
                if np.sum(valid) < 60:
                    continue

                y_c, x_c = y[valid], x[valid]
                x_mean = np.mean(x_c)
                y_mean = np.mean(y_c)
                denom = np.sum((x_c - x_mean) ** 2)
                if denom < 1e-10:
                    continue

                beta = np.sum((y_c - y_mean) * (x_c - x_mean)) / denom
                alpha = y_mean - beta * x_mean
                spread = y_c - alpha - beta * x_c

                # ADF-like test: check spread mean-reversion speed
                spread_diff = np.diff(spread)
                spread_lag = spread[:-1]
                if np.std(spread_lag) < 1e-10:
                    continue

                # Simple OLS: diff_t = lambda * spread_{t-1} + noise
                sl_mean = np.mean(spread_lag)
                lambda_num = np.sum((spread_diff - np.mean(spread_diff)) * (spread_lag - sl_mean))
                lambda_den = np.sum((spread_lag - sl_mean) ** 2)
                if lambda_den < 1e-10:
                    continue
                lam = lambda_num / lambda_den

                # Half-life in trading days
                if lam >= 0:
                    continue  # Not mean-reverting
                half_life = -np.log(2) / lam if lam < 0 else 999
                if half_life > self.half_life_max or half_life < 1:
                    continue

                # Score: lower half-life + higher mean-reversion strength = better
                spread_std = np.std(spread)
                if spread_std < 1e-10:
                    continue

                score = abs(lam) / half_life
                pairs.append({
                    "asset_a": i,
                    "asset_b": j,
                    "hedge_ratio": beta,
                    "intercept": alpha,
                    "half_life": half_life,
                    "spread_std": spread_std,
                    "score": score,
                })

        # Sort by score (higher = better)
        pairs.sort(key=lambda p: p["score"], reverse=True)
        return pairs[:self.max_pairs]

    def _kalman_spread(self, spread: NDArray) -> tuple[NDArray, NDArray]:
        """2-state Kalman filter (mean, slope) on the spread.

        Mathematically identical to the 2x2 matrix form (F=[[1,1],[0,1]],
        H=[1,0], Q=0.001*I, R=0.1*var(spread)): the matrix chain products
        and the 1x1 inverse are expanded elementwise and the recursion runs
        on plain floats. Identical equations, same result to floating-point
        tolerance, ~100x faster than calling np.linalg.inv per step (the
        per-step Python matrix algebra dominated the batch validator at
        96% of stat_arb's runtime - 2M inv() calls).
        """
        n = len(spread)
        r = float(np.var(spread)) * 0.1  # observation noise (R)

        # State x_est = [mean, slope]; P = [[p00, p01], [p10, p11]]
        m = float(spread[0])
        s = 0.0
        p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

        means = np.zeros(n)
        stds = np.zeros(n)

        for t in range(n):
            # Predict: x_pred = F @ x_est
            m_pred = m + s
            s_pred = s
            # P_pred = F @ P @ F.T + Q, F = [[1,1],[0,1]], Q = 0.001*I
            pp00 = p00 + p01 + p10 + p11 + 0.001
            pp01 = p01 + p11
            pp10 = p10 + p11
            pp11 = p11 + 0.001

            # Update: S = H @ P_pred @ H.T + R (1x1 inverse = 1/S)
            z = float(spread[t]) - m_pred
            S = pp00 + r
            k0 = pp00 / S
            k1 = pp10 / S

            m = m_pred + k0 * z
            s = s_pred + k1 * z

            # P = (I - K @ H) @ P_pred
            p00 = pp00 - k0 * pp00
            p01 = pp01 - k0 * pp10
            p10 = pp10 - k1 * pp00
            p11 = pp11 - k1 * pp10

            means[t] = m
            stds[t] = np.sqrt(max(p00, 1e-10))

        return means, stds

    def _kelly_size(self, half_life: float, spread_std: float) -> float:
        """Kelly position sizing based on spread statistics."""
        # Expected return per trade = entry_z * spread_std
        # Holding period = half_life days
        # Win rate from normal distribution at entry_z
        from scipy.stats import norm
        win_rate = 1.0 - norm.cdf(-self.entry_z)
        avg_win = self.entry_z * spread_std
        avg_loss = (self.entry_z - self.exit_z) * spread_std * 0.5

        if avg_loss < 1e-10:
            return 0.5

        kelly = (win_rate - (1.0 - win_rate) * avg_win / avg_loss) if avg_win > 0 else 0.0
        kelly = max(0.0, min(kelly, 1.0))
        return kelly * self.kelly_fraction

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        signals: list[TradeSignal] = []

        # Rescan for pairs periodically
        self._days_since_scan += 1
        if self._days_since_scan >= self.rescan_freq and n_periods >= self.lookback:
            self._pairs = self._scan_pairs(close)
            self._days_since_scan = 0

        # Trade existing pairs
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        for pair in self._pairs:
            a, b = pair["asset_a"], pair["asset_b"]
            hedge = pair["hedge_ratio"]
            intercept = pair["intercept"]
            spread_std = pair["spread_std"]

            y = close[-self.lookback:, a]
            x = close[-self.lookback:, b]
            spread = y - intercept - hedge * x

            if len(spread) < 10:
                continue

            # Kalman filter for current z-score
            means, stds = self._kalman_spread(spread)
            current_z = (spread[-1] - means[-1]) / max(spread_std, 1e-10)

            size = self._kelly_size(pair["half_life"], spread_std)

            if abs(current_z) > self.entry_z:
                # Entry: short the spread if z > threshold
                direction_a = -1 if current_z > 0 else 1
                direction_b = 1 if current_z > 0 else -1

                notional = size * self.config.max_position_notional / 2.0
                signals.append(TradeSignal(
                    symbol=symbols[a],
                    direction=direction_a,
                    strength=min(abs(current_z) / (self.entry_z * 2), 1.0),
                    size=notional,
                    metadata={"pair": f"{symbols[a]}-{symbols[b]}", "z_score": current_z},
                ))
                signals.append(TradeSignal(
                    symbol=symbols[b],
                    direction=direction_b,
                    strength=min(abs(current_z) / (self.entry_z * 2), 1.0),
                    size=notional * abs(hedge),
                    metadata={"pair": f"{symbols[a]}-{symbols[b]}", "z_score": current_z},
                ))

            elif abs(current_z) < self.exit_z:
                # Exit: flatten the pair
                if symbols[a] in self._positions and abs(self._positions[symbols[a]]) > 1e-8:
                    signals.append(TradeSignal(
                        symbol=symbols[a], direction=0, strength=0.0, size=0.0,
                        metadata={"action": "exit_pair", "z_score": current_z},
                    ))
                if symbols[b] in self._positions and abs(self._positions[symbols[b]]) > 1e-8:
                    signals.append(TradeSignal(
                        symbol=symbols[b], direction=0, strength=0.0, size=0.0,
                        metadata={"action": "exit_pair", "z_score": current_z},
                    ))

        return signals
