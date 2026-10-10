"""Statistical arbitrage mean reversion strategy on cointegrated pairs.

Uses pre-identified cointegrated pairs (from PairsFinder) and trades
mean reversion of the spread via z-score signals.

Implements the Strategy ABC for integration with the backtest engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register

log = logging.getLogger(__name__)


@dataclass
class PairState:
    """Runtime state for a single pair."""
    stock_a: str
    stock_b: str
    hedge_ratio: float
    intercept: float
    spread_mean: float
    spread_std: float
    half_life: float
    position: int = 0          # +1 = long A / short B, -1 = short A / long B, 0 = flat
    entry_zscore: float = 0.0
    entry_time: int = 0


@register("equity_stat_arb_mr")
class StatArbMeanReversion(Strategy):
    """Statistical arbitrage mean reversion on cointegrated pairs.

    At each rebalance, computes z-score of each pair's spread.
    Entry: |z-score| > entry_threshold (default 2.0)
    Exit: z-score crosses exit_threshold (default 0.0)
    Stop: |z-score| > stop_threshold (default 4.0)

    Config params:
        entry_threshold: z-score to enter a trade (default 2.0)
        exit_threshold: z-score to exit a trade (default 0.0)
        stop_threshold: z-score to stop-loss (default 4.0)
        lookback_zscore: rolling window for z-score computation in days (default 20)
        max_pairs: max simultaneous pair positions (default 10)
        max_pair_notional: max notional per pair leg (default 50000)
        rebalance_freq: rebalance frequency in bars (default 60 = 1 hour for minute data)
    """

    def __init__(
        self,
        config: StrategyConfig,
        pairs: list[dict] | None = None,
    ) -> None:
        super().__init__(config)
        self.entry_threshold = float(config.params.get("entry_threshold", 2.0))
        self.exit_threshold = float(config.params.get("exit_threshold", 0.0))
        self.stop_threshold = float(config.params.get("stop_threshold", 4.0))
        self.lookback_zscore = int(config.params.get("lookback_zscore", 20))
        self.max_pairs = int(config.params.get("max_pairs", 10))
        self.max_pair_notional = float(config.params.get("max_pair_notional", 50_000))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 60))

        self._pair_states: list[PairState] = []
        self._spread_history: dict[str, list[float]] = {}
        self._bar_count = 0
        self._pairs_data = pairs or []

    def load_pairs(self, pairs: list[dict]) -> None:
        """Load pair definitions from PairsFinder output.

        Each dict should have: stock_a, stock_b, hedge_ratio, intercept,
        spread_mean, spread_std, half_life.
        """
        self._pair_states = []
        self._spread_history = {}
        for p in pairs:
            key = f"{p['stock_a']}_{p['stock_b']}"
            self._pair_states.append(PairState(
                stock_a=p["stock_a"],
                stock_b=p["stock_b"],
                hedge_ratio=p["hedge_ratio"],
                intercept=p["intercept"],
                spread_mean=p["spread_mean"],
                spread_std=p["spread_std"],
                half_life=p["half_life"],
            ))
            self._spread_history[key] = []
        log.info("Loaded %d pairs for stat arb", len(self._pair_states))

    def _compute_spread(
        self,
        pair: PairState,
        close_a: float,
        close_b: float,
    ) -> float:
        """Compute spread = log(B) - hedge_ratio * log(A) - intercept."""
        return np.log(close_b) - pair.hedge_ratio * np.log(close_a) - pair.intercept

    def _compute_zscore(self, pair: PairState, current_spread: float) -> float:
        """Compute z-score using rolling window or training stats."""
        key = f"{pair.stock_a}_{pair.stock_b}"
        history = self._spread_history.get(key, [])

        if len(history) >= self.lookback_zscore:
            window = np.array(history[-self.lookback_zscore:])
            mean = float(np.mean(window))
            std = float(np.std(window))
        else:
            mean = pair.spread_mean
            std = pair.spread_std

        if std < 1e-10:
            return 0.0
        return (current_spread - mean) / std

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Process market data and emit pair trade signals."""
        close = data.get("close")
        if close is None or close.ndim != 2:
            return []

        n_periods, n_assets = close.shape
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])
        sym_to_idx = {s: i for i, s in enumerate(symbols)}

        self._bar_count += 1

        # Rebalance check
        if self._bar_count % self.rebalance_freq != 0:
            return self._current_signals()

        signals: list[TradeSignal] = []
        active_pairs = sum(1 for ps in self._pair_states if ps.position != 0)

        for ps in self._pair_states:
            idx_a = sym_to_idx.get(ps.stock_a)
            idx_b = sym_to_idx.get(ps.stock_b)
            if idx_a is None or idx_b is None:
                continue

            close_a = float(close[-1, idx_a])
            close_b = float(close[-1, idx_b])
            if close_a <= 0 or close_b <= 0:
                continue

            # Compute spread and z-score
            spread = self._compute_spread(ps, close_a, close_b)
            key = f"{ps.stock_a}_{ps.stock_b}"
            self._spread_history.setdefault(key, []).append(spread)

            # Trim history
            max_hist = self.lookback_zscore * 3
            if len(self._spread_history[key]) > max_hist:
                self._spread_history[key] = self._spread_history[key][-max_hist:]

            zscore = self._compute_zscore(ps, spread)

            # Position management
            if ps.position == 0:
                # Entry logic
                if active_pairs >= self.max_pairs:
                    continue

                if zscore > self.entry_threshold:
                    # Spread is high -> short A, long B (bet on narrowing)
                    ps.position = -1
                    ps.entry_zscore = zscore
                    ps.entry_time = self._bar_count
                    active_pairs += 1

                    notional = self.max_pair_notional
                    signals.append(TradeSignal(
                        symbol=ps.stock_a,
                        direction=-1,
                        strength=min(abs(zscore) / self.stop_threshold, 1.0),
                        size=notional,
                        metadata={"pair": key, "leg": "short_a", "zscore": zscore},
                    ))
                    signals.append(TradeSignal(
                        symbol=ps.stock_b,
                        direction=1,
                        strength=min(abs(zscore) / self.stop_threshold, 1.0),
                        size=notional * ps.hedge_ratio,
                        metadata={"pair": key, "leg": "long_b", "zscore": zscore},
                    ))

                elif zscore < -self.entry_threshold:
                    # Spread is low -> long A, short B (bet on widening)
                    ps.position = 1
                    ps.entry_zscore = zscore
                    ps.entry_time = self._bar_count
                    active_pairs += 1

                    notional = self.max_pair_notional
                    signals.append(TradeSignal(
                        symbol=ps.stock_a,
                        direction=1,
                        strength=min(abs(zscore) / self.stop_threshold, 1.0),
                        size=notional,
                        metadata={"pair": key, "leg": "long_a", "zscore": zscore},
                    ))
                    signals.append(TradeSignal(
                        symbol=ps.stock_b,
                        direction=-1,
                        strength=min(abs(zscore) / self.stop_threshold, 1.0),
                        size=notional * ps.hedge_ratio,
                        metadata={"pair": key, "leg": "short_b", "zscore": zscore},
                    ))
            else:
                # Exit logic
                should_exit = False

                if (ps.position == -1 and zscore <= self.exit_threshold) or (ps.position == 1 and zscore >= -self.exit_threshold):
                    should_exit = True

                # Stop-loss
                if abs(zscore) > self.stop_threshold:
                    should_exit = True

                if should_exit:
                    # Flatten both legs
                    signals.append(TradeSignal(
                        symbol=ps.stock_a,
                        direction=0,
                        strength=0.0,
                        size=0.0,
                        metadata={"pair": key, "leg": "exit_a", "zscore": zscore},
                    ))
                    signals.append(TradeSignal(
                        symbol=ps.stock_b,
                        direction=0,
                        strength=0.0,
                        size=0.0,
                        metadata={"pair": key, "leg": "exit_b", "zscore": zscore},
                    ))
                    ps.position = 0
                    ps.entry_zscore = 0.0

        return signals

    def _current_signals(self) -> list[TradeSignal]:
        """Return signals to maintain current positions (no-op for now)."""
        return []
