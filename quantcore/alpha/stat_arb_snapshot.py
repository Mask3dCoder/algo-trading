"""Point-in-time statistical arbitrage mean-reversion on cointegrated pairs.

Native ``on_snapshot`` port of :mod:`quantcore.alpha.stat_arb_mean_reversion`
so it runs through ``BacktestEngine``'s PIT-guarded loop. The legacy module
speaks the old ``on_data`` / ``TradeSignal`` protocol and is *not* engine
compatible; this module ports the identical pair-spread z-score logic onto the
engine's ``on_snapshot(snapshot, portfolio) -> list[SignalData]`` protocol.

Pairs are discovered up-front (out-of-sample) via ``PairsFinder`` on a
point-in-time training window, then traded as market-neutral spread
mean-reversion. Each rebalance, every pair's spread z-score is computed and:

  * entry  : |z| > entry_threshold  -> open long-A/short-B (or inverse) by hedge
  * exit   : z crosses exit_threshold (back toward 0)
  * stop   : |z| > stop_threshold    -> flatten

The engine enforces a strict look-ahead barrier before ``on_snapshot`` runs, so
only prices at or before ``snapshot.timestamp`` are ever visible.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantcore.backtest.engine import SignalData

if TYPE_CHECKING:
    import pandas as pd

_log = logging.getLogger(__name__)

STRATEGY_ID = "stat_arb_mr"


@dataclass
class _PairState:
    """Runtime state for one cointegrated pair (mirrors legacy PairState)."""

    stock_a: str
    stock_b: str
    hedge_ratio: float
    intercept: float
    spread_mean: float
    spread_std: float
    half_life: float
    position: int = 0        # +1 long A/short B, -1 short A/long B, 0 flat
    entry_zscore: float = 0.0
    entry_bar: int = 0


def pairs_from_panel(
    panel: pd.DataFrame,
    train_start: str = "2018-01-01",
    train_end: str = "2021-12-31",
    lookback_days: int = 252,
    max_pvalue: float = 0.05,
    max_pairs: int = 50,
    min_half_life: float = 1.0,
    max_half_life: float = 60.0,
) -> list[dict]:
    """Discover cointegrated pairs on a point-in-time training window.

    ``panel`` is the LONG/tidy frame [symbol, timestamp, open, high, low,
    close, volume]. Pairs are found on ``[train_start, train_end]`` only, so
    no OOS data leaks into pair selection. Returns a list of dicts suitable
    for :class:`StatArbMeanReversionSnapshot`.
    """
    import pandas as pd

    from quantcore.alpha.pairs_finder import PairsFinder

    tr = panel.copy()
    tr["timestamp"] = pd.to_datetime(tr["timestamp"])
    train = tr[(tr["timestamp"] >= train_start) & (tr["timestamp"] <= train_end)]
    wide = train.pivot(index="timestamp", columns="symbol", values="close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.dropna(axis=1, how="any")
    if wide.shape[1] < 2 or len(wide) < lookback_days:
        _log.warning("Not enough training data to find pairs (%d rows, %d names)",
                     len(wide), wide.shape[1])
        return []

    pf = PairsFinder(
        lookback_days=lookback_days,
        max_pvalue=max_pvalue,
        max_pairs=max_pairs,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        bar_size="1d",
    )
    found = pf.find_pairs(wide)
    _log.info("PairsFinder returned %d candidate pairs on training window", len(found))
    return [
        {
            "stock_a": p.stock_a,
            "stock_b": p.stock_b,
            "hedge_ratio": p.hedge_ratio,
            "intercept": p.intercept,
            "spread_mean": p.spread_mean,
            "spread_std": p.spread_std,
            "half_life": p.half_life,
        }
        for p in found
    ]


class StatArbMeanReversionSnapshot:
    """Market-neutral stat-arb mean reversion, engine-native (on_snapshot).

    Parameters
    ----------
    pairs:
        List of dicts from :func:`pairs_from_panel` (or ``PairsFinder``).
    entry_threshold:
        |z| to open a position (default 2.0).
    exit_threshold:
        z level at which to close an open position (default 0.0).
    stop_threshold:
        |z| stop-loss (default 4.0).
    lookback_zscore:
        Rolling window (bars) for the live z-score (default 20).
    max_pairs:
        Max simultaneous open pairs (default 10).
    max_pair_notional:
        Notional per pair leg A (default 50_000).
    rebalance_freq:
        Emit signals only every N bars (default 60).
    regime_filter:
        Optional RegimeFilter; risk-off flattens all open pairs.
    """

    def __init__(
        self,
        *,
        pairs: list[dict],
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.0,
        stop_threshold: float = 4.0,
        lookback_zscore: int = 20,
        max_pairs: int = 10,
        max_pair_notional: float = 50_000.0,
        rebalance_freq: int = 60,
        regime_filter=None,
    ) -> None:
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        self.stop_threshold = float(stop_threshold)
        self.lookback_zscore = int(lookback_zscore)
        self.max_pairs = int(max_pairs)
        self.max_pair_notional = float(max_pair_notional)
        self.rebalance_freq = max(1, int(rebalance_freq))
        self._regime = regime_filter

        self._states: list[_PairState] = []
        for p in pairs:
            self._states.append(_PairState(
                stock_a=p["stock_a"], stock_b=p["stock_b"],
                hedge_ratio=float(p["hedge_ratio"]), intercept=float(p["intercept"]),
                spread_mean=float(p["spread_mean"]), spread_std=float(p["spread_std"]),
                half_life=float(p["half_life"]),
            ))
        self._spread_hist: dict[str, list[float]] = {
            f"{p['stock_a']}_{p['stock_b']}": [] for p in pairs
        }
        self._prices: dict[str, list[float]] = defaultdict(list)
        self._bar = 0
        self._max_hist = self.lookback_zscore * 3
        _log.info("StatArbSnapshot loaded %d pairs", len(self._states))

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _spread(ps: _PairState, close_a: float, close_b: float) -> float:
        return float(np.log(close_b) - ps.hedge_ratio * np.log(close_a) - ps.intercept)

    def _zscore(self, ps: _PairState, spread: float) -> float:
        key = f"{ps.stock_a}_{ps.stock_b}"
        hist = self._spread_hist.get(key, [])
        if len(hist) >= self.lookback_zscore:
            w = np.array(hist[-self.lookback_zscore:], dtype=float)
            mean = float(w.mean())
            std = float(w.std())
        else:
            mean, std = ps.spread_mean, ps.spread_std
        if std < 1e-10:
            return 0.0
        return (spread - mean) / std

    # -- engine interface -------------------------------------------------
    def on_snapshot(self, snapshot, portfolio) -> list[SignalData]:
        self._bar += 1
        for sym, bar in snapshot.bars.items():
            self._prices[sym].append(float(bar.close))

        # Regime gate: risk-off -> flatten every open pair, trade nothing.
        if self._regime is not None and not self._regime.is_risk_on(snapshot.timestamp):
            return [
                SignalData(symbol=s.stock_a, direction="flat", strength=0.0,
                           strategy_id=STRATEGY_ID)
                for s in self._states if s.position != 0
            ] + [
                SignalData(symbol=s.stock_b, direction="flat", strength=0.0,
                           strategy_id=STRATEGY_ID)
                for s in self._states if s.position != 0
            ]

        if self._bar % self.rebalance_freq != 0:
            return []

        signals: list[SignalData] = []
        active = sum(1 for s in self._states if s.position != 0)

        for ps in self._states:
            ca = self._prices.get(ps.stock_a)
            cb = self._prices.get(ps.stock_b)
            if not ca or not cb:
                continue
            close_a, close_b = ca[-1], cb[-1]
            if close_a <= 0 or close_b <= 0:
                continue

            spread = self._spread(ps, close_a, close_b)
            key = f"{ps.stock_a}_{ps.stock_b}"
            hist = self._spread_hist.setdefault(key, [])
            hist.append(spread)
            if len(hist) > self._max_hist:
                del hist[: len(hist) - self._max_hist]

            z = self._zscore(ps, spread)
            strength = float(min(abs(z) / self.stop_threshold, 1.0))

            if ps.position == 0:
                if active >= self.max_pairs:
                    continue
                if z > self.entry_threshold:
                    # Spread rich -> short A, long B (bet on narrowing).
                    ps.position = -1
                    ps.entry_zscore = z
                    ps.entry_bar = self._bar
                    active += 1
                    not_a = self.max_pair_notional
                    not_b = self.max_pair_notional * ps.hedge_ratio
                    signals.append(SignalData(symbol=ps.stock_a, direction="short",
                                               strength=strength, strategy_id=STRATEGY_ID,
                                               notional_size=not_a,
                                               metadata={"pair": key, "leg": "short_a", "z": z}))
                    signals.append(SignalData(symbol=ps.stock_b, direction="long",
                                               strength=strength, strategy_id=STRATEGY_ID,
                                               notional_size=not_b,
                                               metadata={"pair": key, "leg": "long_b", "z": z}))
                elif z < -self.entry_threshold:
                    # Spread cheap -> long A, short B.
                    ps.position = 1
                    ps.entry_zscore = z
                    ps.entry_bar = self._bar
                    active += 1
                    not_a = self.max_pair_notional
                    not_b = self.max_pair_notional * ps.hedge_ratio
                    signals.append(SignalData(symbol=ps.stock_a, direction="long",
                                               strength=strength, strategy_id=STRATEGY_ID,
                                               notional_size=not_a,
                                               metadata={"pair": key, "leg": "long_a", "z": z}))
                    signals.append(SignalData(symbol=ps.stock_b, direction="short",
                                               strength=strength, strategy_id=STRATEGY_ID,
                                               notional_size=not_b,
                                               metadata={"pair": key, "leg": "short_b", "z": z}))
            else:
                should_exit = False
                if (ps.position == -1 and z <= self.exit_threshold) or (ps.position == 1 and z >= -self.exit_threshold):
                    should_exit = True
                if abs(z) > self.stop_threshold:
                    should_exit = True
                if should_exit:
                    signals.append(SignalData(symbol=ps.stock_a, direction="flat",
                                               strength=0.0, strategy_id=STRATEGY_ID,
                                               metadata={"pair": key, "leg": "exit_a", "z": z}))
                    signals.append(SignalData(symbol=ps.stock_b, direction="flat",
                                               strength=0.0, strategy_id=STRATEGY_ID,
                                               metadata={"pair": key, "leg": "exit_b", "z": z}))
                    ps.position = 0
                    ps.entry_zscore = 0.0
        return signals

    def on_bar(self, bar, portfolio) -> list[SignalData]:
        return []

    def on_fill(self, fill, portfolio) -> None:
        return None
