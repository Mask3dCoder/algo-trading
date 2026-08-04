"""Volatility-targeted cross-sectional momentum strategy (native QuantCore).

Generated end-to-end from QuantCore's own building blocks -- no external LLM
runtime, no third-party strategy-generation service. This file is the proof
artifact for DECISION_LOG_vibe_trading.md and the subject of the parameter
sweep in scripts/sweep_vol_target_momentum.py.

Design
------
* Momentum signal: trailing total return per symbol, cross-sectionally
  z-scored. Long the highest-z names; short the lowest-z names.
* Vol targeting: a single ``VolTarget`` tracker (from
  ``quantcore.risk.vol_target``) scales gross exposure toward a target
  annualized volatility using a rolling ``vol_window`` estimate.
* Point-in-time correctness: the strategy only ever reads ``snapshot.bars``.
  ``BacktestEngine`` calls ``PortfolioState.mark_to_market(snapshot)`` and
  enforces a strict look-ahead barrier (engine.py:981) *before* this method
  runs. The optional ``regime_filter`` likewise reads only data at or before
  ``as_of`` (see ``quantcore.alpha.regime_filter``).
* Rebalance cadence: signals are only (re)emitted every ``rebalance_frequency``
  bars to cut turnover/noise.
* Regime gate: when a ``regime_filter`` reports risk-off, the strategy flattens
  all held positions (``direction="flat"``) -- never trades through a panic.

This is the capability Vibe-Trading advertises (natural-language -> strategy)
*minus* the epistemic hazard: an LLM "committee" would emit code whose latent
training-data priors already encode ex-post outcomes. Here the logic is
explicit, deterministic, and auditable.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from quantcore.alpha.regime_filter import RegimeFilter
from quantcore.backtest.engine import SignalData
from quantcore.risk.vol_target import VolTarget, VolTargetConfig

_log = logging.getLogger(__name__)

STRATEGY_ID = "vol_target_momentum"


class VolTargetMomentumStrategy:
    """Cross-sectional momentum with volatility-targeted, regime-gated leverage.

    Parameters
    ----------
    momentum_lookback:
        Trailing window (bars) for the momentum return.
    min_history:
        Minimum bars of history before any signal is emitted.
    n_select:
        Number of names per side (long / short).
    base_notional:
        Gross-notional target at leverage == 1.0 (scaled by the vol target).
    target_vol:
        Target annualized volatility for ``VolTarget``.
    vol_window:
        Rolling window (bars) for realized-vol estimation.
    rebalance_frequency:
        Emit signals only every N bars (turnover control).
    regime_filter:
        Optional ``RegimeFilter``; risk-off => flatten.
    """

    def __init__(
        self,
        *,
        momentum_lookback: int = 60,
        min_history: int = 120,
        n_select: int = 5,
        base_notional: float = 500_000.0,
        target_vol: float = 0.12,
        vol_window: int = 20,
        rebalance_frequency: int = 5,
        regime_filter: RegimeFilter | None = None,
    ) -> None:
        self._lookback = momentum_lookback
        self._min_history = min_history
        self._n_select = n_select
        self._base_notional = base_notional
        self._rebal_freq = max(1, int(rebalance_frequency))
        self._regime = regime_filter
        self._prices: dict[str, list[float]] = defaultdict(list)
        self._vol = VolTarget(
            VolTargetConfig(
                target_vol=target_vol,
                short_window=vol_window,
                min_leverage=0.5,
                max_leverage=2.0,
            )
        )
        self._prev_equity: float | None = None
        self._bar = 0

    # -- strategy interface -------------------------------------------------
    def on_snapshot(self, snapshot, portfolio) -> list[SignalData]:
        self._bar += 1

        # 1) Update price history + vol tracker from THIS timestamp only.
        for sym, bar in snapshot.bars.items():
            hist = self._prices[sym]
            hist.append(float(bar.close))
            cap = self._min_history + self._lookback + 8
            if len(hist) > cap:
                del hist[: len(hist) - cap]

        equity = float(portfolio.total_value)
        if self._prev_equity and self._prev_equity > 0:
            self._vol.update((equity - self._prev_equity) / self._prev_equity)
        self._prev_equity = equity

        # 2) Regime gate: risk-off => flatten everything held, trade nothing.
        if self._regime is not None and not self._regime.is_risk_on(snapshot.timestamp):
            held = getattr(portfolio, "positions", {}) or {}
            return [
                SignalData(
                    symbol=s,
                    direction="flat",
                    strength=0.0,
                    strategy_id=STRATEGY_ID,
                )
                for s in held
            ]

        # 3) Rebalance cadence.
        if self._bar % self._rebal_freq != 0:
            return []

        # 4) Momentum scores (trailing return), z-scored across the universe.
        scores: dict[str, float] = {}
        for sym, hist in self._prices.items():
            if len(hist) < self._min_history + self._lookback:
                continue
            past, now = hist[-(self._lookback + 1)], hist[-1]
            if past > 0:
                scores[sym] = now / past - 1.0

        if len(scores) < 2 * self._n_select:
            return []

        arr = np.array(list(scores.values()), dtype=float)
        mu, sd = arr.mean(), arr.std(ddof=0)
        if sd <= 1e-12:
            return []
        z = {s: (v - mu) / sd for s, v in scores.items()}

        ranked = sorted(z.items(), key=lambda kv: kv[1])
        shorts = ranked[: self._n_select]
        longs = ranked[-self._n_select:]

        lev = self._vol.compute_leverage()
        per_side_notional = self._base_notional * lev / (2.0 * self._n_select)

        signals: list[SignalData] = []
        for sym, zv in longs:
            signals.append(
                SignalData(
                    symbol=sym,
                    direction="long",
                    strength=float(np.clip(abs(zv), 0.0, 3.0)),
                    strategy_id=STRATEGY_ID,
                    notional_size=per_side_notional,
                    metadata={"z": round(float(zv), 4), "leverage": round(float(lev), 3)},
                )
            )
        for sym, zv in shorts:
            signals.append(
                SignalData(
                    symbol=sym,
                    direction="short",
                    strength=float(np.clip(abs(zv), 0.0, 3.0)),
                    strategy_id=STRATEGY_ID,
                    notional_size=per_side_notional,
                    metadata={"z": round(float(zv), 4), "leverage": round(float(lev), 3)},
                )
            )
        return signals

    def on_bar(self, bar, portfolio) -> list[SignalData]:
        # Snapshot-aware strategy; the legacy single-bar path is unused.
        return []

    def on_fill(self, fill, portfolio) -> None:
        # Stateless within a snapshot; no reaction to fills required.
        return None
