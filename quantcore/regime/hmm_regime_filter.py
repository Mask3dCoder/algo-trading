"""HMM Regime Filter for dynamic strategy allocation.

Trains a 3-state Hidden Markov Model on macro features to classify the
market into regimes and adjust strategy allocation dynamically.

States:
  0: Low-vol bull (low VIX, tight spreads) -> full VRP allocation (1.0x)
  1: Moderate vol / transition -> reduced allocation (0.5x)
  2: High vol / crisis (VIX > 30, wide spreads) -> no VRP, cash (0.0x)

The filter uses the forward algorithm for online state probability
estimation (no look-ahead bias), making it suitable for both backtesting
and live trading.

Features used:
  - VIX level and daily change
  - SPY realized volatility (21-day)
  - Credit spread proxy (HYG/SPY ratio change)
  - Yield curve slope proxy (TLT/SPY ratio)

Usage:
    from quantcore.regime.hmm_regime_filter import HMMRegimeFilter

    filt = HMMRegimeFilter()
    filt.train(vix, spy_close, hyg_close, tlt_close, dates)
    scalar = filt.allocation_scalar(date)  # 1.0, 0.5, or 0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegimeFilterConfig:
    """Configuration for HMM regime filter."""
    n_states: int = 3
    covar_type: Literal["full", "diag"] = "diag"
    n_restarts: int = 5
    # Allocation scalars per state (indexed by sorted-vol order)
    # After training, states are sorted by average vol:
    #   state 0 = lowest vol -> highest allocation
    #   state N-1 = highest vol -> lowest allocation
    allocation_scalars: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.0])
    # VIX thresholds for state labeling (used for reporting, not filtering)
    vix_low_threshold: float = 18.0
    vix_crisis_threshold: float = 30.0
    # Minimum training samples
    min_train_samples: int = 252


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def build_regime_features(
    vix: pd.Series,
    spy_close: pd.Series,
    hyg_close: pd.Series | None = None,
    tlt_close: pd.Series | None = None,
    vol_lookback: int = 21,
) -> pd.DataFrame:
    """Build feature matrix for HMM regime detection.

    Features:
      1. VIX level (normalized)
      2. VIX daily change
      3. SPY realized vol (21-day rolling)
      4. Credit spread proxy: HYG/SPY ratio change (if available)
      5. Yield curve proxy: TLT/SPY ratio change (if available)

    All features are z-scored for numerical stability.

    Parameters
    ----------
    vix : pd.Series
        VIX close values (index = DatetimeIndex).
    spy_close : pd.Series
        SPY close prices.
    hyg_close : pd.Series | None
        HYG close prices (credit spread proxy).
    tlt_close : pd.Series | None
        TLT close prices (yield curve proxy).
    vol_lookback : int
        Rolling window for realized vol (default 21).

    Returns
    -------
    pd.DataFrame
        Feature matrix with same index as input.
    """
    # Align all series
    idx = spy_close.index
    vix_aligned = vix.reindex(idx, method="ffill")

    features = pd.DataFrame(index=idx)

    # Feature 1: VIX level
    features["vix_level"] = vix_aligned

    # Feature 2: VIX daily change
    features["vix_change"] = vix_aligned.diff()

    # Feature 3: SPY realized vol (21-day rolling)
    spy_ret = np.log(spy_close / spy_close.shift(1))
    features["spy_rvol"] = spy_ret.rolling(vol_lookback).std() * np.sqrt(252)

    # Feature 4: Credit spread proxy (HYG/SPY ratio)
    if hyg_close is not None:
        hyg_aligned = hyg_close.reindex(idx, method="ffill")
        ratio = hyg_aligned / spy_close
        features["credit_proxy"] = ratio.pct_change(5)  # 5-day change
    else:
        # Fallback: use VIX change as credit proxy
        features["credit_proxy"] = vix_aligned.pct_change(5)

    # Feature 5: Yield curve proxy (TLT/SPY ratio)
    if tlt_close is not None:
        tlt_aligned = tlt_close.reindex(idx, method="ffill")
        ratio = tlt_aligned / spy_close
        features["yc_proxy"] = ratio.pct_change(5)
    else:
        # Fallback: use VIX term structure proxy
        features["yc_proxy"] = -vix_aligned.pct_change(21)

    # Drop warmup rows
    features = features.dropna()

    # Z-score each feature (rolling to avoid look-ahead)
    for col in features.columns:
        expanding_mean = features[col].expanding(min_periods=60).mean()
        expanding_std = features[col].expanding(min_periods=60).std()
        expanding_std = expanding_std.replace(0, 1)
        features[col] = (features[col] - expanding_mean) / expanding_std

    # Clip extreme values
    features = features.clip(-5, 5)

    return features


# ---------------------------------------------------------------------------
# State labeling
# ---------------------------------------------------------------------------


@dataclass
class StateLabel:
    """Label for an HMM state."""
    state_idx: int
    label: str  # "low_vol_bull", "moderate", "crisis"
    allocation_scalar: float  # 1.0, 0.5, 0.0
    avg_vix: float
    avg_rvol: float
    occupancy_pct: float
    avg_duration_days: float


def _label_states(
    states: NDArray,
    features: pd.DataFrame,
    vix: pd.Series,
    config: RegimeFilterConfig,
) -> list[StateLabel]:
    """Label HMM states by their characteristics.

    Sorts states by average realized vol (ascending) and assigns labels:
      - Lowest vol state -> low_vol_bull -> allocation_scalar[0]
      - Middle vol state(s) -> moderate -> allocation_scalar[1]
      - Highest vol state -> crisis -> allocation_scalar[-1]
    """
    n_states = config.n_states
    labels = []

    # Compute per-state statistics
    state_vols = []
    for s in range(n_states):
        mask = states == s
        if mask.sum() == 0:
            state_vols.append(0.0)
            continue

        # Get VIX values for this state
        vix_vals = vix.reindex(features.index, method="ffill")[mask]
        avg_vix = float(vix_vals.mean()) if len(vix_vals) > 0 else 20.0

        # Get realized vol for this state
        rvol_vals = features["spy_rvol"][mask] if "spy_rvol" in features.columns else pd.Series([0.20])
        avg_rvol = float(rvol_vals.mean()) if len(rvol_vals) > 0 else 0.20

        state_vols.append(avg_rvol)

    # Sort states by average vol (ascending)
    sorted_states = np.argsort(state_vols)

    for rank, s in enumerate(sorted_states):
        mask = states == s
        count = mask.sum()

        # Compute occupancy
        occupancy = float(count / len(states) * 100)

        # Compute average duration
        runs = np.diff(np.where(np.concatenate(([False], states == s, [False])))[0])
        avg_dur = float(np.mean(runs)) if len(runs) > 0 else 0.0

        # VIX and vol stats
        vix_vals = vix.reindex(features.index, method="ffill")[mask]
        avg_vix = float(vix_vals.mean()) if len(vix_vals) > 0 else 20.0

        rvol_vals = features["spy_rvol"][mask] if "spy_rvol" in features.columns else pd.Series([0.20])
        avg_rvol = float(rvol_vals.mean()) if len(rvol_vals) > 0 else 0.20

        # Assign label and scalar based on rank
        if rank == 0:
            label = "low_vol_bull"
            scalar = config.allocation_scalars[0]
        elif rank == len(sorted_states) - 1:
            label = "crisis"
            scalar = config.allocation_scalars[-1]
        else:
            label = "moderate"
            scalar_idx = min(rank, len(config.allocation_scalars) - 1)
            scalar = config.allocation_scalars[scalar_idx]

        labels.append(StateLabel(
            state_idx=int(s),
            label=label,
            allocation_scalar=scalar,
            avg_vix=avg_vix,
            avg_rvol=avg_rvol,
            occupancy_pct=occupancy,
            avg_duration_days=avg_dur,
        ))

    return labels


# ---------------------------------------------------------------------------
# Main filter class
# ---------------------------------------------------------------------------


class HMMRegimeFilter:
    """HMM-based regime filter for dynamic strategy allocation.

    Trains a 3-state HMM on macro features and provides:
      - Daily state probabilities (forward algorithm, no look-ahead)
      - Allocation scalar (1.0, 0.5, 0.0) for strategy weighting
      - State labels for reporting

    Usage:
        filt = HMMRegimeFilter()
        filt.train(vix, spy_close, hyg_close, tlt_close, dates)
        scalar = filt.allocation_scalar(date)
    """

    def __init__(self, config: RegimeFilterConfig | None = None) -> None:
        self._cfg = config or RegimeFilterConfig()
        self._trained = False
        self._hmm_result = None
        self._state_labels: list[StateLabel] = []
        self._features: pd.DataFrame | None = None
        self._vix: pd.Series | None = None
        # Mapping from state index -> allocation scalar
        self._state_to_scalar: dict[int, float] = {}

    def train(
        self,
        vix: pd.Series,
        spy_close: pd.Series,
        hyg_close: pd.Series | None = None,
        tlt_close: pd.Series | None = None,
        train_start: str | None = None,
        train_end: str | None = None,
    ) -> None:
        """Train the HMM on historical data.

        Parameters
        ----------
        vix : pd.Series
            VIX close values (index = DatetimeIndex).
        spy_close : pd.Series
            SPY close prices.
        hyg_close : pd.Series | None
            HYG close prices (credit spread proxy).
        tlt_close : pd.Series | None
            TLT close prices (yield curve proxy).
        train_start : str | None
            Training start date. If None, uses all available data.
        train_end : str | None
            Training end date. If None, uses all available data.
        """
        from quantcore.regime.hmm_engine import fit_hmm

        # Build features for FULL period (train + test)
        all_features = build_regime_features(vix, spy_close, hyg_close, tlt_close)
        self._vix = vix

        # Slice training data for HMM fitting
        train_features = all_features.copy()
        if train_start:
            train_features = train_features[train_features.index >= pd.Timestamp(train_start)]
        if train_end:
            train_features = train_features[train_features.index <= pd.Timestamp(train_end)]

        if len(train_features) < self._cfg.min_train_samples:
            raise ValueError(
                f"Insufficient training data: {len(train_features)} < {self._cfg.min_train_samples}"
            )

        # Fit HMM on training data only
        X_train = train_features.values.astype(np.float64)
        X_train = np.nan_to_num(X_train, nan=0.0)

        _log.info(
            "Training HMM: %d samples, %d features, %d states",
            X_train.shape[0], X_train.shape[1], self._cfg.n_states,
        )

        self._hmm_result = fit_hmm(
            X_train,
            n_states=self._cfg.n_states,
            covar_type=self._cfg.covar_type,
            n_restarts=self._cfg.n_restarts,
        )

        # Compute FORWARD-ONLY probabilities on FULL period (no look-ahead)
        # This ensures test period dates get proper state probabilities
        from quantcore.regime.hmm_engine import predict_online

        X_full = all_features.values.astype(np.float64)
        X_full = np.nan_to_num(X_full, nan=0.0)
        self._features = all_features

        T, N = X_full.shape[0], self._hmm_result.params.n_states
        forward_probs = np.zeros((T, N))

        # Initialize with prior
        log_alpha_prev = np.log(np.maximum(self._hmm_result.params.pi, 1e-300))
        _, state_probs_0, log_alpha_prev = predict_online(
            X_full[0], log_alpha_prev, self._hmm_result.params,
        )
        forward_probs[0] = state_probs_0

        for t in range(1, T):
            _, state_probs, log_alpha_prev = predict_online(
                X_full[t], log_alpha_prev, self._hmm_result.params,
            )
            forward_probs[t] = state_probs

        self._forward_probs = forward_probs

        # Label states using training data only
        train_idx = all_features.index.get_indexer(train_features.index)
        train_states = self._hmm_result.states
        self._state_labels = _label_states(
            train_states,
            train_features,
            vix,
            self._cfg,
        )

        # Build state -> scalar mapping
        self._state_to_scalar = {}
        for sl in self._state_labels:
            self._state_to_scalar[sl.state_idx] = sl.allocation_scalar

        self._trained = True

        _log.info("HMM trained. States:")
        for sl in self._state_labels:
            _log.info(
                "  State %d (%s): VIX=%.1f, RVol=%.1f%%, scalar=%.1f, occ=%.1f%%, dur=%.1f days",
                sl.state_idx, sl.label, sl.avg_vix, sl.avg_rvol * 100,
                sl.allocation_scalar, sl.occupancy_pct, sl.avg_duration_days,
            )

    def state_probabilities(self, date: pd.Timestamp) -> dict[int, float]:
        """Get state probabilities for a given date.

        Uses forward-only probabilities (no look-ahead bias).

        The forward algorithm only uses data up to time t, making it
        suitable for backtesting and live trading.

        Parameters
        ----------
        date : pd.Timestamp
            Date to get probabilities for.

        Returns
        -------
        dict[int, float]
            Mapping from state index to probability.
        """
        if not self._trained or self._hmm_result is None:
            raise RuntimeError("Filter not trained. Call train() first.")

        # Find the index for this date
        if self._features is None:
            raise RuntimeError("No features available.")

        # Find closest date
        idx = self._features.index.get_indexer([date], method="ffill")[0]
        if idx < 0 or idx >= len(self._features):
            return {s: 1.0 / self._cfg.n_states for s in range(self._cfg.n_states)}

        # Use forward-only probabilities (no look-ahead)
        if hasattr(self, '_forward_probs') and self._forward_probs is not None:
            probs = self._forward_probs[idx]
        else:
            # Fallback to smoothed (should not happen after training)
            probs = self._hmm_result.state_probs[idx]

        return {s: float(probs[s]) for s in range(self._cfg.n_states)}

    def allocation_scalar(self, date: pd.Timestamp) -> float:
        """Get the allocation scalar for a given date.

        The scalar is the expected allocation multiplier based on state
        probabilities: scalar = sum(p_state * scalar_state).

        This provides a smooth transition between regimes rather than
        a hard switch.

        Parameters
        ----------
        date : pd.Timestamp
            Date to get scalar for.

        Returns
        -------
        float
            Allocation scalar in [0.0, 1.0].
        """
        probs = self.state_probabilities(date)
        scalar = sum(
            p * self._state_to_scalar.get(s, 0.5)
            for s, p in probs.items()
        )
        return float(np.clip(scalar, 0.0, 1.0))

    def most_likely_state(self, date: pd.Timestamp) -> tuple[int, str, float]:
        """Get the most likely state for a given date.

        Returns
        -------
        tuple[int, str, float]
            (state_index, label, probability)
        """
        probs = self.state_probabilities(date)
        best_state = max(probs, key=probs.get)
        label = "unknown"
        for sl in self._state_labels:
            if sl.state_idx == best_state:
                label = sl.label
                break
        return best_state, label, probs[best_state]

    def get_state_labels(self) -> list[StateLabel]:
        """Get all state labels with their characteristics."""
        return list(self._state_labels)

    def get_daily_scalars(self) -> pd.Series:
        """Get allocation scalars for all training dates.

        Uses forward-only probabilities (no look-ahead).

        Returns
        -------
        pd.Series
            Daily allocation scalars (index = dates).
        """
        if not self._trained or self._features is None:
            raise RuntimeError("Filter not trained.")

        # Use forward-only probabilities
        probs_source = (
            self._forward_probs
            if hasattr(self, '_forward_probs') and self._forward_probs is not None
            else self._hmm_result.state_probs
        )

        scalars = []
        for i in range(len(self._features)):
            probs = probs_source[i]
            scalar = sum(
                p * self._state_to_scalar.get(s, 0.5)
                for s, p in enumerate(probs)
            )
            scalars.append(np.clip(scalar, 0.0, 1.0))

        return pd.Series(scalars, index=self._features.index, name="allocation_scalar")

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def n_states(self) -> int:
        return self._cfg.n_states


# ---------------------------------------------------------------------------
# Integration with MultiStrategyAggregator
# ---------------------------------------------------------------------------


class RegimeWeightedAggregator:
    """Wrapper that applies HMM regime scalars to strategy weights.

    Takes a MultiStrategyAggregator and an HMMRegimeFilter, and
    multiplies VRP strategy weights by the regime allocation scalar.

    Usage:
        agg = MultiStrategyAggregator(subs, config)
        regime_agg = RegimeWeightedAggregator(agg, hmm_filter, vrp_strategy_ids)
        signals = regime_agg.on_data(bar_data)
    """

    def __init__(
        self,
        aggregator,
        regime_filter: HMMRegimeFilter,
        vrp_strategy_ids: list[str],
    ) -> None:
        self._agg = aggregator
        self._filter = regime_filter
        self._vrp_ids = set(vrp_strategy_ids)
        self._last_scalar = 1.0

    def on_data(self, data: dict) -> list:
        """Forward to aggregator with regime-adjusted weights."""
        # Get current date from data
        timestamp = data.get("timestamp")
        if timestamp is not None:
            date = pd.Timestamp(timestamp)
            scalar = self._filter.allocation_scalar(date)
        else:
            scalar = 1.0

        self._last_scalar = scalar

        # Adjust VRP strategy weights
        for sub in self._agg._sub_strategies:
            if sub.strategy_id in self._vrp_ids:
                # The aggregator will recompute weights, so we adjust
                # the sub-strategy's enabled flag based on scalar
                sub.enabled = scalar > 0.01

        # Run aggregator
        signals = self._agg.on_data(data)

        # Scale signal sizes for VRP strategies
        if scalar < 0.99:
            adjusted = []
            for sig in signals:
                source = sig.metadata.get("source", "")
                if source in self._vrp_ids:
                    # Scale down the signal
                    adjusted.append(type(sig)(
                        symbol=sig.symbol,
                        direction=sig.direction,
                        strength=sig.strength * scalar,
                        size=sig.size * scalar,
                        metadata={**sig.metadata, "regime_scalar": scalar},
                    ))
                else:
                    adjusted.append(sig)
            return adjusted

        return signals

    @property
    def last_scalar(self) -> float:
        return self._last_scalar
