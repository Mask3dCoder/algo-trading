"""QP-optimized factor-neutral residual strategy.

Uses constrained QP solver for EXACT simultaneous satisfaction of:
  - Factor neutrality: B^T w = 0
  - Market neutrality: sum(w) = 0
  - Position limits: |w_i| <= 0.30

No post-processing, no clipping, no filtering after the solver.
All PnL must come from residual alpha.

Inherits signal extraction from FactorResidualStrategy.
Only replaces portfolio construction with the QP optimizer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.macro.factor_residual import FactorResidualStrategy
from quantcore.strategies.macro.optimizer import (
    solve_factor_neutral_weights,
    scale_to_target_vol,
    enforce_daily_neutrality,
    check_constraints,
)
from quantcore.strategies.base import TradeSignal
from quantcore.strategies.registry import register


@register("macro_factor_residual_qp")
class FactorResidualQPStrategy(FactorResidualStrategy):
    """Factor residual alpha with QP-optimized strict factor neutrality.

    Solves the constrained QP on every rebalance:
        min_w  0.5 * ||w - alpha||^2
        s.t.   B^T w = 0, sum(w) = 0, |w_i| <= max_weight

    Applies daily drift correction when max|B^T w| > threshold.
    Applies scalar vol scaling (preserves all constraints).
    NO post-processing after the solver.

    Config params (all from parent, plus):
        vol_target: annualized volatility target (default 0.15)
        neutrality_threshold: max |B^T w| before daily correction (default 0.005)
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.vol_target = float(config.params.get("vol_target", 0.15))
        self.neutrality_threshold = float(
            config.params.get("neutrality_threshold", 0.005)
        )
        self.max_weight = float(config.params.get("max_weight", 0.30))

        # Tracking state
        self._current_weights: NDArray | None = None
        self._prev_weights: NDArray | None = None
        self._n_assets: int = 0
        self._symbols: list[str] = []

        # Diagnostic accumulators
        self._diag_steps: int = 0
        self._diag_daily_corrections: int = 0
        self._diag_exposure_history: list[float] = []
        self._diag_sum_history: list[float] = []
        self._diag_turnover_history: list[float] = []
        self._diag_distortion_history: list[float] = []

    # ------------------------------------------------------------------
    # Portfolio construction via QP solver
    # ------------------------------------------------------------------

    def _compute_weights(
        self, signals: NDArray, rets: NDArray,
    ) -> NDArray:
        """Compute weights via constrained QP solver.

        Pipeline:
          1. Solve QP: min 0.5||w - signals||^2 s.t. all constraints
          2. Scalar vol scaling (preserves all constraints)
          3. NO post-processing, NO clipping, NO filtering

        Args:
            signals: (n_assets,) residual alpha signals.
            rets: (n_bars, n_assets) daily returns for vol estimation.

        Returns:
            (n_assets,) optimal constraint-satisfying weights.
        """
        B = self._last_betas
        n_assets = len(signals)

        if B is None or B.shape[0] != n_assets or n_assets < 2:
            return signals.copy()

        # Step 1: Solve constrained QP
        w = solve_factor_neutral_weights(
            alpha=signals,
            B=B,
            max_weight=self.max_weight,
        )

        # Step 2: Scalar vol scaling (preserves all constraints exactly)
        if rets.shape[0] >= self.vol_lookback + 2:
            w = scale_to_target_vol(
                w, rets,
                target_vol=self.vol_target,
                lookback=self.vol_lookback,
            )

        return w

    # ------------------------------------------------------------------
    # Main entry point with daily drift correction
    # ------------------------------------------------------------------

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Main entry: signals -> QP weights -> daily drift correction -> emissions.

        On rebalance days:
          - Full QP solve for new weights
        Between rebalances:
          - Check factor drift daily, correct if exceeded

        Args:
            data: dict with keys 'close' (n_periods, n_assets) and 'symbols'.

        Returns:
            List of TradeSignal for assets with meaningful weight changes.
        """
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        # Store symbols for diagnostics
        if not self._symbols:
            self._symbols = symbols
            self._n_assets = n_assets

        # Initialize factor map on first call
        if not self._factor_map:
            self._init_factor_map(symbols)

        # Need enough history for regression window + at least 1 prediction
        if n_periods < self.reg_window + 1:
            return []

        # ---- Compute returns ----
        daily_rets = (
            (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)
        )

        # ---- Build factors ----
        factors = self._build_factors(daily_rets)

        # ---- Compute residuals (also stores self._last_betas) ----
        residuals = self._compute_residuals(daily_rets, factors)

        # ---- Compute residual alpha signals ----
        signals = self._compute_signals(residuals)

        # ---- Update tracking state ----
        self._prev_weights = self._current_weights.copy() if self._current_weights is not None else None

        # ---- Rebalance gate ----
        self._days_since_rebalance += 1
        is_rebalance_day = self._days_since_rebalance >= self.rebalance_freq

        if is_rebalance_day:
            # Full QP solve
            self._current_weights = self._compute_weights(signals, daily_rets)
            self._days_since_rebalance = 0
        else:
            # Keep current weights if we have them
            if self._current_weights is not None:
                self._current_weights = self._current_weights.copy()
            else:
                self._current_weights = self._compute_weights(signals, daily_rets)

        # ---- Daily drift correction (every bar, not just rebalance) ----
        if self._current_weights is not None and self._last_betas is not None:
            corrected = enforce_daily_neutrality(
                self._current_weights,
                self._last_betas,
                threshold=self.neutrality_threshold,
                max_weight=self.max_weight,
            )
            if corrected is not self._current_weights:
                # Correction was applied
                self._current_weights = corrected
                self._diag_daily_corrections += 1

        # ---- Diagnostics ----
        if self._current_weights is not None and self._last_betas is not None:
            diag = check_constraints(
                self._current_weights, self._last_betas,
                max_weight=self.max_weight,
            )
            self._diag_exposure_history.append(diag["max_factor_exposure"])
            self._diag_sum_history.append(diag["sum_violation"])
            self._diag_steps += 1

            # Turnover vs previous weights
            if self._prev_weights is not None and len(self._prev_weights) == len(self._current_weights):
                turnover = float(np.sum(np.abs(self._current_weights - self._prev_weights)))
                self._diag_turnover_history.append(turnover)

            # Distortion: ||w - signals||
            distortion = float(np.sqrt(np.sum((self._current_weights - signals) ** 2)))
            self._diag_distortion_history.append(distortion)

        # ---- Emit trade signals ----
        trade_signals: list[TradeSignal] = []
        if self._current_weights is None:
            return []

        for i in range(n_assets):
            w_i = float(self._current_weights[i])
            if abs(w_i) < self.min_weight:
                continue

            trade_signals.append(TradeSignal(
                symbol=symbols[i],
                direction=1 if w_i > 0 else -1,
                strength=abs(w_i),
                size=abs(w_i) * float(self.config.max_position_notional),
                metadata={
                    "weight": round(w_i, 4),
                    "signal": round(float(signals[i]), 4),
                    "residual": round(float(residuals[-1, i]), 6)
                    if not np.isnan(residuals[-1, i]) else 0.0,
                },
            ))

        return trade_signals

    # ------------------------------------------------------------------
    # Diagnostic summary
    # ------------------------------------------------------------------

    def report_diagnostics(self) -> dict:
        """Return accumulated diagnostic summary."""
        if self._diag_steps == 0:
            return {"status": "no_data"}

        exposure_arr = np.array(self._diag_exposure_history)
        sum_arr = np.array(self._diag_sum_history)
        turnover_arr = np.array(self._diag_turnover_history) if self._diag_turnover_history else np.zeros(1)
        distortion_arr = np.array(self._diag_distortion_history) if self._diag_distortion_history else np.zeros(1)

        constraint_violations = np.sum(exposure_arr > self.neutrality_threshold)
        violation_rate = constraint_violations / self._diag_steps

        return {
            "total_steps": self._diag_steps,
            "daily_corrections": self._diag_daily_corrections,
            "max_factor_exposure": float(np.max(exposure_arr)),
            "mean_factor_exposure": float(np.mean(exposure_arr)),
            "std_factor_exposure": float(np.std(exposure_arr, ddof=1)),
            "max_sum_violation": float(np.max(sum_arr)),
            "mean_turnover": float(np.mean(turnover_arr)),
            "max_turnover": float(np.max(turnover_arr)),
            "mean_distortion": float(np.mean(distortion_arr)),
            "max_distortion": float(np.max(distortion_arr)),
            "constraint_violation_rate": float(violation_rate),
            "all_passed_rate": float(
                np.mean([
                    e < self.neutrality_threshold and s < 1e-6
                    for e, s in zip(self._diag_exposure_history, self._diag_sum_history)
                ])
            ),
        }
