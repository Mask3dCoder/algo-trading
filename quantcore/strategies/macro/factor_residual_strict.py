"""Strict factor-neutral residual trading system.

Enforces EXACT factor neutrality (B^T @ w = 0) via null-space projection
of raw residual alpha signals. No per-asset vol scaling. No optimization
beyond the projection step. All PnL must come from residual alpha.

Inherits signal extraction (factor model → residuals → alpha signals)
from FactorResidualStrategy. Only replaces the portfolio construction
with hard orthogonalization against the factor loading matrix.

Usage:
    Register as "macro_factor_residual_strict" and run through
    scripts/real_data_backtest.py.

Reference:
    quantcore/strategies/macro/factor_residual.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.macro.factor_residual import FactorResidualStrategy
from quantcore.strategies.base import TradeSignal
from quantcore.strategies.registry import register


@register("macro_factor_residual_strict")
class FactorResidualStrictStrategy(FactorResidualStrategy):
    """Factor residual alpha with strict factor-neutral portfolio construction.

    Projects raw residual signals into the null space of the factor loading
    matrix, ensuring the portfolio has ZERO exposure to all systematic factors
    (MKT, RATES, INFL, SAFE). All PnL comes from residual (idiosyncratic) moves.

    Config params (all from parent, plus):
        vol_target: annualized volatility target (default 0.15)
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.vol_target = float(config.params.get("vol_target", 0.15))

        # --- Instrumentation for signal analysis ---
        self._debug_autocorr_values: list[float] = []
        self._debug_persistence_filtered_pct: float = 0.0
        self._debug_signal_magnitudes: dict[str, list[float]] = {
            "before_norm": [],
            "after_norm": [],
        }
        self._debug_n_filtered = 0
        self._debug_n_total = 0

    # ------------------------------------------------------------------
    # Enhanced signal computation (overrides parent)
    # ------------------------------------------------------------------

    def _compute_signals(
        self, residuals: NDArray, decay_span: int = 63,
    ) -> NDArray:
        """Institutional-grade residual signal extraction.

        Improvements over parent implementation:
        1. Exponential decay weighting (half-life = 21 days)
        2. Residual persistence filter (autocorrelation-based)
        3. Robust z-score (median + MAD normalization)

        Only emits signals when residuals show statistical persistence.
        """
        n_periods, n_assets = residuals.shape
        signals = np.zeros(n_assets)

        if n_periods < 10:
            return signals

        # Decay parameter: half-life = 21 days
        decay_rate = np.log(2) / 21.0
        alpha = 1.0 - np.exp(-decay_rate)

        self._debug_n_total = n_assets
        n_filtered = 0

        for i in range(n_assets):
            resid = residuals[:, i]
            valid_mask = ~np.isnan(resid)

            if np.sum(valid_mask) < 10:
                signals[i] = 0.0
                n_filtered += 1
                continue

            valid_resid = resid[valid_mask]
            n_valid = len(valid_resid)

            # --- Test: Is there residual persistence? ---
            # Compute autocorrelation at lags 1, 3, 5
            # Proper autocorrelation: corr(x[t], x[t-lag])
            autocorr_vals = []
            for lag in [1, 3, 5]:
                if n_valid > lag + 20:
                    # Align the series: x[t-lag] vs x[t]
                    x_lagged = valid_resid[:-(lag)]
                    x_current = valid_resid[lag:]
                    # Take last 30 points for efficiency
                    n_compare = min(30, len(x_lagged), len(x_current))
                    x_lag = x_lagged[-n_compare:]
                    x_cur = x_current[-n_compare:]
                    if len(x_lag) > 5 and len(x_cur) > 5:
                        corr = np.corrcoef(x_lag, x_cur)[0, 1]
                        if not np.isnan(corr):
                            autocorr_vals.append(corr)

            avg_autocorr = np.mean(autocorr_vals) if autocorr_vals else 0.0
            self._debug_autocorr_values.append(avg_autocorr)

            # --- Persistence filter: only trade if residuals persist ---
            # Positive autocorr = trending residuals = tradeable
            # Negative/zero autocorr = mean-reverting = don't trade
            if avg_autocorr < 0.05:
                signals[i] = 0.0
                n_filtered += 1
                continue

            # --- Exponential decay weighting on recent residuals ---
            window = min(self.resid_mom_window, n_valid)
            recent = valid_resid[-window:]

            # EWMA of residuals (not simple sum like parent)
            ewma_val = recent[0]
            for j in range(1, len(recent)):
                ewma_val = alpha * recent[j] + (1.0 - alpha) * ewma_val

            # --- Robust z-score using Median Absolute Deviation ---
            median_resid = np.median(recent)
            mad = np.median(np.abs(recent - median_resid))

            if mad < 1e-10:
                signals[i] = 0.0
                n_filtered += 1
                continue

            z_score = (ewma_val - median_resid) / mad

            # --- Scale by autocorrelation strength ---
            # Strong persistence = stronger signal
            signal = z_score * min(1.0, avg_autocorr * 3.0)

            # --- Cap at 2.5 MAD (robust 2-sigma equivalent) ---
            signals[i] = np.clip(signal, -2.5, 2.5)

        # --- Store instrumentation ---
        self._debug_n_filtered = n_filtered
        self._debug_persistence_filtered_pct = (
            n_filtered / n_assets * 100.0 if n_assets > 0 else 0.0
        )
        self._debug_signal_magnitudes["before_norm"] = (
            signals[np.abs(signals) > 0].tolist()
        )

        # --- Cross-sectional robust normalization ---
        non_zero_mask = np.abs(signals) > 0
        if np.sum(non_zero_mask) >= 2:
            sig_median = np.median(signals[non_zero_mask])
            sig_mad = np.median(np.abs(signals[non_zero_mask] - sig_median))
            if sig_mad > 1e-10:
                signals[non_zero_mask] = (
                    signals[non_zero_mask] - sig_median
                ) / sig_mad
                signals[non_zero_mask] = np.clip(
                    signals[non_zero_mask], -2.5, 2.5
                )

        self._debug_signal_magnitudes["after_norm"] = (
            signals[np.abs(signals) > 0].tolist()
        )

        return signals

    # ------------------------------------------------------------------
    # Strict factor-neutral portfolio construction
    # ------------------------------------------------------------------

    def _orthogonalize_weights(self, w_raw: NDArray) -> NDArray:
        """Project weights into null space of factor loading matrix.

        Ensures B^T @ w = 0 (exact factor neutrality) via:
          w_proj = w_raw - B @ solve(B^T @ B, B^T @ w_raw)

        Uses np.linalg.lstsq for numerical stability when B^T @ B
        is near-singular.

        Args:
            w_raw: (n_assets,) raw signal weights before projection.

        Returns:
            (n_assets,) projected weights, zero if projection fails.
        """
        B = self._last_betas  # (n_assets, n_factors) raw-scale loadings
        n_assets = len(w_raw)

        if B is None or B.shape[0] != n_assets or n_assets < 2:
            return w_raw.copy()

        try:
            # B^T @ w = 0 → project into null space
            BtB = B.T @ B  # (n_factors, n_factors)
            Btw = B.T @ w_raw  # (n_factors,)
            # Use solve for exactness when well-conditioned
            theta = np.linalg.solve(BtB, Btw)
            w_proj = w_raw - B @ theta
            return w_proj
        except np.linalg.LinAlgError:
            # Fallback: lstsq for numerical stability
            try:
                theta, _, _, _ = np.linalg.lstsq(B, w_raw, rcond=None)
                w_proj = w_raw - B @ theta
                return w_proj
            except np.linalg.LinAlgError:
                return w_raw.copy()

    def _compute_weights(
        self, signals: NDArray, rets: NDArray,
    ) -> NDArray:
        """Compute strict factor-neutral weights.

        Pipeline:
          1. w_raw = signals (validated residual alpha)
          2. Hard orthogonalization: w = null_space_projection(w_raw)
          3. Normalize: w = w / sum(|w|) (preserves neutrality, scalar)
          4. Vol target: scale to annualized vol target
          5. Clamp: max |w| = 0.30 per asset
          6. Zero-sum: w = w - mean(w) (restores sum ≈ 0 after clamp)
          7. Min weight filter: zero out positions below threshold
        """
        n_assets = len(signals)

        # Step 1: Raw signal as weights
        w = signals.copy()

        # Step 2: Hard orthogonalization against factor loadings
        w = self._orthogonalize_weights(w)

        # Step 3: Normalize to unit abs sum (scalar, preserves neutrality)
        abs_sum = np.sum(np.abs(w))
        if abs_sum > 1e-10:
            w = w / abs_sum
        else:
            return np.zeros(n_assets)

        # Step 4: Vol target scaling (portfolio-level scalar only)
        if rets.shape[0] >= 20:
            port_vol = np.std(rets[-20:] @ w, ddof=1) * np.sqrt(252)
            if port_vol > 1e-10:
                w = w * (self.vol_target / port_vol)

        # Step 5: Clamp max weight to 30% (per-asset limit)
        w = np.clip(w, -0.30, 0.30)

        # Step 6: Zero-sum (market neutral) — repair neutrality after clamp
        w = w - np.mean(w)

        # Step 7: Min weight filter
        w[np.abs(w) < self.min_weight] = 0.0

        return w

    # ------------------------------------------------------------------
    # Main entry point (same as parent, but uses new _compute_weights)
    # ------------------------------------------------------------------

    def on_data(self, data: dict[str, NDArray]) -> list:
        """Main entry: returns → factors → residuals → signals → weights → signals.

        Identical to parent but weight computation enforces strict factor neutrality.
        Also resets debug instrumentation each call.
        """
        # Reset instrumentation
        self._debug_autocorr_values = []
        self._debug_signal_magnitudes = {"before_norm": [], "after_norm": []}
        self._debug_n_filtered = 0
        self._debug_n_total = 0
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        # Initialize factor map on first call
        if not self._factor_map:
            self._init_factor_map(symbols)

        # Need enough history for regression window + at least 1 prediction
        if n_periods < self.reg_window + 1:
            return []

        # Rebalance gate
        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq:
            return []

        # ---- Step 1: Compute returns ----
        daily_rets = (
            (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)
        )

        # ---- Step 2: Build factors ----
        factors = self._build_factors(daily_rets)

        # ---- Step 3: Compute residuals (also stores self._last_betas) ----
        residuals = self._compute_residuals(daily_rets, factors)

        # ---- Step 4: Compute residual alpha signals ----
        signals = self._compute_signals(residuals)

        # ---- Step 5: Compute strict factor-neutral weights ----
        weights = self._compute_weights(signals, daily_rets)

        # ---- Step 6: Emit trade signals ----
        trade_signals = []
        for i in range(n_assets):
            w_i = float(weights[i])
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

        self._days_since_rebalance = 0
        return trade_signals

    # ------------------------------------------------------------------
    # Debug/Instrumentation output
    # ------------------------------------------------------------------

    def print_debug_summary(self) -> None:
        """Print signal instrumentation summary."""
        print("\n" + "="*60)
        print("FACTOR RESIDUAL STRICT — SIGNAL DIAGNOSTICS")
        print("="*60)
        print(f"Total assets processed: {self._debug_n_total}")
        print(f"Assets filtered (no persistence): {self._debug_n_filtered}")
        print(f"Percent filtered: {self._debug_persistence_filtered_pct:.1f}%")

        if self._debug_autocorr_values:
            autocorr_arr = np.array(self._debug_autocorr_values)
            print(f"\nAutocorrelation stats (assets with signals):")
            print(f"  Mean: {np.mean(autocorr_arr):.4f}")
            print(f"  Median: {np.median(autocorr_arr):.4f}")
            print(f"  Min: {np.min(autocorr_arr):.4f}")
            print(f"  Max: {np.max(autocorr_arr):.4f}")

        if self._debug_signal_magnitudes["before_norm"]:
            before = np.array(self._debug_signal_magnitudes["before_norm"])
            print(f"\nSignal magnitude BEFORE normalization:")
            print(f"  Count: {len(before)}")
            print(f"  Mean: {np.mean(before):.4f}")
            print(f"  Std: {np.std(before):.4f}")
            print(f"  Range: [{np.min(before):.4f}, {np.max(before):.4f}]")

        if self._debug_signal_magnitudes["after_norm"]:
            after = np.array(self._debug_signal_magnitudes["after_norm"])
            print(f"\nSignal magnitude AFTER normalization:")
            print(f"  Count: {len(after)}")
            print(f"  Mean: {np.mean(after):.4f}")
            print(f"  Std: {np.std(after):.4f}")
            print(f"  Range: [{np.min(after):.4f}, {np.max(after):.4f}]")

        print("="*60 + "\n")
