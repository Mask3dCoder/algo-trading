"""Factor decomposition residual signal extraction engine with ML.

FIXED: Skip factor assets. Uses ML to predict FUTURE residuals.
This is a signal IMPROVEMENT, not a new strategy.

Pipeline:
  1. Rolling 252d ridge regression → residuals
  2. ML predictor: predict next residual from past residuals + features
  3. Signal = predicted_residual / std(residuals)
  4. Simple z-score weighting, skip factor assets
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, StrategyStatus, TradeSignal
from quantcore.strategies.registry import register

# Import ML predictor
try:
    from quantcore.signals.ml.residual_predictor import ResidualPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# ---------------------------------------------------------------------------
# DEPRECATION NOTICE
# ---------------------------------------------------------------------------
# macro_factor_residual is DISABLED pending rebuild.  The FACTOR_ASSETS set
# causes the strategy to degenerate to a single-asset bet on IEF because 4 of
# 5 universe assets are classified as factors and skipped.  The residual
# momentum framework is being rebuilt from scratch in
# quantcore/alpha/residual_momentum.py (Q2 plan).
# ---------------------------------------------------------------------------

# Factor assets to EXCLUDE from trading (STRUCTURAL BUG: too few non-factor
# assets remain when universe = {SPY, TLT, DBC, GLD, IEF})
FACTOR_ASSETS = {"SPY", "TLT", "DBC", "GLD"}


@register("macro_factor_residual")
class FactorResidualStrategy(Strategy):
    """Residual signal → long-short, ML-enhanced.

    FIXED: Skips factor assets. Uses ML to predict next residual.

    Config params:
        rebalance_freq: days between rebalances (default 5)
        reg_window: rolling window for factor regression (default 252)
        ridge_alpha: ridge regularization strength (default 1e-6)
        residual_window: window for z-scoring residuals (default 63)
        vol_lookback: realized vol estimation window (default 20)
        min_weight: minimum abs weight to emit signal (default 0.01)
        use_ml: use ML predictor (default True)
    """

    STRATEGY_STATUS: StrategyStatus = StrategyStatus.DISABLED  # Degenerate — pending rebuild

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.reg_window = int(config.params.get("reg_window", 252))
        self.ridge_alpha = float(config.params.get("ridge_alpha", 1e-6))
        self.residual_window = int(config.params.get("residual_window", 63))
        self.vol_lookback = int(config.params.get("vol_lookback", 20))
        self.min_weight = float(config.params.get("min_weight", 0.01))
        self.use_ml = bool(config.params.get("use_ml", True))

        self._days_since_rebalance = 0
        self._factor_map: dict[str, tuple[int, int | None, float]] = {}
        self._last_residuals: NDArray | None = None
        self._ml_predictor: ResidualPredictor | None = None

        if self.use_ml and ML_AVAILABLE:
            self._ml_predictor = ResidualPredictor(n_lags=10, alpha=1.0)
        elif self.use_ml:
            print("WARNING: ML requested but not available, using statistical signal")

    # ------------------------------------------------------------------
    # Factor construction
    # ------------------------------------------------------------------

    def _init_factor_map(self, symbols: list[str]) -> None:
        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        self._factor_map = {}
        factor_defs = {
            "MKT": ("SPY", None, 1.0),
            "RATES": ("TLT", "SPY", 1.0),
            "INFL": ("DBC", None, 1.0),
            "SAFE": ("GLD", "TLT", 0.5),
        }
        for fname, (sym1, sym2, scale) in factor_defs.items():
            idx1 = sym_to_idx.get(sym1, -1)
            idx2 = sym_to_idx.get(sym2) if sym2 else None
            if idx1 >= 0:
                self._factor_map[fname] = (idx1, idx2, scale)

    def _build_factors(self, rets: NDArray) -> NDArray:
        n_periods = rets.shape[0]
        factors = np.zeros((n_periods, len(self._factor_map)))
        for j, (fname, (idx1, idx2, scale)) in enumerate(
            sorted(self._factor_map.items())
        ):
            if idx2 is not None:
                factors[:, j] = (rets[:, idx1] - rets[:, idx2]) * scale
            else:
                factors[:, j] = rets[:, idx1] * scale
        return factors

    # ------------------------------------------------------------------
    # Rolling ridge regression
    # ------------------------------------------------------------------

    def _compute_residuals(
        self, rets: NDArray, factors: NDArray,
    ) -> NDArray:
        n_periods, n_assets = rets.shape
        n_factors = factors.shape[1]
        residuals = np.full((n_periods, n_assets), np.nan)

        if n_periods < self.reg_window + 1:
            return residuals

        reg = self.reg_window
        alpha = self.ridge_alpha
        I_reg = alpha * np.eye(n_factors)

        for t in range(reg, n_periods):
            X_train = factors[t - reg:t]
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0, ddof=1)
            X_std_safe = np.maximum(X_std, 1e-10)
            X_train_std = (X_train - X_mean) / X_std_safe
            X_t_std = (factors[t] - X_mean) / X_std_safe

            try:
                XtX_inv = np.linalg.inv(X_train_std.T @ X_train_std + I_reg)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(X_train_std.T @ X_train_std)

            for i in range(n_assets):
                y_train = rets[t - reg:t, i]
                if np.std(y_train) < 1e-10:
                    residuals[t, i] = 0.0
                    continue
                beta = XtX_inv @ (X_train_std.T @ y_train)
                y_pred = float(X_t_std @ beta)
                residuals[t, i] = float(rets[t, i]) - y_pred

        return residuals

    # ------------------------------------------------------------------
    # ML-Enhanced signal generation
    # ------------------------------------------------------------------

    def _compute_signals_ml(
        self, residuals: NDArray, symbols: list[str],
    ) -> NDArray:
        """Use ML to predict next residual, use as signal."""
        n_periods, n_assets = residuals.shape
        signals = np.zeros(n_assets)

        if self._ml_predictor is None:
            return self._compute_signals_statistical(residuals, symbols)

        # Fit ML model on past residuals
        if n_periods > 300:  # Need enough data
            try:
                self._ml_predictor.fit(residuals[:n_periods-1])
            except Exception as e:
                print(f"ML fit failed: {e}")
                return self._compute_signals_statistical(residuals, symbols)

        # Predict next residual
        try:
            predictions = self._ml_predictor.predict(residuals)
        except Exception as e:
            print(f"ML predict failed: {e}")
            return self._compute_signals_statistical(residuals, symbols)

        # Convert predictions to signals (z-score)
        for i in range(n_assets):
            if i < len(symbols) and symbols[i] in FACTOR_ASSETS:
                continue  # Skip factor assets

            if i >= len(predictions) or predictions[i] == 0.0:
                continue

            # Z-score the prediction
            valid = residuals[:, i][~np.isnan(residuals[:, i])]
            if len(valid) < 10:
                continue

            std = np.std(valid[-self.residual_window:]) if len(valid) >= self.residual_window else np.std(valid)
            if std < 1e-10:
                continue

            signals[i] = predictions[i] / std

        return signals

    def _compute_signals_statistical(
        self, residuals: NDArray, symbols: list[str],
    ) -> NDArray:
        """Statistical signal: zscore(current residual)."""
        n_periods, n_assets = residuals.shape
        latest = n_periods - 1
        window = min(self.residual_window, n_periods - 1)

        signals = np.zeros(n_assets)

        for i in range(n_assets):
            if i < len(symbols) and symbols[i] in FACTOR_ASSETS:
                continue  # Skip factor assets

            resid_series = residuals[:latest + 1, i]
            valid = resid_series[~np.isnan(resid_series)]

            if len(valid) < 5:
                continue

            current = valid[-1]
            window_data = valid[-window:] if len(valid) >= window else valid
            std = np.std(window_data)
            if std < 1e-10:
                continue

            signals[i] = current / std

        return signals

    # ------------------------------------------------------------------
    # Portfolio construction (simple z-score)
    # ------------------------------------------------------------------

    def _compute_weights(
        self, signals: NDArray, rets: NDArray,
    ) -> NDArray:
        """Z-score weighting with vol targeting, gross exposure cap, and position limits.

        1. Cross-sectional z-score on signals
        2. Per-asset inverse-vol scaling (dollar vol parity)
        3. Portfolio-level volatility targeting (12% annualized, scale capped 0-2x)
        4. Market-neutral re-centering
        5. Gross exposure constraint (200% max, proportional normalization)
        6. Per-asset position limit (10% max, hard clip)
        """
        n_assets = len(signals)

        if n_assets == 0:
            return np.zeros(0)

        # Z-score signals
        non_zero = signals[np.abs(signals) > 1e-10]
        if len(non_zero) == 0:
            return np.zeros(n_assets)

        sig_mean = np.mean(non_zero)
        sig_std = np.std(non_zero, ddof=1)
        if sig_std < 1e-10:
            return np.zeros(n_assets)

        weights = np.zeros(n_assets)
        for i in range(n_assets):
            if abs(signals[i]) > 1e-10:
                weights[i] = (signals[i] - sig_mean) / sig_std

        # Per-asset inverse-vol scaling (dollar vol parity)
        if rets.shape[0] >= self.vol_lookback + 2:
            vol = np.std(rets[-self.vol_lookback:], axis=0, ddof=1) * np.sqrt(252)
            vol = np.maximum(vol, 1e-10)
            for i in range(n_assets):
                if weights[i] != 0:
                    weights[i] = weights[i] / vol[i]

        # Portfolio-level volatility targeting (12% annualized, no lookahead)
        target_vol = 0.12
        if rets.shape[0] >= self.vol_lookback + 2:
            port_rets = rets[-self.vol_lookback:] @ weights
            port_vol = np.std(port_rets, ddof=1) * np.sqrt(252)
            if port_vol > 1e-10:
                vol_scale = np.clip(target_vol / port_vol, 0.0, 2.0)
                weights = weights * vol_scale

        # Re-center (market neutral)
        weights = weights - np.mean(weights)

        # Normalize
        max_abs = np.max(np.abs(weights))
        if max_abs > 1e-10:
            weights = weights / max_abs

        # Gross exposure constraint (200% max, proportional normalization)
        gross = np.sum(np.abs(weights))
        if gross > 2.0:
            weights = weights * (2.0 / gross)

        # Per-asset position limits (10% max)
        weights = np.clip(weights, -0.10, 0.10)

        return weights

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        if not self._factor_map:
            self._init_factor_map(symbols)

        if n_periods < self.reg_window + 1:
            return []

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq:
            return []

        # Compute returns
        daily_rets = (
            (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)
        )

        # Build factors
        factors = self._build_factors(daily_rets)

        # Compute residuals
        residuals = self._compute_residuals(daily_rets, factors)
        self._last_residuals = residuals

        # Compute signals (ML or Statistical)
        if self.use_ml and self._ml_predictor is not None:
            signals = self._compute_signals_ml(residuals, symbols)
        else:
            signals = self._compute_signals_statistical(residuals, symbols)

        # Compute weights
        weights = self._compute_weights(signals, daily_rets)

        # Emit signals
        trade_signals: list[TradeSignal] = []
        for i in range(n_assets):
            w = float(weights[i])
            if abs(w) < self.min_weight:
                continue

            trade_signals.append(TradeSignal(
                symbol=symbols[i],
                direction=1 if w > 0 else -1,
                strength=abs(w),
                size=abs(w) * float(self.config.max_position_notional),
                metadata={
                    "weight": round(w, 4),
                    "signal": round(float(signals[i]), 4),
                    "residual": round(float(residuals[-1, i]), 6)
                    if not np.isnan(residuals[-1, i]) else 0.0,
                },
            ))

        self._days_since_rebalance = 0
        return trade_signals
