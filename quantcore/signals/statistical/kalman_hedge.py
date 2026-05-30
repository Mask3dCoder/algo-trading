"""Kalman Filter for time-varying hedge ratio estimation.

Replaces expanding-window OLS with an online Bayesian filter that:
  - Tracks a time-varying hedge ratio β_t and intercept α_t
  - Provides built-in uncertainty (covariance) at each step
  - Naturally adapts to structural breaks via process noise
  - Reduces unnecessary rebalancing (smooth updates, not step-function)

The state-space model:
    State:      [α_t, β_t]  (intercept + hedge ratio)
    Transition: [α_t, β_t] = [α_{t-1}, β_{t-1}] + w_t,  w_t ~ N(0, Q)
    Observation: y_t = α_t + β_t * x_t + v_t,            v_t ~ N(0, R)

This is a standard linear-Gaussian model solved by the Kalman filter.

References:
    - Harvey (1989), Forecasting, Structural Time Series Models and the Kalman Filter
    - Elliott & Van Der Hoek (2001), An application of pairs trading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """State of the Kalman filter at a single time step."""
    alpha: float          # intercept estimate
    beta: float           # hedge ratio estimate
    P_alpha: float        # variance of alpha estimate
    P_beta: float         # variance of beta estimate
    P_cross: float        # covariance(alpha, beta)
    spread: float         # y - alpha - beta * x
    innovation: float     # same as spread (prediction error)
    innovation_var: float # variance of innovation (S)


class KalmanHedgeRatio:
    """Online Kalman filter for time-varying hedge ratio with intercept.

    The filter maintains a 2D state [α, β] and updates it with each new
    observation pair (y_t, x_t).  The hedge ratio β adapts over time
    according to the process noise, which controls how fast it can change.

    Parameters
    ----------
    delta : float
        Process noise scaling factor (default 1e-4).
        Larger → faster adaptation but noisier estimates.
        Smaller → smoother but slower to adapt to structural breaks.
        Typical range: 1e-5 to 1e-3.
    Ve : float
        Observation noise variance (default 1e-1).
        Larger → trust observations less, smoother estimates.
        Smaller → trust observations more, noisier estimates.
    initial_beta : float | None
        Starting hedge ratio.  If None, initialized to OLS on first
        `warmup` observations.
    warmup : int
        Number of observations to use for OLS initialization (default 60).
    """

    def __init__(
        self,
        delta: float = 1e-4,
        Ve: float = 1e-1,
        initial_beta: float | None = None,
        warmup: int = 60,
    ) -> None:
        self.delta = delta
        self.Ve = Ve
        self._initial_beta = initial_beta
        self._warmup = warmup

        # State: [alpha, beta]
        self._x = np.zeros(2)  # state estimate
        self._P = np.eye(2) * 1.0  # state covariance

        # Process noise covariance (scaled by delta)
        # Q = delta * I (random walk assumption)
        self._Q_base = np.array([
            [delta, 0.0],
            [0.0, delta],
        ])

        # Observation noise
        self._R = Ve

        # History
        self._states: list[KalmanState] = []
        self._initialized = False
        self._warmup_x: list[float] = []
        self._warmup_y: list[float] = []

    def _ols_init(self, x: NDArray, y: NDArray) -> tuple[float, float]:
        """OLS initialization from warmup data."""
        n = len(x)
        if n < 2:
            return 0.0, 1.0

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xx = np.sum((x - x_mean) ** 2)

        if ss_xx < 1e-15:
            return y_mean, 1.0

        beta = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
        alpha = y_mean - beta * x_mean
        return float(alpha), float(beta)

    def update(self, y_t: float, x_t: float) -> KalmanState:
        """Process one observation and return updated state.

        Parameters
        ----------
        y_t : float
            Dependent variable (e.g., asset A price).
        x_t : float
            Independent variable (e.g., asset B price).

        Returns
        -------
        KalmanState
            Current state estimate after the update.
        """
        # Handle NaN
        if np.isnan(y_t) or np.isnan(x_t) or x_t == 0:
            state = KalmanState(
                alpha=self._x[0], beta=self._x[1],
                P_alpha=self._P[0, 0], P_beta=self._P[1, 1],
                P_cross=self._P[0, 1],
                spread=np.nan, innovation=np.nan, innovation_var=np.nan,
            )
            self._states.append(state)
            return state

        # Warmup phase: collect data for OLS initialization
        if not self._initialized:
            self._warmup_x.append(float(x_t))
            self._warmup_y.append(float(y_t))

            if len(self._warmup_x) >= self._warmup:
                alpha, beta = self._ols_init(
                    np.array(self._warmup_x), np.array(self._warmup_y)
                )
                if self._initial_beta is not None:
                    beta = self._initial_beta
                self._x = np.array([alpha, beta])
                # Initialize covariance from OLS residuals
                residuals = np.array(self._warmup_y) - alpha - beta * np.array(self._warmup_x)
                res_var = float(np.var(residuals)) if len(residuals) > 2 else self.Ve
                self._P = np.array([
                    [res_var, 0.0],
                    [0.0, res_var * 0.1],
                ])
                self._initialized = True
            else:
                # During warmup, return OLS estimate from available data
                alpha, beta = self._ols_init(
                    np.array(self._warmup_x), np.array(self._warmup_y)
                )
                state = KalmanState(
                    alpha=alpha, beta=beta,
                    P_alpha=1.0, P_beta=1.0, P_cross=0.0,
                    spread=y_t - alpha - beta * x_t,
                    innovation=y_t - alpha - beta * x_t,
                    innovation_var=self.Ve,
                )
                self._states.append(state)
                return state

        # --- Predict ---
        # State prediction: x_pred = x (random walk)
        x_pred = self._x.copy()
        P_pred = self._P + self._Q

        # --- Update ---
        # Observation matrix: H = [1, x_t]
        H = np.array([1.0, x_t])

        # Innovation: y_t - H @ x_pred
        y_pred = H @ x_pred
        innovation = y_t - y_pred

        # Innovation variance: S = H @ P_pred @ H^T + R
        S = H @ P_pred @ H + self._R

        # Kalman gain: K = P_pred @ H^T / S
        K = P_pred @ H / S

        # State update
        self._x = x_pred + K * innovation
        I_KH = np.eye(2) - np.outer(K, H)
        self._P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self._R  # Joseph form

        # Build state
        state = KalmanState(
            alpha=float(self._x[0]),
            beta=float(self._x[1]),
            P_alpha=float(self._P[0, 0]),
            P_beta=float(self._P[1, 1]),
            P_cross=float(self._P[0, 1]),
            spread=float(innovation),
            innovation=float(innovation),
            innovation_var=float(S),
        )
        self._states.append(state)
        return state

    @property
    def _Q(self) -> np.ndarray:
        """Process noise covariance (may be time-varying in extensions)."""
        return self._Q_base.copy()

    @property
    def current_beta(self) -> float:
        """Current hedge ratio estimate."""
        return float(self._x[1])

    @property
    def current_alpha(self) -> float:
        """Current intercept estimate."""
        return float(self._x[0])

    @property
    def beta_uncertainty(self) -> float:
        """Standard deviation of hedge ratio estimate."""
        return float(np.sqrt(max(self._P[1, 1], 0.0)))

    @property
    def states(self) -> list[KalmanState]:
        """Full history of filter states."""
        return self._states


def kalman_hedge_ratio_series(
    y: NDArray,
    x: NDArray,
    delta: float = 1e-4,
    Ve: float = 1e-1,
    warmup: int = 60,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute time-varying hedge ratio and spread using Kalman filter.

    Convenience function that runs the full filter over arrays and returns
    the hedge ratio series, spread series, and uncertainty series.

    Parameters
    ----------
    y : NDArray
        Dependent variable series (e.g., asset A prices).
    x : NDArray
        Independent variable series (e.g., asset B prices).
    delta : float
        Process noise scaling (default 1e-4).
    Ve : float
        Observation noise variance (default 1e-1).
    warmup : int
        Warmup period for OLS initialization (default 60).

    Returns
    -------
    betas : NDArray
        Time-varying hedge ratio at each step.
    spreads : NDArray
        Residual spread (y - alpha - beta * x) at each step.
    uncertainties : NDArray
        Standard deviation of beta estimate at each step.
    """
    kf = KalmanHedgeRatio(delta=delta, Ve=Ve, warmup=warmup)

    n = len(y)
    betas = np.full(n, np.nan)
    spreads = np.full(n, np.nan)
    uncertainties = np.full(n, np.nan)

    for t in range(n):
        state = kf.update(float(y[t]), float(x[t]))
        betas[t] = state.beta
        spreads[t] = state.spread
        uncertainties[t] = np.sqrt(max(state.P_beta, 0.0))

    return betas, spreads, uncertainties


def adaptive_kalman_hedge_ratio(
    y: NDArray,
    x: NDArray,
    Ve: float = 1e-1,
    warmup: int = 60,
    delta_min: float = 1e-6,
    delta_max: float = 1e-2,
    innovation_window: int = 20,
) -> tuple[NDArray, NDArray, NDArray]:
    """Adaptive Kalman filter that adjusts process noise based on innovation.

    When the filter's predictions consistently miss (large innovations),
    delta is increased to allow faster adaptation.  When predictions are
    accurate, delta is decreased for smoother estimates.

    This is useful for pairs trading where structural breaks happen
    unpredictably.

    Parameters
    ----------
    y, x : NDArray
        Dependent and independent variable series.
    Ve : float
        Observation noise variance.
    warmup : int
        Warmup period.
    delta_min, delta_max : float
        Bounds on the adaptive process noise.
    innovation_window : int
        Window for computing normalized innovation magnitude.

    Returns
    -------
    betas, spreads, uncertainties : NDArray
        Same as kalman_hedge_ratio_series.
    """
    kf = KalmanHedgeRatio(delta=delta_min, Ve=Ve, warmup=warmup)

    n = len(y)
    betas = np.full(n, np.nan)
    spreads = np.full(n, np.nan)
    uncertainties = np.full(n, np.nan)

    recent_innovations: list[float] = []

    for t in range(n):
        state = kf.update(float(y[t]), float(x[t]))

        # Track normalized innovations
        if not np.isnan(state.innovation) and not np.isnan(state.innovation_var):
            norm_innov = abs(state.innovation) / max(np.sqrt(state.innovation_var), 1e-10)
            recent_innovations.append(norm_innov)
            if len(recent_innovations) > innovation_window:
                recent_innovations.pop(0)

            # Adapt delta: high normalized innovations → increase delta
            if len(recent_innovations) >= 5:
                mean_norm = np.mean(recent_innovations)
                # Scale: when mean normalized innovation > 1, increase delta
                adapt = np.clip(mean_norm * delta_min * 10, delta_min, delta_max)
                kf._Q_base = np.array([[adapt, 0.0], [0.0, adapt]])

        betas[t] = state.beta
        spreads[t] = state.spread
        uncertainties[t] = np.sqrt(max(state.P_beta, 0.0))

    return betas, spreads, uncertainties
