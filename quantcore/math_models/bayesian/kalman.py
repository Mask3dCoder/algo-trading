"""Kalman Filter: standard, extended, unscented with dynamic hedge ratio application."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class KFState:
    """Kalman Filter state at a given time step."""
    state: NDArray
    covariance: NDArray


class KalmanFilter:
    """Standard linear Kalman Filter with predict/update cycle.

    State model:  x_t = F * x_{t-1} + B * u_t + w_t,  w_t ~ N(0, Q)
    Observation:  z_t = H * x_t + v_t,                 v_t ~ N(0, R)
    """

    def __init__(
        self,
        F: NDArray,
        H: NDArray,
        Q: NDArray,
        R: NDArray,
        x0: NDArray | None = None,
        P0: NDArray | None = None,
        B: NDArray | None = None,
    ) -> None:
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B

        n = F.shape[0]
        self.x = x0 if x0 is not None else np.zeros(n)
        self.P = P0 if P0 is not None else np.eye(n) * 1e6

        self.states: list[NDArray] = [self.x.copy()]
        self.covariances: list[NDArray] = [self.P.copy()]

    def predict(self, u: NDArray | None = None) -> None:
        """Predict step: propagate state and covariance forward."""
        if u is not None and self.B is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: NDArray) -> None:
        """Update step: incorporate observation z."""
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P

        self.states.append(self.x.copy())
        self.covariances.append(self.P.copy())

    def filter(self, observations: NDArray) -> tuple[NDArray, NDArray]:
        """Run the full filter on a sequence of observations.

        Returns (states, covariances) arrays.
        """
        for z in observations:
            self.predict()
            self.update(z)

        return np.array(self.states), np.array(self.covariances)


class ExtendedKalmanFilter:
    """Extended Kalman Filter — handles nonlinear state/observation via Jacobians."""

    def __init__(
        self,
        state_fn,
        observation_fn,
        state_jacobian_fn,
        observation_jacobian_fn,
        Q: NDArray,
        R: NDArray,
        x0: NDArray | None = None,
        P0: NDArray | None = None,
    ) -> None:
        self.state_fn = state_fn
        self.observation_fn = observation_fn
        self.state_jacobian = state_jacobian_fn
        self.observation_jacobian = observation_jacobian_fn
        self.Q = Q
        self.R = R

        n = Q.shape[0]
        self.x = x0 if x0 is not None else np.zeros(n)
        self.P = P0 if P0 is not None else np.eye(n) * 1e6

        self.states: list[NDArray] = [self.x.copy()]

    def predict(self, u: NDArray | None = None) -> None:
        self.x = self.state_fn(self.x, u) if u is not None else self.state_fn(self.x)
        F = self.state_jacobian(self.x)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: NDArray) -> None:
        H = self.observation_jacobian(self.x)
        y = z - self.observation_fn(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        self.states.append(self.x.copy())

    def filter(self, observations: NDArray) -> NDArray:
        for z in observations:
            self.predict()
            self.update(z)
        return np.array(self.states)


class UnscentedKalmanFilter:
    """Unscented Kalman Filter — no Jacobians, uses sigma points."""

    def __init__(
        self,
        state_fn,
        observation_fn,
        Q: NDArray,
        R: NDArray,
        x0: NDArray | None = None,
        P0: NDArray | None = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.state_fn = state_fn
        self.observation_fn = observation_fn
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        n = Q.shape[0]
        self.x = x0 if x0 is not None else np.zeros(n)
        self.P = P0 if P0 is not None else np.eye(n) * 1e6
        self.n = n
        self.lambda_ = alpha**2 * (n + kappa) - n

        self.states: list[NDArray] = [self.x.copy()]

    def _sigma_points(self) -> tuple[NDArray, NDArray, NDArray]:
        """Generate sigma points and weights."""
        n = self.n
        lam = self.lambda_
        sigma_pts = np.zeros((2 * n + 1, n))
        sigma_pts[0] = self.x

        try:
            L = np.linalg.cholesky((n + lam) * self.P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky((n + lam) * (self.P + 1e-6 * np.eye(n)))

        for i in range(n):
            sigma_pts[i + 1] = self.x + L[i, :]
            sigma_pts[n + i + 1] = self.x - L[i, :]

        wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        wm[0] = lam / (n + lam)
        wc[0] = lam / (n + lam) + (1 - self.alpha**2 + self.beta)

        return sigma_pts, wm, wc

    def predict(self) -> None:
        sigma_pts, wm, wc = self._sigma_points()
        # Propagate sigma points through state function
        predicted = np.array([self.state_fn(sp) for sp in sigma_pts])
        self.x = np.sum(wm[:, None] * predicted, axis=0)
        self.P = self.Q.copy()
        for i in range(len(sigma_pts)):
            diff = predicted[i] - self.x
            self.P += wc[i] * np.outer(diff, diff)

    def update(self, z: NDArray) -> None:
        sigma_pts, wm, wc = self._sigma_points()
        # Transform through observation function
        obs_pts = np.array([self.observation_fn(sp) for sp in sigma_pts])
        z_pred = np.sum(wm[:, None] * obs_pts, axis=0)

        Pzz = self.R.copy()
        Pxz = np.zeros((self.n, len(z)))
        for i in range(len(sigma_pts)):
            dz = obs_pts[i] - z_pred
            dx = sigma_pts[i] - self.x
            Pzz += wc[i] * np.outer(dz, dz)
            Pxz += wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(Pzz)
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ Pzz @ K.T
        self.states.append(self.x.copy())

    def filter(self, observations: NDArray) -> NDArray:
        for z in observations:
            self.predict()
            self.update(z)
        return np.array(self.states)


def dynamic_hedge_ratio(y: NDArray, x: NDArray) -> NDArray:
    """Estimate time-varying hedge ratio using Kalman Filter.

    State: [alpha_t, beta_t] (intercept, hedge ratio)
    Observation: y_t = alpha_t + beta_t * x_t + epsilon
    """
    n = len(y)
    F = np.eye(2)
    H = np.array([[1.0, x[0]]])
    Q = np.eye(2) * 0.001
    R = np.array([[0.01]])

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=np.array([0.0, 1.0]), P0=np.eye(2) * 1.0)

    betas = [1.0]
    for t in range(1, n):
        kf.H = np.array([[1.0, x[t]]])
        kf.predict()
        kf.update(np.array([y[t]]))
        betas.append(float(kf.x[1]))

    return np.array(betas)
