"""Hidden Markov Model regime detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


@dataclass
class RegimeResult:
    """HMM regime detection result."""
    n_states: int
    states: NDArray[np.integer]  # 0 = bull, 1 = bear, etc.
    probabilities: NDArray[np.floating]  # Smoothed state probabilities (n_periods × n_states)
    current_state: int
    current_probability: float
    transition_matrix: NDArray[np.floating]
    means: NDArray[np.floating]  # Mean return per state
    volatilities: NDArray[np.floating]  # Volatility per state
    aic: float = 0.0
    bic: float = 0.0
    state_labels: list[str] = field(default_factory=list)


class RegimeHMM:
    """Hidden Markov Model for market regime detection.

    Uses hmmlearn for Gaussian/Student-t HMM fitting.
    Supports 2-state (bull/bear), 3-state (bull/bear/sideways),
    and 4-state (trending/mean-reverting/high-vol/low-vol) models.
    """

    def __init__(self, n_states: int = 2, covariance_type: str = "full") -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self._model = None
        self._fitted = False
        self.state_labels: list[str] = []

    def fit(
        self,
        data: NDArray[np.floating],
        n_iter: int = 100,
        tol: float = 1e-4,
    ) -> RegimeResult:
        """Fit HMM to return (or feature) data.

        Args:
            data: array (n_periods,) or (n_periods, n_features).
            n_iter: Max EM iterations.
            tol: Convergence tolerance.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("hmmlearn not installed. Run: pip install hmmlearn")

        X = np.asarray(data, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=42,
        )

        model.fit(X)
        self._model = model
        self._fitted = True

        # Viterbi decoding
        states = model.predict(X)

        # Smoothed probabilities
        probs = model.predict_proba(X)

        # Current state
        current_state = int(states[-1])
        current_prob = float(probs[-1, current_state])

        # Transition matrix
        trans_mat = model.transmat_

        # State characteristics
        means = np.zeros(self.n_states)
        vols = np.zeros(self.n_states)
        for s in range(self.n_states):
            mask = states == s
            if np.any(mask):
                state_data = X[mask, 0]  # First feature (returns)
                means[s] = float(np.mean(state_data))
                vols[s] = float(np.std(state_data))

        # Label states by return (for 2-state)
        if self.n_states == 2:
            if means[0] > means[1]:
                self.state_labels = ["bull", "bear"]
            else:
                self.state_labels = ["bear", "bull"]
        elif self.n_states == 3:
            # Sort by return: bull → sideways → bear
            idx = np.argsort(means)
            self.state_labels = ["bear", "sideways", "bull"]
        elif self.n_states == 4:
            # Sort by return and vol
            self.state_labels = [
                f"state_{i}" for i in range(self.n_states)
            ]

        # AIC/BIC
        log_lik = model.score(X)
        n_params = self.n_states**2 + self.n_states * n_features * 2
        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(n_samples)

        return RegimeResult(
            n_states=self.n_states,
            states=states,
            probabilities=probs,
            current_state=current_state,
            current_probability=current_prob,
            transition_matrix=trans_mat,
            means=means,
            volatilities=vols,
            aic=float(aic),
            bic=float(bic),
            state_labels=self.state_labels,
        )

    def predict_state(
        self,
        data: NDArray[np.floating],
    ) -> tuple[int, float]:
        """Online Viterbi: predict current state.

        Returns:
            (state, probability)
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(data, dtype=float).reshape(1, -1)
        state = int(self._model.predict(X)[0])
        prob = float(self._model.predict_proba(X)[0, state])
        return state, prob

    def smoothed_probabilities(
        self,
        data: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Forward-backward smoothed state probabilities.

        If data is None, returns probabilities from the fitted data.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if data is not None:
            X = np.asarray(data, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return self._model.predict_proba(X)

        # Return fitted probabilities
        return self._model.predict_proba(np.zeros((1, 1)))  # Fallback


def select_hmm_states(
    data: NDArray[np.floating],
    max_states: int = 5,
) -> list[dict]:
    """Model selection: fit HMMs with 2..max_states and rank by AIC/BIC."""
    results = []
    for k in range(2, max_states + 1):
        try:
            hmm = RegimeHMM(n_states=k)
            result = hmm.fit(data)
            results.append({
                "n_states": k,
                "aic": result.aic,
                "bic": result.bic,
                "log_lik": -result.aic / 2 + (k**2 + k),
            })
        except Exception:
            continue

    return sorted(results, key=lambda x: x["aic"])
