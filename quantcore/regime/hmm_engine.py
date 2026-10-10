"""From-scratch Hidden Markov Model engine (Rabiner 1986).

Production-ready, vectorized implementation of the three fundamental HMM problems:
  1. Forward-Backward (evaluation): P(O | λ)
  2. Viterbi (decoding): argmax_I P(O, I | λ)
  3. Baum-Welch / EM (learning): argmax_λ P(O | λ)

Optimized for financial regime detection with:
  - Log-space computation (prevents underflow on long sequences)
  - NaN/inf-safe observation likelihoods
  - Full covariance Gaussian emissions (default) or diagonal
  - Multi-start EM to escape local optima
  - Online (recursive) state probability update for live trading

Reference: L.R. Rabiner, "An Introduction to Hidden Markov Models",
           IEEE ASSP Magazine, January 1986.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HMMParams:
    """HMM parameter set λ = (A, B, π).

    Attributes:
        n_states: N — number of hidden states.
        n_features: D — observation dimensionality.
        pi: (N,) initial state distribution.
        A: (N, N) state transition matrix.
        means: (N, D) per-state emission means.
        covars: (N, D, D) per-state emission covariances.
        covar_type: 'full' or 'diag'.
    """
    n_states: int
    n_features: int
    pi: NDArray[np.floating]
    A: NDArray[np.floating]
    means: NDArray[np.floating]
    covars: NDArray[np.floating]
    covar_type: Literal["full", "diag"] = "full"


@dataclass
class HMMResult:
    """Result from HMM fitting or inference."""
    params: HMMParams
    states: NDArray[np.integer]             # (T,) Viterbi state sequence
    log_likelihood: float                   # log P(O | λ)
    state_probs: NDArray[np.floating]       # (T, N) γ_t(i) — smoothed probs
    n_iter: int                             # EM iterations used
    converged: bool
    bic: float = 0.0
    aic: float = 0.0


# ---------------------------------------------------------------------------
# Observation likelihood (Gaussian, NaN-safe)
# ---------------------------------------------------------------------------

def _log_multivariate_normal(
    X: NDArray[np.floating],
    mean: NDArray[np.floating],
    covar: NDArray[np.floating],
    covar_type: Literal["full", "diag"],
) -> NDArray[np.floating]:
    """Compute log N(x; μ, Σ) for each observation, NaN-safe.

    Args:
        X: (T, D) observations.
        mean: (D,) mean vector.
        covar: (D, D) covariance (full) or (D,) diagonal.
        covar_type: 'full' or 'diag'.

    Returns:
        (T,) log-probabilities. NaN observations → -inf (masked out later).
    """
    T, D = X.shape

    # Replace NaN/inf with zeros for computation; mask later
    mask_nan = ~np.isfinite(X)
    X_clean = np.where(mask_nan, 0.0, X)

    if covar_type == "diag":
        var = np.maximum(covar, 1e-12)  # (D,)
        diff = X_clean - mean  # (T, D)
        mahal = np.sum(diff ** 2 / var, axis=1)  # (T,)
        log_norm = -0.5 * (D * np.log(2 * np.pi) + np.sum(np.log(var)) + mahal)
    else:
        # Full covariance
        try:
            L = np.linalg.cholesky(covar)
        except np.linalg.LinAlgError:
            # Regularize singular covariance
            covar_reg = covar + 1e-6 * np.eye(D)
            L = np.linalg.cholesky(covar_reg)

        diff = X_clean - mean  # (T, D)
        # Solve L @ y = diff^T  →  y = L^{-1} diff^T
        solve = np.linalg.solve(L, diff.T)  # (D, T)
        mahal = np.sum(solve ** 2, axis=0)  # (T,)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        log_norm = -0.5 * (D * np.log(2 * np.pi) + log_det + mahal)

    # Mask NaN observations
    if mask_nan.any():
        # If entire row is NaN → -inf; if partial → use available dims
        row_nan = mask_nan.all(axis=1)
        log_norm[row_nan] = -np.inf
        # Partial NaN: recompute using only finite dims
        partial = mask_nan.any(axis=1) & ~row_nan
        if partial.any():
            for t in np.where(partial)[0]:
                finite_dims = ~mask_nan[t]
                if finite_dims.sum() == 0:
                    log_norm[t] = -np.inf
                    continue
                x_t = X[t, finite_dims]
                m_t = mean[finite_dims]
                if covar_type == "diag":
                    v_t = np.maximum(covar[finite_dims], 1e-12)
                    d_t = finite_dims.sum()
                    mahal_t = np.sum((x_t - m_t) ** 2 / v_t)
                    log_norm[t] = -0.5 * (d_t * np.log(2 * np.pi) + np.sum(np.log(v_t)) + mahal_t)
                else:
                    c_t = covar[np.ix_(finite_dims, finite_dims)]
                    try:
                        L_t = np.linalg.cholesky(c_t)
                    except np.linalg.LinAlgError:
                        c_t = c_t + 1e-6 * np.eye(c_t.shape[0])
                        L_t = np.linalg.cholesky(c_t)
                    d_t = finite_dims.sum()
                    diff_t = x_t - m_t
                    s_t = np.linalg.solve(L_t, diff_t)
                    mahal_t = np.sum(s_t ** 2)
                    log_det_t = 2.0 * np.sum(np.log(np.diag(L_t)))
                    log_norm[t] = -0.5 * (d_t * np.log(2 * np.pi) + log_det_t + mahal_t)

    return log_norm


def _compute_log_B(
    X: NDArray[np.floating],
    params: HMMParams,
) -> NDArray[np.floating]:
    """Compute log emission probabilities for all states.

    Args:
        X: (T, D) observations.
        params: HMM parameters.

    Returns:
        (T, N) log b_j(O_t) matrix.
    """
    T = X.shape[0]
    N = params.n_states
    log_B = np.empty((T, N))

    for j in range(N):
        log_B[:, j] = _log_multivariate_normal(
            X, params.means[j], params.covars[j], params.covar_type
        )

    return log_B


# ---------------------------------------------------------------------------
# Problem 1: Forward-Backward (log-space)
# ---------------------------------------------------------------------------

def forward_log(
    log_B: NDArray[np.floating],
    log_A: NDArray[np.floating],
    log_pi: NDArray[np.floating],
) -> tuple[NDArray[np.floating], float]:
    """Log-space forward algorithm.

    Computes α̂_t(j) = log Σ_i α̂_{t-1}(i) + log a_{ij} + log b_j(O_t)
    with log-sum-exp for numerical stability.

    Args:
        log_B: (T, N) log emission probabilities.
        log_A: (N, N) log transition matrix.
        log_pi: (N,) log initial distribution.

    Returns:
        log_alpha: (T, N) log forward variables.
        log_likelihood: log P(O | λ).
    """
    T, N = log_B.shape
    log_alpha = np.full((T, N), -np.inf)

    # Initialization: α̂_1(i) = log π_i + log b_i(O_1)
    log_alpha[0] = log_pi + log_B[0]

    # Induction — vectorized over states
    for t in range(1, T):
        # (N, 1) + (N, N) → (N, N), logsumexp over axis=0 → (N,)
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_B[t]

    # Termination: log P(O|λ) = logsumexp(α̂_T)
    log_likelihood = float(logsumexp(log_alpha[-1]))

    return log_alpha, log_likelihood


def backward_log(
    log_B: NDArray[np.floating],
    log_A: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Log-space backward algorithm.

    Computes β̂_t(i) = log Σ_j a_{ij} + log b_j(O_{t+1}) + β̂_{t+1}(j)

    Args:
        log_B: (T, N) log emission probabilities.
        log_A: (N, N) log transition matrix.

    Returns:
        log_beta: (T, N) log backward variables.
    """
    T, N = log_B.shape
    log_beta = np.full((T, N), -np.inf)

    # Initialization: β̂_T(i) = 0 (log 1)
    log_beta[-1] = 0.0

    # Induction (backward) — vectorized over states
    for t in range(T - 2, -1, -1):
        # (N, N) + (1, N) + (1, N) → logsumexp over axis=1 → (N,)
        log_beta[t] = logsumexp(log_A + log_B[t + 1, None, :] + log_beta[t + 1, None, :], axis=1)

    return log_beta


def compute_gamma(
    log_alpha: NDArray[np.floating],
    log_beta: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute γ_t(i) = P(i_t = q_i | O, λ) in log-space.

    γ̂_t(i) = α̂_t(i) + β̂_t(i) - logsumexp(α̂_t + β̂_t)

    Returns:
        (T, N) log γ values. Exponentiate for probabilities.
    """
    log_gamma = log_alpha + log_beta
    # Normalize per time step
    log_norm = logsumexp(log_gamma, axis=1, keepdims=True)
    log_gamma = log_gamma - log_norm
    return log_gamma


def compute_xi(
    log_alpha: NDArray[np.floating],
    log_beta: NDArray[np.floating],
    log_B: NDArray[np.floating],
    log_A: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute ξ_t(i,j) = P(i_t=q_i, i_{t+1}=q_j | O, λ) in log-space.

    ξ̂_t(i,j) = α̂_t(i) + log a_{ij} + log b_j(O_{t+1}) + β̂_{t+1}(j) - log P(O|λ)

    Returns:
        (T-1, N, N) log ξ values.
    """
    T, N = log_alpha.shape
    log_xi = np.empty((T - 1, N, N))

    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                log_xi[t, i, j] = (
                    log_alpha[t, i]
                    + log_A[i, j]
                    + log_B[t + 1, j]
                    + log_beta[t + 1, j]
                )
        # Normalize
        log_xi[t] -= logsumexp(log_xi[t].ravel())

    return log_xi


# ---------------------------------------------------------------------------
# Problem 2: Viterbi (log-space)
# ---------------------------------------------------------------------------

def viterbi_log(
    log_B: NDArray[np.floating],
    log_A: NDArray[np.floating],
    log_pi: NDArray[np.floating],
) -> tuple[NDArray[np.integer], float]:
    """Log-space Viterbi algorithm — optimal state sequence.

    δ_t(j) = max_i [δ_{t-1}(i) + log a_{ij}] + log b_j(O_t)
    ψ_t(j) = argmax_i [δ_{t-1}(i) + log a_{ij}]

    Args:
        log_B: (T, N) log emission probabilities.
        log_A: (N, N) log transition matrix.
        log_pi: (N,) log initial distribution.

    Returns:
        states: (T,) optimal state sequence.
        log_prob: log P(O, I* | λ).
    """
    T, N = log_B.shape

    # Initialization
    delta = log_pi + log_B[0]  # (N,)
    psi = np.zeros((T, N), dtype=np.int32)

    # Recursion
    for t in range(1, T):
        # delta[i] + log_A[:, j] for all i,j → (N, N)
        # For each j, find best i
        score = delta[:, None] + log_A  # (N, N)
        psi[t] = np.argmax(score, axis=0)
        delta = np.max(score, axis=0) + log_B[t]

    # Termination
    best_last = int(np.argmax(delta))
    log_prob = float(delta[best_last])

    # Backtrack
    states = np.zeros(T, dtype=np.int32)
    states[-1] = best_last
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states, log_prob


# ---------------------------------------------------------------------------
# Problem 3: Baum-Welch / EM
# ---------------------------------------------------------------------------

def _init_params_kmeans(
    X: NDArray[np.floating],
    n_states: int,
    covar_type: Literal["full", "diag"],
    rng: np.random.Generator,
) -> HMMParams:
    """Initialize HMM parameters via k-means clustering."""
    from scipy.cluster.vq import kmeans2

    T, D = X.shape
    # Remove NaN rows for k-means
    valid = np.all(np.isfinite(X), axis=1)
    X_valid = X[valid]

    if len(X_valid) < n_states:
        # Fallback: random initialization
        means = rng.standard_normal((n_states, D)) * np.std(X_valid, axis=0)
        base_var = np.atleast_1d(np.var(X_valid, axis=0))
        covars = np.tile(np.eye(D), (n_states, 1, 1)) if covar_type == "full" else np.tile(base_var, (n_states, 1))
    else:
        centroids, labels = kmeans2(X_valid, n_states, minit='points', missing='warn')
        means = centroids
        if covar_type == "full":
            covars = np.empty((n_states, D, D))
            for j in range(n_states):
                mask = labels == j
                if mask.sum() > D:
                    covars[j] = np.cov(X_valid[mask].T) + 1e-4 * np.eye(D)
                else:
                    covars[j] = np.eye(D)
        else:
            covars = np.empty((n_states, D))
            for j in range(n_states):
                mask = labels == j
                if mask.sum() > 1:
                    covars[j] = np.atleast_1d(np.var(X_valid[mask], axis=0)) + 1e-4
                else:
                    covars[j] = np.ones(D)

    # Uniform transition matrix with self-transition bias
    A = np.full((n_states, n_states), 0.1 / (n_states - 1))
    np.fill_diagonal(A, 0.9)
    A /= A.sum(axis=1, keepdims=True)

    pi = np.ones(n_states) / n_states

    return HMMParams(
        n_states=n_states,
        n_features=D,
        pi=pi,
        A=A,
        means=means,
        covars=covars,
        covar_type=covar_type,
    )


def baum_welch(
    X: NDArray[np.floating],
    params: HMMParams,
    max_iter: int = 100,
    tol: float = 1e-6,
    min_covar: float = 1e-4,
) -> tuple[HMMParams, float, int, bool]:
    """Baum-Welch (EM) re-estimation — Problem 3.

    Re-estimates π, A, B-parameters to maximize P(O | λ).

    Re-estimation formulas (Rabiner 1986, Box 3):
        π̄_i = γ_1(i)
        ā_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
        For Gaussian: μ̄_j = Σ_t γ_t(j) O_t / Σ_t γ_t(j)
                      Σ̄_j = Σ_t γ_t(j) (O_t - μ̄)(O_t - μ̄)^T / Σ_t γ_t(j)

    Args:
        X: (T, D) observations.
        params: Initial HMM parameters.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance on log-likelihood.
        min_covar: Minimum covariance diagonal.

    Returns:
        (params, log_likelihood, n_iter, converged)
    """
    T, D = X.shape
    N = params.n_states
    current = params
    prev_ll = -np.inf

    for iteration in range(max_iter):
        # E-step: compute log-B, forward, backward, gamma, xi
        log_B = _compute_log_B(X, current)
        log_A = np.log(np.maximum(current.A, 1e-300))
        log_pi = np.log(np.maximum(current.pi, 1e-300))

        log_alpha, ll = forward_log(log_B, log_A, log_pi)
        log_beta = backward_log(log_B, log_A)
        log_gamma = compute_gamma(log_alpha, log_beta)
        gamma = np.exp(log_gamma)  # (T, N)

        # Check convergence
        if abs(ll - prev_ll) < tol:
            return current, ll, iteration, True
        prev_ll = ll

        # M-step: re-estimate parameters
        # π̄_i = γ_1(i)
        new_pi = gamma[0]
        new_pi = np.maximum(new_pi, 1e-300)
        new_pi /= new_pi.sum()

        # ā_ij = Σ_{t=1}^{T-1} ξ_t(i,j) / Σ_{t=1}^{T-1} γ_t(i)
        # Loop over T with vectorized inner state computation
        gamma_sum = np.maximum(gamma[:-1].sum(axis=0), 1e-300)  # (N,)
        new_A = np.zeros((N, N))

        # Precompute common terms: log_α[t,:] + log_β[t,:] = log_γ[t,:] (unnormalized)
        log_B_next = log_B[1:]   # (T-1, N)
        log_beta_next = log_beta[1:]  # (T-1, N)

        for t in range(T - 1):
            # log_xi[i,j] = log_α[t,i] + log_A[i,j] + log_B[t+1,j] + log_β[t+1,j]
            log_xi_t = (
                log_alpha[t, :, None]       # (N, 1)
                + log_A                      # (N, N)
                + log_B_next[t, None, :]     # (1, N)
                + log_beta_next[t, None, :]  # (1, N)
            )
            new_A += np.exp(log_xi_t - logsumexp(log_xi_t.ravel()))

        new_A /= gamma_sum[:, None]
        new_A = np.maximum(new_A, 1e-300)
        new_A /= new_A.sum(axis=1, keepdims=True)

        # Gaussian re-estimation
        new_means = np.zeros((N, D))
        new_covars = np.zeros_like(current.covars)

        for j in range(N):
            w = gamma[:, j]  # (T,)
            w_sum = np.maximum(w.sum(), 1e-300)

            # Handle NaN observations
            valid = np.all(np.isfinite(X), axis=1)  # (T,)
            valid_2d = valid[:, None]  # (T, 1) for broadcasting with (T, D)
            w_valid = w * valid  # (T,)
            w_valid_sum = np.maximum(w_valid.sum(), 1e-300)

            # Weighted mean
            mu_j = np.sum(w_valid[:, None] * np.where(valid_2d, X, 0.0), axis=0) / w_valid_sum
            new_means[j] = mu_j

            # Weighted covariance
            diff = np.where(valid_2d, X - mu_j, 0.0)
            if current.covar_type == "full":
                cov_j = np.zeros((D, D))
                for t in range(T):
                    if valid[t]:
                        cov_j += w[t] * np.outer(diff[t], diff[t])
                cov_j /= w_valid_sum
                cov_j += min_covar * np.eye(D)
                new_covars[j] = cov_j
            else:
                var_j = np.sum(w_valid[:, None] * diff ** 2, axis=0) / w_valid_sum
                new_covars[j] = np.maximum(var_j, min_covar)

        current = HMMParams(
            n_states=N,
            n_features=D,
            pi=new_pi,
            A=new_A,
            means=new_means,
            covars=new_covars,
            covar_type=current.covar_type,
        )

    return current, prev_ll, max_iter, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_hmm(
    X: NDArray[np.floating],
    n_states: int = 3,
    covar_type: Literal["full", "diag"] = "full",
    n_iter: int = 100,
    tol: float = 1e-6,
    n_restarts: int = 5,
    random_state: int | None = 42,
) -> HMMResult:
    """Fit an HMM to observations via Baum-Welch with multi-start.

    Args:
        X: (T, D) observation matrix. Rows with any NaN are partially handled.
        n_states: Number of hidden states N.
        covar_type: 'full' or 'diag' covariance.
        n_iter: Max EM iterations per restart.
        tol: Convergence tolerance.
        n_restarts: Number of random restarts (picks best log-lik).
        random_state: Seed for reproducibility.

    Returns:
        HMMResult with fitted parameters, decoded states, and diagnostics.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T, D = X.shape

    # Clean: replace inf with NaN, then forward-fill NaN for emission computation
    # NaN observations are "missing" — we impute with column median for likelihood
    X = np.where(np.isfinite(X), X, np.nan)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        for d in range(D):
            col_nan = nan_mask[:, d]
            if col_nan.any():
                X[col_nan, d] = col_medians[d] if np.isfinite(col_medians[d]) else 0.0

    rng = np.random.default_rng(random_state)

    best_result: HMMResult | None = None
    best_ll = -np.inf

    for restart in range(n_restarts):
        try:
            # Initialize
            if restart == 0:
                init_params = _init_params_kmeans(X, n_states, covar_type, rng)
            else:
                # Random perturbation of k-means init
                init_params = _init_params_kmeans(X, n_states, covar_type, rng)
                noise = rng.standard_normal(init_params.means.shape) * 0.1
                init_params.means += noise * np.std(X[np.all(np.isfinite(X), axis=1)], axis=0)

            # EM
            fitted_params, ll, iters, converged = baum_welch(
                X, init_params, max_iter=n_iter, tol=tol
            )

            if ll > best_ll:
                best_ll = ll
                # Decode with Viterbi
                log_B = _compute_log_B(X, fitted_params)
                log_A = np.log(np.maximum(fitted_params.A, 1e-300))
                log_pi = np.log(np.maximum(fitted_params.pi, 1e-300))
                states, _ = viterbi_log(log_B, log_A, log_pi)

                # Smoothed probabilities
                log_alpha, _ = forward_log(log_B, log_A, log_pi)
                log_beta = backward_log(log_B, log_A)
                log_gamma = compute_gamma(log_alpha, log_beta)
                state_probs = np.exp(log_gamma)

                # Model selection criteria
                n_params = n_states ** 2 + n_states * D * (1 + D) if covar_type == "full" else n_states ** 2 + n_states * 2 * D
                bic = -2 * ll + n_params * np.log(T)
                aic = -2 * ll + 2 * n_params

                best_result = HMMResult(
                    params=fitted_params,
                    states=states,
                    log_likelihood=ll,
                    state_probs=state_probs,
                    n_iter=iters,
                    converged=converged,
                    bic=bic,
                    aic=aic,
                )

        except Exception as e:
            warnings.warn(f"HMM restart {restart} failed: {e}")
            continue

    if best_result is None:
        raise RuntimeError(f"All {n_restarts} HMM fits failed.")

    return best_result


def predict_online(
    X_new: NDArray[np.floating],
    log_alpha_prev: NDArray[np.floating],
    params: HMMParams,
) -> tuple[int, NDArray[np.floating], NDArray[np.floating]]:
    """Online (recursive) state prediction for live trading.

    One-step forward update:
        α̂_t(j) = logsumexp_i [α̂_{t-1}(i) + log a_{ij}] + log b_j(O_t)

    Args:
        X_new: (D,) or (1, D) new observation.
        log_alpha_prev: (N,) previous log forward variable.
        params: Current HMM parameters.

    Returns:
        state: Most likely current state.
        state_probs: (N,) γ_t(i) probabilities.
        log_alpha_new: (N,) updated log forward variable (pass to next call).
    """
    X_new = np.asarray(X_new, dtype=np.float64).reshape(1, -1)
    N = params.n_states

    log_B = _compute_log_B(X_new, params)  # (1, N)
    log_A = np.log(np.maximum(params.A, 1e-300))

    # Forward update
    log_alpha_new = np.empty(N)
    for j in range(N):
        log_alpha_new[j] = logsumexp(log_alpha_prev + log_A[:, j]) + log_B[0, j]

    # State probability (normalized)
    log_norm = logsumexp(log_alpha_new)
    log_probs = log_alpha_new - log_norm
    state_probs = np.exp(log_probs)

    state = int(np.argmax(state_probs))

    return state, state_probs, log_alpha_new


def decode_states(
    result: HMMResult,
    state_labels: list[str] | None = None,
) -> dict[str, any]:
    """Interpret HMM states by their emission characteristics.

    Returns dict mapping state index → {label, mean_return, volatility, avg_duration}.
    """
    interpretation = {}
    N = result.params.n_states
    states = result.states

    for j in range(N):
        mask = states == j
        count = mask.sum()

        # Average duration of state visits
        runs = np.diff(np.where(np.concatenate(([False], states == j, [False])))[0])
        avg_dur = float(np.mean(runs)) if len(runs) > 0 else 0.0

        label = state_labels[j] if state_labels and j < len(state_labels) else f"regime_{j}"

        interpretation[j] = {
            "label": label,
            "mean_return": float(result.params.means[j, 0]) if result.params.n_features >= 1 else 0.0,
            "volatility": float(np.sqrt(result.params.covars[j, 0, 0])) if result.params.covar_type == "full" else float(np.sqrt(result.params.covars[j, 0])),
            "stationary_prob": float(result.params.pi[j]),
            "occupancy_pct": float(count / len(states) * 100),
            "avg_duration_bars": avg_dur,
            "self_transition_prob": float(result.params.A[j, j]),
        }

    return interpretation


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------

def hmm_regime_detect(
    returns: NDArray[np.floating],
    n_states: int = 3,
    features: NDArray[np.floating] | None = None,
    covar_type: Literal["full", "diag"] = "diag",
    n_restarts: int = 5,
) -> HMMResult:
    """End-to-end regime detection for trading pipeline.

    Args:
        returns: (T,) log or simple returns.
        n_states: Number of regimes (2=bull/bear, 3=+sideways, 4=+high-vol).
        features: Optional (T, D-1) additional features (vol, volume, etc.).
        covar_type: 'diag' recommended for high-D features.
        n_restarts: EM restarts.

    Returns:
        HMMResult with decoded regimes.
    """
    r = np.asarray(returns, dtype=np.float64).flatten()

    if features is not None:
        feat = np.asarray(features, dtype=np.float64)
        if feat.ndim == 1:
            feat = feat.reshape(-1, 1)
        X = np.column_stack([r.reshape(-1, 1), feat])
    else:
        X = r.reshape(-1, 1)

    return fit_hmm(X, n_states=n_states, covar_type=covar_type, n_restarts=n_restarts)
