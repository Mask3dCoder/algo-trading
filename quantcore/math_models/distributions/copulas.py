"""Copula models: Gaussian, t, Clayton, Gumbel, Frank."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t


class GaussianCopula:
    """Bivariate Gaussian copula."""

    def __init__(self, rho: float = 0.5) -> None:
        self.rho = rho

    def pdf(self, u: NDArray, v: NDArray) -> NDArray:
        """Compute copula PDF for uniform marginals u, v."""
        x = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        y = norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
        rho = self.rho
        exponent = -(x**2 - 2 * rho * x * y + y**2) / (2 * (1 - rho**2)) + 0.5 * (x**2 + y**2)
        return np.exp(exponent) / np.sqrt(1 - rho**2)

    def cdf(self, u: NDArray, v: NDArray) -> NDArray:
        """Compute copula CDF. Uses bivariate normal CDF."""
        from scipy.stats import multivariate_normal
        x = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        y = norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        rv = multivariate_normal(mean=[0, 0], cov=cov, allow_singular=True)
        return rv.cdf(np.column_stack([x, y]))

    def simulate(self, n: int, seed: int | None = None) -> tuple[NDArray, NDArray]:
        """Generate correlated uniform samples."""
        rng = np.random.default_rng(seed)
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        z = rng.multivariate_normal([0, 0], cov, n)
        return norm.cdf(z[:, 0]), norm.cdf(z[:, 1])

    def tail_dependence(self) -> tuple[float, float]:
        """Gaussian copula has zero tail dependence (except rho=1)."""
        return 0.0, 0.0

    def fit(self, u: NDArray, v: NDArray) -> float:
        """Fit rho via MLE on uniform marginals."""
        x = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        y = norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
        rho = np.corrcoef(x, y)[0, 1]
        self.rho = rho
        return float(rho)


class StudentTCopula:
    """Bivariate Student-t copula."""

    def __init__(self, rho: float = 0.5, df: float = 5.0) -> None:
        self.rho = rho
        self.df = df

    def simulate(self, n: int, seed: int | None = None) -> tuple[NDArray, NDArray]:
        """Generate correlated uniform samples from t-copula."""
        rng = np.random.default_rng(seed)
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        z = rng.multivariate_normal([0, 0], cov, n)
        chi2 = rng.chisquare(self.df, n)
        t_samples = z / np.sqrt(chi2 / self.df)[:, np.newaxis]
        return student_t.cdf(t_samples[:, 0], self.df), student_t.cdf(t_samples[:, 1], self.df)

    def tail_dependence(self) -> tuple[float, float]:
        """Compute tail dependence coefficients for t-copula."""
        rho = self.rho
        df = self.df
        # Lower tail dependence (same as upper for t-copula due to symmetry)
        x = np.sqrt((df + 1) * (1 - rho) / (1 + rho))
        lower = 2.0 * student_t.cdf(-x, df + 1)
        upper = lower  # Symmetric
        return float(lower), float(upper)

    def fit(self, u: NDArray, v: NDArray) -> tuple[float, float]:
        """Fit rho and df via MLE (simplified: match correlation, estimate df)."""
        x = student_t.ppf(np.clip(u, 1e-10, 1 - 1e-10), self.df)
        y = student_t.ppf(np.clip(v, 1e-10, 1 - 1e-10), self.df)
        self.rho = float(np.corrcoef(x, y)[0, 1])
        # Simple df estimation from kurtosis
        kurt = np.mean(student_t.ppf(np.clip(u, 1e-10, 1 - 1e-10), self.df)**4)
        self.df = max(float(6.0 / (kurt - 3.0)) if kurt > 3.0 else 30.0, 2.1)
        return self.rho, self.df


class ClaytonCopula:
    """Bivariate Clayton copula (asymmetric lower tail dependence)."""

    def __init__(self, theta: float = 1.0) -> None:
        self.theta = theta

    def cdf(self, u: float | NDArray, v: float | NDArray) -> float | NDArray:
        return np.maximum((u ** (-self.theta) + v ** (-self.theta) - 1) ** (-1.0 / self.theta), 0)

    def simulate(self, n: int, seed: int | None = None) -> tuple[NDArray, NDArray]:
        """Generate via conditional inverse method."""
        rng = np.random.default_rng(seed)
        w = rng.uniform(0, 1, n)
        t = self.theta
        u = rng.uniform(0, 1, n)
        v = (1 + u**(-t) * (w**(-t / (t + 1)) - 1)) ** (-1.0 / t)
        return u, np.clip(v, 0, 1)

    def tail_dependence(self) -> tuple[float, float]:
        if self.theta <= 0:
            return 0.0, 0.0
        lower = 2.0 ** (-1.0 / self.theta)
        return float(lower), 0.0  # Clayton has lower tail dependence only

    def fit(self, u: NDArray, v: NDArray) -> float:
        """Fit theta via MLE."""
        def neg_loglik(theta: float) -> float:
            if theta <= 0:
                return 1e10
            t = theta
            c = (u ** (-t) + v ** (-t) - 1) ** (-(1 + t) / t)
            ll = np.sum(np.log(c + 1e-300))
            return -ll

        from scipy.optimize import minimize_scalar
        result = minimize_scalar(neg_loglik, bounds=(0.01, 20.0), method="bounded")
        self.theta = float(result.x)
        return self.theta


class GumbelCopula:
    """Bivariate Gumbel copula (upper tail dependence)."""

    def __init__(self, theta: float = 2.0) -> None:
        self.theta = theta

    def cdf(self, u: float | NDArray, v: float | NDArray) -> float | NDArray:
        return np.exp(-(((-np.log(u)) ** self.theta + (-np.log(v)) ** self.theta) ** (1.0 / self.theta)))

    def tail_dependence(self) -> tuple[float, float]:
        if self.theta < 1:
            return 0.0, 0.0
        upper = 2.0 - 2.0 ** (1.0 / self.theta)
        return 0.0, float(upper)

    def fit(self, u: NDArray, v: NDArray) -> float:
        from scipy.optimize import minimize_scalar
        def neg_loglik(theta: float) -> float:
            if theta < 1:
                return 1e10
            self.theta = theta
            c = self.cdf(u, v)
            return -np.sum(np.log(c + 1e-300))
        result = minimize_scalar(neg_loglik, bounds=(1.01, 20.0), method="bounded")
        self.theta = float(result.x)
        return self.theta


class FrankCopula:
    """Bivariate Frank copula (symmetric, no tail dependence)."""

    def __init__(self, theta: float = 2.0) -> None:
        self.theta = theta

    def cdf(self, u: float | NDArray, v: float | NDArray) -> float | NDArray:
        t = self.theta
        return -1.0 / t * np.log(1 + (np.exp(-t * u) - 1) * (np.exp(-t * v) - 1) / (np.exp(-t) - 1))

    def tail_dependence(self) -> tuple[float, float]:
        return 0.0, 0.0  # Frank copula has no tail dependence

    def fit(self, u: NDArray, v: NDArray) -> float:
        from scipy.optimize import minimize_scalar
        def neg_loglik(theta: float) -> float:
            if abs(theta) < 1e-6:
                return 0.0
            self.theta = theta
            c = self.cdf(u, v)
            return -np.sum(np.log(c + 1e-300))
        result = minimize_scalar(neg_loglik, bounds=(-20.0, 20.0), method="bounded")
        self.theta = float(result.x)
        return self.theta


def fit_copula(
    u: NDArray,
    v: NDArray,
    family: str = "gaussian",
) -> GaussianCopula | StudentTCopula | ClaytonCopula | GumbelCopula | FrankCopula:
    """Fit a copula to uniform marginals."""
    copulas = {
        "gaussian": GaussianCopula,
        "t": StudentTCopula,
        "clayton": ClaytonCopula,
        "gumbel": GumbelCopula,
        "frank": FrankCopula,
    }
    copula = copulas[family.lower()]()
    copula.fit(u, v)
    return copula
