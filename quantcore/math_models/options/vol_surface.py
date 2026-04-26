"""SVI/SSVI volatility surface parameterization and arbitrage checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass
class SVIParams:
    """Raw SVI parameters: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    a: float
    b: float
    rho: float  # in [-1, 1]
    m: float
    sigma: float  # > 0


def svi_total_variance(k: NDArray | float, p: SVIParams) -> NDArray | float:
    """Compute total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sig^2))."""
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x**2 + p.sigma**2))


def svi_implied_vol(
    k: NDArray | float,
    p: SVIParams,
    T: float,
) -> NDArray | float:
    """Implied vol from SVI: σ(k,T) = sqrt(w(k)/T)."""
    return np.sqrt(svi_total_variance(k, p) / T)


def fit_svi(
    strikes: NDArray,
    ivs: NDArray,
    forward: float,
    T: float,
    initial_params: SVIParams | None = None,
) -> SVIParams:
    """Fit raw SVI to market implied vols via least squares.

    strikes: log-moneyness k or absolute strikes (converted to log-moneyness)
    ivs: implied volatilities
    forward: forward price
    T: time to expiry in years
    """
    k = np.asarray(strikes, dtype=float)
    vols = np.asarray(ivs, dtype=float)
    # If strikes are absolute, convert to log-moneyness
    if np.any(k > 5.0):
        k = np.log(k / forward)

    total_var = vols**2 * T

    if initial_params is None:
        initial_params = SVIParams(
            a=max(np.min(total_var) * 0.9, 1e-6),
            b=max(np.max(total_var) - np.min(total_var), 0.01),
            rho=0.0,
            m=0.0,
            sigma=0.1,
        )

    x0 = np.array([initial_params.a, initial_params.b, initial_params.rho, initial_params.m, initial_params.sigma])

    def objective(x: NDArray) -> float:
        p = SVIParams(a=x[0], b=x[1], rho=np.clip(x[2], -0.999, 0.999), m=x[3], sigma=max(x[4], 1e-8))
        w = svi_total_variance(k, p)
        return float(np.sum((w - total_var) ** 2))

    bounds = [
        (1e-8, None),   # a > 0
        (1e-8, None),   # b > 0
        (-0.999, 0.999),  # rho in (-1, 1)
        (-2.0, 2.0),    # m
        (1e-8, 2.0),    # sigma > 0
    ]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})

    return SVIParams(
        a=result.x[0],
        b=result.x[1],
        rho=np.clip(result.x[2], -0.999, 0.999),
        m=result.x[3],
        sigma=max(result.x[4], 1e-8),
    )


@dataclass
class SSVISurface:
    """SSVI surface: collection of SVI slices across expiries."""
    expiries: list[float] = field(default_factory=list)
    slices: list[SVIParams] = field(default_factory=list)

    def iv(self, k: float | NDArray, T: float) -> float | NDArray:
        """Interpolate IV at arbitrary (k, T) via linear interpolation in total variance."""
        if len(self.expiries) < 2:
            if len(self.expiries) == 1:
                return float(svi_implied_vol(k, self.slices[0], self.expiries[0]))
            return np.nan

        k_arr = np.asarray(k, dtype=float) if isinstance(k, (list, np.ndarray)) else float(k)

        # Find bracketing expiries
        T_lo, T_hi = self.expiries[0], self.expiries[-1]
        idx_lo, idx_hi = 0, len(self.expiries) - 1

        for i, exp in enumerate(self.expiries):
            if exp <= T:
                T_lo = exp
                idx_lo = i
            if exp >= T and idx_hi == len(self.expiries) - 1:
                T_hi = exp
                idx_hi = i

        if T_lo >= T:
            return svi_implied_vol(k_arr, self.slices[idx_lo], T_lo)
        if T_hi <= T:
            return svi_implied_vol(k_arr, self.slices[idx_hi], T_hi)

        w_lo = svi_total_variance(k_arr, self.slices[idx_lo])
        w_hi = svi_total_variance(k_arr, self.slices[idx_hi])
        # Linear interpolation in total variance
        alpha = (T - T_lo) / (T_hi - T_lo)
        w_interp = w_lo * (1.0 - alpha) + w_hi * alpha
        return np.sqrt(w_interp / T)


def check_butterfly_arbitrage(
    k_grid: NDArray,
    params: SVIParams,
    tol: float = 1e-8,
) -> list[tuple[float, float]]:
    """Check butterfly arbitrage: g(k) = (1 - k*dw/dk)^2 - (-d²w/dk²)*w ≥ 0.

    Returns list of (k, g(k)) violations where g < -tol.
    """
    dk = k_grid[1] - k_grid[0] if len(k_grid) > 1 else 0.01
    w = svi_total_variance(k_grid, params)
    violations = []
    for i in range(1, len(k_grid) - 1):
        dwdk = (w[i + 1] - w[i - 1]) / (2.0 * dk)
        d2wdk2 = (w[i + 1] - 2.0 * w[i] + w[i - 1]) / (dk**2)
        g = (1.0 - k_grid[i] * dwdk) ** 2 - (-d2wdk2) * w[i] if d2wdk2 < 0 else 1.0
        # If d2w/dk2 >= 0, the slice is convex — always butterfly-arb-free
        if d2wdk2 < 0 and g < -tol:
            violations.append((float(k_grid[i]), float(g)))
    return violations


def check_calendar_spread(
    surface: SSVISurface,
    k_grid: NDArray,
    tol: float = 1e-8,
) -> list[tuple[float, int, int]]:
    """Check calendar spread arbitrage: w(k,T1) <= w(k,T2) for T1 < T2.

    Returns list of (k, slice_idx_1, slice_idx_2) violations.
    """
    violations = []
    for i in range(len(surface.slices) - 1):
        w1 = svi_total_variance(k_grid, surface.slices[i])
        w2 = svi_total_variance(k_grid, surface.slices[i + 1])
        for j in range(len(k_grid)):
            if w2[j] < w1[j] - tol:
                violations.append((float(k_grid[j]), i, i + 1))
    return violations


def plot_surface(surface: SSVISurface, k_range: tuple = (-1.0, 1.0), n_k: int = 50):
    """3D Plotly volatility surface visualization."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("plotly required for surface visualization") from e

    k_grid = np.linspace(k_range[0], k_range[1], n_k)
    K, T = np.meshgrid(k_grid, surface.expiries)
    IV = np.zeros_like(K)
    for i, exp in enumerate(surface.expiries):
        IV[i, :] = svi_implied_vol(k_grid, surface.slices[i], exp)

    fig = go.Figure(data=[go.Surface(x=K, y=T, z=IV * 100, colorscale="Viridis")])
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(xaxis_title="Log-Moneyness", yaxis_title="Expiry (yrs)", zaxis_title="IV (%)"),
    )
    return fig
