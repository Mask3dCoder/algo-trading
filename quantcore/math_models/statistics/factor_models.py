"""Factor models: PCA, ICA, Fama-French, Carhart."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA, FastICA


@dataclass
class PCAResult:
    """PCA result on returns matrix."""
    n_components: int
    explained_variance: NDArray
    explained_variance_ratio: NDArray
    components: NDArray  # (n_components, n_assets)
    loadings: NDArray    # (n_periods, n_components)


def pca_returns(
    returns_matrix: NDArray,
    n_components: int = 5,
) -> PCAResult:
    """PCA on asset returns matrix (T, N).

    Returns principal components, explained variance, and loadings.
    """
    X = np.asarray(returns_matrix, dtype=float)
    # Demean
    X_centered = X - X.mean(axis=0)

    pca = PCA(n_components=n_components)
    loadings = pca.fit_transform(X_centered)

    return PCAResult(
        n_components=n_components,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        components=pca.components_,
        loadings=loadings,
    )


@dataclass
class ICAResult:
    """ICA result on returns matrix."""
    n_components: int
    mixing_matrix: NDArray
    sources: NDArray


def ica_returns(
    returns_matrix: NDArray,
    n_components: int = 5,
) -> ICAResult:
    """ICA on asset returns matrix for independent factor extraction."""
    X = np.asarray(returns_matrix, dtype=float)
    X_centered = X - X.mean(axis=0)

    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
    sources = ica.fit_transform(X_centered)

    return ICAResult(
        n_components=n_components,
        mixing_matrix=ica.mixing_,
        sources=sources,
    )


@dataclass
class FFResult:
    """Fama-French factor regression result."""
    alpha: float
    betas: dict[str, float]
    r_squared: float
    residuals: NDArray
    n_factors: int


def fama_french(
    returns: NDArray,
    factors_df: NDArray,
    n_factors: int = 5,
) -> FFResult:
    """Fama-French factor model regression.

    r = alpha + beta_MKT*MKT + beta_SMB*SMB + beta_HML*HML + ... + epsilon
    factors_df: (T, K) array of factor returns
    """
    r = np.asarray(returns, dtype=float)
    F = np.asarray(factors_df, dtype=float)

    # Use only first n_factors columns
    F = F[:, :n_factors]
    n = len(r)

    # Add intercept
    X = np.column_stack([np.ones(n), F])
    # OLS
    coeffs, residuals_arr, _, _ = np.linalg.lstsq(X, r, rcond=None)

    alpha = float(coeffs[0])
    beta_names = ["MKT", "SMB", "HML", "RMW", "CMA", "UMD"][:n_factors]
    betas = dict(zip(beta_names, coeffs[1:]))

    fitted = X @ coeffs
    ss_res = np.sum((r - fitted)**2)
    ss_tot = np.sum((r - np.mean(r))**2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return FFResult(
        alpha=alpha,
        betas={k: float(v) for k, v in betas.items()},
        r_squared=float(r_sq),
        residuals=np.array(r - fitted),
        n_factors=n_factors,
    )


def carhart(
    returns: NDArray,
    factors_df: NDArray,
) -> FFResult:
    """Carhart 4-factor model: Fama-French 3-factor + momentum."""
    return fama_french(returns, factors_df, n_factors=4)


def download_ff_factors(
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    n_factors: int = 5,
) -> NDArray:
    """Download Fama-French factor data from Kenneth French data library.

    Requires `pandas_datareader` or direct CSV download.
    Returns (T, K) array of factor returns.
    """
    try:
        import pandas_datareader.data as web
        from pandas_datareader.famafrench import get_available_datasets
        dataset = "F-F_Research_Data_5_Factors_2x3"
        if n_factors == 3 or n_factors == 4:
            dataset = "F-F_Research_Data_Factors"

        df = web.DataReader(dataset, "famafrench", start=start, end=end)
        # Monthly data is index 0
        factor_data = df[0]
        return factor_data.values / 100.0  # Convert from percentage
    except ImportError:
        # Fallback: generate placeholder
        raise ImportError("pandas_datareader required for FF factor download. Install with: pip install pandas-datareader")
