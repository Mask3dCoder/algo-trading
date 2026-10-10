"""Advanced risk models for statistical arbitrage portfolios.

Implements:
  - PCA/ICA factor decomposition
  - Extreme Value Theory (EVT) for tail risk
  - Correlation regime detection
  - Dynamic conditional correlation (DCC-GARCH)
  - Risk decomposition (marginal, component, factor contribution)

References:
    - McNeil, Frey, Embrechts (2015): Quantitative Risk Management
    - Alexander (2008): Market Risk Analysis
    - Engle (2002): Dynamic Conditional Correlation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PCAResult:
    """PCA decomposition result."""
    components: NDArray  # Principal components (n_components, n_assets)
    explained_variance: NDArray  # Variance explained by each component
    explained_variance_ratio: NDArray  # Fraction of variance explained
    loadings: NDArray  # Factor loadings (n_assets, n_components)
    scores: NDArray  # Factor scores (n_periods, n_components)
    n_components: int


@dataclass
class TailRiskMetrics:
    """Tail risk metrics from EVT."""
    var_95: float  # 95% Value-at-Risk
    var_99: float  # 99% Value-at-Risk
    es_95: float  # 95% Expected Shortfall (CVaR)
    es_99: float  # 99% Expected Shortfall
    tail_index: float  # Shape parameter (ξ)
    scale_parameter: float  # Scale parameter (β)
    threshold: float  # Threshold for exceedances
    n_exceedances: int


@dataclass
class CorrelationRegime:
    """Correlation regime detection result."""
    regime_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    avg_correlation: float
    correlation_stability: float
    duration_days: int


@dataclass
class RiskDecomposition:
    """Portfolio risk decomposition."""
    total_risk: float  # Total portfolio volatility
    systematic_risk: float  # Factor/systematic risk
    idiosyncratic_risk: float  # Specific/idiosyncratic risk
    marginal_contributions: NDArray  # Marginal contribution to risk
    component_contributions: NDArray  # Component contribution to risk
    factor_contributions: NDArray  # Factor contribution to risk


# ---------------------------------------------------------------------------
# PCA Risk Model
# ---------------------------------------------------------------------------


class PCARiskModel:
    """Principal Component Analysis for factor decomposition.
    
    Extracts principal components from return data to:
      - Reduce dimensionality
      - Identify systematic risk factors
      - Decompose portfolio risk
      - Detect regime changes
    
    Parameters
    ----------
    n_components : int | None
        Number of components to retain. If None, keeps 95% variance.
    rolling_window : int | None
        If set, uses rolling PCA for time-varying factors.
    """

    def __init__(
        self,
        n_components: int | None = None,
        rolling_window: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.rolling_window = rolling_window
        self._pca_result: PCAResult | None = None
        self._is_fitted = False

    def fit(
        self,
        returns: NDArray,
        asset_names: list[str] | None = None,
    ) -> PCARiskModel:
        """Fit PCA model on return data.
        
        Parameters
        ----------
        returns : NDArray
            Return matrix (n_periods, n_assets).
        asset_names : list[str] | None
            Names for each asset.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Standardize returns — store scaler for OOS transform
        self.scaler = StandardScaler()
        returns_scaled = self.scaler.fit_transform(returns)

        # Determine number of components
        if self.n_components is None:
            # Keep components explaining 95% variance
            pca_full = PCA()
            pca_full.fit(returns_scaled)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumvar, 0.95) + 1)
            n_components = min(n_components, returns.shape[1])
        else:
            n_components = self.n_components

        # Fit PCA
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(returns_scaled)

        # Store results
        self._pca_result = PCAResult(
            components=pca.components_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            loadings=pca.components_.T,  # (n_assets, n_components)
            scores=scores,
            n_components=n_components,
        )

        self._is_fitted = True

        _log.info(
            "PCA fitted: %d components explain %.1f%% variance",
            n_components,
            np.sum(pca.explained_variance_ratio_) * 100,
        )

        return self

    def transform(self, returns: NDArray) -> NDArray:
        """Project returns onto principal components.

        Parameters
        ----------
        returns : NDArray
            Return matrix (n_periods, n_assets).

        Returns
        -------
        NDArray
            Factor scores (n_periods, n_components).
        """
        if not self._is_fitted or self._pca_result is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")

        # Use stored scaler from fit() — do NOT refit on OOS data (avoid lookahead bias)
        returns_scaled = self.scaler.transform(returns)

        # Direct projection using stored components — avoids sklearn PCA reconstruction issues
        scores = returns_scaled @ self._pca_result.components.T

        return scores

    def get_factor_loadings(self, asset_names: list[str] | None = None) -> pd.DataFrame:
        """Get factor loadings as DataFrame.
        
        Parameters
        ----------
        asset_names : list[str] | None
            Asset names for index.
            
        Returns
        -------
        pd.DataFrame
            Factor loadings (assets x components).
        """
        if not self._is_fitted or self._pca_result is None:
            raise RuntimeError("PCA not fitted.")

        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(self._pca_result.loadings.shape[0])]

        component_names = [f"PC_{i+1}" for i in range(self._pca_result.n_components)]

        return pd.DataFrame(
            self._pca_result.loadings,
            index=asset_names,
            columns=component_names,
        )

    def compute_factor_risk(
        self,
        weights: NDArray,
        returns: NDArray,
    ) -> dict[str, float]:
        """Compute risk attributable to each factor.
        
        Parameters
        ----------
        weights : NDArray
            Portfolio weights (n_assets,).
        returns : NDArray
            Return matrix (n_periods, n_assets).
            
        Returns
        -------
        dict[str, float]
            Risk contribution per factor.
        """
        if not self._is_fitted or self._pca_result is None:
            raise RuntimeError("PCA not fitted.")

        # Project returns onto factors
        factor_returns = self.transform(returns)

        # Portfolio factor exposures
        factor_exposures = self._pca_result.loadings.T @ weights

        # Factor risk contributions
        factor_vols = np.std(factor_returns, axis=0)
        factor_risks = factor_exposures * factor_vols

        total_risk = np.sqrt(np.sum(factor_risks**2))

        contributions = {}
        for i in range(self._pca_result.n_components):
            contributions[f"PC_{i+1}"] = {
                "exposure": float(factor_exposures[i]),
                "volatility": float(factor_vols[i]),
                "risk_contribution": float(factor_risks[i]),
                "pct_of_total": float(factor_risks[i] / total_risk * 100) if total_risk > 0 else 0.0,
            }

        return contributions


class ICARiskModel:
    """Independent Component Analysis for non-Gaussian factors.
    
    ICA finds statistically independent factors, which may better
    capture tail dependencies and non-linear relationships.
    
    Parameters
    ----------
    n_components : int | None
        Number of independent components.
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = n_components
        self._ica_result = None
        self._is_fitted = False

    def fit(
        self,
        returns: NDArray,
        asset_names: list[str] | None = None,
    ) -> ICARiskModel:
        """Fit ICA model on return data."""
        from sklearn.decomposition import FastICA

        n_components = self.n_components or min(returns.shape[1], 10)

        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        sources = ica.fit_transform(returns)

        self._ica_result = {
            "mixing": ica.mixing_,
            "sources": sources,
            "components": ica.components_,
        }
        self._is_fitted = True

        _log.info("ICA fitted with %d independent components", n_components)

        return self

    def transform(self, returns: NDArray) -> NDArray:
        """Project returns onto independent components."""
        if not self._is_fitted or self._ica_result is None:
            raise RuntimeError("ICA not fitted. Call fit() first.")

        # Direct projection using stored components
        # Note: ICA components are the demixing matrix; sources = returns @ components.T
        scores = returns @ self._ica_result["components"].T
        # Center (FastICA internally whitens and centers)
        scores -= np.mean(scores, axis=0, keepdims=True)

        return scores


# ---------------------------------------------------------------------------
# Extreme Value Theory
# ---------------------------------------------------------------------------


class TailRiskModel:
    """Extreme Value Theory for tail risk estimation.
    
    Uses the Peaks-Over-Threshold (POT) method with Generalized
    Pareto Distribution (GPD) to model extreme losses.
    
    Parameters
    ----------
    threshold_percentile : float
        Percentile for threshold selection (default 0.05 = 5% worst losses).
    """

    def __init__(self, threshold_percentile: float = 0.05) -> None:
        self.threshold_percentile = threshold_percentile
        self._gpd_params: tuple[float, float] | None = None
        self._threshold: float | None = None
        self._is_fitted = False

    def fit(
        self,
        returns: NDArray,
        confidence_levels: list[float] | None = None,
    ) -> TailRiskModel:
        """Fit GPD to extreme losses.
        
        Parameters
        ----------
        returns : NDArray
            Return series (n_periods,).
        confidence_levels : list[float] | None
            VaR/ES confidence levels (default [0.95, 0.99]).
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        # Use losses (negative returns)
        losses = -returns
        losses = losses[np.isfinite(losses)]
        self._n_observations = len(losses)  # store for EVT VaR computation

        # Select threshold
        self._threshold = np.percentile(losses, 100 * (1 - self.threshold_percentile))

        # Exceedances
        exceedances = losses[losses > self._threshold] - self._threshold
        n_exceedances = len(exceedances)

        if n_exceedances < 10:
            _log.warning("Too few exceedances (%d). Using empirical VaR.", n_exceedances)
            self._gpd_params = (0.0, np.std(losses))
            self._is_fitted = True
            return self

        # Fit GPD using MLE
        try:
            # Method of moments initial estimate
            mean_excess = np.mean(exceedances)
            var_excess = np.var(exceedances)

            # Method of moments estimates
            xi_mom = 0.5 * (1 - mean_excess**2 / var_excess)
            beta_mom = 0.5 * mean_excess * (1 + mean_excess**2 / var_excess)

            # MLE refinement
            def neg_log_likelihood(params):
                xi, beta = params
                if beta <= 0:
                    return 1e10
                if abs(xi) < 1e-10:
                    return len(exceedances) * np.log(beta) + np.sum(exceedances) / beta
                n = len(exceedances)
                z = 1 + xi * exceedances / beta
                if np.any(z <= 0):
                    return 1e10
                return n * np.log(beta) + (1 + 1/xi) * np.sum(np.log(z))

            result = minimize(
                neg_log_likelihood,
                [xi_mom, beta_mom],
                method="Nelder-Mead",
            )

            self._gpd_params = (result.x[0], result.x[1])

        except Exception as e:
            _log.warning("GPD fit failed: %s. Using method of moments.", e)
            self._gpd_params = (xi_mom, beta_mom)

        self._is_fitted = True

        _log.info(
            "EVT fitted: threshold=%.4f, exceedances=%d, ξ=%.4f, β=%.4f",
            self._threshold, n_exceedances, self._gpd_params[0], self._gpd_params[1],
        )

        return self

    def compute_var(
        self,
        confidence: float = 0.99,
        n_periods: int = 1,
    ) -> float:
        """Compute Value-at-Risk using fitted GPD.
        
        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.99 for 99% VaR).
        n_periods : int
            Holding period in days.
            
        Returns
        -------
        float
            VaR estimate.
        """
        if not self._is_fitted or self._gpd_params is None or self._threshold is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        xi, beta = self._gpd_params
        # Use actual observation count from fit() instead of hardcoded 1000
        n = getattr(self, '_n_observations', 1000)
        n_exceedances = int(n * self.threshold_percentile)

        # VaR formula from GPD
        p = 1 - confidence
        if abs(xi) < 1e-10:
            var = self._threshold + beta * np.log(n_exceedances / (n * p))
        else:
            var = self._threshold + (beta / xi) * ((n_exceedances / (n * p))**xi - 1)

        # Scale to holding period (square root of time rule)
        var_scaled = var * np.sqrt(n_periods)

        return float(var_scaled)

    def compute_es(
        self,
        confidence: float = 0.99,
        n_periods: int = 1,
    ) -> float:
        """Compute Expected Shortfall (CVaR) using fitted GPD.
        
        Parameters
        ----------
        confidence : float
            Confidence level.
        n_periods : int
            Holding period.
            
        Returns
        -------
        float
            ES estimate.
        """
        if not self._is_fitted or self._gpd_params is None:
            raise RuntimeError("Model not fitted.")

        var = self.compute_var(confidence, n_periods)
        xi, beta = self._gpd_params

        # ES formula from GPD
        if xi < 1:
            es = var / (1 - xi) + (beta - xi * self._threshold) / (1 - xi)
        else:
            es = var * 1.5  # Approximation for ξ ≥ 1

        return float(es)

    def compute_tail_metrics(self, n_periods: int = 1) -> TailRiskMetrics:
        """Compute comprehensive tail risk metrics.
        
        Parameters
        ----------
        n_periods : int
            Holding period.
            
        Returns
        -------
        TailRiskMetrics
            Tail risk metrics.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        return TailRiskMetrics(
            var_95=self.compute_var(0.95, n_periods),
            var_99=self.compute_var(0.99, n_periods),
            es_95=self.compute_es(0.95, n_periods),
            es_99=self.compute_es(0.99, n_periods),
            tail_index=self._gpd_params[0] if self._gpd_params else 0.0,
            scale_parameter=self._gpd_params[1] if self._gpd_params else 0.0,
            threshold=self._threshold if self._threshold else 0.0,
            n_exceedances=0,  # Would need to store during fit
        )


# ---------------------------------------------------------------------------
# Correlation Regime Detection
# ---------------------------------------------------------------------------


class CorrelationRegimeDetector:
    """Detects correlation regime changes in asset returns.
    
    Uses rolling correlation dynamics and breakpoint detection
    to identify when correlation structure changes significantly.
    
    Parameters
    ----------
    lookback : int
        Rolling window for correlation estimation (default 60).
    breakpoint_threshold : float
        Threshold for breakpoint detection (default 0.3).
    """

    def __init__(
        self,
        lookback: int = 60,
        breakpoint_threshold: float = 0.3,
    ) -> None:
        self.lookback = lookback
        self.breakpoint_threshold = breakpoint_threshold
        self._regimes: list[CorrelationRegime] = []

    def detect_regimes(
        self,
        returns: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> list[CorrelationRegime]:
        """Detect correlation regimes.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return matrix (n_periods, n_assets).
        dates : pd.DatetimeIndex
            Date index.
            
        Returns
        -------
        list[CorrelationRegime]
            Detected regimes.
        """
        n_periods = len(returns)
        if n_periods < self.lookback * 2:
            _log.warning("Insufficient data for regime detection")
            return []

        # Compute rolling correlation matrices
        rolling_corrs = []
        for i in range(self.lookback, n_periods):
            window = returns.iloc[i - self.lookback:i]
            corr = window.corr()
            rolling_corrs.append(corr.values)

        # Compute average correlation over time
        avg_corrs = []
        for corr in rolling_corrs:
            # Average off-diagonal elements
            n = corr.shape[0]
            mask = np.ones((n, n), dtype=bool)
            np.fill_diagonal(mask, False)
            avg_corrs.append(np.mean(corr[mask]))

        avg_corrs = np.array(avg_corrs)

        # Detect breakpoints (sudden changes in average correlation)
        regimes = []
        current_regime_start = 0

        for i in range(1, len(avg_corrs)):
            if abs(avg_corrs[i] - avg_corrs[i-1]) > self.breakpoint_threshold:
                # Breakpoint detected
                regime_idx = len(regimes)
                start_date = dates[self.lookback + current_regime_start]
                end_date = dates[self.lookback + i - 1]

                # Compute regime statistics
                regime_corrs = avg_corrs[current_regime_start:i]
                stability = 1.0 - np.std(regime_corrs) / (np.mean(regime_corrs) + 1e-10)

                regimes.append(CorrelationRegime(
                    regime_id=regime_idx,
                    start_date=start_date,
                    end_date=end_date,
                    avg_correlation=float(np.mean(regime_corrs)),
                    correlation_stability=float(stability),
                    duration_days=int((end_date - start_date).days),
                ))

                current_regime_start = i

        # Final regime
        if current_regime_start < len(avg_corrs):
            regime_idx = len(regimes)
            start_date = dates[self.lookback + current_regime_start]
            end_date = dates[-1]

            regime_corrs = avg_corrs[current_regime_start:]
            stability = 1.0 - np.std(regime_corrs) / (np.mean(regime_corrs) + 1e-10)

            regimes.append(CorrelationRegime(
                regime_id=regime_idx,
                start_date=start_date,
                end_date=end_date,
                avg_correlation=float(np.mean(regime_corrs)),
                correlation_stability=float(stability),
                duration_days=int((end_date - start_date).days),
            ))

        self._regimes = regimes

        _log.info("Detected %d correlation regimes", len(regimes))

        return regimes

    def get_current_regime(self, date: pd.Timestamp) -> CorrelationRegime | None:
        """Get regime for a specific date."""
        for regime in self._regimes:
            if regime.start_date <= date <= regime.end_date:
                return regime
        return None


class DCCGARCHModel:
    """Dynamic Conditional Correlation GARCH model.
    
    Models time-varying correlations that respond to market conditions.
    
    Parameters
    ----------
    garch_order : tuple[int, int]
        GARCH(p,q) order (default (1,1)).
    dcc_order : tuple[int, int]
        DCC(a,b) order (default (1,1)).
    """

    def __init__(
        self,
        garch_order: tuple[int, int] = (1, 1),
        dcc_order: tuple[int, int] = (1, 1),
    ) -> None:
        self.garch_order = garch_order
        self.dcc_order = dcc_order
        self._is_fitted = False
        self._params = None

    def fit(self, returns: NDArray) -> DCCGARCHModel:
        """Fit DCC-GARCH model.
        
        Parameters
        ----------
        returns : NDArray
            Return matrix (n_periods, n_assets).
        """
        # Simplified DCC-GARCH implementation
        # In production, use arch library or similar

        n_periods, n_assets = returns.shape

        # Step 1: Fit univariate GARCH for each asset
        garch_params = []
        standardized_returns = np.zeros_like(returns)

        for i in range(n_assets):
            # Simple GARCH(1,1) estimation
            # sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
            r = returns[:, i]
            r_centered = r - np.mean(r)

            # Method of moments estimates
            var_r = np.var(r_centered)
            alpha = 0.1  # Typical value
            beta = 0.85  # Typical value
            omega = var_r * (1 - alpha - beta)

            garch_params.append((omega, alpha, beta))

            # Compute conditional volatility
            h = np.zeros(n_periods)
            h[0] = var_r
            for t in range(1, n_periods):
                h[t] = omega + alpha * r_centered[t-1]**2 + beta * h[t-1]

            # Standardize returns
            standardized_returns[:, i] = r_centered / np.sqrt(h + 1e-10)

        # Step 2: Estimate DCC parameters
        # Q_bar = unconditional correlation of standardized returns
        Q_bar = np.corrcoef(standardized_returns.T)

        # DCC parameters (simplified)
        a, b = 0.05, 0.90  # Typical DCC(1,1) values

        self._params = {
            "garch": garch_params,
            "Q_bar": Q_bar,
            "dcc_a": a,
            "dcc_b": b,
        }
        self._is_fitted = True

        _log.info("DCC-GARCH fitted for %d assets", n_assets)

        return self

    def compute_dynamic_correlation(
        self,
        returns: NDArray,
    ) -> NDArray:
        """Compute time-varying correlation matrices.
        
        Parameters
        ----------
        returns : NDArray
            Return matrix (n_periods, n_assets).
            
        Returns
        -------
        NDArray
            Correlation matrices (n_periods, n_assets, n_assets).
        """
        if not self._is_fitted or self._params is None:
            raise RuntimeError("Model not fitted.")

        n_periods, n_assets = returns.shape
        Q_bar = self._params["Q_bar"]
        a = self._params["dcc_a"]
        b = self._params["dcc_b"]

        # Initialize
        Q_t = Q_bar.copy()
        correlations = np.zeros((n_periods, n_assets, n_assets))

        # Standardize returns (simplified)
        r_std = (returns - np.mean(returns, axis=0)) / (np.std(returns, axis=0) + 1e-10)

        for t in range(n_periods):
            # DCC recursion: Q_t = (1-a-b) * Q_bar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
            z = r_std[t:t+1, :].T
            Q_t = (1 - a - b) * Q_bar + a * z @ z.T + b * Q_t

            # Convert to correlation
            D_t = np.sqrt(np.diag(np.diag(Q_t)))
            D_inv = np.linalg.inv(D_t + np.eye(n_assets) * 1e-10)
            R_t = D_inv @ Q_t @ D_inv

            correlations[t] = R_t

        return correlations


# ---------------------------------------------------------------------------
# Risk Decomposition
# ---------------------------------------------------------------------------


class RiskDecomposer:
    """Decomposes portfolio risk into systematic and idiosyncratic components.
    
    Parameters
    ----------
    n_factors : int | None
        Number of risk factors (default: auto-select via PCA).
    """

    def __init__(self, n_factors: int | None = None) -> None:
        self.n_factors = n_factors
        self._pca_model = PCARiskModel(n_components=n_factors)

    def decompose(
        self,
        weights: NDArray,
        returns: NDArray,
        asset_names: list[str] | None = None,
    ) -> RiskDecomposition:
        """Decompose portfolio risk.
        
        Parameters
        ----------
        weights : NDArray
            Portfolio weights (n_assets,).
        returns : NDArray
            Return matrix (n_periods, n_assets).
        asset_names : list[str] | None
            Asset names.
            
        Returns
        -------
        RiskDecomposition
            Risk decomposition.
        """
        # Fit PCA model
        self._pca_model.fit(returns, asset_names)

        # Get factor exposures
        factor_exposures = self._pca_model._pca_result.loadings.T @ weights

        # Factor returns
        factor_returns = self._pca_model.transform(returns)

        # Factor covariance (ensure at least 2D for single-asset edge case)
        factor_cov = np.atleast_2d(np.cov(factor_returns.T))

        # Portfolio factor risk
        systematic_var = factor_exposures @ factor_cov @ factor_exposures
        systematic_risk = np.sqrt(systematic_var)

        # Total portfolio risk
        port_returns = returns @ weights
        total_risk = np.std(port_returns)

        # Idiosyncratic risk
        idiosyncratic_risk = np.sqrt(max(total_risk**2 - systematic_risk**2, 0))

        # Marginal contribution to risk
        port_vol = total_risk
        cov_matrix = np.atleast_2d(np.cov(returns.T))
        marginal_contrib = ((cov_matrix @ weights[:, None]) / port_vol).ravel()

        # Component contribution to risk
        component_contrib = weights * marginal_contrib

        # Factor contribution
        factor_contrib = factor_exposures * np.sqrt(np.diag(factor_cov))

        return RiskDecomposition(
            total_risk=float(total_risk),
            systematic_risk=float(systematic_risk),
            idiosyncratic_risk=float(idiosyncratic_risk),
            marginal_contributions=marginal_contrib,
            component_contributions=component_contrib,
            factor_contributions=factor_contrib,
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_portfolio_var(
    weights: NDArray,
    returns: NDArray,
    confidence: float = 0.99,
    method: str = "parametric",
    n_periods: int = 1,
) -> float:
    """Compute portfolio Value-at-Risk.
    
    Parameters
    ----------
    weights : NDArray
        Portfolio weights.
    returns : NDArray
        Return matrix.
    confidence : float
        Confidence level.
    method : str
        'parametric', 'historical', or 'evt'.
    n_periods : int
        Holding period.
        
    Returns
    -------
    float
        VaR estimate.
    """
    port_returns = returns @ weights

    if method == "parametric":
        mu = np.mean(port_returns)
        sigma = np.std(port_returns)
        z = stats.norm.ppf(1 - confidence)
        var = -(mu + z * sigma) * np.sqrt(n_periods)

    elif method == "historical":
        var = -np.percentile(port_returns, (1 - confidence) * 100) * np.sqrt(n_periods)

    elif method == "evt":
        model = TailRiskModel()
        model.fit(port_returns)
        var = model.compute_var(confidence, n_periods)

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(var)


def compute_portfolio_es(
    weights: NDArray,
    returns: NDArray,
    confidence: float = 0.99,
    method: str = "historical",
    n_periods: int = 1,
) -> float:
    """Compute portfolio Expected Shortfall.
    
    Parameters
    ----------
    weights : NDArray
        Portfolio weights.
    returns : NDArray
        Return matrix.
    confidence : float
        Confidence level.
    method : str
        'historical' or 'evt'.
    n_periods : int
        Holding period.
        
    Returns
    -------
    float
        ES estimate.
    """
    port_returns = returns @ weights

    if method == "historical":
        var = np.percentile(port_returns, (1 - confidence) * 100)
        es = np.mean(port_returns[port_returns <= var]) * np.sqrt(n_periods)

    elif method == "evt":
        model = TailRiskModel()
        model.fit(port_returns)
        es = -model.compute_es(confidence, n_periods)  # EVT returns positive loss → negate for convention

    else:
        raise ValueError(f"Unknown method: {method}")

    return float(-es)
