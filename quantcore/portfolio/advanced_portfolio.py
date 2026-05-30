"""Advanced portfolio construction for statistical arbitrage.

Implements:
  - Black-Litterman optimization
  - Risk parity (equal risk contribution)
  - Hierarchical Risk Parity (HRP)
  - Mean-CVaR optimization
  - Transaction cost-aware rebalancing

References:
    - Black & Litterman (1992): Global Portfolio Optimization
    - Lopez de Prado (2016): Building Diversified Portfolios that Outperform OOS
    - Rockafellar & Uryasev (2000): Optimization of Conditional Value-at-Risk
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PortfolioResult:
    """Portfolio optimization result."""
    weights: NDArray
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    turnover: float
    n_assets: int


@dataclass
class BlackLittermanResult:
    """Black-Litterman optimization result."""
    posterior_returns: NDArray
    posterior_covariance: NDArray
    weights: NDArray
    view_returns: NDArray
    confidence: NDArray


# ---------------------------------------------------------------------------
# Black-Litterman Optimization
# ---------------------------------------------------------------------------


class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimization.
    
    Combines market equilibrium returns with investor views to
    produce posterior returns and optimal weights.
    
    Parameters
    ----------
    risk_aversion : float
        Risk aversion parameter (default 2.5).
    tau : float
        Uncertainty in prior (default 0.05).
    """
    
    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.tau = tau
        
    def compute_equilibrium_returns(
        self,
        market_weights: NDArray,
        covariance: NDArray,
    ) -> NDArray:
        """Compute equilibrium (prior) returns from market weights.
        
        Parameters
        ----------
        market_weights : NDArray
            Market capitalization weights.
        covariance : NDArray
            Asset covariance matrix.
            
        Returns
        -------
        NDArray
            Implied equilibrium returns.
        """
        # π = δ * Σ * w_mkt
        return self.risk_aversion * covariance @ market_weights
    
    def incorporate_views(
        self,
        equilibrium_returns: NDArray,
        covariance: NDArray,
        view_matrix: NDArray,
        view_returns: NDArray,
        view_confidence: NDArray,
    ) -> BlackLittermanResult:
        """Incorporate investor views using Black-Litterman formula.
        
        Parameters
        ----------
        equilibrium_returns : NDArray
            Prior equilibrium returns.
        covariance : NDArray
            Asset covariance matrix.
        view_matrix : NDArray
            Pick matrix P (n_views, n_assets).
        view_returns : NDArray
            Expected returns from views.
        view_confidence : NDArray
            Confidence in each view [0, 1].
            
        Returns
        -------
        BlackLittermanResult
            Posterior returns and weights.
        """
        n_assets = len(equilibrium_returns)
        
        # Prior covariance (scaled)
        tau_sigma = self.tau * covariance
        
        # View uncertainty (diagonal matrix)
        omega = np.diag(1.0 / (view_confidence + 1e-10))
        
        # Posterior precision
        M = np.linalg.inv(tau_sigma) + view_matrix.T @ np.linalg.inv(omega) @ view_matrix
        
        # Posterior returns
        posterior_returns = np.linalg.solve(
            M,
            np.linalg.inv(tau_sigma) @ equilibrium_returns + view_matrix.T @ np.linalg.inv(omega) @ view_returns,
        )
        
        # Posterior covariance
        posterior_cov = np.linalg.inv(M)
        
        # Optimal weights (mean-variance)
        weights = np.linalg.solve(
            self.risk_aversion * covariance,
            posterior_returns,
        )
        
        # Normalize weights
        weights = np.maximum(weights, 0)  # Long-only
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        
        return BlackLittermanResult(
            posterior_returns=posterior_returns,
            posterior_covariance=posterior_cov,
            weights=weights,
            view_returns=view_returns,
            confidence=view_confidence,
        )
    
    def optimize(
        self,
        market_weights: NDArray,
        covariance: NDArray,
        view_matrix: Optional[NDArray] = None,
        view_returns: Optional[NDArray] = None,
        view_confidence: Optional[NDArray] = None,
        alpha_signals: Optional[NDArray] = None,
        alpha_confidence: float = 0.5,
    ) -> PortfolioResult:
        """Run Black-Litterman optimization.
        
        Parameters
        ----------
        market_weights : NDArray
            Market cap weights.
        covariance : NDArray
            Covariance matrix.
        view_matrix : NDArray | None
            Custom view matrix. If None, uses alpha signals.
        view_returns : NDArray | None
            Custom view returns.
        view_confidence : NDArray | None
            Custom view confidence.
        alpha_signals : NDArray | None
            Alpha signals (used as views if view_matrix is None).
        alpha_confidence : float
            Confidence in alpha signals.
            
        Returns
        -------
        PortfolioResult
            Optimized portfolio.
        """
        n_assets = len(market_weights)
        
        # Compute equilibrium returns
        pi = self.compute_equilibrium_returns(market_weights, covariance)
        
        # If no custom views, use alpha signals as views
        if view_matrix is None and alpha_signals is not None:
            # Each asset is a view
            view_matrix = np.eye(n_assets)
            view_returns = alpha_signals * np.std(pi)  # Scale to prior magnitude
            view_confidence = np.ones(n_assets) * alpha_confidence
        
        if view_matrix is not None and view_returns is not None and view_confidence is not None:
            # Incorporate views
            result = self.incorporate_views(
                pi, covariance, view_matrix, view_returns, view_confidence,
            )
            weights = result.weights
        else:
            # No views - use equilibrium
            weights = market_weights
        
        # Compute portfolio metrics
        port_return = float(weights @ pi)
        port_risk = float(np.sqrt(weights @ covariance @ weights))
        sharpe = port_return / port_risk if port_risk > 0 else 0.0
        
        return PortfolioResult(
            weights=weights,
            expected_return=port_return,
            expected_risk=port_risk,
            sharpe_ratio=sharpe,
            turnover=0.0,
            n_assets=n_assets,
        )


# ---------------------------------------------------------------------------
# Risk Parity
# ---------------------------------------------------------------------------


class RiskParityOptimizer:
    """Risk parity portfolio optimization.
    
    Allocates capital such that each asset contributes equally
    to total portfolio risk.
    
    Parameters
    ----------
    target_risk_contributions : NDArray | None
        Target risk contributions. Default: equal (1/n).
    """
    
    def __init__(
        self,
        target_risk_contributions: Optional[NDArray] = None,
    ) -> None:
        self.target_risk_contributions = target_risk_contributions
        
    def compute_risk_contributions(
        self,
        weights: NDArray,
        covariance: NDArray,
    ) -> NDArray:
        """Compute marginal and component risk contributions.
        
        Parameters
        ----------
        weights : NDArray
            Portfolio weights.
        covariance : NDArray
            Covariance matrix.
            
        Returns
        -------
        NDArray
            Component risk contributions.
        """
        port_vol = np.sqrt(weights @ covariance @ weights)
        marginal_contrib = covariance @ weights / port_vol
        component_contrib = weights * marginal_contrib
        
        return component_contrib
    
    def optimize(
        self,
        covariance: NDArray,
        asset_names: Optional[list[str]] = None,
    ) -> PortfolioResult:
        """Find risk parity portfolio.
        
        Parameters
        ----------
        covariance : NDArray
            Covariance matrix.
        asset_names : list[str] | None
            Asset names.
            
        Returns
        -------
        PortfolioResult
            Risk parity portfolio.
        """
        n_assets = len(covariance)
        
        # Target: equal risk contribution
        if self.target_risk_contributions is None:
            target = np.ones(n_assets) / n_assets
        else:
            target = self.target_risk_contributions
        
        # Objective: minimize difference between actual and target risk contributions
        def objective(w):
            w = np.abs(w)  # Ensure positive
            if np.sum(w) == 0:
                return 1e10
            w = w / np.sum(w)  # Normalize
            
            rc = self.compute_risk_contributions(w, covariance)
            total_risk = np.sum(rc)
            
            if total_risk == 0:
                return 1e10
            
            rc_pct = rc / total_risk
            
            # Sum of squared differences
            return np.sum((rc_pct - target) ** 2)
        
        # Initial guess: inverse volatility
        iv = 1.0 / np.sqrt(np.diag(covariance))
        w0 = iv / np.sum(iv)
        
        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        
        # Bounds: long-only
        bounds = [(0.001, 1.0)] * n_assets
        
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = np.abs(result.x)
        weights /= np.sum(weights)
        
        # Compute portfolio metrics
        port_return = 0.0  # Would need expected returns
        port_risk = float(np.sqrt(weights @ covariance @ weights))
        
        return PortfolioResult(
            weights=weights,
            expected_return=port_return,
            expected_risk=port_risk,
            sharpe_ratio=0.0,
            turnover=0.0,
            n_assets=n_assets,
        )


class HierarchicalRiskParity:
    """Hierarchical Risk Parity (HRP) from Lopez de Prado (2016).
    
    Uses hierarchical clustering to build a diversified portfolio
    without matrix inversion.
    
    Parameters
    ----------
    linkage_method : str
        Linkage method for clustering (default 'ward').
    """
    
    def __init__(self, linkage_method: str = "ward") -> None:
        self.linkage_method = linkage_method
        
    def _tree_clustering(self, correlation: NDArray) -> tuple:
        """Perform hierarchical clustering."""
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        
        # Distance matrix from correlation
        dist = np.sqrt(0.5 * (1 - correlation))
        dist_condensed = squareform(dist, checks=False)
        
        # Hierarchical clustering
        link = linkage(dist_condensed, method=self.linkage_method)
        sort_idx = leaves_list(link)
        
        return link, sort_idx
    
    def _get_quasi_diag(self, link: NDArray) -> list[int]:
        """Get quasi-diagonal ordering."""
        from scipy.cluster.hierarchy import leaves_list
        return leaves_list(link).tolist()
    
    def _recursive_bisection(
        self,
        covariance: NDArray,
        sort_idx: list[int],
    ) -> NDArray:
        """Recursive bisection to allocate weights."""
        n = len(sort_idx)
        weights = np.ones(n)
        
        # Recursive bisection
        cluster_items = [sort_idx]
        
        while cluster_items:
            new_clusters = []
            
            for cluster in cluster_items:
                if len(cluster) <= 1:
                    continue
                
                # Split cluster in half
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]
                
                # Compute cluster variances
                left_var = self._cluster_variance(covariance, left)
                right_var = self._cluster_variance(covariance, right)
                
                # Allocate inversely proportional to variance
                total = left_var + right_var
                if total > 0:
                    alpha = 1 - left_var / total
                else:
                    alpha = 0.5
                
                # Apply allocation
                for i in left:
                    idx = sort_idx.index(i)
                    weights[idx] *= alpha
                for i in right:
                    idx = sort_idx.index(i)
                    weights[idx] *= (1 - alpha)
                
                new_clusters.extend([left, right])
            
            cluster_items = [c for c in new_clusters if len(c) > 1]
        
        return weights
    
    def _cluster_variance(
        self,
        covariance: NDArray,
        cluster: list[int],
    ) -> float:
        """Compute cluster variance (inverse of cluster IVP)."""
        sub_cov = covariance[np.ix_(cluster, cluster)]
        ivp = 1.0 / np.diag(sub_cov)
        ivp /= np.sum(ivp)
        
        return float(ivp @ sub_cov @ ivp)
    
    def optimize(
        self,
        returns: NDArray,
        asset_names: Optional[list[str]] = None,
    ) -> PortfolioResult:
        """Run HRP optimization.
        
        Parameters
        ----------
        returns : NDArray
            Return matrix (n_periods, n_assets).
        asset_names : list[str] | None
            Asset names.
            
        Returns
        -------
        PortfolioResult
            HRP portfolio.
        """
        n_assets = returns.shape[1]
        
        # Compute correlation and covariance
        correlation = np.corrcoef(returns.T)
        covariance = np.cov(returns.T)
        
        # Step 1: Tree clustering
        link, sort_idx = self._tree_clustering(correlation)
        
        # Step 2: Quasi-diagonalization
        sort_idx = self._get_quasi_diag(link)
        
        # Step 3: Recursive bisection
        weights = self._recursive_bisection(covariance, sort_idx)
        
        # Normalize
        weights = np.maximum(weights, 0)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        
        # Compute portfolio metrics
        port_risk = float(np.sqrt(weights @ covariance @ weights))
        
        return PortfolioResult(
            weights=weights,
            expected_return=0.0,
            expected_risk=port_risk,
            sharpe_ratio=0.0,
            turnover=0.0,
            n_assets=n_assets,
        )


# ---------------------------------------------------------------------------
# Mean-CVaR Optimization
# ---------------------------------------------------------------------------


class MeanCVaROptimizer:
    """Mean-CVaR optimization using Conditional Value-at-Risk.
    
    CVaR (Expected Shortfall) is a coherent risk measure that
    better captures tail risk than variance.
    
    Parameters
    ----------
    confidence : float
        CVaR confidence level (default 0.95).
    risk_aversion : float
        Risk aversion parameter (default 1.0).
    """
    
    def __init__(
        self,
        confidence: float = 0.95,
        risk_aversion: float = 1.0,
    ) -> None:
        self.confidence = confidence
        self.risk_aversion = risk_aversion
        
    def compute_cvar(
        self,
        returns: NDArray,
        weights: NDArray,
    ) -> float:
        """Compute portfolio CVaR.
        
        Parameters
        ----------
        returns : NDArray
            Return matrix (n_periods, n_assets).
        weights : NDArray
            Portfolio weights.
            
        Returns
        -------
        float
            CVaR value.
        """
        port_returns = returns @ weights
        var_threshold = np.percentile(port_returns, (1 - self.confidence) * 100)
        
        # CVaR = average of returns below VaR
        tail_returns = port_returns[port_returns <= var_threshold]
        
        if len(tail_returns) > 0:
            return float(-np.mean(tail_returns))
        return 0.0
    
    def optimize(
        self,
        returns: NDArray,
        expected_returns: Optional[NDArray] = None,
        asset_names: Optional[list[str]] = None,
    ) -> PortfolioResult:
        """Optimize portfolio for mean-CVaR.
        
        Parameters
        ----------
        returns : NDArray
            Return matrix (n_periods, n_assets).
        expected_returns : NDArray | None
            Expected returns. If None, uses historical mean.
        asset_names : list[str] | None
            Asset names.
            
        Returns
        -------
        PortfolioResult
            Optimized portfolio.
        """
        n_assets = returns.shape[1]
        
        if expected_returns is None:
            expected_returns = np.mean(returns, axis=0)
        
        # Objective: maximize return - λ * CVaR
        def objective(w):
            port_return = w @ expected_returns
            cvar = self.compute_cvar(returns, w)
            return -(port_return - self.risk_aversion * cvar)
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Constraints
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        
        # Bounds
        bounds = [(0, 1)] * n_assets
        
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = np.maximum(result.x, 0)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        
        # Compute metrics
        port_return = float(weights @ expected_returns)
        port_risk = float(np.std(returns @ weights))
        cvar = self.compute_cvar(returns, weights)
        sharpe = port_return / port_risk if port_risk > 0 else 0.0
        
        return PortfolioResult(
            weights=weights,
            expected_return=port_return,
            expected_risk=port_risk,
            sharpe_ratio=sharpe,
            turnover=0.0,
            n_assets=n_assets,
        )


# ---------------------------------------------------------------------------
# Transaction Cost-Aware Rebalancing
# ---------------------------------------------------------------------------


class TransactionCostOptimizer:
    """Transaction cost-aware portfolio rebalancing.
    
    Optimizes the trade-off between tracking error and transaction costs.
    
    Parameters
    ----------
    cost_per_share : float
        Cost per share traded (default 0.005).
    spread_bps : float
        Bid-ask spread in bps (default 5.0).
    market_impact_coefficient : float
        Market impact coefficient (default 0.1).
    """
    
    def __init__(
        self,
        cost_per_share: float = 0.005,
        spread_bps: float = 5.0,
        market_impact_coefficient: float = 0.1,
    ) -> None:
        self.cost_per_share = cost_per_share
        self.spread_bps = spread_bps
        self.market_impact_coefficient = market_impact_coefficient
        
    def compute_transaction_costs(
        self,
        current_weights: NDArray,
        target_weights: NDArray,
        portfolio_value: float,
        prices: NDArray,
        adv: NDArray,
    ) -> float:
        """Compute transaction costs for rebalancing.
        
        Parameters
        ----------
        current_weights : NDArray
            Current portfolio weights.
        target_weights : NDArray
            Target portfolio weights.
        portfolio_value : float
            Portfolio value.
        prices : NDArray
            Asset prices.
        adv : NDArray
            Average daily volume.
            
        Returns
        -------
        float
            Total transaction costs.
        """
        # Trade quantities
        weight_changes = target_weights - current_weights
        trade_values = weight_changes * portfolio_value
        trade_shares = trade_values / prices
        
        # Spread cost
        spread_cost = np.sum(np.abs(trade_values) * self.spread_bps / 10000)
        
        # Market impact (square-root model)
        participation = np.abs(trade_shares) / np.maximum(adv, 1)
        impact_cost = np.sum(
            np.abs(trade_values) * self.market_impact_coefficient * np.sqrt(participation)
        )
        
        # Commission
        commission = np.sum(np.abs(trade_shares) * self.cost_per_share)
        
        return float(spread_cost + impact_cost + commission)
    
    def optimize_rebalance(
        self,
        current_weights: NDArray,
        target_weights: NDArray,
        covariance: NDArray,
        portfolio_value: float,
        prices: NDArray,
        adv: NDArray,
        max_tracking_error: float = 0.01,
    ) -> NDArray:
        """Optimize rebalancing considering transaction costs.
        
        Parameters
        ----------
        current_weights : NDArray
            Current weights.
        target_weights : NDArray
            Target weights.
        covariance : NDArray
            Covariance matrix.
        portfolio_value : float
            Portfolio value.
        prices : NDArray
            Asset prices.
        adv : NDArray
            Average daily volume.
        max_tracking_error : float
            Maximum allowed tracking error.
            
        Returns
        -------
        NDArray
            Optimized weights (may differ from target to reduce costs).
        """
        n_assets = len(current_weights)
        
        # Objective: minimize tracking error + transaction costs
        def objective(w):
            # Tracking error
            te = np.sqrt((w - target_weights) @ covariance @ (w - target_weights))
            
            # Transaction costs
            tc = self.compute_transaction_costs(
                current_weights, w, portfolio_value, prices, adv,
            )
            
            # Normalize
            tc_normalized = tc / portfolio_value
            
            return te + tc_normalized
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: max_tracking_error - np.sqrt((w - target_weights) @ covariance @ (w - target_weights))},
        ]
        
        # Bounds
        bounds = [(0, 1)] * n_assets
        
        result = minimize(
            objective,
            target_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        weights = np.maximum(result.x, 0)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        
        return weights
    
    def compute_optimal_rebalance_frequency(
        self,
        returns: NDArray,
        current_weights: NDArray,
        target_weights: NDArray,
        portfolio_value: float,
        prices: NDArray,
        adv: NDArray,
        frequencies: list[int] = None,
    ) -> tuple[int, float]:
        """Compute optimal rebalancing frequency.
        
        Parameters
        ----------
        returns : NDArray
            Return matrix.
        current_weights : NDArray
            Current weights.
        target_weights : NDArray
            Target weights.
        portfolio_value : float
            Portfolio value.
        prices : NDArray
            Asset prices.
        adv : NDArray
            Average daily volume.
        frequencies : list[int] | None
            Rebalance frequencies to test (in days).
            
        Returns
        -------
        tuple[int, float]
            (optimal_frequency, cost_savings)
        """
        if frequencies is None:
            frequencies = [1, 5, 10, 20, 60]
        
        # Simulate for each frequency
        costs = {}
        
        for freq in frequencies:
            total_cost = 0.0
            n_rebalances = len(returns) // freq
            
            for i in range(n_rebalances):
                idx = i * freq
                if idx < len(returns):
                    # Cost of rebalancing
                    cost = self.compute_transaction_costs(
                        current_weights, target_weights, portfolio_value, prices, adv,
                    )
                    total_cost += cost
            
            costs[freq] = total_cost
        
        # Find optimal frequency (minimize cost)
        optimal_freq = min(costs, key=costs.get)
        
        # Cost savings vs daily rebalancing
        savings = costs[1] - costs[optimal_freq] if 1 in costs else 0.0
        
        return optimal_freq, savings


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_portfolio_optimizer(
    method: str = "risk_parity",
    **kwargs,
) -> any:
    """Factory function to create portfolio optimizer.
    
    Parameters
    ----------
    method : str
        'black_litterman', 'risk_parity', 'hrp', 'mean_cvar', 'transaction_cost'.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    Portfolio optimizer instance.
    """
    optimizers = {
        "black_litterman": BlackLittermanOptimizer,
        "risk_parity": RiskParityOptimizer,
        "hrp": HierarchicalRiskParity,
        "mean_cvar": MeanCVaROptimizer,
        "transaction_cost": TransactionCostOptimizer,
    }
    
    if method not in optimizers:
        raise ValueError(f"Unknown method: {method}. Choose from {list(optimizers.keys())}")
    
    return optimizers[method](**kwargs)


def compute_portfolio_metrics(
    returns: NDArray,
    weights: NDArray,
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    """Compute comprehensive portfolio metrics.
    
    Parameters
    ----------
    returns : NDArray
        Return matrix (n_periods, n_assets).
    weights : NDArray
        Portfolio weights.
    risk_free_rate : float
        Annual risk-free rate.
        
    Returns
    -------
    dict[str, float]
        Portfolio metrics.
    """
    port_returns = returns @ weights
    
    # Annualization factor
    ann_factor = 252
    
    # Returns
    total_return = float(np.prod(1 + port_returns) - 1)
    ann_return = float(np.mean(port_returns) * ann_factor)
    
    # Risk
    ann_vol = float(np.std(port_returns) * np.sqrt(ann_factor))
    
    # Sharpe ratio
    excess_return = ann_return - risk_free_rate
    sharpe = excess_return / ann_vol if ann_vol > 0 else 0.0
    
    # Sortino ratio
    downside_returns = port_returns[port_returns < 0]
    downside_vol = float(np.std(downside_returns) * np.sqrt(ann_factor)) if len(downside_returns) > 0 else ann_vol
    sortino = excess_return / downside_vol if downside_vol > 0 else 0.0
    
    # Maximum drawdown
    cum_returns = np.cumprod(1 + port_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))
    
    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    # Win rate
    win_rate = float(np.mean(port_returns > 0))
    
    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
    }
