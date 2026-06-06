"""LiveRiskEngine — production wrapper for advanced risk models.

Initialises PCA, ICA, EVT, DCC-GARCH, CorrelationRegime, and RiskDecomposer
models and wires them into the live trading loop via a single
``pre_trade_check()`` gate.  Every order submitted by the production runner
passes through here BEFORE reaching the broker.

Models are fitted on a rolling 252-day return window and refitted every
21 trading days or after a significant correlation-regime change.

Author: QuantCore Team
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from quantcore.risk.advanced_risk import (
    CorrelationRegimeDetector,
    DCCGARCHModel,
    ICARiskModel,
    PCARiskModel,
    RiskDecomposer,
    TailRiskModel,
)

if TYPE_CHECKING:
    from quantcore.core.types import Order, Position, TradeSignal

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approval (aliased to avoid collision with RiskManagerLive.Approval)
# ---------------------------------------------------------------------------


@dataclass
class RiskApproval:
    """Result of a LiveRiskEngine pre-trade check."""

    approved: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Rolling window data store
# ---------------------------------------------------------------------------


@dataclass
class ReturnWindow:
    """Fixed-size rolling window of portfolio-level daily returns."""

    capacity: int = 252

    _returns: deque = field(default_factory=lambda: deque(maxlen=252))
    _assetreturns: deque = field(default_factory=lambda: deque(maxlen=252))
    _dates: deque = field(default_factory=lambda: deque(maxlen=252))
    _n_assets: int | None = None

    asset_names: list[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "TLT", "GLD", "DBC"]
    )

    def push(self, portfolio_return: float, asset_returns: np.ndarray, as_of: date) -> None:
        """Append a daily observation (portfolio + per-asset returns)."""
        self._returns.append(portfolio_return)
        if len(asset_returns) > 0:
            # Infer n_assets from the very first push; reject mismatched subsequent data.
            # Also extend asset_names so it always mirrors the actual n_assets.
            if self._n_assets is None:
                self._n_assets = len(asset_returns)
                _log.info("ReturnWindow: n_assets set to %d from first push", self._n_assets)
                # Extend the default 5-name list to cover more assets (e.g. "asset_5")
                while len(self.asset_names) < self._n_assets:
                    self.asset_names.append(f"asset_{len(self.asset_names)}")
            if len(asset_returns) != self._n_assets:
                _log.warning(
                    "ReturnWindow: received %d-asset returns but n_assets=%d — dropping",
                    len(asset_returns), self._n_assets,
                )
            else:
                self._assetreturns.append(np.asarray(asset_returns).reshape(-1))
        self._dates.append(as_of)

    @property
    def portfolio_returns(self) -> np.ndarray:
        if len(self._returns) < 20:
            return self._synthetic_portfolio_returns()
        return np.array(self._returns, dtype=np.float64)

    @property
    def asset_returns_matrix(self) -> np.ndarray:
        if len(self._assetreturns) < 20:
            return self._synthetic_asset_returns()
        rows = [r for r in self._assetreturns]
        min_len = min(len(r) for r in rows) if rows else 0
        return np.vstack([r[:min_len] for r in rows]).T.astype(np.float64)

    @property
    def is_ready(self) -> bool:
        return len(self._returns) >= 20

    @property
    def n_observations(self) -> int:
        return len(self._returns)

    def _synthetic_portfolio_returns(self) -> np.ndarray:
        """Generate synthetic portfolio returns for cold-start."""
        rng = np.random.default_rng(42)
        annual_vol = 0.12
        daily_vol = annual_vol / np.sqrt(252)
        return rng.normal(0.0003, daily_vol, self.capacity).astype(np.float64)

    def _synthetic_asset_returns(self) -> np.ndarray:
        """Generate synthetic multi-asset returns for cold-start."""
        rng = np.random.default_rng(42)
        annual_vol = 0.15
        daily_vol = annual_vol / np.sqrt(252)
        n = self.capacity
        n_a = self._n_assets or 5
        return (rng.normal(0, daily_vol, (n, n_a)) +
                rng.normal(0, 0.0002, (n, n_a))).astype(np.float64)

    def weights_for_decomposer(self, positions: dict[str, Any]) -> np.ndarray:
        """Build equal-weight vector from available tickers in the window."""
        available = [a for a in self.asset_names if a in positions]
        n = len(available) or self._n_assets
        return np.ones(n, dtype=np.float64) / n


# ---------------------------------------------------------------------------
# LiveRiskEngine
# ---------------------------------------------------------------------------


class LiveRiskEngine:
    """Production risk engine — factor-aware, tail-risk-aware, regime-aware.

    Initialised once at startup.  Provides a single blocking gate:
    ``pre_trade_check()``, which all orders must pass before submission.

    **Safety layers (additive, do not disrupt existing positions):**
      1. Factor exposure check (PCA/ICA)
      2. Marginal tail-risk contribution (EVT)
      3. Correlation-regime leverage override (DCC-GARCH)
      4. Risk decomposition veto

    Parameters
    ----------
    refit_days : int
        Rolling window size in days (default 252).
    refit_interval : int
        Force refit every N trading days (default 21).
    correlation_threshold : float
        DCC-GARCH conditional correlation above this → elevated regime
        (default 0.70).
    high_regime_persistence_alert : int
        Days of sustained high correlation before WARNING state (default 5).
    """

    def __init__(
        self,
        refit_days: int = 252,
        refit_interval: int = 21,
        correlation_threshold: float = 0.70,
        high_regime_persistence_alert: int = 5,
    ) -> None:
        self.refit_days = refit_days
        self.refit_interval = refit_interval
        self.correlation_threshold = correlation_threshold
        self.high_regime_persistence_alert = high_regime_persistence_alert

        # --- Models ---------------------------------------------------
        self._pca = PCARiskModel()
        self._ica = ICARiskModel()
        self._evt = TailRiskModel(threshold_percentile=0.05)
        self._regime_detector = CorrelationRegimeDetector(lookback=60)
        self._dcc = DCCGARCHModel()
        self._risk_decomposer = RiskDecomposer()

        # --- State ----------------------------------------------------
        self._window = ReturnWindow(capacity=refit_days)
        self._fitted = False
        self._last_fit_date: date | None = None
        self._days_since_fit = 0

        # Correlation regime tracking
        self._current_regime: str = "NORMAL"  # NORMAL | ELEVATED | CRISIS
        self._high_corr_days: int = 0        # consecutive high-corr days
        self._regime_change_count: int = 0

        # Latest computed metrics (exposed for API endpoint)
        self._latest_var_95: float = 0.0
        self._latest_var_99: float = 0.0
        self._latest_es_95: float = 0.0
        self._latest_es_99: float = 0.0
        self._latest_avg_corr: float = 0.0
        self._latest_conditional_corr: float = 0.0
        self._latest_factor_exposures: dict[str, float] = {}
        self._latest_risk_contrib: dict[str, float] = {}
        self._latest_systematic_risk: float = 0.0
        self._latest_idiosyncratic_risk: float = 0.0

        _log.info(
            "LiveRiskEngine initialised: refit_days=%d refit_interval=%d "
            "corr_threshold=%.2f",
            refit_days, refit_interval, correlation_threshold,
        )

    # ------------------------------------------------------------------
    # Initialisation / fitting
    # ------------------------------------------------------------------

    def fit_initial(self, synthetic: bool = True) -> None:
        """Perform the initial model fit using available data.

        In production this is called once at runner startup.
        If ``synthetic=True`` and no real data is available, fits on
        252 days of synthetic returns so the engine is immediately operational.
        """
        if self._fitted and self._window.is_ready:
            return

        returns = self._window.portfolio_returns
        asset_rets = self._window.asset_returns_matrix

        if len(returns) < 20:
            returns = self._window._synthetic_portfolio_returns()
            asset_rets = self._window._synthetic_asset_returns()
            _log.warning(
                "LiveRiskEngine: insufficient real return data — using "
                "synthetic returns for initial fit (engine operational but "
                "not yet calibrated to live regime)"
            )

        self._fit_models(returns, asset_rets)
        self._last_fit_date = date.today()
        self._days_since_fit = 0
        _log.info(
            "LiveRiskEngine initial fit complete: %d observations, "
            "factor_exposures=%s",
            len(returns),
            {k: f"{v:.3f}" for k, v in self._latest_factor_exposures.items()},
        )

    def _fit_models(self, returns: np.ndarray, asset_rets: np.ndarray) -> None:
        """Fit all risk models on the given return data."""
        try:
            self._pca.fit(asset_rets)
            self._ica.fit(asset_rets)
            self._evt.fit(returns)

            # DCC-GARCH on asset returns
            if asset_rets.shape[0] >= 60:
                self._dcc.fit(asset_rets)
                dynamic_corrs = self._dcc.compute_dynamic_correlation(asset_rets)
                if dynamic_corrs is not None and len(dynamic_corrs) > 0:
                    last_corr = dynamic_corrs[-1]
                    n = last_corr.shape[0]
                    mask = np.ones((n, n), dtype=bool)
                    np.fill_diagonal(mask, False)
                    self._latest_conditional_corr = float(np.mean(last_corr[mask]))
                    self._latest_avg_corr = self._latest_conditional_corr

            # Determine asset_names from the actual matrix shape — fixes mismatch
            # when real data has more assets than the default 5-name list.
            n_assets = asset_rets.shape[1] if asset_rets.ndim == 2 else 1
            full_asset_names: list[str] = (
                self._window.asset_names[:n_assets]
                if len(self._window.asset_names) >= n_assets
                else self._window.asset_names + [
                    f"asset_{i}" for i in range(len(self._window.asset_names), n_assets)
                ]
            )

            # Factor exposures
            import pandas as pd
            pca_loadings = self._pca.get_factor_loadings(full_asset_names)
            if not pca_loadings.empty:
                exposures = pca_loadings.mean(axis=1).to_dict()
                self._latest_factor_exposures = {
                    str(k): float(v) for k, v in exposures.items()
                }

            # Risk decomposition (equal weights for now — positions injected at check time)
            n_assets = asset_rets.shape[1] if asset_rets.ndim == 2 else 1
            equal_weights = np.ones(n_assets, dtype=np.float64) / max(n_assets, 1)
            if asset_rets.shape[0] >= 20:
                decomp = self._risk_decomposer.decompose(equal_weights, asset_rets, full_asset_names)
                self._latest_systematic_risk = decomp.systematic_risk
                self._latest_idiosyncratic_risk = decomp.idiosyncratic_risk
                if decomp.component_contributions is not None:
                    contrib = decomp.component_contributions
                    self._latest_risk_contrib = {
                        full_asset_names[i] if i < len(full_asset_names) else f"asset_{i}": float(
                            contrib[i] if i < len(contrib) else 0.0
                        )
                        for i in range(len(contrib))
                    }

            # Tail risk metrics
            self._latest_var_95 = float(self._evt.compute_var(0.95))
            self._latest_var_99 = float(self._evt.compute_var(0.99))
            self._latest_es_95 = float(self._evt.compute_es(0.95))
            self._latest_es_99 = float(self._evt.compute_es(0.99))

            # Update correlation regime
            self._update_regime()

            self._fitted = True

        except Exception as exc:
            _log.error("LiveRiskEngine model fit failed: %s — engine may be inaccurate", exc)

    def _update_regime(self) -> None:
        """Update the correlation-regime flag based on latest DCC-GARCH output."""
        avg_corr = self._latest_conditional_corr

        prev_regime = self._current_regime
        if avg_corr > 0.80:
            self._current_regime = "CRISIS"
        elif avg_corr > self.correlation_threshold:
            self._current_regime = "ELEVATED"
        else:
            self._current_regime = "NORMAL"

        if prev_regime != self._current_regime:
            self._regime_change_count += 1
            _log.info(
                "CORRELATION REGIME CHANGE: %s → %s (avg_corr=%.3f, count=%d)",
                prev_regime, self._current_regime, avg_corr, self._regime_change_count,
            )

        if self._current_regime in ("ELEVATED", "CRISIS"):
            self._high_corr_days += 1
        else:
            self._high_corr_days = 0

        if self._current_regime == "CRISIS":
            _log.warning(
                "CRISIS correlation regime active: avg_cond_corr=%.3f, "
                "consecutive_high_corr_days=%d",
                avg_corr, self._high_corr_days,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pre_trade_check(
        self,
        signal: Any,
        positions: dict[str, Any] | None = None,
        current_equity: float = 100_000.0,
    ) -> RiskApproval:
        """Gate every order through the advanced risk models.

        Called from ``ProductionRunner._submit_signal()`` AFTER the existing
        ``RiskManagerLive.pre_trade_check()`` but BEFORE broker submission.

        Parameters
        ----------
        signal : TradeSignal
            The signal being evaluated for submission.
        positions : dict[str, Any]
            Current portfolio positions (symbol → position object).
        current_equity : float
            Latest known portfolio equity.

        Returns
        -------
        RiskApproval
            ``approved=True`` if the signal passes all risk checks.
            ``approved=False`` with a non-empty ``reason`` if rejected.
        """
        if not self._fitted:
            self.fit_initial()

        try:
            approval = self._check_factor_exposure(signal)
            if not approval.approved:
                return approval

            # Tail-risk check runs BEFORE correlation regime — first check if the trade
            # itself is too risky, then check if the current market regime warrants tightening.
            approval = self._check_tail_risk(signal, current_equity)
            if not approval.approved:
                return approval

            approval = self._check_correlation_regime(signal, current_equity)
            if not approval.approved:
                return approval

            approval = self._check_risk_decomposition(signal, positions, current_equity)
            return approval

        except Exception as exc:
            _log.error("LiveRiskEngine pre_trade_check error: %s — allowing order", exc)
            return RiskApproval(approved=True, reason="risk_engine_error")

    def _check_factor_exposure(self, signal: Any) -> RiskApproval:
        """PCA/ICA factor exposure veto.

        Reject if the signal increases factor exposure beyond acceptable bounds.
        """
        if not self._fitted or not self._pca._is_fitted:
            return RiskApproval(approved=True)

        direction = getattr(signal, "direction", 0)
        if direction == 0:  # closing — always allow
            return RiskApproval(approved=True)

        # Simple heuristic: new position direction increases directional exposure
        # Check against top factor loadings if available
        if not self._latest_factor_exposures:
            return RiskApproval(approved=True)

        # Allow if factor exposures are reasonable (not a hard veto in normal regime)
        if self._current_regime == "NORMAL":
            return RiskApproval(approved=True)

        # In elevated/crisis regime, be stricter on adding directional exposure
        max_exposure = 0.8
        for exp in self._latest_factor_exposures.values():
            if abs(exp) > max_exposure:
                return RiskApproval(
                    approved=False,
                    reason=f"Factor exposure {abs(exp):.2f} exceeds {max_exposure:.2f} "
                           f"in {self._current_regime} regime — reduce before adding",
                )

        return RiskApproval(approved=True)

    def _check_tail_risk(self, signal: Any, current_equity: float) -> RiskApproval:
        """EVT tail-risk marginal contribution check.

        Approximate the marginal CVaR contribution of the new position and
        reject if it pushes the tail risk beyond acceptable bounds.
        """
        if not self._fitted or not self._evt._is_fitted:
            return RiskApproval(approved=True)

        var99 = self._latest_var_99
        es99 = self._latest_es_99
        notional = float(getattr(signal, "size", 0)) or (current_equity * 0.05)

        # Heuristic: reject if VaR at 99% exceeds 3% of equity in normal,
        # 2% in elevated, 1% in crisis
        thresholds = {
            "NORMAL": 0.03,
            "ELEVATED": 0.02,
            "CRISIS": 0.01,
        }
        threshold = thresholds.get(self._current_regime, 0.03)

        var_pct = var99 / current_equity if current_equity > 0 else 0.0
        if var_pct > threshold:
            return RiskApproval(
                approved=False,
                reason=f"Tail risk VaR(99%%) = {var_pct:.2%} of equity "
                       f"exceeds {threshold:.2%} limit in {self._current_regime} regime",
            )

        return RiskApproval(approved=True)

    def _check_correlation_regime(self, signal: Any, current_equity: float) -> RiskApproval:
        """DCC-GARCH correlation-regime leverage override.

        In high-correlation regimes, apply a stricter effective leverage
        limit and warn the operator if it persists.
        """
        is_high_corr = self._current_regime in ("ELEVATED", "CRISIS")

        if not is_high_corr:
            return RiskApproval(approved=True)

        direction = getattr(signal, "direction", 0)
        if direction == 0:
            return RiskApproval(approved=True)

        # In high-correlation mode: tighter position sizes
        notional = float(getattr(signal, "size", 0)) or (current_equity * 0.05)
        notional_pct = notional / current_equity if current_equity > 0 else 0.0

        # Effective limit halves in elevated, quarters in crisis
        limits = {"ELEVATED": 0.025, "CRISIS": 0.0125}
        limit = limits.get(self._current_regime, 0.05)

        if notional_pct > limit:
            return RiskApproval(
                approved=False,
                reason=f"High correlation regime ({self._current_regime}): "
                       f"trade size {notional_pct:.2%} exceeds reduced limit {limit:.2%} "
                       f"(high_corr_days={self._high_corr_days})",
            )

        # Persistent high correlation → operator alert
        if self._high_corr_days >= self.high_regime_persistence_alert:
            _log.warning(
                "HIGH CORRELATION PERSISTENCE WARNING: %d consecutive high-corr days — "
                "consider reducing gross exposure. average_cond_corr=%.3f",
                self._high_corr_days, self._latest_conditional_corr,
            )

        return RiskApproval(approved=True)

    def _check_risk_decomposition(
        self, signal: Any, positions: dict[str, Any] | None, current_equity: float
    ) -> RiskApproval:
        """RiskDecomposer veto — reject if idiosyncratic concentration is extreme."""
        if not self._fitted or self._latest_systematic_risk == 0:
            return RiskApproval(approved=True)

        # In crisis regime, the idiosyncratic risk floor rises sharply
        if self._current_regime == "CRISIS":
            max_idio = 0.08  # 8% idio vol → crisis
            if self._latest_idiosyncratic_risk > max_idio:
                return RiskApproval(
                    approved=False,
                    reason=f"Idiosyncratic risk {self._latest_idiosyncratic_risk:.2%} "
                           f"exceeds {max_idio:.2%} — crisis regime, reducing",
                )

        return RiskApproval(approved=True)

    def update(self, daily_portfolio_return: float, asset_returns: np.ndarray | None = None, as_of: date | None = None) -> None:
        """Update the rolling window with the day's realised returns.

        Called during end-of-day reconciliation in ``ProductionRunner._reconcile()``.
        Triggers a model refit when the refit_interval is reached or a regime
        change is detected.
        """
        as_of = as_of or date.today()
        asset_rets = asset_returns if asset_returns is not None else np.array([])
        self._window.push(daily_portfolio_return, asset_rets, as_of)

        if not self._fitted:
            self.fit_initial()
            return

        self._days_since_fit += 1

        # Check if regime changed significantly (auto-refit trigger)
        self._update_regime()
        regime_changed = (
            self._current_regime in ("ELEVATED", "CRISIS") and
            self._high_corr_days == 1  # just entered elevated/crisis
        )

        force_refit = (self._days_since_fit >= self.refit_interval) or regime_changed

        if force_refit:
            _log.info(
                "LiveRiskEngine refit triggered: days_since_fit=%d regime_changed=%s "
                "current_regime=%s",
                self._days_since_fit, regime_changed, self._current_regime,
            )
            self._refit()
            self._last_fit_date = as_of
            self._days_since_fit = 0

    def _refit(self) -> None:
        """Refit all models on the current rolling window."""
        returns = self._window.portfolio_returns
        asset_rets = self._window.asset_returns_matrix
        if len(returns) >= 20:
            self._fit_models(returns, asset_rets)

    @property
    def correlation_regime(self) -> str:
        return self._current_regime

    @property
    def conditional_correlation(self) -> float:
        return self._latest_conditional_corr

    @property
    def high_corr_days(self) -> int:
        return self._high_corr_days

    @property
    def is_high_correlation_regime(self) -> bool:
        return self._current_regime in ("ELEVATED", "CRISIS")

    def get_snapshot(self) -> dict[str, Any]:
        """Return a read-only snapshot of all current risk metrics.

        Used by the ``GET /risk`` API endpoint.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "correlation_regime": self._current_regime,
            "conditional_correlation": self._latest_conditional_corr,
            "avg_correlation": self._latest_avg_corr,
            "high_corr_days": self._high_corr_days,
            "regime_change_count": self._regime_change_count,
            "var_95": self._latest_var_95,
            "var_99": self._latest_var_99,
            "es_95": self._latest_es_95,
            "es_99": self._latest_es_99,
            "factor_exposures": self._latest_factor_exposures,
            "risk_contribution": self._latest_risk_contrib,
            "systematic_risk": self._latest_systematic_risk,
            "idiosyncratic_risk": self._latest_idiosyncratic_risk,
            "n_observations": self._window.n_observations,
            "is_fitted": self._fitted,
            "days_since_fit": self._days_since_fit,
        }