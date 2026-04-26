"""Risk Engine: unified orchestrator for all risk computations.

Single entry point that chains VaR, stress tests, limits, drawdown, greeks,
metrics, attribution, and reporting into one comprehensive risk assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from quantcore.core.types import PortfolioSnapshot
from quantcore.risk.attribution import AttributionResult, risk_attribution
from quantcore.risk.drawdown import DrawdownConfig, DrawdownState, DrawdownTracker
from quantcore.risk.greeks import GreeksSnapshot, aggregate_greeks
from quantcore.risk.limits import (
    CircuitBreaker,
    LimitCheckResult,
    RiskLimits,
    check_post_trade,
)
from quantcore.risk.metrics import compute_all_metrics
from quantcore.risk.stress import (
    StressResult,
    apply_scenario,
    build_scenario_library,
    concentration_report,
    reverse_stress,
)
from quantcore.risk.var import (
    VaRResult,
    cornish_fisher_var,
    cvar,
    historical_var,
    monte_carlo_var,
    parametric_var,
)


@dataclass
class RiskAssessment:
    """Complete risk assessment for a portfolio at a point in time."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    portfolio_value: float = 0.0

    # VaR (multiple methods)
    var_historical: VaRResult | None = None
    var_parametric: VaRResult | None = None
    var_cornish_fisher: VaRResult | None = None
    var_monte_carlo: VaRResult | None = None
    cvar_value: float = 0.0

    # Stress
    stress_results: dict[str, StressResult] = field(default_factory=dict)
    reverse_stress_losses: NDArray[np.floating] | None = None

    # Greeks
    greeks: GreeksSnapshot | None = None

    # Limits
    limit_breaches: list[LimitCheckResult] = field(default_factory=list)
    circuit_breaker_active: bool = False

    # Drawdown
    drawdown: DrawdownState | None = None
    max_drawdown: float = 0.0

    # Metrics
    metrics: dict[str, float] = field(default_factory=dict)

    # Attribution
    attribution: AttributionResult | None = None
    risk_contributions: dict[str, float] = field(default_factory=dict)

    # Flags
    risk_flags: list[str] = field(default_factory=list)
    risk_level: str = "normal"  # low, normal, elevated, high, critical


class RiskEngine:
    """Central risk engine that orchestrates all risk computations.

    Feed it returns history + portfolio snapshot, get back a complete
    RiskAssessment with VaR, stress tests, limits, greeks, metrics, and
    attribution.

    Usage:
        engine = RiskEngine(RiskLimits(max_concentration_pct=0.25))
        assessment = engine.assess(returns_history, portfolio_snapshot)
        print(assessment.risk_level, assessment.risk_flags)
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        drawdown_config: DrawdownConfig | None = None,
        var_confidence: float = 0.95,
        stress_scenarios: list[str] | None = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.drawdown_tracker = DrawdownTracker(drawdown_config)
        self.var_confidence = var_confidence
        self.stress_scenarios = stress_scenarios or [
            "tech_crash", "rate_hike", "credit_crunch", "vol_spike",
        ]
        self._returns_history: list[float] = []
        self._circuit_breaker = CircuitBreaker(limits=self.limits)

    def update_returns(self, returns: NDArray[np.floating]) -> None:
        """Accumulate returns history for rolling computations."""
        r = np.asarray(returns, dtype=float).flatten()
        self._returns_history.extend(r.tolist())
        # Keep last 5 years
        self._returns_history = self._returns_history[-1260:]

    def assess(
        self,
        portfolio: PortfolioSnapshot,
        returns: NDArray[np.floating] | None = None,
        market_data: dict[str, float] | None = None,
        option_greeks: dict[str, dict[str, float]] | None = None,
    ) -> RiskAssessment:
        """Run complete risk assessment on a portfolio.

        Args:
            portfolio: Current portfolio snapshot.
            returns: Recent portfolio returns (daily). If None, uses history.
            market_data: {symbol: price} for current prices.
            option_greeks: {symbol: {delta, gamma, vega, theta}} for options.

        Returns:
            RiskAssessment with all risk metrics.
        """
        if returns is not None:
            self.update_returns(returns)

        r = np.array(self._returns_history, dtype=float)
        if len(r) < 2:
            return RiskAssessment(portfolio_value=float(portfolio.total_value))

        assessment = RiskAssessment(
            timestamp=datetime.utcnow(),
            portfolio_value=float(portfolio.total_value),
        )

        # 1. VaR
        assessment.var_historical = historical_var(r, confidence=self.var_confidence)
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        assessment.var_parametric = parametric_var(mu, sigma, confidence=self.var_confidence)
        assessment.cvar_value = cvar(r, confidence=self.var_confidence)
        if len(r) >= 100:
            from scipy import stats as sp_stats
            skew = float(sp_stats.skew(r))
            kurt = float(sp_stats.kurtosis(r))
            assessment.var_cornish_fisher = cornish_fisher_var(
                mu, sigma, skew, kurt, confidence=self.var_confidence,
            )
        if len(r) >= 252:
            assessment.var_monte_carlo = monte_carlo_var(r, confidence=self.var_confidence, n_scenarios=10_000)

        # 2. Stress tests
        scenarios = build_scenario_library()
        positions_dict = {
            sym: pos for sym, pos in portfolio.positions.items()
        }
        for scenario_name in self.stress_scenarios:
            if scenario_name in scenarios:
                try:
                    assessment.stress_results[scenario_name] = apply_scenario(
                        scenario_name, positions_dict, scenarios,
                    )
                except Exception:
                    pass

        # Concentration report
        try:
            assessment.stress_results["concentration"] = concentration_report(portfolio)
        except Exception:
            pass

        # Reverse stress
        if len(r) >= 100:
            values = np.array([
                float(pos.market_value) for pos in portfolio.positions.values()
            ])
            if len(values) > 0:
                try:
                    loss_threshold = -0.20 * float(portfolio.total_value)
                    rev_result = reverse_stress(values, float(loss_threshold))
                    assessment.reverse_stress_losses = rev_result.shocks if rev_result.converged else None
                except Exception:
                    pass

        # 3. Greeks
        try:
            assessment.greeks = aggregate_greeks(
                portfolio,
                option_greeks=option_greeks,
            )
        except Exception:
            pass

        # 4. Limits
        assessment.limit_breaches = check_post_trade(portfolio, self.limits)
        assessment.circuit_breaker_active = self._circuit_breaker.check(
            assessment.limit_breaches
        )

        # 5. Drawdown
        total_val = float(portfolio.total_value)
        assessment.drawdown = self.drawdown_tracker.update(total_val)
        assessment.max_drawdown = self.drawdown_tracker.max_drawdown

        # 6. Performance metrics
        assessment.metrics = compute_all_metrics(r)

        # 7. Risk attribution (if we have weights and cov)
        if len(portfolio.positions) >= 2 and len(r) >= 50:
            try:
                weights = np.array([
                    float(abs(p.market_value)) / total_val
                    for p in portfolio.positions.values()
                    if float(abs(p.market_value)) > 0
                ])
                if len(weights) >= 2:
                    # Approximate covariance from returns
                    cov = np.cov(r.reshape(-1, 1).T) if r.ndim == 1 else np.eye(len(weights)) * np.var(r)
                    if len(weights) == cov.shape[0]:
                        attr = risk_attribution(weights, cov, confidence=self.var_confidence)
                        assessment.risk_contributions = attr.component_var_pct
            except Exception:
                pass

        # 8. Risk level and flags
        assessment.risk_level, assessment.risk_flags = self._classify_risk(assessment)

        return assessment

    def _classify_risk(self, a: RiskAssessment) -> tuple[str, list[str]]:
        """Classify overall risk level and generate flags."""
        flags: list[str] = []
        score = 0

        # VaR-based
        if a.var_historical and a.portfolio_value > 0:
            var_pct = abs(a.var_historical.var) / a.portfolio_value
            if var_pct > 0.05:
                score += 1
                flags.append(f"VaR {var_pct:.1%} exceeds 5%")
            if var_pct > 0.10:
                score += 1
                flags.append(f"VaR {var_pct:.1%} exceeds 10% (critical)")

        # Stress losses
        worst_stress = 0.0
        for name, result in a.stress_results.items():
            if abs(result.total_pnl) > worst_stress:
                worst_stress = abs(result.total_pnl)
        if a.portfolio_value > 0:
            stress_pct = worst_stress / a.portfolio_value
            if stress_pct > 0.15:
                score += 1
                flags.append(f"Worst stress loss {stress_pct:.1%} exceeds 15%")

        # Limits
        if a.limit_breaches:
            score += 1
            flags.append(f"{len(a.limit_breaches)} limit breaches")

        if a.circuit_breaker_active:
            score += 2
            flags.append("Circuit breaker activated")

        # Drawdown
        if a.drawdown and a.drawdown.drawdown_pct < -10:
            score += 1
            flags.append(f"Drawdown {a.drawdown.drawdown_pct:.0f}%")
        if a.drawdown and a.drawdown.drawdown_pct < -20:
            score += 1
            flags.append(f"Drawdown {a.drawdown.drawdown_pct:.0f}% exceeds 20% (critical)")

        # Sharpe
        sharpe = a.metrics.get("sharpe", 0)
        if sharpe < 0:
            score += 1
            flags.append("Negative Sharpe ratio")

        # Risk level
        if score <= 1:
            level = "normal"
        elif score <= 3:
            level = "elevated"
        elif score <= 5:
            level = "high"
        else:
            level = "critical"

        return level, flags

    @property
    def is_circuit_breaker_active(self) -> bool:
        return self._circuit_breaker.activated

    def reset_circuit_breaker(self) -> None:
        self._circuit_breaker.reset()
