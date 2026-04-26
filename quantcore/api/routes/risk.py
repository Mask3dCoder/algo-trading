"""Risk management API routes: VaR, stress tests, limits, metrics, reports."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from quantcore.risk.limits import RiskLimits
from quantcore.risk.metrics import compute_all_metrics, drawdown_stats
from quantcore.risk.reports import generate_daily_risk_report, generate_weekly_summary
from quantcore.risk.stress import (
    apply_scenario,
    build_scenario_library,
    correlation_stress,
    factor_stress,
    reverse_stress,
)
from quantcore.risk.var import (
    cornish_fisher_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
    portfolio_parametric_var,
)
from quantcore.risk.var_backtest import var_backtest_report

router = APIRouter(prefix="/risk", tags=["risk"])


# ---------------------------------------------------------------------------
# VaR endpoints
# ---------------------------------------------------------------------------


@router.post("/var/historical")
def api_historical_var(
    returns: list[float],
    confidence: float = Query(0.95, ge=0.90, le=0.999),
    horizon_days: int = Query(1, ge=1, le=60),
) -> dict[str, Any]:
    """Compute Historical VaR and CVaR."""
    r = np.array(returns)
    result = historical_var(r, confidence=confidence, horizon_days=horizon_days)
    return {
        "method": result.method,
        "confidence": result.confidence,
        "var": result.var,
        "cvar": result.cvar,
        "horizon_days": horizon_days,
    }


@router.post("/var/parametric")
def api_parametric_var(
    returns: list[float],
    confidence: float = Query(0.95, ge=0.90, le=0.999),
    horizon_days: int = Query(1, ge=1, le=60),
) -> dict[str, Any]:
    """Compute Parametric VaR (normal assumption)."""
    r = np.array(returns)
    result = parametric_var(r, confidence=confidence, horizon_days=horizon_days)
    return {
        "method": result.method,
        "confidence": result.confidence,
        "var": result.var,
        "cvar": result.cvar,
    }


@router.post("/var/portfolio")
def api_portfolio_var(
    returns: list[list[float]],
    weights: list[float],
    confidence: float = Query(0.95, ge=0.90, le=0.999),
) -> dict[str, Any]:
    """Compute portfolio parametric VaR."""
    r = np.array(returns)
    w = np.array(weights)
    result = portfolio_parametric_var(r, w, confidence=confidence)
    return {
        "var": result.var,
        "cvar": result.cvar,
        "method": "portfolio_parametric",
    }


@router.post("/var/cornish-fisher")
def api_cornish_fisher_var(
    returns: list[float],
    confidence: float = Query(0.95, ge=0.90, le=0.999),
) -> dict[str, Any]:
    """Compute Cornish-Fisher VaR (accounts for skew/kurtosis)."""
    r = np.array(returns)
    result = cornish_fisher_var(r, confidence=confidence)
    return {"var": result.var, "cvar": result.cvar, "method": "cornish_fisher"}


@router.post("/var/monte-carlo")
def api_monte_carlo_var(
    returns: list[float],
    confidence: float = Query(0.95),
    n_simulations: int = Query(10_000, le=100_000),
) -> dict[str, Any]:
    """Compute Monte Carlo VaR."""
    r = np.array(returns)
    result = monte_carlo_var(r, confidence=confidence, n_simulations=n_simulations)
    return {"var": result.var, "cvar": result.cvar, "method": "monte_carlo"}


# ---------------------------------------------------------------------------
# Stress test endpoints
# ---------------------------------------------------------------------------


@router.get("/stress/scenarios")
def api_list_scenarios() -> dict[str, Any]:
    """List available stress scenarios."""
    scenarios = build_scenario_library()
    return {
        "scenarios": [
            {"name": name, "description": s.description}
            for name, s in scenarios.items()
        ]
    }


@router.post("/stress/apply")
def api_apply_scenario(
    scenario_name: str,
    positions: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Apply a named stress scenario to a portfolio."""
    scenarios = build_scenario_library()
    if scenario_name not in scenarios:
        raise HTTPException(404, f"Scenario '{scenario_name}' not found")

    result = apply_scenario(scenario_name, positions, scenarios)
    return {
        "scenario": scenario_name,
        "total_pnl": result.total_pnl,
        "pnl_by_symbol": result.pnl_by_symbol,
        "worst_symbol": result.worst_symbol,
    }


@router.post("/stress/correlation")
def api_correlation_stress(
    positions: dict[str, dict[str, float]],
    target_corr: float = Query(0.80, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Run a correlation stress test."""
    result = correlation_stress(type("Portfolio", (), {
        "positions": positions,
        "total_value": sum(p.get("market_value", 0) for p in positions.values()),
    })(), target_corr)
    return {"total_pnl": result.total_pnl, "worst_symbol": result.worst_symbol}


@router.post("/stress/reverse")
def api_reverse_stress(
    positions: dict[str, dict[str, float]],
    loss_threshold: float = Query(..., description="Max acceptable loss (negative)"),
) -> dict[str, Any]:
    """Run a reverse stress test — find shocks that breach loss threshold."""
    values = np.array([p.get("market_value", 0) for p in positions.values()])
    result = reverse_stress(values, loss_threshold)
    return {
        "shocks": result.shocks.tolist() if hasattr(result.shocks, "tolist") else list(result.shocks),
        "worst_pnl": result.worst_pnl,
        "converged": result.converged,
    }


@router.post("/stress/factor")
def api_factor_stress(
    positions: dict[str, dict[str, float]],
    factor_shocks: dict[str, float],
) -> dict[str, Any]:
    """Run a macro factor stress test."""
    result = factor_stress(positions, factor_shocks)
    return {"total_pnl": result.total_pnl, "pnl_by_factor": result.pnl_by_factor}


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


@router.get("/limits")
def api_get_limits() -> dict[str, Any]:
    """Get default risk limits."""
    limits = RiskLimits()
    return {
        "max_position_pct_adv": limits.max_position_pct_adv,
        "max_dollar_exposure_per_name": limits.max_dollar_exposure_per_name,
        "max_loss_per_position": limits.max_loss_per_position,
        "max_net_delta": limits.max_net_delta,
        "max_gross_gamma": limits.max_gross_gamma,
        "max_concentration_pct": limits.max_concentration_pct,
        "max_var_utilization_pct": limits.max_var_utilization_pct,
        "circuit_breaker_threshold_pct": limits.circuit_breaker_threshold_pct,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@router.post("/metrics")
def api_compute_metrics(
    returns: list[float],
    market_returns: list[float] | None = None,
    rf: float = 0.0,
) -> dict[str, Any]:
    """Compute all risk-adjusted performance metrics."""
    r = np.array(returns)
    m = np.array(market_returns) if market_returns else None
    metrics = compute_all_metrics(r, m, rf)
    return {k: float(v) if v is not None else None for k, v in metrics.items()}


@router.post("/metrics/drawdown")
def api_drawdown_stats(returns: list[float]) -> dict[str, Any]:
    """Compute drawdown statistics."""
    r = np.array(returns)
    stats = drawdown_stats(r)
    return {
        "max_drawdown": stats.max_drawdown,
        "avg_drawdown": stats.avg_drawdown,
        "max_drawdown_duration": stats.max_drawdown_duration,
        "avg_drawdown_duration": stats.avg_drawdown_duration,
    }


# ---------------------------------------------------------------------------
# VaR backtest
# ---------------------------------------------------------------------------


@router.post("/var/backtest")
def api_var_backtest(
    returns: list[float],
    var_series: list[float],
    confidence: float = Query(0.95),
) -> dict[str, Any]:
    """Backtest a VaR model against realized returns."""
    r = np.array(returns)
    v = np.array(var_series)
    report = var_backtest_report(r, v, confidence=confidence)
    return {
        "n_observations": report.n_observations,
        "n_violations": report.n_violations,
        "violation_rate": report.violation_rate,
        "expected_rate": report.expected_rate,
        "kupiec_statistic": report.kupiec_stat,
        "kupiec_pvalue": report.kupiec_pvalue,
        "christoffersen_statistic": report.christoffersen_stat,
        "christoffersen_pvalue": report.christoffersen_pvalue,
        "traffic_light": report.traffic_light,
        "avg_violation_loss": report.avg_violation_loss,
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@router.post("/reports/daily")
def api_daily_report(
    returns_history: list[float],
    portfolio_value: float,
    greeks: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate a daily risk report."""
    from quantcore.core.types import PortfolioSnapshot
    snap = PortfolioSnapshot(
        timestamp=datetime.utcnow(),
        positions={},
        total_value=portfolio_value,
    )
    report = generate_daily_risk_report(snap, np.array(returns_history), greeks)
    return {
        "date": report.date,
        "pnl": report.pnl,
        "var_95": report.var_95,
        "var_99": report.var_99,
        "cvar_975": report.cvar_975,
        "net_delta": report.net_delta,
        "current_drawdown": report.current_drawdown,
        "max_drawdown": report.max_drawdown,
        "sharpe_rolling": report.sharpe_rolling,
    }


@router.post("/reports/weekly")
def api_weekly_report(
    daily_returns: list[float],
    attribution: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Generate a weekly performance summary."""
    r = np.array(daily_returns)
    summary = generate_weekly_summary(r, attribution=attribution)
    return {
        "pnl": summary.pnl,
        "sharpe": summary.sharpe,
        "sortino": summary.sortino,
        "calmar": summary.calmar,
        "max_dd_week": summary.max_dd_week,
        "win_rate": summary.win_rate,
        "profit_factor": summary.profit_factor,
        "top_contributors": summary.top_contributors,
        "top_detractors": summary.top_detractors,
    }
