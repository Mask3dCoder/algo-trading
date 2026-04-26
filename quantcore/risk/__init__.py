"""Risk management: VaR, stress testing, limits, drawdown, greeks, reports."""

from quantcore.risk.var import (
    VaRResult,
    historical_var,
    parametric_var,
    portfolio_parametric_var,
    cornish_fisher_var,
    cvar,
    marginal_cvar,
    component_cvar,
    incremental_cvar,
    monte_carlo_var,
    monte_carlo_var_copula,
    filtered_historical_var,
)
from quantcore.risk.stress import (
    Scenario,
    StressResult,
    build_scenario_library,
    get_scenario_library,
    apply_scenario,
    stress_report,
    stress_report_dataframe,
    custom_scenario,
    correlation_stress,
    factor_stress,
    liquidity_stress,
    reverse_stress,
    concentration_report,
)
from quantcore.risk.limits import (
    LimitStatus,
    LimitCheckResult,
    RiskLimits,
    CircuitBreaker,
    check_pre_trade,
    check_post_trade,
    check_order,
)
from quantcore.risk.drawdown import (
    DrawdownAction,
    DrawdownState,
    DrawdownConfig,
    DrawdownTracker,
    drawdown_at_risk,
    conditional_dar,
    recovery_estimate,
    compute_calmar_ratio,
)
from quantcore.risk.greeks import (
    GreeksSnapshot,
    DTE_BUCKETS,
    aggregate_greeks,
    delta_adjusted_exposure,
    gamma_by_tenor,
    vega_by_tenor,
    greeks_stress,
)
from quantcore.risk.metrics import (
    DrawdownStats,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    information_ratio,
    treynor_ratio,
    jensens_alpha,
    m_squared,
    sterling_ratio,
    ulcer_index,
    pain_index,
    pain_ratio,
    burke_ratio,
    skewness,
    kurtosis,
    jarque_bera,
    drawdown_stats,
    rolling,
    bootstrap_ci,
    compute_all_metrics,
)
from quantcore.risk.var_backtest import (
    VaRBacktestReport,
    kupiec_pof_test,
    christoffersen_test,
    traffic_light,
    var_backtest_report,
)
from quantcore.risk.reports import (
    DailyRiskReport,
    WeeklySummary,
    generate_daily_risk_report,
    generate_weekly_summary,
    generate_html_report,
    generate_daily_and_weekly,
)
from quantcore.risk.attribution import (
    AttributionResult,
    RiskAttribution,
    brinson_attribution,
    factor_attribution,
    risk_attribution,
    pnl_explain,
)
from quantcore.risk.engine import (
    RiskAssessment,
    RiskEngine,
)
from quantcore.risk.budget import (
    RiskBudget,
    equal_risk_contribution,
    risk_budget_optimize,
    hierarchical_risk_parity,
    maximum_diversification,
    inverse_volatility,
    risk_decomposition,
)
from quantcore.risk.monitor import (
    AlertSeverity,
    AlertCategory,
    RiskAlert,
    MonitorConfig,
    MonitorState,
    RiskMonitor,
    detect_correlation_regime,
    detect_vol_regime,
    detect_liquidity_stress,
)
from quantcore.risk.signals import (
    RiskSignal,
    RiskRegime,
    volatility_target_scale,
    risk_on_off,
    drawdown_protection,
    tail_hedge_signal,
    correlation_hedge_signal,
    aggregate_risk_regime,
    cross_sectional_risk,
    position_sizing_gate,
)

__all__ = [
    # var
    "VaRResult", "historical_var", "parametric_var", "portfolio_parametric_var",
    "cornish_fisher_var", "cvar", "marginal_cvar", "component_cvar",
    "incremental_cvar", "monte_carlo_var", "monte_carlo_var_copula",
    "filtered_historical_var",
    # stress
    "Scenario", "StressResult", "build_scenario_library", "get_scenario_library",
    "apply_scenario", "stress_report", "stress_report_dataframe",
    "custom_scenario", "correlation_stress", "factor_stress", "liquidity_stress",
    "reverse_stress", "concentration_report",
    # limits
    "LimitStatus", "LimitCheckResult", "RiskLimits", "CircuitBreaker",
    "check_pre_trade", "check_post_trade", "check_order",
    # drawdown
    "DrawdownAction", "DrawdownState", "DrawdownConfig", "DrawdownTracker",
    "drawdown_at_risk", "conditional_dar", "recovery_estimate", "compute_calmar_ratio",
    # greeks
    "GreeksSnapshot", "DTE_BUCKETS", "aggregate_greeks", "delta_adjusted_exposure",
    "gamma_by_tenor", "vega_by_tenor", "greeks_stress",
    # metrics
    "DrawdownStats", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
    "information_ratio", "treynor_ratio", "jensens_alpha", "m_squared",
    "sterling_ratio", "ulcer_index", "pain_index", "pain_ratio", "burke_ratio",
    "skewness", "kurtosis", "jarque_bera", "drawdown_stats", "rolling",
    "bootstrap_ci", "compute_all_metrics",
    # var_backtest
    "VaRBacktestReport", "kupiec_pof_test", "christoffersen_test", "traffic_light",
    "var_backtest_report",
    # reports
    "DailyRiskReport", "WeeklySummary", "generate_daily_risk_report",
    "generate_weekly_summary", "generate_html_report", "generate_daily_and_weekly",
    # attribution
    "AttributionResult", "RiskAttribution", "brinson_attribution",
    "factor_attribution", "risk_attribution", "pnl_explain",
    # engine
    "RiskAssessment", "RiskEngine",
    # budget
    "RiskBudget", "equal_risk_contribution", "risk_budget_optimize",
    "hierarchical_risk_parity", "maximum_diversification", "inverse_volatility",
    "risk_decomposition",
    # monitor
    "AlertSeverity", "AlertCategory", "RiskAlert", "MonitorConfig",
    "MonitorState", "RiskMonitor", "detect_correlation_regime",
    "detect_vol_regime", "detect_liquidity_stress",
    # signals
    "RiskSignal", "RiskRegime", "volatility_target_scale", "risk_on_off",
    "drawdown_protection", "tail_hedge_signal", "correlation_hedge_signal",
    "aggregate_risk_regime", "cross_sectional_risk", "position_sizing_gate",
]
