"""Risk management: VaR, stress testing, limits, drawdown, greeks, reports."""

from quantcore.risk.attribution import (
    AttributionResult,
    RiskAttribution,
    brinson_attribution,
    factor_attribution,
    pnl_explain,
    risk_attribution,
)
from quantcore.risk.budget import (
    RiskBudget,
    equal_risk_contribution,
    hierarchical_risk_parity,
    inverse_volatility,
    maximum_diversification,
    risk_budget_optimize,
    risk_decomposition,
)
from quantcore.risk.config_loader import (
    KillSwitchConfig,
    PaperExecutionConfig,
    PreflightConfig,
    RiskConfig,
    load_risk_config,
    run_preflight_checks,
    validate_kill_switch_mechanism,
)
from quantcore.risk.drawdown import (
    DrawdownAction,
    DrawdownConfig,
    DrawdownState,
    DrawdownTracker,
    compute_calmar_ratio,
    conditional_dar,
    drawdown_at_risk,
    recovery_estimate,
)
from quantcore.risk.engine import (
    RiskAssessment,
    RiskEngine,
)
from quantcore.risk.greeks import (
    DTE_BUCKETS,
    GreeksSnapshot,
    aggregate_greeks,
    delta_adjusted_exposure,
    gamma_by_tenor,
    greeks_stress,
    vega_by_tenor,
)
from quantcore.risk.limits import (
    CircuitBreaker,
    LimitCheckResult,
    LimitStatus,
    RiskLimits,
    check_order,
    check_post_trade,
    check_pre_trade,
)
from quantcore.risk.metrics import (
    DrawdownStats,
    bootstrap_ci,
    burke_ratio,
    calmar_ratio,
    compute_all_metrics,
    drawdown_stats,
    information_ratio,
    jarque_bera,
    jensens_alpha,
    kurtosis,
    m_squared,
    omega_ratio,
    pain_index,
    pain_ratio,
    rolling,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    sterling_ratio,
    treynor_ratio,
    ulcer_index,
)
from quantcore.risk.monitor import (
    AlertCategory,
    AlertSeverity,
    MonitorConfig,
    MonitorState,
    RiskAlert,
    RiskMonitor,
    detect_correlation_regime,
    detect_liquidity_stress,
    detect_vol_regime,
)
from quantcore.risk.reports import (
    DailyRiskReport,
    WeeklySummary,
    generate_daily_and_weekly,
    generate_daily_risk_report,
    generate_html_report,
    generate_weekly_summary,
)
from quantcore.risk.signals import (
    RiskRegime,
    RiskSignal,
    aggregate_risk_regime,
    correlation_hedge_signal,
    cross_sectional_risk,
    drawdown_protection,
    position_sizing_gate,
    risk_on_off,
    tail_hedge_signal,
    volatility_target_scale,
)
from quantcore.risk.stress import (
    Scenario,
    StressResult,
    apply_scenario,
    build_scenario_library,
    concentration_report,
    correlation_stress,
    custom_scenario,
    factor_stress,
    get_scenario_library,
    liquidity_stress,
    reverse_stress,
    stress_report,
    stress_report_dataframe,
)
from quantcore.risk.var import (
    VaRResult,
    component_cvar,
    cornish_fisher_var,
    cvar,
    filtered_historical_var,
    historical_var,
    incremental_cvar,
    marginal_cvar,
    monte_carlo_var,
    monte_carlo_var_copula,
    parametric_var,
    portfolio_parametric_var,
)
from quantcore.risk.var_backtest import (
    VaRBacktestReport,
    christoffersen_test,
    kupiec_pof_test,
    traffic_light,
    var_backtest_report,
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
    # config_loader
    "RiskConfig", "PreflightConfig", "KillSwitchConfig", "PaperExecutionConfig",
    "load_risk_config", "run_preflight_checks", "validate_kill_switch_mechanism",
    # signals
    "RiskSignal", "RiskRegime", "volatility_target_scale", "risk_on_off",
    "drawdown_protection", "tail_hedge_signal", "correlation_hedge_signal",
    "aggregate_risk_regime", "cross_sectional_risk", "position_sizing_gate",
]
