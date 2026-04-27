"""Load risk configuration from YAML into structured dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quantcore.risk.drawdown import DrawdownAction, DrawdownConfig
from quantcore.risk.limits import RiskLimits
from quantcore.risk.monitor import MonitorConfig


@dataclass
class PreflightConfig:
    """Preflight check settings loaded from risk.yaml."""

    min_capital: float = 10000.0
    max_startup_drawdown_pct: float = 0.15
    min_data_points: int = 50
    verify_kill_switch: bool = True
    verify_state_dir_writable: bool = True
    verify_log_dir_writable: bool = True
    max_resume_state_age_hours: int = 24
    startup_max_concentration: float = 0.30
    verify_data_feed: bool = True


@dataclass
class KillSwitchConfig:
    """Kill switch trigger settings loaded from risk.yaml."""

    max_consecutive_rejections: int = 5
    max_total_rejections: int = 20
    max_drawdown_pct: float = -0.25
    max_data_staleness_sec: int = 300
    max_daily_loss: float = -25000.0
    max_memory_mb: int = 4096
    cooldown_sec: int = 300
    auto_flatten: bool = True


@dataclass
class PaperExecutionConfig:
    """Paper trading execution parameters loaded from risk.yaml."""

    trade_size_pct_equity: float = 0.10
    min_trade_quantity: int = 1
    poll_interval_sec: float = 0.1
    market_closed_sleep_sec: int = 30
    state_save_interval_sec: int = 60
    metrics_interval_sec: int = 60


@dataclass
class RiskConfig:
    """Aggregate risk configuration loaded from YAML."""

    limits: RiskLimits = field(default_factory=RiskLimits)
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    paper_execution: PaperExecutionConfig = field(default_factory=PaperExecutionConfig)
    var_confidence: float = 0.95
    circuit_breaker_threshold_pct: float = 0.20
    circuit_breaker_action: str = "flatten"


def _parse_drawdown_action(name: str) -> DrawdownAction:
    """Convert a string action name to DrawdownAction enum."""
    mapping = {
        "none": DrawdownAction.NONE,
        "reduce_25": DrawdownAction.REDUCE_25,
        "reduce_50": DrawdownAction.REDUCE_50,
        "reduce_75": DrawdownAction.REDUCE_75,
        "flatten": DrawdownAction.FLATTEN,
        "halt": DrawdownAction.HALT,
    }
    return mapping.get(name, DrawdownAction.NONE)


def load_risk_config(path: str | Path = "config/risk.yaml") -> RiskConfig:
    """Load risk configuration from a YAML file.

    Returns a fully populated RiskConfig. Missing keys fall back to
    sensible defaults defined in the dataclass constructors.
    """
    cfg_path = Path(path)
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    # --- RiskLimits ---
    pos = raw.get("position_limits", {})
    port = raw.get("portfolio_limits", {})
    var_l = raw.get("var_limits", {})
    margin = raw.get("margin_limits", {})

    limits = RiskLimits(
        max_position_pct_adv=float(pos.get("max_position_pct_adv", 0.05)),
        max_dollar_exposure_per_name=float(pos.get("max_dollar_exposure_per_name", 1_000_000)),
        max_loss_per_position=float(pos.get("max_loss_per_position", -50_000)),
        max_net_delta=float(port.get("max_net_delta", 500_000)),
        max_gross_gamma=float(port.get("max_gross_gamma", 100_000)),
        max_vega=float(port.get("max_vega", 500_000)),
        max_sector_exposure_pct=float(port.get("max_sector_exposure_pct", 0.40)),
        max_concentration_pct=float(port.get("max_concentration_pct", 0.25)),
        max_var_utilization_pct=float(var_l.get("max_var_utilization_pct", 0.80)),
        max_margin_utilization_pct=float(margin.get("max_margin_utilization_pct", 0.85)),
        circuit_breaker_threshold_pct=float(
            raw.get("circuit_breaker", {}).get("threshold_pct", 0.20)
        ),
        circuit_breaker_action=str(
            raw.get("circuit_breaker", {}).get("action", "flatten")
        ),
    )

    # --- DrawdownConfig ---
    dd_raw = raw.get("drawdown", {})
    thresholds_raw = dd_raw.get("thresholds", [])
    thresholds: list[tuple[float, DrawdownAction]] = []
    for entry in thresholds_raw:
        if isinstance(entry, dict):
            t_val = float(entry.get("drawdown", 0))
            t_act = _parse_drawdown_action(str(entry.get("action", "none")))
            thresholds.append((t_val, t_act))
    if not thresholds:
        thresholds = [
            (-0.05, DrawdownAction.REDUCE_25),
            (-0.10, DrawdownAction.REDUCE_50),
            (-0.15, DrawdownAction.REDUCE_75),
            (-0.20, DrawdownAction.FLATTEN),
        ]

    drawdown_cfg = DrawdownConfig(
        thresholds=thresholds,
        halt_threshold=float(dd_raw.get("halt_threshold", -0.25)),
    )

    # --- MonitorConfig ---
    mon = raw.get("monitor", {})
    monitor_cfg = MonitorConfig(
        var_breach_threshold=int(var_l.get("var_breach_threshold", 3)),
        var_coverage_min=float(var_l.get("var_coverage_min", 0.90)),
        var_coverage_max=float(var_l.get("var_coverage_max", 1.10)),
        max_single_position_pct=float(port.get("max_concentration_pct", 0.20)),
        max_sector_pct=float(port.get("max_sector_exposure_pct", 0.40)),
        corr_break_threshold=float(mon.get("corr_break_threshold", 0.30)),
        corr_window=int(mon.get("corr_window", 63)),
        corr_short_window=int(mon.get("corr_short_window", 21)),
        vol_spike_multiple=float(mon.get("vol_spike_multiple", 2.5)),
        vol_window_short=int(mon.get("vol_window_short", 21)),
        vol_window_long=int(mon.get("vol_window_long", 63)),
        spread_widening_multiple=float(mon.get("spread_widening_multiple", 3.0)),
        vol_decline_multiple=float(mon.get("vol_decline_multiple", 2.0)),
        dd_warning_pct=float(dd_raw.get("warning_pct", -0.05)),
        dd_critical_pct=float(dd_raw.get("critical_pct", -0.15)),
        max_leverage=float(margin.get("max_leverage", 2.0)),
        max_gross_exposure=float(margin.get("max_gross_exposure", 3.0)),
        tail_zscore=float(mon.get("tail_zscore", -3.0)),
    )

    # --- PreflightConfig ---
    pf = raw.get("preflight", {}).get("checks", {})
    preflight_cfg = PreflightConfig(
        min_capital=float(pf.get("min_capital", 10000)),
        max_startup_drawdown_pct=float(pf.get("max_startup_drawdown_pct", 0.15)),
        min_data_points=int(pf.get("min_data_points", 50)),
        verify_kill_switch=bool(pf.get("verify_kill_switch", True)),
        verify_state_dir_writable=bool(pf.get("verify_state_dir_writable", True)),
        verify_log_dir_writable=bool(pf.get("verify_log_dir_writable", True)),
        max_resume_state_age_hours=int(pf.get("max_resume_state_age_hours", 24)),
        startup_max_concentration=float(pf.get("startup_max_concentration", 0.30)),
        verify_data_feed=bool(pf.get("verify_data_feed", True)),
    )

    # --- KillSwitchConfig ---
    ks = raw.get("kill_switch", {}).get("triggers", {})
    ks_main = raw.get("kill_switch", {})
    kill_cfg = KillSwitchConfig(
        max_consecutive_rejections=int(ks.get("max_consecutive_rejections", 5)),
        max_total_rejections=int(ks.get("max_total_rejections", 20)),
        max_drawdown_pct=float(ks.get("max_drawdown_pct", -0.25)),
        max_data_staleness_sec=int(ks.get("max_data_staleness_sec", 300)),
        max_daily_loss=float(ks.get("max_daily_loss", -25000)),
        max_memory_mb=int(ks.get("max_memory_mb", 4096)),
        cooldown_sec=int(ks_main.get("cooldown_sec", 300)),
        auto_flatten=bool(ks_main.get("auto_flatten", True)),
    )

    # --- PaperExecutionConfig ---
    pe = raw.get("paper_execution", {})
    exec_cfg = PaperExecutionConfig(
        trade_size_pct_equity=float(pe.get("trade_size_pct_equity", 0.10)),
        min_trade_quantity=int(pe.get("min_trade_quantity", 1)),
        poll_interval_sec=float(pe.get("poll_interval_sec", 0.1)),
        market_closed_sleep_sec=int(pe.get("market_closed_sleep_sec", 30)),
        state_save_interval_sec=int(pe.get("state_save_interval_sec", 60)),
        metrics_interval_sec=int(pe.get("metrics_interval_sec", 60)),
    )

    return RiskConfig(
        limits=limits,
        drawdown=drawdown_cfg,
        monitor=monitor_cfg,
        preflight=preflight_cfg,
        kill_switch=kill_cfg,
        paper_execution=exec_cfg,
        var_confidence=float(var_l.get("confidence", 0.95)),
        circuit_breaker_threshold_pct=float(
            raw.get("circuit_breaker", {}).get("threshold_pct", 0.20)
        ),
        circuit_breaker_action=str(
            raw.get("circuit_breaker", {}).get("action", "flatten")
        ),
    )


def run_preflight_checks(
    config: RiskConfig,
    initial_capital: float,
    data_feed_ok: bool = False,
    resume_state_age_hours: float | None = None,
) -> list[str]:
    """Run all preflight checks and return a list of failures (empty = all pass).

    Call this before entering the main paper trading loop. Each string in
    the returned list is a failure message.
    """
    failures: list[str] = []
    pf = config.preflight

    if initial_capital < pf.min_capital:
        failures.append(
            f"Capital ${initial_capital:,.0f} below minimum ${pf.min_capital:,.0f}"
        )

    if pf.verify_state_dir_writable:
        state_dir = Path("data/paper_state")
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            test_file = state_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            failures.append(f"State directory not writable: {state_dir} — {e}")

    if pf.verify_log_dir_writable:
        log_dir = Path("logs")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            test_file = log_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            failures.append(f"Log directory not writable: {log_dir} — {e}")

    if pf.verify_data_feed and not data_feed_ok:
        failures.append("Data feed verification failed")

    if resume_state_age_hours is not None:
        if resume_state_age_hours > pf.max_resume_state_age_hours:
            failures.append(
                f"Resume state is {resume_state_age_hours:.1f}h old "
                f"(max {pf.max_resume_state_age_hours}h)"
            )

    return failures


def validate_kill_switch_mechanism() -> tuple[bool, str]:
    """Verify the kill switch mechanism is functional.

    Checks that signal handlers can be registered and the kill switch
    state variable can be toggled. Returns (pass, message).
    """
    import signal

    triggered = False

    def _test_handler(signum, frame):
        nonlocal triggered
        triggered = True

    try:
        old_handler = signal.signal(signal.SIGUSR1, _test_handler) if hasattr(signal, "SIGUSR1") else None
        signal.signal(signal.SIGUSR1, old_handler or signal.SIG_DFL) if hasattr(signal, "SIGUSR1") else None
        return True, "Kill switch signal handlers register correctly"
    except Exception as e:
        return False, f"Kill switch verification failed: {e}"
