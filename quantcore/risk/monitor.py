"""Real-time risk monitoring: alerts, threshold surveillance, tail risk detection.

Monitors:
- VaR breaches and cumulative violations
- Concentration drift (position, sector, factor)
- Correlation breakdowns between assets
- Volatility regime shifts
- Liquidity deterioration signals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    VAR = "var"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    DRAWDOWN = "drawdown"
    LEVERAGE = "leverage"
    TAIL = "tail"
    GENERAL = "general"


@dataclass
class RiskAlert:
    """A single risk alert/monitoring event."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: AlertSeverity = AlertSeverity.INFO
    category: AlertCategory = AlertCategory.GENERAL
    message: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    breach_pct: float = 0.0  # How far past threshold
    acknowledged: bool = False


@dataclass
class MonitorConfig:
    """Configuration for risk monitoring thresholds."""
    # VaR
    var_breach_threshold: int = 3  # Consecutive VaR breaches to trigger
    var_coverage_min: float = 0.90  # Minimum acceptable VaR coverage
    var_coverage_max: float = 1.10  # Maximum acceptable coverage

    # Concentration
    max_single_position_pct: float = 0.20
    max_sector_pct: float = 0.40
    max_single_factor_pct: float = 0.35

    # Correlation
    corr_break_threshold: float = 0.30  # Absolute change to flag
    corr_window: int = 63  # Lookback for correlation comparison
    corr_short_window: int = 21

    # Volatility
    vol_spike_multiple: float = 2.5  # Short-term vol / long-term vol
    vol_window_short: int = 21
    vol_window_long: int = 63

    # Liquidity
    spread_widening_multiple: float = 3.0
    vol_decline_multiple: float = 2.0  # Volume decline relative to average

    # Drawdown
    dd_warning_pct: float = -0.05
    dd_critical_pct: float = -0.15

    # Leverage
    max_leverage: float = 2.0
    max_gross_exposure: float = 3.0

    # Tail
    tail_zscore: float = -3.0  # Z-score for tail event detection


@dataclass
class MonitorState:
    """Aggregated monitoring state at a point in time."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    active_alerts: list[RiskAlert] = field(default_factory=list)
    alert_count_24h: int = 0
    current_risk_score: float = 0.0
    highest_severity: AlertSeverity = AlertSeverity.INFO
    is_healthy: bool = True


class RiskMonitor:
    """Real-time risk monitor with threshold-based alerting.

    Usage:
        monitor = RiskMonitor(MonitorConfig())
        monitor.update_portfolio(value, positions)
        alerts = monitor.check(returns, var_estimates)
        if alerts:
            for alert in alerts:
                log.warning(alert.message)
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig()
        self._alert_history: list[RiskAlert] = []
        self._var_breach_count: int = 0
        self._consecutive_var_breaches: int = 0
        self._prev_corr: NDArray[np.floating] | None = None
        self._prev_portfolio_value: float = 0.0
        self._prev_vol: float = 0.0
        self._health_score: float = 100.0

    def check(
        self,
        returns: NDArray[np.floating],
        var_estimates: NDArray[np.floating] | None = None,
        positions: dict[str, float] | None = None,
        portfolio_value: float = 0.0,
        sector_exposures: dict[str, float] | None = None,
        factor_exposures: dict[str, float] | None = None,
        spreads: dict[str, float] | None = None,
        volumes: dict[str, float] | None = None,
        correlation_matrix: NDArray[np.floating] | None = None,
    ) -> list[RiskAlert]:
        """Run all risk monitoring checks.

        Returns list of triggered alerts (empty if all clear).
        """
        alerts: list[RiskAlert] = []

        r = np.asarray(returns, dtype=float).flatten()

        # VaR breach monitoring
        if var_estimates is not None and len(r) > 0 and len(var_estimates) > 0:
            var_alerts = self._check_var_breaches(r, var_estimates)
            alerts.extend(var_alerts)

        # Concentration monitoring
        if positions and portfolio_value > 0:
            conc_alerts = self._check_concentration(positions, portfolio_value, sector_exposures)
            alerts.extend(conc_alerts)

        # Volatility monitoring
        if len(r) >= self.config.vol_window_long:
            vol_alerts = self._check_volatility(r)
            alerts.extend(vol_alerts)

        # Drawdown monitoring
        if portfolio_value > 0:
            dd_alerts = self._check_drawdown(portfolio_value)
            alerts.extend(dd_alerts)

        # Correlation breakdown
        if correlation_matrix is not None:
            corr_alerts = self._check_correlation_break(correlation_matrix)
            alerts.extend(corr_alerts)

        # Leverage
        if positions and portfolio_value > 0:
            lev_alerts = self._check_leverage(positions, portfolio_value)
            alerts.extend(lev_alerts)

        # Tail risk
        if len(r) >= 100:
            tail_alerts = self._check_tail_risk(r)
            alerts.extend(tail_alerts)

        # Factor exposure
        if factor_exposures:
            factor_alerts = self._check_factor_concentration(factor_exposures)
            alerts.extend(factor_alerts)

        # Record and update health
        self._alert_history.extend(alerts)
        self._prune_history()
        self._update_health(alerts)

        # Update state
        if portfolio_value > 0:
            self._prev_portfolio_value = portfolio_value

        return alerts

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_var_breaches(
        self, returns: NDArray[np.floating], var_estimates: NDArray[np.floating],
    ) -> list[RiskAlert]:
        """Check for VaR violations and cumulative breach patterns."""
        alerts: list[RiskAlert] = []

        r = np.asarray(returns, dtype=float).flatten()
        v = np.asarray(var_estimates, dtype=float).flatten()
        n = min(len(r), len(v))

        recent_breaches = 0
        for i in range(max(0, n - 5), n):
            if r[i] < v[i]:
                recent_breaches += 1
                self._consecutive_var_breaches += 1
            else:
                self._consecutive_var_breaches = 0

        if self._consecutive_var_breaches >= self.config.var_breach_threshold:
            alerts.append(RiskAlert(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.VAR,
                message=f"VaR breached {self._consecutive_var_breaches} consecutive days",
                current_value=float(self._consecutive_var_breaches),
                threshold_value=float(self.config.var_breach_threshold),
                breach_pct=float(self._consecutive_var_breaches) / self.config.var_breach_threshold,
            ))

        # Coverage check over trailing window
        if n >= 20:
            breaches = int(np.sum(r[-min(n, 252):] < v[-min(n, 252):]))
            total = min(n, 252)
            actual_rate = breaches / total
            expected = 1.0 - 0.99  # Assuming 99% VaR

            if expected > 0 and actual_rate > expected * self.config.var_coverage_max:
                alerts.append(RiskAlert(
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.VAR,
                    message=f"VaR coverage {actual_rate:.1%} exceeds expected {expected:.1%}",
                    current_value=actual_rate,
                    threshold_value=expected * self.config.var_coverage_max,
                ))

        return alerts

    def _check_concentration(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        sector_exposures: dict[str, float] | None,
    ) -> list[RiskAlert]:
        """Check position and sector concentration limits."""
        alerts: list[RiskAlert] = []

        for symbol, mv in positions.items():
            pct = abs(mv) / portfolio_value
            if pct > self.config.max_single_position_pct:
                alerts.append(RiskAlert(
                    severity=AlertSeverity.WARNING if pct < self.config.max_single_position_pct * 1.5 else AlertSeverity.CRITICAL,
                    category=AlertCategory.CONCENTRATION,
                    message=f"Position {symbol} at {pct:.1%} exceeds {self.config.max_single_position_pct:.0%} limit",
                    current_value=pct,
                    threshold_value=self.config.max_single_position_pct,
                    breach_pct=pct / self.config.max_single_position_pct,
                ))

        if sector_exposures:
            for sector, pct in sector_exposures.items():
                if abs(pct) > self.config.max_sector_pct:
                    alerts.append(RiskAlert(
                        severity=AlertSeverity.WARNING,
                        category=AlertCategory.CONCENTRATION,
                        message=f"Sector {sector} at {pct:.1%} exceeds {self.config.max_sector_pct:.0%}",
                        current_value=abs(pct),
                        threshold_value=self.config.max_sector_pct,
                    ))

        return alerts

    def _check_volatility(self, returns: NDArray[np.floating]) -> list[RiskAlert]:
        """Detect volatility regime shifts/spikes."""
        r = np.asarray(returns, dtype=float).flatten()

        vol_short = float(np.std(r[-self.config.vol_window_short:], ddof=1))
        vol_long = float(np.std(r[-self.config.vol_window_long:], ddof=1))

        if vol_long > 0 and vol_short / vol_long > self.config.vol_spike_multiple:
            spike_ratio = vol_short / vol_long
            return [RiskAlert(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.VOLATILITY,
                message=f"Volatility spike: short-term vol {vol_short:.4f} is {spike_ratio:.1f}x long-term {vol_long:.4f}",
                current_value=spike_ratio,
                threshold_value=self.config.vol_spike_multiple,
                breach_pct=spike_ratio / self.config.vol_spike_multiple,
            )]

        self._prev_vol = vol_short
        return []

    def _check_drawdown(self, portfolio_value: float) -> list[RiskAlert]:
        """Check drawdown severity."""
        alerts: list[RiskAlert] = []

        if self._prev_portfolio_value > 0:
            dd_pct = (portfolio_value - self._prev_portfolio_value) / self._prev_portfolio_value

            if dd_pct < self.config.dd_critical_pct:
                alerts.append(RiskAlert(
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.DRAWDOWN,
                    message=f"Critical drawdown: {dd_pct:.1%} drop from peak",
                    current_value=dd_pct,
                    threshold_value=self.config.dd_critical_pct,
                ))
            elif dd_pct < self.config.dd_warning_pct:
                alerts.append(RiskAlert(
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.DRAWDOWN,
                    message=f"Drawdown warning: {dd_pct:.1%} from peak",
                    current_value=dd_pct,
                    threshold_value=self.config.dd_warning_pct,
                ))

        return alerts

    def _check_correlation_break(
        self, corr_matrix: NDArray[np.floating],
    ) -> list[RiskAlert]:
        """Detect correlation structure breakdowns."""
        alerts: list[RiskAlert] = []

        corr = np.asarray(corr_matrix, dtype=float)

        if self._prev_corr is not None and corr.shape == self._prev_corr.shape:
            diff = corr - self._prev_corr
            max_abs_change = float(np.max(np.abs(diff)))

            if max_abs_change > self.config.corr_break_threshold:
                # Find the pair with the biggest change
                idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
                alerts.append(RiskAlert(
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.CORRELATION,
                    message=f"Correlation break detected: max change {max_abs_change:.3f} at pair {idx}",
                    current_value=max_abs_change,
                    threshold_value=self.config.corr_break_threshold,
                ))

        self._prev_corr = corr.copy()
        return alerts

    def _check_leverage(
        self, positions: dict[str, float], portfolio_value: float,
    ) -> list[RiskAlert]:
        """Monitor leverage and gross exposure."""
        alerts: list[RiskAlert] = []

        gross = sum(abs(v) for v in positions.values())
        sum(v for v in positions.values())
        leverage = gross / portfolio_value if portfolio_value > 0 else 0.0

        if leverage > self.config.max_leverage:
            alerts.append(RiskAlert(
                severity=AlertSeverity.CRITICAL if leverage > self.config.max_gross_exposure else AlertSeverity.WARNING,
                category=AlertCategory.LEVERAGE,
                message=f"Leverage {leverage:.1f}x exceeds {self.config.max_leverage:.1f}x limit",
                current_value=leverage,
                threshold_value=self.config.max_leverage,
            ))

        return alerts

    def _check_tail_risk(self, returns: NDArray[np.floating]) -> list[RiskAlert]:
        """Check for tail events (extreme negative returns)."""
        r = np.asarray(returns, dtype=float).flatten()
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))

        if sigma < 1e-12:
            return []

        # Most recent return
        latest = r[-1]
        zscore = (latest - mu) / sigma

        if zscore < self.config.tail_zscore:
            return [RiskAlert(
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.TAIL,
                message=f"Tail event: return {latest:.4f} at {zscore:.1f}σ below mean",
                current_value=float(zscore),
                threshold_value=self.config.tail_zscore,
                breach_pct=abs(float(zscore)) / abs(self.config.tail_zscore),
            )]

        return []

    def _check_factor_concentration(
        self, factor_exposures: dict[str, float],
    ) -> list[RiskAlert]:
        """Check factor exposure concentration."""
        alerts: list[RiskAlert] = []

        for factor, exposure in factor_exposures.items():
            if abs(exposure) > self.config.max_single_factor_pct:
                alerts.append(RiskAlert(
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.CONCENTRATION,
                    message=f"Factor {factor} exposure {exposure:.1%} exceeds {self.config.max_single_factor_pct:.0%}",
                    current_value=abs(exposure),
                    threshold_value=self.config.max_single_factor_pct,
                ))

        return alerts

    # ------------------------------------------------------------------
    # Health & history
    # ------------------------------------------------------------------

    def _prune_history(self, max_age_hours: int = 72) -> None:
        """Remove alerts older than max_age_hours."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        self._alert_history = [a for a in self._alert_history if a.timestamp > cutoff]

    def _update_health(self, new_alerts: list[RiskAlert]) -> None:
        """Update health score based on alert severity and frequency."""
        for alert in new_alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                self._health_score = max(0.0, self._health_score - 15.0)
            elif alert.severity == AlertSeverity.WARNING:
                self._health_score = max(0.0, self._health_score - 5.0)
            elif alert.severity == AlertSeverity.INFO:
                self._health_score = max(0.0, self._health_score - 1.0)

        # Gradual recovery
        self._health_score = min(100.0, self._health_score + 0.1)

    def get_state(self) -> MonitorState:
        """Get current monitoring state."""
        now = datetime.utcnow()
        active = [a for a in self._alert_history if not a.acknowledged]
        alerts_24h = sum(1 for a in self._alert_history if a.timestamp > now - timedelta(hours=24))

        if any(a.severity == AlertSeverity.CRITICAL for a in active):
            highest = AlertSeverity.CRITICAL
        elif any(a.severity == AlertSeverity.WARNING for a in active):
            highest = AlertSeverity.WARNING
        else:
            highest = AlertSeverity.INFO

        return MonitorState(
            timestamp=now,
            active_alerts=active,
            alert_count_24h=alerts_24h,
            current_risk_score=self._health_score,
            highest_severity=highest,
            is_healthy=self._health_score >= 70.0,
        )

    def acknowledge(self, alert_index: int) -> None:
        """Acknowledge an active alert by index."""
        active = [a for a in self._alert_history if not a.acknowledged]
        if 0 <= alert_index < len(active):
            active[alert_index].acknowledged = True

    def acknowledge_all(self) -> int:
        """Acknowledge all active alerts. Returns count acknowledged."""
        count = 0
        for alert in self._alert_history:
            if not alert.acknowledged:
                alert.acknowledged = True
                count += 1
        return count

    def reset(self) -> None:
        """Reset all monitoring state."""
        self._alert_history.clear()
        self._var_breach_count = 0
        self._consecutive_var_breaches = 0
        self._prev_corr = None
        self._health_score = 100.0

    @property
    def health_score(self) -> float:
        return self._health_score

    @property
    def alert_history(self) -> list[RiskAlert]:
        return list(self._alert_history)


# ---------------------------------------------------------------------------
# Standalone monitoring checks
# ---------------------------------------------------------------------------


def detect_correlation_regime(
    returns: NDArray[np.floating],
    window: int = 63,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """Detect if correlation structure has changed.

    Compares short-window correlations to long-window to identify
    regime breaks in the correlation matrix.

    Args:
        returns: (n_periods, n_assets) return matrix.
        window: Short window length.
        threshold: Correlation change to flag as regime shift.

    Returns:
        {regime_change: bool, max_change: float, changed_pairs: list}
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)

    n_periods, n_assets = r.shape
    if n_periods < 2 * window or n_assets < 2:
        return {"regime_change": False, "max_change": 0.0, "changed_pairs": []}

    corr_long = np.corrcoef(r[-2 * window:-window].T)
    corr_short = np.corrcoef(r[-window:].T)

    diff = corr_short - corr_long
    np.fill_diagonal(diff, 0.0)

    max_change = float(np.max(np.abs(diff)))
    regime_change = bool(max_change > threshold)

    changed_pairs: list[dict[str, Any]] = []
    if regime_change:
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if abs(diff[i, j]) > threshold:
                    changed_pairs.append({
                        "pair": (i, j),
                        "corr_long": float(corr_long[i, j]),
                        "corr_short": float(corr_short[i, j]),
                        "change": float(diff[i, j]),
                    })

    return {
        "regime_change": regime_change,
        "max_change": max_change,
        "changed_pairs": changed_pairs,
    }


def detect_vol_regime(
    returns: NDArray[np.floating],
    short_window: int = 21,
    long_window: int = 63,
    spike_threshold: float = 2.0,
) -> dict[str, Any]:
    """Detect volatility regime shift.

    Returns:
        {regime: low/normal/elevated/high, vol_short, vol_long, ratio, is_spike}
    """
    r = np.asarray(returns, dtype=float).flatten()

    if len(r) < long_window:
        return {"regime": "unknown", "vol_short": 0.0, "vol_long": 0.0, "ratio": 1.0, "is_spike": False}

    vol_short = float(np.std(r[-short_window:], ddof=1))
    vol_long = float(np.std(r[-long_window:], ddof=1))

    if vol_long < 1e-12:
        ratio = 1.0
    else:
        ratio = vol_short / vol_long

    is_spike = bool(ratio > spike_threshold)

    if ratio < 0.5:
        regime = "low"
    elif ratio < 1.25:
        regime = "normal"
    elif ratio < spike_threshold:
        regime = "elevated"
    else:
        regime = "high"

    return {
        "regime": regime,
        "vol_short": vol_short,
        "vol_long": vol_long,
        "ratio": ratio,
        "is_spike": is_spike,
    }


def detect_liquidity_stress(
    spreads: NDArray[np.floating],
    volumes: NDArray[np.floating],
    spread_threshold: float = 3.0,
    volume_threshold: float = 2.0,
) -> dict[str, bool]:
    """Detect liquidity stress from spreads and volumes.

    Args:
        spreads: Historical bid-ask spread series.
        volumes: Historical volume series.
        spread_threshold: Multiple of avg spread to flag.
        volume_threshold: Multiple of avg volume decline to flag.

    Returns:
        {spread_stress, volume_stress, liquidity_crunch}
    """
    s = np.asarray(spreads, dtype=float).flatten()
    v = np.asarray(volumes, dtype=float).flatten()

    avg_spread = float(np.mean(s)) if len(s) > 0 else 0.0
    avg_volume = float(np.mean(v)) if len(v) > 0 else 0.0

    latest_spread = s[-1] if len(s) > 0 else 0.0
    latest_volume = v[-1] if len(v) > 0 else 0.0

    spread_stress = bool(latest_spread > avg_spread * spread_threshold) if avg_spread > 0 else False
    volume_stress = bool(latest_volume < avg_volume / volume_threshold) if avg_volume > 0 else False

    return {
        "spread_stress": spread_stress,
        "volume_stress": volume_stress,
        "liquidity_crunch": spread_stress and volume_stress,
    }
