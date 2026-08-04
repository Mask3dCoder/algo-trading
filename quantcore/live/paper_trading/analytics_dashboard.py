"""Performance & Drift Analytics Dashboard for Delta Neutral VRP Paper Trading.

Daily logging schema (CSV) that tracks execution metrics against backtest
expectations.  Alerts on significant drift from benchmark performance.

Output schema
-------------
- ``reports/paper_trading/daily_log.csv``
- ``reports/paper_trading/alerts.log``
- ``reports/paper_trading/drift_summary.json``
"""

from __future__ import annotations

import csv
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AnalyticsConfig:
    """Dashboard configuration — no overrides without committee approval."""

    # -- Paths -------------------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path("reports/paper_trading"))

    # -- Backtest benchmark ------------------------------------------------------
    # Expected Sharpe from walk-forward OOS (used for drift calculation)
    expected_sharpe_annual: float = 1.30  # walk-forward OOS Sharpe
    expected_annual_return: float = 0.12  # ~12% annual from backtest
    expected_max_dd: float = 0.1543  # 15.43% from walk-forward
    expected_win_rate: float = 0.867  # 86.7% from 2022/2023 ledger

    # -- Alert thresholds --------------------------------------------------------
    # Win rate drift
    win_rate_warn_threshold: float = 0.65  # < 65% 3-month win rate → WARNING
    win_rate_halt_threshold: float = 0.50  # < 50% → HALT review

    # P&L drift vs backtest expectations
    drift_sigma_warn: float = 2.0  # 2 sigma from expected P&L → WARNING
    drift_sigma_halt: float = 3.0  # 3 sigma → HALT review

    # Rolling Sharpe
    rolling_sharpe_window: int = 63  # ~3 months
    rolling_sharpe_halt: float = 0.0  # halt if < 0 (already enforced by RiskController)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Analytics Dashboard
# ---------------------------------------------------------------------------


class PaperTradingAnalytics:
    """Daily performance logging and drift analytics.

    Parameters
    ----------
    config : AnalyticsConfig
        Immutable configuration.
    benchmark_equity_curve : pd.Series | None
        Walk-forward OOS equity curve for P&L drift comparison.
    """

    def __init__(
        self,
        config: AnalyticsConfig,
        benchmark_equity_curve: pd.Series | None = None,
    ) -> None:
        self._cfg = config
        self._benchmark = benchmark_equity_curve

        # Daily rolling buffers
        self._daily_returns: deque[float] = deque(maxlen=config.rolling_sharpe_window)
        self._equity_history: list[tuple[date, float]] = []
        self._peak_equity = 0.0

        # Trade-level tracking
        self._closed_trades: list[dict[str, Any]] = []
        self._winning_trades: deque[float] = deque(maxlen=config.rolling_sharpe_window)
        self._trade_pnls: deque[float] = deque(maxlen=config.rolling_sharpe_window)

        # Counters
        self._today_pnl = 0.0
        self._today_trades = 0
        self._today_slippage_bps = 0.0
        self._today_slippage_dollars = 0.0

        # File handles
        self._csv_path = self._cfg.output_dir / "daily_log.csv"
        self._alert_path = self._cfg.output_dir / "alerts.log"
        self._init_csv()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_daily_result(
        self,
        dt: date,
        equity: float,
        realized_pnl: float,
        expected_pnl: float | None = None,
        slippage_bps: float = 0.0,
        slippage_dollars: float = 0.0,
        trades_executed: int = 0,
        open_positions: int = 0,
        net_delta: float = 0.0,
        vix_close: float | None = None,
        macro_exit_triggered: bool = False,
    ) -> dict[str, Any]:
        """Log a single day's results and return the generated row.

        Call at market close (or after the last fill of the day).
        """
        # -- State updates ----------------------------------------------------
        self._equity_history.append((dt, equity))
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Daily return for Sharpe
        if len(self._equity_history) >= 2:
            prev_equity = self._equity_history[-2][1]
            if prev_equity > 0:
                daily_ret = (equity - prev_equity) / prev_equity
                self._daily_returns.append(daily_ret)
        else:
            daily_ret = 0.0

        # -- Compute metrics --------------------------------------------------
        current_dd = (equity - self._peak_equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        max_dd = self._compute_max_dd()

        rolling_sharpe_1m = self._compute_rolling_sharpe(window=21)
        rolling_sharpe_3m = self._compute_rolling_sharpe(window=63)

        win_rate_all = self._compute_win_rate()
        win_rate_3m = self._compute_win_rate(window=63)
        profit_factor = self._compute_profit_factor()

        # -- P&L drift vs backtest benchmark -----------------------------------
        if expected_pnl is not None:
            drift_pct = self._compute_pnl_drift(realized_pnl, expected_pnl)
        else:
            drift_pct = None

        # -- Build log row ----------------------------------------------------
        row = {
            "date": str(dt),
            "equity": round(equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "current_drawdown": round(current_dd * 100, 2),  # percentage
            "max_drawdown": round(max_dd * 100, 2),
            "realized_pnl": round(realized_pnl, 2),
            "expected_pnl": round(expected_pnl, 2) if expected_pnl is not None else None,
            "pnl_drift_pct": round(drift_pct * 100, 2) if drift_pct is not None else None,
            "daily_return": round(daily_ret * 100, 4) if len(self._equity_history) >= 2 else 0.0,
            "rolling_sharpe_1m": round(rolling_sharpe_1m, 4) if not np.isnan(rolling_sharpe_1m) else None,
            "rolling_sharpe_3m": round(rolling_sharpe_3m, 4) if not np.isnan(rolling_sharpe_3m) else None,
            "slippage_bps": round(slippage_bps, 2),
            "slippage_dollars": round(slippage_dollars, 2),
            "win_rate_all": round(win_rate_all * 100, 1) if not np.isnan(win_rate_all) else None,
            "win_rate_3m": round(win_rate_3m * 100, 1) if not np.isnan(win_rate_3m) else None,
            "win_rate_drift_alert": win_rate_3m < self._cfg.win_rate_warn_threshold if not np.isnan(win_rate_3m) else False,
            "trades_today": trades_executed,
            "open_positions": open_positions,
            "net_delta": round(net_delta, 2),
            "vix_close": vix_close if vix_close is not None else None,
            "macro_exit_triggered": macro_exit_triggered,
            "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
        }

        # -- Write to CSV ----------------------------------------------------
        self._write_row(row)

        # -- Evaluate alerts --------------------------------------------------
        alerts = self._evaluate_alerts(row, dt)

        return {"row": row, "alerts": alerts}

    def record_trade(self, pnl: float, trade_date: date) -> None:
        """Record a closed trade for win rate / profit factor tracking."""
        self._closed_trades.append({"pnl": pnl, "date": trade_date})
        self._trade_pnls.append(pnl)
        if pnl > 0:
            self._winning_trades.append(1.0)
        else:
            self._winning_trades.append(0.0)

    def generate_summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of all paper trading activity."""
        if not self._equity_history:
            return {"status": "no_data"}

        rets = [r for d, r in self._equity_history if d]
        rets = [r for r in rets if r is not None]

        total_trades = len(self._closed_trades)
        winning = sum(1 for t in self._closed_trades if t["pnl"] > 0)

        return {
            "status": "ok",
            "total_trading_days": len(self._equity_history),
            "total_closed_trades": total_trades,
            "overall_win_rate": (winning / total_trades * 100) if total_trades > 0 else None,
            "current_equity": self._equity_history[-1][1] if self._equity_history else None,
            "peak_equity": self._peak_equity,
            "max_drawdown": self._compute_max_dd(),
            "latest_rolling_sharpe_3m": self._compute_rolling_sharpe(window=63),
            "backtest_bench_sharpe": self._cfg.expected_sharpe_annual,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        """Create CSV with headers if it doesn't exist."""
        if self._csv_path.exists():
            return

        headers = [
            "date", "equity", "peak_equity", "current_drawdown", "max_drawdown",
            "realized_pnl", "expected_pnl", "pnl_drift_pct", "daily_return",
            "rolling_sharpe_1m", "rolling_sharpe_3m",
            "slippage_bps", "slippage_dollars",
            "win_rate_all", "win_rate_3m", "win_rate_drift_alert",
            "trades_today", "open_positions", "net_delta",
            "vix_close", "macro_exit_triggered", "profit_factor",
        ]
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def _write_row(self, row: dict[str, Any]) -> None:
        """Append a row to the CSV log."""
        headers = list(row.keys())
        try:
            # Check if file exists and has content
            mode = "a" if self._csv_path.exists() else "w"
            with open(self._csv_path, mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)
        except Exception as exc:
            _log.error("CSV write failed: %s", exc)

    def _compute_max_dd(self) -> float:
        """Peak-to-trrough max drawdown from equity history."""
        if not self._equity_history or len(self._equity_history) < 2:
            return 0.0
        peak = 0.0
        max_dd = 0.0
        for d, eq in self._equity_history:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (eq - peak) / peak
                if dd < max_dd:
                    max_dd = dd
        return max_dd

    def _compute_rolling_sharpe(self, window: int) -> float:
        """Rolling Sharpe over ``window`` daily returns."""
        rets = list(self._daily_returns)
        if len(rets) < window // 3:  # need at least 1/3 of window for meaningful Sharpe
            return float("nan")
        arr = np.array(rets[-window:])
        mean_r = np.mean(arr)
        std_r = np.std(arr, ddof=1)
        if std_r < 1e-12:
            return float("nan")
        return mean_r / std_r * np.sqrt(252)

    def _compute_win_rate(self, window: int | None = None) -> float:
        """Win rate from closed trades."""
        if not self._closed_trades:
            return float("nan")
        if window:
            recent = self._closed_trades[-window:]
        else:
            recent = self._closed_trades
        if not recent:
            return float("nan")
        wins = sum(1 for t in recent if t["pnl"] > 0)
        return wins / len(recent)

    def _compute_profit_factor(self) -> float | None:
        """Gross profit / gross loss."""
        if not self._closed_trades:
            return None
        gross_profit = sum(t["pnl"] for t in self._closed_trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in self._closed_trades if t["pnl"] < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else None
        return gross_profit / gross_loss

    def _compute_pnl_drift(self, realized: float, expected: float) -> float:
        """P&L drift vs backtest benchmark as a percentage."""
        if expected == 0:
            return 0.0
        return (realized - expected) / abs(expected)

    def _evaluate_alerts(self, row: dict[str, Any], dt: date) -> list[dict[str, Any]]:
        """Check alert conditions and return list of triggered alerts."""
        alerts: list[dict[str, Any]] = []

        # 1. Win rate drift
        w3 = row.get("win_rate_3m")
        if w3 is not None and w3 < self._cfg.win_rate_warn_threshold * 100:
            alert = f"WARNING: 3-month win rate {w3:.1f}% below threshold {self._cfg.win_rate_warn_threshold * 100:.0f}%"
            _log.warning("%s", alert)
            alerts.append({"type": "win_rate_drift", "severity": "WARNING", "message": alert})
            self._write_alert(dt, "WARNING", alert)

        # 2. P&L drift
        drift = row.get("pnl_drift_pct")
        if drift is not None:
            drift_sigma = abs(drift) / 100  # scale to sigma (approximate)
            if drift_sigma > self._cfg.drift_sigma_halt:
                alert = f"HALT: P&L drift {drift:.1f}% (> {self._cfg.drift_sigma_halt}σ from benchmark). Manual review required."
                _log.critical("%s", alert)
                alerts.append({"type": "pnl_drift", "severity": "CRITICAL", "message": alert})
                self._write_alert(dt, "CRITICAL", alert)
            elif drift_sigma > self._cfg.drift_sigma_warn:
                alert = f"WARNING: P&L drift {drift:.1f}% (> {self._cfg.drift_sigma_warn}σ from benchmark)"
                _log.warning("%s", alert)
                alerts.append({"type": "pnl_drift", "severity": "WARNING", "message": alert})
                self._write_alert(dt, "WARNING", alert)

        # 3. Rolling Sharpe early warning
        rs_1m = row.get("rolling_sharpe_1m")
        if rs_1m is not None and rs_1m < 0:
            alert = f"WARNING: 1-month rolling Sharpe {rs_1m:.4f} is negative"
            _log.warning("%s", alert)
            alerts.append({"type": "sharpe_warning", "severity": "WARNING", "message": alert})
            self._write_alert(dt, "WARNING", alert)

        rs_3m = row.get("rolling_sharpe_3m")
        if rs_3m is not None and rs_3m < self._cfg.rolling_sharpe_halt:
            alert = f"HALT: 3-month rolling Sharpe {rs_3m:.4f} below threshold {self._cfg.rolling_sharpe_halt}"
            _log.critical("%s", alert)
            alerts.append({"type": "sharpe_halt", "severity": "CRITICAL", "message": alert})
            self._write_alert(dt, "CRITICAL", alert)

        # 4. Max drawdown approaching threshold (already enforced by RiskController, but log as INFO)
        cdd = row.get("current_drawdown", 0)
        if cdd is not None and cdd < -0.05:  # > 5% drawdown
            alert = f"INFO: Current drawdown {cdd:.1f}%"
            _log.info("%s", alert)
            alerts.append({"type": "drawdown", "severity": "INFO", "message": alert})

        return alerts

    def _write_alert(self, dt: date, severity: str, message: str) -> None:
        """Append an alert to the alerts log file."""
        try:
            with open(self._alert_path, "a") as f:
                f.write(f"{dt.isoformat()} [{severity}] {message}\n")
        except Exception as exc:
            _log.error("Alert log write failed: %s", exc)