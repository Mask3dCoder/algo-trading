#!/usr/bin/env python3
"""Options Strategy Sweep — walk-forward validation for volatility strategies.

Tests the four options strategies (Iron Condor, Delta Neutral, Earnings Vol Crush,
Tail Hedge) using anchored walk-forward with the same statistical gates as the
honest sweep: CI(90%) lower > 0 and DSR p-value < 0.1.

Options strategies are synthetic-IV backtests: they use close prices to compute
realized volatility as a proxy for implied volatility, then apply the strategy
logic.  This is conservative (realized vol is typically lower than IV for puts).

Outputs:
    reports/options_sweep.json      — structured results
    reports/options_sweep.md        — analysis and decisions

Usage:
    python scripts/options_sweep.py
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────

# Shorter window: options data availability is limited
TRAIN_YEARS = 2
TEST_YEARS = 1
DATA_START = "2020-01-01"
DATA_END = "2025-12-31"
OOS_START = "2022-01-01"

COST_PARAMS = {
    "spread_bps": 10.0,      # wider for options
    "impact_bps": 15.0,      # wider for options
    "commission_per_share": 0.005,
}

FILTER_MIN_SHARPE = -0.5
FILTER_MAX_DD = 50.0  # percent
FILTER_MIN_TRADES = 5  # lower for options (fewer trades expected)

N_TRIALS_DSR = 15

# Options underlyings — the 5 most liquid ETFs for options
OPTIONS_UNDERLYINGS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]

# Strategy specs for the four options strategies
OPTIONS_STRATEGY_SPECS = [
    {
        "name": "options_iron_condor",
        "module": "quantcore.strategies.options.iron_condor",
        "class_hint": "IronCondor",
        "yaml_config": "quantcore/strategy_factory/configs/iron_condor.yaml",
        "params": {
            "delta_short": 16,
            "delta_wing_width": 10,
            "dte_entry_min": 30,
            "dte_entry_max": 45,
            "ivr_entry": 40,
            "ivr_exit": 20,
            "profit_target_pct": 50,
            "max_loss_pct": 200,
            "adjustment_threshold": 3,
            "rebalance_freq": 1,
            "max_position_notional": 1.0,
            "risk_budget_pct": 0.20,
        },
    },
    {
        "name": "options_delta_neutral",
        "module": "quantcore.strategies.options.delta_neutral",
        "class_hint": "DeltaNeutral",
        "yaml_config": "quantcore/strategy_factory/configs/delta_neutral.yaml",
        "params": {
            "ivr_entry": 50,
            "ivr_exit": 25,
            "dte_min": 30,
            "dte_max": 60,
            "delta_target": 0.3,
            "hedge_freq": 1,
            "gamma_scalp_threshold": 0.01,
            "max_positions": 5,
            "rebalance_freq": 1,
            "max_position_notional": 1.0,
            "risk_budget_pct": 0.20,
        },
    },
    {
        "name": "options_earnings_vol_crush",
        "module": "quantcore.strategies.options.earnings_vol",
        "class_hint": "EarningsVol",
        "yaml_config": "quantcore/strategy_factory/configs/earnings_vol_crush.yaml",
        "params": {
            "ivr_entry": 60,
            "min_iv_premium": 0.3,
            "dte_min": 5,
            "dte_max": 14,
            "delta_target": 0.3,
            "profit_target_pct": 30,
            "max_loss_pct": 150,
            "hold_through_earnings": False,
            "rebalance_freq": 1,
            "max_position_notional": 1.0,
            "risk_budget_pct": 0.10,
        },
    },
    {
        "name": "options_tail_hedge",
        "module": "quantcore.strategies.options.tail_hedge",
        "class_hint": "TailHedge",
        "yaml_config": "quantcore/strategy_factory/configs/tail_hedge.yaml",
        "params": {
            "otm_pct": 20,
            "dte_target": 30,
            "roll_dte": 7,
            "max_allocation_pct": 0.05,
            "ivr_entry": 0,
            "ivr_exit": 999,
            "allocation_pct": 0.02,
            "rebalance_freq": 21,
            "max_position_notional": 1.0,
            "risk_budget_pct": 0.05,
        },
    },
]


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_ohlcv_wide(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data and pivot into wide DataFrames keyed by field name."""
    from quantcore.data.yfinance_loader import load_multiple

    _log.info("Downloading OHLCV for %d tickers (%s to %s)...", len(tickers), start, end)
    data = load_multiple(tickers, start=start, end=end, use_cache=True)

    fields = ["close", "high", "low", "open", "volume"]
    wide: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}

    for sym, df in data.items():
        if df.empty:
            continue
        df = df.copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        for f in fields:
            if f in df.columns:
                wide[f][sym] = df[f]

    result = {}
    for f in fields:
        if not wide[f]:
            continue
        frame = pd.DataFrame(wide[f]).sort_index().dropna(how="all")
        result[f] = frame

    if "close" not in result or result["close"].empty:
        raise ValueError("No close price data loaded")

    n_days = len(result["close"])
    n_tickers = len(result["close"].columns)
    _log.info("Loaded %d days x %d tickers", n_days, n_tickers)

    return result


def compute_realized_vol(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Compute annualized realized volatility from close prices.

    Uses rolling standard deviation of log returns, annualized by sqrt(252).
    This serves as a conservative proxy for implied volatility.
    """
    log_returns = np.log(close / close.shift(1))
    rv = log_returns.rolling(window=window, min_periods=max(5, window // 2)).std() * np.sqrt(252)
    return rv


# ─── Options Strategy Wrapper ────────────────────────────────────────────────


class OptionsStrategyWrapper:
    """Wrap an on_data()-based options strategy to expose run_returns_backtest().

    Provides close prices and realized volatility (as IV proxy) to the
    strategy's on_data() method.
    """

    def __init__(
        self,
        strategy,
        config,
        ohlcv: dict[str, pd.DataFrame],
        *,
        spread_bps: float = 10.0,
        impact_bps: float = 15.0,
        commission_per_share: float = 0.005,
    ) -> None:
        from quantcore.strategies.base import StrategyConfig

        self.strategy = strategy
        self.config: StrategyConfig = config

        close = ohlcv["close"].sort_index()
        self._close = close
        self._high = ohlcv.get("high", close * 1.005).reindex(close.index).sort_index()
        self._low = ohlcv.get("low", close * 0.995).reindex(close.index).sort_index()
        self._open = ohlcv.get("open", close).reindex(close.index).sort_index()
        self._volume = ohlcv.get("volume", pd.DataFrame(1_000_000, index=close.index, columns=close.columns)).reindex(close.index).sort_index()

        # Compute realized vol as IV proxy
        self._implied_vol = compute_realized_vol(close, window=21)

        self._all_dates = close.index
        self._tickers = list(close.columns)

        self._spread_bps = spread_bps
        self._impact_bps = impact_bps
        self._commission = commission_per_share

        self._simple_rets = close.pct_change().fillna(0)

        self._days_since_rebalance: int = 0
        self._rebalance_freq: int = int(config.params.get("rebalance_freq", 1))
        self._positions: dict[str, float] = {}
        self._prev_positions: dict[str, float] = {}
        self._trade_count: int = 0

    def run_returns_backtest(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        training_start: pd.Timestamp,
        initial_capital: float = 1_000_000.0,
        **kwargs,
    ) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
        mask = (self._all_dates >= start) & (self._all_dates <= end)
        test_dates = self._all_dates[mask]

        self._positions = {t: 0.0 for t in self._tickers}
        self._prev_positions = {t: 0.0 for t in self._tickers}
        self._days_since_rebalance = 0
        self._rebalance_freq = int(self.config.params.get("rebalance_freq", 1))
        self._trade_count = 0

        equity: list[float] = [initial_capital]
        daily_rets: list[float] = [0.0]
        turnover_list: list[float] = [0.0]
        layer_logs: dict[str, list[int]] = {}

        for date_i in test_dates:
            date_i_ts = pd.Timestamp(date_i)
            hist_dates = self._all_dates[self._all_dates <= date_i_ts]

            if len(hist_dates) < 20:
                equity.append(equity[-1])
                daily_rets.append(0.0)
                turnover_list.append(0.0)
                continue

            close_hist = self._close.loc[hist_dates]
            iv_hist = self._implied_vol.loc[hist_dates]
            high_hist = self._high.loc[hist_dates]
            low_hist = self._low.loc[hist_dates]
            open_hist = self._open.loc[hist_dates]
            vol_hist = self._volume.loc[hist_dates]

            data_dict: dict[str, object] = {
                "close": close_hist.values,
                "implied_vol": iv_hist.values,
                "high": high_hist.values,
                "low": low_hist.values,
                "open": open_hist.values,
                "volume": vol_hist.values,
                "symbols": self._tickers,
                "timestamp": date_i_ts,
            }

            self._days_since_rebalance += 1
            do_rebalance = self._days_since_rebalance >= self._rebalance_freq

            if do_rebalance:
                self._prev_positions = dict(self._positions)
                self._days_since_rebalance = 0

                signals = self.strategy.on_data(data_dict)
                new_positions = self._signals_to_weights(signals)
                self._positions = new_positions

                turnover = sum(
                    abs(self._positions.get(t, 0.0) - self._prev_positions.get(t, 0.0))
                    for t in set(self._positions) | set(self._prev_positions)
                )
                turnover_list.append(float(turnover))

                if turnover > 1e-10:
                    for t in set(self._positions) | set(self._prev_positions):
                        delta = abs(self._positions.get(t, 0.0) - self._prev_positions.get(t, 0.0))
                        if delta > 1e-10:
                            self._trade_count += 1

                delta = turnover
                cost_bps = self._spread_bps / 2.0 + self._impact_bps
                cost = equity[-1] * delta * (cost_bps / 10_000.0)
                avg_price = 100.0
                shares = equity[-1] * delta / avg_price
                commission = shares * self._commission
                total_cost = cost + commission
            else:
                turnover_list.append(0.0)
                total_cost = 0.0

            today_rets = self._simple_rets.loc[date_i_ts]
            port_ret = sum(
                self._positions.get(t, 0.0) * today_rets.get(t, 0.0)
                for t in self._positions
            )
            gross_equity = equity[-1] * (1.0 + port_ret)
            after_cost = gross_equity - total_cost

            equity.append(after_cost)
            day_ret = (equity[-1] - equity[-2]) / equity[-2] if equity[-2] != 0 else 0.0
            daily_rets.append(float(day_ret))

            n_long = sum(1 for v in self._positions.values() if v > 0)
            n_short = sum(1 for v in self._positions.values() if v < 0)
            layer_logs[str(date_i.date())] = [n_long + n_short, n_long, n_short]

        eq_series = pd.Series(equity[1:], index=test_dates)
        rets_series = pd.Series(daily_rets[1:], index=test_dates)
        turnover_series = pd.Series(turnover_list[1:], index=test_dates)

        return eq_series, rets_series, turnover_series, layer_logs

    def _signals_to_weights(self, signals) -> dict[str, float]:
        weights: dict[str, float] = {}
        max_notional = float(self.config.max_position_notional) if self.config.max_position_notional else 1.0
        if max_notional <= 0:
            max_notional = 1.0

        for sig in signals:
            if sig.direction == 0:
                weights[sig.symbol] = 0.0
                continue
            if sig.size > 0:
                w = sig.direction * sig.size / max_notional
            else:
                w = sig.direction * sig.strength
            weights[sig.symbol] = w

        return weights


# ─── Strategy Instantiation ─────────────────────────────────────────────────


def _find_strategy_class(module_path: str, class_hint: str):
    """Import module and find strategy class by hint."""
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        _log.error("Could not import %s: %s", module_path, exc)
        return None

    candidates = [class_hint + "Strategy", class_hint]
    for name in candidates:
        if hasattr(mod, name):
            cls = getattr(mod, name)
            if isinstance(cls, type):
                return cls

    for attr_name in dir(mod):
        if attr_name.endswith("Strategy") and not attr_name.startswith("_"):
            cls = getattr(mod, attr_name)
            if isinstance(cls, type) and not getattr(cls, "__abstractmethods__", None):
                return cls

    return None


def _build_config(name: str, params: dict):
    from quantcore.strategies.base import StrategyConfig, StrategyType

    default_params = {"rebalance_freq": 1, "max_position_notional": 1.0}
    default_params.update(params)

    return StrategyConfig(
        name=name,
        strategy_type=StrategyType.OPTIONS,
        params=default_params,
    )


# ─── Walk-Forward Runner ─────────────────────────────────────────────────────


def run_strategy_walkforward(
    spec: dict,
    ohlcv: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
) -> dict:
    """Run walk-forward backtest for one options strategy."""
    name = spec["name"]
    _log.info("=" * 70)
    _log.info("Testing: %s", name)
    _log.info("=" * 70)

    StratClass = _find_strategy_class(spec["module"], spec["class_hint"])
    if StratClass is None:
        _log.error("  Could not find strategy class for %s", name)
        return _error_result(name, "Strategy class not found")

    config = _build_config(name, spec.get("params", {}))

    try:
        base_instance = StratClass(config)
    except Exception as exc:
        _log.error("  Strategy.__init__ failed: %s", exc)
        return _error_result(name, f"Init failed: {exc}")

    wrapper = OptionsStrategyWrapper(
        strategy=base_instance,
        config=config,
        ohlcv=ohlcv,
        **COST_PARAMS,
    )

    from quantcore.research.walkforward import WalkForwardBacktester

    wf = WalkForwardBacktester(wrapper, train_years=TRAIN_YEARS, test_years=TEST_YEARS)

    try:
        result = wf.run(dates=dates, n_trials=N_TRIALS_DSR)
    except Exception as exc:
        _log.exception("  WalkForwardBacktester.run failed: %s", exc)
        return _error_result(name, f"Walk-forward failed: {exc}")

    oos_sharpe = result.get("annualised_sharpe", float("nan"))
    ci_lower = result.get("sharpe_ci_90_lower", float("nan"))
    dsr_p = result.get("deflated_sharpe_pvalue", 1.0)
    dsr_stat = result.get("deflated_sharpe_stat", 0.0)
    max_dd = result.get("max_drawdown", 0.0)
    n_trades = wrapper._trade_count

    oos_equity = result.get("catalogues_equity", np.array([]))
    if len(oos_equity) > 0:
        total_ret = (oos_equity[-1] / oos_equity[0] - 1.0) * 100.0
        n_years = len(oos_equity) / 252.0
        annual_ret = ((oos_equity[-1] / oos_equity[0]) ** (1.0 / max(n_years, 0.01)) - 1.0) * 100.0
    else:
        total_ret = 0.0
        annual_ret = 0.0

    max_dd_pct = max_dd * 100.0
    sharpe_ok = oos_sharpe > FILTER_MIN_SHARPE if not np.isnan(oos_sharpe) else False
    dd_ok = max_dd_pct < FILTER_MAX_DD
    trades_ok = n_trades >= FILTER_MIN_TRADES
    passes = sharpe_ok and dd_ok and trades_ok

    # Additional gate: CI(90%) lower > 0 and DSR p-value < 0.1
    ci_passes = ci_lower > 0 if not np.isnan(ci_lower) else False
    dsr_passes = dsr_p < 0.1 if not np.isnan(dsr_p) else False
    stat_passes = ci_passes and dsr_passes

    verdict = "PASS" if passes and stat_passes else "FAIL"
    _log.info(
        "  → %s | Sharpe=%+.3f | CI_low=%+.3f | DSR_p=%.4f | MaxDD=%.1f%% | Trades=%d | %s",
        name, oos_sharpe, ci_lower, dsr_p, max_dd_pct, n_trades, verdict,
    )

    return {
        "name": name,
        "status": "completed",
        "oos_sharpe": round(oos_sharpe, 4) if not np.isnan(oos_sharpe) else None,
        "sharpe_ci_90_lower": round(ci_lower, 4) if not np.isnan(ci_lower) else None,
        "deflated_sharpe_pvalue": round(dsr_p, 4),
        "deflated_sharpe_stat": round(dsr_stat, 4),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "annual_return_pct": round(annual_ret, 2),
        "total_return_pct": round(total_ret, 2),
        "n_oos_trades": n_trades,
        "n_folds": len(result.get("fold_results", [])),
        "passes_filter": passes,
        "passes_statistical": stat_passes,
        "ci_lower_passes": ci_passes,
        "dsr_pvalue_passes": dsr_passes,
        "filter_details": {
            "sharpe_ok": sharpe_ok,
            "sharpe_value": round(oos_sharpe, 4) if not np.isnan(oos_sharpe) else None,
            "drawdown_ok": dd_ok,
            "drawdown_pct": round(max_dd_pct, 2),
            "trades_ok": trades_ok,
            "trades_count": n_trades,
            "ci_lower_passes": ci_passes,
            "ci_lower_value": round(ci_lower, 4) if not np.isnan(ci_lower) else None,
            "dsr_pvalue_passes": dsr_passes,
            "dsr_pvalue": round(dsr_p, 4),
        },
    }


def _error_result(name: str, reason: str) -> dict:
    return {
        "name": name,
        "status": "ERROR",
        "reason": reason,
        "oos_sharpe": None,
        "sharpe_ci_90_lower": None,
        "deflated_sharpe_pvalue": None,
        "deflated_sharpe_stat": None,
        "max_drawdown_pct": None,
        "annual_return_pct": None,
        "total_return_pct": None,
        "n_oos_trades": None,
        "passes_filter": False,
        "passes_statistical": False,
        "filter_details": {"sharpe_ok": False, "drawdown_ok": False, "trades_ok": False},
    }


# ─── Output Generation ──────────────────────────────────────────────────────


def write_json_report(results: list[dict], path: Path) -> None:
    completed = [r for r in results if r["status"] == "completed"]
    passed = [r for r in results if r.get("passes_filter") and r.get("passes_statistical")]
    failed = [r for r in results if r["status"] == "completed" and not (r.get("passes_filter") and r.get("passes_statistical"))]
    errors = [r for r in results if r["status"] == "ERROR"]

    report = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "period": f"{OOS_START} to {DATA_END}",
            "train_window": f"{TRAIN_YEARS} years (anchored, expanding)",
            "test_window": f"{TEST_YEARS} year",
            "universe": f"Options underlyings: {', '.join(OPTIONS_UNDERLYINGS)}",
            "cost_model": COST_PARAMS,
            "filters": {
                "min_oos_sharpe": FILTER_MIN_SHARPE,
                "max_drawdown_pct": FILTER_MAX_DD,
                "min_oos_trades": FILTER_MIN_TRADES,
                "ci_90_lower_gt_0": True,
                "dsr_pvalue_lt_0_1": True,
            },
            "iv_proxy": "21-day annualized realized volatility",
        },
        "strategies": results,
        "summary": {
            "total_strategies": len(results),
            "completed": len(completed),
            "errors": len(errors),
            "passed_filter": len(passed),
            "failed_filter": len(failed),
        },
    }

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log.info("JSON report written to %s", path)


def write_markdown_report(results: list[dict], path: Path) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M %Z")
    completed = [r for r in results if r["status"] == "completed"]
    passed = [r for r in results if r.get("passes_filter") and r.get("passes_statistical")]

    lines = [
        "# Options Strategy Sweep Report",
        f"_Generated: {now}_",
        "",
        "## Context",
        "",
        "The daily-bar alpha engine has been statistically proven dead (DSR p-values = 1.0). "
        "This sweep evaluates four options strategies that harvest the volatility risk premium: "
        "Iron Condor, Delta Neutral, Earnings Vol Crush, and Tail Hedge.",
        "",
        "These strategies use realized volatility (21-day) as a proxy for implied volatility. "
        "In production, they would use actual options chain data from the Polygon.io pipeline.",
        "",
        "## Framework",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Training window | {TRAIN_YEARS} yr anchored (expanding) |",
        f"| Test window | {TEST_YEARS} yr |",
        f"| Data period | {DATA_START} to {DATA_END} |",
        f"| OOS period | {OOS_START} to {DATA_END} |",
        f"| Universe | {', '.join(OPTIONS_UNDERLYINGS)} |",
        f"| Costs | {COST_PARAMS['spread_bps']}bps spread + {COST_PARAMS['impact_bps']}bps impact + ${COST_PARAMS['commission_per_share']}/share |",
        f"| DSR trials | {N_TRIALS_DSR} |",
        f"| IV proxy | 21-day annualized realized volatility |",
        "",
        "## Filter Criteria",
        "",
        f"- OOS Sharpe > {FILTER_MIN_SHARPE}",
        f"- Max Drawdown < {FILTER_MAX_DD}%",
        f"- OOS Trades >= {FILTER_MIN_TRADES}",
        "- CI(90%) lower > 0 (Sharpe significance)",
        "- DSR p-value < 0.1 (deflated significance)",
        "",
        "## Results",
        "",
        "| Strategy | Status | OOS Sharpe | CI(90%) Low | DSR p-val | MaxDD% | Annual% | Trades | Verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for r in sorted(completed, key=lambda x: x.get("oos_sharpe") or -999, reverse=True):
        sr = r.get("oos_sharpe")
        ci = r.get("sharpe_ci_90_lower")
        dsr = r.get("deflated_sharpe_pvalue")
        dd = r.get("max_drawdown_pct")
        ar = r.get("annual_return_pct")
        trades = r.get("n_oos_trades")
        passes = r.get("passes_filter") and r.get("passes_statistical")
        verdict = "PASS" if passes else "FAIL"

        sr_s = f"{sr:+.4f}" if sr is not None else "N/A"
        ci_s = f"{ci:+.4f}" if ci is not None else "N/A"
        dsr_s = f"{dsr:.4f}" if dsr is not None else "N/A"
        dd_s = f"{dd:.2f}" if dd is not None else "N/A"
        ar_s = f"{ar:+.2f}" if ar is not None else "N/A"
        tr_s = str(trades) if trades is not None else "N/A"

        lines.append(
            f"| {r['name']} | {r['status']} | {sr_s} | {ci_s} | {dsr_s} "
            f"| {dd_s} | {ar_s} | {tr_s} | **{verdict}** |"
        )

    errors = [r for r in results if r["status"] == "ERROR"]
    if errors:
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        for r in errors:
            lines.append(f"- **{r['name']}**: {r.get('reason', 'unknown')}")

    # Detailed analysis per strategy
    lines.append("")
    lines.append("## Strategy Analysis")
    lines.append("")

    for r in sorted(completed, key=lambda x: x.get("oos_sharpe") or -999, reverse=True):
        sr = r.get("oos_sharpe", 0) or 0
        ci = r.get("sharpe_ci_90_lower", 0) or 0
        dsr = r.get("deflated_sharpe_pvalue", 1.0) or 1.0
        dd = r.get("max_drawdown_pct", 0) or 0
        passes = r.get("passes_filter") and r.get("passes_statistical")

        lines.append(f"### {r['name']}")
        lines.append("")
        if passes:
            lines.append("**EDGE CONFIRMED** — passes all statistical gates.")
        else:
            reasons = []
            if sr <= FILTER_MIN_SHARPE:
                reasons.append(f"OOS Sharpe {sr:+.4f} <= {FILTER_MIN_SHARPE}")
            if dd >= FILTER_MAX_DD:
                reasons.append(f"Max DD {dd:.1f}% >= {FILTER_MAX_DD}%")
            if (r.get("n_oos_trades") or 0) < FILTER_MIN_TRADES:
                reasons.append(f"Trades {r.get('n_oos_trades', 0)} < {FILTER_MIN_TRADES}")
            if ci <= 0:
                reasons.append(f"CI(90%) lower {ci:+.4f} <= 0")
            if dsr >= 0.1:
                reasons.append(f"DSR p-value {dsr:.4f} >= 0.1")
            lines.append(f"**EDGE NOT CONFIRMED** — {', '.join(reasons)}.")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    if passed:
        lines.append(f"**{len(passed)} strategies passed** the statistical gates:")
        lines.append("")
        for r in sorted(passed, key=lambda x: x.get("oos_sharpe") or -999, reverse=True):
            sr = r.get("oos_sharpe", 0)
            lines.append(f"- `{r['name']}`: OOS Sharpe {sr:+.4f}")
        lines.append("")
        lines.append(
            "These strategies demonstrate a real, bias-free edge with the volatility risk premium. "
            "Proceed to paper trading with actual options chain data from the Polygon.io pipeline."
        )
    else:
        lines.append("**No strategies passed** the statistical gates.")
        lines.append("")
        lines.append(
            "The volatility risk premium strategies do not demonstrate a statistically significant "
            "edge using realized volatility as an IV proxy. Possible next steps:"
        )
        lines.append("")
        lines.append("1. Test with actual implied volatility data from the options chain pipeline")
        lines.append("2. Adjust strategy parameters (wider DTE, different delta targets)")
        lines.append("3. Consider that the VRP may be too efficiently priced in liquid ETFs")

    path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("Markdown report written to %s", path)


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Options strategy sweep — volatility risk premium validation")
    parser.add_argument("--data-start", default=DATA_START, help="Data start date")
    parser.add_argument("--data-end", default=DATA_END, help="Data end date")
    args = parser.parse_args()

    print("=" * 70)
    print("  OPTIONS STRATEGY SWEEP — VOLATILITY RISK PREMIUM VALIDATION")
    print("=" * 70)
    print(f"  Data:       {args.data_start} to {args.data_end}")
    print(f"  OOS:        {OOS_START} to {args.data_end}")
    print(f"  Train:      {TRAIN_YEARS}yr anchored expanding")
    print(f"  Test:       {TEST_YEARS}yr")
    print(f"  Costs:      {COST_PARAMS['spread_bps']}bps spread + {COST_PARAMS['impact_bps']}bps impact")
    print(f"  Filters:    Sharpe>{FILTER_MIN_SHARPE}, MaxDD<{FILTER_MAX_DD}%, Trades>={FILTER_MIN_TRADES}")
    print(f"  Stat gates: CI(90%) lower > 0, DSR p-value < 0.1")
    print(f"  Strategies: {len(OPTIONS_STRATEGY_SPECS)}")
    print(f"  IV proxy:   21-day realized volatility")
    print()

    # Load data
    ohlcv = load_ohlcv_wide(OPTIONS_UNDERLYINGS, args.data_start, args.data_end)
    dates = ohlcv["close"].index

    # Run each strategy
    results: list[dict] = []
    for i, spec in enumerate(OPTIONS_STRATEGY_SPECS, 1):
        print(f"\n[{i}/{len(OPTIONS_STRATEGY_SPECS)}] {spec['name']}...")
        result = run_strategy_walkforward(spec, ohlcv, dates)
        results.append(result)

    # Write outputs
    json_path = REPORTS_DIR / "options_sweep.json"
    md_path = REPORTS_DIR / "options_sweep.md"

    write_json_report(results, json_path)
    write_markdown_report(results, md_path)

    # Summary
    completed = [r for r in results if r["status"] == "completed"]
    passed = [r for r in results if r.get("passes_filter") and r.get("passes_statistical")]

    print()
    print("=" * 70)
    print("  SWEEP COMPLETE")
    print("=" * 70)
    print(f"  Tested:    {len(completed)}")
    print(f"  Passed:    {len(passed)}")
    print(f"  Failed:    {len(completed) - len(passed)}")
    print()

    if completed:
        print("  Ranked by OOS Sharpe:")
        for r in sorted(completed, key=lambda x: x.get("oos_sharpe") or -999, reverse=True):
            sr = r.get("oos_sharpe", 0) or 0
            ci = r.get("sharpe_ci_90_lower", 0) or 0
            dsr = r.get("deflated_sharpe_pvalue", 1.0) or 1.0
            passes = r.get("passes_filter") and r.get("passes_statistical")
            verdict = "PASS" if passes else "FAIL"
            print(f"    {verdict:4s}  {r['name']:35s}  Sharpe={sr:+.4f}  CI_low={ci:+.4f}  DSR_p={dsr:.4f}")

    print()
    print(f"  Reports: {json_path}")
    print(f"           {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
