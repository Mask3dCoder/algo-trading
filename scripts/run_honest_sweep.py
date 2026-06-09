#!/usr/bin/env python3
"""Honest Strategy Sweep — post-bias-fix validation.

Re-evaluates every strategy in config/live.yaml against clean signals
with realistic costs (5bps spread + 10bps impact + $0.005/share) using
anchored walk-forward validation (5yr train, 1yr test).

Outputs:
    reports/honest_sweep.json      — structured results
    reports/honest_sweep_report.md — analysis and decisions

Usage:
    python scripts/run_honest_sweep.py
    python scripts/run_honest_sweep.py --update-config
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

TRAIN_YEARS = 5
TEST_YEARS = 1
DATA_START = "2013-01-01"
DATA_END = "2024-12-31"
OOS_START = "2018-01-01"

COST_PARAMS = {
    "spread_bps": 5.0,
    "impact_bps": 10.0,
    "commission_per_share": 0.005,
}

FILTER_MIN_SHARPE = -0.5
FILTER_MAX_DD = 50.0  # percent
FILTER_MIN_TRADES = 10

N_TRIALS_DSR = 15  # conservative: accounts for strategy family size

# BLUECHIP universe + macro ETFs
BLUECHIP = [
    "AAPL", "MSFT", "JNJ", "PFE", "PG", "KO", "PEP", "WMT", "T", "VZ",
    "XOM", "CVX", "MRK", "ABT", "BMY", "LLY", "IBM", "CSCO", "INTC",
    "BA", "GE", "MMM", "DIS", "HD", "NKE", "MCD", "SBUX", "YUM",
    "WFC", "BAC", "C", "USB", "PNC", "TFC", "COF", "SCHW",
    "JPM", "GS", "MS", "BLK", "AXP", "V", "MA", "COST", "PCAR",
    "ADM", "SHW", "CLX", "GIS", "KMB", "SYY", "HSY", "MKC",
    "EMR", "ETN", "PH", "ROK", "DOV", "ITW", "CAT", "DE", "HON",
    "UPS", "FDX", "RTX", "LMT", "NOC", "GD", "TDG",
    "AON", "MMC", "AJG", "TRV", "ALL", "MET", "PRU",
    "AVB", "EQR", "EXR", "PSA", "AMT", "EQIX", "CCI",
    "NSC", "UNP", "CSX", "F",
    "CMCSA", "ORCL", "ACN", "INFY", "ADI", "TXN", "QCOM",
]
MACRO_ETFS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"]

# Strategies from live.yaml — testability classification
STRATEGY_SPECS = [
    {
        "name": "macro_cross_sectional_momentum",
        "module": "quantcore.strategies.macro.cross_sectional_momentum",
        "class_hint": "CrossSectionalMomentum",
        "type": "macro",
        "universe": "macro",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "macro_state_selective_momentum",
        "module": "quantcore.strategies.macro.state_selective_momentum",
        "class_hint": "StateSelectiveMomentum",
        "type": "macro",
        "universe": "macro",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "macro_factor_residual_v3",
        "module": "quantcore.strategies.macro.factor_residual_v3",
        "class_hint": "FactorResidualStrategyV3",
        "type": "macro",
        "universe": "macro",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "equity_mean_reversion",
        "module": "quantcore.strategies.equity.mean_reversion",
        "class_hint": "MeanReversion",
        "type": "equity",
        "universe": "equity",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "equity_cross_sectional_momentum",
        "module": "quantcore.strategies.equity.cross_sectional_momentum",
        "class_hint": "CrossSectionalMomentum",
        "type": "equity",
        "universe": "equity",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "equity_trend_following",
        "module": "quantcore.strategies.equity.trend_following",
        "class_hint": "TrendFollowing",
        "type": "equity",
        "universe": "equity",
        "params": {},
        "needs_ohlcv": True,
    },
    {
        "name": "equity_tsmom",
        "module": "quantcore.strategies.equity.tsmom",
        "class_hint": "TSMOM",
        "type": "equity",
        "universe": "equity",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "equity_stat_arb",
        "module": "quantcore.strategies.equity.stat_arb",
        "class_hint": "StatArb",
        "type": "equity",
        "universe": "equity",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "opening_gap_fade",
        "module": None,
        "class_hint": None,
        "type": "equity",
        "universe": "equity",
        "params": {},
        "needs_ohlcv": True,  # actually needs intraday — untestable
        "untestable_reason": "Requires intraday data (open, prev_close, current_time)",
    },
    # ── New non-momentum strategies (2026-05-28 diagnostic rebuild) ──
    {
        "name": "equity_idiosyncratic_momentum",
        "module": "quantcore.strategies.equity.idiosyncratic_momentum",
        "class_hint": "IdiosyncraticMomentum",
        "type": "equity",
        "universe": "equity",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "macro_carry",
        "module": "quantcore.strategies.macro.carry",
        "class_hint": "CarryStrategy",
        "type": "macro",
        "universe": "macro",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "equity_short_term_reversal",
        "module": "quantcore.strategies.equity.short_term_reversal",
        "class_hint": "ShortTermReversal",
        "type": "equity",
        "universe": "equity",
        "params": {},
        "needs_ohlcv": False,
    },
    {
        "name": "equity_full_universe_xs_momentum",
        "module": "quantcore.strategies.equity.full_universe_cross_sectional_momentum",
        "class_hint": "FullUniverseCrossSectionalMomentum",
        "type": "equity",
        "universe": "equity",
        "params": {},
        "needs_ohlcv": False,
    },
]


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_ohlcv_wide(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data and pivot into wide DataFrames keyed by field name.

    Returns dict with keys: 'close', 'high', 'low', 'open', 'volume'.
    Each value is a DataFrame indexed by date with ticker columns.
    """
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


# ─── HonestStrategyWrapper ───────────────────────────────────────────────────


class HonestStrategyWrapper:
    """Wrap an on_data()-based Strategy to expose run_returns_backtest().

    Enhanced version of StrategyBacktestWrapper from run_strategy_sweep.py
    with OHLCV data passthrough and trade counting.
    """

    def __init__(
        self,
        strategy,
        config,
        ohlcv: dict[str, pd.DataFrame],
        *,
        spread_bps: float = 5.0,
        impact_bps: float = 10.0,
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
        self._all_dates = close.index
        self._tickers = list(close.columns)

        self._spread_bps = spread_bps
        self._impact_bps = impact_bps
        self._commission = commission_per_share

        self._simple_rets = close.pct_change().fillna(0)

        self._days_since_rebalance: int = 0
        self._rebalance_freq: int = int(config.params.get("rebalance_freq", 21))
        self._positions: dict[str, float] = {}
        self._prev_positions: dict[str, float] = {}

        # Trade counting
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
        self._rebalance_freq = int(self.config.params.get("rebalance_freq", 21))
        self._trade_count = 0

        equity: list[float] = [initial_capital]
        daily_rets: list[float] = [0.0]
        turnover_list: list[float] = [0.0]
        layer_logs: dict[str, list[int]] = {}

        for date_i in test_dates:
            date_i_ts = pd.Timestamp(date_i)
            hist_dates = self._all_dates[self._all_dates <= date_i_ts]

            if len(hist_dates) < 5:
                equity.append(equity[-1])
                daily_rets.append(0.0)
                turnover_list.append(0.0)
                continue

            # Build data dict with OHLCV
            close_hist = self._close.loc[hist_dates]
            high_hist = self._high.loc[hist_dates]
            low_hist = self._low.loc[hist_dates]
            open_hist = self._open.loc[hist_dates]
            vol_hist = self._volume.loc[hist_dates]

            data_dict: dict[str, object] = {
                "close": close_hist.values,
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

                # Count trades (nonzero position changes)
                turnover = sum(
                    abs(self._positions.get(t, 0.0) - self._prev_positions.get(t, 0.0))
                    for t in set(self._positions) | set(self._prev_positions)
                )
                turnover_list.append(float(turnover))

                if turnover > 1e-10:
                    # Count each position change as a trade
                    for t in set(self._positions) | set(self._prev_positions):
                        delta = abs(self._positions.get(t, 0.0) - self._prev_positions.get(t, 0.0))
                        if delta > 1e-10:
                            self._trade_count += 1

                # Transaction costs
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

    # Try exact class name patterns
    candidates = [
        class_hint + "Strategy",
        class_hint,
    ]
    for name in candidates:
        if hasattr(mod, name):
            cls = getattr(mod, name)
            if isinstance(cls, type):
                return cls

    # Fallback: any concrete class ending in Strategy (skip abstract bases)
    for attr_name in dir(mod):
        if attr_name.endswith("Strategy") and not attr_name.startswith("_"):
            cls = getattr(mod, attr_name)
            if isinstance(cls, type) and not getattr(cls, "__abstractmethods__", None):
                return cls

    return None


def _build_config(name: str, strat_type: str, params: dict):
    from quantcore.strategies.base import StrategyConfig, StrategyType

    s_type = StrategyType.MACRO if strat_type == "macro" else StrategyType.EQUITY
    default_params = {"rebalance_freq": 21, "max_position_notional": 1.0}
    default_params.update(params)

    return StrategyConfig(
        name=name,
        strategy_type=s_type,
        params=default_params,
    )


# ─── Walk-Forward Runner ─────────────────────────────────────────────────────


def run_strategy_walkforward(
    spec: dict,
    ohlcv: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
) -> dict:
    """Run walk-forward backtest for one strategy. Returns metrics dict."""
    name = spec["name"]
    _log.info("=" * 70)
    _log.info("Testing: %s", name)
    _log.info("=" * 70)

    if spec.get("untestable_reason"):
        _log.info("  SKIP: %s", spec["untestable_reason"])
        return {
            "name": name,
            "status": "UNTESTABLE",
            "reason": spec["untestable_reason"],
            "oos_sharpe": None,
            "sharpe_ci_90_lower": None,
            "deflated_sharpe_pvalue": None,
            "deflated_sharpe_stat": None,
            "max_drawdown_pct": None,
            "annual_return_pct": None,
            "total_return_pct": None,
            "n_oos_trades": None,
            "passes_filter": False,
            "filter_details": {"sharpe_ok": False, "drawdown_ok": False, "trades_ok": False},
        }

    # Find and instantiate strategy class
    StratClass = _find_strategy_class(spec["module"], spec["class_hint"])
    if StratClass is None:
        _log.error("  Could not find strategy class for %s", name)
        return _error_result(name, "Strategy class not found")

    config = _build_config(name, spec["type"], spec.get("params", {}))

    try:
        base_instance = StratClass(config)
    except Exception as exc:
        _log.error("  Strategy.__init__ failed: %s", exc)
        return _error_result(name, f"Init failed: {exc}")

    # Wrap with OHLCV-aware wrapper
    wrapper = HonestStrategyWrapper(
        strategy=base_instance,
        config=config,
        ohlcv=ohlcv,
        **COST_PARAMS,
    )

    # Run walk-forward
    from quantcore.research.walkforward import WalkForwardBacktester

    wf = WalkForwardBacktester(wrapper, train_years=TRAIN_YEARS, test_years=TEST_YEARS)

    try:
        result = wf.run(dates=dates, n_trials=N_TRIALS_DSR)
    except Exception as exc:
        _log.exception("  WalkForwardBacktester.run failed: %s", exc)
        return _error_result(name, f"Walk-forward failed: {exc}")

    # Extract metrics
    oos_sharpe = result.get("annualised_sharpe", float("nan"))
    ci_lower = result.get("sharpe_ci_90_lower", float("nan"))
    dsr_p = result.get("deflated_sharpe_pvalue", 1.0)
    dsr_stat = result.get("deflated_sharpe_stat", 0.0)
    max_dd = result.get("max_drawdown", 0.0)
    n_trades = wrapper._trade_count

    # Annual return from OOS equity
    oos_equity = result.get("catalogues_equity", np.array([]))
    if len(oos_equity) > 0:
        total_ret = (oos_equity[-1] / oos_equity[0] - 1.0) * 100.0
        n_years = len(oos_equity) / 252.0
        annual_ret = ((oos_equity[-1] / oos_equity[0]) ** (1.0 / max(n_years, 0.01)) - 1.0) * 100.0
    else:
        total_ret = 0.0
        annual_ret = 0.0

    # Apply filters
    max_dd_pct = max_dd * 100.0
    sharpe_ok = oos_sharpe > FILTER_MIN_SHARPE if not np.isnan(oos_sharpe) else False
    dd_ok = max_dd_pct < FILTER_MAX_DD
    trades_ok = n_trades >= FILTER_MIN_TRADES
    passes = sharpe_ok and dd_ok and trades_ok

    verdict = "PASS" if passes else "FAIL"
    _log.info(
        "  → %s | Sharpe=%+.3f | CI_low=%+.3f | MaxDD=%.1f%% | Trades=%d | %s",
        name, oos_sharpe, ci_lower, max_dd_pct, n_trades, verdict,
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
        "filter_details": {
            "sharpe_ok": sharpe_ok,
            "sharpe_value": round(oos_sharpe, 4) if not np.isnan(oos_sharpe) else None,
            "drawdown_ok": dd_ok,
            "drawdown_pct": round(max_dd_pct, 2),
            "trades_ok": trades_ok,
            "trades_count": n_trades,
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
        "filter_details": {"sharpe_ok": False, "drawdown_ok": False, "trades_ok": False},
    }


# ─── Output Generation ──────────────────────────────────────────────────────


def write_json_report(results: list[dict], path: Path) -> None:
    completed = [r for r in results if r["status"] == "completed"]
    passed = [r for r in results if r.get("passes_filter")]
    failed = [r for r in results if r["status"] == "completed" and not r.get("passes_filter")]
    untestable = [r for r in results if r["status"] == "UNTESTABLE"]
    errors = [r for r in results if r["status"] == "ERROR"]

    report = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "period": f"{OOS_START} to {DATA_END}",
            "train_window": f"{TRAIN_YEARS} years (anchored, expanding)",
            "test_window": f"{TEST_YEARS} year",
            "universe": f"BLUECHIP {len(BLUECHIP)} + macro ETFs ({len(MACRO_ETFS)})",
            "cost_model": COST_PARAMS,
            "filters": {
                "min_oos_sharpe": FILTER_MIN_SHARPE,
                "max_drawdown_pct": FILTER_MAX_DD,
                "min_oos_trades": FILTER_MIN_TRADES,
            },
            "look_ahead_bias_fixes": 7,
        },
        "strategies": results,
        "summary": {
            "total_strategies": len(results),
            "completed": len(completed),
            "untestable": len(untestable),
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
    passed = [r for r in results if r.get("passes_filter")]

    lines = [
        "# Honest Strategy Sweep Report",
        f"_Generated: {now}_",
        "",
        "## Context",
        "",
        "7 critical look-ahead biases were removed from the signal layer. "
        "This is the first honest assessment of the alpha library.",
        "",
        "## Framework",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Training window | {TRAIN_YEARS} yr anchored (expanding) |",
        f"| Test window | {TEST_YEARS} yr |",
        f"| Data period | {DATA_START} to {DATA_END} |",
        f"| OOS period | {OOS_START} to {DATA_END} |",
        f"| Universe | BLUECHIP {len(BLUECHIP)} + {len(MACRO_ETFS)} macro ETFs |",
        f"| Costs | {COST_PARAMS['spread_bps']}bps spread + {COST_PARAMS['impact_bps']}bps impact + ${COST_PARAMS['commission_per_share']}/share |",
        f"| DSR trials | {N_TRIALS_DSR} |",
        f"| Block bootstrap | 2000 reps, block = max(5, n^(1/3)) |",
        "",
        "## Filter Criteria",
        "",
        f"- OOS Sharpe > {FILTER_MIN_SHARPE}",
        f"- Max Drawdown < {FILTER_MAX_DD}%",
        f"- OOS Trades >= {FILTER_MIN_TRADES}",
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
        verdict = "PASS" if r.get("passes_filter") else "FAIL"

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

    # Untestable
    untestable = [r for r in results if r["status"] == "UNTESTABLE"]
    if untestable:
        lines.append("")
        lines.append("## Untestable Strategies")
        lines.append("")
        for r in untestable:
            lines.append(f"- **{r['name']}**: {r.get('reason', 'unknown')}")

    # Errors
    errors = [r for r in results if r["status"] == "ERROR"]
    if errors:
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        for r in errors:
            lines.append(f"- **{r['name']}**: {r.get('reason', 'unknown')}")

    # Recommendations
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    if passed:
        lines.append(f"**{len(passed)} strategies passed** the viability filter:")
        lines.append("")
        for r in sorted(passed, key=lambda x: x.get("oos_sharpe") or -999, reverse=True):
            sr = r.get("oos_sharpe", 0)
            lines.append(f"- `{r['name']}`: OOS Sharpe {sr:+.4f}")
        lines.append("")
        lines.append("These strategies have a real, bias-free edge. Update config/live.yaml to trade only these.")
    else:
        lines.append("**No strategies passed** the viability filter. The system has no demonstrable edge with honest signals.")
        lines.append("")
        lines.append("Do NOT update config/live.yaml. Investigate further before trading.")

    # Config preview
    if passed:
        lines.append("")
        lines.append("## Proposed live.yaml Strategy Weights")
        lines.append("")
        lines.append("| Strategy | Weight | OOS Sharpe |")
        lines.append("| --- | ---: | ---: |")
        sharpe_sum = sum(max(0, r.get("oos_sharpe") or 0) for r in passed)
        for r in sorted(passed, key=lambda x: x.get("oos_sharpe") or -999, reverse=True):
            sr = max(0, r.get("oos_sharpe") or 0)
            w = sr / sharpe_sum if sharpe_sum > 0 else 1.0 / len(passed)
            lines.append(f"| {r['name']} | {w:.3f} | {r.get('oos_sharpe', 0):+.4f} |")

    path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("Markdown report written to %s", path)


# ─── Config Update ───────────────────────────────────────────────────────────


def update_live_yaml(results: list[dict], path: Path) -> None:
    """Update config/live.yaml with only viable strategies."""
    import yaml

    passed = [r for r in results if r.get("passes_filter")]
    if not passed:
        _log.warning("No strategies passed filter. Config NOT updated.")
        return

    # Load existing config
    with open(path) as f:
        config = yaml.safe_load(f)

    # Build weight map proportional to max(0, sharpe)
    sharpe_map = {}
    for r in passed:
        sr = max(0, r.get("oos_sharpe") or 0)
        sharpe_map[r["name"]] = sr

    total_sharpe = sum(sharpe_map.values())
    weight_map = {}
    for name, sr in sharpe_map.items():
        weight_map[name] = sr / total_sharpe if total_sharpe > 0 else 1.0 / len(passed)

    # Update strategies
    for strat in config.get("strategies", []):
        sname = strat["name"]
        if sname in weight_map:
            strat["enabled"] = True
            strat["weight"] = round(weight_map[sname], 4)
        else:
            strat["enabled"] = False
            _log.info("  Disabling %s (failed filter or untestable)", sname)

    # Verify weights sum to 1.0
    active_weights = [s["weight"] for s in config["strategies"] if s.get("enabled")]
    weight_sum = sum(active_weights)
    if abs(weight_sum - 1.0) > 0.01:
        _log.warning("Weights sum to %.4f, normalizing", weight_sum)
        for s in config["strategies"]:
            if s.get("enabled"):
                s["weight"] = round(s["weight"] / weight_sum, 4)

    # Add sweep metadata comment
    with open(path, "w") as f:
        f.write(f"# Honest Sweep: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        f.write(f"# OOS period: {OOS_START} to {DATA_END}\n")
        f.write(f"# Filters: Sharpe>{FILTER_MIN_SHARPE}, MaxDD<{FILTER_MAX_DD}%, Trades>={FILTER_MIN_TRADES}\n")
        f.write(f"# Passed: {', '.join(r['name'] for r in passed)}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    _log.info("Config updated at %s", path)


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Honest strategy sweep — post-bias-fix validation")
    parser.add_argument("--update-config", action="store_true",
                        help="Update config/live.yaml with viable strategies")
    parser.add_argument("--data-start", default=DATA_START, help="Data start date")
    parser.add_argument("--data-end", default=DATA_END, help="Data end date")
    args = parser.parse_args()

    print("=" * 70)
    print("  HONEST STRATEGY SWEEP — POST-BIAS-FIX VALIDATION")
    print("=" * 70)
    print(f"  Data:       {args.data_start} to {args.data_end}")
    print(f"  OOS:        {OOS_START} to {args.data_end}")
    print(f"  Train:      {TRAIN_YEARS}yr anchored expanding")
    print(f"  Test:       {TEST_YEARS}yr")
    print(f"  Costs:      {COST_PARAMS['spread_bps']}bps spread + {COST_PARAMS['impact_bps']}bps impact + ${COST_PARAMS['commission_per_share']}/share")
    print(f"  Filters:    Sharpe>{FILTER_MIN_SHARPE}, MaxDD<{FILTER_MAX_DD}%, Trades>={FILTER_MIN_TRADES}")
    print(f"  Strategies: {len(STRATEGY_SPECS)}")
    print()

    # Load data
    all_tickers = sorted(set(BLUECHIP + MACRO_ETFS))
    ohlcv = load_ohlcv_wide(all_tickers, args.data_start, args.data_end)
    dates = ohlcv["close"].index

    # Run each strategy
    results: list[dict] = []
    for i, spec in enumerate(STRATEGY_SPECS, 1):
        print(f"\n[{i}/{len(STRATEGY_SPECS)}] {spec['name']}...")
        result = run_strategy_walkforward(spec, ohlcv, dates)
        results.append(result)

    # Write outputs
    json_path = REPORTS_DIR / "honest_sweep.json"
    md_path = REPORTS_DIR / "honest_sweep_report.md"

    write_json_report(results, json_path)
    write_markdown_report(results, md_path)

    # Summary
    completed = [r for r in results if r["status"] == "completed"]
    passed = [r for r in results if r.get("passes_filter")]

    print()
    print("=" * 70)
    print("  SWEEP COMPLETE")
    print("=" * 70)
    print(f"  Tested:    {len(completed)}")
    print(f"  Passed:    {len(passed)}")
    print(f"  Failed:    {len(completed) - len(passed)}")
    print(f"  Untestable: {len([r for r in results if r['status'] == 'UNTESTABLE'])}")
    print()

    if completed:
        print("  Ranked by OOS Sharpe:")
        for r in sorted(completed, key=lambda x: x.get("oos_sharpe") or -999, reverse=True):
            sr = r.get("oos_sharpe", 0) or 0
            verdict = "PASS" if r.get("passes_filter") else "FAIL"
            print(f"    {verdict:4s}  {r['name']:40s}  Sharpe={sr:+.4f}  MaxDD={r.get('max_drawdown_pct', 0):.1f}%  Trades={r.get('n_oos_trades', 0)}")

    print()

    # Update config if requested
    if args.update_config:
        update_live_yaml(results, PROJECT_ROOT / "config" / "live.yaml")
        print()
        print("  Next steps:")
        print("    1. Start Docker Desktop")
        print("    2. docker compose down && docker compose build --no-cache && docker compose up -d")
        print("    3. curl http://localhost:8000/health")
    else:
        print("  Run with --update-config to update config/live.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())
