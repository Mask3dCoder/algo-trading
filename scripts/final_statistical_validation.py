#!/usr/bin/env python3
"""Final Statistical Validation — Alpha Engine Gate.

Re-evaluates top strategy candidates using institutional-grade criteria:
1. 90% block-bootstrap CI lower bound (block_size=63 days, 2000 reps)
2. Deflated Sharpe Ratio p-value (Bailey & Lopez de Prado, N_trials=12)

PASS criteria: CI_lower > 0.0 AND DSR p-value < 0.1

Usage:
    python scripts/final_statistical_validation.py
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
from scipy import stats

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

# Bootstrap parameters
BOOT_BLOCK_SIZE = 63  # days (~3 months, captures momentum autocorrelation)
BOOT_N_REPS = 2000
BOOT_CI = 0.90

# DSR parameters
N_TRIALS = 12  # total unique strategies tested across all sweeps

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

# Candidates: strategies with OOS Sharpe > 0 and >= 10 trades
CANDIDATES = [
    {
        "name": "macro_cross_sectional_momentum",
        "module": "quantcore.strategies.macro.cross_sectional_momentum",
        "class_hint": "CrossSectionalMomentum",
        "type": "macro",
        "universe": "macro",
    },
    {
        "name": "equity_tsmom",
        "module": "quantcore.strategies.equity.tsmom",
        "class_hint": "TSMOM",
        "type": "equity",
        "universe": "equity",
    },
    {
        "name": "equity_full_universe_xs_momentum",
        "module": "quantcore.strategies.equity.full_universe_cross_sectional_momentum",
        "class_hint": "FullUniverseCrossSectionalMomentum",
        "type": "equity",
        "universe": "equity",
    },
    {
        "name": "equity_idiosyncratic_momentum",
        "module": "quantcore.strategies.equity.idiosyncratic_momentum",
        "class_hint": "IdiosyncraticMomentum",
        "type": "equity",
        "universe": "equity",
    },
    {
        "name": "macro_carry",
        "module": "quantcore.strategies.macro.carry",
        "class_hint": "CarryStrategy",
        "type": "macro",
        "universe": "macro",
    },
    {
        "name": "equity_short_term_reversal",
        "module": "quantcore.strategies.equity.short_term_reversal",
        "class_hint": "ShortTermReversal",
        "type": "equity",
        "universe": "equity",
    },
    {
        "name": "equity_cross_sectional_momentum",
        "module": "quantcore.strategies.equity.cross_sectional_momentum",
        "class_hint": "CrossSectionalMomentum",
        "type": "equity",
        "universe": "equity",
    },
]


# ─── Statistical Functions ──────────────────────────────────────────────────


def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    block_size: int = 63,
    n_boot: int = 2000,
    ci: float = 0.90,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Block bootstrap confidence interval for the Sharpe ratio.

    Returns (ci_lower, ci_upper, bootstrap_sharpes).
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < 30:
        return float("nan"), float("nan"), np.array([])

    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        # Circular block bootstrap
        idx = 0
        resampled = []
        while len(resampled) < n:
            start = rng.integers(0, n)
            for j in range(block_size):
                resampled.append(returns[(start + j) % n])
                if len(resampled) >= n:
                    break
        resampled = np.array(resampled[:n])
        mean_r = np.mean(resampled)
        std_r = np.std(resampled, ddof=1)
        if std_r > 1e-10:
            sharpes[b] = mean_r / std_r * np.sqrt(252)
        else:
            sharpes[b] = 0.0

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(sharpes, alpha * 100))
    ci_upper = float(np.percentile(sharpes, (1 - alpha) * 100))
    return ci_lower, ci_upper, sharpes


def expected_max_sharpe(n_trials: int) -> float:
    """Expected maximum of N independent standard normals."""
    if n_trials <= 1:
        return 0.0
    euler_mascheroni = 0.5772156649015329
    log_n = np.log(n_trials)
    return (
        np.sqrt(2.0 * log_n)
        - (np.log(log_n) + np.log(4.0 * np.pi)) / (2.0 * np.sqrt(2.0 * log_n))
        + euler_mascheroni / np.sqrt(2.0 * log_n)
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    num_trials: int,
    sample_length: int,
    skewness: float | None = None,
    kurtosis_: float | None = None,
) -> tuple[float, float]:
    """Bailey & Lopez de Prado DSR. Returns (p_value, dsr_stat)."""
    n = float(sample_length)
    expected_max = expected_max_sharpe(num_trials)

    if skewness is None or kurtosis_ is None:
        variance_term = 1.0
    else:
        variance_term = np.sqrt(
            1.0
            - skewness * observed_sharpe
            + ((kurtosis_ + 3.0) / 4.0) * observed_sharpe ** 2
        )

    dsr = (observed_sharpe - expected_max) * np.sqrt(n - 0.25) / variance_term
    p_value = 1.0 - stats.norm.cdf(dsr)
    return float(p_value), float(dsr)


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_ohlcv_wide(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
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

    return result


# ─── Strategy Wrapper (same as honest sweep) ────────────────────────────────


class HonestStrategyWrapper:
    def __init__(self, strategy, config, ohlcv, **cost_params):
        self.strategy = strategy
        self.config = config

        close = ohlcv["close"].sort_index()
        self._close = close
        self._high = ohlcv.get("high", close * 1.005).reindex(close.index).sort_index()
        self._low = ohlcv.get("low", close * 0.995).reindex(close.index).sort_index()
        self._open = ohlcv.get("open", close).reindex(close.index).sort_index()
        self._volume = ohlcv.get("volume", pd.DataFrame(1_000_000, index=close.index, columns=close.columns)).reindex(close.index).sort_index()
        self._all_dates = close.index
        self._tickers = list(close.columns)

        self._spread_bps = cost_params.get("spread_bps", 5.0)
        self._impact_bps = cost_params.get("impact_bps", 10.0)
        self._commission = cost_params.get("commission_per_share", 0.005)

        self._simple_rets = close.pct_change().fillna(0)

        self._days_since_rebalance: int = 0
        self._rebalance_freq: int = int(config.params.get("rebalance_freq", 21))
        self._positions: dict[str, float] = {}
        self._prev_positions: dict[str, float] = {}
        self._trade_count: int = 0

    def run_returns_backtest(self, *, start, end, training_start, initial_capital=1_000_000.0, **kwargs):
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

            close_hist = self._close.loc[hist_dates]
            high_hist = self._high.loc[hist_dates]
            low_hist = self._low.loc[hist_dates]
            open_hist = self._open.loc[hist_dates]
            vol_hist = self._volume.loc[hist_dates]

            data_dict = {
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


def _build_config(name: str, strat_type: str, params: dict):
    from quantcore.strategies.base import StrategyConfig, StrategyType
    s_type = StrategyType.MACRO if strat_type == "macro" else StrategyType.EQUITY
    default_params = {"rebalance_freq": 21, "max_position_notional": 1.0}
    default_params.update(params)
    return StrategyConfig(name=name, strategy_type=s_type, params=default_params)


# ─── Walk-Forward for One Strategy ──────────────────────────────────────────


def run_strategy_wf(spec: dict, ohlcv: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> dict:
    """Run walk-forward and return OOS daily returns + metrics."""
    name = spec["name"]
    _log.info("Testing: %s", name)

    StratClass = _find_strategy_class(spec["module"], spec["class_hint"])
    if StratClass is None:
        _log.error("  Could not find strategy class for %s", name)
        return {"name": name, "status": "ERROR", "reason": "Class not found"}

    config = _build_config(name, spec["type"], spec.get("params", {}))
    try:
        base_instance = StratClass(config)
    except Exception as exc:
        _log.error("  Init failed: %s", exc)
        return {"name": name, "status": "ERROR", "reason": str(exc)}

    wrapper = HonestStrategyWrapper(
        strategy=base_instance,
        config=config,
        ohlcv=ohlcv,
        **COST_PARAMS,
    )

    from quantcore.research.walkforward import WalkForwardBacktester
    wf = WalkForwardBacktester(wrapper, train_years=TRAIN_YEARS, test_years=TEST_YEARS)

    try:
        result = wf.run(dates=dates, n_trials=N_TRIALS)
    except Exception as exc:
        _log.exception("  Walk-forward failed: %s", exc)
        return {"name": name, "status": "ERROR", "reason": str(exc)}

    # Get OOS daily returns from the walk-forward
    oos_rets = result.get("catalogues_returns", np.array([]))
    oos_sharpe = result.get("annualised_sharpe", float("nan"))
    n_trades = wrapper._trade_count

    return {
        "name": name,
        "status": "completed",
        "oos_returns": oos_rets,
        "oos_sharpe": oos_sharpe,
        "n_oos_trades": n_trades,
        "n_oos_days": len(oos_rets),
        "max_drawdown_pct": result.get("max_drawdown", 0.0) * 100.0,
    }


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    _log.info("=" * 70)
    _log.info("FINAL STATISTICAL VALIDATION")
    _log.info("=" * 70)
    _log.info("Bootstrap: block_size=%d, n_reps=%d, CI=%.0f%%", BOOT_BLOCK_SIZE, BOOT_N_REPS, BOOT_CI * 100)
    _log.info("DSR: N_trials=%d", N_TRIALS)
    _log.info("Pass criteria: CI_lower > 0.0 AND DSR p-value < 0.1")
    _log.info("")

    # Load data
    universe = BLUECHIP + MACRO_ETFS
    ohlcv = load_ohlcv_wide(universe, DATA_START, DATA_END)
    dates = ohlcv["close"].index

    # Run walk-forward for each candidate
    results = []
    for spec in CANDIDATES:
        res = run_strategy_wf(spec, ohlcv, dates)
        results.append(res)

    # Compute bootstrap CI and DSR for each
    _log.info("")
    _log.info("=" * 70)
    _log.info("STATISTICAL TESTS")
    _log.info("=" * 70)

    validation_results = []
    for res in results:
        name = res["name"]

        if res["status"] != "completed":
            validation_results.append({
                "name": name,
                "status": res["status"],
                "reason": res.get("reason", ""),
                "oos_sharpe": None,
                "ci_lower": None,
                "ci_upper": None,
                "dsr_pvalue": None,
                "dsr_stat": None,
                "n_trades": None,
                "verdict": "ERROR",
            })
            continue

        oos_rets = res["oos_returns"]
        oos_sharpe = res["oos_sharpe"]
        n_trades = res["n_oos_trades"]

        if len(oos_rets) < 30:
            _log.warning("  %s: only %d OOS days, skipping", name, len(oos_rets))
            validation_results.append({
                "name": name,
                "status": "INSUFFICIENT_DATA",
                "oos_sharpe": round(oos_sharpe, 4),
                "ci_lower": None,
                "ci_upper": None,
                "dsr_pvalue": None,
                "dsr_stat": None,
                "n_trades": n_trades,
                "verdict": "INSUFFICIENT_DATA",
            })
            continue

        # 1. Bootstrap CI
        ci_lower, ci_upper, boot_sharpes = block_bootstrap_sharpe_ci(
            oos_rets, block_size=BOOT_BLOCK_SIZE, n_boot=BOOT_N_REPS, ci=BOOT_CI,
        )

        # 2. DSR
        skew = float(pd.Series(oos_rets).skew())
        kurt = float(pd.Series(oos_rets).kurtosis())  # excess kurtosis
        dsr_p, dsr_stat = deflated_sharpe_ratio(
            observed_sharpe=oos_sharpe,
            num_trials=N_TRIALS,
            sample_length=len(oos_rets),
            skewness=skew,
            kurtosis_=kurt,
        )

        # 3. Verdict
        ci_pass = ci_lower > 0.0
        dsr_pass = dsr_p < 0.1
        verdict = "PASS" if (ci_pass and dsr_pass) else "FAIL"

        _log.info(
            "  %s: Sharpe=%+.4f | CI_lower=%+.4f | DSR_p=%.4f | Trades=%d | %s",
            name, oos_sharpe, ci_lower, dsr_p, n_trades, verdict,
        )

        validation_results.append({
            "name": name,
            "status": "completed",
            "oos_sharpe": round(oos_sharpe, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "dsr_pvalue": round(dsr_p, 6),
            "dsr_stat": round(dsr_stat, 4),
            "skewness": round(skew, 4),
            "excess_kurtosis": round(kurt, 4),
            "n_oos_days": len(oos_rets),
            "n_trades": n_trades,
            "verdict": verdict,
        })

    # Write JSON results
    json_path = REPORTS_DIR / "final_statistical_validation.json"
    json_path.write_text(json.dumps({
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "bootstrap_block_size": BOOT_BLOCK_SIZE,
            "bootstrap_n_reps": BOOT_N_REPS,
            "bootstrap_ci": BOOT_CI,
            "dsr_n_trials": N_TRIALS,
            "pass_criteria": "CI_lower > 0.0 AND DSR p-value < 0.1",
        },
        "results": validation_results,
    }, indent=2, default=str), encoding="utf-8")

    # Write markdown report
    md_path = REPORTS_DIR / "final_statistical_validation.md"
    md_lines = [
        "# Final Statistical Validation",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Bootstrap:** block_size={BOOT_BLOCK_SIZE} days, {BOOT_N_REPS} replications, {BOOT_CI:.0%} CI",
        f"**DSR:** Bailey & Lopez de Prado, N_trials={N_TRIALS}",
        f"**Pass criteria:** CI_lower > 0.0 AND DSR p-value < 0.1",
        "",
        "## Results",
        "",
        "| Strategy | OOS Sharpe | CI(90%) Lower | CI(90%) Upper | DSR p-value | Skew | Kurt | Trades | Verdict |",
        "|----------|------------|---------------|---------------|-------------|------|------|--------|---------|",
    ]

    for r in validation_results:
        if r["status"] != "completed":
            md_lines.append(f"| {r['name']} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | {r['verdict']} |")
            continue
        md_lines.append(
            f"| {r['name']} | {r['oos_sharpe']:+.4f} | {r['ci_lower']:+.4f} | {r['ci_upper']:+.4f} "
            f"| {r['dsr_pvalue']:.6f} | {r['skewness']:.2f} | {r['excess_kurtosis']:.2f} "
            f"| {r['n_trades']} | **{r['verdict']}** |"
        )

    passed = [r for r in validation_results if r.get("verdict") == "PASS"]
    failed = [r for r in validation_results if r.get("verdict") == "FAIL"]

    md_lines.extend([
        "",
        "## Statistical Context",
        "",
        f"The expected maximum Sharpe under the null (with N={N_TRIALS} trials) is "
        f"**{expected_max_sharpe(N_TRIALS):.2f}**. This means a strategy needs an observed "
        f"Sharpe above ~{expected_max_sharpe(N_TRIALS):.2f} to even begin showing statistical "
        f"significance after multiple-testing correction.",
        "",
        "The DSR formula accounts for:",
        "- Non-normality (skewness and excess kurtosis of returns)",
        "- Multiple testing (number of strategies ever tested)",
        "- Sample length (number of OOS observations)",
        "",
        "## Decision",
        "",
    ])

    if passed:
        md_lines.append(f"### GO — {len(passed)} strateg{'y' if len(passed)==1 else 'ies'} passed:")
        md_lines.append("")
        for r in passed:
            md_lines.append(f"- **{r['name']}**: Sharpe={r['oos_sharpe']:+.4f}, CI_lower={r['ci_lower']:+.4f}, DSR_p={r['dsr_pvalue']:.6f}")
        md_lines.append("")
        md_lines.append("Proceed to paper trading with these strategies.")
    else:
        md_lines.extend([
            "### NO-GO — No strategy passed both statistical gates.",
            "",
            "The daily-bar alpha engine is declared **exhausted** for the current signal set.",
            "",
            "### Why all strategies failed:",
            "",
        ])
        for r in failed:
            if r.get("ci_lower") is not None:
                ci_status = "CI_lower <= 0" if r["ci_lower"] <= 0 else "CI_lower > 0"
                dsr_status = "DSR p >= 0.1" if r["dsr_pvalue"] >= 0.1 else "DSR p < 0.1"
                md_lines.append(f"- **{r['name']}**: Sharpe={r['oos_sharpe']:+.4f}, {ci_status}, {dsr_status}")

        md_lines.extend([
            "",
            "### Recommended pivot:",
            "",
            "1. **Execution alpha**: Focus on reducing transaction costs, improving fill quality, "
            "and optimizing rebalance timing rather than finding new signals.",
            "2. **Alternative data**: Incorporate non-price data (earnings, sentiment, options flow) "
            "that is not yet arbitraged by systematic funds.",
            "3. **Higher-frequency signals**: Move from daily to intraday where informational "
            "advantages decay faster and are harder to arbitrage.",
            "4. **Factor timing**: Instead of trading momentum directly, try to time when momentum "
            "works vs. when it doesn't (regime-conditional allocation).",
        ])

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    _log.info("")
    _log.info("Reports written to:")
    _log.info("  %s", json_path)
    _log.info("  %s", md_path)

    # Print summary
    _log.info("")
    _log.info("=" * 70)
    _log.info("VERDICT: %s", "GO" if passed else "NO-GO")
    _log.info("=" * 70)


if __name__ == "__main__":
    main()
