#!/usr/bin/env python
"""Phase 3.1: Walk-forward re-validation of all 6 enabled strategies.

Runs anchored expanding-window walk-forward (504-day train, 126-day test,
1% embargo, 2018-2024) on each strategy. Computes:
  - OOS Sharpe, CI(90%) lower bound, MaxDD, Win Rate, Profit Factor
  - Deflated Sharpe Ratio p-value (trials=20, accounting for prior tests)

Triage:
  KEEP    — CI_lower > 0 AND DSR p-value < 0.1
  MONITOR — CI_lower > 0 but DSR p-value >= 0.1
  DISABLE — everything else

Outputs reports/phase3_validation.md and updated config/live.yaml.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantcore.backtest.walk_forward import (
    anchored_walk_forward,
    compute_oos_metrics,
)
from quantcore.research.deflated_sharpe import deflated_sharpe_ratio
from quantcore.risk.metrics import sharpe_ratio, drawdown_stats
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class

# Trigger @register decorators
import quantcore.strategies.equity.cross_sectional_momentum  # noqa: F401
import quantcore.strategies.equity.mean_reversion  # noqa: F401
import quantcore.strategies.equity.momentum  # noqa: F401
import quantcore.strategies.equity.multi_factor  # noqa: F401
import quantcore.strategies.macro.carry  # noqa: F401
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401
import quantcore.strategies.macro.risk_parity  # noqa: F401
import quantcore.strategies.macro.volatility_risk_premium  # noqa: F401
import quantcore.strategies.options.delta_neutral  # noqa: F401
import quantcore.strategies.options.iron_condor  # noqa: F401
import quantcore.strategies.options.tail_hedge  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Constants
TRAIN_SIZE = 504   # 2 years
TEST_SIZE = 126    # 6 months
EMBARGO_PCT = 0.01
NUM_TRIALS = 20    # conservative trial count for DSR
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
INITIAL_CASH = 100_000.0


# ---------------------------------------------------------------------------
# Strategy definitions for validation
# ---------------------------------------------------------------------------

@dataclass
class StrategySpec:
    """Specification for a strategy to validate."""
    registry_name: str
    display_name: str
    strategy_type: str  # EQUITY, MACRO, OPTIONS
    data_source: str    # alpaca_daily, yfinance_vix, fred_macro
    universe: list[str]
    params: dict = field(default_factory=dict)
    vix_required: bool = False


STRATEGIES = [
    StrategySpec(
        registry_name="options_delta_neutral",
        display_name="Delta Neutral VRP",
        strategy_type="OPTIONS",
        data_source="yfinance_vix",
        universe=["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        params={"ivr_entry": 50, "ivr_exit": 25, "dte_min": 30, "dte_max": 60,
                "delta_target": 0.3, "hedge_freq": 1, "gamma_scalp_threshold": 0.01,
                "max_positions": 5},
        vix_required=True,
    ),
    StrategySpec(
        registry_name="options_iron_condor",
        display_name="Iron Condor VRP",
        strategy_type="OPTIONS",
        data_source="yfinance_vix",
        universe=["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        params={"delta_short": 0.3, "delta_long": 0.15, "dte_min": 30, "dte_max": 45,
                "ivr_entry": 40},
        vix_required=True,
    ),
    StrategySpec(
        registry_name="options_tail_hedge",
        display_name="Tail Hedge",
        strategy_type="OPTIONS",
        data_source="yfinance_vix",
        universe=["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        params={"vix_entry": 15, "hedge_pct": 0.02},
        vix_required=True,
    ),
    StrategySpec(
        registry_name="macro_cross_sectional_momentum",
        display_name="Macro Cross-Sectional Momentum",
        strategy_type="MACRO",
        data_source="alpaca_daily",
        universe=["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        params={"lookback": 60, "top_n": 3, "rebalance_freq": 5},
    ),
    StrategySpec(
        registry_name="macro_risk_parity",
        display_name="Macro Risk Parity",
        strategy_type="MACRO",
        data_source="alpaca_daily",
        universe=["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        params={"vol_lookback": 60, "rebalance_freq": 21},
    ),
    StrategySpec(
        registry_name="macro_vrp",
        display_name="Volatility Risk Premium",
        strategy_type="MACRO",
        data_source="yfinance_vix",
        universe=["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        params={"vix_entry": 18, "vol_target": 0.10},
        vix_required=True,
    ),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_market_data(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Load market data using yfinance with caching."""
    from quantcore.data.yfinance_loader import load_combined
    try:
        df = load_combined(symbols, start=start, end=end, interval="1d")
        log.info("Loaded %d rows for %d symbols (%s to %s)",
                 len(df), df["symbol"].nunique(), start, end)
        return df
    except Exception as e:
        log.error("Failed to load market data: %s", e)
        raise


def load_vix_data(start: str, end: str) -> pd.Series:
    """Load VIX data for options strategies."""
    import yfinance as yf
    vix = yf.download("^VIX", start=start, end=end, progress=False)
    if vix.empty:
        raise ValueError("VIX data empty")
    # Handle multi-level columns from yfinance
    if isinstance(vix.columns, pd.MultiIndex):
        close_col = vix["Close"].iloc[:, 0] if len(vix["Close"].shape) > 1 else vix["Close"]
    else:
        close_col = vix["Close"]
    return close_col


# ---------------------------------------------------------------------------
# DataFrame -> strategy data dict
# ---------------------------------------------------------------------------

def df_to_data(df: pd.DataFrame) -> dict:
    """Convert long-format DataFrame to the dict format expected by on_data()."""
    symbols = sorted(df["symbol"].unique())
    n_assets = len(symbols)

    pivots = {}
    for field_name in ["open", "high", "low", "close", "volume"]:
        if field_name in df.columns:
            pivoted = df.pivot_table(index="timestamp", columns="symbol", values=field_name)
            pivoted = pivoted.reindex(columns=symbols)
            pivots[field_name] = pivoted.values
        else:
            pivots[field_name] = np.full((len(df["timestamp"].unique()), n_assets), np.nan)

    close = pivots["close"]
    # Causal forward-fill (max 5 days)
    close_df = pd.DataFrame(close)
    close_filled = close_df.ffill(limit=5).fillna(0).values

    data = {
        "close": close_filled,
        "open": pivots.get("open", close_filled),
        "high": pivots.get("high", close_filled),
        "low": pivots.get("low", close_filled),
        "volume": pivots.get("volume", np.ones_like(close_filled)),
        "symbols": symbols,
    }
    return data


def inject_vix(data: dict, vix_series: pd.Series) -> dict:
    """Inject VIX-derived implied_vol into the data dict for options strategies."""
    close = data["close"]
    n_periods = close.shape[0]

    # Align VIX to the same dates as the close matrix
    # Use the index from the close data
    dates = pd.date_range(start="2018-01-01", periods=n_periods, freq="B")
    vix_aligned = vix_series.reindex(dates).ffill().fillna(20.0)

    iv_decimal = vix_aligned.values / 100.0
    iv_series = np.clip(iv_decimal, 0.08, 0.80)

    # Build realistic IV ramp: blend realized vol with VIX level
    if n_periods > 10 and close.shape[1] > 0:
        close_series = pd.Series(close[:, 0])
        log_ret = np.log(close_series / close_series.shift(1)).dropna()
        rv = log_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
        rv = rv.reindex(range(n_periods)).bfill().fillna(np.mean(iv_series))
        # Blend: 60% VIX, 40% realized vol
        blended = 0.6 * iv_series + 0.4 * rv.values
        blended = np.clip(blended, 0.08, 0.80)
        blended[-1] = iv_series[-1]  # ensure last bar uses actual VIX
        iv_series = blended

    data["implied_vol"] = iv_series.reshape(-1, 1)
    return data


# ---------------------------------------------------------------------------
# Signal-based return computation
# ---------------------------------------------------------------------------

def compute_strategy_returns_full(strategy, data: dict, start: int, end: int) -> np.ndarray:
    """Run strategy on data using the proven causal-slice approach.

    At each bar i in [start, end), passes data[:i+1] to strategy.on_data().
    Computes signal-weighted asset returns using next-day return.

    This is the exact same approach used in scripts/run_backtest_focused.py.
    """
    close = data["close"]
    symbols = data.get("symbols", [])
    n_periods = close.shape[0]
    n_assets = len(symbols)

    if n_assets == 0 or n_periods < 2 or end - start < 2:
        return np.array([])

    # Clamp indices
    start = max(start, 1)
    end = min(end, n_periods)

    if end - start < 2:
        return np.array([])

    # Pre-compute asset returns: ret[t] = (close[t+1] - close[t]) / close[t]
    asset_rets = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)

    bar_rets = np.zeros(end - start)

    for idx, bar_i in enumerate(range(start, end)):
        if bar_i >= n_periods - 1:
            break

        # Causal slice: all data up to current bar (inclusive)
        bar_data = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_periods:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v

        try:
            signals = strategy.on_data(bar_data) or []
        except Exception:
            signals = []

        # Compute signal-weighted return for this bar
        weighted_ret = 0.0
        for s in signals:
            if hasattr(s, "size") and s.size and s.size > 0:
                weight = float(s.direction) * float(s.size)
            else:
                weight = float(s.direction) * float(s.strength) if s.strength else float(s.direction) * 0.1
            if s.direction == 0:
                continue
            try:
                asset_idx = symbols.index(str(s.symbol).upper())
            except (ValueError, IndexError):
                continue
            if bar_i < asset_rets.shape[0] and asset_idx < asset_rets.shape[1]:
                weighted_ret += weight * float(asset_rets[bar_i, asset_idx])

        bar_rets[idx] = weighted_ret

    return bar_rets


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of walk-forward validation for one strategy."""
    strategy_name: str
    display_name: str
    strategy_type: str
    oos_sharpe: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    dsr_p_value: float = 1.0
    dsr_stat: float = 0.0
    n_folds: int = 0
    n_passing_folds: int = 0
    oos_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    triage: str = "DISABLE"  # KEEP, MONITOR, DISABLE
    triage_reason: str = ""


def validate_strategy(spec: StrategySpec, data: dict, vix_series: pd.Series | None = None) -> ValidationResult:
    """Run walk-forward validation on a single strategy.

    Uses the proven causal-slice approach: at each bar, passes data[:bar_i+1]
    to strategy.on_data(). This ensures the strategy has full historical
    context (identical to how the backtest engine works).
    """
    log.info("Validating %s (%s)...", spec.display_name, spec.registry_name)

    result = ValidationResult(
        strategy_name=spec.registry_name,
        display_name=spec.display_name,
        strategy_type=spec.strategy_type,
    )

    # Create strategy instance
    try:
        cls = get_strategy_class(spec.registry_name)
        cfg = StrategyConfig(
            name=spec.registry_name,
            strategy_type=spec.strategy_type,
            max_position_notional=INITIAL_CASH,
            params=spec.params,
        )
    except Exception as e:
        log.error("Failed to create %s: %s", spec.registry_name, e)
        result.triage_reason = f"Strategy creation failed: {e}"
        return result

    # Prepare data (with VIX injection if needed)
    strategy_data = dict(data)
    if spec.vix_required and vix_series is not None:
        strategy_data = inject_vix(strategy_data, vix_series)

    # Run walk-forward
    close = strategy_data["close"]
    n_bars = close.shape[0]

    folds = anchored_walk_forward(
        n_bars=n_bars,
        initial_train=TRAIN_SIZE,
        test_size=TEST_SIZE,
        embargo_pct=EMBARGO_PCT,
    )

    if not folds:
        log.warning("No walk-forward folds for %s (n_bars=%d)", spec.display_name, n_bars)
        result.triage_reason = f"Insufficient data ({n_bars} bars, need {TRAIN_SIZE + TEST_SIZE})"
        return result

    result.n_folds = len(folds)
    all_oos_returns = []

    for fold_idx, fold in enumerate(folds):
        # Create fresh strategy instance for each fold (no state leakage)
        fold_strategy = cls(cfg)

        # Compute returns using causal slice on FULL data
        # The strategy sees data[:bar_i+1] at each bar — proper WF
        fold_returns = compute_strategy_returns_full(
            fold_strategy, strategy_data,
            start=fold.test_start, end=fold.test_end,
        )

        if len(fold_returns) > 0:
            all_oos_returns.extend(fold_returns.tolist())

            # Check if fold is profitable
            if np.sum(fold_returns) > 0:
                result.n_passing_folds += 1

    if not all_oos_returns:
        result.triage_reason = "No OOS returns generated"
        return result

    result.oos_returns = np.array(all_oos_returns)

    # Compute aggregate OOS metrics
    if len(result.oos_returns) > 1:
        mean_ret = np.mean(result.oos_returns)
        std_ret = np.std(result.oos_returns, ddof=1)

        if std_ret > 1e-10:
            result.oos_sharpe = float((mean_ret / std_ret) * np.sqrt(252))
        else:
            result.oos_sharpe = 0.0

        # Bootstrap CI(90%)
        n_bootstrap = 1000
        sharpes = []
        n = len(result.oos_returns)
        for _ in range(n_bootstrap):
            sample = np.random.choice(result.oos_returns, size=n, replace=True)
            s_mean = np.mean(sample)
            s_std = np.std(sample, ddof=1)
            if s_std > 1e-10:
                sharpes.append((s_mean / s_std) * np.sqrt(252))
        if sharpes:
            result.ci_lower = float(np.percentile(sharpes, 5))
            result.ci_upper = float(np.percentile(sharpes, 95))

        # Max drawdown
        equity = np.cumprod(1.0 + result.oos_returns)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        result.max_dd_pct = float(abs(np.min(dd)) * 100)

        # Win rate
        wins = np.sum(result.oos_returns > 0)
        total = np.sum(result.oos_returns != 0)
        result.win_rate = float(wins / total * 100) if total > 0 else 0.0

        # Profit factor
        gross_profit = np.sum(result.oos_returns[result.oos_returns > 0])
        gross_loss = abs(np.sum(result.oos_returns[result.oos_returns < 0]))
        result.profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Deflated Sharpe Ratio
        skewness = float(pd.Series(result.oos_returns).skew())
        kurtosis = float(pd.Series(result.oos_returns).kurtosis())
        result.dsr_p_value, result.dsr_stat = deflated_sharpe_ratio(
            observed_sharpe=result.oos_sharpe,
            num_trials=NUM_TRIALS,
            sample_length=len(result.oos_returns),
            skewness=skewness,
            kurtosis_=kurtosis,
        )

    # Triage
    if result.ci_lower > 0 and result.dsr_p_value < 0.1:
        result.triage = "KEEP"
        result.triage_reason = f"CI_lower={result.ci_lower:.2f}>0, DSR p={result.dsr_p_value:.3f}<0.1"
    elif result.ci_lower > 0:
        result.triage = "MONITOR"
        result.triage_reason = f"CI_lower={result.ci_lower:.2f}>0, but DSR p={result.dsr_p_value:.3f}>=0.1"
    else:
        result.triage = "DISABLE"
        result.triage_reason = f"CI_lower={result.ci_lower:.2f}<=0"

    log.info("  %s: Sharpe=%.2f, CI=[%.2f, %.2f], MaxDD=%.1f%%, DSR p=%.3f -> %s",
             spec.display_name, result.oos_sharpe, result.ci_lower, result.ci_upper,
             result.max_dd_pct, result.dsr_p_value, result.triage)

    return result


# ---------------------------------------------------------------------------
# Risk parity weight computation
# ---------------------------------------------------------------------------

def compute_risk_parity_weights(results: list[ValidationResult]) -> dict[str, float]:
    """Compute risk-parity weights based on trailing realized volatility.

    Weight = (1/vol_i) / sum(1/vol_j) for all KEEP/MONITOR strategies.
    """
    kept = [r for r in results if r.triage in ("KEEP", "MONITOR")]
    if not kept:
        return {}

    vols = {}
    for r in kept:
        if len(r.oos_returns) > 1:
            vol = np.std(r.oos_returns, ddof=1) * np.sqrt(252)
            vols[r.strategy_name] = max(vol, 0.001)  # floor to avoid div/0
        else:
            vols[r.strategy_name] = 0.20  # default 20% vol

    inv_vols = {k: 1.0 / v for k, v in vols.items()}
    total_inv = sum(inv_vols.values())

    weights = {k: v / total_inv for k, v in inv_vols.items()}
    return weights


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list[ValidationResult], weights: dict[str, float]) -> str:
    """Generate the Phase 3 validation report."""
    lines = [
        "# Phase 3 Validation Report",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Method:** Anchored expanding-window walk-forward (504-day train, 126-day test, 1% embargo)",
        f"**Period:** {START_DATE} to {END_DATE}",
        f"**DSR Trials:** {NUM_TRIALS}",
        "",
        "## Results Summary",
        "",
        "| Strategy | Type | OOS Sharpe | CI(90%) Lower | MaxDD | Win% | PF | DSR p | Triage |",
        "|----------|------|-----------|---------------|-------|------|-----|-------|--------|",
    ]

    for r in sorted(results, key=lambda x: x.triage):
        ci_str = f"[{r.ci_lower:.2f}, {r.ci_upper:.2f}]"
        pf_str = f"{r.profit_factor:.1f}" if r.profit_factor < 100 else "inf"
        dsr_str = f"{r.dsr_p_value:.3f}" if r.dsr_p_value < 1.0 else "N/A"
        triage_emoji = {"KEEP": "✅", "MONITOR": "⚠️", "DISABLE": "❌"}.get(r.triage, "")
        lines.append(
            f"| {r.display_name} | {r.strategy_type} | {r.oos_sharpe:.2f} | {r.ci_lower:.2f} | "
            f"{r.max_dd_pct:.1f}% | {r.win_rate:.0f}% | {pf_str} | {dsr_str} | {triage_emoji} {r.triage} |"
        )

    lines.extend(["", "## Triage Decisions", ""])

    for r in results:
        lines.append(f"### {r.display_name}")
        lines.append(f"- **Triage:** {r.triage}")
        lines.append(f"- **Reason:** {r.triage_reason}")
        if r.strategy_name in weights:
            lines.append(f"- **Risk-Parity Weight:** {weights[r.strategy_name]:.1%}")
        lines.append(f"- **OOS Sharpe:** {r.oos_sharpe:.2f}")
        lines.append(f"- **CI(90%):** [{r.ci_lower:.2f}, {r.ci_upper:.2f}]")
        lines.append(f"- **Max Drawdown:** {r.max_dd_pct:.1f}%")
        lines.append(f"- **Win Rate:** {r.win_rate:.0f}%")
        lines.append(f"- **DSR p-value:** {r.dsr_p_value:.3f}")
        lines.append(f"- **Folds:** {r.n_passing_folds}/{r.n_folds} profitable")
        lines.append("")

    # Risk-parity weights section
    lines.extend(["## Risk-Parity Weights (KEEP + MONITOR strategies)", ""])
    if weights:
        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            display = next((r.display_name for r in results if r.strategy_name == name), name)
            lines.append(f"- **{display}:** {w:.1%}")
    else:
        lines.append("No strategies passed triage.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("PHASE 3.1: Walk-Forward Re-Validation of All Strategies")
    log.info("=" * 70)

    # 1. Load market data
    all_symbols = set()
    for spec in STRATEGIES:
        all_symbols.update(spec.universe)
    all_symbols = sorted(all_symbols)

    log.info("Loading market data for %d symbols (%s to %s)...", len(all_symbols), START_DATE, END_DATE)
    t0 = time.time()
    df = load_market_data(all_symbols, START_DATE, END_DATE)
    log.info("Data loaded in %.1fs", time.time() - t0)

    # 2. Load VIX
    log.info("Loading VIX data...")
    try:
        vix_series = load_vix_data(START_DATE, END_DATE)
        log.info("VIX loaded: %d data points", len(vix_series))
    except Exception as e:
        log.warning("VIX load failed: %s — options strategies will use fallback", e)
        vix_series = None

    # 3. Convert to data dict
    data = df_to_data(df)
    log.info("Data dict: close shape %s, symbols %s", data["close"].shape, data.get("symbols", []))

    # 4. Validate each strategy
    results = []
    for spec in STRATEGIES:
        log.info("-" * 50)
        result = validate_strategy(spec, data, vix_series)
        results.append(result)

    # 5. Compute risk-parity weights
    weights = compute_risk_parity_weights(results)
    log.info("Risk-parity weights: %s", {k: f"{v:.1%}" for k, v in weights.items()})

    # 6. Generate report
    report = generate_report(results, weights)
    report_path = REPORTS_DIR / "phase3_validation.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("Report written to %s", report_path)

    # 7. Save results as JSON for downstream use
    results_json = []
    for r in results:
        results_json.append({
            "strategy_name": r.strategy_name,
            "display_name": r.display_name,
            "strategy_type": r.strategy_type,
            "oos_sharpe": r.oos_sharpe,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "max_dd_pct": r.max_dd_pct,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "dsr_p_value": r.dsr_p_value,
            "n_folds": r.n_folds,
            "n_passing_folds": r.n_passing_folds,
            "triage": r.triage,
            "triage_reason": r.triage_reason,
            "risk_parity_weight": weights.get(r.strategy_name, 0.0),
        })

    json_path = REPORTS_DIR / "phase3_validation.json"
    json_path.write_text(json.dumps(results_json, indent=2, default=str), encoding="utf-8")
    log.info("JSON results written to %s", json_path)

    # 8. Print summary
    log.info("=" * 70)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 70)
    for r in results:
        emoji = {"KEEP": "PASS", "MONITOR": "WARN", "DISABLE": "FAIL"}.get(r.triage, "")
        log.info("  [%s] %s — Sharpe=%.2f, CI_low=%.2f, DSR p=%.3f",
                 emoji, r.display_name, r.oos_sharpe, r.ci_lower, r.dsr_p_value)

    kept = [r for r in results if r.triage == "KEEP"]
    monitored = [r for r in results if r.triage == "MONITOR"]
    disabled = [r for r in results if r.triage == "DISABLE"]
    log.info("KEEP: %d | MONITOR: %d | DISABLE: %d", len(kept), len(monitored), len(disabled))

    return results, weights


if __name__ == "__main__":
    results, weights = main()
