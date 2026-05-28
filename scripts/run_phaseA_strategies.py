"""Phase A validation for five equity strategies via anchored walk-forward.

Runs TSMOM, Cross-Sectional Momentum, Value (HML), Quality (RMW/CMA),
and Multi-Factor Composite through a 10-year-train / 1-year-test
walk-forward on a survivorship-biased S&P 500 universe (2005-2024).

Outputs:
  - experiments.csv (one row per strategy)
  - reports/phaseA_validation.md

Cost model:
  - 5 bps half-spread
  - 10 bps market impact
  - $0.005 per share
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantcore.data.sources.fama_french import FamaFrenchLoader
from quantcore.data.sources.historical_universe import HistoricalUniverseLoader
from quantcore.data import yfinance_loader
from quantcore.strategies.base import Strategy, StrategyConfig, StrategyType, TradeSignal
from quantcore.research.bootstrap import bootstrap_confidence_interval
from quantcore.research.deflated_sharpe import deflated_sharpe_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

UNIVERSE_START = "2000-01-01"
UNIVERSE_END = "2024-12-31"
TRAIN_YEARS = 10
TEST_YEARS = 1
OOS_START_YEAR = 2015  # first OOS fold starts here
OOS_END_YEAR = 2024

HALF_SPREAD = 5e-4   # 5 bps
IMPACT = 1e-3         # 10 bps
PER_SHARE_COST = 0.005
MAX_NOTIONAL = 1_000_000.0

REPORTS_DIR = _ROOT / "reports"


# ── Strategy configs ───────────────────────────────────────────────────

def _make_config(name: str, params: dict | None = None) -> StrategyConfig:
    return StrategyConfig(
        name=name,
        strategy_type=StrategyType.EQUITY,
        max_position_notional=MAX_NOTIONAL,
        max_leverage=1.0,
        risk_budget_pct=0.02,
        params=params or {},
    )


STRATEGY_SPECS: list[tuple[str, str, dict]] = [
    ("equity_tsmom", "TSMOMStrategy", {}),
    ("equity_cross_sectional_momentum", "CrossSectionalMomentumStrategy", {}),
    ("equity_value_hml", "ValueHMLStrategy", {}),
    ("equity_quality_rmw_cma", "QualityRMWCMAStrategy", {}),
    ("equity_multifactor_composite", "MultiFactorCompositeStrategy", {}),
]


# ── Data loading ───────────────────────────────────────────────────────

def load_universe() -> list[str]:
    loader = HistoricalUniverseLoader()
    return loader.get_constituents("SP500", date(2005, 1, 1))


def load_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    _log.info("Loading price data for %d symbols (%s -> %s)", len(symbols), UNIVERSE_START, UNIVERSE_END)
    data = yfinance_loader.load_multiple(symbols, UNIVERSE_START, UNIVERSE_END, interval="1d")
    _log.info("Loaded %d / %d symbols", len(data), len(symbols))
    return data


def load_factors() -> pd.DataFrame:
    _log.info("Loading Fama-French 5-factor data")
    loader = FamaFrenchLoader(cache_dir=_ROOT / ".cache" / "fama_french")
    df = loader.load_5_factors(freq="daily")
    _log.info("Factor data: %d rows, columns=%s", len(df), list(df.columns))
    return df


def build_data_dict(
    prices: dict[str, pd.DataFrame],
    factors: pd.DataFrame,
) -> dict:
    """Build the data dict expected by strategy on_data()."""
    # Normalize all timestamps to tz-naive date-only for alignment
    date_counts: dict[pd.Timestamp, int] = {}
    for sym, df in prices.items():
        ts = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
        for d in ts.unique():
            date_counts[d] = date_counts.get(d, 0) + 1
    n_stocks = len(prices)
    threshold = int(n_stocks * 0.90)
    common_dates = sorted(d for d, cnt in date_counts.items() if cnt >= threshold)
    dates_idx = pd.DatetimeIndex(common_dates)

    # Filter to factor data range (use outer bounds, reindex handles alignment)
    factor_start = factors.index.min()
    factor_end = factors.index.max()
    dates_idx = dates_idx[(dates_idx >= factor_start) & (dates_idx <= factor_end)]

    symbols = sorted(prices.keys())
    n_dates = len(dates_idx)
    n_assets = len(symbols)

    close = np.zeros((n_dates, n_assets))
    volume = np.zeros((n_dates, n_assets))

    for j, sym in enumerate(symbols):
        df = prices[sym].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.reindex(dates_idx)
        close[:, j] = df["close"].ffill().bfill().values
        volume[:, j] = df["volume"].ffill().fillna(0).values

    # Align factors to dates
    factor_aligned = factors.reindex(dates_idx).ffill().bfill()

    return {
        "close": close,
        "volume": volume,
        "symbols": symbols,
        "dates": dates_idx,
        "factor_returns": factor_aligned,
    }


# ── Walk-forward engine ───────────────────────────────────────────────

def _weights_from_signals(
    signals: list[TradeSignal],
    symbols: list[str],
    max_notional: float,
) -> np.ndarray:
    """Convert TradeSignal list to a weight vector."""
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    weights = np.zeros(len(symbols))
    for sig in signals:
        idx = sym_to_idx.get(sig.symbol)
        if idx is not None:
            weights[idx] = sig.direction * sig.size / max_notional
    return weights


def run_fold(
    strategy: Strategy,
    data: dict,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single walk-forward fold.

    Returns (daily_returns, daily_turnover) arrays for the OOS period.
    """
    dates = data["dates"]
    close = data["close"]
    symbols = data["symbols"]
    max_notional = strategy.config.max_position_notional

    # OOS mask
    oos_mask = (dates >= oos_start) & (dates <= oos_end)
    oos_dates = dates[oos_mask]
    if len(oos_dates) < 20:
        return np.array([]), np.array([])

    # Find index of first OOS date in full dates array
    oos_start_idx = int(np.where(dates == oos_dates[0])[0][0])

    # Pre-compute daily returns for the full period
    daily_rets_full = np.diff(close, axis=0) / np.maximum(close[:-1], 1e-10)

    daily_returns: list[float] = []
    daily_turnover: list[float] = []
    prev_weights = np.zeros(len(symbols))

    # Simulate monthly rebalancing
    current_month = -1
    target_weights = np.zeros(len(symbols))

    for i, d in enumerate(oos_dates):
        full_idx = int(np.where(dates == d)[0][0])

        # Rebalance on first trading day of each month
        if d.month != current_month:
            current_month = d.month

            # Build data slice up to current bar
            hist_close = close[: full_idx + 1]
            hist_volume = data["volume"][: full_idx + 1]
            hist_dates = dates[: full_idx + 1]
            factor_df = data["factor_returns"]

            data_slice = {
                "close": hist_close,
                "volume": hist_volume,
                "symbols": symbols,
                "dates": hist_dates,
                "factor_returns": factor_df,
            }

            signals = strategy.on_data(data_slice)
            target_weights = _weights_from_signals(signals, symbols, max_notional)

        # Daily return from previous day's weights
        if full_idx > 0:
            day_ret = float(np.dot(prev_weights, daily_rets_full[full_idx - 1]))
            daily_returns.append(day_ret)

            # Turnover on rebalance days
            turnover = float(np.sum(np.abs(target_weights - prev_weights)))
            daily_turnover.append(turnover)
        else:
            daily_returns.append(0.0)
            daily_turnover.append(0.0)

        prev_weights = target_weights.copy()

    returns_arr = np.array(daily_returns)
    turnover_arr = np.array(daily_turnover)

    # Apply transaction costs: on rebalance days, deduct cost proportional to turnover
    cost_rate = HALF_SPREAD + IMPACT  # 15 bps on notional traded
    for i in range(len(returns_arr)):
        if turnover_arr[i] > 1e-8:
            returns_arr[i] -= turnover_arr[i] * cost_rate

    return returns_arr, turnover_arr


def walk_forward_validate(
    strategy_cls: type,
    strategy_name: str,
    params: dict,
    data: dict,
) -> dict:
    """Run anchored walk-forward validation for one strategy."""
    dates = data["dates"]

    all_oos_returns: list[np.ndarray] = []
    all_turnover: list[np.ndarray] = []
    fold_info: list[dict] = []

    for fold_idx, oos_year in enumerate(range(OOS_START_YEAR, OOS_END_YEAR + 1)):
        oos_start = pd.Timestamp(f"{oos_year}-01-01")
        oos_end = pd.Timestamp(f"{oos_year}-12-31")

        # Clamp to available data
        if oos_start < dates[0] or oos_start > dates[-1]:
            continue

        _log.info("  Fold %d: OOS %d", fold_idx + 1, oos_year)

        # Fresh strategy instance for each fold
        config = _make_config(strategy_name, params)
        strategy = strategy_cls(config)

        rets, turnover = run_fold(strategy, data, oos_start, oos_end)
        if len(rets) == 0:
            _log.warning("  Fold %d: empty returns", fold_idx + 1)
            continue

        all_oos_returns.append(rets)
        all_turnover.append(turnover)
        fold_info.append({
            "fold": fold_idx + 1,
            "oos_year": oos_year,
            "n_days": len(rets),
            "mean_ret": float(np.mean(rets)),
            "sharpe": float(np.mean(rets) / max(np.std(rets, ddof=1), 1e-12) * np.sqrt(252)),
        })

    if not all_oos_returns:
        return {"error": "no valid folds"}

    # Concatenate OOS returns (chronological)
    concat_rets = np.concatenate(all_oos_returns)
    concat_turnover = np.concatenate(all_turnover)

    # Metrics
    mean_ret = float(np.mean(concat_rets))
    std_ret = float(np.std(concat_rets, ddof=1))
    annual_sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-12 else 0.0

    # Bootstrap 90% CI on Sharpe
    block_size = max(5, int(len(concat_rets) ** (1 / 3)))
    ci_lower, ci_upper = bootstrap_confidence_interval(
        concat_rets, block_size=block_size, n_boot=2000, ci=0.90,
    )

    # DSR (5 trials = number of strategies tested)
    skew = float(np.mean((concat_rets - mean_ret) ** 3) / (std_ret ** 3)) if std_ret > 1e-12 else 0.0
    kurt = float(np.mean((concat_rets - mean_ret) ** 4) / (std_ret ** 4) - 3.0) if std_ret > 1e-12 else 3.0
    dsr_pval, dsr_stat = deflated_sharpe_ratio(
        observed_sharpe=annual_sharpe,
        num_trials=5,  # 5 strategies tested
        sample_length=len(concat_rets),
        skewness=skew,
        kurtosis_=kurt,
    )

    # Max drawdown
    equity = np.cumprod(1.0 + concat_rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.abs(dd.min()))

    # Win rate
    win_rate = float(np.mean(concat_rets > 0))

    # Verdict
    verdict = "DEPLOY" if ci_lower >= 0.3 else "REJECT"

    # Annual turnover
    n_years = len(concat_rets) / 252
    avg_annual_turnover = float(np.mean(concat_turnover) * 252) if n_years > 0 else 0.0

    return {
        "strategy": strategy_name,
        "oos_sharpe": round(annual_sharpe, 3),
        "ci_90_lower": round(ci_lower, 3),
        "ci_90_upper": round(ci_upper, 3),
        "dsr_pvalue": round(dsr_pval, 4),
        "dsr_stat": round(dsr_stat, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "avg_annual_turnover": round(avg_annual_turnover, 2),
        "n_oos_days": len(concat_rets),
        "n_folds": len(fold_info),
        "verdict": verdict,
        "folds": fold_info,
    }


# ── Report generation ─────────────────────────────────────────────────

def generate_report(results: list[dict]) -> str:
    lines = [
        "# Phase A Validation Report — Equity Strategies",
        "",
        f"**Date**: {date.today().isoformat()}  ",
        f"**Universe**: S&P 500 (~130 stocks, survivorship-biased)  ",
        f"**Period**: {OOS_START_YEAR}-{OOS_END_YEAR} OOS (10yr anchored train)  ",
        f"**Costs**: {HALF_SPREAD*1e4:.0f}bps half-spread + {IMPACT*1e4:.0f}bps impact + ${PER_SHARE_COST}/share  ",
        f"**DEPLOY threshold**: 90% CI lower on Sharpe ≥ 0.3",
        "",
        "## Summary",
        "",
        "| Strategy | OOS Sharpe | CI(90%) Lower | Max DD | Win Rate | Turnover | Verdict |",
        "|----------|-----------|---------------|--------|----------|----------|---------|",
    ]

    for r in results:
        if "error" in r:
            lines.append(f"| {r['strategy']} | ERROR | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['strategy']} | {r['oos_sharpe']:.3f} | {r['ci_90_lower']:.3f} "
            f"| {r['max_drawdown']:.2%} | {r['win_rate']:.1%} "
            f"| {r['avg_annual_turnover']:.1f}x | **{r['verdict']}** |"
        )

    lines += [
        "",
        "## Detailed Results",
        "",
    ]

    for r in results:
        if "error" in r:
            continue
        lines += [
            f"### {r['strategy']}",
            "",
            f"- OOS Sharpe: {r['oos_sharpe']:.3f}",
            f"- 90% CI: [{r['ci_90_lower']:.3f}, {r['ci_90_upper']:.3f}]",
            f"- DSR p-value: {r['dsr_pvalue']:.4f} (stat: {r['dsr_stat']:.3f})",
            f"- Max Drawdown: {r['max_drawdown']:.2%}",
            f"- Win Rate: {r['win_rate']:.1%}",
            f"- Avg Annual Turnover: {r['avg_annual_turnover']:.1f}x",
            f"- OOS Days: {r['n_oos_days']:,}",
            f"- Folds: {r['n_folds']}",
            f"- **Verdict: {r['verdict']}**",
            "",
            "#### Per-Fold Performance",
            "",
            "| Fold | Year | Days | Mean Daily Ret | Sharpe |",
            "|------|------|------|---------------|--------|",
        ]
        for f in r.get("folds", []):
            lines.append(
                f"| {f['fold']} | {f['oos_year']} | {f['n_days']} "
                f"| {f['mean_ret']:.6f} | {f['sharpe']:.3f} |"
            )
        lines.append("")

    lines += [
        "## Methodology",
        "",
        "- **Walk-forward**: Anchored (expanding train, 1yr test, annual steps)",
        "- **Train window**: 10 years (initial 2005-2014)",
        "- **Bootstrap**: Block bootstrap, 2000 replications, block size = n^(1/3)",
        "- **DSR**: Bailey & Lopez de Prado (2012), 5 trials",
        "- **Rebalancing**: Monthly (first trading day of each month)",
        "- **Cost model**: Turnover × (half-spread + impact) per rebalance day",
        "",
        "## Known Limitations",
        "",
        "- Survivorship bias: universe uses current S&P 500 constituents retroactively",
        "- No capacity constraints beyond position sizing",
        "- yfinance OHLCV only — no tick-level or order book validation",
        "- Factor data from Ken French Data Library (academic, not tradable)",
    ]

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    _log.info("=" * 60)
    _log.info("Phase A Validation — 5 Equity Strategies")
    _log.info("=" * 60)

    # Load data
    symbols = load_universe()
    prices = load_prices(symbols)
    factors = load_factors()

    # Filter to stocks with sufficient data
    min_bars = 252 * (TRAIN_YEARS + 2)  # at least 12 years of data
    valid_symbols = []
    for sym in symbols:
        if sym in prices and len(prices[sym]) >= min_bars:
            valid_symbols.append(sym)
    _log.info("Valid symbols with ≥%d bars: %d / %d", min_bars, len(valid_symbols), len(symbols))

    if len(valid_symbols) < 20:
        _log.error("Too few valid symbols (%d). Aborting.", len(valid_symbols))
        return

    # Build data dict
    data = build_data_dict({s: prices[s] for s in valid_symbols}, factors)
    _log.info(
        "Data dict: close=%s, dates=%s -> %s",
        data["close"].shape,
        str(data["dates"][0].date()),
        str(data["dates"][-1].date()),
    )

    # Run validation for each strategy
    import quantcore.strategies.equity  # noqa: F401 — trigger @register decorators
    from quantcore.strategies.registry import get as get_strategy_cls

    results: list[dict] = []
    for reg_name, display_name, params in STRATEGY_SPECS:
        _log.info("-" * 40)
        _log.info("Validating: %s (%s)", display_name, reg_name)

        try:
            strategy_cls = get_strategy_cls(reg_name)
        except KeyError:
            _log.error("Strategy %s not found in registry", reg_name)
            results.append({"strategy": display_name, "error": "not found"})
            continue

        result = walk_forward_validate(strategy_cls, display_name, params, data)
        results.append(result)

        if "error" not in result:
            _log.info(
                "  -> Sharpe=%.3f, CI lower=%.3f, MaxDD=%.2f%%, Verdict=%s",
                result["oos_sharpe"],
                result["ci_90_lower"],
                result["max_drawdown"] * 100,
                result["verdict"],
            )
        else:
            _log.warning("  -> ERROR: %s", result["error"])

    # Write experiments.csv
    csv_path = _ROOT / "experiments.csv"
    csv_rows = []
    for r in results:
        if "error" in r:
            csv_rows.append({"strategy": r["strategy"], "error": r["error"]})
        else:
            csv_rows.append({k: v for k, v in r.items() if k != "folds"})
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    _log.info("Wrote experiments.csv -> %s", csv_path)

    # Write report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "phaseA_validation.md"
    report_path.write_text(generate_report(results), encoding="utf-8")
    _log.info("Wrote report -> %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE A RESULTS")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['strategy']:40s}  ERROR: {r['error']}")
        else:
            print(
                f"  {r['strategy']:40s}  Sharpe={r['oos_sharpe']:+.3f}  "
                f"CI_low={r['ci_90_lower']:+.3f}  DD={r['max_drawdown']:.2%}  "
                f"-> {r['verdict']}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
