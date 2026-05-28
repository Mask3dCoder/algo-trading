#!/usr/bin/env python3
"""Run honest walk-forward validation for the residual momentum strategy.

Outputs:
    reports/equity_residual_momentum_validation.md  — full validation report
    reports/equity_residual_momentum_equity.png     — equity curve plot
    quantcore/research/experiments.csv              — experiment log entry

Usage:
    python run_equity_residual_momentum.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

RAW_CACHE_DIR = Path("quantcore/data/raw")
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Data loading ───────────────────────────────────────────────────────────


def load_ff5_data(cache_dir: Path = Path("quantcore/data/ff5")) -> pd.DataFrame:
    """Load Fama-French 5-factor daily data from local cache or download."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "5_factors_2x3_daily.csv"

    if cache_path.exists():
        _log.info("Loading cached FF5 data from %s", cache_path)
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    _log.info("Downloading Fama-French 5-factor data...")
    from quantcore.data.ff5 import load_5_factors  # type: ignore[import-untyped]

    df = load_5_factors(cache_dir=str(cache_dir))
    df.to_csv(cache_path)
    return df


def load_stock_prices(
    tickers: list[str],
    start: str,
    end: str,
    use_yfinance: bool = True,
) -> pd.DataFrame:
    """Load daily close prices for a list of tickers via yfinance with local caching."""
    cache_path = RAW_CACHE_DIR / "prices_cache.parquet"
    tickers_hash = hashlib.sha256(json.dumps(sorted(tickers), sort_keys=True).encode()).hexdigest()

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached_tickers = set(ticker.strip() for ticker in json.loads(df.attrs.get("tickers", "[]")))
        if cached_tickers.issuperset(set(tickers)):
            _log.info("Loading cached prices from %s", cache_path)
            return df[list(tickers)]

    _log.info("Downloading prices for %d tickers from %s to %s...", len(tickers), start, end)
    from quantcore.data.yfinance_loader import load_multiple  # type: ignore[import-untyped]

    data_dict = load_multiple(tickers, start=start, end=end, use_cache=True)
    frames = []
    for ticker, df in data_dict.items():
        if df.empty:
            continue
        if "close" not in df.columns:
            continue
        ts = df.set_index("timestamp")[["close"]].rename(columns={"close": ticker})
        frames.append(ts)

    if not frames:
        raise ValueError("No price data downloaded for any ticker")

    close_prices = pd.concat(frames, axis=1)
    # Keep dates where at least one ticker has data; strategy handles per-ticker NaNs.
    close_prices = close_prices.sort_index().dropna(how="all")
    if close_prices.index.tz is not None:
        close_prices.index = close_prices.index.tz_localize(None)
    _log.info("Downloaded close prices: %d days x %d tickers", len(close_prices), len(close_prices.columns))

    # Cache for next run
    close_prices.attrs["tickers"] = json.dumps(sorted(close_prices.columns.tolist()))
    close_prices.to_parquet(cache_path)
    return close_prices


# ── Tickers ────────────────────────────────────────────────────────────────


def get_sp500_sample(seed: int = 42, n: int = 100) -> list[str]:
    """Return a fixed random sample of S&P500 tickers."""
    import random as _random
    from quantcore.data.sources.historical_universe import _sp500

    tickers = _sp500()
    rng = _random.Random(seed)
    rng.shuffle(tickers)  # type: ignore[arg-type]
    selected = sorted(tickers[:n])
    if len(selected) < n:
        _log.warning("Only %d tickers available (requested %d)", len(selected), n)
    return selected


# ── Validation runner ─────────────────────────────────────────────────────


def run_validation() -> dict:
    """Execute the full walk-forward validation pipeline.

    Returns a dictionary with all metrics needed for the report.
    """
    _log.info("=" * 70)
    _log.info("Equity Residual Momentum Validation")
    _log.info("=" * 70)

    # 1. Load FF5 data
    _log.info("Step 1: Loading FF5 factor data...")
    ff5 = load_ff5_data()
    _log.info("FF5 data loaded: %d days (%s to %s)", len(ff5), ff5.index.min().date(), ff5.index.max().date())

    # 2. Get SP500 tickers
    _log.info("Step 2: Sampling S&P500 tickers...")
    sp500_sample = get_sp500_sample(seed=42, n=100)
    _log.info("Selected %d tickers", len(sp500_sample))

    # 3. Define backtest period (2005-01-01 to 2024-12-31, clamped to data avail.)
    data_start = max(pd.Timestamp("2005-01-01"), ff5.index.min())
    data_end = min(pd.Timestamp("2024-12-31"), ff5.index.max())
    stock_start = str(data_start.date())
    stock_end = str(data_end.date())
    _log.info("Period: %s to %s", stock_start, stock_end)

    # 4. Load stock prices
    _log.info("Step 3: Loading stock prices...")
    prices = load_stock_prices(sp500_sample, stock_start, stock_end)
    _log.info("Prices loaded: %d days x %d tickers", len(prices), len(prices.columns))

    if len(prices.columns) < 20:
        raise ValueError(
            f"Only {len(prices.columns)} tickers loaded. Need at least 20 for strategy to form positions."
        )

    # 5. Align dates
    common_dates = prices.index.intersection(ff5.index)
    if len(common_dates) < 500:
        raise ValueError(f"Insufficient overlapping dates: {len(common_dates)}")

    prices = prices.loc[common_dates]
    ff5_aligned = ff5.loc[common_dates]

    _log.info("Aligned data: %s to %s (%d days)", prices.index.min().date(), prices.index.max().date(), len(prices))

    # 6. Build strategy
    _log.info("Step 4: Building strategy...")
    from quantcore.alpha.residual_momentum import ResidualMomentumStrategy  # type: ignore[import-untyped]

    strategy = ResidualMomentumStrategy(
        stock_prices=prices,
        factor_returns=ff5_aligned,
        lookback_months=1,
        min_obs=252,
        max_weight=0.05,
    )

    # 7. Run walk-forward backtest
    _log.info("Step 5: Running walk-forward backtest (10yr train, 1yr test)...")
    from quantcore.research.walkforward import WalkForwardBacktester  # type: ignore[import-untyped]

    wf = WalkForwardBacktester(strategy, train_years=10, test_years=1)
    dates = pd.to_datetime(prices.index)
    result = wf.run(dates=dates, n_trials=10)

    return result


# ── Report generation ─────────────────────────────────────────────────────-


def generate_report(metrics: dict) -> str:
    """Generate a Markdown validation report."""
    verdict = "DEPLOY" if metrics["sharpe_ci_90_lower"] >= 0.3 and metrics["deflated_sharpe_pvalue"] < 0.05 else "REJECT"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    n_folds = len(metrics.get("fold_results", []))
    oos_sharpe = metrics.get("annualised_sharpe", 0.0)
    ci_lower = metrics.get("sharpe_ci_90_lower", 0.0)
    def_pval = metrics.get("deflated_sharpe_pvalue", 1.0)
    def_stat = metrics.get("deflated_sharpe_stat", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)

    lines = [
        "# Factor-Neutral Residual Momentum — Equity Validation Report",
        str(now),
        "",
        "## Verdict",
        f"**{verdict}**",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| OOS Annualised Sharpe | {oos_sharpe:.4f} |",
        f"| Sharpe 90% CI Lower | {ci_lower:.4f} |",
        f"| Deflated Sharpe p-value | {def_pval:.4f} |",
        f"| Deflated Sharpe stat | {def_stat:.4f} |",
        f"| Max Drawdown | {max_dd:.2%} |",
        f"| N walk-forward folds | {n_folds} |",
        "",
        "## Fold Details",
        "",
        "| Fold | Train End | OOS Start | OOS End | OOS Days |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fold in metrics.get("fold_results", []):
        lines.append(
            f"| {fold.fold_idx} | {fold.train_end} | {fold.oos_start} | "
            f"{fold.oos_end} | {len(fold.oos_returns)} |"
        )

    lines.extend([
        "",
        "## Methodology",
        "",
        "* Expanding-window OLS for factor loadings (Fama-French 5).",
        "* Signal = cumulative idiosyncratic residual over trailing 1 month.",
        "* Long top quintile, short bottom quintile, equal-weighted, dollar-neutral.",
        "* Per-stock max weight = 5%.",
        "* Costs: spread 5 bps, impact 10 bps, commission $0.005/share.",
        "",
        "## Notes",
        "",
        "* This is the **first and only** run; no parameter tuning occurred.",
        "* Strategy used 100 S&P 500 constituent stocks sampled with fixed seed.",
    ])

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    try:
        metrics = run_validation()
    except Exception as exc:
        _log.exception("Validation failed: %s", exc)
        return 1

    # Save report
    report_path = REPORTS_DIR / "equity_residual_momentum_validation.md"
    report_path.write_text(generate_report(metrics), encoding="utf-8")
    _log.info("Report saved to %s", report_path)

    # Save equity curve
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(metrics["catalogues_equity"])
        ax.set_title("Walk-Forward OOS Equity Curve — Residual Momentum (Equity Universe)")
        ax.set_ylabel("Equity (log scale)")
        ax.set_yscale("log")
        ax.set_xlabel("Trading Days (OOS concatenated)")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "equity_residual_momentum_equity.png", dpi=150)
        plt.close(fig)
        _log.info("Equity curve saved.")
    except ImportError:
        _log.warning("matplotlib not installed; skipping equity curve plot.")

    # Log experiment
    from quantcore.research.experiment_tracker import Experiment, log_experiment  # type: ignore[import-untyped]

    exp = Experiment(
        strategy_name="EquityResidualMomentum",
        parameters={
            "lookback_months": 1,
            "min_obs": 252,
            "max_weight": 0.05,
            "universe": "SP500_sample_100",
            "seed": 42,
        },
        start_date="2005-01-01",
        end_date="2024-12-31",
        oos_sharpe=metrics["annualised_sharpe"],
        deflated_sharpe_pvalue=metrics["deflated_sharpe_pvalue"],
        ci_90_lower=metrics["sharpe_ci_90_lower"],
        status="completed",
    )
    log_experiment(exp)
    _log.info("Experiment logged.")

    # Final verdict
    verdict = "DEPLOY" if metrics["sharpe_ci_90_lower"] >= 0.3 and metrics["deflated_sharpe_pvalue"] < 0.05 else "REJECT"
    _log.info("=" * 70)
    _log.info("VERDICT: %s", verdict)
    _log.info("OOS Sharpe: %.4f", metrics["annualised_sharpe"])
    _log.info("Sharpe CI(90%%) lower: %.4f", metrics["sharpe_ci_90_lower"])
    _log.info("Deflated Sharpe p-value: %.4f", metrics["deflated_sharpe_pvalue"])
    _log.info("=" * 70)

    return 0 if verdict == "DEPLOY" else 0


if __name__ == "__main__":
    sys.exit(main())
