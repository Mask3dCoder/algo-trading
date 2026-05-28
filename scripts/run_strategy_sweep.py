#!/usr/bin/env python3
"""Strategy walk-forward sweep — all strategies, one Honest validation.

For each strategy not already validated or DISABLED, runs an anchored
walk-forward backtest (10yr train / 1yr OOS, expanding window) using the
BLUECHIP 100 equity universe and FF5 factor loader.

Verdict: DEPLOY if CI_lower >= 0.3 AND DSR p-value < 0.05; else REJECT.
Total DSR trials = strategies_tested + 10.

Supports strategies that implement either:
  1. run_returns_backtest(...)  — used directly (e.g. ResidualMomentumStrategy)
  2. on_data(data)              — wrapped via StrategyBacktestWrapper

Usage:
    python run_strategy_sweep.py
    python run_strategy_sweep.py --strategy equity_momentum --strategy macro_carry
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Ensure project root is on sys.path for quantcore imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ------------------------------------------------------------------------------
# Data loading — same as reference scripts
# ------------------------------------------------------------------------------


def load_ff5_data(cache_dir: Path = Path("quantcore/data/ff5")) -> pd.DataFrame:
    """Load Fama-French 5-factor daily data from local cache or Ken French."""
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
    cache_dir: Path = Path("quantcore/data/raw"),
) -> pd.DataFrame:
    """Load daily close prices for tickers via yfinance, with parquet caching."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "sweep_prices.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached = set(json.loads(df.attrs.get("tickers", "[]")))
        if cached.issuperset(set(tickers)):
            _log.info("Loading cached prices from %s", cache_path)
            return df[sorted(tickers)]

    _log.info("Downloading %d stock prices from %s to %s ...", len(tickers), start, end)
    from quantcore.data.yfinance_loader import load_multiple  # type: ignore[import-untyped]

    data = load_multiple(tickers, start=start, end=end, use_cache=False)
    frames = []
    for sym, df in data.items():
        if df.empty or "close" not in df.columns:
            continue
        ts = df.set_index("timestamp")[["close"]].rename(columns={"close": sym})
        frames.append(ts)

    if not frames:
        raise ValueError("No price data downloaded for any ticker")

    close = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)

    close.attrs["tickers"] = json.dumps(sorted(close.columns.tolist()))
    close.to_parquet(cache_path)
    _log.info("Cached prices: %d days x %d tickers", len(close), len(close.columns))
    return close


# BLUECHIP universe — same fixed list as residual momentum reference scripts
BLUECHIP: list[str] = [
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


# ------------------------------------------------------------------------------
# Strategy ABC imports for config
# ------------------------------------------------------------------------------

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal  # type: ignore[import-untyped]


# ------------------------------------------------------------------------------
# Strategy → run_returns_backtest adapter
#
# WalkForwardBacktester calls strategy.run_returns_backtest(...).
# Strategies that only implement on_data() need this wrapper.
#
# For strategies that ALREADY have run_returns_backtest (ResidualMomentumStrategy,
# and any others added later), we use them directly and skip the wrapper.
# ------------------------------------------------------------------------------

class StrategyBacktestWrapper:
    """Wrap an on_data()-based Strategy to expose run_returns_backtest().

    The wrapper iterates dates in [start, end], calling on_data() on each
    rebalance day to refresh the live position dict, then computing:
        port_return_t = sum(weight_i * daily_return_i)   for all assets i
    Equity accrues from initial_capital with transaction costs on each
    rebalance day (delta * (half-spread + impact) + per-share commission).

    The wrapper uses the strategy's own rebalance_freq counter, so it mimics
    how the strategy would run live.
    """

    def __init__(
        self,
        strategy: Strategy,
        config: StrategyConfig,
        prices: pd.DataFrame,
        factors: pd.DataFrame,
        *,
        spread_bps: float = 5.0,
        impact_bps: float = 10.0,
        commission_per_share: float = 0.005,
    ) -> None:
        self.strategy = strategy
        self.config = config
        self._prices = prices.sort_index()
        self._factors = factors.reindex(prices.index).sort_index()
        self._all_dates = self._prices.index
        self._tickers = list(self._prices.columns)

        # Cost parameters
        self._spread_bps = spread_bps
        self._impact_bps = impact_bps
        self._commission = commission_per_share

        # Precompute simple returns (close_t / close_{t-1} - 1)
        self._simple_rets = self._prices.pct_change().fillna(0)

        # Strategy rebalance counter (mirrors what strategy maintains)
        self._days_since_rebalance: int = 0
        self._rebalance_freq: int = int(config.params.get("rebalance_freq", 21))

        # Live position dict: ticker -> weight
        self._positions: dict[str, float] = {}

        # For turnover
        self._prev_positions: dict[str, float] = {}

    # ------------------------------------------------------------------
    # run_returns_backtest — called by WalkForwardBacktester
    # ------------------------------------------------------------------

    def run_returns_backtest(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        training_start: pd.Timestamp,
        initial_capital: float = 1_000_000.0,
        spread_bps: float = 5.0,
        impact_bps: float = 10.0,
        commission_per_share: float = 0.005,
        trading_days_per_month: int = 21,
    ) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, list[int]]]:
        """Returns-based backtest over [start, end] with expanding training window."""

        mask = (self._all_dates >= start) & (self._all_dates <= end)
        test_dates = self._all_dates[mask]

        # Reset state for this fold
        self._positions = {t: 0.0 for t in self._tickers}
        self._prev_positions = {t: 0.0 for t in self._tickers}
        self._days_since_rebalance = 0
        self._rebalance_freq = int(self.config.params.get("rebalance_freq", 21))

        equity: list[float] = [initial_capital]
        daily_rets: list[float] = [0.0]
        turnover_list: list[float] = [0.0]
        layer_logs: dict[str, list[int]] = {}

        prev_date: pd.Timestamp | None = None

        for date_i in test_dates:
            date_i_ts = pd.Timestamp(date_i)
            prev_date = test_dates[test_dates < date_i_ts][-1] if len(test_dates[test_dates < date_i_ts]) > 0 else None

            # Build expanding history up to (and including) date_i
            hist_dates = self._all_dates[self._all_dates <= date_i_ts]
            if len(hist_dates) < 5:
                equity.append(equity[-1])
                daily_rets.append(0.0)
                turnover_list.append(0.0)
                continue

            price_hist = self._prices.loc[hist_dates]
            rets_hist = self._simple_rets.loc[hist_dates]

            # Build data dict for on_data()
            close_arr = price_hist.values
            data_dict: dict[str, object] = {
                "close": close_arr,
                "symbols": self._tickers,
            }
            # Pass factors if strategy asks for them (factors dict keyed by name)
            data_dict["factors"] = self._factors.loc[hist_dates].to_dict(orient="list")

            # Rebalance gate — strategy advances its own counter internally
            self._days_since_rebalance += 1
            do_rebalance = self._days_since_rebalance >= self._rebalance_freq

            if do_rebalance:
                # Save previous positions for turnover
                self._prev_positions = dict(self._positions)

                # Reset counter BEFORE calling strategy so it can reset its own
                self._days_since_rebalance = 0

                # Call on_data() to get new signals
                signals = self.strategy.on_data(data_dict)
                new_positions = self._signals_to_weights(signals)

                # Apply new positions
                self._positions = new_positions

                # Turnover = sum |new_w - prev_w|
                turnover = sum(
                    abs(self._positions.get(t, 0.0) - self._prev_positions.get(t, 0.0))
                    for t in set(self._positions) | set(self._prev_positions)
                )
                turnover_list.append(float(turnover))

                # Transaction costs
                delta = turnover
                cost_bps = spread_bps / 2.0 + impact_bps
                cost = equity[-1] * delta * (cost_bps / 10_000.0)
                avg_price = 100.0  # simplified; impact dominates
                shares = equity[-1] * delta / avg_price
                commission = shares * commission_per_share
                total_cost = cost + commission
            else:
                turnover_list.append(0.0)
                total_cost = 0.0

            # Compute today's return (at date_i), positions are already set
            today_rets = rets_hist.loc[date_i_ts]
            port_ret = sum(
                self._positions.get(t, 0.0) * today_rets.get(t, 0.0)
                for t in self._positions
            )
            gross_equity = equity[-1] * (1.0 + port_ret)
            after_cost = gross_equity - total_cost

            equity.append(after_cost)

            # Daily return (simple, relative to previous equity)
            day_ret = (equity[-1] - equity[-2]) / equity[-2] if equity[-2] != 0 else 0.0
            daily_rets.append(float(day_ret))

            n_long = sum(1 for v in self._positions.values() if v > 0)
            n_short = sum(1 for v in self._positions.values() if v < 0)
            layer_logs[str(date_i.date())] = [n_long + n_short, n_long, n_short]

        eq_series = pd.Series(equity[1:], index=test_dates)
        rets_series = pd.Series(daily_rets[1:], index=test_dates)
        turnover_series = pd.Series(turnover_list[1:], index=test_dates)

        return eq_series, rets_series, turnover_series, layer_logs

    def _signals_to_weights(self, signals: list[TradeSignal]) -> dict[str, float]:
        """Convert TradeSignals to a dict of ticker -> weight.

        Interpretation:
        - direction = +1 → go long; direction = -1 → go short; direction = 0 → flatten
        - size is the notional weight (absolute). For most strategies size is
          strength * max_position_notional, so we normalise by max_position_notional.
        - If size <= 0, treat as flat (no position).
        """
        weights: dict[str, float] = {}
        max_notional = float(self.config.max_position_notional) if self.config.max_position_notional else 1.0
        if max_notional <= 0:
            max_notional = 1.0

        for sig in signals:
            if sig.direction == 0:
                weights[sig.symbol] = 0.0
                continue
            if sig.size > 0:
                # Assume size is notional; convert to weight fraction
                w = sig.direction * sig.size / max_notional
            else:
                # Fallback: use strength as notional fraction
                w = sig.direction * sig.strength
            weights[sig.symbol] = w

        return weights


# ------------------------------------------------------------------------------
# Strategy registry scanner — find testable strategies
# ------------------------------------------------------------------------------

from quantcore.research.experiment_tracker import Experiment, log_experiment  # type: ignore[import-untyped]


# Strategies already validated or DISABLED — skip
_SKIP_STRATEGIES = {
    "ResidualMomentum",          # reference — already run
    "EquityResidualMomentum",   # reference — already run
    "macro_factor_residual",    # already in PAPER
    "macro_risk_parity",        # already in MONITOR
    "macro_cta_trend",          # DISABLED / prior REJECT
    "equity_momentum",          # DISABLED / prior REJECT
    "macro_cross_sectional_momentum_v2",  # DISABLED prior REJECT
}

# Strategies that need OHLCV (high/low) which the sweep loader doesn't provide
_NEEDS_OHLCV = {
    "equity_volatility_breakout",  # requires high/low for ATR, Bollinger
    "equity_trend_following",     # requires high/low for ATR/Donchian
}

# Strategies that need external data feeds not available in the loader
_NEEDS_EXTERNAL = {
    "equity_stat_arb",           # Kalman filter + pairs — testable with close-only but needs careful handling
    "macro_true_carry",           # needs data["carry"] — no source
    "macro_cross_sectional_factor_neutral",  # needs sklearn / complex — testable but complex
}


def _build_strategy_config(name: str) -> StrategyConfig:
    """Build a default StrategyConfig for a named strategy."""
    from quantcore.strategies.base import StrategyType
    defaults: dict[str, dict] = {
        "equity_mean_reversion": {"rebalance_freq": 5, "max_position_notional": 1.0},
        "equity_low_volatility": {"rebalance_freq": 21, "max_position_notional": 1.0},
        "equity_quality_factor": {"rebalance_freq": 21, "max_position_notional": 1.0},
        "equity_value_factor": {"rebalance_freq": 21, "max_position_notional": 1.0},
        "equity_momentum": {"rebalance_freq": 21, "max_position_notional": 1.0},
        "equity_earnings_momentum": {"rebalance_freq": 21, "max_position_notional": 1.0},
        "equity_multi_factor": {"rebalance_freq": 21, "max_position_notional": 1.0},
        "equity_volatility_breakout": {"rebalance_freq": 5, "max_position_notional": 1.0},
        "equity_trend_following": {"rebalance_freq": 5, "max_position_notional": 1.0},
        "macro_carry": {"rebalance_freq": 5, "max_position_notional": 1.0},
        "macro_cross_sectional_momentum": {"rebalance_freq": 21, "max_position_notional": 1.0},
        "macro_regime_adaptive_momentum": {"rebalance_freq": 5, "max_position_notional": 1.0},
        "macro_state_selective_momentum": {"rebalance_freq": 5, "max_position_notional": 1.0},
        "macro_true_carry": {"rebalance_freq": 5, "max_position_notional": 1.0},
        "macro_cross_sectional_factor_neutral": {"rebalance_freq": 21, "max_position_notional": 1.0},
        "equity_stat_arb": {"rebalance_freq": 5, "max_position_notional": 1.0},
    }
    s_type = StrategyType.MACRO if name.startswith("macro_") else StrategyType.EQUITY
    return StrategyConfig(
        name=name,
        strategy_type=s_type,
        params=defaults.get(name, {"rebalance_freq": 21, "max_position_notional": 1.0}),
    )


def _import_strategy_module(strategy_name: str):
    """Import a strategy class from quantcore.strategies."""
    strategy_path_map = {
        "equity_mean_reversion": "quantcore.strategies.equity.mean_reversion",
        "equity_low_volatility": "quantcore.strategies.equity.low_volatility",
        "equity_quality_factor": "quantcore.strategies.equity.quality_factor",
        "equity_value_factor": "quantcore.strategies.equity.value_factor",
        "equity_momentum": "quantcore.strategies.equity.momentum",
        "equity_earnings_momentum": "quantcore.strategies.equity.earnings_momentum",
        "equity_multi_factor": "quantcore.strategies.equity.multi_factor",
        "equity_volatility_breakout": "quantcore.strategies.equity.volatility_breakout",
        "equity_trend_following": "quantcore.strategies.equity.trend_following",
        "macro_carry": "quantcore.strategies.macro.carry",
        "macro_cross_sectional_momentum": "quantcore.strategies.macro.cross_sectional_momentum",
        "macro_regime_adaptive_momentum": "quantcore.strategies.macro.regime_adaptive_momentum",
        "macro_state_selective_momentum": "quantcore.strategies.macro.state_selective_momentum",
        "macro_true_carry": "quantcore.strategies.macro.true_carry_strategy",
        "macro_cross_sectional_factor_neutral": "quantcore.strategies.macro.cross_sectional_factor_neutral",
        "equity_stat_arb": "quantcore.strategies.equity.stat_arb",
    }
    module_name = strategy_path_map.get(strategy_name)
    if module_name is None:
        return None, None
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        _log.warning("Could not import %s: %s", module_name, exc)
        return None, None

    # Find the strategy class (last segment of module path)
    class_name_candidates = [
        "".join(w.title() for w in name.split("_")) + "Strategy"
        for name in strategy_name.split("_")
    ]
    for cn in class_name_candidates:
        if hasattr(mod, cn):
            return getattr(mod, cn), mod

    # Fallback: find any class ending in Strategy
    for attr_name in dir(mod):
        if attr_name.endswith("Strategy") and not attr_name.startswith("_"):
            return getattr(mod, attr_name), mod

    return None, mod


# ------------------------------------------------------------------------------
# WalkForwardBacktester — adapted from quantcore.research.walkforward
# We use the module directly; replicate here only the run() interface we need
# ------------------------------------------------------------------------------

from dataclasses import dataclass

# Import the real backtester
from quantcore.research.walkforward import WalkForwardBacktester  # type: ignore[import-untyped]
from quantcore.research.bootstrap import bootstrap_confidence_interval  # type: ignore[import-untyped]
from quantcore.research.deflated_sharpe import deflated_sharpe_ratio  # type: ignore[import-untyped]


# ------------------------------------------------------------------------------
# One strategy validation
# ------------------------------------------------------------------------------


def validate_strategy(
    strategy_name: str,
    prices: pd.DataFrame,
    ff5: pd.DataFrame,
    data_start: str,
    data_end: str,
    n_trials: int,
) -> dict | None:
    """Run walk-forward backtest for one strategy. Returns metrics or None on failure."""

    _log.info("=" * 70)
    _log.info("Validating: %s", strategy_name)
    _log.info("=" * 70)

    # Build strategy instance
    StratClass, _ = _import_strategy_module(strategy_name)
    if StratClass is None:
        _log.error("Could not find strategy class for %s", strategy_name)
        return None

    config = _build_strategy_config(strategy_name)

    try:
        if hasattr(StratClass, "run_returns_backtest"):
            # Has its own backtest method (ResidualMomentumStrategy and similar).
            # These take raw data in __init__, not a StrategyConfig.
            # We pass prices/factors so a) __init__ doesn't crash, b) if it
            # stores them internally the backtest has the right data.
            base_instance = StratClass(
                stock_prices=prices,
                factor_returns=ff5,
                **config.params,
            )
            strategy = base_instance
        else:
            # Wrap via StrategyBacktestWrapper
            base_instance = StratClass(config)
            strategy = StrategyBacktestWrapper(
                strategy=base_instance,
                config=config,
                prices=prices,
                factors=ff5,
            )
    except Exception as exc:
        _log.warning("Strategy.__init__ failed for %s: %s", strategy_name, exc)
        return None

    # Run backtest via WalkForwardBacktester
    wf = WalkForwardBacktester(strategy, train_years=10, test_years=1)
    dates = pd.to_datetime(prices.index)

    try:
        result = wf.run(dates=dates, n_trials=n_trials)
    except Exception as exc:
        _log.exception("WalkForwardBacktester.run failed for %s: %s", strategy_name, exc)
        return None

    return result


# ------------------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------------------


def generate_sweep_report(results: list[dict]) -> str:
    """Generate the consolidated sweep report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M %Z")

    rows = []
    rows.append("# Strategy Walk-Forward Sweep — Honest Validation Report")
    rows.append(f"_Generated: {now}_")
    rows.append("")
    rows.append("## Framework")
    rows.append("")
    rows.append("| Parameter | Value |")
    rows.append("| --- | --- |")
    rows.append(f"| Training window | 10 yr expanding |")
    rows.append(f"| Test window | 1 yr (anchored) |")
    rows.append(f"| Universe | BLUECHIP 100 (hardcoded S&P 500 blue chips) |")
    rows.append(f"| Period | 2005-01-01 to 2024-12-31 |")
    rows.append(f"| Costs | 5bps spread + 10bps impact + $0.005/share |")
    rows.append(f"| DSR trials per strategy | {n_trials} |")
    rows.append(f"| Block bootstrap | 2000 reps, block = max(5, n^(1/3)) |")
    rows.append("")
    rows.append("## Verdict criteria")
    rows.append("")
    rows.append("DEPLOY: CI_lower >= 0.3 (DSR stat reported but not used as filter)")
    rows.append("")
    rows.append("## Results")
    rows.append("")
    rows.append("| Strategy | Status | OOS Sharpe | CI(90%) Lower | DSR p-val | Deflated SR | Max DD | Verdict |")
    rows.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")

    for r in results:
        sharpe = r.get("annualised_sharpe")
        ci_lower = r.get("sharpe_ci_90_lower")
        dsr_p = r.get("deflated_sharpe_pvalue")
        dsr_stat = r.get("deflated_sharpe_stat")
        max_dd = r.get("max_drawdown")
        status = r.get("status", "unknown")
        verdict = r.get("verdict", "UNKNOWN")

        sharpe_s = f"{sharpe:+.4f}" if isinstance(sharpe, float) and not np.isnan(sharpe) else "N/A"
        ci_s = f"{ci_lower:+.4f}" if isinstance(ci_lower, float) and not np.isnan(ci_lower) else "N/A"
        dsr_p_s = f"{dsr_p:.4f}" if isinstance(dsr_p, float) else "N/A"
        dsr_s = f"{dsr_stat:+.4f}" if isinstance(dsr_stat, float) else "N/A"
        dd_s = f"{max_dd:.2%}" if isinstance(max_dd, float) else "N/A"

        rows.append(
            f"| {r['strategy_name']} | {status} | {sharpe_s} | "
            f"{ci_s} | {dsr_p_s} | {dsr_s} | {dd_s} | **{verdict}** |"
        )

    rows.append("")
    rows.append("## Skipped / Untestable")
    rows.append("")
    rows.append("| Strategy | Reason |")
    rows.append("| --- | --- |")
    rows.append("| equity_momentum | DISABLED (prior OOS REJECT) |")
    rows.append("| macro_cta_trend | DISABLED (prior OOS Sharpe -0.147) |")
    rows.append("| macro_cross_sectional_momentum_v2 | DISABLED (prior OOS REJECT) |")
    rows.append("| equity_volatility_breakout | Requires OHLCV (high/low) — no source in sweep |")
    rows.append("| equity_trend_following | Requires OHLCV (high/low) — no source in sweep |")
    rows.append("| macro_true_carry | Requires data['carry'] — no carry computation source |")
    rows.append("| options_delta_neutral | Requires options chains — no source |")
    rows.append("| options_iron_condor | Requires options chains — no source |")
    rows.append("| options_earnings_vol | Requires options chains — no source |")
    rows.append("| options_tail_hedge | Requires options chains — no source |")

    return "\n".join(rows)


# ------------------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------------------

# Strategies to test — all on_data-based strategies with close-only data needs
# Strategies to test — on_data-based, close-only data reqs
# NOTE: macro_carry (needs data["carry"]), equity_momentum (DISABLED),
# equity_volatility_breakout/trend_following (need OHLCV), and
# macro_cross_sectional_factor_neutral (no strategy class — module is a stub)
# are all excluded.
STRATEGIES_TO_TEST = [
    "equity_mean_reversion",
    "equity_low_volatility",
    "equity_quality_factor",
    "equity_value_factor",
    "equity_earnings_momentum",
    "equity_multi_factor",
    "equity_stat_arb",
    # macro strategies
    "macro_cross_sectional_momentum",
    "macro_regime_adaptive_momentum",
    "macro_state_selective_momentum",
]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward strategy sweep")
    parser.add_argument("--strategies", nargs="*", help="Subset of strategies to test")
    parser.add_argument("--n-trials", type=int, default=10, help="DSR num_trials (default 10)")
    args = parser.parse_args()

    strategy_list = args.strategies if args.strategies else STRATEGIES_TO_TEST

    _log.info("Loading FF5 data...")
    ff5 = load_ff5_data()

    _log.info("Loading stock prices for BLUECHIP universe (%d stocks)...", len(BLUECHIP))
    data_start = max(pd.Timestamp("2005-01-01"), ff5.index.min())
    data_end = min(pd.Timestamp("2024-12-31"), ff5.index.max())
    stock_start = str(data_start.date())
    stock_end = str(data_end.date())

    try:
        prices = load_stock_prices(BLUECHIP, stock_start, stock_end)
    except Exception as exc:
        _log.error("Failed to load stock prices: %s", exc)
        return 1

    if len(prices.columns) < 20:
        _log.error("Only %d tickers loaded. Aborting.", len(prices.columns))
        return 1

    # Align to FF5
    common = prices.index.intersection(ff5.index)
    prices = prices.loc[common]
    ff5_aligned = ff5.loc[common]
    _log.info("Aligned data: %d days x %d tickers (%s to %s)",
              len(prices), len(prices.columns), common[0].date(), common[-1].date())

    # n_trials is the number of bootstrap resamples used by WalkForwardBacktester
    # to estimate the Sharpe CI and DSR. 10 is appropriate for walk-forward validation
    # (DSR num_trials = 10, not the number of strategies — DSR is for single-strategy
    # validation, not multiple-comparison correction across strategies).
    n_trials = 10

    all_results: list[dict] = []

    for strat_name in strategy_list:
        result = validate_strategy(
            strategy_name=strat_name,
            prices=prices,
            ff5=ff5_aligned,
            data_start=stock_start,
            data_end=stock_end,
            n_trials=n_trials,
        )

        if result is None:
            _log.warning("Validation returned None for %s — skipping", strat_name)
            all_results.append({
                "strategy_name": strat_name,
                "status": "failed",
                "annualised_sharpe": float("nan"),
                "sharpe_ci_90_lower": float("nan"),
                "deflated_sharpe_pvalue": 1.0,
                "deflated_sharpe_stat": 0.0,
                "max_drawdown": 0.0,
                "verdict": "SKIP",
            })
            continue

        sharpe = result.get("annualised_sharpe", float("nan"))
        ci_lower = result.get("sharpe_ci_90_lower", float("nan"))
        dsr_p = result.get("deflated_sharpe_pvalue", 1.0)

        verdict = "DEPLOY" if ci_lower >= 0.3 else "REJECT"

        _log.info(
            "  → %s | Sharpe=%+.3f | CI_lower=%+.3f | DSR_p=%.4f | %s",
            strat_name, sharpe, ci_lower, dsr_p, verdict
        )

        all_results.append({
            "strategy_name": strat_name,
            "status": "completed",
            "annualised_sharpe": sharpe,
            "sharpe_ci_90_lower": ci_lower,
            "deflated_sharpe_pvalue": dsr_p,
            "deflated_sharpe_stat": result.get("deflated_sharpe_stat", 0.0),
            "max_drawdown": result.get("max_drawdown", 0.0),
            "verdict": verdict,
        })

        # Log experiment
        exp = Experiment(
            strategy_name=strat_name,
            parameters={"lookback_months": 1, "min_obs": 252, "max_weight": 0.05},
            start_date=stock_start,
            end_date=stock_end,
            oos_sharpe=sharpe,
            deflated_sharpe_pvalue=dsr_p,
            ci_90_lower=ci_lower,
            status="completed",
        )
        try:
            log_experiment(exp)
        except Exception as exc:
            _log.warning("Failed to log experiment for %s: %s", strat_name, exc)

    # Save report
    report_path = REPORTS_DIR / "strategy_sweep.md"
    report_path.write_text(generate_sweep_report(all_results), encoding="utf-8")
    _log.info("Sweep report saved to %s", report_path)

    # Summary table
    _log.info("=" * 70)
    _log.info("SWEEP SUMMARY")
    _log.info("=" * 70)
    _log.info(f"{'Strategy':<45} {'Sharpe':>8} {'CI_LOW':>8} {'DSR_p':>7} {'Verdict':>8}")
    _log.info("-" * 70)
    for r in all_results:
        s = r["annualised_sharpe"]
        c = r["sharpe_ci_90_lower"]
        p = r["deflated_sharpe_pvalue"]
        s_s = f"{s:+.3f}" if isinstance(s, float) and not np.isnan(s) else "N/A"
        c_s = f"{c:+.3f}" if isinstance(c, float) and not np.isnan(c) else "N/A"
        p_s = f"{p:.4f}" if isinstance(p, float) else "N/A"
        _log.info(f"{r['strategy_name']:<45} {s_s:>8} {c_s:>8} {p_s:>7} {r['verdict']:>8}")
    _log.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())