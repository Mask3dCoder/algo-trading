"""ETF Universe Expansion Test — Walk-forward validation on 20 macro ETFs.

Tests macro_cross_sectional_momentum and macro_state_selective_momentum
on a 20-ETF universe vs the BLUECHIP 100 baseline.

Walk-forward: 10-year expanding train, 1-year anchored test, 2005-2024.
Cost model: 5 bps round-trip turnover cost.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantcore.backtest.walk_forward import anchored_walk_forward, compute_oos_metrics
from quantcore.research.bootstrap import bootstrap_confidence_interval
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class

# Trigger @register decorators
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401
import quantcore.strategies.macro.state_selective_momentum  # noqa: F401

from quantcore.data import yfinance_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

ETF_20 = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",   # Equities
    "TLT", "IEF", "SHY",                    # Bonds
    "GLD", "SLV",                            # Precious metals
    "USO", "DBC", "UNG",                     # Commodities
    "UUP", "FXY",                            # Currencies
    "VNQ",                                   # REITs
    "TIP",                                   # TIPS
    "HYG", "LQD",                            # Credit
    "VXX",                                   # Volatility
]

BLUECHIP_100 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "DIS", "NFLX", "ADBE", "CRM", "PYPL",
    "INTC", "CMCSA", "PEP", "ABT", "KO", "NKE", "MRK", "TMO", "AVGO", "COST",
    "WMT", "DHR", "ACN", "LLY", "TXN", "MDT", "NEE", "PM", "UNP", "HON",
    "LOW", "AMGN", "IBM", "BA", "GE", "CAT", "MMM", "DE", "AXP", "SYK",
    "ISRG", "BLK", "GILD", "ADI", "MMC", "CB", "PLD", "SCHW", "CI", "SO",
    "DUK", "CL", "ZTS", "BDX", "ICE", "CME", "AON", "SHW", "EL", "LHX",
    "PNC", "USB", "TFC", "TGT", "PSA", "EQIX", "SLB", "EOG", "COP", "PXD",
    "OXY", "F", "GM", "DAL", "UAL", "LUV", "AAL", "ROK", "EMR", "ETN",
    "ITW", "PH", "FAST", "ODFL", "CSX", "NSC", "UNH", "PRU", "MET", "AFL",
    "TRV", "PGR", "ALL", "CPT", "MAA", "O", "DOW", "DD", "HON",
]

START_DATE = "2005-01-01"
END_DATE = "2024-12-31"
INITIAL_TRAIN = 2520   # 10 years
TEST_SIZE = 252        # 1 year
TC_BPS = 5.0           # 5 bps round-trip turnover cost
N_BOOT = 2000          # bootstrap reps for CI

# ── Baseline results (BLUECHIP 100) ───────────────────────────────────

BASELINE = {
    "macro_cross_sectional_momentum": {
        "oos_sharpe": 0.8521,
        "ci_90_lower": 0.3337,
        "max_dd": 6.29,
    },
    "macro_state_selective_momentum": {
        "oos_sharpe": 0.8114,
        "ci_90_lower": 0.4000,
        "max_dd": float("nan"),
    },
}

# ── Strategy configs ──────────────────────────────────────────────────

STRATEGY_CONFIGS = {
    "macro_cross_sectional_momentum": {
        "rebalance_freq": 5,
        "mom_windows": "20,60,120",
        "mom_weights": "0.2,0.5,0.3",
        "target_vol": 0.12,
        "vol_lookback": 20,
        "use_vol_adjustment": True,
        "max_gross_exposure": 1.0,
        "beta_neutral": True,
        "regime_filter": True,
        "dd_control": True,
        "min_weight": 0.01,
    },
    "macro_state_selective_momentum": {
        "short_window": 21,
        "medium_window": 63,
        "vol_target": 0.15,
        "vol_lookback": 60,
        "rebalance_freq": 5,
        "min_position": 0.05,
        "instability_buffer": 252,
    },
}


# ── Data loading ──────────────────────────────────────────────────────

def load_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Load daily close prices for symbols."""
    _log.info("Loading %d symbols: %s -> %s", len(symbols), start, end)
    data = yfinance_loader.load_multiple(symbols, start, end, interval="1d")
    _log.info("Loaded %d / %d symbols", len(data), len(symbols))

    frames = {}
    for sym, df in data.items():
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        frames[sym] = df["close"]

    prices = pd.DataFrame(frames)
    prices = prices.ffill().bfill()
    # Drop symbols with >10% missing
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.90))
    _log.info("Price matrix: %s", prices.shape)
    return prices


def prices_to_data(prices: pd.DataFrame) -> dict:
    """Convert price DataFrame to strategy data dict."""
    symbols = list(prices.columns)
    close = prices.values
    n_periods, n_assets = close.shape

    # Build OHLCV (OHLC all = close for daily)
    data = {
        "close": close,
        "open": close,
        "high": close,
        "low": close,
        "volume": np.ones((n_periods, n_assets)),
        "symbols": symbols,
        "timestamps": list(prices.index),
    }
    return data


# ── Walk-forward backtest ─────────────────────────────────────────────

def run_walk_forward(
    strategy_name: str,
    prices: pd.DataFrame,
) -> tuple[np.ndarray, list]:
    """Run anchored walk-forward backtest for a strategy.

    Returns (oos_returns, fold_results).
    """
    data = prices_to_data(prices)
    close = data["close"]
    symbols = data["symbols"]
    n_periods = close.shape[0]

    _log.info("  Data: %d bars x %d assets", n_periods, len(symbols))

    folds = anchored_walk_forward(
        n_bars=n_periods,
        initial_train=INITIAL_TRAIN,
        test_size=TEST_SIZE,
        embargo_pct=0.01,
    )
    _log.info("  Walk-forward folds: %d", len(folds))

    if not folds:
        return np.array([]), []

    all_oos_returns: list[np.ndarray] = []
    fold_results: list[dict] = []

    strat_cls = get_strategy_class(strategy_name)
    strat_params = STRATEGY_CONFIGS[strategy_name]

    for fold_idx, fold in enumerate(folds):
        cfg = StrategyConfig(
            name=strategy_name,
            strategy_type=StrategyType.MACRO,
            params=strat_params.copy(),
        )
        strat = strat_cls(cfg)
        strat.initialize()

        # Warmup on training data
        warmup_end = fold.train_end
        for bar_i in range(0, warmup_end):
            bar_data = {}
            for k, v in data.items():
                if k == "symbols" or k == "timestamps":
                    bar_data[k] = v
                elif isinstance(v, np.ndarray) and v.ndim >= 1:
                    bar_data[k] = v[:bar_i + 1]
                else:
                    bar_data[k] = v
            if bar_i < len(data["timestamps"]):
                bar_data["timestamp"] = data["timestamps"][bar_i]
            try:
                strat.on_data(bar_data)
            except Exception:
                pass

        # Run on test period
        test_start = fold.test_start
        test_end = fold.test_end
        n_test = test_end - test_start

        if n_test < 2:
            continue

        asset_rets = (close[test_start + 1:test_end] - close[test_start:test_end - 1]) / np.maximum(close[test_start:test_end - 1], 1e-10)

        bar_rets = np.zeros(n_test - 1)
        carried_weights: dict[str, float] = {}

        for idx, bar_i in enumerate(range(test_start, test_end - 1)):
            bar_data = {}
            for k, v in data.items():
                if k == "symbols" or k == "timestamps":
                    bar_data[k] = v
                elif isinstance(v, np.ndarray) and v.ndim >= 1:
                    bar_data[k] = v[:bar_i + 1]
                else:
                    bar_data[k] = v
            if bar_i < len(data["timestamps"]):
                bar_data["timestamp"] = data["timestamps"][bar_i]

            try:
                signals = strat.on_data(bar_data) or []
            except Exception:
                signals = []

            weighted_ret = 0.0
            total_weight = 0.0
            current_weights: dict[str, float] = {}

            for s in signals:
                weight = float(s.direction) * float(s.strength)
                if s.direction != 0:
                    current_weights[s.symbol] = weight
                try:
                    asset_idx = symbols.index(s.symbol)
                except (ValueError, IndexError):
                    continue
                weighted_ret += weight * float(asset_rets[idx, asset_idx])
                total_weight += abs(float(s.strength))

            if total_weight > 1e-10:
                bar_rets[idx] = weighted_ret / total_weight

            # Transaction cost
            if TC_BPS > 0 and current_weights:
                turnover = 0.0
                all_syms = set(carried_weights.keys()) | set(current_weights.keys())
                for sym in all_syms:
                    prev_w = carried_weights.get(sym, 0.0)
                    curr_w = current_weights.get(sym, 0.0)
                    turnover += abs(curr_w - prev_w)
                bar_rets[idx] -= turnover * TC_BPS / 10000.0

            carried_weights = current_weights

        all_oos_returns.append(bar_rets)

        # Fold metrics
        if len(bar_rets) > 1:
            fold_mean = float(np.mean(bar_rets))
            fold_std = float(np.std(bar_rets, ddof=1))
            fold_sharpe = fold_mean / fold_std * np.sqrt(252) if fold_std > 1e-10 else 0.0
        else:
            fold_sharpe = 0.0

        fold_results.append({
            "fold": fold_idx,
            "train_bars": fold.train_end,
            "test_bars": n_test,
            "sharpe": fold_sharpe,
        })
        _log.info("  Fold %d: train=%d, test=%d, sharpe=%.3f", fold_idx, fold.train_end, n_test, fold_sharpe)

    if not all_oos_returns:
        return np.array([]), fold_results

    return np.concatenate(all_oos_returns), fold_results


# ── Metrics ───────────────────────────────────────────────────────────

def compute_full_metrics(returns: np.ndarray) -> dict:
    """Compute OOS Sharpe, bootstrap CI, max DD, win rate."""
    if len(returns) < 2:
        return {"error": "insufficient returns"}

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1))
    annual_sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 1e-10 else 0.0

    # Bootstrap CI
    block_size = max(5, int(len(returns) ** (1 / 3)))
    ci_lower, ci_upper = bootstrap_confidence_interval(
        returns, block_size=block_size, n_boot=N_BOOT, ci=0.90,
    )

    # Max drawdown
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.abs(dd.min())) * 100.0

    # Win rate
    win_rate = float(np.mean(returns > 0)) * 100.0

    return {
        "oos_sharpe": round(annual_sharpe, 4),
        "ci_90_lower": round(ci_lower, 4),
        "ci_90_upper": round(ci_upper, 4),
        "max_dd_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "n_oos_days": len(returns),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    _log.info("=" * 60)
    _log.info("ETF Universe Expansion Test")
    _log.info("=" * 60)

    # Load both universes in parallel-ish
    _log.info("Loading 20-ETF universe...")
    prices_20 = load_prices(ETF_20, START_DATE, END_DATE)

    _log.info("Loading BLUECHIP 100 universe...")
    prices_100 = load_prices(BLUECHIP_100, START_DATE, END_DATE)

    strategies = ["macro_cross_sectional_momentum", "macro_state_selective_momentum"]
    results: dict[str, dict] = {}

    for strat_name in strategies:
        _log.info("-" * 40)
        _log.info("Testing: %s", strat_name)

        # 20-ETF
        _log.info("  Running 20-ETF walk-forward...")
        oos_20, folds_20 = run_walk_forward(strat_name, prices_20)
        metrics_20 = compute_full_metrics(oos_20)

        # BLUECHIP 100 (re-run for comparison)
        _log.info("  Running BLUECHIP 100 walk-forward...")
        oos_100, folds_100 = run_walk_forward(strat_name, prices_100)
        metrics_100 = compute_full_metrics(oos_100)

        results[strat_name] = {
            "etf_20": metrics_20,
            "bluechip_100": metrics_100,
            "baseline": BASELINE[strat_name],
        }

    # ── Print results ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ETF EXPANSION TEST RESULTS")
    print("=" * 80)
    print(f"Period: {START_DATE} -> {END_DATE}")
    print(f"Walk-forward: {INITIAL_TRAIN//252}yr train, {TEST_SIZE//252}yr test")
    print(f"Cost: {TC_BPS} bps round-trip turnover")
    print(f"Bootstrap CI: {N_BOOT} reps, 90%")
    print()

    for strat_name, res in results.items():
        m20 = res["etf_20"]
        m100 = res["bluechip_100"]
        base = res["baseline"]

        print(f"{'='*60}")
        print(f"Strategy: {strat_name}")
        print(f"{'='*60}")

        if "error" not in m20:
            verdict_20 = "DEPLOY" if m20["ci_90_lower"] > 0.3 else "MONITOR" if m20["oos_sharpe"] > 0.5 else "RESEARCH" if m20["oos_sharpe"] > 0 else "REJECT"
            print(f"  20-ETF:        Sharpe={m20['oos_sharpe']:+.4f}  CI=[{m20['ci_90_lower']:+.4f}, {m20['ci_90_upper']:+.4f}]  MaxDD={m20['max_dd_pct']:.2f}%  WinRate={m20['win_rate_pct']:.1f}%  Verdict={verdict_20}")
        else:
            print(f"  20-ETF:        ERROR - {m20['error']}")

        if "error" not in m100:
            verdict_100 = "DEPLOY" if m100["ci_90_lower"] > 0.3 else "MONITOR" if m100["oos_sharpe"] > 0.5 else "RESEARCH" if m100["oos_sharpe"] > 0 else "REJECT"
            print(f"  BLUECHIP 100:  Sharpe={m100['oos_sharpe']:+.4f}  CI=[{m100['ci_90_lower']:+.4f}, {m100['ci_90_upper']:+.4f}]  MaxDD={m100['max_dd_pct']:.2f}%  WinRate={m100['win_rate_pct']:.1f}%  Verdict={verdict_100}")
        else:
            print(f"  BLUECHIP 100:  ERROR - {m100['error']}")

        print(f"  Baseline:      Sharpe={base['oos_sharpe']:+.4f}  CI_lower={base['ci_90_lower']:+.4f}  MaxDD={base['max_dd']:.2f}%")

        # Comparison
        if "error" not in m20 and "error" not in m100:
            ci_diff = m20["ci_90_lower"] - m100["ci_90_lower"]
            sr_diff = m20["oos_sharpe"] - m100["oos_sharpe"]
            print(f"  Delta (20ETF - 100):  Sharpe {sr_diff:+.4f}  CI_lower {ci_diff:+.4f}")

            if m20["ci_90_lower"] > m100["ci_90_lower"]:
                print(f"  -> 20-ETF IMPROVES CI_lower by {abs(ci_diff):.4f}")
            else:
                print(f"  -> BLUECHIP 100 BETTER by {abs(ci_diff):.4f} in CI_lower")

        print()

    # ── Decision ──────────────────────────────────────────────────────
    print("=" * 80)
    print("DECISION")
    print("=" * 80)

    for strat_name, res in results.items():
        m20 = res["etf_20"]
        m100 = res["bluechip_100"]

        if "error" in m20 or "error" in m100:
            print(f"  {strat_name}: INSUFFICIENT DATA — no change")
            continue

        if m20["ci_90_lower"] > m100["ci_90_lower"]:
            print(f"  {strat_name}: EXPAND to 20 ETFs (CI_lower improves {m100['ci_90_lower']:+.4f} -> {m20['ci_90_lower']:+.4f})")
        else:
            print(f"  {strat_name}: KEEP BLUECHIP 100 (CI_lower degrades {m100['ci_90_lower']:+.4f} -> {m20['ci_90_lower']:+.4f})")

    print("=" * 80)


if __name__ == "__main__":
    main()
