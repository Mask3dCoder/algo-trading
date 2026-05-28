"""Walk-forward backtest for the Volatility Risk Premium strategy.

Runs 6-fold anchored walk-forward validation using real VIX/VVIX data
from yfinance and SPY OHLCV as the primary instrument.

Usage::

    python scripts/vrp_backtest.py
    python scripts/vrp_backtest.py --start 2015-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantcore.backtest.walk_forward import (
    WFResult,
    aggregate_oos_equity,
    anchored_walk_forward,
    check_acceptance_criteria,
    compute_oos_metrics,
)
from quantcore.data.sources.volatility_data import VolatilityDataLoader
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.macro.volatility_risk_premium import (
    VolatilityRiskPremiumStrategy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# Strategy parameters (from macro_vrp.toml defaults)
DEFAULT_PARAMS = {
    "lookback": 21,
    "ewm_span": 5,
    "entry_threshold": 0.25,
    "vix_crisis_threshold": 35,
    "target_vol": 0.10,
    "kelly_fraction": 0.5,
    "max_position_pct": 0.25,
    "stop_loss_sigma_normal": 2.0,
    "stop_loss_sigma_crisis": 3.0,
    "profit_target_pct": 0.05,
    "max_holding_days": 30,
    "signal_exit_threshold": 0.0,
    "tail_hedge_pct": 0.02,
    "vol_lookback": 21,
    "rebalance_freq": 5,
}


def load_data(
    start: str = "2015-01-01",
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SPY OHLCV and vol index data.

    Returns:
        (spy_df, vol_df) — both indexed by date, aligned.
    """
    _log.info("Loading SPY OHLCV data...")
    spy = yf.download("SPY", start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy[["Open", "High", "Low", "Close", "Volume"]].copy()
    spy.columns = ["open", "high", "low", "close", "volume"]
    spy.index = pd.to_datetime(spy.index).tz_localize(None)

    _log.info("Loading vol index data...")
    vol_loader = VolatilityDataLoader()
    vol_data = vol_loader.load_all(start=start, end=end)

    # Build vol DataFrame
    vol_df = pd.DataFrame({
        "vix": vol_data.vix,
        "vvix": vol_data.vvix,
        "ts_slope": vol_data.term_structure_slope,
    })

    # Align on common dates
    common_idx = spy.index.intersection(vol_df.index)
    spy = spy.loc[common_idx]
    vol_df = vol_df.loc[common_idx]

    _log.info("Loaded %d bars (%s to %s)", len(spy), spy.index[0].date(), spy.index[-1].date())
    return spy, vol_df


def run_fold(
    spy: pd.DataFrame,
    vol_df: pd.DataFrame,
    fold_idx: int,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    params: dict,
) -> WFResult:
    """Run a single walk-forward fold."""
    _log.info(
        "Fold %d: train=[%d:%d] test=[%d:%d]",
        fold_idx, train_start, train_end, test_start, test_end,
    )

    # Training period (for reference — strategy doesn't train, but we compute
    # in-sample metrics)
    train_spy = spy.iloc[train_start:train_end]
    train_close = train_spy["close"].values.reshape(-1, 1)

    # Test period
    test_spy = spy.iloc[test_start:test_end]
    test_vol = vol_df.iloc[test_start:test_end]

    if len(test_spy) < 10:
        return WFResult(
            fold=None,  # type: ignore
            in_sample_sharpe=0.0,
            out_of_sample_sharpe=0.0,
        )

    # Create strategy instance
    config = StrategyConfig(
        name="macro_vrp",
        strategy_type=StrategyType.MACRO,
        max_position_notional=5_000_000,
        max_leverage=1.0,
        risk_budget_pct=0.01,
        params=params,
    )
    strategy = VolatilityRiskPremiumStrategy(config)

    # Attach full vol arrays — strategy indexes by bar_idx = n_periods - 1
    vix_full = vol_df["vix"].values
    vvix_full = vol_df["vvix"].values
    ts_full = vol_df["ts_slope"].values

    strategy.attach_vol_data(vix_full, vvix_full, ts_full)

    # Use FULL spy data for building data dicts (strategy needs 262+ bars)
    full_close = spy["close"].values
    full_high = spy["high"].values
    full_low = spy["low"].values
    full_open = spy["open"].values

    # Pre-compute daily realized vol (annualized) on full data for P&L
    spy_log_ret = np.diff(np.log(full_close))
    daily_rv = np.zeros(len(full_close))
    rv_window = 21
    for j in range(rv_window, len(spy_log_ret)):
        daily_rv[j] = np.std(spy_log_ret[j - rv_window : j]) * np.sqrt(252)

    # Pre-compute daily simple returns for gamma P&L
    spy_daily_ret = np.zeros(len(full_close))
    for j in range(1, len(full_close)):
        if full_close[j - 1] > 0:
            spy_daily_ret[j] = (full_close[j] - full_close[j - 1]) / full_close[j - 1]

    returns: list[float] = []
    position = 0.0
    entry_vix = 0.0
    entry_day = 0
    holding_days = 0

    # Iterate over test period indices in the full array
    for i in range(test_start, test_end):
        # Build data dict from full data up to current bar
        bar_close = full_close[: i + 1].reshape(-1, 1)
        bar_high = full_high[: i + 1].reshape(-1, 1)
        bar_low = full_low[: i + 1].reshape(-1, 1)
        bar_open = full_open[: i + 1].reshape(-1, 1)

        data = {
            "close": bar_close,
            "high": bar_high,
            "low": bar_low,
            "open": bar_open,
            "symbols": ["SPY"],
        }

        signals = strategy.on_data(data)

        # VRP return model: short vol has two components:
        # 1. Theta/VRP yield: collects (IV - RV) premium daily
        # 2. Gamma P&L: short gamma loses on large moves (-0.5 × gamma × dS²)
        # 3. Transaction costs: spread + commissions on entry/exit
        daily_ret = 0.0
        if position != 0 and i < len(daily_rv):
            current_vix = float(vix_full[i]) / 100.0  # annualized IV
            current_rv = daily_rv[i]
            # Premium collected (positive when IV > RV)
            vrp_daily = (current_vix - current_rv) / np.sqrt(252)
            theta_component = position * vrp_daily

            # Gamma P&L: short gamma loses on large moves
            # Approximate: -0.5 × position × (daily_return)² × scaling
            # The scaling factor converts from raw gamma to portfolio impact
            gamma_scaling = 30.0  # calibrated for realistic vol drag
            gamma_component = -position * gamma_scaling * spy_daily_ret[i] ** 2

            daily_ret = theta_component + gamma_component

            # Tail hedge cost (2% annual drag)
            daily_ret -= params.get("tail_hedge_pct", 0.02) / 252

            holding_days += 1

        # Process signals — scale strength back to actual portfolio weight
        max_pos_pct = params.get("max_position_pct", 0.25)
        tc_bps = 5  # 5 bps round-trip cost for options/ETFs
        for sig in signals:
            if sig.metadata.get("type") == "tail_hedge":
                continue
            if sig.direction == -1:  # Short vol entry
                new_position = sig.strength * max_pos_pct
                # Transaction cost on position change
                daily_ret -= abs(new_position - position) * tc_bps / 10000
                position = new_position
                entry_vix = float(vix_full[i]) / 100.0
                entry_day = i
                holding_days = 0
            elif sig.direction == 1:  # Exit
                # Transaction cost on exit
                daily_ret -= abs(position) * tc_bps / 10000
                position = 0.0

        returns.append(daily_ret)

    returns_arr = np.array(returns)

    # In-sample metrics (from training period — simple buy-and-hold for reference)
    if len(train_close) > 1:
        train_returns = np.diff(train_close.flatten()) / train_close[:-1].flatten()
        train_equity = np.cumprod(1 + train_returns)
        is_metrics = compute_oos_metrics(train_returns, train_equity)
        is_sharpe = is_metrics.get("sharpe", 0.0)
        if np.isnan(is_sharpe):
            is_sharpe = 0.0
    else:
        is_sharpe = 0.0

    # Out-of-sample metrics
    if len(returns_arr) > 1:
        oos_equity = np.cumprod(1 + returns_arr)
        oos_metrics = compute_oos_metrics(returns_arr, oos_equity)
        oos_sharpe = oos_metrics.get("sharpe", 0.0)
        if np.isnan(oos_sharpe):
            oos_sharpe = 0.0
    else:
        oos_sharpe = 0.0

    return WFResult(
        fold=None,  # type: ignore
        in_sample_sharpe=is_sharpe,
        out_of_sample_sharpe=oos_sharpe,
        out_of_sample_returns=returns_arr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VRP Walk-Forward Backtest")
    parser.add_argument("--start", default="2015-01-01", help="Data start date")
    parser.add_argument("--end", default=None, help="Data end date")
    parser.add_argument("--train-size", type=int, default=504, help="Training bars")
    parser.add_argument("--test-size", type=int, default=126, help="Test bars")
    args = parser.parse_args()

    # Load data
    spy, vol_df = load_data(start=args.start, end=args.end)
    n_bars = len(spy)

    if n_bars < args.train_size + args.test_size + 10:
        _log.error("Insufficient data: %d bars (need %d)", n_bars, args.train_size + args.test_size)
        sys.exit(1)

    # Generate walk-forward folds
    folds = anchored_walk_forward(
        n_bars=n_bars,
        initial_train=args.train_size,
        test_size=args.test_size,
    )
    _log.info("Generated %d walk-forward folds", len(folds))

    # Run each fold
    results: list[WFResult] = []
    for i, fold in enumerate(folds):
        result = run_fold(
            spy=spy,
            vol_df=vol_df,
            fold_idx=i,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            params=DEFAULT_PARAMS,
        )
        results.append(result)

    # Aggregate OOS results
    oos_equity = aggregate_oos_equity(results)
    if len(oos_equity) > 0:
        oos_returns = oos_equity.pct_change().dropna().values
        oos_eq = oos_equity.values
    else:
        oos_returns = np.array([])
        oos_eq = np.array([])

    # Compute metrics
    metrics = compute_oos_metrics(oos_returns, oos_eq) if len(oos_returns) > 1 else {}

    # Count passing folds
    passing = sum(
        1 for r in results
        if r.out_of_sample_sharpe > 0.5
    )

    # Print report
    print("\n" + "=" * 60)
    print("VRP Strategy — Walk-Forward Results")
    print("=" * 60)
    print(f"Period: {spy.index[0].date()} to {spy.index[-1].date()}")
    print(f"Total bars: {n_bars}")
    print(f"Folds: {len(results)}")
    print(f"Passing folds (OOS Sharpe > 0.5): {passing}/{len(results)}")
    print()

    for i, r in enumerate(results):
        print(f"  Fold {i}: IS Sharpe={r.in_sample_sharpe:.3f}, OOS Sharpe={r.out_of_sample_sharpe:.3f}")

    print()
    if metrics:
        print(f"Aggregate OOS Sharpe:      {metrics.get('sharpe', 'N/A')}")
        print(f"Aggregate OOS Max DD:      {metrics.get('max_dd_pct', 'N/A')}%")
        print(f"Aggregate Annual Return:   {metrics.get('annual_return', 'N/A')}%")
        print(f"Aggregate Total Return:    {metrics.get('total_return_pct', 'N/A')}%")

    # Acceptance criteria
    oos_sharpe = metrics.get("sharpe", 0.0)
    if np.isnan(oos_sharpe):
        oos_sharpe = 0.0
    max_dd = metrics.get("max_dd_pct", 0.0)

    is_valid, reason = check_acceptance_criteria(
        oos_sharpe=oos_sharpe,
        max_dd_pct=max_dd,
        periods_passed=passing,
        total_periods=len(results),
    )

    print()
    print(f"Acceptance: {'PASS' if is_valid else 'FAIL'} — {reason}")
    print("=" * 60)

    # Save equity curve
    if len(oos_equity) > 0:
        out_path = Path(__file__).resolve().parent.parent / "backtest_equity_curve.csv"
        oos_equity.to_csv(out_path)
        _log.info("Equity curve saved to %s", out_path)


if __name__ == "__main__":
    main()
