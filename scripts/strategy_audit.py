#!/usr/bin/env python
"""QUANTCORE STRATEGY AUDIT — Phases 2-7.

Phase 2: Walk-forward analysis for all 15 registered strategies
Phase 3: Master audit table sorted by OOS Sharpe
Phase 4: Engine waterfall decomposition (non-DISCARD strategies)
Phase 5: Failure analysis for DISCARD/INVESTIGATE
Phase 6: Portfolio-level analysis for survivors
Phase 7: Final deliverable
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# Import all strategy modules to trigger @register decorators
import quantcore.strategies.equity.mean_reversion  # noqa: F401
import quantcore.strategies.equity.momentum  # noqa: F401
import quantcore.strategies.equity.stat_arb  # noqa: F401
import quantcore.strategies.equity.trend_following  # noqa: F401
import quantcore.strategies.equity.intraday_breakout  # noqa: F401
import quantcore.strategies.equity.intraday_mean_reversion  # noqa: F401
import quantcore.strategies.equity.intraday_momentum  # noqa: F401
import quantcore.strategies.equity.intraday_trend  # noqa: F401
import quantcore.strategies.macro.carry  # noqa: F401
import quantcore.strategies.macro.cta_trend  # noqa: F401
import quantcore.strategies.macro.risk_parity  # noqa: F401
import quantcore.strategies.options.delta_neutral  # noqa: F401
import quantcore.strategies.options.earnings_vol  # noqa: F401
import quantcore.strategies.options.iron_condor  # noqa: F401
import quantcore.strategies.options.tail_hedge  # noqa: F401

from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import create, list_strategies
from quantcore.backtest.walk_forward import anchored_walk_forward
from quantcore.risk.metrics import sharpe_ratio, sortino_ratio, calmar_ratio
from quantcore.risk.metrics import compute_all_metrics
from quantcore.risk.psr import probabilistic_sharpe_ratio, deflated_sharpe_ratio
from quantcore.risk.attribution import capm_attribution

# ── Phase 2: Strategy configs ──────────────────────────────────────────────────

_EQUITY_PARAMS: dict[str, dict] = {
    "equity_momentum": {"rebalance_freq": 1, "lookback_months": 6, "skip_months": 1},
    "equity_stat_arb": {"rescan_freq": 1, "entry_z": 1.5, "lookback": 100, "max_pairs": 2},
    "equity_mean_reversion": {"rsi_oversold": 30, "rsi_overbought": 70},
    "equity_trend_following": {"trend_window": 50, "fast_window": 10, "slow_window": 30},
}

_OPTIONS_PARAMS: dict[str, dict] = {
    "options_delta_neutral": {"ivr_entry": 30, "hedge_freq": 1},
    "options_iron_condor": {"ivr_entry": 30},
    "options_earnings_vol": {},
    "options_tail_hedge": {},
}

_MACRO_PARAMS: dict[str, dict] = {
    "macro_risk_parity": {"rebalance_freq": 1, "lookback": 100},
    "macro_cta_trend": {"rebalance_freq": 1, "slow_window": 100},
    "macro_carry": {"rebalance_freq": 1, "n_long": 2, "n_short": 2},
}

_INTRADAY_PARAMS: dict[str, dict] = {
    "intraday_breakout": {"range_bars": 10, "atr_window": 7, "session_bars": 20,
                          "exit_before_close": 3, "cooldown_bars": 2},
    "intraday_mean_reversion": {"lookback": 10, "rsi_period": 7, "bb_period": 10},
    "intraday_momentum": {"lookback_bars": 6, "rebalance_freq": 3, "n_long": 2, "n_short": 2},
    "intraday_trend": {"trend_window": 15, "fast_window": 5, "slow_window": 10},
}

ALL_PARAMS: dict[str, dict] = {}
ALL_PARAMS.update(_EQUITY_PARAMS)
ALL_PARAMS.update(_OPTIONS_PARAMS)
ALL_PARAMS.update(_MACRO_PARAMS)
ALL_PARAMS.update(_INTRADAY_PARAMS)

_STRATEGY_TYPE_MAP: dict[str, StrategyType] = {}
for name in _EQUITY_PARAMS:
    _STRATEGY_TYPE_MAP[name] = StrategyType.EQUITY
for name in _OPTIONS_PARAMS:
    _STRATEGY_TYPE_MAP[name] = StrategyType.OPTIONS
for name in _MACRO_PARAMS:
    _STRATEGY_TYPE_MAP[name] = StrategyType.MACRO
for name in _INTRADAY_PARAMS:
    _STRATEGY_TYPE_MAP[name] = StrategyType.EQUITY

# ── Data generation ────────────────────────────────────────────────────────────

N_BARS = 500
N_SYMBOLS = 10
RNG_SEED = 42


def make_data(n_bars: int = N_BARS, n_symbols: int = N_SYMBOLS,
              seed: int = RNG_SEED) -> dict:
    """Multi-symbol synthetic OHLCV data with all required fields."""
    rng = np.random.default_rng(seed)

    # Correlated random walk
    common = rng.normal(0.0003, 0.005, n_bars)
    specific = rng.normal(0.0005, 0.015, (n_bars, n_symbols))
    returns = specific + common[:, np.newaxis] * 0.3

    log_prices = np.zeros((n_bars, n_symbols))
    log_prices[0] = np.log(100.0)
    for i in range(1, n_bars):
        log_prices[i] = log_prices[i - 1] + returns[i]
    close = np.exp(log_prices)

    noise_hi = np.abs(rng.normal(0, 0.008, (n_bars, n_symbols)))
    noise_lo = np.abs(rng.normal(0, 0.008, (n_bars, n_symbols)))
    high = close * (1 + noise_hi)
    low = close * (1 - noise_lo)
    low = np.minimum(low, close)

    open_ = np.zeros_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.002, (n_bars - 1, n_symbols)))

    volume = rng.integers(500_000, 5_000_000, (n_bars, n_symbols)).astype(float)

    implied_vol = np.maximum(0.25 + rng.normal(0, 0.02, (n_bars, n_symbols)), 0.05)

    yields = np.maximum(0.03 + rng.normal(0, 0.005, (n_bars, n_symbols)), 0.001)

    # Earnings: ~quarterly events — set to 1.0 in 5-bar window before each event
    earnings_upcoming = np.zeros((n_bars, n_symbols))
    for j in range(n_symbols):
        for i in range(0, n_bars, 63):
            for k in range(1, 6):
                if i - k >= 0:
                    earnings_upcoming[i - k, j] = 1.0

    symbols = [f"A{j}" for j in range(n_symbols)]

    return {
        "close": close,
        "high": high,
        "low": low,
        "open": open_,
        "volume": volume,
        "implied_vol": implied_vol,
        "yields": yields,
        "earnings_upcoming": earnings_upcoming,
        "symbols": symbols,
    }


# ── Signal-based return computation ───────────────────────────────────────────


def compute_signal_returns(strat, data: dict, start: int, end: int) -> np.ndarray:
    """Compute bar-level signal-weighted returns over data[start:end).

    For each bar i in [start, end-1): calls strat.on_data() with data[:i+1],
    then computes weighted forward return using asset return from i to i+1.

    Returns array of length (end - start - 1). Empty if < 2 bars.
    """
    if end - start < 2:
        return np.array([], dtype=float)

    close = data["close"]
    symbols = data["symbols"]
    n_assets = len(symbols)

    # Forward asset returns over the evaluation window
    asset_rets = (close[start + 1: end] - close[start: end - 1]) / np.maximum(close[start: end - 1], 1e-10)

    bar_rets = np.zeros(end - start - 1)

    for idx, bar_i in enumerate(range(start, end - 1)):
        bar_data: dict = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v

        try:
            signals = strat.on_data(bar_data) or []
        except Exception:
            signals = []

        weighted_ret = 0.0
        total_weight = 0.0
        for s in signals:
            if s.direction != 0:
                try:
                    asset_idx = symbols.index(s.symbol)
                except (ValueError, IndexError):
                    continue
                weighted_ret += float(s.direction) * float(s.strength) * float(asset_rets[idx, asset_idx])
                total_weight += abs(float(s.strength))

        if total_weight > 1e-10:
            bar_rets[idx] = weighted_ret / total_weight

    return bar_rets


def estimate_turnover(strat, data: dict, start: int, end: int) -> float:
    """Estimate annual turnover from signal changes."""
    if end - start < 3:
        return 0.0

    prev_directions: dict[str, int] = {}
    total_churn = 0.0
    n_changes = 0

    for bar_i in range(start, end - 1):
        bar_data: dict = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v

        try:
            signals = strat.on_data(bar_data) or []
        except Exception:
            signals = []

        current: dict[str, int] = {}
        for s in signals:
            if s.direction != 0:
                current[s.symbol] = s.direction

        for sym, dir_ in current.items():
            prev = prev_directions.get(sym, 0)
            if prev != 0 and dir_ != prev:
                total_churn += 1.0
        for sym in prev_directions:
            if sym not in current and prev_directions[sym] != 0:
                total_churn += 1.0

        n_changes += 1
        prev_directions = current

    n_years = (end - start) / 252
    if n_years < 0.01 or n_changes < 1:
        return 0.0
    return total_churn / n_years


# ── Walk-forward analysis ─────────────────────────────────────────────────────


def analyze_strategy(name: str, data: dict, folds: list,
                     all_is_sharpes: list[float]) -> dict:
    """Run anchored walk-forward for a single strategy.

    Returns dict of all IS, OOS, and quality metrics.
    """
    params = ALL_PARAMS.get(name, {})
    stype = _STRATEGY_TYPE_MAP.get(name, StrategyType.EQUITY)
    cfg = StrategyConfig(name=name, strategy_type=stype, params=params)

    fold_is_sharpes: list[float] = []
    fold_oos_sharpes: list[float] = []
    all_oos_rets: list[np.ndarray] = []

    for fold in folds:
        try:
            strat = create(name, cfg)
            strat.initialize()

            # Training — feed bars to build state, collect IS returns
            is_rets = compute_signal_returns(strat, data, fold.train_start, fold.train_end)

            # Test — continue from trained state, collect OOS returns
            oos_rets = compute_signal_returns(strat, data, fold.test_start, fold.test_end)

            if len(is_rets) > 2:
                fold_is_sharpes.append(sharpe_ratio(is_rets, annualize=False))
            if len(oos_rets) > 2:
                sr = sharpe_ratio(oos_rets, annualize=False)
                fold_oos_sharpes.append(sr)
                all_oos_rets.append(oos_rets)

        except Exception as e:
            continue

    if not all_oos_rets:
        return {"name": name, "error": "No folds completed", "verdict": "DISCARD"}

    # Aggregate OOS returns across folds
    oos_returns = np.concatenate(all_oos_rets) if len(all_oos_rets) > 0 else np.array([])

    if len(oos_returns) < 3:
        return {"name": name, "error": "Insufficient OOS data", "verdict": "DISCARD"}

    # IS metrics
    mean_is_sr = float(np.mean(fold_is_sharpes)) if fold_is_sharpes else 0.0
    std_is_sr = float(np.std(fold_is_sharpes, ddof=1)) if len(fold_is_sharpes) > 1 else 0.0

    # OOS metrics
    oos_sr = sharpe_ratio(oos_returns)
    oos_sortino = sortino_ratio(oos_returns)
    oos_calmar = calmar_ratio(oos_returns)

    # PSR: probability that true OOS Sharpe > 0
    oos_psr = probabilistic_sharpe_ratio(oos_returns, benchmark_sr=0.0)

    # DSR: probability that OOS Sharpe exceeds what random search over N strategies yields
    oos_dsr = deflated_sharpe_ratio(all_is_sharpes, oos_sr, oos_returns) if all_is_sharpes else oos_psr

    # Degradation ratio (OOS / IS) — handle degenerate cases explicitly
    deg_ratio: float | None = None
    if abs(mean_is_sr) < 0.10:
        deg_ratio = None  # IS Sharpe near zero — ratio undefined
    elif mean_is_sr < 0 and oos_sr > 0:
        deg_ratio = oos_sr / mean_is_sr  # Will be negative (sign flip)
    else:
        deg_ratio = oos_sr / mean_is_sr

    # CAPM attribution (use equal-weight portfolio as market proxy)
    close = data["close"]
    ew_rets = np.mean(close[1:] / close[:-1] - 1.0, axis=1)
    # Align OOS returns — use bars corresponding to OOS periods
    oos_start = folds[0].test_start
    oos_end = folds[-1].test_end
    oos_market = ew_rets[oos_start:oos_end - 1]
    n_min = min(len(oos_returns), len(oos_market))
    if n_min > 5:
        capm = capm_attribution(oos_returns[:n_min], oos_market[:n_min])
        alpha = capm.alpha
        beta = capm.beta
        r_squared = capm.r_squared
    else:
        alpha = beta = r_squared = 0.0

    # Turnover (estimate on full sample)
    full_strat = create(name, cfg)
    full_strat.initialize()
    turnover = estimate_turnover(full_strat, data, 100, min(len(data["close"]), 400))

    # Full-sample metrics
    fs = compute_all_metrics(oos_returns)

    return {
        "name": name,
        "type": stype.value,
        "n_folds": len([f for f in fold_oos_sharpes if f is not None]),
        "is_sharpe": mean_is_sr,
        "is_sharpe_std": std_is_sr,
        "oos_sharpe": oos_sr,
        "oos_sortino": oos_sortino,
        "oos_calmar": oos_calmar,
        "oos_psr": oos_psr,
        "oos_dsr": oos_dsr,
        "deg_ratio": deg_ratio,
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "turnover": turnover,
        "max_dd": fs.get("max_drawdown", 0.0),
        "skew": fs.get("skewness", 0.0),
        "kurtosis": fs.get("kurtosis", 0.0),
        "n_oos_bars": len(oos_returns),
        "is_sharpes_raw": fold_is_sharpes,
        "oos_sharpes_raw": fold_oos_sharpes,
    }


# ── Verdict logic ──────────────────────────────────────────────────────────────


def determine_verdict(r: dict) -> str:
    """DEPLOY, PAPER, INVESTIGATE, or DISCARD based on OOS metrics.

    When deg_ratio is None (IS near zero), degradation is undefined and does
    not block PAPER/DEPLOY verdicts — the strategy should be judged on OOS
    metrics alone. Other flags (IS_OOS_SIGN_FLIP) may still warrant caution.
    """
    if r.get("error"):
        return "DISCARD"
    oos = r.get("oos_sharpe", 0.0)
    psr = r.get("oos_psr", 0.0)
    dsr = r.get("oos_dsr", 0.0)
    deg = r.get("deg_ratio")
    # When deg is None (IS near zero), treat degradation check as satisfied
    deg_ok = deg is None or deg > 0.5
    deg_ok_paper = deg is None or deg > 0.4

    if oos > 0.8 and dsr > 0.90 and psr > 0.90 and deg_ok:
        return "DEPLOY CANDIDATE"
    if oos > 0.5 and dsr > 0.5 and psr > 0.75 and deg_ok_paper:
        return "PAPER TRADE FIRST"
    if oos > 0.0:
        return "INVESTIGATE"
    return "DISCARD"


def get_flags(r: dict) -> str:
    """Generate quality flags."""
    flags: list[str] = []
    deg = r.get("deg_ratio")
    if deg is None:
        flags.append("UNDEFINED_IS_NEAR_ZERO")
    elif deg < 0.3:
        flags.append("HIGH_DEGRADATION")
    # Sign flip: IS negative but OOS positive
    is_sr = r.get("is_sharpe", 0.0)
    oos_sr = r.get("oos_sharpe", 0.0)
    if is_sr < 0 and oos_sr > 0:
        flags.append("IS_OOS_SIGN_FLIP")
    if r.get("oos_psr", 0) < 0.5:
        flags.append("LOW_PSR")
    if r.get("oos_dsr", 0) < 0.3:
        flags.append("SNOOPING_RISK")
    if r.get("turnover", 0) > 500:
        flags.append("HIGH_TURNOVER")
    if abs(r.get("beta", 0)) > 0.8:
        flags.append("HIGH_BETA")
    if r.get("max_dd", 0) < -0.3:
        flags.append("DEEP_DD")
    if r.get("oos_sharpe", 0) < -0.5:
        flags.append("NEGATIVE_OOS")
    return " | ".join(flags) if flags else "—"


# ── Phases 2-3: Run audit, produce master table ────────────────────────────────


def run_audit() -> list[dict]:
    """Phase 2: run walk-forward on all strategies. Returns result dicts."""
    print("=" * 72)
    print("  PHASE 2: WALK-FORWARD ANALYSIS — ALL 15 STRATEGIES")
    print("=" * 72)

    data = make_data()
    close = data["close"]
    n = close.shape[0]
    print(f"\n  Data: {n} bars x {len(data['symbols'])} symbols")

    # Walk-forward folds
    folds = anchored_walk_forward(n, initial_train=200, test_size=50, embargo_pct=0.02)
    print(f"  Walk-forward: {len(folds)} anchored folds (train=200->, test=50, embargo=1)")

    for i, f in enumerate(folds):
        print(f"    Fold {i}: train [{f.train_start}:{f.train_end}] "
              f"→ test [{f.test_start}:{f.test_end}]")

    # Collect IS Sharpes first for DSR computation
    all_names = list(ALL_PARAMS.keys())
    all_is_sharpes: list[float] = []

    print(f"\n  Computing IS Sharpes for DSR baseline...")
    for name in all_names:
        params = ALL_PARAMS[name]
        stype = _STRATEGY_TYPE_MAP[name]
        cfg = StrategyConfig(name=name, strategy_type=stype, params=params)
        try:
            strat = create(name, cfg)
            strat.initialize()
            for fold in folds[:1]:
                is_rets = compute_signal_returns(strat, data, fold.train_start, fold.train_end)
                if len(is_rets) > 2:
                    all_is_sharpes.append(sharpe_ratio(is_rets, annualize=False))
        except Exception:
            pass

    print(f"    Collected {len(all_is_sharpes)} IS Sharpe estimates")

    # Run full walk-forward for each strategy
    results: list[dict] = []
    total = len(all_names)

    for idx, name in enumerate(all_names):
        elapsed = time.perf_counter()
        r = analyze_strategy(name, data, folds, all_is_sharpes)
        elapsed = time.perf_counter() - elapsed
        r["verdict"] = determine_verdict(r)
        r["flags"] = get_flags(r)
        results.append(r)

        status = f"{r.get('oos_sharpe', 0.0):+.3f}" if "oos_sharpe" in r else "ERROR"
        print(f"  [{idx + 1}/{total}] {name:<30s} OOS Sharpe={status:>8s}  "
              f"verdict={r['verdict']:<18s}  ({elapsed:.1f}s)")

    # ── Phase 3: Master table ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  PHASE 3: MASTER AUDIT TABLE (sorted by OOS Sharpe)")
    print("=" * 72)

    results.sort(key=lambda x: x.get("oos_sharpe", -999), reverse=True)

    header = (
        f"  {'Rank':>4s} {'Strategy':<28s} {'Type':<9s} {'IS_SR':>7s} {'OOS_SR':>7s} "
        f"{'PSR':>5s} {'DSR':>5s} {'Deg':>5s} {'Alpha':>7s} {'Beta':>5s} "
        f"{'DD':>6s} {'Verdict':<18s}"
    )
    sep = "  " + "-" * len(header.expandtabs())
    print(f"\n{header}")
    print(sep)

    for rank, r in enumerate(results, 1):
        is_s = f"{r.get('is_sharpe', 0):+.3f}" if not r.get("error") else "FAIL"
        oos_s = f"{r.get('oos_sharpe', 0):+.3f}" if not r.get("error") else "FAIL"
        psr = f"{r.get('oos_psr', 0):.3f}" if not r.get("error") else "—"
        dsr = f"{r.get('oos_dsr', 0):.3f}" if not r.get("error") else "—"
        deg_raw = r.get("deg_ratio")
        deg = f"{deg_raw:.2f}" if deg_raw is not None else "UNDEF"
        alp = f"{r.get('alpha', 0):+.3f}" if not r.get("error") else "—"
        bet = f"{r.get('beta', 0):.3f}" if not r.get("error") else "—"
        dd = f"{r.get('max_dd', 0):.1%}" if not r.get("error") else "—"

        print(f"  {rank:>4d} {r['name']:<28s} {r.get('type', '?'):<9s} "
              f"{is_s:>7s} {oos_s:>7s} {psr:>5s} {dsr:>5s} {deg:>5s} "
              f"{alp:>7s} {bet:>5s} {dd:>6s} {r['verdict']:<18s}")

    print(sep)
    n_deploy = sum(1 for r in results if r["verdict"] == "DEPLOY CANDIDATE")
    n_paper = sum(1 for r in results if r["verdict"] == "PAPER TRADE FIRST")
    n_invest = sum(1 for r in results if r["verdict"] == "INVESTIGATE")
    n_discard = sum(1 for r in results if r["verdict"] == "DISCARD")
    print(f"\n  Summary: {n_deploy} DEPLOY | {n_paper} PAPER | {n_invest} INVESTIGATE | {n_discard} DISCARD")

    # ── Phase 4: Run engine waterfall for non-DISCARD ──────────────────────
    survivors = [r for r in results if r["verdict"] in ("DEPLOY CANDIDATE", "PAPER TRADE FIRST")]
    if survivors:
        run_waterfall(survivors, data)
    else:
        print("\n" + "=" * 72)
        print("  PHASE 4: ENGINE WATERFALL — skipped (no survivors)")
        print("=" * 72)

    # ── Phase 5: Failure analysis ──────────────────────────────────────────
    failures = [r for r in results if r["verdict"] in ("DISCARD", "INVESTIGATE")]
    analyze_failures(failures)

    # ── Phase 6: Portfolio analysis ────────────────────────────────────────
    deploy = [r for r in results if r["verdict"] in ("DEPLOY CANDIDATE", "PAPER TRADE FIRST")]
    if deploy:
        portfolio_analysis(deploy, data)
    else:
        print("\n" + "=" * 72)
        print("  PHASE 6: PORTFOLIO ANALYSIS — skipped (no survivors)")
        print("=" * 72)

    return results


# ── Phase 4: Engine waterfall ──────────────────────────────────────────────────


def run_waterfall(survivors: list[dict], data: dict) -> None:
    """Phase 4: impact decomposition via BacktestEngine + StrategyAdapter."""
    print("\n" + "=" * 72)
    print("  PHASE 4: IMPACT DECOMPOSITION (ENGINE WATERFALL)")
    print("=" * 72)

    from quantcore.backtest.engine import BacktestEngine
    from quantcore.backtest.financing import BorrowCostRegistry, FinancingConfig
    from quantcore.backtest.impact_config import MarketImpactConfig, VolumeFilterConfig
    from scripts.strategy_adapter import StrategyAdapter
    from quantcore.strategies.registry import get as get_strategy_class

    # Create single-symbol OHLCV data for the engine
    rng = np.random.default_rng(142)
    n_bars = 252
    rows = []
    price = 100.0
    for i in range(n_bars):
        from datetime import datetime, timedelta
        ts = datetime(2020, 1, 1) + timedelta(days=i)
        ret = rng.normal(0.0005, 0.015)
        price = max(price * (1 + ret), 10.0)
        o = price * (1 + rng.uniform(-0.003, 0.003))
        c = price * (1 + rng.uniform(-0.003, 0.003))
        o, c = max(o, 1.0), max(c, 1.0)
        h = max(o, c) * (1 + abs(rng.uniform(0.001, 0.008)))
        l = min(o, c) * (1 - abs(rng.uniform(0.001, 0.008)))
        rows.append({"symbol": "TEST", "timestamp": ts,
                     "open": round(o, 2), "high": round(h, 2),
                     "low": round(l, 2), "close": round(c, 2),
                     "volume": 2_000_000})
    engine_data = __import__("pandas").DataFrame(rows)

    impact_cfg = MarketImpactConfig(model_type="sqrt", eta=1.0, sigma=0.02, adv=2_000_000)
    vol_filter = VolumeFilterConfig(max_participation_rate=0.10)
    fin_cfg = FinancingConfig(
        borrow_cost_registry=BorrowCostRegistry(default_rate=0.05),
        margin_rate=0.065,
    )

    print(f"\n  {'Strategy':<30s} {'Raw SR':>7s} {'Imp SR':>7s} {'Fin SR':>7s} "
          f"{'Full SR':>7s} {'ImpCost':>9s} {'Borrow':>9s} {'Margin':>9s}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*9}")

    for sr in survivors:
        name = sr["name"]
        params = ALL_PARAMS.get(name, {})
        stype = _STRATEGY_TYPE_MAP.get(name, StrategyType.EQUITY)
        cfg = StrategyConfig(name=name, strategy_type=stype, params=params)
        try:
            cls = get_strategy_class(name)
        except KeyError:
            continue

        # 4 engine passes
        passes = [
            ("raw", BacktestEngine, {"initial_cash": 1_000_000.0, "slippage_bps": 5.0,
                                      "sizing_pct": 0.20}),
            ("impact", BacktestEngine, {"initial_cash": 1_000_000.0, "slippage_bps": 5.0,
                                         "sizing_pct": 0.20,
                                         "impact_config": impact_cfg,
                                         "volume_filter_config": vol_filter}),
            ("financing", BacktestEngine, {"initial_cash": 1_000_000.0, "slippage_bps": 5.0,
                                            "sizing_pct": 0.20,
                                            "financing_config": fin_cfg}),
            ("full", BacktestEngine, {"initial_cash": 1_000_000.0, "slippage_bps": 5.0,
                                       "sizing_pct": 0.20,
                                       "impact_config": impact_cfg,
                                       "volume_filter_config": vol_filter,
                                       "financing_config": fin_cfg}),
        ]

        # Single-symbol strategies
        n_sym = 1
        # Check if multi-symbol
        if name in ("equity_momentum", "equity_stat_arb", "macro_carry",
                     "macro_cta_trend", "macro_risk_parity"):
            n_sym = 6
            # Re-generate multi-symbol engine data sorted by (timestamp, symbol)
            rows_m = []
            rng2 = np.random.default_rng(142)
            prices = [100.0] * n_sym
            for i in range(n_bars):
                ts = datetime(2020, 1, 1) + timedelta(days=i)
                for sym_i in range(n_sym):
                    ret = rng2.normal(0.0005, 0.015)
                    prices[sym_i] = max(prices[sym_i] * (1 + ret), 10.0)
                    p = prices[sym_i]
                    o = p * (1 + rng2.uniform(-0.003, 0.003))
                    c = p * (1 + rng2.uniform(-0.003, 0.003))
                    o, c = max(o, 1.0), max(c, 1.0)
                    h = max(o, c) * (1 + abs(rng2.uniform(0.001, 0.008)))
                    l = min(o, c) * (1 - abs(rng2.uniform(0.001, 0.008)))
                    rows_m.append({"symbol": f"SYM{sym_i}", "timestamp": ts,
                                   "open": round(o, 2), "high": round(h, 2),
                                   "low": round(l, 2), "close": round(c, 2),
                                   "volume": 2_000_000})
            engine_data = __import__("pandas").DataFrame(rows_m)

        row = f"  {name:<30s}"
        raw_sr = fin_sr = imp_sr = full_sr = 0.0
        imp_cost_total = borrow_total = margin_total = 0.0

        for pass_name, eng_cls, kwargs in passes:
            try:
                adapt = StrategyAdapter(cls, cfg, n_symbols=n_sym)
                eng = eng_cls(strategy=adapt, **kwargs)
                eng.run(engine_data)
                rep = eng.summary()
                sr_val = rep.sharpe
                if pass_name == "raw":
                    raw_sr = sr_val
                elif pass_name == "impact":
                    imp_sr = sr_val
                    imp_cost_total = rep.total_market_impact_cost
                elif pass_name == "financing":
                    fin_sr = sr_val
                    borrow_total = rep.total_borrow_cost
                    margin_total = rep.total_margin_interest
                elif pass_name == "full":
                    full_sr = sr_val
            except Exception:
                sr_val = 0.0

        row += f" {raw_sr:>7.3f} {imp_sr:>7.3f} {fin_sr:>7.3f} {full_sr:>7.3f} "
        row += f" ${imp_cost_total:>7.2f} ${borrow_total:>7.2f} ${margin_total:>7.2f}"
        print(row)

    print(f"\n  Note: Multi-symbol strategies use {n_sym}-asset engine data.")


# ── Phase 5: Failure analysis ──────────────────────────────────────────────────


def analyze_failures(failures: list[dict]) -> None:
    """Phase 5: specific failure analysis for DISCARD and INVESTIGATE."""
    print("\n" + "=" * 72)
    print("  PHASE 5: FAILURE ANALYSIS — DISCARD / INVESTIGATE")
    print("=" * 72)

    if not failures:
        print("\n  No failures to analyze.")
        return

    for r in failures:
        name = r["name"]
        verdict = r["verdict"]
        oos = r.get("oos_sharpe", 0.0)
        psr = r.get("oos_psr", 0.0)
        dsr = r.get("oos_dsr", 0.0)
        flags = r.get("flags", "—")

        print(f"\n  [{verdict}] {name}")
        print(f"      OOS Sharpe: {oos:+.4f}   PSR: {psr:.4f}   DSR: {dsr:.4f}")
        print(f"      Flags: {flags}")

        reasons = []

        if r.get("error"):
            reasons.append(f"  ERROR: {r['error']}")
        if oos <= 0:
            reasons.append(f"  → OOS Sharpe <= 0 ({oos:+.4f}): strategy destroys value OOS")
        if psr < 0.5:
            reasons.append(f"  → PSR < 0.5: cannot distinguish strategy returns from noise")
        if dsr < 0.3:
            reasons.append(f"  → DSR < 0.3: high risk of data snooping / selection bias")
        deg_ratio = r.get("deg_ratio")
        if deg_ratio is not None and deg_ratio < 0.1 and oos > 0:
            reasons.append(f"  → Degradation ratio {deg_ratio:.2f}: IS performance collapses OOS")
        if r.get("max_dd", 0) < -0.4:
            reasons.append(f"  → Max drawdown {r['max_dd']:.1%}: excessive downside")
        if r.get("turnover", 0) > 1000:
            reasons.append(f"  → Annual turnover {r['turnover']:.0f}x: impractical for live trading")

        for reason in reasons:
            print(f"  {reason}")

        if not reasons and verdict == "INVESTIGATE":
            print("  → Marginal performance: positive OOS Sharpe but below deploy threshold.")
            print(f"    Needs parameter tuning, regime filtering, or ensemble approach.")


# ── Phase 6: Portfolio analysis ────────────────────────────────────────────────


def portfolio_analysis(survivors: list[dict], data: dict) -> None:
    """Phase 6: portfolio-level metrics for surviving strategies."""
    print("\n" + "=" * 72)
    print("  PHASE 6: PORTFOLIO ANALYSIS — SURVIVOR STRATEGIES")
    print("=" * 72)

    names = [r["name"] for r in survivors]
    print(f"\n  Survivors ({len(names)}): {', '.join(names)}")

    # Compute pairwise OOS return correlations
    from quantcore.strategies.base import StrategyConfig
    from quantcore.strategies.registry import create
    import numpy as np

    folds = anchored_walk_forward(N_BARS, initial_train=200, test_size=50, embargo_pct=0.02)

    strat_returns: dict[str, np.ndarray] = {}
    for r in survivors:
        name = r["name"]
        params = ALL_PARAMS.get(name, {})
        stype = _STRATEGY_TYPE_MAP.get(name, StrategyType.EQUITY)
        cfg = StrategyConfig(name=name, strategy_type=stype, params=params)

        all_oos: list[np.ndarray] = []
        for fold in folds:
            try:
                strat = create(name, cfg)
                strat.initialize()
                oos_rets = compute_signal_returns(strat, data, fold.test_start, fold.test_end)
                if len(oos_rets) > 2:
                    all_oos.append(oos_rets)
            except Exception:
                continue

        if all_oos:
            strat_returns[name] = np.concatenate(all_oos)

    if len(strat_returns) < 2:
        print("\n  Too few survivors for correlation analysis.")
        return

    # Correlation matrix
    names_list = list(strat_returns.keys())
    n_s = len(names_list)
    corr_matrix = np.eye(n_s)

    print(f"\n  Pairwise OOS return correlations:")
    print(f"  {'':<28s}", end="")
    for n2 in names_list:
        print(f" {n2:<20s}", end="")
    print()

    for i, n1 in enumerate(names_list):
        r1 = strat_returns[n1]
        n_min = len(r1)
        print(f"  {n1:<28s}", end="")
        for j, n2 in enumerate(names_list):
            if i == j:
                corr_matrix[i, j] = 1.0
                print(f" {'1.000':>8s}   ", end="")
            elif j < i:
                corr_matrix[i, j] = corr_matrix[j, i]
                print(f" {corr_matrix[i, j]:>8.3f}   ", end="")
            else:
                r2 = strat_returns[n2]
                m = min(n_min, len(r2))
                if m > 5:
                    c = float(np.corrcoef(r1[:m], r2[:m])[0, 1])
                    corr_matrix[i, j] = c
                    print(f" {c:>8.3f}   ", end="")
                else:
                    corr_matrix[i, j] = 0.0
                    print(f" {'0.000':>8s}   ", end="")
        print()

    avg_corr = (np.sum(np.abs(corr_matrix)) - n_s) / (n_s * (n_s - 1))
    print(f"\n  Mean absolute correlation: {avg_corr:.4f}")
    if avg_corr < 0.3:
        print("  → Low correlation: good diversification potential")
    elif avg_corr < 0.6:
        print("  → Moderate correlation: partial diversification")
    else:
        print("  → High correlation: strategies capture similar risk premia")

    # Equal-weight portfolio
    eq_portfolio = np.zeros_like(list(strat_returns.values())[0])
    min_len = min(len(v) for v in strat_returns.values())
    for v in strat_returns.values():
        eq_portfolio[:min_len] += v[:min_len] / n_s

    port_sr = sharpe_ratio(eq_portfolio)
    port_sortino = sortino_ratio(eq_portfolio)
    port_calmar = calmar_ratio(eq_portfolio)
    try:
        port_psr = probabilistic_sharpe_ratio(eq_portfolio, benchmark_sr=0.0)
    except Exception:
        port_psr = 0.5

    print(f"\n  Equal-Weight Portfolio (n={n_s} strategies):")
    print(f"    Sharpe: {port_sr:.4f}")
    print(f"    Sortino: {port_sortino:.4f}")
    print(f"    Calmar: {port_calmar:.4f}")
    print(f"    PSR: {port_psr:.4f}")

    # Kelly optimal sizing for portfolio
    annual_ret = float(np.mean(eq_portfolio)) * 252
    annual_vol = float(np.std(eq_portfolio, ddof=1)) * np.sqrt(252)
    if annual_vol > 1e-10:
        kelly_f = annual_ret / (annual_vol ** 2)
        kelly_pct = min(kelly_f, 1.0) * 100
        print(f"    Kelly allocation: {kelly_pct:.1f}% of capital")
        quarter_kelly = min(kelly_f * 0.25, 1.0) * 100
        print(f"    Quarter-Kelly (recommended): {quarter_kelly:.1f}% of capital")
    else:
        print("    Kelly: insufficient volatility for estimation")

    # Capacity estimate
    avg_turnover = np.mean([r.get("turnover", 0) for r in survivors])
    if avg_turnover > 0:
        print(f"    Avg strategy turnover: {avg_turnover:.0f}x / year")
        capacity = 50_000_000 / avg_turnover if avg_turnover > 1 else 50_000_000
        print(f"    Estimated capacity (10% market participation): ~${capacity:,.0f}")


# ── Phase 7: Final summary ─────────────────────────────────────────────────────


def print_final_summary(results: list[dict]) -> None:
    """Phase 7: final deliverable."""
    print("\n" + "=" * 72)
    print("  PHASE 7: AUDIT SUMMARY")
    print("=" * 72)

    deploy = [r for r in results if r["verdict"] == "DEPLOY CANDIDATE"]
    paper = [r for r in results if r["verdict"] == "PAPER TRADE FIRST"]
    invest = [r for r in results if r["verdict"] == "INVESTIGATE"]
    discard = [r for r in results if r["verdict"] == "DISCARD"]

    print(f"\n  Total strategies audited: {len(results)}")
    print(f"  DEPLOY CANDIDATE: {len(deploy)}")
    print(f"  PAPER TRADE FIRST: {len(paper)}")
    print(f"  INVESTIGATE: {len(invest)}")
    print(f"  DISCARD: {len(discard)}")

    if deploy:
        print(f"\n  DEPLOY CANDIDATES:")
        for r in deploy:
            print(f"    {r['name']:<30s} OOS Sharpe={r['oos_sharpe']:+.3f}  "
                  f"PSR={r['oos_psr']:.3f}  DSR={r['oos_dsr']:.3f}")

    if paper:
        print(f"\n  PAPER TRADE FIRST:")
        for r in paper:
            print(f"    {r['name']:<30s} OOS Sharpe={r['oos_sharpe']:+.3f}  "
                  f"PSR={r['oos_psr']:.3f}  Flags: {r['flags']}")

    if discard:
        print(f"\n  DISCARDED:")
        for r in discard:
            err = f" ({r.get('error', 'OOS Sharpe <= 0')})"
            print(f"    {r['name']:<30s} OOS Sharpe={r.get('oos_sharpe', 0):+.3f}{err}")

    print(f"\n  {'=' * 72}")
    print(f"  RECOMMENDATION")
    print(f"  {'=' * 72}")

    if deploy:
        print(f"\n  → Proceed to live deployment pipeline for {len(deploy)} candidate(s).")
        print(f"    Run paper trading validation for a minimum of 3 months.")
        print(f"    Monitor slippage, capacity, and regime sensitivity.")
    if paper:
        print(f"\n  → Begin paper trading for {len(paper)} strategy(ies).")
        print(f"    Collect 3-6 months of live OOS data before considering deployment.")
        print(f"    Address flagged issues ({' | '.join(set(r['flags'] for r in paper))}).")
    if invest:
        print(f"\n  → Review {len(invest)} INVESTIGATE strategy(ies) for parameter optimization.")
        print(f"    Consider regime filters, ensemble approaches, or revised entry logic.")
    if discard:
        print(f"\n  → {len(discard)} strategy(ies) discarded: destroy value OOS or fail to execute.")
        print(f"    Remove from active monitoring. Retain for reference only.")

    print(f"\n  {'=' * 72}")
    print(f"  AUDIT COMPLETE — {len(results)} strategies evaluated")
    print(f"  {'=' * 72}")


# ── Main ────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    results = run_audit()
    print_final_summary(results)
