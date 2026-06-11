#!/usr/bin/env python3
"""Research validation harness for Atlas RAMFA."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantcore.backtest.walk_forward import rolling_walk_forward
from quantcore.risk.metrics import sharpe_ratio
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.equity.atlas_ramfa import AtlasRamfaStrategy


def _synthetic_data(n_bars: int, n_assets: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    market_ret = rng.normal(0.00025, 0.010, n_bars)
    market_close = 100.0 * np.exp(np.cumsum(market_ret))
    betas = rng.normal(1.0, 0.20, n_assets)
    idio_vol = rng.uniform(0.008, 0.035, n_assets)
    drift = rng.normal(0.00005, 0.00020, n_assets)
    asset_ret = market_ret[:, None] * betas[None, :] + rng.normal(
        drift[None, :], idio_vol[None, :], (n_bars, n_assets)
    )
    close = 100.0 * np.exp(np.cumsum(asset_ret, axis=0))
    volume = rng.lognormal(13.8, 0.35, size=(n_bars, n_assets))
    volume *= 1.0 + 0.15 * np.maximum(market_ret[:, None], 0.0)
    spread_bps = rng.uniform(3.0, 22.0, size=(n_bars, n_assets))
    base_iv = 0.20 + 0.35 * np.std(asset_ret[-63:], axis=0)
    iv_noise = rng.normal(0.0, 0.025, size=(n_bars, n_assets))
    implied_vol = base_iv[np.newaxis, :] + iv_noise
    implied_vol = np.clip(implied_vol, 0.08, 0.80)
    symbols = np.array([f"ATLAS_{idx:03d}" for idx in range(n_assets)], dtype=object)
    sector = np.array([f"sector_{idx % 5}" for idx in range(n_assets)], dtype=object)
    book_value = close * rng.uniform(0.55, 1.35, size=(n_bars, n_assets))
    earnings = close * rng.uniform(0.02, 0.08, size=(n_bars, n_assets))
    cash_flow = close * rng.uniform(0.01, 0.05, size=(n_bars, n_assets))
    enterprise_value = close * rng.uniform(0.80, 2.20, size=(n_bars, n_assets))
    adv_dollar = volume * close * 21.0
    float_value = close * rng.uniform(0.10, 0.90, size=(n_bars, n_assets))
    institutional_ownership = rng.uniform(20.0, 90.0, size=(n_bars, n_assets))
    days_to_liquidate = rng.uniform(0.5, 8.0, size=(n_bars, n_assets))

    rolling_ret = np.zeros_like(market_ret)
    rolling_ret[21:] = market_ret[21:] / market_ret[:-21] - 1.0
    rolling_vol = np.zeros_like(market_ret)
    for idx in range(21, n_bars):
        rolling_vol[idx] = float(np.std(market_ret[idx - 21 : idx], ddof=1) * math.sqrt(252.0))
    regime_score = 55.0 + 25.0 * np.tanh(rolling_ret / 0.08) - 10.0 * rolling_vol

    return {
        "close": close,
        "volume": volume,
        "spread_bps": spread_bps,
        "implied_vol": implied_vol,
        "regime_score": np.clip(regime_score, 0.0, 100.0),
        "symbols": symbols,
        "sector": sector,
        "earnings": earnings,
        "book_value": book_value,
        "cash_flow": cash_flow,
        "enterprise_value": enterprise_value,
        "adv_dollar": adv_dollar,
        "float_value": float_value,
        "institutional_ownership": institutional_ownership,
        "days_to_liquidate": days_to_liquidate,
        "market_close": market_close,
    }


def _slice(data: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    return {
        key: value[start:end] if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }


def _params() -> dict[str, float | int | str | bool]:
    return {
        "rebalance_freq": 1,
        "lookback_days": 252,
        "skip_days": 21,
        "vol_lookback": 63,
        "n_long": 8,
        "n_short": 8,
        "min_liquidity_score": 0.0,
        "max_gross_exposure": 1.5,
        "min_net_exposure": -0.25,
        "max_net_exposure": 0.5,
        "single_name_max_pct": 0.02,
        "use_regime_filter": True,
        "min_regime_score": 55.0,
        "regime_exposure_multiplier": True,
        "max_regime_multiplier": 1.0,
        "min_regime_multiplier": 0.15,
    }


def _strategy(portfolio_value: float) -> AtlasRamfaStrategy:
    cfg = StrategyConfig(
        name="atlas_ramfa_research_validation",
        strategy_type=StrategyType.EQUITY,
        max_position_notional=portfolio_value,
        max_leverage=1.5,
        risk_budget_pct=0.02,
        params=_params(),
    )
    strategy = AtlasRamfaStrategy(cfg)
    strategy.initialize()
    return strategy


def _weights(signals: list[Any], portfolio_value: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for signal in signals:
        if signal.direction == 0 or signal.size <= 0:
            continue
        out[str(signal.symbol)] = float(signal.direction) * float(signal.size) / portfolio_value
    return out


def _metrics(returns: NDArray[np.float64], equity: NDArray[np.float64]) -> dict[str, Any]:
    returns = np.asarray(returns, dtype=float)
    equity = np.asarray(equity, dtype=float)
    if len(returns) == 0:
        return {
            "n_periods": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "volatility": 0.0,
            "sharpe": float("nan"),
            "max_drawdown": 0.0,
        }
    total_return = float(equity[-1] / equity[0] - 1.0) if equity[0] > 0 else 0.0
    n_years = len(returns) / 252.0
    annual_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    volatility = float(np.std(returns, ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return {
        "n_periods": int(len(returns)),
        "total_return": total_return,
        "annual_return": float(annual_return),
        "volatility": volatility,
        "sharpe": float(sharpe_ratio(returns)) if len(returns) > 1 else float("nan"),
        "max_drawdown": float(np.max(drawdown)) if len(drawdown) else 0.0,
    }

def _run_backtest(
    data: dict[str, Any],
    portfolio_value: float,
    cost_bps: float,
    borrow_bps: float,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
) -> dict[str, Any]:
    strategy = _strategy(portfolio_value)
    close = np.asarray(data["close"], dtype=float)
    symbol_index = {str(symbol): idx for idx, symbol in enumerate(data["symbols"])}
    prev: dict[str, float] = {}
    returns: list[float] = []
    equity = [portfolio_value]
    turnover_total = 0.0
    transaction_cost_total = 0.0
    borrow_cost_total = 0.0
    signal_count = 0
    gross_values: list[float] = []
    net_values: list[float] = []
    position_counts: list[int] = []

    for idx in range(train_start, test_end - 1):
        signals = strategy.on_data(_slice(data, train_start, idx + 1))
        signal_count += sum(1 for sig in signals if sig.direction != 0 and sig.size > 0)
        if idx < test_start:
            prev = _weights(signals, portfolio_value)
            continue

        target = _weights(signals, portfolio_value)
        turnover = 0.5 * sum(
            abs(target.get(sym, 0.0) - prev.get(sym, 0.0))
            for sym in set(target) | set(prev)
        )
        gross = sum(abs(weight) for weight in target.values())
        net = sum(target.values())
        short_notional = sum(-weight for weight in target.values() if weight < 0.0) * portfolio_value
        next_returns = (close[idx + 1] - close[idx]) / np.maximum(close[idx], 1e-10)
        pnl = sum(
            weight * float(next_returns[symbol_index[symbol]])
            for symbol, weight in target.items()
            if symbol in symbol_index
        )
        transaction_cost = turnover * portfolio_value * cost_bps / 10_000.0
        borrow_cost = short_notional * borrow_bps / 10_000.0 / 252.0
        daily_return = (pnl * portfolio_value - transaction_cost - borrow_cost) / portfolio_value
        returns.append(float(daily_return))
        equity.append(equity[-1] * (1.0 + daily_return))
        turnover_total += turnover
        transaction_cost_total += transaction_cost
        borrow_cost_total += borrow_cost
        gross_values.append(gross)
        net_values.append(net)
        position_counts.append(len(target))
        prev = target

    metrics = _metrics(np.asarray(returns, dtype=float), np.asarray(equity, dtype=float))
    metrics.update(
        {
            "signal_count": signal_count,
            "turnover": turnover_total,
            "transaction_cost": transaction_cost_total,
            "borrow_cost": borrow_cost_total,
            "max_gross_exposure": max(gross_values) if gross_values else 0.0,
            "max_net_exposure": max(abs(x) for x in net_values) if net_values else 0.0,
            "mean_gross_exposure": float(np.mean(gross_values)) if gross_values else 0.0,
            "mean_net_exposure": float(np.mean(net_values)) if net_values else 0.0,
            "max_position_count": max(position_counts) if position_counts else 0,
            "returns": returns,
            "equity": equity,
        }
    )
    return metrics


def _run_walk_forward(
    data: dict[str, Any],
    portfolio_value: float,
    cost_bps: float,
    borrow_bps: float,
    train_size: int,
    test_size: int,
    embargo_pct: float,
) -> list[dict[str, Any]]:
    n_bars = int(np.asarray(data["close"]).shape[0])
    folds = rolling_walk_forward(
        n_bars, train_size=train_size, test_size=test_size, embargo_pct=embargo_pct
    )
    results: list[dict[str, Any]] = []
    for fold_idx, fold in enumerate(folds):
        result = _run_backtest(
            data=data,
            portfolio_value=portfolio_value,
            cost_bps=cost_bps,
            borrow_bps=borrow_bps,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
        )
        results.append(
            {
                "fold_index": fold_idx,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **result,
            }
        )
    return results


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _quarantine_checks(
    paper_config: Path,
    live_config: Path,
    strategy_status: Path,
) -> dict[str, Any]:
    paper = _load_yaml(paper_config)
    live = _load_yaml(live_config)
    status = _load_yaml(strategy_status)
    paper_names = {item.get("name") for item in paper.get("strategies", [])}
    live_names = {item.get("name") for item in live.get("strategies", [])}
    status_entry = status.get("strategies", {}).get("atlas_ramfa", {})
    return {
        "atlas_ramfa_in_paper_config": "atlas_ramfa" in paper_names,
        "atlas_ramfa_in_live_config": "atlas_ramfa" in live_names,
        "atlas_ramfa_status": status_entry.get("phase"),
        "quarantined": (
            "atlas_ramfa" not in paper_names
            and "atlas_ramfa" not in live_names
            and str(status_entry.get("phase", "")).upper() == "RESEARCH"
        ),
    }


def _paper_readiness(paper_config_path: Path, quarantine: dict[str, Any]) -> dict[str, Any]:
    config = _load_toml(paper_config_path)
    return {
        "paper_readiness_config_present": paper_config_path.exists(),
        "paper_readiness_config_name": config.get("strategy", {}).get("name"),
        "paper_readiness_config_status": config.get("strategy", {}).get("status"),
        "quarantine_passed": bool(quarantine.get("quarantined")),
        "ready_for_paper": False,
        "notes": [
            "Research-only readiness snapshot; not wired into config/paper.yaml or config/live.yaml.",
            "Paper activation remains blocked until point-in-time, walk-forward, transaction-cost, and model-risk gates pass.",
        ],
    }


def _model_risk_review(review_path: Path) -> dict[str, Any]:
    return {
        "model_risk_review_present": review_path.exists(),
        "review_path": str(review_path),
        "key_risks": [
            "Forward-looking bias if any future data leaks into signal generation.",
            "Survivorship bias if the synthetic or production universe is not point-in-time reconstructed.",
            "Liquidity and borrow constraints may invalidate the long/short exposure assumptions.",
            "Regime filter can suppress signals in adverse markets, creating clustering risk.",
            "IC weighting is unstable with short histories or low cross-sectional dispersion.",
        ],
    }

def _transaction_cost_sensitivity(
    data: dict[str, Any],
    portfolio_value: float,
    borrow_bps: float,
    train_size: int,
    test_size: int,
    embargo_pct: float,
) -> list[dict[str, Any]]:
    scenarios = [0.0, 5.0, 10.0, 20.0, 40.0]
    rows: list[dict[str, Any]] = []
    for cost_bps in scenarios:
        folds = _run_walk_forward(
            data=data,
            portfolio_value=portfolio_value,
            cost_bps=cost_bps,
            borrow_bps=borrow_bps,
            train_size=train_size,
            test_size=test_size,
            embargo_pct=embargo_pct,
        )
        oos_returns = (
            np.concatenate([np.asarray(item["returns"], dtype=float) for item in folds])
            if folds
            else np.array([])
        )
        rows.append(
            {
                "cost_bps": cost_bps,
                "oos_sharpe": float(sharpe_ratio(oos_returns)) if len(oos_returns) > 1 else float("nan"),
                "avg_turnover": float(np.mean([item["turnover"] for item in folds])) if folds else 0.0,
                "avg_transaction_cost": (
                    float(np.mean([item["transaction_cost"] for item in folds])) if folds else 0.0
                ),
                "avg_borrow_cost": float(np.mean([item["borrow_cost"] for item in folds])) if folds else 0.0,
            }
        )
    return rows


def _run_validation(args: argparse.Namespace) -> dict[str, Any]:
    data = _synthetic_data(args.n_bars, args.n_assets, args.seed)
    portfolio_value = args.portfolio_value
    pit = _run_backtest(
        data=data,
        portfolio_value=portfolio_value,
        cost_bps=args.cost_bps,
        borrow_bps=args.borrow_bps,
        train_start=0,
        train_end=args.train_size,
        test_start=args.train_size,
        test_end=min(args.n_bars, args.train_size + args.test_size),
    )
    folds = _run_walk_forward(
        data=data,
        portfolio_value=portfolio_value,
        cost_bps=args.cost_bps,
        borrow_bps=args.borrow_bps,
        train_size=args.train_size,
        test_size=args.test_size,
        embargo_pct=args.embargo_pct,
    )
    quarantine = _quarantine_checks(
        paper_config=REPO_ROOT / "config" / "paper.yaml",
        live_config=REPO_ROOT / "config" / "live.yaml",
        strategy_status=REPO_ROOT / "strategy_status.yaml",
    )
    paper_readiness = _paper_readiness(REPO_ROOT / args.paper_readiness_config, quarantine)
    model_risk = _model_risk_review(REPO_ROOT / args.model_risk_review)
    tc_rows = _transaction_cost_sensitivity(
        data=data,
        portfolio_value=portfolio_value,
        borrow_bps=args.borrow_bps,
        train_size=args.train_size,
        test_size=args.test_size,
        embargo_pct=args.embargo_pct,
    )
    oos_returns = (
        np.concatenate([np.asarray(item["returns"], dtype=float) for item in folds])
        if folds
        else np.array([])
    )
    oos_equity = (
        np.concatenate([np.asarray(item["equity"], dtype=float)[1:] for item in folds])
        if folds
        else np.array([1.0])
    )
    peak = np.maximum.accumulate(oos_equity) if len(oos_equity) else np.array([1.0])
    max_dd = (
        float(np.max((peak - oos_equity) / np.where(peak > 0, peak, 1.0)))
        if len(oos_equity)
        else 0.0
    )
    oos_sharpe = sharpe_ratio(oos_returns) if len(oos_returns) > 1 else float("nan")
    return {
        "strategy": "atlas_ramfa",
        "status": "RESEARCH",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "point_in_time_backtest": {
            "uses_expanding_slices": True,
            "lagged_execution": True,
            "min_required_history": args.train_size + args.test_size,
            "result": pit,
        },
        "walk_forward_validation": {
            "folds": folds,
            "aggregate_oos_sharpe": float(oos_sharpe),
            "aggregate_max_drawdown": max_dd,
            "passes_thresholds": bool(
                (not math.isnan(oos_sharpe))
                and oos_sharpe >= args.min_oos_sharpe
                and max_dd <= args.max_drawdown
            ),
        },
        "transaction_cost_calibration": tc_rows,
        "paper_trading_readiness": paper_readiness,
        "model_risk_review": model_risk,
        "production_quarantine": quarantine,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Atlas RAMFA research strategy")
    parser.add_argument("--n-bars", type=int, default=1260)
    parser.add_argument("--n-assets", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--portfolio-value", type=float, default=1_000_000.0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--borrow-bps", type=float, default=200.0)
    parser.add_argument("--train-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=63)
    parser.add_argument("--embargo-pct", type=float, default=0.01)
    parser.add_argument("--min-oos-sharpe", type=float, default=0.3)
    parser.add_argument("--max-drawdown", type=float, default=0.10)
    parser.add_argument(
        "--paper-readiness-config",
        default="research/atlas_ramfa/paper_readiness.toml",
    )
    parser.add_argument(
        "--model-risk-review",
        default="research/atlas_ramfa/model_risk_review.md",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = _run_validation(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_threshold and not report["walk_forward_validation"]["passes_thresholds"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
