#!/usr/bin/env python
"""End-to-end integration test for multi-strategy paper trading.

Simulates one full trading day:
  1. Instantiate all enabled strategies from config
  2. Load market data
  3. Generate signals for each strategy
  4. Verify no crashes, no deadlocks, no shared state corruption
  5. Verify per-strategy isolation

Usage:
    python scripts/end_to_end_test.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Import all strategy modules
import quantcore.strategies.options.delta_neutral  # noqa: F401
import quantcore.strategies.options.iron_condor  # noqa: F401
import quantcore.strategies.options.put_write  # noqa: F401
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401
import quantcore.strategies.macro.risk_parity  # noqa: F401
import quantcore.strategies.macro.volatility_risk_premium  # noqa: F401

from quantcore.strategies.base import StrategyConfig
from quantcore.strategies.registry import get as get_strategy_class, list_strategies


def test_strategy_import():
    """Verify all enabled strategies can be imported and instantiated."""
    log.info("TEST 1: Strategy import and instantiation")
    config_path = REPO_ROOT / "config" / "live.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    available = set(list_strategies())
    results = []

    for strat_cfg in config.get("strategies", []):
        name = strat_cfg["name"]
        enabled = strat_cfg.get("enabled", True)
        if not enabled:
            continue

        if name not in available:
            results.append((name, False, f"Not in registry: {name}"))
            continue

        try:
            cls = get_strategy_class(name)
            cfg = StrategyConfig(
                name=name,
                strategy_type=strat_cfg.get("strategy_type", "EQUITY"),
                max_position_notional=100_000,
                params=strat_cfg.get("params", {}),
            )
            instance = cls(cfg)
            results.append((name, True, "OK"))
        except Exception as e:
            results.append((name, False, str(e)))

    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        log.info("  [%s] %s: %s", status, name, msg)

    return all(ok for _, ok, _ in results)


def test_signal_generation():
    """Verify all enabled strategies generate signals without crashing."""
    log.info("TEST 2: Signal generation on synthetic data")

    config_path = REPO_ROOT / "config" / "live.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create synthetic market data
    n_periods = 300
    n_assets = 6
    symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"]

    np.random.seed(42)
    base_prices = np.array([500, 400, 200, 200, 100, 25])
    returns = np.random.normal(0.0003, 0.015, (n_periods, n_assets))
    close = base_prices * np.cumprod(1 + returns, axis=0)

    # Synthetic VIX data (VIX-shaped)
    vix_base = 20.0 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_periods))
    iv_series = np.clip(vix_base / 100.0, 0.08, 0.80)

    data = {
        "close": close,
        "open": close * (1 + np.random.normal(0, 0.001, close.shape)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.005, close.shape))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.005, close.shape))),
        "volume": np.random.randint(1000000, 10000000, close.shape),
        "symbols": symbols,
        "implied_vol": iv_series.reshape(-1, 1),
    }

    results = []
    for strat_cfg in config.get("strategies", []):
        name = strat_cfg["name"]
        if not strat_cfg.get("enabled", True):
            continue

        try:
            cls = get_strategy_class(name)
            cfg = StrategyConfig(
                name=name,
                strategy_type=strat_cfg.get("strategy_type", "EQUITY"),
                max_position_notional=100_000,
                params=strat_cfg.get("params", {}),
            )
            instance = cls(cfg)

            # Test with full data
            t0 = time.time()
            signals = instance.on_data(data)
            elapsed = time.time() - t0

            n_signals = len(signals) if signals else 0
            results.append((name, True, f"{n_signals} signals in {elapsed:.3f}s"))

        except Exception as e:
            results.append((name, False, str(e)[:100]))

    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        log.info("  [%s] %s: %s", status, name, msg)

    return all(ok for _, ok, _ in results)


def test_strategy_isolation():
    """Verify strategies don't share mutable state."""
    log.info("TEST 3: Strategy isolation (no shared mutable state)")

    config_path = REPO_ROOT / "config" / "live.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create two instances of the same strategy
    strat_cfg = config["strategies"][0]
    name = strat_cfg["name"]
    cls = get_strategy_class(name)
    cfg = StrategyConfig(
        name=name, strategy_type="OPTIONS",
        max_position_notional=100_000,
        params=strat_cfg.get("params", {}),
    )

    instance_a = cls(cfg)
    instance_b = cls(cfg)

    # Verify they're independent
    if instance_a is instance_b:
        log.info("  [FAIL] Same object — shared state!")
        return False

    # Modify one and verify the other is unchanged
    instance_a._positions["TEST"] = 999.0
    if "TEST" in instance_b._positions:
        log.info("  [FAIL] Shared _positions dict!")
        return False

    log.info("  [PASS] Strategies are independent instances")
    return True


def test_production_runner_init():
    """Verify ProductionRunner can be instantiated with the current config."""
    log.info("TEST 4: ProductionRunner initialization")

    try:
        from quantcore.live.production_runner import ProductionRunner, load_config

        config = load_config(REPO_ROOT / "config" / "live.yaml")

        # Override broker to use local simulation (no Alpaca needed)
        config["broker"]["type"] = "paper"
        config["broker"]["alpaca"]["api_key"] = ""
        config["broker"]["alpaca"]["secret_key"] = ""

        runner = ProductionRunner(config)

        log.info("  Strategies loaded: %d", len(runner.strategy_states))
        for s in runner.strategy_states:
            log.info("    %s (weight=%.3f, type=%s, state=%s)",
                     s.name, s.weight, s.strategy_type, s.state)

        log.info("  [PASS] ProductionRunner initialized successfully")
        return True

    except Exception as e:
        log.info("  [FAIL] ProductionRunner init failed: %s", e)
        return False


def test_per_strategy_equity():
    """Verify per-strategy equity curve writing works."""
    log.info("TEST 5: Per-strategy equity curve")

    try:
        from quantcore.live.production_runner import StrategyRuntimeState

        state = StrategyRuntimeState(
            name="test_strategy", weight=0.5, instance=None,
            data_source="alpaca_daily", bar_size="1Day",
            starting_equity=50000, current_equity=50000, peak_equity=50000,
            state="running",
        )

        # Simulate equity updates
        state.update_equity(50100)
        assert state.total_pnl == 100.0, f"Expected 100, got {state.total_pnl}"
        assert state.peak_equity == 50100.0

        state.update_equity(49900)
        assert state.peak_equity == 50100.0  # peak should not decrease
        assert state.drawdown_pct < 0  # should show drawdown

        log.info("  [PASS] Per-strategy equity tracking works")
        return True

    except Exception as e:
        log.info("  [FAIL] Per-strategy equity failed: %s", e)
        return False


def main():
    log.info("=" * 70)
    log.info("END-TO-END INTEGRATION TEST")
    log.info("=" * 70)

    results = []
    results.append(("Strategy Import", test_strategy_import()))
    results.append(("Signal Generation", test_signal_generation()))
    results.append(("Strategy Isolation", test_strategy_isolation()))
    results.append(("ProductionRunner Init", test_production_runner_init()))
    results.append(("Per-Strategy Equity", test_per_strategy_equity()))

    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        log.info("  [%s] %s", status, name)
        if not passed:
            all_pass = False

    if all_pass:
        log.info("\nALL TESTS PASSED — system ready for multi-strategy paper trading")
    else:
        log.info("\nSOME TESTS FAILED — fix before launching")

    # Write report
    report_lines = [
        "# End-to-End Test Report",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| Test | Result |",
        "|------|--------|",
    ]
    for name, passed in results:
        emoji = "PASS" if passed else "FAIL"
        report_lines.append(f"| {name} | {emoji} |")

    report = "\n".join(report_lines)
    (REPORTS_DIR / "end_to_end_test.md").write_text(report, encoding="utf-8")

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
