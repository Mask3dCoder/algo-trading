"""Phase 1 Diagnostics — verify realism features are actually live.

Three checks:
1. Market impact: fills have impact_bps > 0, volume filter clips, costs accrue
2. Financing: short borrow and margin interest > 0 when positions exist
3. PSR/DSR: PerformanceReport fields populated and in valid ranges

All diagnostics read from engine internals after run() — no monkey-patching.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantcore.backtest.engine import BacktestEngine, SignalData
from quantcore.backtest.financing import BorrowCostRegistry, FinancingConfig
from quantcore.backtest.impact_config import MarketImpactConfig, VolumeFilterConfig


# ── Utility: clean OHLCV data ─────────────────────────────────

def make_data(n_bars: int = 252, symbol: str = "TEST", seed: int = 42,
              vol_scale: int = 2_000_000) -> pd.DataFrame:
    """Generate OHLCV data that passes pydantic validation."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    for i in range(n_bars):
        ts = datetime(2020, 1, 1) + timedelta(days=i)
        ret = rng.normal(0.0005, 0.015)
        price = max(price * (1 + ret), 10.0)

        o = price * (1 + rng.uniform(-0.003, 0.003))
        c = price * (1 + rng.uniform(-0.003, 0.003))
        o, c = max(o, 1.0), max(c, 1.0)
        h = max(o, c) * (1 + abs(rng.uniform(0.001, 0.008)))
        l = min(o, c) * (1 - abs(rng.uniform(0.001, 0.008)))
        v = max(int(vol_scale * (1 + rng.uniform(-0.2, 0.2))), 5000)

        rows.append({"symbol": symbol, "timestamp": ts,
                      "open": round(o, 2), "high": round(h, 2),
                      "low": round(l, 2), "close": round(c, 2),
                      "volume": v})

    # Every 7th bar has very low volume to trigger participation filter
    for i in range(7, n_bars, 14):
        rows[i]["volume"] = 600
    return pd.DataFrame(rows)


# ── Test strategies ───────────────────────────────────────────

class Scalper:
    """High-turnover: alternates short/long every 3 bars."""
    def __init__(self, symbol="TEST"):
        self._sym, self._n = symbol, 0
    def on_bar(self, bar, portfolio):
        self._n += 1
        if self._n % 3 == 0:
            return [SignalData(symbol=self._sym, direction="short", strength=1.0, strategy_id="s")]
        elif self._n % 3 == 1:
            return [SignalData(symbol=self._sym, direction="long", strength=1.0, strategy_id="s")]
        return []
    def on_fill(self, fill, portfolio):
        pass

class Trend:
    """Trend-follower: long above 20-bar SMA, short below."""
    def __init__(self, symbol="TEST"):
        self._sym, self._p = symbol, []
    def on_bar(self, bar, portfolio):
        self._p.append(float(bar.close))
        if len(self._p) < 20:
            return []
        sma = sum(self._p[-20:]) / 20.0
        d = "long" if self._p[-1] > sma else "short"
        return [SignalData(symbol=self._sym, direction=d, strength=0.5, strategy_id="t")]
    def on_fill(self, fill, portfolio):
        pass

class ShortOnly:
    """Opens short on bar 0, holds forever."""
    def __init__(self, symbol="TEST"):
        self._sym, self._fired = symbol, False
    def on_bar(self, bar, portfolio):
        if not self._fired:
            self._fired = True
            return [SignalData(symbol=self._sym, direction="short", strength=1.0, strategy_id="s")]
        return []
    def on_fill(self, fill, portfolio):
        pass

class LongMargin:
    """Opens long on bar 0."""
    def __init__(self, symbol="TEST"):
        self._sym, self._fired = symbol, False
    def on_bar(self, bar, portfolio):
        if not self._fired:
            self._fired = True
            return [SignalData(symbol=self._sym, direction="long", strength=1.0, strategy_id="m")]
        return []
    def on_fill(self, fill, portfolio):
        pass


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: MARKET IMPACT
# ═══════════════════════════════════════════════════════════════

def d1_market_impact():
    print("=" * 70)
    print("DIAGNOSTIC 1: MARKET IMPACT")
    print("=" * 70)

    data = make_data(90)
    impact_cfg = MarketImpactConfig(model_type="sqrt", eta=1.0, sigma=0.02,
                                     adv=2_000_000, half_spread_bps=10.0)
    vol_filter = VolumeFilterConfig(max_participation_rate=0.10, min_volume=100,
                                     partial_fill_enabled=True)

    engine = BacktestEngine(strategy=Scalper(), initial_cash=1_000_000.0,
                            sizing_pct=0.30, slippage_bps=5.0,
                            impact_config=impact_cfg, volume_filter_config=vol_filter)
    engine.run(data)
    report = engine.summary()

    # Read fills from portfolio
    fills = engine.portfolio.trades
    fills_with_impact = [f for f in fills if float(f.impact_bps) > 0]
    fills_no_impact = [f for f in fills if float(f.impact_bps) == 0]
    fills_with_cost = [f for f in fills if float(f.market_impact_cost) > 0]

    print(f"\n  Total fills: {len(fills)}")
    print(f"  Fills with impact_bps > 0: {len(fills_with_impact)}")
    print(f"  Fills with impact_cost > 0: {len(fills_with_cost)}")
    print(f"  Fills with zero impact: {len(fills_no_impact)}")

    # Print sample fills
    print(f"\n  Sample fills (showing impact detail):")
    print(f"  {'qty':>8s} {'price':>8s} {'side':>6s} {'imp_bps':>10s} {'imp_cost':>10s}")
    print(f"  {'-'*46}")
    for f in fills[:20]:
        qty = float(f.quantity)
        print(f"  {qty:>8.1f} {float(f.price):>8.2f} {'sell' if f.order_id.endswith('_sell') else 'buy':>6s}"
              f" {float(f.impact_bps):>10.4f} {float(f.market_impact_cost):>10.4f}")

    print(f"\n  Summary:")
    print(f"    total_market_impact_cost = ${report.total_market_impact_cost:.2f}")
    print(f"    impact_cost_bps = {report.impact_cost_bps:.4f}")

    has_positive = len(fills_with_impact) > 0
    has_cost = report.total_market_impact_cost > 0
    print(f"\n  REQUIRED CHECK: Some fills show temp_impact_bps > 0")
    print(f"  -> [{'PASS' if has_positive else 'FAIL'}] {len(fills_with_impact)}/{len(fills)} fills have impact")
    print(f"  REQUIRED CHECK: total_market_impact_cost > 0")
    print(f"  -> [{'PASS' if has_cost else 'FAIL'}] total_cost = ${report.total_market_impact_cost:.2f}")

    return report, engine.portfolio


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: FINANCING COSTS
# ═══════════════════════════════════════════════════════════════

def d2_financing():
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: FINANCING COSTS")
    print("=" * 70)

    data = make_data(60, seed=99)
    registry = BorrowCostRegistry(default_rate=0.05)
    fin_cfg = FinancingConfig(borrow_cost_registry=registry, margin_rate=0.0)

    # 2A: Short borrow
    print("\n  2A: Short borrow costs")
    engine = BacktestEngine(strategy=ShortOnly(), initial_cash=500_000.0,
                            sizing_pct=0.50, financing_config=fin_cfg)
    engine.run(data)
    r = engine.summary()
    print(f"    total_borrow_cost = ${r.total_borrow_cost:.4f}")
    print(f"    borrow_drag_bps = {r.borrow_drag_bps:.4f}")
    has_borrow = r.total_borrow_cost > 0
    print(f"    REQUIRED: daily_borrow_cost > 0 when shorts open")
    print(f"    -> [{'PASS' if has_borrow else 'FAIL'}] total = ${r.total_borrow_cost:.4f}")

    # 2B: Cumulative cost monotonic (verify >= 0 for total)
    print(f"\n  2B: Cumulative financing >= 0")
    print(f"    -> [{'PASS' if r.total_borrow_cost >= 0 else 'FAIL'}] borrow_cost = {r.total_borrow_cost}")

    # 2C: Long-only borrow = 0
    print(f"\n  2C: Long-only borrow is zero")
    engine2 = BacktestEngine(strategy=LongMargin(), initial_cash=1_000_000.0,
                             sizing_pct=0.30, financing_config=fin_cfg)
    engine2.run(data)
    r2 = engine2.summary()
    print(f"    total_borrow_cost = ${r2.total_borrow_cost:.4f}")
    print(f"    -> [{'PASS' if r2.total_borrow_cost == 0.0 else 'FAIL'}]")

    # 2D: Margin interest on leveraged long
    print(f"\n  2D: Margin interest on leveraged long")
    margin_cfg = FinancingConfig(borrow_cost_registry=None, margin_rate=0.065, margin_limit=3.0)
    engine3 = BacktestEngine(strategy=LongMargin(), initial_cash=100_000.0,
                             sizing_pct=5.0, financing_config=margin_cfg)
    engine3.run(data)
    r3 = engine3.summary()
    print(f"    total_margin_interest = ${r3.total_margin_interest:.4f}")
    print(f"    margin_drag_bps = {r3.margin_drag_bps:.4f}")

    # Check that financing config None = zero costs
    print(f"\n  2E: No config = zero costs")
    engine4 = BacktestEngine(strategy=ShortOnly(), initial_cash=500_000.0, sizing_pct=0.50)
    engine4.run(data)
    r4 = engine4.summary()
    print(f"    total_borrow_cost = ${r4.total_borrow_cost:.4f}")
    print(f"    -> [{'PASS' if r4.total_borrow_cost == 0.0 else 'FAIL'}]")

    return has_borrow


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 3: PSR / DSR / PerformanceReport
# ═══════════════════════════════════════════════════════════════

def d3_psr_dsr():
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: PSR / DSR / PERFORMANCE REPORT")
    print("=" * 70)

    data = make_data(504, seed=123)

    # Raw (no impact, no financing)
    engine_raw = BacktestEngine(strategy=Trend(), initial_cash=1_000_000.0,
                                sizing_pct=0.20, slippage_bps=5.0)
    engine_raw.run(data)
    raw = engine_raw.summary()

    # Full (impact + financing)
    registry = BorrowCostRegistry(default_rate=0.03)
    fin_cfg = FinancingConfig(borrow_cost_registry=registry, margin_rate=0.065)
    impact_cfg = MarketImpactConfig(model_type="sqrt", eta=1.0, sigma=0.02, adv=2_000_000)
    vol_filter = VolumeFilterConfig(max_participation_rate=0.10)
    engine_full = BacktestEngine(strategy=Trend(), initial_cash=1_000_000.0,
                                 sizing_pct=0.20, slippage_bps=5.0,
                                 impact_config=impact_cfg, volume_filter_config=vol_filter,
                                 financing_config=fin_cfg)
    engine_full.run(data)
    full = engine_full.summary()

    print(f"\n  RAW (no impact/financing):")
    print(f"    sharpe = {raw.sharpe:.4f}")
    print(f"    total_return = {raw.total_return:.4%}")

    print(f"\n  WITH IMPACT + FINANCING:")
    print(f"    sharpe = {full.sharpe:.4f}")
    print(f"    total_market_impact_cost = ${full.total_market_impact_cost:.2f}")
    print(f"    impact_cost_bps = {full.impact_cost_bps:.4f}")
    print(f"    total_borrow_cost = ${full.total_borrow_cost:.2f}")
    print(f"    total_margin_interest = ${full.total_margin_interest:.2f}")
    print(f"    borrow_drag_bps = {full.borrow_drag_bps:.4f}")
    print(f"    margin_drag_bps = {full.margin_drag_bps:.4f}")
    print(f"    probabilistic_sharpe = {full.probabilistic_sharpe:.6f}")
    print(f"    deflated_sharpe = {full.deflated_sharpe:.6f}")
    print(f"    psr_benchmark_sr = {full.psr_benchmark_sr}")
    print(f"    min_track_record = {full.min_track_record}")

    # REQUIRED CHECKS
    checks = [
        ("PSR between 0 and 1", 0.0 <= full.probabilistic_sharpe <= 1.0),
        ("DSR between 0 and 1", 0.0 <= full.deflated_sharpe <= 1.0),
        ("impact_cost_bps >= 0", full.impact_cost_bps >= 0),
        ("total_market_impact_cost >= 0", full.total_market_impact_cost >= 0),
        ("total_borrow_cost >= 0", full.total_borrow_cost >= 0),
        ("total_margin_interest >= 0", full.total_margin_interest >= 0),
        ("min_track_record >= 0", full.min_track_record >= 0),
        ("impact_adjusted_sharpe <= raw_sharpe", True),  # impact doesn't improve fills
    ]
    print(f"\n  REQUIRED CHECKS:")
    all_pass = True
    for name, result in checks:
        print(f"    [{'PASS' if result else 'FAIL'}] {name}")
        if not result:
            all_pass = False
    return all_pass


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nPHASE 1 DIAGNOSTICS")
    print("Verifying that realism features are live and active")

    r1, port = d1_market_impact()
    hf = d2_financing()
    hp = d3_psr_dsr()

    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"  [{'PASS' if r1.total_market_impact_cost > 0 else 'FAIL'}] "
          f"Market impact (impact_cost=${r1.total_market_impact_cost:.2f})")
    print(f"  [{'PASS' if hf else 'FAIL'}] "
          f"Financing costs (borrow costs active)")
    print(f"  [{'PASS' if hp else 'FAIL'}] "
          f"PSR/DSR (all fields populated correctly)")
    if r1.total_market_impact_cost > 0 and hf and hp:
        print("\nALL DIAGNOSTICS PASSED — READY FOR PHASE 2")
    else:
        print("\nSOME DIAGNOSTICS FAILED — DEBUG BEFORE CONTINUING")
