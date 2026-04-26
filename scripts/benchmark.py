"""Performance benchmarking for QuantCore hot paths.

Covers: backtest engine, signal generation, OMS operations.
Run with: python scripts/benchmark.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Ensure quantcore is importable when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def generate_ohlcv_data(n_bars: int = 50_000, n_symbols: int = 1) -> pd.DataFrame:
    """Generate realistic OHLCV bars as a random walk with realistic vol."""
    rows = []
    start = datetime(2020, 1, 2, tzinfo=UTC)
    for sym_idx in range(n_symbols):
        symbol = f"STK{sym_idx:04d}"
        close = 100.0
        for i in range(n_bars):
            ret = RNG.normal(0.0001, 0.015)
            close = close * (1.0 + ret)
            intraday_range = close * abs(RNG.normal(0, 0.005))
            open_p = close * (1.0 + RNG.normal(0, 0.002))
            high = max(open_p, close) + abs(RNG.normal(0, intraday_range * 0.3))
            low = min(open_p, close) - abs(RNG.normal(0, intraday_range * 0.3))
            volume = int(abs(RNG.normal(1_000_000, 500_000)))
            ts = start + timedelta(minutes=i)
            rows.append({
                "symbol": symbol,
                "timestamp": ts,
                "open": float(open_p),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": volume,
            })
    return pd.DataFrame(rows)


def generate_price_series(n: int = 10_000) -> np.ndarray:
    """Generate a random walk price series."""
    returns = RNG.normal(0.0001, 0.015, n)
    prices = 100.0 * np.cumprod(1.0 + returns)
    return prices.astype(np.float64)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def benchmark_backtest_engine(data: pd.DataFrame, name: str) -> dict:
    """Profile the backtest engine run() method."""
    from quantcore.backtest.engine import BacktestEngine, PortfolioState
    from quantcore.core.types import OHLCV, Direction, Signal

    class BuyAndHoldStrategy:
        def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list:
            if len(portfolio.trades) == 0:
                return [Signal(
                    name="bnh_entry", symbol=bar.symbol, direction=Direction.LONG,
                    strength=1.0, confidence=1.0,
                )]
            return []

        def on_fill(self, fill, portfolio):
            pass

    engine = BacktestEngine(strategy=BuyAndHoldStrategy(), initial_cash=100_000.0)

    t0 = time.perf_counter()
    engine.run(data)
    elapsed = time.perf_counter() - t0

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    engine2 = BacktestEngine(strategy=BuyAndHoldStrategy(), initial_cash=100_000.0)
    engine2.run(data)
    profiler.disable()

    buf = io.StringIO()
    ps = pstats.Stats(profiler, stream=buf).sort_stats("cumtime")
    ps.print_stats(20)

    return {
        "name": name,
        "n_bars": len(data),
        "wall_seconds": elapsed,
        "bars_per_second": len(data) / elapsed if elapsed > 0 else 0,
        "profile_top20": buf.getvalue(),
    }


def benchmark_signal_generation(prices: np.ndarray, name: str) -> dict:
    """Profile technical indicator computation."""
    from quantcore.signals.technical.mean_reversion import (
        bollinger_bands,
        rsi,
        stochastic,
    )
    from quantcore.signals.technical.trend import (
        adx,
        breakout_signal,
        ema,
        hull_ma,
        macd,
        sma,
        supertrend,
        zscore_normalize,
    )
    from quantcore.signals.technical.volatility import atr as atr_fn

    high = prices * (1.0 + np.abs(RNG.normal(0, 0.005, len(prices))))
    low = prices * (1.0 - np.abs(RNG.normal(0, 0.005, len(prices))))
    close = prices

    profiler = cProfile.Profile()
    profiler.enable()

    t0 = time.perf_counter()
    # Compute all indicators
    sma(close, 20)
    ema(close, 20)
    hull_ma(close, 20)
    macd(close)
    adx(high, low, close, 14)
    supertrend(high, low, close)
    breakout_signal(high, low, close)
    zscore_normalize(close)
    bollinger_bands(close, 20)
    rsi(close, 14)
    stochastic(high, low, close)
    try:
        atr_fn(high, low, close, 14)
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    profiler.disable()

    buf = io.StringIO()
    ps = pstats.Stats(profiler, stream=buf).sort_stats("cumtime")
    ps.print_stats(20)

    return {
        "name": name,
        "n_points": len(prices),
        "wall_seconds": elapsed,
        "signals_per_second": len(prices) / elapsed if elapsed > 0 else 0,
        "profile_top20": buf.getvalue(),
    }


def benchmark_oms(n_orders: int = 10_000, name: str = "OMS") -> dict:
    """Profile OMS order lifecycle."""
    from quantcore.core.types import Order, OrderType, Side
    from quantcore.execution.oms import OrderManager

    oms = OrderManager()
    orders = [
        Order(
            id=f"ord_{i}", symbol=f"STK{i % 10:04d}",
            side=Side.BUY if i % 3 != 0 else Side.SELL,
            type=OrderType.MARKET,
            quantity=Decimal(str(RNG.integers(1, 1000))),
        )
        for i in range(n_orders)
    ]

    t0 = time.perf_counter()
    for i, order in enumerate(orders):
        oms.submit(order)
        if i % 2 == 0:
            try:
                oms.fill(order.id, Decimal(str(order.quantity)), Decimal("100.0"))
            except Exception:
                pass
        elif i % 3 == 0:
            try:
                oms.cancel(order.id)
            except Exception:
                pass
    elapsed = time.perf_counter() - t0

    return {
        "name": name,
        "n_orders": n_orders,
        "wall_seconds": elapsed,
        "orders_per_second": n_orders / elapsed if elapsed > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Kill switch benchmark
# ---------------------------------------------------------------------------


def benchmark_kill_switch(n_iterations: int = 100) -> dict:
    """Measure time from trigger to halt."""
    from quantcore.live.trader import LiveTrader, TraderConfig

    times = []
    for _ in range(n_iterations):
        trader = LiveTrader(config=TraderConfig())
        t0 = time.perf_counter()
        trader._emergency_flatten()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    return {
        "name": "Kill Switch",
        "n_iterations": n_iterations,
        "mean_seconds": float(np.mean(times)),
        "min_seconds": float(np.min(times)),
        "max_seconds": float(np.max(times)),
        "p99_seconds": float(np.percentile(times, 99)),
        "target_seconds": 1.0,
        "passes": float(np.mean(times)) < 1.0,
    }


# ---------------------------------------------------------------------------
# Alert latency benchmark
# ---------------------------------------------------------------------------


def benchmark_alert_latency(n_fires: int = 50) -> dict:
    """Measure time from condition evaluation to alert emission."""
    from quantcore.dashboard.alerts import AlertEngine, build_default_alert_rules

    times = []
    for _ in range(n_fires):
        engine = AlertEngine()
        for rule in build_default_alert_rules():
            engine.add_rule(rule)
        # State dict triggering drawdown alert (drawdown_pct < -10%)
        state = {
            "drawdown_pct": -15.0,
            "var_breach": False,
            "circuit_breaker_active": False,
            "max_concentration": 0.12,
            "correlation_break": False,
            "daily_loss_pct": 1.2,
            "data_stale_seconds": 5.0,
            "details": "test",
        }
        t0 = time.perf_counter()
        engine.evaluate(state)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    return {
        "name": "Alert Engine",
        "n_fires": n_fires,
        "mean_seconds": float(np.mean(times)),
        "min_seconds": float(np.min(times)),
        "max_seconds": float(np.max(times)),
        "p99_seconds": float(np.percentile(times, 99)),
        "target_seconds": 5.0,
        "passes": float(np.mean(times)) < 5.0,
    }


# ---------------------------------------------------------------------------
# Dashboard load benchmark
# ---------------------------------------------------------------------------


def benchmark_dashboard_load(n_trials: int = 10) -> dict:
    """Measure dashboard app creation + first render time."""
    from quantcore.dashboard.app import create_app

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        app = create_app()
        # Simulate first page render: serialize layout to string
        if app is not None:
            _ = str(app.layout)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    return {
        "name": "Dashboard Load",
        "n_trials": n_trials,
        "mean_seconds": float(np.mean(times)),
        "min_seconds": float(np.min(times)),
        "max_seconds": float(np.max(times)),
        "p99_seconds": float(np.percentile(times, 99)),
        "target_seconds": 3.0,
        "passes": float(np.mean(times)) < 3.0,
    }


# ---------------------------------------------------------------------------
# Feed handler throughput benchmark
# ---------------------------------------------------------------------------


def benchmark_feed_handler(n_messages: int = 10_000) -> dict:
    """Measure feed handler message processing throughput per second."""
    from quantcore.data.feeds.base import AlpacaFeed

    events = []
    for i in range(n_messages):
        events.append({
            "T": "t", "S": "AAPL", "t": 1700000000000000 + i * 1000,
            "p": 150.0 + (i % 100) * 0.01, "s": 100, "x": "N",
        })


    feed = AlpacaFeed(api_key="test", secret_key="test")
    feed._connected = True  # bypass websocket

    import asyncio

    async def _run():
        t0 = time.perf_counter()
        for event in events:
            await feed._process_event(event)
        return time.perf_counter() - t0

    elapsed = asyncio.run(_run())

    return {
        "name": "Feed Handler (_process_event)",
        "n_messages": n_messages,
        "wall_seconds": elapsed,
        "msgs_per_second": n_messages / elapsed if elapsed > 0 else 0,
        "target_msgs_per_sec": 100_000,
        "passes": (n_messages / elapsed) > 100_000 if elapsed > 0 else False,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("QuantCore Performance Benchmarks")
    print("=" * 70)

    # --- Backtest Engine ---
    print("\n[1/7] Benchmarking backtest engine (10K bars)...")
    data = generate_ohlcv_data(n_bars=10_000)
    bt_result = benchmark_backtest_engine(data, "Backtest Engine (10K bars)")
    print(f"  Wall time: {bt_result['wall_seconds']:.2f}s")
    print(f"  Bars/sec:  {bt_result['bars_per_second']:.0f}")

    # --- Signal Generation ---
    print("\n[2/7] Benchmarking signal generation (50K points)...")
    prices = generate_price_series(50_000)
    sig_result = benchmark_signal_generation(prices, "Signal Generation (50K)")
    print(f"  Wall time: {sig_result['wall_seconds']:.2f}s")
    print(f"  Points/sec: {sig_result['signals_per_second']:.0f}")

    # --- OMS ---
    print("\n[3/7] Benchmarking OMS (10K orders)...")
    oms_result = benchmark_oms(10_000)
    print(f"  Wall time: {oms_result['wall_seconds']:.3f}s")
    print(f"  Orders/sec: {oms_result['orders_per_second']:.0f}")

    # --- Kill Switch ---
    print("\n[4/7] Benchmarking kill switch latency...")
    ks_result = benchmark_kill_switch(100)
    print(f"  Mean: {ks_result['mean_seconds']*1000:.2f}ms")
    print(f"  P99:  {ks_result['p99_seconds']*1000:.2f}ms")
    print(f"  Passes <1s target: {ks_result['passes']}")

    # --- Alerts ---
    print("\n[5/7] Benchmarking alert latency...")
    alert_result = benchmark_alert_latency(50)
    print(f"  Mean: {alert_result['mean_seconds']*1000:.2f}ms")
    print(f"  P99:  {alert_result['p99_seconds']*1000:.2f}ms")
    print(f"  Passes <5s target: {alert_result['passes']}")

    # --- Dashboard ---
    print("\n[6/7] Benchmarking dashboard load time...")
    dash_result = benchmark_dashboard_load(10)
    print(f"  Mean: {dash_result['mean_seconds']*1000:.0f}ms")
    print(f"  P99:  {dash_result['p99_seconds']*1000:.0f}ms")
    print(f"  Passes <3s target: {dash_result['passes']}")

    # --- Feed Handler ---
    print("\n[7/7] Benchmarking feed handler throughput (10K msgs)...")
    feed_result = benchmark_feed_handler(10_000)
    print(f"  Wall time: {feed_result['wall_seconds']:.3f}s")
    print(f"  Msgs/sec:  {feed_result['msgs_per_second']:.0f}")
    print(f"  Passes >100K msgs/sec target: {feed_result['passes']}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_results = {
        "backtest": bt_result,
        "signals": sig_result,
        "oms": oms_result,
        "kill_switch": ks_result,
        "alerts": alert_result,
        "dashboard": dash_result,
        "feed": feed_result,
    }
    print(f"  Backtest:  {bt_result['bars_per_second']:.0f} bars/sec")
    print(f"  Signals:   {sig_result['signals_per_second']:.0f} points/sec")
    print(f"  OMS:       {oms_result['orders_per_second']:.0f} orders/sec")
    print(f"  Kill switch: {ks_result['mean_seconds']*1000:.1f}ms (target <1000ms)")
    print(f"  Alerts:      {alert_result['mean_seconds']*1000:.1f}ms (target <5000ms)")
    print(f"  Dashboard:   {dash_result['mean_seconds']*1000:.0f}ms (target <3000ms)")
    print(f"  Feed:        {feed_result['msgs_per_second']:.0f} msgs/sec (target >100K)")

    # Save profiles
    out = Path("scripts/bench_results.txt")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for key in ["backtest", "signals"]:
            f.write(f"\n{'='*70}\n{key.upper()} PROFILE\n{'='*70}\n")
            f.write(all_results[key].get("profile_top20", "N/A"))
        f.write(f"\nKill switch mean: {ks_result['mean_seconds']*1000:.2f}ms\n")
        f.write(f"Alert mean: {alert_result['mean_seconds']*1000:.2f}ms\n")
        f.write(f"Dashboard mean: {dash_result['mean_seconds']*1000:.0f}ms\n")
        f.write(f"Feed throughput: {feed_result['msgs_per_second']:.0f} msgs/sec\n")
    print(f"\nProfiles saved to {out}")


if __name__ == "__main__":
    main()
