from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from quantcore.backtest.engine import BacktestEngine, SignalData
from quantcore.core.types import Fill


class _SnapshotCanary:
    def __init__(self) -> None:
        self.snapshot_shapes: list[tuple[object, tuple[str, ...]]] = []
        self.fill_times: list[object] = []
        self.signal_times: list[object] = []

    def on_snapshot(self, snapshot, portfolio):
        self.snapshot_shapes.append((snapshot.timestamp, snapshot.symbols))
        self.signal_times.append(snapshot.timestamp)
        if snapshot.timestamp.date() == datetime(2020, 1, 3).date():
            return [
                SignalData(
                    symbol="AAPL",
                    direction="long",
                    strength=1.0,
                    strategy_id="snapshot_canary",
                    notional_size=1_000.0,
                )
            ]
        return []

    def on_fill(self, fill: Fill, portfolio) -> None:
        self.fill_times.append(fill.timestamp)


def _ohlcv(timestamp: datetime, symbol: str, close: float = 100.0) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": 1_000_000,
    }


def test_snapshot_loop_groups_cross_sectional_bars_before_strategy() -> None:
    start = datetime(2020, 1, 2)
    data = pd.DataFrame(
        [
            _ohlcv(start, "AAPL", 100.0),
            _ohlcv(start, "MSFT", 50.0),
            _ohlcv(start + timedelta(days=1), "AAPL", 101.0),
            _ohlcv(start + timedelta(days=1), "MSFT", 51.0),
        ]
    )

    strategy = _SnapshotCanary()
    engine = BacktestEngine(strategy=strategy, initial_cash=100_000.0)
    engine.run(data)

    assert strategy.snapshot_shapes == [
        (start, ("AAPL", "MSFT")),
        (start + timedelta(days=1), ("AAPL", "MSFT")),
    ]
    assert strategy.signal_times == [start, start + timedelta(days=1)]


def test_snapshot_loop_queues_strategy_orders_until_next_timestamp() -> None:
    start = datetime(2020, 1, 2)
    data = pd.DataFrame(
        [
            _ohlcv(start, "AAPL", 100.0),
            _ohlcv(start + timedelta(days=1), "AAPL", 101.0),
            _ohlcv(start + timedelta(days=2), "AAPL", 102.0),
        ]
    )

    strategy = _SnapshotCanary()
    engine = BacktestEngine(strategy=strategy, initial_cash=100_000.0)
    engine.run(data)

    assert strategy.fill_times[0] == start + timedelta(days=2)
    assert all(fill_time != start + timedelta(days=1) for fill_time in strategy.fill_times)
