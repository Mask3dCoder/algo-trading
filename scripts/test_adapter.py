"""Quick test: verify StrategyAdapter works with BacktestEngine."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from quantcore.backtest.engine import BacktestEngine
from quantcore.backtest.financing import BorrowCostRegistry, FinancingConfig
from quantcore.backtest.impact_config import MarketImpactConfig, VolumeFilterConfig
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
import sys
sys.path.insert(0, ".")
from scripts.strategy_adapter import StrategyAdapter  # noqa: E402


def make_single_data(n_bars=400):
    rng = np.random.default_rng(42)
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
        rows.append({"symbol": "TEST", "timestamp": ts,
                     "open": round(o,2), "high": round(h,2), "low": round(l,2),
                     "close": round(c,2), "volume": 2_000_000})
    return pd.DataFrame(rows)


data = make_single_data(400)

cfg = StrategyConfig(name="test_mr", strategy_type=StrategyType.EQUITY,
                     params={"rsi_oversold": 30, "rsi_overbought": 70})
adapter = StrategyAdapter(MeanReversionStrategy, cfg, n_symbols=1)

impact = MarketImpactConfig(model_type="sqrt", eta=1.0, sigma=0.02, adv=2_000_000)
vol_filter = VolumeFilterConfig(max_participation_rate=0.10)
fin = FinancingConfig(borrow_cost_registry=BorrowCostRegistry(default_rate=0.05), margin_rate=0.065)

engine = BacktestEngine(strategy=adapter, initial_cash=1_000_000.0,
                        sizing_pct=0.20, slippage_bps=5.0,
                        impact_config=impact, volume_filter_config=vol_filter,
                        financing_config=fin)
engine.run(data)
report = engine.summary()

print(f"  MeanReversion audit:")
print(f"    total_return={report.total_return:.4%}")
print(f"    sharpe={report.sharpe:.4f}")
print(f"    impact_cost=${report.total_market_impact_cost:.2f}")
print(f"    borrow_cost=${report.total_borrow_cost:.2f}")
print(f"    margin_interest=${report.total_margin_interest:.2f}")
print(f"    n_trades={report.n_trades}")
print(f"    win_rate={report.win_rate:.4%}")
print(f"    PSR={report.probabilistic_sharpe:.4f}")
print(f"    DSR={report.deflated_sharpe:.4f}")
print(f"    min_track_record={report.min_track_record}")
print("SUCCESS: Adapter works!")
