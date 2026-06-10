from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).parents[2] / "scripts" / "strategy_risk_dashboard.py"
spec = importlib.util.spec_from_file_location("strategy_risk_dashboard", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)  # type: ignore[attr-defined]


def test_strategy_daily_attribution_uses_inferred_weights(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    equity_path = tmp_path / "equity.csv"
    events_path.write_text(
        "\n".join(
            [
                '{"event":"fill","strategy":"equity_stat_arb","symbol":"AAPL","side":"buy","qty":1,"notional":100}',
                '{"event":"fill","strategy":"macro_carry","symbol":"TLT","side":"buy","qty":1,"notional":300}',
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": ["2026-06-09", "2026-06-10"],
            "total_equity": [1000.0, 1020.0],
        }
    ).to_csv(equity_path, index=False)

    events = module.load_events(events_path)
    equity = module.load_equity(equity_path)
    attribution = module.strategy_daily_attribution(events, equity)

    assert set(attribution["strategy"]) == {"equity_stat_arb", "macro_carry"}
    assert attribution.loc[attribution["strategy"] == "equity_stat_arb", "allocation_weight"].iloc[0] == 0.25
    assert attribution.loc[attribution["strategy"] == "macro_carry", "allocation_weight"].iloc[0] == 0.75


def test_strategy_attribution_export_writes_csv(tmp_path: Path) -> None:
    attribution = pd.DataFrame(
        [
            {
                "date": "2026-06-09",
                "strategy": "equity_stat_arb",
                "ret": 0.01,
                "pnl": 10.0,
                "allocation_weight": 0.5,
            }
        ]
    )
    output_path = tmp_path / "attribution.csv"

    module.write_attribution_csv(attribution, output_path)

    exported = pd.read_csv(output_path)
    assert exported.equals(attribution)


def test_strategy_metrics_detects_kill_switch_breach() -> None:
    metrics = module.strategy_metrics(
        events=pd.DataFrame(
            [
                {
                    "event": "fill",
                    "strategy": "equity_stat_arb",
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 1,
                    "notional": 100,
                }
            ]
        ),
        equity=pd.DataFrame(
            {
                "date": ["2026-06-09", "2026-06-10"],
                "total_equity": [100.0, 80.0],
                "ret": [0.0, -0.20],
            }
        ),
    )

    assert not metrics.empty
    assert bool(metrics.loc[metrics["strategy"] == "equity_stat_arb", "kill_switch"].iloc[0]) is True
