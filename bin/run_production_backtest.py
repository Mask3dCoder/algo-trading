"""Historical Backtest Harness — drives PaperTradingRunner.run_once(date).

Reads config/historical_backtest.yaml, ingests synthetic macro data,
instantiates the production runner, and chronologically walks each
trading day.
"""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from quantcore.testing.synthetic_universe import download_spy_vix_history, build_chain_history, make_snapshot
from quantcore.live.paper_trading.runner import PaperTradingConfig, PaperTradingRunner

CONFIG = Path(__file__).resolve().parents[1] / "config" / "historical_backtest.yaml"


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def main(start: str | None = None, end: str | None = None, force: bool = False) -> None:
    cfg_dict = _load_yaml(CONFIG)
    h = cfg_dict["harness"]
    start_dt = date.fromisoformat(start) if start else date.fromisoformat(h["start_date"])
    end_dt = date.fromisoformat(end) if end else date.fromisoformat(h["end_date"])

    history = download_spy_vix_history(start_dt, end_dt, force_redownload=force)
    chains = build_chain_history(history, dte=h.get("dte", 45))

    class FakeFeed:
        def __init__(self, hist: pd.DataFrame, lookback: int = 252) -> None:
            self._h = hist
            self._lb = lookback

        def get_daily_snapshot(self, as_of, lookback_days: int = 252) -> dict:
            snap = make_snapshot(self._h, as_of, lookback=lookback_days)
            # Inject VIX so the runner can read macro kill-switch levels
            window = self._h.loc[:as_of].tail(lookback_days)
            vix = window["VIX_close"].fillna(15.0).to_numpy().reshape(-1, 1)
            snap["close"] = np.concatenate([snap["close"], vix], axis=1)
            snap["symbols"] = ["SPY", "VIX"]
            return snap

    ptc = PaperTradingConfig.from_dict(cfg_dict)
    runner = PaperTradingRunner(config=ptc, data_feed=FakeFeed(history))

    trading_days = [d.date() for d in pd.bdate_range(start_dt, end_dt) if d.date() in chains]
    total = len(trading_days)
    print(f"Running {total} trading days …")
    for i, d in enumerate(trading_days):
        out = runner.run_once(d)
        if out.get("status") != "ok":
            print(f"  {d}: {out}")
        if (i + 1) % 50 == 0:
            print(f"  …{i + 1}/{total} days ({(i+1)/total:.0%})")
        if runner.risk_controller.is_halted:
            print(f"\n[*** HALTED ***] {d} — risk controller tripped.")
            break

    out_dir = Path("reports/paper_trading")
    print(f"\nDone.  Reports → {out_dir.resolve()}")
    print(f"  daily_log.csv      → portfolio equity / drawdown / Sharpe")
    print(f"  tca_performance.csv → slippage & spread execution costs")
    print(f"  delta_telemetry.csv → delta drift tracking")
    print(f"  state/risk_controller_state.json → kill-switch audit trail")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="YYYY-MM-DD (defaults from config)")
    ap.add_argument("--end", help="YYYY-MM-DD (defaults from config)")
    ap.add_argument("--force-redownload", action="store_true", dest="force")
    args = ap.parse_args()
    main(start=args.start, end=args.end, force=args.force)
