"""Honest single-strategy backtest for PutWriteVRPStrategy (2023-2024).

Loads SPY historical data via yfinance, VIX from the local fred/vix.parquet,
and runs the Phase-2 PutWrite strategy with BSM-based OptionFillSimulator.

No backtest leakage: each day the strategy sees only data up to and
including that day.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from quantcore.backtest.simulation import OptionFillSimulator
from quantcore.core.types import Fill, Order, OrderType, Side
from quantcore.strategies.base import StrategyConfig
from quantcore.strategies.options.put_write import PutWriteVRPStrategy

_log = logging.getLogger(__name__)


@dataclass
class OptionPosition:
    """Backtest position tracker for one short-put leg."""
    contract_id: str
    contracts: int
    strike: float
    dte: int
    entry_credit: float  # per-share premium received
    iv_at_entry: float
    spot_at_entry: float
    entry_costs: float  # commission paid


@dataclass
class BacktestState:
    cash: float = 1_000_000.0
    positions: dict[str, OptionPosition] = field(default_factory=dict)
    equity_curve: list[tuple[date, float]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)


def _load_spy_data(start: date, end: date) -> pd.DataFrame:
    """Download SPY daily OHLCV from yfinance."""
    _log.info("Downloading SPY %s to %s ...", start, end)
    df = yf.download("SPY", start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("No SPY data returned from yfinance")
    # yfinance returns multi-index columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    return df


def _load_vix_data(start: date, end: date) -> pd.Series:
    """Load VIX from local FRED parquet (daily, 2015-2026)."""
    vix_path = Path("quantcore/data/vol/vix.parquet")
    if not vix_path.exists():
        # Fallback: download via yfinance ^VIX
        _log.warning("Local VIX parquet not found, downloading via yfinance")
        df = yf.download("^VIX", start=start, end=end, progress=False)
        return df["Close"].rename("vix")
    df = pd.read_parquet(vix_path)
    # The parquet has a 'vix' column and Date index
    s = df["vix"].loc[start:end]
    return s


def _simulate_option_fill_from_signal(
    sim: OptionFillSimulator,
    signal: Any,
    spot: float,
) -> dict | None:
    """Run OptionFillSimulator for a TradeSignal's metadata."""
    m = signal.metadata
    strike = float(m.get("strike", 0.0))
    iv = float(m.get("iv", 0.0))
    dte = float(m.get("dte", 0.0))
    contracts = int(m.get("contracts", 0))
    if contracts <= 0 or strike <= 0 or iv <= 0 or dte <= 0:
        return None
    return sim.simulate_option_fill(
        underlying_price=spot,
        strike=strike,
        iv=iv,
        dte=dte,
        r=float(m.get("risk_free_rate", 0.045)),
        is_call=False,
        side=Side.SELL if signal.direction == -1 else Side.BUY,
        contracts=contracts,
        spread_pct=float(m.get("bid_ask_spread_pct", 0.05)),
        commission_per_contract=float(m.get("commission_per_contract", 0.65)),
    )


def run_backtest(start: date, end: date) -> pd.DataFrame:
    """Run the put-write backtest and return an equity-curve DataFrame."""
    # --- Data load ----------------------------------------------------------
    spy = _load_spy_data(start, end)
    vix = _load_vix_data(start, end)
    # Align VIX to SPY trading days (forward fill on SPY index)
    vix_aligned = vix.reindex(spy.index).ffill().bfill()
    if vix_aligned.isna().any():
        vix_aligned = vix_aligned.fillna(vix_aligned.median())

    # --- Strategy init ------------------------------------------------------
    cfg = StrategyConfig(
        name="options_put_write",
        strategy_type="OPTIONS",
        max_position_notional=None,
        params={
            "vix_entry": 15,
            "otm_pct": 0.95,
            "dte": 30,
            "use_enhanced_iv": True,
            "min_vrp": 0.03,
            "profit_take_pct": 0.50,
            "dte_exit": 21,
            "stop_loss_mult": 2.0,
            "kelly_fraction": 0.25,
            "max_notional_pct": 0.05,
            "gap_percentile": 0.99,
            "commission_per_contract": 0.65,
            "bid_ask_spread_pct": 0.05,
            "risk_free_rate": 0.045,
            "multiplier": 100,
        },
    )
    strategy = PutWriteVRPStrategy(cfg)
    strategy.initialize()

    sim = OptionFillSimulator()
    state = BacktestState()
    dates = spy.index.date if hasattr(spy.index, "date") else spy.index

    # --- Day-by-day simulation ---------------------------------------------
    for i, (ts, row) in enumerate(spy.iterrows(), start=1):
        current_date = ts.date() if isinstance(ts, datetime) else ts
        close_slice = spy["close"].iloc[:i].values
        n = len(close_slice)

        # Build implied_vol from VIX (percent → decimal)
        vix_val = float(vix_aligned.iloc[i - 1])
        iv_decimal = vix_val / 100.0

        # Assemble data dict
        data = {
            "close": np.column_stack([close_slice]),
            "implied_vol": np.full((n, 1), iv_decimal),
            "symbols": ["SPY"],
            "equity": np.array(state.cash + _position_value(state, row["close"])),
            "current_date": current_date,
        }

        signals = strategy.on_data(data)

        # Process fills
        for sig in signals:
            if sig.instrument_type != "option":
                continue
            m = sig.metadata
            action = m.get("action", "")
            contract_id = m.get("contract_id", "")
            spot = float(row["close"])

            if action == "SELL_TO_OPEN" and sig.direction == -1:
                result = _simulate_option_fill_from_signal(sim, sig, spot)
                if result is None:
                    continue
                contracts = int(m.get("contracts", 0))
                if contracts <= 0:
                    continue
                # Check cash-secured margin: strike * contracts * 100
                strike = float(m.get("strike", 0))
                cash_secured = strike * contracts * strategy.multiplier
                if state.cash < cash_secured:
                    _log.warning(
                        "[%s] Insufficient cash for %s (need %.0f, have %.0f)",
                        current_date, contract_id, cash_secured, state.cash,
                    )
                    continue

                # Credit received (premium inflow)
                premium = float(result["notional"])
                commission = float(result["commission"])
                state.cash += premium - commission
                pos = OptionPosition(
                    contract_id=contract_id,
                    contracts=contracts,
                    strike=strike,
                    dte=int(m.get("dte", 30)),
                    entry_credit=float(result["fill_price"]),
                    iv_at_entry=float(m.get("iv", 0)),
                    spot_at_entry=spot,
                    entry_costs=commission,
                )
                state.positions[contract_id] = pos
                state.trades.append({
                    "date": current_date,
                    "action": "SELL_TO_OPEN",
                    "contract_id": contract_id,
                    "contracts": contracts,
                    "strike": strike,
                    "premium": premium,
                    "commission": commission,
                    "spot": spot,
                })

            elif action == "BUY_TO_CLOSE" and sig.direction == 0:
                if contract_id not in state.positions:
                    continue
                pos = state.positions[contract_id]
                # Current IV for marking
                iv_now = iv_decimal
                dte_now = int(m.get("dte", 0))
                if dte_now <= 0:
                    # Expired ITM or worthless
                    payoff = max(0.0, pos.strike - spot)
                    close_cost = payoff * pos.contracts * strategy.multiplier
                else:
                    result = _simulate_option_fill_from_signal(sim, sig, spot)
                    if result is None:
                        continue
                    close_cost = float(result["notional"])

                commission = strategy.commission_per_contract * pos.contracts
                pnl = (
                    pos.entry_credit * pos.contracts * strategy.multiplier
                    - close_cost
                    - pos.entry_costs
                    - commission
                )
                state.cash -= close_cost + commission
                _log.info(
                    "[%s] Close %s: pnl=%.2f", current_date, contract_id, pnl
                )
                del state.positions[contract_id]
                state.trades.append({
                    "date": current_date,
                    "action": "BUY_TO_CLOSE",
                    "contract_id": contract_id,
                    "contracts": pos.contracts,
                    "strike": pos.strike,
                    "close_cost": close_cost,
                    "commission": commission,
                    "pnl": pnl,
                    "spot": spot,
                })

        # Mark-to-market: add unrealized P&L to equity
        state.equity_curve.append((current_date, _total_value(state, row["close"])))

    return pd.DataFrame(state.equity_curve, columns=["date", "equity"])


def _position_value(state: BacktestState, spot: float) -> float:
    """Rough mark-to-market value of open short puts (liability)."""
    val = 0.0
    for pos in state.positions.values():
        # Intrinsic + rough time-value estimate (simplified)
        intrinsic = max(0.0, pos.strike - spot)
        # For a quick mark: use entry_credit as time-value proxy scaled by DTE decay
        time_val = pos.entry_credit * max(0, pos.dte) / 30.0
        val += (intrinsic + time_val) * pos.contracts * 100
    return val


def _total_value(state: BacktestState, spot: float) -> float:
    """Cash + unrealized P&L of short options."""
    # For short puts, unrealized is negative of current liability
    liability = _position_value(state, spot)
    return state.cash - liability


def compute_metrics(equity_df: pd.DataFrame) -> dict[str, float]:
    """Compute basic performance metrics."""
    e = equity_df["equity"].values
    if len(e) < 2:
        return {}
    total_return = (e[-1] - e[0]) / e[0]
    daily_rets = np.diff(e) / e[:-1]
    sharpe = 0.0
    if daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    max_dd = 0.0
    peak = e[0]
    for v in e:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return {
        "start_equity": float(e[0]),
        "end_equity": float(e[-1]),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_annual": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_days": int(len(e)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", default="reports/put_write_backtest_2023_2024.csv")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    equity_df = run_backtest(start, end)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    equity_df.to_csv(out_path, index=False)

    metrics = compute_metrics(equity_df)
    _log.info("\n=== PutWrite Backtest %s → %s ===", start, end)
    for k, v in metrics.items():
        _log.info("%s: %s", k, v)
    _log.info("Equity curve saved to %s", out_path)

    # Load state from the run (not pretty, but works for diagnostics)
    # We'll re-run a quick version to count trades
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from quantcore.backtest.simulation import OptionFillSimulator
    from quantcore.strategies.base import StrategyConfig
    from quantcore.strategies.options.put_write import PutWriteVRPStrategy

    spy_diag = yf.download("SPY", start=start, end=end, progress=False)
    if isinstance(spy_diag.columns, pd.MultiIndex):
        spy_diag.columns = [c[0].lower() for c in spy_diag.columns]
    else:
        spy_diag.columns = [c.lower() for c in spy_diag.columns]
    spy_diag = spy_diag[["open", "high", "low", "close", "volume"]].dropna()
    vix_path = Path("quantcore/data/vol/vix.parquet")
    vix_raw = pd.read_parquet(vix_path)["vix"] if vix_path.exists() else pd.Series()
    vix_diag = vix_raw.reindex(spy_diag.index, method="ffill").ffill()

    cfg = StrategyConfig(
        name="options_put_write", strategy_type="OPTIONS", max_position_notional=None,
        params={
            "vix_entry": 15, "otm_pct": 0.95, "dte": 30,
            "use_enhanced_iv": True, "min_vrp": 0.03,
            "profit_take_pct": 0.50, "dte_exit": 21,
            "stop_loss_mult": 2.0, "kelly_fraction": 0.25,
            "max_notional_pct": 0.05, "gap_percentile": 0.99,
            "commission_per_contract": 0.65, "bid_ask_spread_pct": 0.05,
            "risk_free_rate": 0.045, "multiplier": 100,
        }
    )
    s = PutWriteVRPStrategy(cfg)
    s.initialize()
    n_entries = 0
    n_exits = 0
    for i in range(len(spy_diag)):
        close_slice = spy_diag["close"].iloc[:i+1].values
        v = float(vix_diag.iloc[i])
        data = {
            "close": np.column_stack([close_slice]),
            "implied_vol": np.full((len(close_slice), 1), v / 100.0),
            "symbols": ["SPY"],
            "equity": np.array(1_000_000.0),
        }
        sigs = s.on_data(data)
        for sig in sigs:
            if sig.metadata.get("action") == "SELL_TO_OPEN":
                n_entries += 1
            if sig.metadata.get("action") == "BUY_TO_CLOSE":
                n_exits += 1
    _log.info("Total signals: entries=%d, exits=%d", n_entries, n_exits)


if __name__ == "__main__":
    main()
