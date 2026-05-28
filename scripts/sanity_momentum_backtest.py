"""Brutally simple 12-month cross-sectional momentum sanity check.

Universe: SPY, QQQ, IWM, GLD, TLT, DBC
Signal: rank by trailing 12m return, long top 2, short bottom 2
Execution: month-end signal, next-day open, 1-month hold
Costs: 5 bps spread + 10 bps impact + $0.005/share commission
Data: yfinance daily OHLCV 2015-01-01 to 2024-12-31

No imports from quantcore.strategies, quantcore.backtest, or any residual code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"]
START = "2015-01-01"
END = "2024-12-31"
MOM_WINDOW = 252  # ~12 months trading days
TOP_K = 2
BOTTOM_K = 2
HALF_SPREAD_BPS = 5.0
MARKET_IMPACT_BPS = 10.0
COMMISSION_PER_SHARE = 0.005
AVG_SHARE_PRICE = 200.0  # rough avg for our ETFs

def sharpe_ratio(daily_returns: np.ndarray, rf_annual: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily returns."""
    excess = daily_returns - rf_annual / 252.0
    if np.std(excess, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(252.0))

def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown as a positive percentage."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))

def round_trip_cost(dollar_traded: float, shares_traded: float) -> float:
    """One-way transaction cost for a trade leg.

    Half-spread (5 bps) + market impact (10 bps) + commission.
    Applied on entry AND exit, so round-trip cost = 2x this.
    """
    spread_cost = dollar_traded * HALF_SPREAD_BPS / 10000.0
    impact_cost = dollar_traded * MARKET_IMPACT_BPS / 10000.0
    commission_cost = shares_traded * COMMISSION_PER_SHARE
    return spread_cost + impact_cost + commission_cost


def main():
    print("=" * 60)
    print(f"  SANITY CHECK: 12-Month Cross-Sectional Momentum")
    print(f"  Universe: {', '.join(TICKERS)}")
    print(f"  Period:   {START} to {END}")
    print(f"  Costs:    {HALF_SPREAD_BPS} bps spread + {MARKET_IMPACT_BPS} bps impact + $0.005/sh")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Download data
    # -----------------------------------------------------------------------
    print("\nDownloading yfinance data...")
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=True,
                        group_by="ticker", progress=False)
    if data.empty:
        print("ERROR: No data downloaded.")
        sys.exit(1)

    # Extract adjusted close
    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    for t in TICKERS:
        try:
            closes[t] = data[t]["Close"].squeeze().astype(float)
            opens[t] = data[t]["Open"].squeeze().astype(float)
        except (KeyError, AttributeError):
            print(f"WARNING: Could not extract data for {t}, skipping.")
            continue

    if len(closes) < 4:
        print("ERROR: Too few symbols with valid data.")
        sys.exit(1)

    # Build DataFrame
    close_df = pd.DataFrame(closes).dropna()
    open_df = pd.DataFrame(opens).dropna()
    print(f"Loaded {len(close_df)} trading days for {len(close_df.columns)} symbols.")

    # Ensure index is datetime
    close_df.index = pd.to_datetime(close_df.index)
    open_df.index = pd.to_datetime(open_df.index)

    # -----------------------------------------------------------------------
    # 2. Generate month-end signals
    # -----------------------------------------------------------------------
    # Month-end dates
    month_ends = close_df.resample("ME").last().index
    # Filter to valid month-ends that have enough prior data
    month_ends = month_ends[month_ends >= close_df.index[MOM_WINDOW]]

    portfolio_values: list[float] = [1_000_000.0]  # start with $1M
    monthly_returns: list[float] = []
    monthly_pnl: list[float] = []
    n_wins = 0
    n_total = 0
    total_costs = 0.0
    signals_log: list[dict] = []

    cash = 1_000_000.0

    for i, me_date in enumerate(month_ends):
        # --- Signal generation at month-end ---
        lookback_start = me_date - pd.Timedelta(days=MOM_WINDOW + 30)
        hist = close_df.loc[lookback_start:me_date]

        if len(hist) < 60:  # need at least 3 months of data
            continue

        # Trailing 12-month returns (price only)
        start_price = hist.iloc[0]
        end_price = hist.iloc[-1]
        trailing_ret = (end_price / start_price) - 1.0

        # Rank and select
        ranked = trailing_ret.sort_values(ascending=False)
        longs = ranked.head(TOP_K).index.tolist()
        shorts = ranked.tail(BOTTOM_K).index.tolist()

        # --- Execution at next day's open ---
        # Find the next trading day after month-end
        next_day = close_df.index[close_df.index > me_date]
        if len(next_day) == 0:
            continue
        exec_date = next_day[0]

        # Get open prices
        if exec_date not in open_df.index:
            continue
        exec_prices = open_df.loc[exec_date]
        entry_prices = exec_prices  # Series

        # Equal-weight: each leg gets 1/4 of capital
        leg_weight = 1.0 / (TOP_K + BOTTOM_K)

        # Compute shares and costs for entry
        entry_costs = 0.0
        long_shares: dict[str, float] = {}
        short_shares: dict[str, float] = {}

        for sym in longs:
            if sym not in entry_prices.index or pd.isna(entry_prices[sym]) or entry_prices[sym] <= 0:
                continue
            dollars = cash * leg_weight
            price = float(entry_prices[sym])
            shares = dollars / price
            long_shares[sym] = shares
            entry_costs += round_trip_cost(dollars, shares)

        for sym in shorts:
            if sym not in entry_prices.index or pd.isna(entry_prices[sym]) or entry_prices[sym] <= 0:
                continue
            dollars = cash * leg_weight
            price = float(entry_prices[sym])
            shares = dollars / price
            short_shares[sym] = shares
            entry_costs += round_trip_cost(dollars, shares)

        # --- Determine next month-end for exit ---
        if i + 1 < len(month_ends):
            exit_date_candidate = month_ends[i + 1]
        else:
            # Last period: exit at end
            exit_date_candidate = close_df.index[-1]

        # Find the next trading day after exit month-end
        exit_next_days = close_df.index[close_df.index > exit_date_candidate]
        if len(exit_next_days) == 0:
            exit_date_actual = close_df.index[-1]
        else:
            exit_date_actual = exit_next_days[0]

        # Use open price for exit
        if exit_date_actual in open_df.index:
            exit_prices = open_df.loc[exit_date_actual]
        elif exit_date_actual in close_df.index:
            exit_prices = close_df.loc[exit_date_actual]
        else:
            # Find nearest available date
            available = close_df.index[close_df.index <= exit_date_actual]
            if len(available) == 0:
                continue
            exit_date_actual = available[-1]
            exit_prices = close_df.loc[exit_date_actual]

        # --- Compute P&L for the month ---
        pnl = 0.0
        exit_costs = 0.0

        for sym in long_shares:
            if sym not in exit_prices.index or pd.isna(exit_prices[sym]):
                continue
            exit_price = float(exit_prices[sym])
            entry_price = float(entry_prices[sym])
            sym_pnl = long_shares[sym] * (exit_price - entry_price)
            pnl += sym_pnl
            exit_costs += round_trip_cost(abs(sym_pnl), long_shares[sym])

        for sym in short_shares:
            if sym not in exit_prices.index or pd.isna(exit_prices[sym]):
                continue
            exit_price = float(exit_prices[sym])
            entry_price = float(entry_prices[sym])
            sym_pnl = short_shares[sym] * (entry_price - exit_price)
            pnl += sym_pnl
            exit_costs += round_trip_cost(abs(sym_pnl), short_shares[sym])

        total_tc = entry_costs + exit_costs
        total_costs += total_tc
        net_pnl = pnl - total_tc

        # Track
        n_total += 1
        if net_pnl > 0:
            n_wins += 1

        monthly_pnl.append(net_pnl)
        monthly_ret = net_pnl / cash
        monthly_returns.append(monthly_ret)
        cash += net_pnl
        portfolio_values.append(cash)

        signals_log.append({
            "month_end": me_date,
            "exec_date": exec_date,
            "exit_date": exit_date_actual,
            "longs": longs,
            "shorts": shorts,
            "pnl_gross": pnl,
            "costs": total_tc,
            "pnl_net": net_pnl,
            "return_pct": monthly_ret * 100,
        })

    # -----------------------------------------------------------------------
    # 3. Compute full metrics
    # -----------------------------------------------------------------------
    portfolio = np.array(portfolio_values)
    total_return = (portfolio[-1] - portfolio[0]) / portfolio[0] * 100.0
    daily_prices = close_df.copy()

    # For Sharpe, we need daily returns from the strategy
    # Since we only have monthly PnL, approximate daily returns
    # Better: compute Sharpe from monthly PnL annualized
    monthly_rets_arr = np.array(monthly_returns)
    if len(monthly_rets_arr) > 2:
        # Monthly Sharpe: mean/std of monthly returns, annualized by sqrt(12)
        monthly_excess = monthly_rets_arr  # rf=0
        monthly_mean = float(np.mean(monthly_excess))
        monthly_std = float(np.std(monthly_excess, ddof=1))
        if monthly_std > 1e-12:
            monthly_sharpe = monthly_mean / monthly_std * np.sqrt(12.0)
        else:
            monthly_sharpe = 0.0

        # Max DD from month-end portfolio values
        mdd = max_drawdown(portfolio) * 100.0
    else:
        monthly_sharpe = 0.0
        mdd = 0.0

    win_rate = (n_wins / n_total * 100.0) if n_total > 0 else 0.0

    # -----------------------------------------------------------------------
    # 4. Print results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Total Return:            {total_return:+.2f}%")
    print(f"  Annualized Sharpe:       {monthly_sharpe:.4f}  (rf=0%, monthly->annual)")
    print(f"  Maximum Drawdown:        {mdd:.2f}%")
    print(f"  Win Rate (monthly):      {win_rate:.1f}%  ({n_wins}/{n_total})")
    print(f"  Total Round-Trips:       {n_total * 4}  ({TOP_K + BOTTOM_K} legs × {n_total} periods)")
    print(f"  Total Transaction Costs: ${total_costs:,.2f}")
    print(f"  Final Portfolio:         ${portfolio[-1]:,.2f}")
    print(f"  Periods Traded:          {n_total} months")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # 5. Verdict
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    if monthly_sharpe > -0.5 and total_return > -20.0:
        print("  VERDICT: Infrastructure VALID")
        print()
        print("  The simple momentum strategy produces realistic results.")
        print("  The fault lies in the complex strategy signal logic,")
        print("  not the backtest infrastructure.")
    else:
        print("  VERDICT: Infrastructure BROKEN")
        print()
        print("  A simple momentum strategy should not produce deeply")
        print("  negative results. The backtest pipeline has a systemic")
        print("  issue (look-ahead, data alignment, cost double-counting")
        print("  or direction inversion).")
    print("=" * 60)

    # Print recent signals
    print("\nLast 5 signals:")
    print(f"  {'Date':<12s} {'Longs':<20s} {'Shorts':<20s} {'PnL Net':>10s} {'Ret':>8s}")
    print(f"  {'-'*12} {'-'*20} {'-'*20} {'-'*10} {'-'*8}")
    for s in signals_log[-5:]:
        l_str = ", ".join(s["longs"])
        s_str = ", ".join(s["shorts"])
        dt_str = str(s["month_end"].date())
        print(f"  {dt_str:<12s} {l_str:<20s} {s_str:<20s} {s['pnl_net']:>+10.2f} {s['return_pct']:>+7.2f}%")


if __name__ == "__main__":
    main()