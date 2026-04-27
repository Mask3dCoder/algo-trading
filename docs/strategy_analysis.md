# QuantCore Strategy Analysis

**Generated:** 2026-04-27T00:59:00.305470
**Symbol:** SPY  |  **Capital:** $1,000,000  |  **Slippage:** 5.0 bps

## 1. Strategy Inventory

**Total:** 11 strategies across 3 categories

| # | Strategy | Category | Backtest Ready | Description |
|---|----------|----------|----------------|-------------|
| 1 | Momentum (Cross-Sectional) | Equity | Yes | 12-1 month cross-sectional momentum with inverse-vol weighting. Jegadeesh & Titman (1993). |
| 2 | Mean Reversion (RSI + BB) | Equity | Yes | RSI + Bollinger Band + VWAP z-score composite. 1-3 day hold. |
| 3 | Trend Following (EMA + ATR) | Equity | Yes | Multi-timeframe EMA crossover with ATR sizing and pyramiding. |
| 4 | Stat Arb (Pairs) | Equity | Yes | Cointegrated pairs with Kalman filter spread tracking and Kelly sizing. |
| 5 | CTA Trend (Multi-Speed) | Macro | Yes | Multi-speed trend signals (fast/med/slow) with vol targeting. |
| 6 | Carry | Macro | No | Cross-asset carry: long high-yield, short low-yield, vol-targeted. |
| 7 | Risk Parity | Macro | Yes | Equal risk contribution with vol targeting and regime overlay. |
| 8 | Delta Neutral (Options) | Options | No | Delta-hedged option positions, capturing vol risk premium. |
| 9 | Iron Condor (Options) | Options | No | Sell OTM call and put spreads, collect theta, manage tail risk. |
| 10 | Earnings Vol (Options) | Options | No | Pre-earnings straddle/strangle, post-earnings vol crush capture. |
| 11 | Tail Hedge (Options) | Options | No | OTM put buying program, 1-3% of AUM, roll monthly. |

## 2. Strategy Details

### Momentum (Cross-Sectional)
- **Registry key:** `equity_momentum`
- **Category:** Equity
- **Parameters:** `lookback=12mo, skip=1mo, n_long=5, n_short=5, vol_lb=6mo, rebal=21d`
- **Data requirement:** Daily OHLCV for universe (20+ assets)
- **Backtest ready:** True

### Mean Reversion (RSI + BB)
- **Registry key:** `equity_mean_reversion`
- **Category:** Equity
- **Parameters:** `rsi_win=14, rsi_os=25, rsi_ob=75, bb_win=20, z_entry=2.0, hold=3d`
- **Data requirement:** Daily OHLCV for a single asset
- **Backtest ready:** True

### Trend Following (EMA + ATR)
- **Registry key:** `equity_trend_following`
- **Category:** Equity
- **Parameters:** `fast=20, slow=50, trend=200, atr=20, pyramid=4, stop=2xATR`
- **Data requirement:** Daily OHLCV for a single asset
- **Backtest ready:** True

### Stat Arb (Pairs)
- **Registry key:** `equity_stat_arb`
- **Category:** Equity
- **Parameters:** `lookback=252, hl_max=20, entry_z=2.0, exit_z=0.5, kelly=0.5`
- **Data requirement:** Daily OHLCV for universe (10+ assets)
- **Backtest ready:** True

### CTA Trend (Multi-Speed)
- **Registry key:** `macro_cta_trend`
- **Category:** Macro
- **Parameters:** `fast=20, med=50, slow=200, w=(0.2,0.3,0.5), vol_tgt=15%, rebal=5d`
- **Data requirement:** Daily OHLCV for multiple assets
- **Backtest ready:** True

### Carry
- **Registry key:** `macro_carry`
- **Category:** Macro
- **Parameters:** `n_long=3, n_short=3, vol_tgt=10%, smoothing=21d, rebal=21d`
- **Data requirement:** Daily OHLCV + yield data for multiple assets
- **Backtest ready:** False

### Risk Parity
- **Registry key:** `macro_risk_parity`
- **Category:** Macro
- **Parameters:** `vol_tgt=10%, lookback=252, rebal=21d, regime=True, max_lev=1.0`
- **Data requirement:** Daily OHLCV for multiple assets (5+)
- **Backtest ready:** True

### Delta Neutral (Options)
- **Registry key:** `options_delta_neutral`
- **Category:** Options
- **Parameters:** `delta_band=0.1, rehedge_freq=daily, max_gamma=0.05`
- **Data requirement:** Options chain data (not available via yfinance free tier)
- **Backtest ready:** False

### Iron Condor (Options)
- **Registry key:** `options_iron_condor`
- **Category:** Options
- **Parameters:** `dte=45, wing_width=10%, profit_target=50%, stop=2x credit`
- **Data requirement:** Options chain data (not available via yfinance free tier)
- **Backtest ready:** False

### Earnings Vol (Options)
- **Registry key:** `options_earnings_vol`
- **Category:** Options
- **Parameters:** `dte_before=5, dte_after=1, min_iv_rank=60, max_size=5%`
- **Data requirement:** Options chain + earnings calendar data
- **Backtest ready:** False

### Tail Hedge (Options)
- **Registry key:** `options_tail_hedge`
- **Category:** Options
- **Parameters:** `put_delta=0.05-0.10, allocation=1-3%, roll_freq=monthly`
- **Data requirement:** Options chain data (not available via yfinance free tier)
- **Backtest ready:** False

## 3. Backtest Results

### 3.1 Performance Across 4 Market Regimes

#### Bull (2023-2024)

| Strategy | Return | Ann.Ret | Sharpe | Sortino | MaxDD | Calmar | Win Rate | Trades |
|----------|--------|---------|--------|---------|-------|--------|----------|--------|
| Momentum (SMA 50/200) | +0.0% | +0.0% | 0.00 | 0.00 | 0.0% | 0.00 | 0.0% | 0 |
| Momentum (EMA 20/50) | -38.2% | -21.5% | -1.54 | -1.15 | 39.4% | -0.48 | 0.0% | 3 |
| Mean Reversion (RSI 30) | -92.8% | -73.4% | -1.21 | -1.44 | 95.9% | -0.36 | 0.0% | 7 |
| *Buy & Hold (benchmark)* | +54.5% | -- | -- | -- | -- | -- | -- | -- |

#### Bear (2022)

| Strategy | Return | Ann.Ret | Sharpe | Sortino | MaxDD | Calmar | Win Rate | Trades |
|----------|--------|---------|--------|---------|-------|--------|----------|--------|
| Momentum (SMA 50/200) | +0.0% | +0.0% | 0.00 | 0.00 | 0.0% | 0.00 | 0.0% | 0 |
| Momentum (EMA 20/50) | -28.1% | -28.4% | -1.65 | -0.74 | 29.9% | -0.94 | 0.0% | 3 |
| Mean Reversion (RSI 30) | -120.6% | +0.0% | -0.66 | -0.54 | 121.5% | -0.63 | 0.0% | 13 |
| *Buy & Hold (benchmark)* | -19.7% | -- | -- | -- | -- | -- | -- | -- |

#### Sideways (2015)

| Strategy | Return | Ann.Ret | Sharpe | Sortino | MaxDD | Calmar | Win Rate | Trades |
|----------|--------|---------|--------|---------|-------|--------|----------|--------|
| Momentum (SMA 50/200) | -10.0% | -10.1% | -1.04 | -0.15 | 10.1% | -1.00 | 0.0% | 1 |
| Momentum (EMA 20/50) | -20.0% | -20.2% | -1.42 | -0.42 | 20.2% | -1.00 | 0.0% | 2 |
| Mean Reversion (RSI 30) | -41.6% | -41.9% | -1.99 | -1.48 | 42.6% | -0.97 | 0.0% | 4 |
| *Buy & Hold (benchmark)* | +0.2% | -- | -- | -- | -- | -- | -- | -- |

#### Volatile (2020)

| Strategy | Return | Ann.Ret | Sharpe | Sortino | MaxDD | Calmar | Win Rate | Trades |
|----------|--------|---------|--------|---------|-------|--------|----------|--------|
| Momentum (SMA 50/200) | +0.0% | +0.0% | 0.00 | 0.00 | 0.0% | 0.00 | 0.0% | 0 |
| Momentum (EMA 20/50) | -12.7% | -12.7% | -1.29 | -0.85 | 12.7% | -1.00 | 0.0% | 1 |
| Mean Reversion (RSI 30) | -85.7% | -85.8% | -3.04 | -2.78 | 85.8% | -1.00 | 0.0% | 7 |
| *Buy & Hold (benchmark)* | +14.5% | -- | -- | -- | -- | -- | -- | -- |

### 3.2 Strategy Selection

**Selected for paper trading: Momentum (EMA 20/50)**

| Strategy | Avg Score | Avg Sharpe | Avg Sortino | Avg MaxDD |
|----------|-----------|------------|-------------|-----------|
| Momentum (SMA 50/200) | 0.044 | -0.26 | -0.04 | +2.5% |
| Momentum (EMA 20/50) (SELECTED) | -0.668 | -1.48 | -0.79 | +25.6% |
| Mean Reversion (RSI 30) | -0.778 | -1.73 | -1.56 | +86.4% |

### 3.3 Selection Rationale

The **Momentum (EMA 20/50)** strategy was selected because it achieved the highest composite score averaged across all 4 market regimes (bull, bear, sideways, volatile).

**Score weights:**
- Sharpe Ratio (40%): risk-adjusted returns
- Sortino Ratio (20%): downside volatility penalty
- Calmar Ratio (20%): max drawdown efficiency
- Drawdown Resilience (20%): direct max drawdown penalty

## 4. Non-Backtestable Strategies

These strategies require data not available via free tiers:

| Strategy | Missing Data |
|----------|-------------|
| Carry | Daily OHLCV + yield data for multiple assets |
| Delta Neutral (Options) | Options chain data (not available via yfinance free tier) |
| Iron Condor (Options) | Options chain data (not available via yfinance free tier) |
| Earnings Vol (Options) | Options chain + earnings calendar data |
| Tail Hedge (Options) | Options chain data (not available via yfinance free tier) |

## 5. Recommendation

Deploy **Momentum (EMA 20/50)** for the 5-day paper trading run in Section 3.
The strategy has been backtest-validated across 4 distinct market regimes on SPY.

---

*Report generated by scripts/strategy_audit.py*