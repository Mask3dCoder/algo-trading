# QuantCore Daily Operations Quick Reference

## Pre-Market (8:00–9:00 ET)

```bash
# 1. Verify data feeds are healthy
python scripts/trade_live.py --strategy momentum --symbols SPY --mode dry-run --skip-preflight

# 2. Run full pre-flight (checks market, data, strategy, PDT)
python scripts/trade_live.py --strategy rsi --symbols GLD --mode dry-run

# 3. Review yesterday's P&L
python scripts/analyze_paper_trade.py --all-candidates --json outputs/daily_pnl.json
```

### Pre-flight checklist

| Check | Command/Source | Pass Criteria |
|---|---|---|
| Market open | Calendar (NYSE) | Not a holiday/weekend |
| Data feed | Pre-flight `data_available` | PASS |
| Account capital | Pre-flight `pdt_rule` | `>= $25,000` |
| Strategy health | Pre-flight `strategy_health` | PASS |
| Positions flat | `--mode dry-run` shows no hanging positions | 0 open positions |
| Risk limits | Review `risk.limits.RiskLimits` | All within bounds |

## Market Hours (9:30–16:00 ET)

### Launch live trading (paper first!)
```bash
# STEP 1: Start with paper trading to validate
python scripts/trade_live.py --strategy momentum --symbols SPY,QQQ --mode paper

# STEP 2: After 1+ week of clean paper trading, go live
python scripts/trade_live.py --strategy momentum --symbols SPY,QQQ --mode live
```

### Monitor during session
- Watch for circuit breaker alerts (daily loss >= $50K)
- Check fill confirmations within 10 seconds of order
- Verify position sizes match expected (25% of capital per position)
- Monitor max drawdown stays below 30%

### Emergency stop
```bash
# Ctrl+C initiates graceful shutdown — flattens positions
# If unresponsive, use broker's own cancel-all:
#   IBKR: TWS > Account > Close All Positions
#   Alpaca: Dashboard > Positions > Liquidate
```

## Post-Market (after 16:00 ET)

```bash
# 1. Generate post-trade analysis
python scripts/analyze_paper_trade.py --all-candidates --json outputs/$(date +%Y-%m-%d)_pnl.json --output outputs/$(date +%Y-%m-%d)_report.html

# 2. Run walk-forward validation weekly to confirm edge persists
python scripts/walk_forward_validation.py

# 3. Check full backtest health
python scripts/strategy_validation.py
```

### Daily P&L reconciliation
- Compare broker statement P&L vs. post-trade analysis output
- Flag any discrepancy > $10
- Log daily P&L to tracking spreadsheet

## Confirmed Live Candidates (from walk-forward)

| Symbol | Strategy | OOS Sharpe | OOS Return | Max DD | Trade Frequency |
|---|---|---|---|---|---|
| QQQ | Momentum (SMA 50/200) | 1.99 | +19.7% | 5.0% | ~1/year |
| SPY | Momentum (SMA 50/200) | 1.83 | +10.9% | 3.4% | ~1/year |
| GLD | Mean Reversion (RSI 30) | 2.71 | +4.6% | 1.3% | ~15/year |

## Risk Limits at a Glance

```
Max position per symbol:     25% of capital
Max drawdown (circuit):      30% from peak
Daily loss limit:            $50,000
PDT minimum:                 $25,000
Cooldown per symbol:         1 hour
Max concurrent positions:    5
```

## Troubleshooting

| Symptom | Diagnosis | Action |
|---|---|---|
| No signals generated | Strategy warmup failed | Re-run preflight; check data feed |
| Orders rejected | PDT/limits breach | Check account equity; review limits |
| Slippage > 50 bps | Low liquidity | Widen stops; reduce position size |
| Circuit breaker tripped | Daily loss > $50K | Stop trading; investigate; do not override |
| Data gaps | Feed disconnected | Restart feed; validate cache |

## Weekly Audit (Every Friday)

```bash
# Full validation suite
python scripts/strategy_validation.py          # All 5 strategies x 8 symbols
python scripts/walk_forward_validation.py       # OOS edge check
python -m pytest tests/ -q --tb=short           # Full test suite
```

## Key Paths

```
scripts/strategy_validation.py    — Validation matrix (40 backtests)
scripts/walk_forward_validation.py — Out-of-sample edge verification
scripts/analyze_paper_trade.py    — Post-trade P&L and slippage analysis
scripts/trade_live.py             — Live trading launcher with safety checks
outputs/                          — Reports, JSON exports, logs
quantcore/risk/limits.py          — Risk limit definitions
quantcore/backtest/engine.py      — Backtest engine (event-driven)
```
