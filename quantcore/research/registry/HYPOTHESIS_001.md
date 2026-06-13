# Hypothesis Doc #001: Systematic Volatility Risk Premium (VRP) Harvesting

> **Status:** PRE-REGISTERED — No data analysis performed before this document.
> **Registered:** 2026-06-13
> **Author:** QuantCore Research Pipeline
> **Hypothesis ID:** HYPOTHESIS_001

---

## 1. Core Economic Rationale

The Volatility Risk Premium (VRP) exists due to a variance risk asymmetry: options buyers purchase implied volatility ($\sigma_{IV}$) as insurance against catastrophic market drawdowns, paying a structural premium to options sellers. Therefore, the expected implied volatility overestimates the subsequent realized volatility ($\sigma_{RV}$) over an investment horizon $\tau$:

$$\mathbb{E}[\sigma_{IV, t}] - \mathbb{E}[\sigma_{RV, t, t+\tau}] > 0$$

This strategy intends to systematically harvest this variance premium using liquid index options (SPY/SPX), under the structural assumption that risk-averse market participants structurally overpay for tail-risk insurance.

### Why SPY/SPX?
- Extreme liquidity (tight bid-ask spreads, deep book)
- No dividend risk complications (cash index settlement)
- OPRA options market is highly efficient — VRP signal is pure, not confounded by liquidity premiums

---

## 2. Mathematical Definition & Alpha Signal

| Component | Definition |
|---|---|
| **Signal Source** | VRP spread = VIX (or front-month ATM IV) − 21-day realized volatility (rolling std of log returns) |
| **Signal Conditioning** | Enter short vol only when normalized VRP spread is above the 25th percentile of its rolling 2-year historical distribution |
| **Entry Timing** | Weekly rebalance, every Friday at market close (or nearest trading day) |

### Execution Vehicle: Two Approved Variants

#### Option A — Short ATM Straddle (Higher VRP capture, unbounded risk)
- Sell 1× ATM call + 1× ATM put, same expiry, same strike (approximately ATM via 50-delta)
- Delta-hedge with underlying when |portfolio delta| > 0.15
- Expiry: 30 DTE at entry (next expiry cycle)

#### Option B — Iron Condor (Defined risk, lower VRP capture)
- Sell 10-delta put spread + 30-delta call spread (wings at 10-delta and 30-delta)
- Maximum loss is capped at wing width
- Expiry: 30 DTE at entry

**Decision rule:** Option A (straddle) vs Option B (iron condor) is a live configuration parameter. Both are tested. TriageGate decides which survives.

### Expiry / Roll Schedule
- Enter at 30 DTE
- Exit at expiration (not earlier)
- Roll every Friday at close on new 30-DTE contracts

---

## 3. Risk Controls

| Parameter | Value | Rationale |
|---|---|---|
| Max position size per leg | $10,000 notional | Prevents margin cascade |
| Max total portfolio delta | ±0.15 before hedge | Delta-neutral target |
| Max portfolio gamma | ±0.05 per 1% move | Prevents blow-up near expiry |
| Max loss per trade | 3× premium received | Hard stop, no exceptions |
| Margin buffer | ≥25% of max loss | Prevents forced liquidation |

---

## 4. Data Split Architecture (The Hard Isolation)

**No code, no analyst, and no system may access the Vault data until this hypothesis passes TriageGate.**

| Dataset | Date Range | Purpose | Access |
|---|---|---|---|
| **In-Sample (IS)** | 2016-01-01 → 2021-12-31 | Feature calibration, distribution mapping, entry threshold discovery | Permitted |
| **Validation (Val)** | 2022-01-01 → 2023-12-31 | Hyperparameter tuning, initial DSR computation, fold-level performance | Permitted |
| **Vault (OOS)** | 2024-01-01 → 2026-06-01 | One-time TriageGate evaluation | **PROHIBITED** until promotion |

The walk-forward backtester (`quantcore/research/walkforward.py`) generates anchored folds within these boundaries automatically.

---

## 5. Strict Falsification Criteria

**Any one of these being triggered invalidates the hypothesis. The experiment is immediately logged as FAILED in `experiments.csv` with `status=rejected`.**

| Gate | Threshold | Failure Condition |
|---|---|---|
| **DSR p-value** | p < 0.05 | DSR p-value ≥ 0.05 at Validation phase |
| **Friction Erosion** | < 35% gross premium retained | Transaction costs (bid-ask + slippage + fees) erode ≥ 65% of gross premium |
| **Max Drawdown** | < 20% on 1× gross exposure | Peak-to-trough drawdown ≥ 20% during Validation period |
| **OOS Sharpe** | ≥ 0.3 annualized | OOS Sharpe < 0.3 in any walk-forward fold |
| **Consistency** | ≥ 70% fold pass rate | < 70% of walk-forward folds individually pass all gates |

---

## 6. Economic Model of Transaction Costs

To prevent the friction erosion gate from being gamed:
- Assume bid-ask spread = 1.5× measured market spread at time of trade
- Slippage = 5 bps of notional (conservative, non-zero)
- Fee = $0.65 per contract (Alpaca standard retail rate)
- No rebating, no maker/taker distinction (assume taker on entry and exit)

> **Note:** These costs are *conservative lower bounds*. Real execution will be worse.

---

## 7. Known Failure Modes (Pre-registered)

Before any data analysis, we pre-register what we believe will kill this hypothesis:

1. **Volmageddon scenario**: 2020 COVID crash causes vol spike to 80+, short straddle gamma blowup causes loss > 3× premium
2. **Volatility persistence**: 2017–2019 low-vol regime causes VRP signal to stay below 25th percentile for extended period, generating no trades
3. **Margin cascade**: After a large vol spike, margin requirements expand faster than equity buffer, forcing liquidation at worst possible moment
4. **Delta-hedging friction**: Weekly delta hedging in a trending vol environment causes cumulative hedging costs to exceed premium collected

---

## 8. Trial Counter

The trials ledger (`experiments.csv`) must be updated *before* each backtest run. This is the N for DSR computation.

Current trial count before this experiment: **0** (HYPOTHESIS_001 is the first formal trial)

---

## 9. Success Definition

The hypothesis is considered **validated** if and only if:
1. All Falsification Criteria gates pass in Validation (2022–2023)
2. TriageGate promotion check returns `passed=True`
3. One single run is performed against Vault data (2024–2026) — the first and only time
4. Vault run passes DSR p < 0.10, friction erosion < 35%, MaxDD < 20%

**If all above are met → strategy promoted to `PAPER_TRADING` status in `live.yaml`**

---

*This document is immutable once saved. Any subsequent changes constitute a new hypothesis (HYPOTHESIS_002) and start a fresh trials count.*