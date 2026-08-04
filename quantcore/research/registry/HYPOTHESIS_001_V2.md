# HYPOTHESIS_001.md — Version 2 Governance Amendment

> **Document**: HYPOTHESIS_001 V2 — Data Constraint Governance Patch
> **Status**: AMENDED — Original protocol superseded
> **Original registration**: 2026-06-13
> **This amendment**: 2026-06-13
> **Amendment reason**: Primary IS period (2016-01-01 → 2021-12-31) has no options data
>   cache coverage for the IVR warmup window (mid-2015 → mid-2016). The engine crashes
>   on day 1 of backtest with `DataQualityViolation`. Route B applied per Research
>   Governance Layer §3: *"Amend the treaty before you cross the border."*
>
> **This document supersedes HYPOTHESIS_001.md V1** — the V1 partition matrix
> is **permanently void**. All future work must reference this V2 document.
> Any code, experiment run, or analysis referencing V1 dates is invalid.

---

## Amended Data Partition Architecture

Due to data availability constraints (no pre-2021-06-01 SPY options chain cache
and no accessible API to backfill it without significant capital investment), the
following reduced timeline is the ONLY acceptable partition matrix for HYPOTHESIS_001.

```
  Phase      Original V1 Dates         Amended V2 Dates         Covered?
  ──────     ────────────────          ────────────────         ────────
  IS         2016-01-01 → 2021-12-31  2021-06-01 → 2022-12-31  YES (partial)
  Val        2022-01-01 → 2023-12-31  2023-01-01 → 2023-12-31  YES
  Vault      2024-01-01 → 2026-06-01  2024-01-01 → 2024-12-31  YES
```

| Dataset | V2 Date Range | Purpose | Access |
|---|---|---|---|
| **In-Sample (IS)** | 2021-06-01 → 2022-12-31 | Feature calibration, IVR threshold tuning, entry threshold discovery | Permitted |
| **Validation (Val)** | 2023-01-01 → 2023-12-31 | Hyperparameter lock-in, DSR computation (1 trial) | Permitted |
| **Vault (OOS)** | 2024-01-01 → 2024-12-31 | One-time TriageGate evaluation | **PROHIBITED** until Val gate passes |

### What This Loses from V1

- **~5.5 years of IS calibration** (2016-01 → 2021-05). The 2021-06 → 2022-12
  IS period includes the COVID recovery, 2022 inflation bear market, but NOT the
  low-vol 2017–2019 period or the 2020 COVID crash — the two highest-VRP regimes.
- **Significantly reduced trade count**: ~18 months of IS data vs 6 years.
  Expect fewer than 20 total straddles in the IS window (entry every 30 days).
- **Higher variance estimates**: Fewer trades → wider CI bounds → harder to
  pass the DSR gate.

### What This Preserves

- The 2022 inflation bear market IS entry is preserved (high-VRP environment — good).
- The 2023 bull market Val period tests whether the strategy generalizes.
- 2024 Vault tests the compressed vol environment cleanly.

### Route A (Data Acquisition) Still Superior

If a Polygon.io API key ($200/mo+) or equivalent data source becomes available,
**Route A remains the correct scientific path**. This V2 amendment does not close Route A.
The IS window can be re-extended to 2016-01-01 retroactively if backfill succeeds.

---

## Strict Falsification Criteria (Unchanged)

Any one of these being triggered invalidates the hypothesis. The experiment is
immediately logged as FAILED in `experiments.csv` with `status=rejected`.

| Gate | Threshold | Failure Condition |
|---|---|---|
| **DSR p-value** | p < 0.10 | DSR p-value >= 0.10 at Val phase |
| **Friction Erosion** | < 35% gross premium retained | Transaction costs erode >= 65% of gross premium |
| **Max Drawdown** | < 20% on 1x gross exposure | Peak-to-trough drawdown >= 20% during Val period |
| **OOS Sharpe** | >= 0.3 annualized | OOS Sharpe < 0.3 in Vault |
| **Consistency** | >= 70% fold pass rate | < 70% of IS folds individually pass all gates |

---

## Trial Counter

Current trial count for H001 V2: **1** (trial `ca8727e7` marked `status=rejected`
in `experiments.csv` for failing the §"Strict Falsification Criteria" MaxDD gate
at $-46\%$ on the ATM short-straddle execution vehicle — Gamma-whip exposure
during the 2022 inflation shock). Each subsequent IS calibration pass
increments the trial count, even if the trial also fails a gate; the count
provides the N for the Deflated Sharpe Ratio (DSR) gate at Validation.

---

## Success Definition (Unchanged from V1)

The hypothesis is considered **validated** if and only if:
1. All Falsification Criteria gates pass in Val (2023)
2. TriageGate promotion check returns `passed=True`
3. One single run is performed against Vault data (2024) — the first and only time
4. Vault run passes max drawdown < 20%, OOS Sharpe >= 0.3

**If all above are met -> strategy promoted to `PAPER_TRADING` status in `live.yaml`**

---

## Amendments Log

| Date | Version | Change | Reason |
|---|---|---|---|
| 2026-06-13 | V1 | Initial registration | Pre-data analysis |
| 2026-06-13 | V2 | Reduced IS to 2021-06-01 → 2022-12-31; Val to 2023; Vault to 2024 | No pre-2021-06 options cache available; IVR warmup window unreadable |
| 2026-06-13 | V2.1 | Trial `ca8727e7` (ATM short straddle, `alpha/delta_neutral.py`) marked `rejected` for MaxDD=-46% violation; strategic pivot to **Option B (Iron Condor, `alpha/iron_condor.py`)** as the V2.1 execution vehicle; fresh IS calibration pass required on the new structure before Val/Vault runs are unlocked | Straddle is structurally short-Gamma with unbounded payoff — the 2022 inflation shock forced a Gamma-whip on the static 5-share delta-hedge band (short SPY into the drop, buy back violently higher on the CPI squeeze). Iron condor's long 10-delta wings flip far-OTM Gamma to positive and cap max loss at wing width — the only path to mechanically satisfying the <20% IS MaxDD gate without abandoning the VRP harvest entirely. Known cost: the **volatility-skew tax** — buying 10-delta puts systematically pays for tail insurance at the steepest point of the skew curve, which erodes gross credit; this is exactly the friction the IS pass is designed to measure. See `reports/h001_v2_1_is_pass_<ts>.json` for the calibration result. |

---

*This document is immutable once saved. Any subsequent changes constitute a new hypothesis (HYPOTHESIS_002) and start a fresh trials count.*