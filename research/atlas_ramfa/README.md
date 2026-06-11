# Atlas RAMFA Research

Atlas RAMFA is a **regime-adaptive multi-factor long/short equity research strategy**.

## Current status

- Status: **RESEARCH**
- Unit tests: **PASS** (`tests/unit/strategies/test_atlas_ramfa.py`)
- Validation harness: fixed on **2026-06-11**
- Paper/live activation: **blocked**

The validation harness now runs successfully:

```bash
python research/atlas_ramfa/validate.py --output reports/atlas_ramfa_validation.json
```

Threshold failure mode remains active when requested:

```bash
python research/atlas_ramfa/validate.py --fail-on-threshold
```

The synthetic default run generates a clean research report but does **not** approve the strategy for paper or live trading.

## Gating checklist

Atlas RAMFA must satisfy all gates below before it can leave `RESEARCH`:

1. Point-in-time backtest with lagged execution.
2. Walk-forward validation with embargo.
3. Transaction-cost and borrow-cost sensitivity.
4. Paper-readiness configuration review.
5. Model-risk review approval.
6. Explicit migration from `RESEARCH` to `PAPER` or `LIVE` in governance docs and configs.

## Production quarantine

Atlas RAMFA must remain absent from:

- `config/paper.yaml`
- `config/live.yaml`

until all validation gates pass and model-risk review is approved.

The strategy is currently quarantined by design:

- `research/atlas_ramfa/paper_readiness.toml` marks `status = "RESEARCH"`.
- `config/strategies/atlas_ramfa.toml` documents research-only parameters.
- `research/atlas_ramfa/model_risk_review.md` records forward-looking-bias, survivorship-bias, liquidity, borrow, regime, and IC-weighting risks.

## Important discipline

Do not tune Atlas RAMFA parameters after observing paper or live P&L. Any future paper activation must be treated as a fixed out-of-sample validation window with no parameter changes during the run.
