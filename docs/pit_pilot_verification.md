# PIT Pilot Verification Checklist

**Pre-written 2026-08-03** — the moment a real Polygon key lands, the pilot is a
same-day execution, not another research cycle. Window: **2022-01-01 →
2024-12-31**, 20–30 names. Purpose: hand-verify the PIT pipeline (bars,
corporate actions, universe gating) against **independently confirmable**
real-world events BEFORE scaling to the 500×7yr backfill.

Rollback / reference: tag `pre-forensic-audit` (`61daebe`); scaffold +
gate commits `5a675af` · `91a4ba6` · `04a6b18`.

---

## Pilot universe (20–30 names)

| Name | Role | Anchor event |
|------|------|--------------|
| AMZN | Split case | 20:1 ex-date 2022-06-06 |
| ATVI | Delisting case | last trade 2023-10-12, closed 2023-10-13 |
| MSFT | Dividend gap case | announced 2024-03-12, ex-date 2024-05-15 |
| GOOGL | Second split | 20:1 ex-date 2022-07-15 |
| TSLA | Third split | 5:1 ex-date 2022-08-25 |
| NVDA | Fourth split | 10:1 ex-date 2024-06-10 |
| RDDT | Not-yet-listed control | IPO ~2024-03-21 → zero bars before |
| ARM | Not-yet-listed control | IPO ~2023-09-14 → zero bars before |
| SPY | Control | cross-check vs committed Yahoo fixture |
| AAPL, META, JPM, V, WMT, XOM, UNH, JNJ, PG, KO, PEP, HD, LLY, AVGO, ORCL, AMD, DIS, NFLX | Baseline (no corp action in window) | — |

Every name: query Polygon's ticker metadata and record the exact IPO/listing
date + `delisted_utc` from **Polygon's own fields** (not our assumption).

## Case 1 — Split: AMZN 20:1, ex-date 2022-06-06

Independent facts (verified: Amazon 2022-03-09 announcement; Nasdaq):
record 2022-05-27, payable 2022-06-03, **ex/effective 2022-06-06**, ratio 20:1.
Pre-split price ~$2,400–2,800; post-split ~$120–140.

Checks:
1. PIT reader asof=2022-06-03 → AMZN last bar ≈ $2,5xx (unadjusted).
2. PIT reader asof=2022-06-06 → first bar ≈ $1xx (unadjusted).
3. Corp-action table row: ratio `20.0` on `2022-06-06`.
4. Engine rescale on a held position: qty ×20, avg_cost ÷20, total value
   continuous across the price gap (no P&L jump).
5. Diff AMZN unadjusted OHLC vs Yahoo `auto_adjust=False` — must match.

## Case 2 — Delisting: ATVI, last trade 2023-10-12

Independent facts (verified: Nasdaq Equity Corporate Actions Alert #2023-588):
last trading date **2023-10-12**, merger closed **2023-10-13** pre-market
($95.00/share cash), suspended effective **2023-10-16**.

Checks:
1. PIT reader asof=2023-10-10 → ATVI present; last bar date 2023-10-12,
   price ≈ $94.42.
2. PIT reader asof=2023-10-16 → ATVI **absent** (or zero bars).
3. Record Polygon's actual `delisted_utc` for ATVI. If it reads 2023-10-13
   (close date) vs 2023-10-16 (suspension) vs 2023-10-12 (last trade),
   adopt the convention consciously — this is a documented field-semantics
   decision, not a silent one.

## Case 3 — Dividend announcement→ex-date gap: MSFT

Independent facts (verified: Microsoft board press releases):
- $0.75 — announced **2024-03-12**, ex **2024-05-15**, record 2024-05-16, pay 2024-06-13.
- $0.83 (10% raise) — announced **2024-09-16**, ex **2024-11-21**, record 2024-11-21, pay 2024-12-12.

Checks (stresses "invisible until announced" PIT logic):
1. Polygon dividend record returns `declaration_date`/`announcement_date`,
   `ex_dividend_date`, `record_date`, `pay_date`.
2. PIT gate requires `announced_at <= asof AND ex_date <= asof`.
3. Mini buy-and-hold(MSFT) through the engine ending **2024-05-15** → NO
   dividend credited. Ending **2024-05-16** → $0.75/share credited.
4. Same at the 2024-09/11 boundary for $0.83.

## Baseline integrity checks (every symbol)

1. `python bin/polygon_ingest.py check` with the real key — confirm the API
   base URL responds. **Post-Massive-rebrand note:** docs/signup moved to
   massive.com; the scaffold still points at `https://api.polygon.io`
   (`quantcore/data/pit/market/client.py:19`). Verify that host still works
   with the new key, or update BASE_URL before the run.
2. Bars: no gaps on trading days (compare to a calendar/benchmark), no
   duplicate timestamps, high ≥ low, volume ≥ 0.
3. Corp-action counts: splits/dividends/delistings in the table match what
   Polygon's reference endpoints return for each symbol in the window.
4. Unadjusted OHLC matches Yahoo `auto_adjust=False` to float precision.

## Acceptance criteria

- All three cases assert clean on the first pass, OR failures are explainable
  **field-semantics** differences (ex-dividend vs ex-date keying, `delisted_utc`
  = last trade vs close) that we consciously adopt and document — never silent.
- No lookahead: a bar at T never reflects corporate actions effective after T.
- Zero-cost buy-and-hold over the window reproduces each name's independent
  total return (price + cash-credited dividends, splits applied) within the
  same honesty band used by `tests/integration/test_backtest_sanity.py`.

Only after all of the above: scale to `--top-n 500` (flat files on Starter are
included in all paid plans — see FREEZE/data notes), then run the locked
decisive experiment per `docs/cross_sectional_factor_protocol.md`.

> **Amended 2026-08-04:** the earlier instruction to re-test
> `options_delta_neutral` with walk-forward + DSR is superseded by the
> pre-registered factor protocol (`cross_sectional_factor_protocol.md`, locked
> 2026-08-04). The live system stays dormant pending the gate verdict.
