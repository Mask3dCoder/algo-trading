"""Phase 4 — Master Evaluation Script.

Orchestrates all 12 stages of the Institutional Signal Validation pipeline
with multi-horizon support:

  Stages 1-2:   Data + factor extraction (shared across all horizons)
  Stage  3:     IC / predictive analytics   [per horizon: h=1, 5, 21]
  Stage  4:     Bootstrap CI + FDR + permutation [per horizon]
  Stage  5:     Economic validation          [once]
  Stage  6:     Regime analysis               [per horizon]
  Stages 7-8:   Capacity & cost                [per horizon, with holding_days scaling]
  Stage  9:     Signal correlation             [once — IC corr is signal-intrinsic]
  Stage  10:    Factor library                  [once, includes min_IC per horizon]
  Stage  11:    Research journal               [once]
  Stage  12:    Promotion committee report      [multi-horizon matrix]

Usage:
    python -m research.phase4.run_phase4
    python research/phase4/run_phase4.py --quick
    python research/phase4/run_phase4.py --no-cache
    python research/phase4/run_phase4.py -h 1    # single horizon only
    python research/phase4/run_phase4.py --horizons 5,21  # specific horizons
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Bootstrap project root onto sys.path ──────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Logging ────────────────────────────────────────────────────────────────────
def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


log = logging.getLogger("phase4")

OUTPUT_DIR    = Path(__file__).resolve().parent / "output"
FACTOR_LIB_DIR = Path(__file__).resolve().parent / "factor_library"
JOURNAL_DIR    = Path(__file__).resolve().parent / "journal" / "output"


def _save_report(content: str, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    log.info("Saved: %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  ECONOMIC SCORING (Stage 5) — fixed once, not per-horizon
# ═══════════════════════════════════════════════════════════════════════════════

_ECON_SCORES = {
    "mom_1m":      (3, "Moderate — well-documented but weak in large-caps"),
    "mom_3m":      (4, "Strong — Jegadeesh-Titman, cross-sectional persistence documented"),
    "mom_6m":      (4, "Persistent intermediate-term momentum"),
    "mom_12_1":    (5, "Very strong — flagship academic anomaly, hundreds of replications"),
    "ts_mom":      (4, "Moskowitz 2012 — trend-following with economic rationale"),
    "rev_1m":      (3, "Short-term reversal, may be bid-ask bounce artifact"),
    "rev_1w":      (2, "Weak — dominated by microstructure at daily frequency"),
    "zscore_20":   (3, "Mean reversion with economic basis but fragile at large N"),
    "rsi_14":      (2, "Weak — popular but academically weak, often over-fitted"),
    "low_vol":     (4, "Strong — Blitz & Van Vliet 2007, leverage constraints theory"),
    "vrp_proxy":  (3, "Moderate — VRP exists but this is a noisy proxy"),
    "quality":     (3, "Moderate — Asness 2013, our proxy is too simple"),
    "vol_regime":  (2, "Weak — regime indicator, not a standalone alpha source"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase4(
    quick: bool = False,
    use_cache: bool = True,
    verbose: bool = False,
    horizons: Optional[tuple[int, ...]] = None,
) -> dict:
    """Execute the complete Phase 4 evaluation pipeline.

    Args:
        quick:     Reduce bootstrap/perm counts for fast test.
        use_cache: Use cached yfinance data if available.
        verbose:   Enable debug logging.
        horizons:  Tuple of forward horizons to evaluate. Default: (1, 5, 21).

    Returns:
        dict with all results.
    """
    _setup_logging(verbose)
    total_start = time.time()

    n_bootstrap  = 200 if quick else 1000
    n_permutations = 100 if quick else 500
    eval_horizons = horizons if horizons is not None else (1, 5, 21)

    # holding_days equals the horizon itself (rebalances every period)
    HOLDING_DAYS: dict[int, int] = {1: 1, 5: 5, 21: 21}

    results: dict = {"reports": {}, "factor_results": {}, "horizon_results": {}}

    # ── Stage 1: Data Download & Validation ──────────────────────────────────
    log.info("=" * 65)
    log.info("STAGE 1: Institutional Data Validation")
    log.info("=" * 65)
    stage_start = time.time()

    from research.phase4.data.loader import build_research_dataset
    from research.phase4.data.validator import validate_dataset

    try:
        dataset = build_research_dataset(
            start="2013-01-01", end="2024-12-31",
            fwd_horizons=eval_horizons,
            use_cache=use_cache,
        )
    except Exception as e:
        log.error("Data download failed: %s", e)
        log.error("Ensure yfinance is installed: pip install yfinance")
        return results

    prices       = dataset["prices"]
    volumes      = dataset["volumes"]
    fwd_all      = dataset["fwd"]          # dict[int, pd.DataFrame]
    benchmark    = dataset["benchmark"]
    vix          = dataset["vix"]
    symbols      = dataset["symbols"]
    train_mask   = dataset["train_mask"]

    log.info("Downloaded: %d symbols, %d trading days, horizons=%s",
             len(symbols), len(prices), eval_horizons)

    # Validate at h=1 (primary horizon) — costs are worst here
    quality_report = validate_dataset(prices, dataset["returns"], fwd_all[1], benchmark)
    quality_report.display()

    if not quality_report.passed:
        log.error("DATA QUALITY FAILED. Experiment blocked.")
        _save_report(
            "# Data Quality Report\n\nCRITICAL ISSUES FOUND — EXPERIMENT BLOCKED\n\n"
            + "\n".join(f"- {i}" for i in quality_report.critical_issues),
            "DATA_QUALITY_REPORT.md",
        )
        return results

    data_quality_md = _format_data_quality_report(quality_report)
    _save_report(data_quality_md, "DATA_QUALITY_REPORT.md")
    results["reports"]["data_quality"] = str(OUTPUT_DIR / "DATA_QUALITY_REPORT.md")
    log.info("Stage 1 complete: %.1fs", time.time() - stage_start)

    # ── Stage 2: Factor Extraction ────────────────────────────────────────────
    log.info("=" * 65)
    log.info("STAGE 2: Factor Extraction")
    log.info("=" * 65)
    stage_start = time.time()

    from research.phase4.signals.extractor import extract_all_factors, FACTOR_CATALOG

    all_signals = extract_all_factors(prices, vix)
    log.info("Extracted %d factors: %s", len(all_signals), list(all_signals.keys()))
    results["factor_names"] = list(all_signals.keys())

    # ── Attach Econ Scores ────────────────────────────────────────────────────
    factor_ics: dict[str, dict] = {}
    for fname in all_signals:
        score, rationale = _ECON_SCORES.get(fname, (2, "Undocumented"))
        factor_ics[fname] = {"econ_score": score, "econ_rationale": rationale}

    log.info("Stage 2 complete: %.1fs", time.time() - stage_start)

    # ══════════════════════════════════════════════════════════════════════════
    #  MULTI-HORIZON IC EVALUATION  (Stage 3 + 4 + Stage 7-8 per horizon)
    # ══════════════════════════════════════════════════════════════════════════

    from research.phase4.analytics.metrics import (
        pearson_ic, spearman_ic, daily_ic_series,
        ic_information_ratio, hit_rate, coverage, signal_persistence,
        quintile_return_analysis,
    )
    from research.phase4.analytics.stats import (
        run_full_stat_test, benjamini_hochberg_correction,
    )
    from research.phase4.analytics.regime import (
        classify_regimes, compute_regime_ic, generate_regime_report,
    )
    from research.phase4.analytics.capacity import (
        compute_capacity_report, generate_capacity_report,
    )
    from research.phase4.analytics.correlation import (
        run_correlation_analysis,
        generate_correlation_report,
    )

    # All per-horizon results: horizon → factor_name → HorizonResult
    all_horizon_results: dict[int, dict[str, dict]] = {}

    # Correlation baseline uses h=1 IC series (signal is same, only returns change)
    corr_signals_used = False   # run correlation once

    for h in eval_horizons:
        log.info("")
        log.info("=" * 63)
        log.info("══ HORIZON h=%d ══", h)
        log.info("=" * 63)

        fwd_h = fwd_all[h]
        holding_days = HOLDING_DAYS.get(h, h)

        # ── Stage 3: IC Analytics (h-specific) ─────────────────────────────
        log.info("STAGE 3: IC / Predictive Analytics (h=%d)", h)
        ic_stage_start = time.time()

        horizon_factor_data: dict[str, dict] = {}

        for fname, signal in all_signals.items():
            try:
                p_ic  = pearson_ic(signal, fwd_h)
                s_ic  = spearman_ic(signal, fwd_h)
                ic_series = daily_ic_series(signal, fwd_h, method="pearson")
                icir = ic_information_ratio(ic_series)
                hit  = hit_rate(ic_series)
                cov  = coverage(signal)
                pers = signal_persistence(signal)
                quint = quintile_return_analysis(signal, fwd_h)

                horizon_factor_data[fname] = {
                    "ic": p_ic, "rank_ic": s_ic,
                    "icir": icir, "hit_rate": hit,
                    "coverage": cov, "persistence": pers,
                    "quintiles": quint,
                    "daily_ic_series": ic_series,
                }
            except Exception as e:
                log.warning("  %s [h=%d]: IC failed — %s", fname, h, e)
                horizon_factor_data[fname] = {"ic": float("nan"), "rank_ic": float("nan")}

            log.info(
                "  %s [h=%d]: IC=%+.4f RankIC=%+.4f ICIR=%.2f Hit=%.1f%%",
                fname, h,
                horizon_factor_data[fname]["ic"],
                horizon_factor_data[fname]["rank_ic"],
                horizon_factor_data[fname]["icir"],
                horizon_factor_data[fname]["hit_rate"] * 100,
            )

        log.info("Stage 3 [h=%d] complete: %.1fs", h, time.time() - ic_stage_start)

        # ── Stage 4: Statistical Tests (h-specific) ───────────────────────
        log.info("STAGE 4: Statistical Validation (h=%d)", h)
        stat_stage_start = time.time()

        stat_results: dict = {}
        pvalues_for_fdr: dict[str, float] = {}

        for fname, data in horizon_factor_data.items():
            ic_series = data.get("daily_ic_series")
            if ic_series is not None and len(ic_series) > 20:
                res = run_full_stat_test(
                    fname + f"_h{h}", ic_series,
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                )
                stat_results[fname] = res
                pvalues_for_fdr[fname] = res.ic_pvalue

                data["reject_null"]   = res.reject_null
                data["bootstrap_ci"]  = (res.bootstrap_ci_lower, res.bootstrap_ci_upper)
                data["t_stat"]        = res.ic_tstat
                data["t_pvalue"]      = res.ic_pvalue
                data["permutation_pvalue"] = res.permutation_pvalue
            else:
                pvalues_for_fdr[fname] = 1.0

        # FDR correction (computed per-horizon independently)
        fdr_results = benjamini_hochberg_correction(dict(pvalues_for_fdr))
        for fname, survives in fdr_results.items():
            horizon_factor_data[fname]["fdr_survives"] = survives

        n_fdr = sum(1 for v in fdr_results.values() if v)
        log.info("BH-FDR [h=%d]: %d/%d factors survive", h, n_fdr, len(fdr_results))
        log.info("Stage 4 [h=%d] complete: %.1fs", h, time.time() - stat_stage_start)

        # ── Stage 6: Regime Analysis (h-specific) ────────────────────────────
        regime_stage_start = time.time()
        log.info("STAGE 6: Regime Analysis (h=%d)", h)
        regime_results_h: dict = {}
        if not vix.empty and not benchmark.empty:
            regime_map = classify_regimes(benchmark, vix)
            for fname, signal in all_signals.items():
                try:
                    r_ic = compute_regime_ic(fname, signal, fwd_h, regime_map)
                    regime_results_h[fname] = r_ic
                    horizon_factor_data[fname]["regime_robust"] = r_ic.robust
                    horizon_factor_data[fname]["regime_stable"] = r_ic.stable
                    horizon_factor_data[fname]["regime_ics"] = r_ic.regime_ics
                except Exception as e:
                    log.warning("Regime analysis [h=%d, %s] failed: %s", h, fname, e)
                    horizon_factor_data[fname]["regime_robust"] = False
            reg_md = generate_regime_report(regime_results_h, regime_map)
            _save_report(reg_md, f"REGIME_STABILITY_REPORT_h{h}.md")
        log.info("Stage 6 [h=%d] complete: %.1fs", h, time.time() - regime_stage_start)

        # ── Stage 7-8: Capacity & Cost (h-specific) ──────────────────────
        cap_stage_start = time.time()
        log.info("STAGE 7-8: Capacity & Cost Estimation (h=%d)", h)
        capacity_results_h: dict = {}
        for fname, signal in all_signals.items():
            ic_val = horizon_factor_data.get(fname, {}).get("ic", 0.0)
            est = compute_capacity_report(
                fname, signal, volumes, prices,
                ic_value=ic_val,
                holding_days=holding_days,
                round_trip_bps=10.0,
                target_annual_vol=0.15,
            )
            capacity_results_h[fname] = est
            horizon_factor_data[fname]["avg_adv_usd"]      = est.avg_adv_usd
            horizon_factor_data[fname]["max_capacity_usd"]  = est.max_capacity_usd
            horizon_factor_data[fname]["annual_turnover"]  = est.avg_turnover_annual
            horizon_factor_data[fname]["annual_cost_bps"]  = est.annual_cost_bps
            horizon_factor_data[fname]["min_ic"]           = est.min_ic_to_survive
            horizon_factor_data[fname]["cost_viable"]      = est.viable

        cap_md = generate_capacity_report(capacity_results_h)
        _save_report(cap_md, f"CAPACITY_COST_REPORT_h{h}.md")
        log.info("Stage 7-8 [h=%d] complete: %.1fs", h, time.time() - cap_stage_start)

        # ── Stage 9: Correlation (run once — use h=1 IC series) ───────────
        if not corr_signals_used:
            log.info("")
            log.info("STAGE 9: Signal Correlation Analysis (baseline: h=1 IC series)")
            corr_stage_start = time.time()

            factor_ic_map = {n: d.get("ic", 0.0) for n, d in horizon_factor_data.items()}
            corr_result = run_correlation_analysis(all_signals, fwd_all[1], factor_ics=factor_ic_map)

            # Apply correlation info to ALL horizon data (it's signal-intrinsic)
            for fname in horizon_factor_data:
                if fname in corr_result.corr_matrix.columns:
                    row = corr_result.corr_matrix[fname].drop(fname, errors="ignore")
                    if not row.empty:
                        abs_row = row.abs()
                        max_name = abs_row.idxmax()
                        max_val  = float(row[max_name])
                        for _h in eval_horizons:
                            if _h not in all_horizon_results:
                                all_horizon_results[_h] = {}
                            if fname not in all_horizon_results[_h]:
                                all_horizon_results[_h][fname] = {}
                            all_horizon_results[_h][fname]["max_corr_with"]    = max_name
                            all_horizon_results[_h][fname]["max_corr_value"]   = max_val
                            all_horizon_results[_h][fname]["is_independent"] = (
                                fname in corr_result.independent_factors
                            )

            corr_md = generate_correlation_report(corr_result)
            _save_report(corr_md, "SIGNAL_CORRELATION_REPORT.md")
            log.info("Stage 9 complete: %.1fs", time.time() - corr_stage_start)
            corr_signals_used = True

        # ── Assemble HorizonResult for Stage 12 ──────────────────────────────
        from research.phase4.reports.promotion import HorizonResult, decide_promotion

        for fname, data in horizon_factor_data.items():
            stat = stat_results.get(fname)
            if fname not in all_horizon_results:
                all_horizon_results[h][fname] = {}
            hr = HorizonResult(
                horizon=h,
                ic=data.get("ic", float("nan")),
                icir=data.get("icir", float("nan")),
                reject_null=data.get("reject_null", False),
                fdr_survives=data.get("fdr_survives", False),
                regime_robust=data.get("regime_robust", False),
                cost_viable=data.get("cost_viable", False),
                is_independent=all_horizon_results[h][fname].get("is_independent", True),
                min_ic=data.get("min_ic", float("nan")),
                permutation_pvalue=stat.permutation_pvalue if stat else float("nan"),
                t_pvalue=stat.ic_pvalue if stat else float("nan"),
                bootstrap_ci=data.get("bootstrap_ci", (float("nan"), float("nan"))),
            )

            vote = decide_promotion(
                factor_name=fname,
                ic_value=hr.ic,
                icir=hr.icir,
                reject_null=hr.reject_null,
                fdr_survives=hr.fdr_survives,
                regime_robust=hr.regime_robust,
                cost_viable=hr.cost_viable,
                is_independent=hr.is_independent,
                econ_rationale=factor_ics.get(fname, {}).get("econ_rationale", ""),
            )

            hr.decision = vote.decision
            hr.blocking_issues = vote.blocking_issues

            # Store in both per-horizon dict and in master factor_ics
            all_horizon_results[h][fname] = hr

            if fname not in factor_ics:
                factor_ics[fname].update({})
            factor_ics[fname][f"ic_h{h}"]         = data.get("ic", float("nan"))
            factor_ics[fname][f"reject_null_h{h}"] = hr.reject_null
            factor_ics[fname][f"permutation_pvalue_h{h}"] = hr.permutation_pvalue
            factor_ics[fname][f"min_ic_h{h}"]      = hr.min_ic
            factor_ics[fname][f"cost_viable_h{h}"] = hr.cost_viable
            factor_ics[fname][f"fdr_survives_h{h}"] = hr.fdr_survives
            factor_ics[fname][f"decision_h{h}"]    = hr.decision
            factor_ics[fname][f"regime_robust_h{h}"] = hr.regime_robust

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 12: Promotion Committee — multi-horizon
    # ══════════════════════════════════════════════════════════════════════════
    log.info("")
    log.info("STAGE 12: Promotion Committee Report")
    log.info("=" * 65)

    from research.phase4.reports.promotion import (
        FactorPromotions, generate_promotion_report, DECISION_ORDER,
        PROMOTE, REFINE, HOLD, ARCHIVE, REJECT,
    )

    # Build FactorPromotions per factor
    all_factor_promotions: dict[str, FactorPromotions] = {}

    for fname in all_signals:
        fp = FactorPromotions(
            factor_name=fname,
            all_horizons=list(eval_horizons),
            horizon_results={
                h: all_horizon_results.get(h, {}).get(fname, None)
                for h in eval_horizons
            },
            econ_rationale=factor_ics.get(fname, {}).get("econ_rationale", ""),
            econ_score=factor_ics.get(fname, {}).get("econ_score", 0),
        )

        # Best decision across horizons
        best = HOLD
        for dec in DECISION_ORDER:
            for h in eval_horizons:
                hr = fp.horizon_results.get(h)
                if hr and hr.decision == dec:
                    best = dec
                    break
            if best != HOLD:
                break
        fp.best_decision = best
        all_factor_promotions[fname] = fp

    promotion_md = generate_promotion_report(all_factor_promotions)
    _save_report(promotion_md, "PROMOTION_COMMITTEE_REPORT.md")
    results["reports"]["promotion"] = str(OUTPUT_DIR / "PROMOTION_COMMITTEE_REPORT.md")

    # ── Final summary ────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    promoted = [fp.factor_name for fp in all_factor_promotions.values() if fp.best_decision == "PROMOTE"]
    refined  = [fp.factor_name for fp in all_factor_promotions.values() if fp.best_decision == "REFINE"]
    held     = [fp.factor_name for fp in all_factor_promotions.values() if fp.best_decision == "HOLD"]
    archived = [fp.factor_name for fp in all_factor_promotions.values() if fp.best_decision == "ARCHIVE"]
    rejected = [fp.factor_name for fp in all_factor_promotions.values() if fp.best_decision == "REJECT"]

    print("\n" + "=" * 65)
    print("  PHASE 4 RESULTS SUMMARY  (multi-horizon)")
    print("=" * 65)
    print(f"  Universe:    {len(symbols)} stocks  |  Period: 2013-2024")
    print(f"  Horizons:    {eval_horizons}  |  Bootstrap={n_bootstrap}  Perm={n_permutations}")
    print(f"  Factors:     {len(all_signals)}")
    print("-" * 65)
    print(f"  PROMOTE:   {len(promoted):>3}  {promoted  if promoted  else '—'}")
    print(f"  REFINE:    {len(refined):>3}  {refined   if refined   else '—'}")
    print(f"  HOLD:      {len(held):>3}  {held      if held      else '—'}")
    print(f"  ARCHIVE:   {len(archived):>3}  {archived if archived else '—'}")
    print(f"  REJECT:    {len(rejected):>3}  {rejected if rejected else '—'}")
    print("=" * 65)
    print(f"  Wall time: {total_time:.1f}s")
    print(f"  Reports:   {OUTPUT_DIR}")
    print("=" * 65)

    results["factor_results"]     = factor_ics
    results["all_horizon_results"] = all_horizon_results
    results["factor_promotions"]    = all_factor_promotions
    results["total_seconds"]        = total_time

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  REPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _format_data_quality_report(report) -> str:
    lines = [
        "# Data Quality Report",
        "",
        f"**Status:** {'PASSED' if report.passed else 'FAILED'}",
        f"**Universe:** {report.n_symbols} large-cap US equities",
        f"**Trading Days:** {report.n_dates:,}",
        f"**Date Range:** {report.date_range[0]} → {report.date_range[1]}",
        f"**Avg Cross-Sectional Coverage:** {report.avg_cross_sectional_coverage:.1%}",
        f"**Look-ahead Bias:** {'DETECTED' if report.lookahead_bias_detected else 'None detected'}",
        "",
        "## Survivorship Bias Declaration",
        "",
        report.survivorship_bias_warning or (
            "Data sourced from yfinance (adjusted close, split/dividend adjusted). "
            "Back-tested results may be optimistic relative to live trading."
        ),
        "",
    ]
    if report.critical_issues:
        lines.append("## Critical Issues")
        lines.append("")
        for i in report.critical_issues:
            lines.append(f"- {i}")
        lines.append("")
    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Per-Symbol Quality")
    lines.append("")
    lines.append("| Symbol | Missing % | Outlier Days |")
    lines.append("|--------|----------:|-------------:|")
    for sym in sorted(report.missing_pct.keys()):
        miss = report.missing_pct[sym]
        outliers = report.outlier_return_days.get(sym, 0)
        lines.append(f"| {sym} | {miss:.2%} | {outliers} |")
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: Institutional Signal Validation — Multi-Horizon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Reduced bootstrap/perm (fast test)")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download from yfinance")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,5,21",
        help="Comma-separated forward horizons to evaluate (default: 1,5,21)",
    )
    args = parser.parse_args()

    h_tuple = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())

    run_phase4(
        quick=args.quick,
        use_cache=not args.no_cache,
        verbose=args.verbose,
        horizons=h_tuple,
    )