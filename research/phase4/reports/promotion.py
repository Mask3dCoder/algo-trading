"""Phase 4 — Promotion Committee Report Generator (Stage 12).

Produces the final PROMOTION_COMMITTEE_REPORT.md with structured decisions
for each factor across all forward horizons.

Decision Logic (per horizon):
  PROMOTE:  IC significant + FDR survives + regime-robust + cost-viable + independent
  REFINE:   IC significant but fails 1-2 non-critical criteria
  HOLD:     Marginal IC — worth revisiting with better data or larger universe
  ARCHIVE:  No IC but economically sound theory — park for future data acquisition
  REJECT:   No IC and no economic rationale

Multi-Horizon Feature:
  The same factor is re-evaluated independently at h=1, h=5, and h=21.
  A factor that fails at h=1 but passes at h=21 is explicitly flagged as
  a "horizon rescue" — the reduced turnover and cost hurdle at longer
  holding periods enables viability.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Promotion Decisions ────────────────────────────────────────────────────────

PROMOTE = "PROMOTE"
REFINE = "REFINE"
HOLD   = "HOLD"
ARCHIVE = "ARCHIVE"
REJECT  = "REJECT"

DECISION_ORDER = [PROMOTE, REFINE, HOLD, ARCHIVE, REJECT]


@dataclass
class PromotionVote:
    """Structured promotion decision for one factor at one horizon."""
    factor_name:       str
    holding_days:     int
    decision:          str
    rationale:         str
    ic_evidence:        str
    blocking_issues:   list[str] = field(default_factory=list)
    next_steps:        list[str] = field(default_factory=list)


@dataclass
class HorizonResult:
    """Per-horizon statistical and decision results for one factor."""
    horizon:           int
    ic:                 float              = np.nan
    icir:              float              = np.nan
    reject_null:        bool              = False
    fdr_survives:       bool              = False
    regime_robust:      bool              = False
    cost_viable:        bool              = False
    is_independent:     bool              = True
    min_ic:             float             = np.nan
    permutation_pvalue: float             = np.nan
    t_pvalue:           float             = np.nan
    bootstrap_ci:       tuple[float, float] = (np.nan, np.nan)
    decision:           str               = HOLD
    blocking_issues:   list[str]         = field(default_factory=list)


@dataclass
class FactorPromotions:
    """All promotion results for one factor across all horizons."""
    factor_name:      str
    all_horizons:      list[int]        = field(default_factory=list)
    horizon_results:  dict[int, HorizonResult] = field(default_factory=dict)
    # Best decision across all horizons (for overall factor status)
    best_decision:    str               = HOLD
    econ_rationale:   str               = ""
    econ_score:       int               = 0


# ── Deciding ────────────────────────────────────────────────────────────────────

def decide_promotion(
    factor_name: str,
    ic_value: float,
    icir: float,
    reject_null: bool,
    fdr_survives: bool,
    regime_robust: bool,
    cost_viable: bool,
    is_independent: bool,
    econ_rationale: str = "",
) -> PromotionVote:
    """Apply the institutional promotion decision logic for a single factor.

    Decision tree (strict — all criteria must pass for PROMOTE):
      1. PROMOTE: IC significant ✓ AND FDR survives ✓ AND regime-robust ✓
                  AND cost-viable ✓ AND independent ✓
      2. REFINE:  IC significant ✓ but fails 1-2 non-critical criteria
      3. HOLD:    Positive IC but not statistically significant
      4. ARCHIVE: No IC but economically sound theory
      5. REJECT:  No IC and no redeeming economic rationale
    """
    blocking: list[str]  = []
    next_steps: list[str] = []

    if not reject_null:
        blocking.append("IC not statistically significant (t-test + permutation fail)")
    if not fdr_survives:
        blocking.append("Failed FDR multiple-testing correction")
    if not regime_robust:
        blocking.append("Not regime-robust (IC not consistent across regimes)")
    if not cost_viable:
        blocking.append("Not cost-viable (IC below min_IC for this holding horizon)")
    if not is_independent:
        blocking.append("Redundant with higher-IC factor (IC correlation > 0.7)")

    ic_str = f"IC={ic_value:+.4f}, ICIR={icir:.2f}" if not np.isnan(ic_value) else "IC not computable"
    if reject_null and fdr_survives:
        ic_str += " (significant ✓)"
    else:
        ic_str += " (not significant ✗)"

    if (reject_null and fdr_survives and regime_robust
            and cost_viable and is_independent):
        decision = PROMOTE
        rationale = (
            f"All 5 criteria met. Factor shows genuine predictive power after "
            f"multiple-testing adjustment, stable across market regimes, "
            f"survives realistic cost assumptions, and is not redundant."
        )
        next_steps = [
            "Proceed to paper-trading validation with position sizing",
            "Monitor out-of-sample IC decay",
            "Add to live config with initially small allocation",
        ]
    elif reject_null and fdr_survives and len(blocking) <= 2:
        decision = REFINE
        rationale = (
            f"IC statistically significant, but {len(blocking)} criteria unmet: "
            f"{'; '.join(blocking)}. Factor has signal that needs refinement."
        )
        next_steps = [
            "Address blocking issues: " + "; ".join(blocking),
            "Consider alternative signal construction",
            "Re-run with larger universe for better statistical power",
        ]
    elif not np.isnan(ic_value) and ic_value > 0 and not reject_null:
        decision = HOLD
        rationale = (
            f"Positive IC ({ic_value:+.4f}) but not statistically significant. "
            f"Signal may exist but is too weak for this universe. "
            f"Worth revisiting with 100+ stocks and longer sample."
        )
        next_steps = [
            "Expand universe to S&P 500 (more cross-sectional statistical power)",
            "Obtain point-in-time data to eliminate survivorship bias",
            "Test signal with sector-neutral construction",
        ]
    elif not np.isnan(ic_value) and ic_value <= 0 and econ_rationale:
        decision = ARCHIVE
        rationale = (
            f"IC ≤ 0 but strong economic rationale exists. The theory may be "
            f"correct but our data cannot capture it. Archive until better data."
        )
        next_steps = [
            "Acquire options IV surface data (ORATS/OptionsDX) for true VRP",
            "Acquire earnings SUE data (FactSet) for PEAD",
            "Re-evaluate when paid data sources are available",
        ]
    else:
        decision = REJECT
        rationale = (
            f"IC ≤ 0 and no strong economic rationale. Factor shows no predictive "
            f"power and is unlikely to with better data."
        )
        next_steps = ["Do not deploy; document in research log"]

    return PromotionVote(
        factor_name=factor_name,
        holding_days=-1,   # populated by caller
        decision=decision,
        rationale=rationale,
        ic_evidence=ic_str,
        blocking_issues=blocking,
        next_steps=next_steps,
    )


# ── Report Generation ──────────────────────────────────────────────────────────

def generate_promotion_report(
    all_factor_results: dict[str, FactorPromotions],
) -> str:
    """Generate PROMOTION_COMMITTEE_REPORT.md with multi-horizon comparison layout.

    Args:
        all_factor_results: dict mapping factor_name → FactorPromotions object
                             containing results for each horizon.
    """
    lines: list[str] = []

    decision_to_emoji = {
        PROMOTE: "🟢", REFINE: "🟡", HOLD: "⚪", ARCHIVE: "🔵", REJECT: "🔴",
    }

    # ── Header ─────────────────────────────────────────────────────────────
    lines.append("# Promotion Committee Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append(f"**Universe:** 100 large-cap US equities (S&P 100)")
    lines.append(f"**Period:** 2013-01-01 → 2024-12-31")
    lines.append("**Horizons evaluated:** 1-day, 5-day, 21-day forward returns")
    lines.append(f"**Significance Level:** α = 0.10 (t-test + permutation + BH-FDR)")
    lines.append("**Multi-Horizon Gate:** Factor must pass gate at any single horizon to PROMOTE/REFINE")
    lines.append("")

    # ── Summary Table ───────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    # Count per decision (count each factor once — best across horizons)
    counts = {d: 0 for d in DECISION_ORDER}
    rescued: list[str] = []   # factors rescued (HOLD/ARCHIVE at h=1 → PROMOTE/REFINE at h=21)

    for fp in all_factor_results.values():
        counts[fp.best_decision] += 1
        h1 = fp.horizon_results.get(1)
        h21 = fp.horizon_results.get(21)
        if h1 and h21:
            if h1.decision in (HOLD, ARCHIVE) and h21.decision in (PROMOTE, REFINE):
                rescued.append(fp.factor_name)

    lines.append("| Decision | Count | Factors |")
    lines.append("|----------|------:|---------|")
    for dec in DECISION_ORDER:
        factors_with = [fp.factor_name for fp in all_factor_results.values() if fp.best_decision == dec]
        factor_list = ", ".join(factors_with) if factors_with else "—"
        lines.append(f"| {decision_to_emoji[dec]} {dec} | {counts[dec]} | {factor_list} |")

    lines.append("")
    total = sum(counts.values())
    lines.append(f"**Total factors evaluated:** {total}")
    lines.append(f"**Promoted:** {counts[PROMOTE]} | **Refinable:** {counts[REFINE]} | "
                 f"**Pending:** {counts[HOLD]+counts[ARCHIVE]} | **Rejected:** {counts[REJECT]}")
    lines.append("")

    # Horizon-rescue table
    if rescued:
        lines.append("### Horizon Rescue — Factors Viabled at h=21 but not h=1")
        lines.append("")
        lines.append("Longer holding horizons reduce turnover → lower min_IC → viability.")
        lines.append("")
        lines.append("| Factor | h=1 Decision | h=1 IC | h=21 Decision | h=21 IC |")
        lines.append("|--------|:-------------:|-------:|:--------------:|---------:|")
        for name in rescued:
            fp = all_factor_results[name]
            h1 = fp.horizon_results.get(1)
            h21 = fp.horizon_results.get(21)
            if h1 and h21:
                ic_h1 = f"{h1.ic:+.4f}" if not np.isnan(h1.ic) else "—"
                ic_h21 = f"{h21.ic:+.4f}" if not np.isnan(h21.ic) else "—"
                lines.append(f"| {name} | {h1.decision} | {ic_h1} | **{h21.decision}** | {ic_h21} |")
        lines.append("")

    # ── Horizon Comparison Matrix ────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Horizon Comparison Matrix")
    lines.append("")
    lines.append(
        "Each factor evaluated at h=1, h=5, h=21 independently. "
        "Decision at each horizon uses the 3-way AND-gate "
        "(t-test p<0.10 AND permutation p<0.10 AND bootstrap CI lower > 0)."
    )
    lines.append("")

    lines.append(
        "| Factor | h | IC | Perm p | Min IC | Cost Viable | Decision |"
        "|------|--|------|--------|--------|-------------|----------|"
    )

    for fp in sorted(all_factor_results.values(), key=lambda x: x.factor_name):
        for h in [1, 5, 21]:
            hr = fp.horizon_results.get(h)
            if hr is None:
                continue
            ic_s = f"{hr.ic:+.4f}" if not np.isnan(hr.ic) else "—"
            pp_s = f"{hr.permutation_pvalue:.3f}" if not np.isnan(hr.permutation_pvalue) else "—"
            mi_s = f"{hr.min_ic:.4f}" if not np.isnan(hr.min_ic) else "—"
            cv_s = "✓" if hr.cost_viable else "✗"
            dec_emoji = decision_to_emoji.get(hr.decision, "⚪")
            row_marker = "**" if hr.decision in (PROMOTE, REFINE) else ""
            lines.append(
                f"| {fp.factor_name} | {h} | {ic_s} | {pp_s} | {mi_s} | {cv_s} | "
                f"{row_marker}{dec_emoji} {hr.decision}{row_marker} |"
            )
    lines.append("")

    # ── Detailed Per-Factor Reports ─────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Per-Factor Decisions")
    lines.append("")

    for fp in sorted(all_factor_results.values(), key=lambda x: x.factor_name):
        emoji = decision_to_emoji.get(fp.best_decision, "⚪")

        lines.append(f"### {emoji} {fp.factor_name}: {fp.best_decision}")
        lines.append("")
        lines.append(f"**Economic Rationale:** {fp.econ_rationale}")
        lines.append("")

        if fp.best_decision == PROMOTE:
            h_best = min(fp.horizon_results.items(), key=lambda x: x[1].decision == PROMOTE)
            hr = h_best[1]
            lines.append(f"**Passing Horizon:** h={h_best[0]}")
            lines.append(f"**IC at passing horizon:** {hr.ic:+.4f}")
            lines.append("")
            lines.append("**Next Steps:**")
            lines.append("- Proceed to paper-trading validation")
            lines.append("- Monitor OOS IC decay")
            lines.append("- Add to live config at initial small weight")
            lines.append("")
            continue

        lines.append("| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |")
        lines.append("|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|")

        for h in [1, 5, 21]:
            hr = fp.horizon_results.get(h)
            if hr is None:
                continue
            ic_s  = f"{hr.ic:+.4f}" if not np.isnan(hr.ic) else "—"
            pp_s  = f"{hr.permutation_pvalue:.3f}" if not np.isnan(hr.permutation_pvalue) else "—"
            tp_s  = f"{hr.t_pvalue:.3f}" if not np.isnan(hr.t_pvalue) else "—"
            ci_lo, ci_hi = hr.bootstrap_ci
            ci_s  = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]" if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else "—"
            fdr_s = "✓" if hr.fdr_survives else "✗"
            reg_s = "✓" if hr.regime_robust else "✗"
            cv_s  = "✓" if hr.cost_viable else "✗"
            lines.append(
                f"| {h} | {ic_s} | {pp_s} | {tp_s} | {ci_s} | {fdr_s} | {reg_s} | {cv_s} "
                f"| {hr.decision} |"
            )

        lines.append("")
        # Blocking issues from the BEST horizon only
        best_hr = None
        for dec in DECISION_ORDER:
            for h, hr in fp.horizon_results.items():
                if hr.decision == dec:
                    best_hr = hr
                    break
            if best_hr:
                break

        if best_hr and best_hr.blocking_issues:
            lines.append("**Blocking Issues:**")
            for issue in best_hr.blocking_issues:
                lines.append(f"- {issue}")
            lines.append("")

        lines.append("**Next Steps:**")
        if fp.best_decision == ARCHIVE:
            next_list = [
                "Acquire options IV surface data (ORATS) for true VRP evaluation",
                "Re-evaluate when paid data sources are available",
                "Longer horizon (h=21) may rescue this factor once turnover is reduced",
            ]
        elif fp.best_decision == HOLD:
            next_list = [
                "Expand universe to S&P 500 for more statistical power",
                "Obtain point-in-time data (Polygon/FirstRate) to fix survivorship bias",
                "Test sector-neutral variant",
            ]
        else:
            next_list = [
                "Do not deploy",
                "Document rejection in research log",
            ]
        for s in next_list:
            lines.append(f"- {s}")
        lines.append("")

    # ── Recommendations ──────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Overall Assessment")
    lines.append("")

    promoted = [fp.factor_name for fp in all_factor_results.values() if fp.best_decision == PROMOTE]
    refined  = [fp.factor_name for fp in all_factor_results.values() if fp.best_decision == REFINE]
    held     = [fp.factor_name for fp in all_factor_results.values() if fp.best_decision == HOLD]
    archived = [fp.factor_name for fp in all_factor_results.values() if fp.best_decision == ARCHIVE]

    if promoted:
        lines.append(f"🟢 **{len(promoted)} factor(s) promoted** — institutional-grade evidence.")
        lines.append(f"   Candidates: {', '.join(promoted)}")
        lines.append("   These are the only factors approved for paper-trading evaluation.")
    elif refined:
        lines.append(f"⚠ **No factors promoted**, but {len(refined)} deserve refinement.")
        lines.append(f"   Candidates: {', '.join(refined)}")
        lines.append("   No strategy code should be written until at least one factor reaches PROMOTE.")
    else:
        lines.append("⚠ **No factors promoted or refinable.**")
        lines.append("   This evaluation confirms: no deployable edge with current data/methodology.")
        lines.append("   All factors suffer from either IC non-significance or prohibitive turnover costs.")

    lines.append("")
    lines.append("### Recommendations")
    lines.append("")
    lines.append("1. **Do NOT deploy any strategy.**")
    lines.append("2. Expand universe to S&P 500 (100 stocks is still underpowered).")
    lines.append("3. Acquire point-in-time data (Polygon/FirstRate) for survivorship-free research.")
    lines.append("4. Acquire ORATS IV surface for true per-stock VRP evaluation.")
    if rescued:
        lines.append(f"5. **{len(rescued)} factors may be rescued at h=21** once turnover is corrected:")
        lines.append(f"   {', '.join(rescued)}")
        lines.append("   Re-test these with the 100-stock universe and h=21 horizon as first priority.")
    lines.append(f"6. Re-run Phase 4 pipeline when data infrastructure is upgraded.")
    lines.append("")

    return "\n".join(lines)