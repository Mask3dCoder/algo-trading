"""Phase 4 — Research Journal Auto-Generator (Stage 11).

Generates a structured .md file per hypothesis mapping, including:
  - Findings summary
  - Statistical evidence
  - Failure analysis (if applicable)
  - Lessons learned
  - Next steps

The journal is the human-readable research artifact that accompanies
the machine-readable factor library.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

JOURNAL_DIR = Path(__file__).resolve().parent

# ── Hypothesis catalog ────────────────────────────────────────────────────────

HYPOTHESIS_MAP = {
    "H001": {
        "title": "Cross-Sectional Momentum (Jegadeesh-Titman)",
        "factors": ["mom_1m", "mom_3m", "mom_6m", "mom_12_1", "ts_mom"],
        "economic_rationale": (
            "Underreaction: investors gradually incorporate new information into prices. "
            "Herding: trend-following reinforces momentum until reversal. "
            "The 12-1 month skip avoids contamination from short-term reversal."
        ),
        "prior_expectation": "Positive IC, stronger in 3-12m horizons, weaker in 1m due to reversal contamination.",
        "failure_modes": [
            "Momentum crash during sharp reversals (Mar 2009, Mar 2020)",
            "High turnover increases costs beyond IC benefit",
            "IC concentrated in small/mid-cap — weak in large-cap universe",
        ],
    },
    "H002": {
        "title": "Short-Term Mean Reversion",
        "factors": ["rev_1w", "rev_1m", "zscore_20", "rsi_14"],
        "economic_rationale": (
            "Overreaction: investors overreact to short-term news, creating "
            "predictable reversion. Liquidity provision: market makers demand "
            "compensation for absorbing order flow."
        ),
        "prior_expectation": "Negative 1-week IC (reversal), stronger in high-turnover stocks.",
        "failure_modes": [
            "Reversal profits dominated by bid-ask bounce (not captured in daily close data)",
            "Costs from high turnover may exceed signal profits",
            "Signal decay too fast — 1-week IC may not survive execution lag",
        ],
    },
    "H003": {
        "title": "Volatility Risk Premium (VRP Proxy)",
        "factors": ["vrp_proxy", "vol_regime"],
        "economic_rationale": (
            "Implied vol systematically exceeds realized vol due to: "
            "(1) risk aversion — investors overpay for downside protection, "
            "(2) volatility selling as insurance analog. The VRP is one of the "
            "most documented risk premia in options literature."
        ),
        "prior_expectation": "Positive IC (VIX > RV → premium available to sellers), regime-conditioned.",
        "failure_modes": [
            "VIX is SPX-specific — poor proxy for individual stock IV",
            "Equity proxy cannot capture the true options-based VRP",
            "Tail events (vol spikes) can wipe months of premium",
        ],
    },
    "H004": {
        "title": "Low Volatility Anomaly",
        "factors": ["low_vol"],
        "economic_rationale": (
            "Blitz & Van Vliet (2007): Low-beta stocks earn higher risk-adjusted returns. "
            "Causes: leverage constraints, benchmarking, lottery preference. "
            "Institutional investors constrained to hold high-beta stocks, overpaying for risk."
        ),
        "prior_expectation": "Positive IC (low vol → higher future returns), slow signal, low turnover.",
        "failure_modes": [
            "IC vs. Sharpe trade-off: high Sharpe but low absolute return",
            "Underperforms in strong bull markets (missing upside)",
            "Concentrated in utilities/REITs — sector bet disguised as factor",
        ],
    },
    "H005": {
        "title": "Quality Factor (Risk-Adjusted Return)",
        "factors": ["quality"],
        "economic_rationale": (
            "Asness, Frazzini, Pedersen (2013): Quality Minus Junk. "
            "High-quality firms (profitable, stable, low-leverage) earn higher "
            "risk-adjusted returns than theory predicts. Our proxy uses "
            "Sharpe-as-factor (return/vol) as a simple quality measure."
        ),
        "prior_expectation": "Weak to moderate positive IC. Close to low_vol factor.",
        "failure_modes": [
            "Quality proxy too simple — real QMJ uses ROE, earnings stability, leverage",
            "High correlation with low_vol → likely redundant",
            "Data-mined: trailing Sharpe is a noisy proxy for true quality",
        ],
    },
}


def _ic_verdict(ic_value: float, reject_null: bool, ci_lower: float) -> str:
    """Generate a human-readable IC verdict."""
    if np.isnan(ic_value):
        return "INCONCLUSIVE — Could not compute IC"
    if not reject_null:
        return f"WEAK — IC={ic_value:+.4f} but not statistically significant"
    if ci_lower <= 0:
        return f"MARGINAL — IC={ic_value:+.4f} but CI includes zero"
    return f"CONFIRMED — IC={ic_value:+.4f} (statistically significant)"


def generate_hypothesis_journal(
    hypothesis_id: str,
    factor_results: dict[str, dict],
) -> str:
    """Generate a research journal .md for a single hypothesis.

    Args:
        hypothesis_id: "H001" through "H005".
        factor_results: dict mapping factor_name → evaluation result dict.
            Keys: ic, rank_ic, icir, hit_rate, reject_null, bootstrap_ci,
            regime_robust, cost_viable, etc.

    Returns:
        Markdown string of the journal entry.
    """
    hypo = HYPOTHESIS_MAP.get(hypothesis_id, {})
    if not hypo:
        return f"# {hypothesis_id}: Unknown hypothesis\n"

    lines: list[str] = []
    lines.append(f"# {hypothesis_id}: {hypo['title']}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append(f"**Factors:** {', '.join(hypo['factors'])}")
    lines.append("")

    # ── Economic Rationale ─────────────────────────────────────────────────
    lines.append("## Economic Rationale")
    lines.append("")
    lines.append(hypo["economic_rationale"])
    lines.append("")
    lines.append(f"**Prior Expectation:** {hypo['prior_expectation']}")
    lines.append("")

    # ── Findings ───────────────────────────────────────────────────────────
    lines.append("## Findings")
    lines.append("")
    lines.append("| Factor | IC | Rank IC | ICIR | Hit Rate | Stat. Sig. | CI (95%) | Verdict |")
    lines.append("|--------|------:|--------:|-----:|---------:|-----------:|----------|---------|")

    confirmed = []
    marginal = []
    weak = []

    for factor_name in hypo["factors"]:
        data = factor_results.get(factor_name, {})
        ic = data.get("ic", float("nan"))
        rank_ic = data.get("rank_ic", float("nan"))
        icir = data.get("icir", float("nan"))
        hit = data.get("hit_rate", float("nan"))
        reject = data.get("reject_null", False)
        ci = data.get("bootstrap_ci", (0, 0))

        verdict = _ic_verdict(ic, reject, ci[0])

        ic_str = f"{ic:+.4f}" if not np.isnan(ic) else "—"
        ric_str = f"{rank_ic:+.4f}" if not np.isnan(rank_ic) else "—"
        icir_str = f"{icir:.2f}" if not np.isnan(icir) else "—"
        hit_str = f"{hit:.1%}" if not np.isnan(hit) else "—"
        sig_str = "✓" if reject else "✗"
        ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if not np.isnan(ci[0]) else "—"

        lines.append(f"| {factor_name} | {ic_str} | {ric_str} | {icir_str} | {hit_str} | {sig_str} | {ci_str} | {verdict.split('—')[0].strip()} |")

        if "CONFIRMED" in verdict:
            confirmed.append(factor_name)
        elif "MARGINAL" in verdict:
            marginal.append(factor_name)
        else:
            weak.append(factor_name)

    lines.append("")

    # ── Summary ─────────────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    if confirmed:
        lines.append(f"**Confirmed factors:** {', '.join(confirmed)}")
    if marginal:
        lines.append(f"**Marginal factors:** {', '.join(marginal)}")
    if weak:
        lines.append(f"**Weak/Inconclusive factors:** {', '.join(weak)}")
    lines.append("")

    # ── Regime Analysis ─────────────────────────────────────────────────────
    lines.append("## Regime Stability")
    lines.append("")
    any_robust = False
    for factor_name in hypo["factors"]:
        data = factor_results.get(factor_name, {})
        robust = data.get("regime_robust", False)
        stable = data.get("regime_stable", False)
        if robust:
            lines.append(f"- **{factor_name}:** Robust across regimes (✓)")
            any_robust = True
        elif stable:
            lines.append(f"- **{factor_name}:** Directionally stable but not consistently positive")
            any_robust = True
        else:
            lines.append(f"- **{factor_name}:** Fragile — IC varies by regime (✗)")
    lines.append("")

    # ── Capacity & Cost ─────────────────────────────────────────────────────
    lines.append("## Capacity & Cost Viability")
    lines.append("")
    for factor_name in hypo["factors"]:
        data = factor_results.get(factor_name, {})
        viable = data.get("cost_viable", False)
        cost_bps = data.get("annual_cost_bps", 0.0)
        capacity = data.get("max_capacity_usd", 0.0)
        lines.append(
            f"- **{factor_name}:** "
            f"{'✓ Cost-viable' if viable else '✗ Not cost-viable'} "
            f"(Cost: {cost_bps:.1f} bps/yr, Capacity: ${capacity/1e6:.0f}M)"
        )
    lines.append("")

    # ── Failure Analysis ────────────────────────────────────────────────────
    lines.append("## Failure Analysis")
    lines.append("")
    if not confirmed:
        lines.append("⚠ **No factors confirmed statistically significant.**")
        lines.append("")
        lines.append("Potential root causes:")
        for mode in hypo.get("failure_modes", []):
            lines.append(f"- {mode}")
    else:
        lines.append(f"✓ **{len(confirmed)} of {len(hypo['factors'])} factors confirmed.**")
        if weak:
            lines.append(f"  Weak factors ({', '.join(weak)}) may fail due to:")
            for mode in hypo.get("failure_modes", []):
                lines.append(f"- {mode}")
    lines.append("")

    # ── Lessons Learned ─────────────────────────────────────────────────────
    lines.append("## Lessons Learned")
    lines.append("")
    if confirmed:
        lines.append("1. The confirmed factors provide genuine cross-sectional predictive power.")
        lines.append("2. Bootstrap CI and FDR correction reduce false discovery risk.")
        if not any_robust:
            lines.append("3. ⚠ No factor is regime-robust — survival in all market conditions is uncertain.")
    else:
        lines.append("1. Signal extraction is correct, but no factor survives statistical validation.")
        lines.append("2. This does NOT mean the economic rationale is wrong — it may mean:")
        lines.append("   - The signal is too weak for this universe size (25 stocks)")
        lines.append("   - The 10-year sample includes regime shifts that dilute average IC")
        lines.append("   - Transaction costs and implementation lag erase the predicted edge")
        lines.append("3. **Larger universe (100+ stocks) is needed for reliable cross-sectional IC estimation.**")
    lines.append("")

    # ── Next Steps ──────────────────────────────────────────────────────────
    lines.append("## Next Steps")
    lines.append("")
    if confirmed:
        lines.append("1. Promote confirmed factors to paper-trading evaluation.")
        lines.append("2. Validate regime robustness with extended out-of-sample period.")
        lines.append("3. Check correlation with existing promoted factors.")
    else:
        lines.append("1. Expand universe to 100+ stocks for statistical power.")
        lines.append("2. Obtain point-in-time data to eliminate survivorship bias.")
        lines.append("3. Consider alternative signal construction (nonlinear, interaction).")
        lines.append("4. Do NOT deploy any strategy until at least one factor is confirmed with cost viability.")
    lines.append("")

    return "\n".join(lines)


def generate_all_journals(
    all_factor_results: dict[str, dict],
    output_dir: Path | None = None,
) -> list[Path]:
    """Generate journal .md files for all hypotheses.

    Args:
        all_factor_results: dict mapping factor_name → evaluation result dict.
        output_dir: Override output directory.

    Returns:
        List of paths to generated journal files.
    """
    if output_dir is None:
        output_dir = JOURNAL_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for hypo_id in ["H001", "H002", "H003", "H004", "H005"]:
        content = generate_hypothesis_journal(hypo_id, all_factor_results)
        filepath = output_dir / f"{hypo_id}_journal.md"
        filepath.write_text(content, encoding="utf-8")
        log.info("Generated journal: %s → %s", hypo_id, filepath)
        paths.append(filepath)

    return paths
