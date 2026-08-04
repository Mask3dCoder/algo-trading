"""Phase 4 — Regime Analysis (Stage 6).

Decomposes factor IC performance across market regimes:
  1. Bull / Bear: SPY rolling 12-month return > 0 (Bull) or ≤ 0 (Bear)
  2. High Vol / Low Vol: VIX 21-day rolling rank > 75th pct (High) or < 25th pct (Low)
  3. Cross-regime: 4 buckets (Bull+LowVol, Bull+HighVol, Bear+LowVol, Bear+HighVol)

Institutional relevance: A factor that only works in one regime is fragile.
Surviving factors must show either:
  a) Positive IC across ALL regimes (robust), or
  b) Clear economic rationale for regime-specific behavior with evidence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Regime Labels ─────────────────────────────────────────────────────────────

REGIME_BULL_LOW   = "Bull+LowVol"
REGIME_BULL_HIGH  = "Bull+HighVol"
REGIME_BEAR_LOW   = "Bear+LowVol"
REGIME_BEAR_HIGH  = "Bear+HighVol"

BULL = "Bull"
BEAR = "Bear"
HIGH_VOL = "HighVol"
LOW_VOL = "LowVol"


# ── Regime Classification ─────────────────────────────────────────────────────

@dataclass
class RegimeMap:
    """Maps each date to its regime classification."""
    trend: pd.Series           # "Bull" or "Bear"
    vol_regime: pd.Series      # "HighVol", "LowVol", or "MidVol"
    composite: pd.Series       # 4-way composite
    date_range: tuple[str, str]

    def summary(self) -> dict[str, int]:
        """Count dates per composite regime."""
        return self.composite.value_counts().to_dict()


def classify_regimes(
    spy_returns: pd.Series,
    vix: pd.Series,
    trend_window: int = 252,
    vol_window: int = 21,
    high_vol_pct: float = 0.75,
    low_vol_pct: float = 0.25,
) -> RegimeMap:
    """Classify each date into market regimes.

    Args:
        spy_returns: SPY daily log returns (indexed by date).
        vix:         VIX closing level (indexed by date).
        trend_window: Rolling window for bull/bear classification (default 252 = 12m).
        vol_window:   Rolling window for vol regime (default 21 days).
        high_vol_pct: VIX percentile above which = high vol (default 75th).
        low_vol_pct:  VIX percentile below which = low vol (default 25th).

    Returns:
        RegimeMap with trend, vol_regime, and composite classifications.
    """
    # Align indices
    common_idx = spy_returns.index.intersection(vix.index)
    spy_ret = spy_returns.reindex(common_idx)
    vix_aligned = vix.reindex(common_idx)

    # ── Trend regime ───────────────────────────────────────────────────────
    # Rolling 12-month cumulative return of SPY
    spy_cum_ret = spy_ret.rolling(trend_window, min_periods=trend_window // 2).sum()
    trend = pd.Series(
        np.where(spy_cum_ret > 0, BULL, BEAR),
        index=common_idx,
        name="trend_regime",
    )

    # ── Vol regime ─────────────────────────────────────────────────────────
    # Rank VIX within a 252-day rolling window
    vix_pct = vix_aligned.rolling(252, min_periods=126).rank(pct=True)
    vol_labels = pd.Series("MidVol", index=common_idx, name="vol_regime")
    vol_labels[vix_pct >= high_vol_pct] = HIGH_VOL
    vol_labels[vix_pct <= low_vol_pct] = LOW_VOL

    # ── Composite ──────────────────────────────────────────────────────────
    composite = pd.Series("", index=common_idx, name="composite_regime")
    for i in range(len(composite)):
        t = trend.iloc[i]
        v = vol_labels.iloc[i]
        if v == LOW_VOL:
            composite.iloc[i] = f"{t}+LowVol"
        elif v == HIGH_VOL:
            composite.iloc[i] = f"{t}+HighVol"
        else:
            composite.iloc[i] = f"{t}+MidVol"  # Not used in primary analysis

    return RegimeMap(
        trend=trend,
        vol_regime=vol_labels,
        composite=composite,
        date_range=(str(common_idx[0].date()), str(common_idx[-1].date())),
    )


# ── Regime-Conditioned IC ─────────────────────────────────────────────────────

@dataclass
class RegimeICResult:
    """IC statistics conditioned on market regime."""
    factor_name: str
    regime_ics: dict[str, dict] = field(default_factory=dict)
    # Each dict: {"ic": float, "rank_ic": float, "n_dates": int, "coverage": float}
    robust: bool = False     # Positive IC in ALL 4 primary regimes
    stable: bool = False     # IC direction same in ≥3/4 regimes


def compute_regime_ic(
    factor_name: str,
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    regime_map: RegimeMap,
    min_dates: int = 50,
) -> RegimeICResult:
    """Compute IC for a single factor, conditioned on each regime.

    Args:
        factor_name: Name of the factor.
        signal:       (date × symbol) signal panel.
        fwd_returns:  (date × symbol) 1-day forward returns.
        regime_map:   Regime classification from classify_regimes().
        min_dates:    Minimum number of dates per regime to compute IC.

    Returns:
        RegimeICResult with per-regime IC statistics.
    """
    from research.phase4.analytics.metrics import pearson_ic, spearman_ic, daily_ic_series

    result = RegimeICResult(factor_name=factor_name)

    # Get daily IC series (Pearson and Spearman)
    ic_pearson = daily_ic_series(signal, fwd_returns, method="pearson")
    ic_spearman = daily_ic_series(signal, fwd_returns, method="spearman")

    # Align regime dates with IC dates
    common = ic_pearson.index.intersection(regime_map.composite.index)

    primary_regimes = [REGIME_BULL_LOW, REGIME_BULL_HIGH, REGIME_BEAR_LOW, REGIME_BEAR_HIGH]
    positive_count = 0
    consistent_direction = 0
    overall_sign = None

    for regime_name in primary_regimes + ["Bull", "Bear", HIGH_VOL, LOW_VOL]:
        # Determine which dates belong to this regime
        if regime_name in primary_regimes:
            mask = regime_map.composite.reindex(common) == regime_name
        elif regime_name == BULL:
            mask = regime_map.trend.reindex(common) == BULL
        elif regime_name == BEAR:
            mask = regime_map.trend.reindex(common) == BEAR
        elif regime_name == HIGH_VOL:
            mask = regime_map.vol_regime.reindex(common) == HIGH_VOL
        elif regime_name == LOW_VOL:
            mask = regime_map.vol_regime.reindex(common) == LOW_VOL
        else:
            continue

        regime_dates = common[mask]
        n_dates = len(regime_dates)

        if n_dates < min_dates:
            result.regime_ics[regime_name] = {
                "ic": float("nan"),
                "rank_ic": float("nan"),
                "n_dates": n_dates,
                "coverage": float(n_dates) / len(common) if len(common) > 0 else 0.0,
                "note": f"Insufficient dates ({n_dates} < {min_dates})",
            }
            continue

        regime_ic_p = ic_pearson.reindex(regime_dates).dropna()
        regime_ic_s = ic_spearman.reindex(regime_dates).dropna()

        mean_ic = float(regime_ic_p.mean())
        mean_rank_ic = float(regime_ic_s.mean())
        coverage = float(n_dates) / len(common) if len(common) > 0 else 0.0

        result.regime_ics[regime_name] = {
            "ic": round(mean_ic, 6),
            "rank_ic": round(mean_rank_ic, 6),
            "n_dates": n_dates,
            "coverage": round(coverage, 4),
        }

        # Track positives and direction consistency
        if regime_name in primary_regimes:
            if not np.isnan(mean_ic) and mean_ic > 0:
                positive_count += 1
            if overall_sign is None and not np.isnan(mean_ic):
                overall_sign = np.sign(mean_ic)
            if not np.isnan(mean_ic) and np.sign(mean_ic) == overall_sign:
                consistent_direction += 1

    result.robust = positive_count >= 3  # Positive IC in 3/4 or more primary regimes
    result.stable = consistent_direction >= 3

    return result


def generate_regime_report(
    all_regime_results: dict[str, RegimeICResult],
    regime_map: RegimeMap,
) -> str:
    """Generate REGIME_STABILITY_REPORT.md as a markdown string.

    Args:
        all_regime_results: dict mapping factor_name → RegimeICResult.
        regime_map:         RegimeMap with date range and summary.

    Returns:
        Markdown string for the regime stability report.
    """
    lines: list[str] = []

    lines.append("# Regime Stability Report")
    lines.append("")
    lines.append("## Regime Classification")
    lines.append("")
    lines.append(f"- **Date Range:** {regime_map.date_range[0]} → {regime_map.date_range[1]}")
    lines.append(f"- **Trend:** Bull = SPY 12-month rolling return > 0; Bear ≤ 0")
    lines.append(f"- **Vol:** HighVol = VIX ≥ 75th percentile; LowVol = VIX ≤ 25th percentile")
    lines.append("")
    summary = regime_map.summary()
    lines.append("### Regime Distribution")
    lines.append("")
    lines.append("| Regime | Dates | % of Total |")
    lines.append("|--------|-------:|-----------:|")
    total = sum(summary.values())
    for regime, count in sorted(summary.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"| {regime} | {count} | {pct:.1f}% |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Factor Regime Performance")
    lines.append("")

    # Build the main table
    primary_regimes = [REGIME_BULL_LOW, REGIME_BULL_HIGH, REGIME_BEAR_LOW, REGIME_BEAR_HIGH]
    header = "| Factor | " + " | ".join(primary_regimes) + " | Robust | Stable |"
    sep =    "|--------|" + "|".join(["-------:" for _ in primary_regimes]) + "|--------|--------|"
    lines.append(header)
    lines.append(sep)

    for factor_name, result in sorted(all_regime_results.items()):
        cells = []
        for regime in primary_regimes:
            ic_val = result.regime_ics.get(regime, {}).get("ic", float("nan"))
            if np.isnan(ic_val):
                cells.append("—")
            else:
                cells.append(f"{ic_val:+.4f}")
        robust = "✓" if result.robust else "✗"
        stable = "✓" if result.stable else "✗"
        lines.append(f"| {factor_name} | " + " | ".join(cells) + f" | {robust} | {stable} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Per-Factor Regime Analysis")
    lines.append("")

    for factor_name, result in sorted(all_regime_results.items()):
        lines.append(f"### {factor_name}")
        lines.append("")
        lines.append(f"- **Robust:** {'Yes — positive IC in ≥3/4 regimes' if result.robust else 'No — IC not consistently positive across regimes'}")
        lines.append(f"- **Stable:** {'Yes — IC direction consistent across ≥3/4 regimes' if result.stable else 'No — IC direction flips across regimes'}")
        lines.append("")

        detail_header = "| Regime | IC | Rank IC | N Dates | Coverage | Note |"
        detail_sep =    "|--------|------:|--------:|--------:|---------:|------|"
        lines.append(detail_header)
        lines.append(detail_sep)

        for regime_name, stats in result.regime_ics.items():
            ic_v = stats.get("ic", float("nan"))
            ric_v = stats.get("rank_ic", float("nan"))
            n_d = stats.get("n_dates", 0)
            cov = stats.get("coverage", 0)
            note = stats.get("note", "")
            ic_str = f"{ic_v:+.4f}" if not np.isnan(ic_v) else "—"
            ric_str = f"{ric_v:+.4f}" if not np.isnan(ric_v) else "—"
            lines.append(f"| {regime_name} | {ic_str} | {ric_str} | {n_d} | {cov:.1%} | {note} |")

        lines.append("")

    # Summary verdicts
    lines.append("---")
    lines.append("")
    lines.append("## Summary Verdicts")
    lines.append("")
    robust_factors = [name for name, r in all_regime_results.items() if r.robust]
    fragile_factors = [name for name, r in all_regime_results.items() if not r.robust and not r.stable]
    regime_dependent = [name for name, r in all_regime_results.items() if not r.robust and r.stable]

    if robust_factors:
        lines.append(f"**Robust (positive IC ≥3/4 regimes):** {', '.join(robust_factors)}")
    if regime_dependent:
        lines.append(f"**Regime-Dependent (stable direction but not consistently positive):** {', '.join(regime_dependent)}")
    if fragile_factors:
        lines.append(f"**Fragile (direction flips, no consistency):** {', '.join(fragile_factors)}")

    return "\n".join(lines)
