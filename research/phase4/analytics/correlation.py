"""Phase 4 — Signal Correlation Matrix (Stage 9).

Computes pairwise IC correlations between all 13 factors, then applies
hierarchical clustering to identify redundant factors.

Institutional standard: factors with IC correlation > 0.7 are considered
redundant — they capture the same underlying risk premium. Only the
higher-IC factor from each redundant pair should be retained.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REDUNDANCY_THRESHOLD = 0.70  # Pairwise IC correlation threshold for redundancy


@dataclass
class CorrelationResult:
    """Result of pairwise factor correlation analysis."""
    corr_matrix: pd.DataFrame       # Factor × Factor IC correlation matrix
    clusters: list[list[str]]        # Hierarchical clusters
    redundant_pairs: list[tuple[str, str, float]]  # (factor_a, factor_b, corr)
    independent_factors: list[str]   # Factors not in any redundant pair


def compute_daily_ic_panel(
    all_signals: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute daily IC for every factor, returning a (date × factor) panel.

    This is the "IC panel" — each column is a factor's daily IC time series,
    which we then correlate to measure factor inter-dependence.
    """
    from research.phase4.analytics.metrics import daily_ic_series

    ic_dict: dict[str, pd.Series] = {}
    for name, signal in all_signals.items():
        try:
            ic_series = daily_ic_series(signal, fwd_returns, method=method)
            if len(ic_series) > 20:
                ic_dict[name] = ic_series
        except Exception as e:
            log.warning("Cannot compute IC for %s: %s", name, e)

    return pd.DataFrame(ic_dict)


def compute_ic_correlation_matrix(
    ic_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise correlation of daily IC series between factors.

    This measures whether factors' cross-sectional predictions move together.
    High correlation = factors capture same risk premium → redundancy.
    """
    return ic_panel.corr()


def hierarchical_cluster(
    corr_matrix: pd.DataFrame,
    n_clusters: int | None = None,
) -> list[list[str]]:
    """Hierarchical agglomerative clustering of factors by IC correlation.

    Uses 1 - |corr| as distance. If n_clusters is None, uses automatic
    selection via the inconsistency coefficient (scipy default).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # Distance = 1 - |correlation| (absolute — sign doesn't matter for redundancy)
    dist = 1.0 - corr_matrix.abs()
    # Ensure diagonal is 0 and matrix is symmetric
    np.fill_diagonal(dist.values, 0.0)
    dist = (dist + dist.T) / 2.0  # force symmetry

    # Convert to condensed distance matrix
    condensed = squareform(dist.values, checks=False)

    # Ward linkage on the distance matrix
    Z = linkage(condensed, method="average")

    if n_clusters is None:
        # Auto-select: use inconsistency cutoff of 0.8
        n_clusters = max(2, min(len(corr_matrix) // 2, 6))

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Group factors by cluster
    clusters: dict[int, list[str]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(corr_matrix.index[i])

    return list(clusters.values())


def identify_redundant_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = REDUNDANCY_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Find all factor pairs with |IC correlation| > threshold."""
    redundant = []
    factors = list(corr_matrix.columns)
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            a, b = factors[i], factors[j]
            corr_val = float(corr_matrix.loc[a, b])
            if abs(corr_val) > threshold:
                redundant.append((a, b, corr_val))
    return sorted(redundant, key=lambda x: -abs(x[2]))


def select_independent_factors(
    all_factors: list[str],
    redundant_pairs: list[tuple[str, str, float]],
    factor_ics: dict[str, float],
) -> list[str]:
    """From each redundant pair, keep the factor with higher absolute IC.

    Returns the set of "independent" factors after redundancy removal.
    """
    dropped: set[str] = set()
    for a, b, corr_val in redundant_pairs:
        if a in dropped or b in dropped:
            continue  # Already handled by transitive removal
        # Keep the higher-IC factor
        ic_a = abs(factor_ics.get(a, 0.0))
        ic_b = abs(factor_ics.get(b, 0.0))
        if ic_a >= ic_b:
            dropped.add(b)
            log.info("  Redundant pair (%s, %s) corr=%.3f → dropping %s (IC=%.4f < %.4f)",
                      a, b, corr_val, b, ic_b, ic_a)
        else:
            dropped.add(a)
            log.info("  Redundant pair (%s, %s) corr=%.3f → dropping %s (IC=%.4f < %.4f)",
                      a, b, corr_val, a, ic_a, ic_b)

    independent = [f for f in all_factors if f not in dropped]
    return independent


def run_correlation_analysis(
    all_signals: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
    factor_ics: dict[str, float] | None = None,
) -> CorrelationResult:
    """Run the full correlation analysis pipeline.

    Args:
        all_signals:  dict mapping factor_name → (date × symbol) signal DataFrame.
        fwd_returns:  (date × symbol) 1-day forward returns.
        factor_ics:   Optional dict mapping factor_name → full-sample IC for redundancy resolution.

    Returns:
        CorrelationResult with matrix, clusters, redundant pairs, and independent factors.
    """
    factor_names = list(all_signals.keys())
    if factor_ics is None:
        factor_ics = {}

    # Step 1: Daily IC panel
    ic_panel = compute_daily_ic_panel(all_signals, fwd_returns)
    if ic_panel.empty:
        log.error("No factor ICs could be computed. Cannot run correlation analysis.")
        return CorrelationResult(
            corr_matrix=pd.DataFrame(),
            clusters=[],
            redundant_pairs=[],
            independent_factors=factor_names,
        )

    # Step 2: Correlation matrix
    corr_matrix = compute_ic_correlation_matrix(ic_panel)

    # Step 3: Hierarchical clustering
    n_clusters = max(2, len(factor_names) // 3)
    clusters = hierarchical_cluster(corr_matrix, n_clusters=n_clusters)

    # Step 4: Identify redundant pairs
    redundant_pairs = identify_redundant_pairs(corr_matrix)

    # Step 5: Select independent factors
    independent = select_independent_factors(factor_names, redundant_pairs, factor_ics)

    log.info(
        "Correlation analysis: %d factors, %d redundant pairs, %d independent factors",
        len(factor_names), len(redundant_pairs), len(independent),
    )

    return CorrelationResult(
        corr_matrix=corr_matrix,
        clusters=clusters,
        redundant_pairs=redundant_pairs,
        independent_factors=independent,
    )


def generate_correlation_report(
    result: CorrelationResult,
) -> str:
    """Generate SIGNAL_CORRELATION_REPORT.md as a markdown string."""
    lines: list[str] = []

    lines.append("# Signal Correlation Report")
    lines.append("")
    lines.append("## IC Correlation Matrix")
    lines.append("")
    lines.append("Pairwise correlation of daily IC time series between factors.")
    lines.append("High correlation (>0.7) indicates redundancy — same risk premium.")
    lines.append("")

    # Build correlation table
    factors = list(result.corr_matrix.columns)
    if factors:
        header = "| | " + " | ".join(factors) + " |"
        sep = "|---|" + "|".join(["------:" for _ in factors]) + "|"
        lines.append(header)
        lines.append(sep)
        for row_f in factors:
            cells = []
            for col_f in factors:
                val = result.corr_matrix.loc[row_f, col_f]
                if np.isnan(val):
                    cells.append("—")
                elif row_f == col_f:
                    cells.append("1.00")
                else:
                    cells.append(f"{val:.2f}")
            lines.append(f"| **{row_f}** | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Hierarchical Clusters")
    lines.append("")
    lines.append("Factors grouped by IC correlation distance (1 - |corr|):")
    lines.append("")
    for i, cluster in enumerate(result.clusters, 1):
        lines.append(f"**Cluster {i}:** {', '.join(cluster)}")
    lines.append("")

    # Redundancy analysis
    lines.append("---")
    lines.append("")
    lines.append("## Redundancy Analysis")
    lines.append("")
    lines.append(f"Threshold: |IC correlation| > {REDUNDANCY_THRESHOLD:.0%}")
    lines.append("")

    if result.redundant_pairs:
        lines.append("| Factor A | Factor B | IC Correlation | Verdict |")
        lines.append("|----------|----------|---------------:|---------|")
        for a, b, corr_val in result.redundant_pairs:
            lines.append(f"| {a} | {b} | {corr_val:+.3f} | REDUNDANT |")
    else:
        lines.append("No redundant pairs found. All factors are independent.")
    lines.append("")

    # Independent factors
    lines.append("---")
    lines.append("")
    lines.append("## Independent Factor Set")
    lines.append("")
    lines.append("After removing redundant factors (keeping higher-IC from each pair):")
    lines.append("")
    for f in result.independent_factors:
        lines.append(f"- {f}")
    lines.append("")
    if len(result.independent_factors) < len(factors):
        dropped_count = len(factors) - len(result.independent_factors)
        lines.append(f"**{dropped_count} factor(s) removed as redundant.**")

    return "\n".join(lines)
