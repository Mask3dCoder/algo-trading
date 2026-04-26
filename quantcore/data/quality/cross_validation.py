"""Cross-validation convenience re-export."""

from quantcore.data.quality.stale_quotes import DiscrepancyReport, cross_validate, majority_vote

__all__ = ["DiscrepancyReport", "cross_validate", "majority_vote"]
