"""Cross-validation convenience re-export."""

from quantcore.data.quality.stale_quotes import cross_validate, majority_vote, DiscrepancyReport

__all__ = ["cross_validate", "majority_vote", "DiscrepancyReport"]
