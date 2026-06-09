"""Phase 1 strategy registry checks.

Walk-forward validation is intentionally disabled until Phase 4, after one
hypothesis has been researched and one clean strategy has been implemented.
"""

from __future__ import annotations

from quantcore.strategies.registry import list_strategies


def test_phase1_registry_is_empty() -> None:
    """All strategy implementations were removed in the Phase 1 strip."""
    assert list_strategies() == []


def test_phase1_walk_forward_test_suite_is_disabled() -> None:
    """No strategy configs remain until a Phase 4 strategy is registered."""
    assert True
