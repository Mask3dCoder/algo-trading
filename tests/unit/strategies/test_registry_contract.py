"""Phase 3a+3b: Registry contract and count tests.

Every registered strategy must be a concrete subclass of BaseStrategy and
the registry must contain the expected number of entries.
"""

from __future__ import annotations

import pytest

from quantcore.strategies.base import Strategy
from quantcore.strategies.registry import _registry, list_strategies


def _force_registry_populated() -> None:
    """Ensure imports have triggered all @register decorators."""
    if not _registry:
        import quantcore.strategies  # noqa: F401


_force_registry_populated()

EXPECTED_COUNT = 49


# ── Phase 3a: contract test ────────────────────────────────────────────────


def test_all_registered_are_strategy_subclasses() -> None:
    """Every entry in _registry must be a concrete subclass of Strategy."""
    for name, cls in _registry.items():
        assert isinstance(cls, type), f"'{name}' is not a type: {type(cls)}"
        assert issubclass(cls, Strategy), (
            f"'{name}' ({cls.__module__}.{cls.__qualname__}) "
            f"is not a subclass of Strategy"
        )


def test_all_registered_have_required_methods() -> None:
    """Every registered strategy must implement on_data (abstract)."""
    for name, cls in _registry.items():
        assert hasattr(cls, "on_data"), (
            f"'{name}' is missing required method 'on_data'"
        )


# ── Phase 3b: count test ──────────────────────────────────────────────────


def test_registry_count_matches_expected() -> None:
    """Registry must contain exactly the expected number of strategies."""
    actual = len(list_strategies())
    assert actual == EXPECTED_COUNT, (
        f"Expected {EXPECTED_COUNT} registered strategies, got {actual}. "
        f"Available: {list_strategies()}"
    )


def test_registry_has_no_duplicates() -> None:
    """Each registered name must map to exactly one class."""
    seen: dict[str, type] = {}
    for name, cls in _registry.items():
        if name in seen:
            pytest.fail(
                f"Duplicate registration for '{name}': "
                f"{seen[name]} vs {cls}"
            )
        seen[name] = cls


def test_list_strategies_is_sorted() -> None:
    """list_strategies() must return a sorted list."""
    names = list_strategies()
    assert names == sorted(names)
