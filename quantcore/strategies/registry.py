"""Strategy registry: register, lookup, and list all available strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantcore.strategies.base import Strategy, StrategyConfig

_registry: dict[str, type[Strategy]] = {}


def register(name: str) -> type:
    """Class decorator to register a strategy class by name."""
    def decorator(cls: type[Strategy]) -> type[Strategy]:
        if name in _registry:
            raise ValueError(f"Strategy '{name}' already registered")
        _registry[name] = cls
        return cls
    return decorator


def get(name: str) -> type[Strategy]:
    """Look up a strategy class by name."""
    if name not in _registry:
        raise KeyError(f"Strategy '{name}' not found. Available: {list(_registry.keys())}")
    return _registry[name]


def list_strategies() -> list[str]:
    """Return sorted list of registered strategy names."""
    return sorted(_registry.keys())


def create(name: str, config: StrategyConfig) -> Strategy:
    """Create a strategy instance from registry by name and config."""
    cls = get(name)
    return cls(config)


def clear() -> None:
    """Clear the registry (for testing)."""
    _registry.clear()
