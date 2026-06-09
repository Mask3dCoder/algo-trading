"""Survivorship-bias-free universe sampler.

Provides ``SurvivorshipFreeUniverse`` that returns the set of tickers
valid for any historical date, using the point-in-time constituent data
from ``HistoricalUniverseLoader``.
"""

from __future__ import annotations

import random
from datetime import date

from quantcore.data.sources.historical_universe import get_loader


class SurvivorshipFreeUniverse:
    """Point-in-time  universe with survivorship-bias correction.

    Args:
        index_name: Name of the index (e.g. ``"SP500"``).
        seed: Random seed for deterministic sampling of sub-universes.
    """

    def __init__(self, index_name: str = "SP500", seed: int = 42) -> None:
        self._index = index_name
        self._loader = get_loader()
        self._seed = seed
        self._rng = random.Random(seed)

    def get_tickers(self, as_of: date) -> list[str]:
        """Return constituent tickers as of ``as_of``.

        Falls back on current constituents if point-in-time data is
        unavailable, but **warns** about survivorship bias.
        """
        return self._loader.get_constituents(self._index, as_of)

    def sample(self, as_of: date, n: int = 100) -> list[str]:
        """Return a fixed random sample of ``n`` tickers from the universe at ``as_of``.

        The sample is deterministic (seeded) so it is reproducible across
        reruns.  If the universe has fewer than ``n`` tickers, all available
        tickers are returned.
        """
        tickers = self.get_tickers(as_of)
        if len(tickers) <= n:
            return tickers
        self._rng.shuffle(tickers)
        return sorted(tickers[:n])

    def seeded_sample(self, as_of: date, n: int = 100, seed: int = 42) -> list[str]:
        """Explicitly seeded version of ``sample``.

        Use this when you need a *different* sample from the same date.
        """
        rng = random.Random(seed)
        tickers = self.get_tickers(as_of)
        if len(tickers) <= n:
            return tickers
        rng.shuffle(tickers)
        return sorted(tickers[:n])
