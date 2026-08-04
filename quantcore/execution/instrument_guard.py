# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
"""Instrument routing guard — Audit Mandate 2026-07-12 (Phase 4).

Physically prevents an OPTIONS strategy from routing an EQUITY order.

Root cause this guards against
------------------------------
`production_runner._submit_option_signal` historically built a plain equity
``Order(symbol="SPY", qty=contracts)`` and merely *tagged* it
``metadata={"instrument": "OPTION"}``.  The paper/Alpaca broker ignored the
tag and filled **SPY stock** at the equity spot (~$752), producing the
``BUY 1 SPY`` loop seen in ``reports/live_events.jsonl`` while the
``options_delta_neutral`` strategy believed it was short vol.

This module is broker-agnostic and side-effect-light so it can be unit tested
(``tests/test_option_routing.py``) without a live broker.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

KILL_SWITCH_PATH = os.environ.get("QUANTCORE_KILL_SWITCH", "/tmp/KILL_SWITCH")

# Canonical asset-type tokens.
ASSET_OPTION = "OPTION_CONTRACT"
ASSET_EQUITY = "EQUITY"


@dataclass(frozen=True)
class RouteDecision:
    """Result of a routing guard evaluation."""

    allowed: bool
    reason: str = ""
    killed: bool = False


def _trip_kill_switch(path: str = KILL_SWITCH_PATH) -> None:
    """Create the kill-switch sentinel file (idempotent)."""
    try:
        with open(path, "a", encoding="utf-8"):
            os.utime(path, None)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        log.error("Failed to write kill switch at %s: %s", path, exc)


def guard_route(
    *,
    strategy_type: str,
    asset_type: str,
    strategy_name: str = "unknown",
    kill_on_violation: bool = True,
    kill_switch_path: str = KILL_SWITCH_PATH,
) -> RouteDecision:
    """Return a :class:`RouteDecision`; trip the kill switch on a violation.

    Audit Rule #1: an ``OPTIONS`` strategy MUST trade ``OPTION_CONTRACT``.
    Any attempt to send it down the equity path is a mis-route: block the
    order and trip the kill switch (a mis-route means the strategy's real
    economic exposure is undefined - safest action is halt).

    Parameters
    ----------
    strategy_type : str
        Declared strategy type, e.g. ``"OPTIONS"`` or ``"EQUITY"``.
    asset_type : str
        Asset the pending order would actually trade
        (``"OPTION_CONTRACT"`` or ``"EQUITY"``).
    """
    stype = (strategy_type or "").strip().upper()
    atype = (asset_type or "").strip().upper()

    if stype == "OPTIONS" and atype != ASSET_OPTION:
        reason = (
            f"BLOCKING ORDER: {strategy_name} (OPTIONS) tried to trade "
            f"{atype or 'UNKNOWN'} instead of {ASSET_OPTION}."
        )
        log.critical(reason)
        # Mocked alert channel - CAPS print stands in for Telegram/Slack.
        print("\U0001f6a8 KILL SWITCH TRIGGERED: MISROUTING DETECTED \U0001f6a8")
        print(reason)
        killed = False
        if kill_on_violation:
            _trip_kill_switch(kill_switch_path)
            killed = True
        return RouteDecision(allowed=False, reason=reason, killed=killed)

    return RouteDecision(allowed=True, reason="ok")


def build_occ_symbol(underlying: str, expiry: str, right: str, strike: float) -> str:
    """Construct an OCC option symbol, e.g. ``SPY   240720C00750000``.

    Parameters
    ----------
    underlying : str
        Root symbol (space-padded to 6 chars per OCC spec).
    expiry : str
        Expiration ``YYMMDD``.
    right : str
        ``"C"`` or ``"P"``.
    strike : float
        Strike price (encoded as price * 1000, zero-padded to 8 digits).
    """
    root = f"{underlying.strip().upper():<6}"
    r = right.strip().upper()[0]
    if r not in ("C", "P"):
        raise ValueError(f"right must be C or P, got {right!r}")
    if len(expiry) != 6 or not expiry.isdigit():
        raise ValueError(f"expiry must be YYMMDD, got {expiry!r}")
    strike_int = int(round(float(strike) * 1000))
    return f"{root}{expiry}{r}{strike_int:08d}"
