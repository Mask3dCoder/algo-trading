"""Max Pain calculation: the strike where option buyers lose the most at expiration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class MaxPainConfig:
    include_weekly_expiry: bool = True
    min_open_interest: int = 100


@dataclass
class MaxPainResult:
    symbol: str = ""
    expiration: str = ""
    max_pain: float = 0.0
    call_resistance: float = 0.0
    put_support: float = 0.0
    max_pain_distance_pct: float = 0.0
    magnet_force: float = 0.0


class MaxPainCalculator:
    """Calculate Max Pain — the strike price where option buyers lose the most.

    At expiration, call options are ITM when spot > strike, puts when spot < strike.
    Total pain = sum of ITM call payoff * call OI + ITM put payoff * put OI.
    Max Pain is the strike that minimizes total pain.
    """

    def __init__(self, config: MaxPainConfig | None = None) -> None:
        self.config = config or MaxPainConfig()

    def compute(self, spot: float, options: list[dict]) -> MaxPainResult:
        """Compute max pain from an option chain.

        Each option dict needs: strike, option_type (call/put), open_interest, expiration.
        """
        strikes = sorted(set(float(o["strike"]) for o in options))
        if not strikes:
            return MaxPainResult()

        total_pain_by_strike: dict[float, float] = {}
        for candidate_strike in strikes:
            total_pain = 0.0
            for opt in options:
                strike = float(opt["strike"])
                oi = float(opt.get("open_interest", 0))
                opt_type = str(opt.get("option_type", "")).lower()
                if oi <= 0 or oi < self.config.min_open_interest:
                    continue
                if opt_type == "call":
                    payoff = max(candidate_strike - strike, 0.0)
                elif opt_type == "put":
                    payoff = max(strike - candidate_strike, 0.0)
                else:
                    continue
                total_pain += payoff * oi
            total_pain_by_strike[candidate_strike] = total_pain

        max_pain = min(total_pain_by_strike, key=total_pain_by_strike.get)  # type: ignore[arg-type]

        # Call resistance: strike above spot with max call OI
        calls_above = [o for o in options if str(o.get("option_type", "")).lower() == "call" and float(o["strike"]) > spot]
        call_resistance = max(
            calls_above, key=lambda o: float(o.get("open_interest", 0))
        )["strike"] if calls_above else spot

        # Put support: strike below spot with max put OI
        puts_below = [o for o in options if str(o.get("option_type", "")).lower() == "put" and float(o["strike"]) < spot]
        put_support = max(
            puts_below, key=lambda o: float(o.get("open_interest", 0))
        )["strike"] if puts_below else spot

        distance_pct = abs(spot - max_pain) / spot * 100.0 if spot > 0 else 0.0

        # Magnet force: how concentrated the pain minimum is
        pain_range = max(total_pain_by_strike.values()) - min(total_pain_by_strike.values())
        magnet_force = 1.0 - (pain_range / (abs(max(total_pain_by_strike.values())) + 1e-10))
        magnet_force = max(0.0, min(1.0, float(magnet_force)))

        return MaxPainResult(
            symbol=options[0].get("symbol", "") if options else "",
            expiration=str(options[0].get("expiration", "")) if options else "",
            max_pain=float(max_pain),
            call_resistance=float(call_resistance),
            put_support=float(put_support),
            max_pain_distance_pct=float(distance_pct),
            magnet_force=float(magnet_force),
        )

    def compute_multi_expiry(
        self, spot: float, options_by_expiry: dict[str, list[dict]],
    ) -> dict[str, MaxPainResult]:
        """Compute max pain for each expiration in the chain."""
        results: dict[str, MaxPainResult] = {}
        for expiry, options in options_by_expiry.items():
            results[expiry] = self.compute(spot, options)
        return results

    def max_pain_trend(self, results: list[MaxPainResult]) -> str:
        """Determine trend of max pain over time: rising, falling, or stable."""
        if len(results) < 2:
            return "stable"
        values = [r.max_pain for r in results[-5:]]
        diffs = np.diff(values)
        if np.all(diffs > 0):
            return "rising"
        if np.all(diffs < 0):
            return "falling"
        return "stable"


def put_call_ratio(call_volume: float, put_volume: float) -> float:
    """Put/call volume ratio. > 1 = bearish, < 1 = bullish."""
    return put_volume / call_volume if call_volume > 0 else float("inf")


def put_call_oi_ratio(call_oi: float, put_oi: float) -> float:
    """Put/call open interest ratio."""
    return put_oi / call_oi if call_oi > 0 else float("inf")


def pcr_signal(pcr: float, historical_pcr: NDArray[np.floating]) -> dict[str, float]:
    """Compute z-score and percentile of current PCR vs history."""
    if len(historical_pcr) < 10:
        return {"pcr_zscore": 0.0, "pcr_percentile": 50.0}
    mean = float(np.mean(historical_pcr))
    std = float(np.std(historical_pcr, ddof=1))
    zscore = float((pcr - mean) / std) if std > 0 else 0.0
    percentile = float(np.mean(historical_pcr <= pcr) * 100.0)
    return {"pcr_zscore": zscore, "pcr_percentile": percentile}
