"""Risk-based trading signals: volatility gating, risk-on/off, stress indicators.

Signals derived from risk metrics that can gate, scale, or override strategy signals:
- Volatility targeting / scaling
- Risk-on / risk-off regime detection
- Drawdown protection signals
- Correlation-based hedging triggers
- Tail-risk hedging signals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class RiskSignal:
    """A risk-derived signal for strategy gating or position sizing."""
    name: str
    value: float
    direction: str = "neutral"  # long, short, neutral, reduce, flatten
    confidence: float = 0.0  # 0-1
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskRegime:
    """Aggregated risk regime assessment."""
    regime: str = "normal"  # calm, normal, cautious, stressed, crisis
    risk_on: bool = True
    score: float = 0.5  # 0 = max defensive, 1 = max aggressive
    vol_target_scale: float = 1.0
    drawdown_scale: float = 1.0
    correlation_hedge_ratio: float = 0.0
    tail_hedge_ratio: float = 0.0
    signals: list[RiskSignal] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Volatility targeting
# ---------------------------------------------------------------------------


def volatility_target_scale(
    returns: NDArray[np.floating],
    target_vol: float = 0.15,  # 15% annualized
    current_vol: float | None = None,
    window: int = 63,
    max_scale: float = 2.0,
    min_scale: float = 0.1,
) -> float:
    """Compute position scaling factor for volatility targeting.

    scale = target_vol / realized_vol, clamped to [min_scale, max_scale].

    Args:
        returns: Historical return series.
        target_vol: Target annualized volatility.
        current_vol: Override current volatility estimate.
        window: Lookback window for vol estimation.
        max_scale: Maximum scale-up factor.
        min_scale: Minimum scale-down factor.

    Returns:
        Scaling factor ∈ [min_scale, max_scale].
    """
    r = np.asarray(returns, dtype=float).flatten()

    if current_vol is None:
        if len(r) < window:
            return 1.0
        current_vol = float(np.std(r[-window:], ddof=1)) * np.sqrt(252)

    if current_vol < 1e-12:
        return 1.0

    scale = target_vol / current_vol
    return float(np.clip(scale, min_scale, max_scale))


# ---------------------------------------------------------------------------
# Risk-on / Risk-off regime
# ---------------------------------------------------------------------------


def risk_on_off(
    returns: NDArray[np.floating],
    vol_short: int = 21,
    vol_long: int = 63,
    drawdown_window: int = 252,
) -> RiskSignal:
    """Classify current market as risk-on or risk-off.

    Uses:
    - Volatility ratio (short vs long)
    - Drawdown depth
    - Recent Sharpe
    - Skewness

    Returns a RiskSignal with direction = "long" (risk-on), "reduce" (cautious),
    or "flatten" (risk-off).
    """
    r = np.asarray(returns, dtype=float).flatten()

    if len(r) < vol_long:
        return RiskSignal(name="risk_on_off", value=0.5, direction="neutral", confidence=0.0)

    # Vol ratio
    vol_s = float(np.std(r[-vol_short:], ddof=1))
    vol_l = float(np.std(r[-vol_long:], ddof=1))
    vol_ratio = vol_s / vol_l if vol_l > 0 else 1.0

    # Drawdown from equity curve
    recent_r = r[-drawdown_window:] if len(r) >= drawdown_window else r
    equity = np.cumprod(1.0 + recent_r)
    running_peak = np.maximum.accumulate(equity)
    dd = float(np.min(equity / running_peak - 1.0))

    # Trailing Sharpe
    if vol_s > 0 and len(r) >= vol_short:
        sharpe = float(np.mean(r[-vol_short:])) / vol_s * np.sqrt(252)
    else:
        sharpe = 0.0

    # Skewness
    if len(r) >= 21 and vol_s > 0:
        skew = float(np.mean(((r[-vol_short:] - np.mean(r[-vol_short:])) / vol_s) ** 3))
    else:
        skew = 0.0

    # Score: 0 = max risk-off, 1 = max risk-on
    score = 0.5

    # Vol ratio adjust: high vol → risk-off
    if vol_ratio > 2.0:
        score -= 0.2
    elif vol_ratio > 1.5:
        score -= 0.1
    elif vol_ratio < 0.7:
        score += 0.1

    # Drawdown adjust: deep dd → risk-off
    if dd < -0.15:
        score -= 0.25
    elif dd < -0.10:
        score -= 0.15
    elif dd < -0.05:
        score -= 0.05

    # Sharpe adjust
    if sharpe > 1.0:
        score += 0.1
    elif sharpe < -0.5:
        score -= 0.15
    elif sharpe < 0.0:
        score -= 0.05

    # Skewness adjust: negative skew → risk-off
    if skew < -1.0:
        score -= 0.1

    score = max(0.0, min(1.0, score))

    if score >= 0.6:
        direction = "long"
    elif score >= 0.4:
        direction = "reduce"
    else:
        direction = "flatten"

    return RiskSignal(
        name="risk_on_off",
        value=score,
        direction=direction,
        confidence=abs(score - 0.5) * 2.0,
        metadata={
            "vol_ratio": vol_ratio,
            "drawdown": dd,
            "sharpe": sharpe,
            "skewness": skew,
        },
    )


# ---------------------------------------------------------------------------
# Drawdown protection signals
# ---------------------------------------------------------------------------


def drawdown_protection(
    equity_curve: NDArray[np.floating],
    max_dd_threshold: float = 0.10,
    recovery_threshold: float = 0.05,
) -> RiskSignal:
    """Generate a drawdown protection signal.

    - If drawdown > threshold → reduce/flatten
    - If recovering (up from trough by recovery_threshold) → gradual re-entry

    Args:
        equity_curve: Cumulative equity values.
        max_dd_threshold: Drawdown % that triggers protection (as positive decimal).
        recovery_threshold: Recovery % from trough to signal re-entry.

    Returns:
        RiskSignal with appropriate direction.
    """
    eq = np.asarray(equity_curve, dtype=float).flatten()

    # Compute running peak and drawdown
    running_peak = np.maximum.accumulate(eq)
    dd = (eq - running_peak) / running_peak

    current_dd = float(dd[-1])
    trough_idx = int(np.argmin(dd))
    trough_dd = float(dd[trough_idx])

    # Check if recovering
    if current_dd < -max_dd_threshold:
        direction = "flatten"
        confidence = min(1.0, abs(current_dd) / max_dd_threshold)
    elif trough_dd < -max_dd_threshold and current_dd < 0:
        # Was in protection, recovering but not yet positive
        recovery = current_dd - trough_dd
        if recovery > recovery_threshold:
            direction = "reduce"  # Still cautious
        else:
            direction = "flatten"  # Still in protection
        confidence = min(1.0, abs(trough_dd) / max_dd_threshold)
    elif current_dd < -max_dd_threshold * 0.5:
        direction = "reduce"
        confidence = abs(current_dd) / max_dd_threshold
    else:
        direction = "long"
        confidence = 0.0

    return RiskSignal(
        name="drawdown_protection",
        value=current_dd,
        direction=direction,
        confidence=confidence,
        metadata={
            "current_dd": current_dd,
            "trough_dd": trough_dd,
            "trough_idx": int(trough_idx),
        },
    )


# ---------------------------------------------------------------------------
# Tail risk hedging signal
# ---------------------------------------------------------------------------


def tail_hedge_signal(
    returns: NDArray[np.floating],
    var_95: float | None = None,
    cvar_95: float | None = None,
    skew_window: int = 63,
) -> RiskSignal:
    """Determine whether tail risk hedging is warranted.

    Uses:
    - VaR / CVaR ratio (how fat the tail is)
    - Recent skewness
    - Maximum 1-day loss relative to VaR

    Returns a RiskSignal where value = recommended hedge ratio (0 to 1).
    """
    r = np.asarray(returns, dtype=float).flatten()

    if len(r) < skew_window:
        return RiskSignal(name="tail_hedge", value=0.0, direction="neutral", confidence=0.0)

    # Compute VaR and CVaR if not provided
    if var_95 is None:
        var_95 = float(np.percentile(r, 5))
    if cvar_95 is None:
        cvar_95 = float(np.mean(r[r <= var_95])) if np.any(r <= var_95) else var_95

    var_abs = abs(var_95) if var_95 < 0 else 0.0
    cvar_abs = abs(cvar_95) if cvar_95 < 0 else 0.0

    # Tail ratio: how much worse is CVaR than VaR
    tail_ratio = cvar_abs / var_abs if var_abs > 1e-10 else 1.0

    # Recent skewness
    recent = r[-skew_window:]
    skew = float(np.mean(((recent - np.mean(recent)) / np.std(recent, ddof=1)) ** 3)) if np.std(recent, ddof=1) > 0 else 0.0

    # Hedge score
    hedge = 0.0

    if tail_ratio > 1.5:
        hedge += 0.3
    elif tail_ratio > 1.3:
        hedge += 0.15

    if skew < -1.0:
        hedge += 0.25
    elif skew < -0.5:
        hedge += 0.1

    # Recent extreme moves
    max_1d_loss = float(np.min(recent))
    if var_abs > 0 and abs(max_1d_loss) > var_abs * 2.0:
        hedge += 0.2

    hedge = min(1.0, hedge)

    if hedge > 0.5:
        direction = "reduce"
    elif hedge > 0.25:
        direction = "reduce"
    else:
        direction = "neutral"

    return RiskSignal(
        name="tail_hedge",
        value=hedge,
        direction=direction,
        confidence=hedge,
        metadata={
            "var_95": var_95,
            "cvar_95": cvar_95,
            "tail_ratio": tail_ratio,
            "skewness": skew,
            "max_1d_loss": float(max_1d_loss),
        },
    )


# ---------------------------------------------------------------------------
# Correlation-based hedging
# ---------------------------------------------------------------------------


def correlation_hedge_signal(
    returns: NDArray[np.floating],
    benchmark_returns: NDArray[np.floating],
    window: int = 63,
    corr_threshold: float = 0.7,
) -> RiskSignal:
    """Signal to hedge when correlation to a risk factor spikes.

    High correlation to a benchmark during selloffs indicates
    the portfolio is not diversifying when needed most.

    Args:
        returns: Portfolio return series.
        benchmark_returns: Benchmark/factor return series.
        window: Rolling window.
        corr_threshold: Correlation level that triggers hedging.

    Returns:
        RiskSignal where value = recommended hedge ratio.
    """
    r = np.asarray(returns, dtype=float).flatten()
    b = np.asarray(benchmark_returns, dtype=float).flatten()
    n = min(len(r), len(b))

    if n < window:
        return RiskSignal(name="correlation_hedge", value=0.0, direction="neutral", confidence=0.0)

    rw = r[-window:]
    bw = b[-window:]

    # Full period correlation
    full_corr = float(np.corrcoef(rw, bw)[0, 1]) if np.std(rw) > 0 and np.std(bw) > 0 else 0.0

    # Down-market correlation (when benchmark is negative)
    down_mask = bw < 0
    down_corr = 0.0
    if down_mask.sum() >= 10:
        r_down = rw[down_mask]
        b_down = bw[down_mask]
        down_corr = float(np.corrcoef(r_down, b_down)[0, 1]) if np.std(r_down) > 0 and np.std(b_down) > 0 else 0.0

    # Asymmetric correlation (down - up) — measures crash correlation
    up_mask = bw >= 0
    up_corr = 0.0
    if up_mask.sum() >= 10:
        r_up = rw[up_mask]
        b_up = bw[up_mask]
        up_corr = float(np.corrcoef(r_up, b_up)[0, 1]) if np.std(r_up) > 0 and np.std(b_up) > 0 else 0.0

    asym_corr = down_corr - up_corr  # Positive = correlated in selloffs

    # Compute hedge ratio
    hedge = 0.0
    if down_corr > corr_threshold:
        hedge = (down_corr - corr_threshold) / (1.0 - corr_threshold)
    if asym_corr > 0.2:
        hedge = max(hedge, 0.5)

    hedge = min(1.0, hedge)

    direction = "reduce" if hedge > 0.3 else "neutral"

    return RiskSignal(
        name="correlation_hedge",
        value=hedge,
        direction=direction,
        confidence=hedge,
        metadata={
            "full_corr": full_corr,
            "down_corr": down_corr,
            "up_corr": up_corr,
            "asym_corr": asym_corr,
        },
    )


# ---------------------------------------------------------------------------
# Aggregated risk regime
# ---------------------------------------------------------------------------


def aggregate_risk_regime(
    returns: NDArray[np.floating],
    equity_curve: NDArray[np.floating] | None = None,
    benchmark_returns: NDArray[np.floating] | None = None,
    target_vol: float = 0.15,
) -> RiskRegime:
    """Aggregate all risk signals into a single regime assessment.

    Args:
        returns: Portfolio return series.
        equity_curve: Optional equity curve for drawdown signals.
        benchmark_returns: Optional benchmark for correlation hedging.
        target_vol: Target annualized volatility for vol scaling.

    Returns:
        RiskRegime with combined assessment and all component signals.
    """
    r = np.asarray(returns, dtype=float).flatten()
    signals: list[RiskSignal] = []

    # 1. Volatility targeting
    vol_scale = volatility_target_scale(r, target_vol=target_vol)
    signals.append(RiskSignal(
        name="vol_target",
        value=vol_scale,
        direction="long" if vol_scale >= 0.9 else "reduce",
        confidence=abs(1.0 - vol_scale),
        metadata={"target_vol": target_vol},
    ))

    # 2. Risk-on/off
    roo = risk_on_off(r)
    signals.append(roo)

    # 3. Drawdown protection
    if equity_curve is not None:
        dd_signal = drawdown_protection(equity_curve)
        signals.append(dd_signal)
        dd_scale = 1.0 if dd_signal.direction == "long" else (0.5 if dd_signal.direction == "reduce" else 0.05)
    else:
        dd_scale = 1.0

    # 4. Tail hedge
    tail = tail_hedge_signal(r)
    signals.append(tail)

    # 5. Correlation hedge
    if benchmark_returns is not None:
        corr_hedge = correlation_hedge_signal(r, benchmark_returns)
        signals.append(corr_hedge)
        corr_hedge_ratio = corr_hedge.value
    else:
        corr_hedge_ratio = 0.0

    # Combine into regime
    score = 0.5
    score += (vol_scale - 1.0) * 0.2
    score += (roo.value - 0.5) * 0.3
    score += (dd_scale - 1.0) * 0.15
    score -= tail.value * 0.2
    score -= corr_hedge_ratio * 0.15

    score = max(0.0, min(1.0, score))

    if score >= 0.7:
        regime = "calm"
        risk_on = True
    elif score >= 0.5:
        regime = "normal"
        risk_on = True
    elif score >= 0.35:
        regime = "cautious"
        risk_on = False
    elif score >= 0.2:
        regime = "stressed"
        risk_on = False
    else:
        regime = "crisis"
        risk_on = False

    return RiskRegime(
        regime=regime,
        risk_on=risk_on,
        score=score,
        vol_target_scale=vol_scale,
        drawdown_scale=dd_scale,
        correlation_hedge_ratio=corr_hedge_ratio,
        tail_hedge_ratio=tail.value,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Cross-sectional risk signals
# ---------------------------------------------------------------------------


def cross_sectional_risk(
    returns_matrix: NDArray[np.floating],
    window: int = 63,
) -> dict[str, Any]:
    """Cross-sectional risk signals across assets.

    Detects:
    - Rising cross-sectional dispersion (markets getting choppier)
    - Convergence/divergence patterns
    - Factor crowding signals

    Args:
        returns_matrix: (n_periods, n_assets) array of returns.
        window: Rolling window.

    Returns:
        Dict with cross-sectional risk indicators.
    """
    r = np.asarray(returns_matrix, dtype=float)

    if r.ndim == 1:
        r = r.reshape(-1, 1)

    n_periods, n_assets = r.shape

    if n_periods < window or n_assets < 2:
        return {"dispersion": 0.0, "dispersion_trend": "stable", "crowding": 0.0}

    # Cross-sectional dispersion (std of returns across assets)
    # Compare recent to historical
    recent_disp = np.std(r[-window // 2:], axis=1)
    hist_disp = np.std(r[-window:], axis=1)

    avg_recent = float(np.mean(recent_disp))
    avg_hist = float(np.mean(hist_disp))

    disp_ratio = avg_recent / avg_hist if avg_hist > 0 else 1.0

    if disp_ratio > 1.5:
        trend = "rising"
    elif disp_ratio < 0.7:
        trend = "falling"
    else:
        trend = "stable"

    # Pairwise correlation — average correlation as crowding indicator
    recent_corr = np.corrcoef(r[-window // 2:].T)
    np.fill_diagonal(recent_corr, 0.0)
    avg_corr = float(np.mean(recent_corr[np.triu_indices_from(recent_corr, k=1)]))

    long_corr = np.corrcoef(r[-window:].T)
    np.fill_diagonal(long_corr, 0.0)
    long_avg_corr = float(np.mean(long_corr[np.triu_indices_from(long_corr, k=1)]))

    crowding = avg_corr - long_avg_corr  # Positive = increasing crowding

    return {
        "dispersion": avg_recent,
        "dispersion_ratio": disp_ratio,
        "dispersion_trend": trend,
        "avg_correlation": avg_corr,
        "crowding": crowding,
        "crowding_signal": "crowded" if crowding > 0.1 else ("fragmented" if crowding < -0.1 else "normal"),
    }


def position_sizing_gate(
    returns: NDArray[np.floating],
    max_dd: float = 0.0,
    sharpe_min: float = -1.0,
    vol_max: float = 0.50,
) -> RiskSignal:
    """Simple risk gate: determine if strategies should be active.

    Returns a RiskSignal that can stop, scale, or allow trading.

    Args:
        returns: Historical returns.
        max_dd: Current max drawdown (0 = no drawdown).
        sharpe_min: Minimum acceptable trailing Sharpe.
        vol_max: Maximum acceptable annualized volatility.

    Returns:
        RiskSignal with direction = "long" (trade), "reduce" (scale), or "flatten" (stop).
    """
    r = np.asarray(returns, dtype=float).flatten()

    if len(r) < 21:
        return RiskSignal(name="position_gate", value=1.0, direction="long", confidence=0.0)

    vol = float(np.std(r[-63:], ddof=1)) * np.sqrt(252) if len(r) >= 63 else float(np.std(r, ddof=1)) * np.sqrt(252)
    mu = float(np.mean(r[-63:])) if len(r) >= 63 else float(np.mean(r))
    sharpe = mu / (vol / np.sqrt(252)) if vol > 0 else 0.0

    reasons: list[str] = []

    if max_dd < -0.20:
        return RiskSignal(
            name="position_gate", value=0.0, direction="flatten", confidence=1.0,
            metadata={"reasons": ["drawdown > 20%"]},
        )

    if vol > vol_max:
        reasons.append(f"vol {vol:.1%} > {vol_max:.0%}")

    if sharpe < sharpe_min:
        reasons.append(f"sharpe {sharpe:.2f} < {sharpe_min}")

    if reasons:
        if len(reasons) >= 2 or vol > vol_max * 1.5:
            return RiskSignal(
                name="position_gate", value=0.2, direction="flatten", confidence=0.8,
                metadata={"reasons": reasons},
            )
        return RiskSignal(
            name="position_gate", value=0.5, direction="reduce", confidence=0.5,
            metadata={"reasons": reasons},
        )

    return RiskSignal(name="position_gate", value=1.0, direction="long", confidence=0.8)
