"""Cross-asset carry factor harvesting strategy.

Carry is the return earned if nothing changes — the "yield" component of returns.
This is the most robust macro factor with near-zero correlation to momentum.

Signal construction (asset-class-specific proxies using OHLCV only):
  - Equity (SPY, QQQ, IWM): earnings yield proxy via long-term vs short-term mean
  - Bond (TLT): yield curve slope proxy via SMA divergence
  - Gold (GLD): opportunity cost proxy via mean-reversion
  - Commodity (DBC): roll yield proxy via backwardation/contango approximation

Multi-horizon composite: 0.2 * 21d + 0.5 * 63d + 0.3 * 252d carry signals.

Risk controls (Phase 4.5): volatility scaling, regime filter, drawdown control,
gross exposure cap, beta-neutral constraint.

Fixed event-driven interface: tracks timestamps to count calendar days properly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register
from quantcore.backtest.engine import SignalData


# ---------------------------------------------------------------------------
# Risk control configuration (identical to cross_sectional_momentum)
# ---------------------------------------------------------------------------

class VolRegime:
    LOW = 0       # vol < p25  -> 100% exposure
    NORMAL = 1    # p25-p75    -> 100% exposure
    ELEVATED = 2  # p75-p90    -> 75% exposure
    CRISIS = 3    # p90-p99    -> 50% exposure
    EXTREME = 4   # > p99      -> FLAT (0% exposure)


# Drawdown action -> exposure fraction
DD_ACTION_SCALE = {
    0: 1.00,   # NONE
    1: 0.75,   # REDUCE_25
    2: 0.50,   # REDUCE_50
    3: 0.25,   # REDUCE_75
    4: 0.00,   # FLATTEN
}

# Drawdown thresholds (fraction of peak equity)
DD_THRESHOLDS = [
    (-0.05, 0),   # > -5% DD: no action
    (-0.10, 1),   # -5% to -10%: reduce 25%
    (-0.15, 2),   # -10% to -15%: reduce 50%
    (-0.20, 3),   # -15% to -20%: reduce 75%
]

DD_FLATTEN = -0.20  # Below -20% -> FLAT


def _classify_vol_regime(
    current_vol: float, vol_percentiles: dict[str, float],
) -> int:
    """Classify vol level into regime using rolling percentiles."""
    if current_vol > vol_percentiles.get("p99", 1.0):
        return VolRegime.EXTREME
    if current_vol > vol_percentiles.get("p90", 0.6):
        return VolRegime.CRISIS
    if current_vol > vol_percentiles.get("p75", 0.4):
        return VolRegime.ELEVATED
    return VolRegime.NORMAL


def _compute_dd_action(drawdown_pct: float) -> int:
    """Return drawdown action index from drawdown percentage (negative)."""
    if drawdown_pct <= DD_FLATTEN:
        return 4  # FLATTEN
    for threshold, action in DD_THRESHOLDS:
        if drawdown_pct > threshold:
            return action
    return 0


# ---------------------------------------------------------------------------
# Asset-class carry signal computation
# ---------------------------------------------------------------------------

# Macro universe indices (assumed order: SPY, QQQ, IWM, GLD, TLT, DBC)
EQUITY_INDICES = {0, 1, 2}   # SPY, QQQ, IWM
GOLD_INDEX = 3                # GLD
BOND_INDEX = 4                # TLT
COMMODITY_INDEX = 5           # DBC


def _compute_equity_carry(close_matrix: NDArray, window: int) -> NDArray:
    """Equity carry proxy: earnings yield via long-term vs short-term mean.

    When long-term returns exceed short-term, valuations are depressed
    (higher earnings yield = higher carry). This captures mean-reversion
    of equity valuations as a carry proxy.

    Formula: (252d_return - window_return) / window_return
    Positive = long-term outpacing short-term = depressed valuation = high carry.
    """
    n_periods, n_assets = close_matrix.shape
    if n_periods < max(window, 21) + 1:
        return np.zeros(n_assets)

    # Short-term return (21 days)
    short_ret = (close_matrix[-1, :] - close_matrix[-22, :]) / np.maximum(
        close_matrix[-22, :], 1e-10
    )

    # Long-term return (window days)
    long_ret = (close_matrix[-1, :] - close_matrix[-window - 1, :]) / np.maximum(
        close_matrix[-window - 1, :], 1e-10
    )

    # Carry = long-term minus short-term (mean-reversion proxy)
    carry = long_ret - short_ret
    return carry


def _compute_bond_carry(close_matrix: NDArray, window: int) -> NDArray:
    """Bond carry proxy: yield curve slope via SMA divergence.

    When long-term SMA > short-term SMA, yields are falling (positive carry
    for duration holders). This approximates the term premium.

    Formula: (SMA_window - SMA_21) / SMA_21
    Positive = long-term average above short-term = falling yields = positive carry.
    """
    n_periods = close_matrix.shape[0]
    if n_periods < max(window, 21) + 1:
        return 0.0

    sma_long = np.mean(close_matrix[-window:, :])
    sma_short = np.mean(close_matrix[-21:, :])

    carry = (sma_long - sma_short) / np.maximum(sma_short, 1e-10)
    return carry


def _compute_gold_carry(close_matrix: NDArray, window: int) -> float:
    """Gold carry proxy: opportunity cost via mean-reversion.

    Gold has negative carry (storage cost, no yield). The "carry" is the
    opportunity cost vs risk-free rate. We approximate this as the negative
    of recent price appreciation — when gold has risen a lot, it's expensive
    relative to its mean (negative carry for new longs).

    Formula: -(close[-1] / close[-window] - 1)
    Negative of return = mean-reversion pressure.
    """
    n_periods = close_matrix.shape[0]
    if n_periods < window + 1:
        return 0.0

    ret = close_matrix[-1, 0] / close_matrix[-window - 1, 0] - 1.0
    return -ret


def _compute_commodity_carry(close_matrix: NDArray, window: int) -> float:
    """Commodity carry proxy: roll yield via backwardation approximation.

    When commodities are in backwardation (near > far), carry is positive.
    We approximate this with the negative of recent price change — rising
    prices often indicate backwardation (supply tightness).

    Formula: -(close[-1] / close[-window] - 1)
    This captures the mean-reversion component of roll yield.
    """
    n_periods = close_matrix.shape[0]
    if n_periods < window + 1:
        return 0.0

    ret = close_matrix[-1, 0] / close_matrix[-window - 1, 0] - 1.0
    return -ret


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@register("macro_carry")
class CarryStrategy(Strategy):
    """Cross-asset carry factor harvesting with integrated risk controls.

    Carry is the return earned if nothing changes — the "yield" component.
    This strategy harvests carry across equities, bonds, gold, and commodities
    using OHLCV-only proxies.

    Risk controls (all active by default):
    - Volatility scaling: reduce exposure when realized vol > target
    - Regime filter: scale down in crisis/extreme vol regimes
    - Drawdown control: progressive position reduction as DD deepens
    - Gross exposure cap: max 100% (not 200%)
    - Beta-neutral: hedge residual market beta

    Config params:
        carry_windows: lookback windows for carry computation (default "21,63,252")
        carry_weights: weights for multi-horizon composite (default "0.2,0.5,0.3")
        target_vol: annualized vol target (default 0.12)
        vol_lookback: window for realized vol estimation (default 20)
        max_gross_exposure: max sum of absolute weights (default 1.0)
        beta_neutral: enable beta neutralization (default True)
        regime_filter: enable vol regime filter (default True)
        dd_control: enable drawdown control (default True)
        rebalance_freq: rebalance interval in calendar days (default 21)
        min_weight: minimum absolute weight to emit signal (default 0.01)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        # Signal params -------------------------------------------------------
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.min_weight = float(config.params.get("min_weight", 0.01))

        raw_windows = config.params.get("carry_windows", "21,63,252")
        if isinstance(raw_windows, str):
            self.carry_windows = tuple(int(w) for w in raw_windows.split(","))
        else:
            self.carry_windows = tuple(int(w) for w in raw_windows)

        raw_weights = config.params.get("carry_weights", "0.2,0.5,0.3")
        if isinstance(raw_weights, str):
            self.carry_weights = tuple(float(w) for w in raw_weights.split(","))
        else:
            self.carry_weights = tuple(float(w) for w in raw_weights)

        # Risk control params -------------------------------------------------
        self.target_vol = float(config.params.get("target_vol", 0.12))
        self.vol_lookback = int(config.params.get("vol_lookback", 20))
        self.vol_percentile_window = int(config.params.get("vol_percentile_window", 252))
        self.max_gross_exposure = float(config.params.get("max_gross_exposure", 1.0))
        self.beta_neutral = bool(config.params.get("beta_neutral", True))
        self.regime_filter = bool(config.params.get("regime_filter", True))
        self.dd_control = bool(config.params.get("dd_control", True))

        # State ---------------------------------------------------------------
        self._price_history: dict[str, list[float]] = {}
        self._symbols: list[str] = []
        self._cached_signals: dict[str, SignalData] = {}
        self._last_rebalance_ts: float | None = None
        self._current_day_ts: float | None = None
        self._days_since_rebalance: int = 0
        self._max_window: int = max(self.carry_windows) if self.carry_windows else 252

        # Risk state ----------------------------------------------------------
        self._equity_peak: float = 1.0
        self._current_equity: float = 1.0
        self._equity_history: list[float] = []
        self._vol_history: list[float] = []

    # ------------------------------------------------------------------
    # Volatility & regime state update
    # ------------------------------------------------------------------

    def _update_risk_state(self, close_matrix: NDArray) -> None:
        """Update rolling vol history and equity tracking."""
        if close_matrix.shape[0] < 5:
            return

        # Portfolio-level realized vol (from equal-weight returns)
        n = close_matrix.shape[0]
        w = min(self.vol_lookback, n - 1)
        rets = np.diff(close_matrix[-w - 1:, :], axis=0) / np.maximum(
            close_matrix[-w - 1:-1, :], 1e-10
        )
        eq_wt_rets = np.nanmean(rets, axis=1)
        port_vol = float(np.std(eq_wt_rets, ddof=1) * np.sqrt(252))
        self._vol_history.append(port_vol)

        # Keep only percentile_window entries
        if len(self._vol_history) > self.vol_percentile_window:
            self._vol_history = self._vol_history[-self.vol_percentile_window:]

    def _get_vol_percentiles(self) -> dict[str, float]:
        """Compute rolling vol percentiles for regime classification."""
        if len(self._vol_history) < 63:
            return {"p75": 0.40, "p90": 0.60, "p99": 1.0}
        vols = np.array(self._vol_history)
        return {
            "p75": float(np.percentile(vols, 75)),
            "p90": float(np.percentile(vols, 90)),
            "p99": float(np.percentile(vols, 99)),
        }

    # ------------------------------------------------------------------
    # Carry signal computation
    # ------------------------------------------------------------------

    def _compute_carry_scores(self, close_matrix: NDArray) -> NDArray:
        """Compute multi-horizon carry scores for each asset.

        Returns cross-sectional z-scored composite carry signal.
        Each asset class uses its own carry proxy appropriate to its economics.
        """
        n_periods, n_assets = close_matrix.shape
        scores = np.zeros(n_assets)

        for window, weight in zip(self.carry_windows, self.carry_weights):
            if n_periods < window + 1:
                continue

            # Compute per-asset carry at this horizon
            carry = np.zeros(n_assets)

            for i in range(n_assets):
                # Extract single-asset column for scalar carry functions
                asset_col = close_matrix[:, i:i + 1]

                if i in EQUITY_INDICES:
                    # Equity carry: long-term vs short-term mean (earnings yield proxy)
                    equity_carry = _compute_equity_carry(close_matrix, window)
                    carry[i] = equity_carry[i]

                elif i == BOND_INDEX:
                    # Bond carry: SMA divergence (term premium proxy)
                    bond_carry_val = _compute_bond_carry(asset_col, window)
                    carry[i] = bond_carry_val

                elif i == GOLD_INDEX:
                    # Gold carry: opportunity cost (mean-reversion proxy)
                    gold_carry_val = _compute_gold_carry(asset_col, window)
                    carry[i] = gold_carry_val

                elif i == COMMODITY_INDEX:
                    # Commodity carry: roll yield (backwardation proxy)
                    comm_carry_val = _compute_commodity_carry(asset_col, window)
                    carry[i] = comm_carry_val

            # Cross-sectional z-score at this horizon
            mean_carry = np.mean(carry)
            std_carry = np.std(carry, ddof=1)
            if std_carry > 1e-10:
                z = (carry - mean_carry) / std_carry
            else:
                z = np.zeros_like(carry)

            scores += weight * z

        return scores

    def _compute_vol_ranks(self, close_matrix: NDArray) -> NDArray:
        """Rank assets by volatility (0=low vol, 1=high vol)."""
        n_periods, n_assets = close_matrix.shape
        vol_window = min(20, n_periods - 1)
        if vol_window < 5:
            return np.full(n_assets, 0.5)

        segment = close_matrix[-vol_window - 1:, :]
        daily_rets = np.diff(segment, axis=0) / np.maximum(segment[:-1, :], 1e-10)
        vol = np.std(daily_rets, axis=0, ddof=1)

        order = np.argsort(vol)
        ranks = np.zeros_like(vol)
        ranks[order] = np.linspace(0, 1, len(vol))
        return ranks

    # ------------------------------------------------------------------
    # Beta neutralization
    # ------------------------------------------------------------------

    def _neutralize_beta(self, weights: NDArray, close_matrix: NDArray) -> NDArray:
        """Remove residual market beta from portfolio weights.

        Regresses asset returns against the equal-weight portfolio return
        and subtracts the beta component from weights.
        """
        n_periods = close_matrix.shape[0]
        window = min(63, n_periods - 1)
        if window < 10:
            return weights

        rets = np.diff(close_matrix[-window - 1:, :], axis=0) / np.maximum(
            close_matrix[-window - 1:-1, :], 1e-10
        )
        n_days, n_assets = rets.shape

        # Equal-weight portfolio as market proxy (avoids single-asset noise)
        mkt_rets = np.nanmean(rets, axis=1)

        # Compute beta for each asset
        mkt_var = np.var(mkt_rets, ddof=1)
        if mkt_var < 1e-10:
            return weights

        betas = np.zeros(n_assets)
        for i in range(n_assets):
            valid = ~np.isnan(rets[:, i])
            if valid.sum() < 10:
                continue
            cov = np.cov(rets[valid, i], mkt_rets[valid], ddof=1)[0, 1]
            betas[i] = cov / mkt_var

        # Portfolio beta
        port_beta = float(np.dot(weights, betas))

        # Remove beta exposure
        if abs(port_beta) > 1e-10:
            beta_sum_sq = float(np.dot(betas, betas))
            if beta_sum_sq > 1e-10:
                weights = weights - betas * (port_beta / beta_sum_sq)

        return weights

    # ------------------------------------------------------------------
    # Risk scaling factor
    # ------------------------------------------------------------------

    def _compute_risk_scale(self) -> float:
        """Compute combined risk scaling factor (0.0 to 1.0).

        Returns the product of:
        - Vol targeting scale: min(1.0, target_vol / current_vol)
        - Regime scale: based on vol percentile classification
        - Drawdown scale: based on current drawdown level
        """
        scale = 1.0

        # 1. Volatility targeting
        if len(self._vol_history) >= 5:
            current_vol = self._vol_history[-1]
            if current_vol > 1e-10:
                vol_scale = np.clip(self.target_vol / current_vol, 0.2, 1.0)
                scale *= vol_scale

        # 2. Regime filter
        if self.regime_filter and len(self._vol_history) >= 63:
            percentiles = self._get_vol_percentiles()
            current_vol = self._vol_history[-1]
            regime = _classify_vol_regime(current_vol, percentiles)

            regime_scales = {
                VolRegime.LOW: 1.0,
                VolRegime.NORMAL: 1.0,
                VolRegime.ELEVATED: 0.75,
                VolRegime.CRISIS: 0.50,
                VolRegime.EXTREME: 0.0,
            }
            scale *= regime_scales.get(regime, 1.0)

        # 3. Drawdown control
        if self.dd_control and len(self._equity_history) >= 5:
            peak = max(self._equity_history)
            current = self._equity_history[-1]
            if peak > 1e-10:
                dd_pct = (current - peak) / peak
                action = _compute_dd_action(dd_pct)
                scale *= DD_ACTION_SCALE.get(action, 1.0)

        return float(np.clip(scale, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_all_signals(self) -> dict[str, SignalData]:
        """Compute carry signals for all symbols with sufficient history."""
        valid_symbols = [
            s for s in self._symbols
            if len(self._price_history.get(s, [])) >= self._max_window + 2
        ]
        if not valid_symbols:
            return {}

        # Build aligned close matrix
        max_len = max(len(self._price_history[s]) for s in valid_symbols)
        n_assets = len(valid_symbols)
        close_matrix = np.zeros((max_len, n_assets))

        for i, sym in enumerate(valid_symbols):
            hist = self._price_history[sym]
            start = max(0, len(hist) - max_len)
            close_matrix[:len(hist) - start, i] = hist[start:]

        # Update risk state
        self._update_risk_state(close_matrix)

        # Compute carry scores
        scores = self._compute_carry_scores(close_matrix)
        vol_ranks = self._compute_vol_ranks(close_matrix)

        # Vol-adjust carry (carry-to-vol ratio)
        adjusted = scores / (vol_ranks + 0.5)

        # Softmax to get portfolio weights (dollar-neutral)
        weights = softmax(adjusted) - np.mean(softmax(adjusted))

        # Beta-neutral constraint
        if self.beta_neutral:
            weights = self._neutralize_beta(weights, close_matrix)

        # Normalize weights to [-1, 1]
        max_abs = np.max(np.abs(weights))
        if max_abs > 1e-10:
            weights = weights / max_abs

        # Gross exposure cap
        gross = np.sum(np.abs(weights))
        if gross > self.max_gross_exposure:
            weights = weights * (self.max_gross_exposure / gross)

        # Apply risk scaling
        risk_scale = self._compute_risk_scale()
        weights = weights * risk_scale

        # Per-asset position limit (10% max)
        weights = np.clip(weights, -0.10, 0.10)

        # Re-center after all adjustments
        weights = weights - np.mean(weights)

        # Convert to signals
        signals = {}
        for i, sym in enumerate(valid_symbols):
            w = weights[i]
            if abs(w) < self.min_weight:
                continue
            signals[sym] = SignalData(
                symbol=sym,
                direction="long" if w > 0 else "short",
                strength=abs(w),
                strategy_id="macro_carry",
                notional_size=abs(w) * float(self.config.max_position_notional or 10000),
            )

        return signals

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def on_bar(self, bar, portfolio) -> list[SignalData]:
        """Event-driven entry point."""
        sym = bar.symbol
        price = float(bar.close)

        if hasattr(bar.timestamp, "timestamp"):
            ts = bar.timestamp.timestamp()
        else:
            ts = float(bar.timestamp)

        # Track price history
        if sym not in self._price_history:
            self._price_history[sym] = []
            self._symbols.append(sym)
        self._price_history[sym].append(price)

        # Update equity tracking
        if portfolio is not None and hasattr(portfolio, "nav"):
            self._equity_history.append(float(portfolio.nav))
            if len(self._equity_history) > 252:
                self._equity_history = self._equity_history[-252:]

        # Track calendar days using timestamps
        if self._current_day_ts is None:
            self._current_day_ts = ts
        elif ts > self._current_day_ts + 86400:
            self._current_day_ts = ts
            self._days_since_rebalance += 1

        # Check if we should rebalance
        if self._last_rebalance_ts is None:
            min_hist = min(
                (len(h) for h in self._price_history.values()), default=0
            )
            if min_hist >= self._max_window + 2:
                self._cached_signals = self._compute_all_signals()
                self._last_rebalance_ts = ts
                self._days_since_rebalance = 0
        elif self._days_since_rebalance >= self.rebalance_freq:
            self._cached_signals = self._compute_all_signals()
            self._last_rebalance_ts = ts
            self._days_since_rebalance = 0

        if sym in self._cached_signals:
            return [self._cached_signals[sym]]
        return []

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Batch data entry point (legacy)."""
        close = data.get("close")
        if close is None:
            return []
        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        if n_periods < self._max_window + 2:
            return []

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        # Update risk state
        self._update_risk_state(close)

        # Update equity tracking from data
        timestamp = data.get("timestamp")
        if timestamp is not None:
            self._equity_history.append(1.0)
            if len(self._equity_history) > 252:
                self._equity_history = self._equity_history[-252:]

        # Compute carry scores (same logic as _compute_carry_scores)
        scores = np.zeros(n_assets)

        for window, weight in zip(self.carry_windows, self.carry_weights):
            if n_periods < window + 1:
                continue

            carry = np.zeros(n_assets)
            for i in range(n_assets):
                asset_col = close[:, i:i + 1]

                if i in EQUITY_INDICES:
                    equity_carry = _compute_equity_carry(close, window)
                    carry[i] = equity_carry[i]
                elif i == BOND_INDEX:
                    bond_carry_val = _compute_bond_carry(asset_col, window)
                    carry[i] = bond_carry_val
                elif i == GOLD_INDEX:
                    gold_carry_val = _compute_gold_carry(asset_col, window)
                    carry[i] = gold_carry_val
                elif i == COMMODITY_INDEX:
                    comm_carry_val = _compute_commodity_carry(asset_col, window)
                    carry[i] = comm_carry_val

            # Cross-sectional z-score
            mean_carry = np.mean(carry)
            std_carry = np.std(carry, ddof=1)
            if std_carry > 1e-10:
                z = (carry - mean_carry) / std_carry
            else:
                z = np.zeros_like(carry)

            scores += weight * z

        # Vol ranks
        vol_window = min(20, n_periods - 1)
        if vol_window >= 5:
            segment = close[-vol_window - 1:, :]
            daily_rets = np.diff(segment, axis=0) / np.maximum(
                segment[:-1, :], 1e-10
            )
            vol = np.std(daily_rets, axis=0, ddof=1)
        else:
            vol = np.ones(n_assets) * 1e-10

        order = np.argsort(vol)
        vol_ranks = np.zeros_like(vol)
        vol_ranks[order] = np.linspace(0, 1, len(vol))

        # Vol-adjust carry
        adjusted = scores / (vol_ranks + 0.5)

        # Portfolio weights (dollar-neutral)
        weights = softmax(adjusted) - np.mean(softmax(adjusted))

        # Beta-neutral constraint
        if self.beta_neutral:
            weights = self._neutralize_beta(weights, close)

        max_abs = np.max(np.abs(weights))
        if max_abs > 1e-10:
            weights = weights / max_abs

        # Gross exposure cap
        gross = np.sum(np.abs(weights))
        if gross > self.max_gross_exposure:
            weights = weights * (self.max_gross_exposure / gross)

        # Apply risk scaling
        risk_scale = self._compute_risk_scale()
        weights = weights * risk_scale

        # Per-asset position limit (10% max)
        weights = np.clip(weights, -0.10, 0.10)

        # Re-center
        weights = weights - np.mean(weights)

        # Signals
        signals = []
        for i in range(n_assets):
            w = weights[i]
            if abs(w) < self.min_weight:
                continue
            signals.append(TradeSignal(
                symbol=symbols[i],
                direction=1 if w > 0 else -1,
                strength=abs(w),
                size=abs(w) * float(self.config.max_position_notional or 10000),
                metadata={
                    "score": float(scores[i]),
                    "weight": float(w),
                    "vol_rank": float(vol_ranks[i]),
                    "risk_scale": risk_scale,
                    "strategy": "macro_carry",
                },
            ))
        return signals
