"""Full-universe cross-sectional momentum on 56 tickers.

Addresses the two fatal flaws of prior momentum strategies:
- macro_cross_sectional_momentum: only 6 macro ETFs (insufficient cross-sectional DOF)
- equity_cross_sectional_momentum: only 18 trades over 7 years (untestable)

This strategy runs cross-sectional momentum on the full 56-ticker universe
(50 blue-chip stocks + 6 macro ETFs), providing sufficient DOF for meaningful
z-scoring and enough trades for statistical validation.

Signal construction matches macro_xs_momentum but with tighter position caps
(3% per name vs 10%) due to 3x more assets.

Risk controls: volatility scaling, regime filter, drawdown control,
gross exposure cap, beta-neutral constraint.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register
from quantcore.backtest.engine import SignalData


# ---------------------------------------------------------------------------
# Risk control configuration (identical to macro cross_sectional_momentum)
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
# Default 56-ticker universe (loaded from config/live.yaml at runtime)
# ---------------------------------------------------------------------------

_DEFAULT_UNIVERSE = [
    # 6 macro ETFs
    "SPY", "QQQ", "IWM", "GLD", "TLT", "DBC",
    # 50 blue-chip stocks
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK",
    "LLY", "ABBV", "PFE", "KO", "AVGO", "PEP", "COST", "TMO", "MCD",
    "WMT", "CSCO", "ACN", "ABT", "DHR", "LIN", "TXN", "NEER", "UPS",
    "RTX", "QCOM", "PM", "HON", "LOW", "AMGN", "INTU", "BA", "GE",
    "CAT", "SBUX", "ISRG", "AMD", "PLD",
]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@register("equity_full_universe_xs_momentum")
class FullUniverseCrossSectionalMomentum(Strategy):
    """Full-universe (56 tickers) cross-sectional momentum with integrated risk controls.

    Improves on macro_cross_sectional_momentum by:
    - 56 assets (vs 6): robust z-scores, sufficient cross-sectional DOF
    - 3% per-name cap (vs 10%): tighter diversification for large universe
    - Beta neutralization against SPY (vs equal-weight proxy)

    Risk controls (all active by default):
    - Volatility scaling: reduce exposure when realized vol > target
    - Regime filter: scale down in crisis/extreme vol regimes
    - Drawdown control: progressive position reduction as DD deepens
    - Gross exposure cap: max 100% (not 200%)
    - Beta-neutral: hedge residual market beta vs SPY
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        # Signal params -------------------------------------------------------
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.min_weight = float(config.params.get("min_weight", 0.01))

        raw_windows = config.params.get("mom_windows", "20,60,120")
        if isinstance(raw_windows, str):
            self.mom_windows = tuple(int(w) for w in raw_windows.split(","))
        else:
            self.mom_windows = tuple(int(w) for w in raw_windows)

        raw_weights = config.params.get("mom_weights", "0.2,0.5,0.3")
        if isinstance(raw_weights, str):
            self.mom_weights = tuple(float(w) for w in raw_weights.split(","))
        else:
            self.mom_weights = tuple(float(w) for w in raw_weights)

        self.use_vol_adjustment = bool(config.params.get("use_vol_adjustment", True))
        self.max_position_pct = float(config.params.get("max_position_pct", 0.03))

        # Risk control params -------------------------------------------------
        self.target_vol = float(config.params.get("target_vol", 0.12))
        self.vol_lookback = int(config.params.get("vol_lookback", 20))
        self.vol_percentile_window = int(config.params.get("vol_percentile_window", 252))
        self.max_gross_exposure = float(config.params.get("max_gross_exposure", 1.0))
        self.beta_neutral = bool(config.params.get("beta_neutral", True))
        self.regime_filter = bool(config.params.get("regime_filter", True))
        self.dd_control = bool(config.params.get("dd_control", True))

        # Universe ------------------------------------------------------------
        raw_universe = config.params.get("universe", _DEFAULT_UNIVERSE)
        if isinstance(raw_universe, str):
            self._universe = [t.strip() for t in raw_universe.split(",")]
        else:
            self._universe = list(raw_universe)

        # State ---------------------------------------------------------------
        self._price_history: dict[str, list[float]] = {}
        self._symbols: list[str] = []
        self._cached_signals: dict[str, SignalData] = {}
        self._last_rebalance_ts: float | None = None
        self._current_day_ts: float | None = None
        self._days_since_rebalance: int = 0
        self._max_window: int = max(self.mom_windows) if self.mom_windows else 120

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
        rets = np.diff(close_matrix[-w-1:, :], axis=0) / np.maximum(
            close_matrix[-w-1:-1, :], 1e-10
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
    # Momentum score computation
    # ------------------------------------------------------------------

    def _compute_momentum_scores(self, close_matrix: NDArray) -> NDArray:
        """Compute cross-sectional momentum scores."""
        n_periods, n_assets = close_matrix.shape
        scores = np.zeros(n_assets)

        for window, weight in zip(self.mom_windows, self.mom_weights):
            if n_periods < window + 1:
                continue
            ret = (close_matrix[-1, :] - close_matrix[-window-1, :]) / \
                  np.maximum(close_matrix[-window-1, :], 1e-10)
            mean_ret = np.mean(ret)
            std_ret = np.std(ret, ddof=1)
            if std_ret > 1e-10:
                z = (ret - mean_ret) / std_ret
            else:
                z = np.zeros_like(ret)
            scores += weight * z

        return scores

    def _compute_vol_ranks(self, close_matrix: NDArray) -> NDArray:
        """Rank assets by volatility (0=low vol, 1=high vol)."""
        n_periods, n_assets = close_matrix.shape
        vol_window = min(20, n_periods - 1)
        if vol_window < 5:
            return np.full(n_assets, 0.5)

        segment = close_matrix[-vol_window-1:, :]
        daily_rets = np.diff(segment, axis=0) / np.maximum(segment[:-1, :], 1e-10)
        vol = np.std(daily_rets, axis=0, ddof=1)

        order = np.argsort(vol)
        ranks = np.zeros_like(vol)
        ranks[order] = np.linspace(0, 1, len(vol))
        return ranks

    # ------------------------------------------------------------------
    # Beta neutralization (vs SPY for 56-ticker universe)
    # ------------------------------------------------------------------

    def _find_spy_index(self, symbols: list[str]) -> int | None:
        """Find SPY index in the symbol list."""
        for i, sym in enumerate(symbols):
            if sym == "SPY":
                return i
        return None

    def _neutralize_beta(self, weights: NDArray, close_matrix: NDArray,
                         symbols: list[str]) -> NDArray:
        """Remove residual market beta from portfolio weights.

        Uses SPY returns as market proxy (more robust than equal-weight
        portfolio for a 56-asset universe). Falls back to equal-weight
        proxy if SPY is not in the universe.
        """
        n_periods = close_matrix.shape[0]
        window = min(63, n_periods - 1)
        if window < 10:
            return weights

        rets = np.diff(close_matrix[-window-1:, :], axis=0) / np.maximum(
            close_matrix[-window-1:-1, :], 1e-10
        )
        n_days, n_assets = rets.shape

        # Use SPY as market proxy if available, else equal-weight portfolio
        spy_idx = self._find_spy_index(symbols)
        if spy_idx is not None:
            mkt_rets = rets[:, spy_idx]
        else:
            mkt_rets = np.nanmean(rets, axis=1)

        # Compute beta for each asset
        mkt_var = np.var(mkt_rets, ddof=1)
        if mkt_var < 1e-10:
            return weights

        betas = np.zeros(n_assets)
        for i in range(n_assets):
            if spy_idx is not None and i == spy_idx:
                betas[i] = 1.0
                continue
            valid = ~np.isnan(rets[:, i])
            if valid.sum() < 10:
                continue
            cov = np.cov(rets[valid, i], mkt_rets[valid], ddof=1)[0, 1]
            betas[i] = cov / mkt_var

        # Portfolio beta
        port_beta = float(np.dot(weights, betas))

        # Remove beta exposure while preserving dollar neutrality
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
        """Compute signals for all symbols with sufficient history."""
        valid_symbols = [s for s in self._symbols
                        if len(self._price_history.get(s, [])) >= self._max_window + 2]
        if not valid_symbols:
            return {}

        # Build aligned close matrix
        max_len = max(len(self._price_history[s]) for s in valid_symbols)
        n_assets = len(valid_symbols)
        close_matrix = np.zeros((max_len, n_assets))

        for i, sym in enumerate(valid_symbols):
            hist = self._price_history[sym]
            start = max(0, len(hist) - max_len)
            close_matrix[:len(hist)-start, i] = hist[start:]

        # Update risk state
        self._update_risk_state(close_matrix)

        # Compute scores
        scores = self._compute_momentum_scores(close_matrix)
        vol_ranks = self._compute_vol_ranks(close_matrix)

        # Adjust by volatility
        if self.use_vol_adjustment:
            adjusted = scores / (vol_ranks + 0.5)
        else:
            adjusted = scores

        # Softmax to get portfolio weights (dollar-neutral)
        weights = softmax(adjusted) - np.mean(softmax(adjusted))

        # Beta-neutral constraint (uses SPY as market proxy)
        if self.beta_neutral:
            weights = self._neutralize_beta(weights, close_matrix, valid_symbols)

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

        # Per-asset position limit (3% for 56-ticker universe)
        weights = np.clip(weights, -self.max_position_pct, self.max_position_pct)

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
                strategy_id="equity_full_universe_xs_momentum",
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

        if hasattr(bar.timestamp, 'timestamp'):
            ts = bar.timestamp.timestamp()
        else:
            ts = float(bar.timestamp)

        # Track price history
        if sym not in self._price_history:
            self._price_history[sym] = []
            self._symbols.append(sym)
        self._price_history[sym].append(price)

        # Update equity tracking
        if portfolio is not None and hasattr(portfolio, 'nav'):
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
            min_hist = min((len(h) for h in self._price_history.values()), default=0)
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

        # Compute momentum scores
        scores = np.zeros(n_assets)
        for window, weight in zip(self.mom_windows, self.mom_weights):
            if n_periods < window + 1:
                continue
            ret = (close[-1, :] - close[-window-1, :]) / \
                  np.maximum(close[-window-1, :], 1e-10)
            mean_ret = np.mean(ret)
            std_ret = np.std(ret, ddof=1)
            if std_ret > 1e-10:
                z = (ret - mean_ret) / std_ret
            else:
                z = np.zeros_like(ret)
            scores += weight * z

        # Vol ranks
        vol_window = min(20, n_periods - 1)
        if vol_window >= 5:
            segment = close[-vol_window-1:, :]
            daily_rets = np.diff(segment, axis=0) / np.maximum(segment[:-1, :], 1e-10)
            vol = np.std(daily_rets, axis=0, ddof=1)
        else:
            vol = np.ones(n_assets) * 1e-10

        order = np.argsort(vol)
        vol_ranks = np.zeros_like(vol)
        vol_ranks[order] = np.linspace(0, 1, len(vol))

        # Adjust
        if self.use_vol_adjustment:
            adjusted = scores / (vol_ranks + 0.5)
        else:
            adjusted = scores

        # Portfolio weights (dollar-neutral)
        weights = softmax(adjusted) - np.mean(softmax(adjusted))

        # Beta-neutral constraint
        if self.beta_neutral:
            weights = self._neutralize_beta(weights, close, list(symbols))

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

        # Per-asset position limit (3% for 56-ticker universe)
        weights = np.clip(weights, -self.max_position_pct, self.max_position_pct)

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
                },
            ))
        return signals
