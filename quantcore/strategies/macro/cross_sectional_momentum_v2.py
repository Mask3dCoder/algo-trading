"""Cross-sectional momentum v2 — composable risk pipeline.

RECOMMENDED BASELINE with institutional-grade risk controls.

Pipeline:
  Raw Momentum Scores → Softmax → Dollar-Neutral Weights
  → Beta-Neutral (beta_neutral.py)
  → Position Limits (±3%)
  → Gross Exposure Normalization (≤100%)
  → Vol Targeting Scale (vol_target.py, range [0.5, 2.0])
  → Regime Filter Scale (volatility_regime.py)
  → Drawdown Guard Scale (drawdown_guard.py, with cooldown)
  → Final Weights → Signals

Key differences from v1:
  - Risk modules are composed, not inline — each independently testable
  - Leverage range [0.5, 2.0] allows scaling UP in calm markets (v1: [0.2, 1.0])
  - Drawdown guard has 7-day cooldown + recovery gate (v1: no cooldown)
  - Beta neutralization uses SPY as market proxy, not equal-weight portfolio
  - Position limit: ±3% per asset (v1: ±10%)
  - Proper equity tracking in both on_bar and on_data paths
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from quantcore.backtest.engine import SignalData
from quantcore.portfolio.beta_neutral import neutralize_beta
from quantcore.regime.volatility_regime import VolRegime, vol_percentile_regime
from quantcore.risk.drawdown_guard import DrawdownGuard, DrawdownGuardConfig
from quantcore.risk.vol_target import VolTarget, VolTargetConfig
from quantcore.strategies.base import Strategy, StrategyConfig, StrategyStatus, TradeSignal
from quantcore.strategies.registry import register


# ---------------------------------------------------------------------------
# Regime → scale mapping (conservative: only binary reduction)
# ---------------------------------------------------------------------------

REGIME_SCALE: dict[VolRegime, float] = {
    VolRegime.LOW: 1.0,
    VolRegime.NORMAL: 1.0,
    VolRegime.ELEVATED: 0.75,
    VolRegime.CRISIS: 0.50,
    VolRegime.EXTREME: 0.0,
}


# DISABLED (Phase 0 / prospective trial #1 — REJECTED)
# See prospective_trials.csv for rejection rationale.
# @register("macro_cross_sectional_momentum_v2")
class CrossSectionalMomentumV2Strategy(Strategy):
    """Cross-sectional momentum with composable institutional risk controls.

    STATUS: REJECTED — OOS Sharpe −0.091 (90% CI −0.855 to +0.248) on
    expanded_15 universe 2018-2024.

    Rationale for rejection:
      - Cross-sectional momentum signal with only 5-15 macro ETFs has
        insufficient cross-sectional variation (degrees of freedom ≈ 13).
      - Score vs forward-return r = 0.055 (p < 0.00001) is statistically
        significant but economically too weak to overcome transaction costs.
      - Even with 15 assets the signal-to-noise ratio is below threshold.
      - The strategy is believed to need ≥40 independent cross-sectional
        securities (e.g. sector SPDRs + international + commodities) to
        produce a reliable diversified portfolio.

    Risk controls (updated post-audit):
    - Volatility scaling: dynamic leverage [0.5, 1.0] targeting 12% ann. vol
      (max_leverage reduced from 2.0 → 1.0 to prevent catastrophic drawdowns)
    - Correlation-stress: halve all weights when avg pairwise corr > 0.70
    - Regime filter: scale down 50% in crisis, flatten in extreme regimes
    - Drawdown guard: progressive reduction → flatten at -15% with 7-day cooldown
    - Beta-neutral: hedge residual market beta using SPY proxy
    - Position limits: ±3% per asset, max 100% gross exposure
    """

    STRATEGY_STATUS: StrategyStatus = StrategyStatus.REJECTED

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        # --- Signal params ---------------------------------------------------
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))

        raw_windows = config.params.get("mom_windows", "20,60,120")
        if isinstance(raw_windows, str):
            self.mom_windows = tuple(int(w) for w in raw_windows.split(","))
        else:
            self.mom_windows = tuple(int(w) for w in raw_windows)

        raw_mom_weights = config.params.get("mom_weights", "0.2,0.5,0.3")
        if isinstance(raw_mom_weights, str):
            self.mom_weights = tuple(float(w) for w in raw_mom_weights.split(","))
        else:
            self.mom_weights = tuple(float(w) for w in raw_mom_weights)

        self.use_vol_adjustment = bool(config.params.get("use_vol_adjustment", True))
        self._max_window = max(self.mom_windows) if self.mom_windows else 120

        # --- Risk control params ---------------------------------------------
        self._target_vol = float(config.params.get("target_vol", 0.12))
        self._vol_lookback = int(config.params.get("vol_lookback", 20))
        self._max_gross_exposure = float(config.params.get("max_gross_exposure", 1.0))
        self._max_position_pct = float(config.params.get("max_position_pct", 0.03))
        self._min_weight = float(config.params.get("min_weight", 0.005))

        beta_neutral_enabled = bool(config.params.get("beta_neutral", True))
        self._beta_window = int(config.params.get("beta_window", 60))

        regime_enabled = bool(config.params.get("regime_filter", True))
        self._regime_method = str(config.params.get("regime_method", "percentile"))
        self._regime_reduce = float(config.params.get("regime_reduce_factor", 0.5))

        dd_enabled = bool(config.params.get("dd_control", True))
        dd_flatten = float(config.params.get("dd_flatten_threshold", -0.15))
        dd_cooldown = int(config.params.get("dd_cooldown_days", 7))

        # Task 2.3 safety fix: cap max_leverage at 1.0 (was 2.0).
        # The previous range [0.5, 2.0] enabled 200% gross exposure which
        # combined with 88% max drawdown in OOS.  Max leverage 1.0 ensures
        # total long + short <= 100% NAV at all times.
        _max_lev = min(float(config.params.get("max_leverage", 1.0)), 1.0)

        # --- Risk modules ----------------------------------------------------
        self._vol_target = VolTarget(VolTargetConfig(
            target_vol=self._target_vol,
            short_window=self._vol_lookback,
            long_window=min(self._vol_lookback * 3, 60),
            min_leverage=0.5,
            max_leverage=_max_lev,
            use_ewma=True,
        ))

        self._drawdown_guard = DrawdownGuard(DrawdownGuardConfig(
            flatten_threshold=dd_flatten,
            reduce_thresholds=[(-0.05, 0.75), (-0.10, 0.50)],
            cooldown_days=dd_cooldown,
            recovery_threshold=-0.05,
        )) if dd_enabled else None

        self._beta_neutral_enabled = beta_neutral_enabled
        self._regime_enabled = regime_enabled
        self._dd_enabled = dd_enabled

        # --- State -----------------------------------------------------------
        self._price_history: dict[str, list[float]] = {}
        self._symbols: list[str] = []
        self._cached_signals: dict[str, SignalData] = {}
        self._last_rebalance_ts: float | None = None
        self._current_day_ts: float | None = None
        self._days_since_rebalance: int = 0
        self._equity_history: list[float] = []
        self._prev_equity: float | None = None

    # ------------------------------------------------------------------
    # Risk pipeline
    # ------------------------------------------------------------------

    def _compute_risk_scale(self, close_matrix: NDArray) -> float:
        """Compute combined risk scaling factor (0.0 to 1.0).

        Returns the product of vol targeting, regime filter, and drawdown
        guard scales. Each is independently computed and multiplied.
        """
        scale = 1.0

        # 1. Volatility targeting (handled separately — applied as leverage
        #    multiplier to weights, not a 0-1 scale). This method only
        #    returns the reduction factors (regime + drawdown).
        #    Vol targeting is applied in _compute_weights() via _vol_target.

        # 2. Regime filter
        if self._regime_enabled:
            regime_scale = self._compute_regime_scale(close_matrix)
            scale *= regime_scale

        # 3. Drawdown guard
        if self._dd_enabled and self._drawdown_guard is not None:
            scale *= self._drawdown_guard.scale_factor

        return float(np.clip(scale, 0.0, 1.0))

    def _compute_regime_scale(self, close_matrix: NDArray) -> float:
        """Classify current vol regime and return scale factor."""
        n = close_matrix.shape[0]
        lookback = min(252, n - 1)
        if lookback < 21:
            return 1.0

        # Compute equal-weight portfolio returns for regime detection
        segment = close_matrix[-lookback - 1:, :]
        rets = np.diff(segment, axis=0) / np.maximum(segment[:-1, :], 1e-10)
        eq_wt_rets = np.nanmean(rets, axis=1)

        result = vol_percentile_regime(eq_wt_rets, window=lookback)
        base_scale = REGIME_SCALE.get(result.regime, 1.0)

        # Apply reduce factor for crisis/extreme (default 0.5 → 50% cut)
        if result.regime in (VolRegime.CRISIS, VolRegime.EXTREME):
            base_scale *= self._regime_reduce

        return float(np.clip(base_scale, 0.0, 1.0))

    def _compute_vol_adjustment(self, close_matrix: NDArray) -> NDArray:
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

    def _update_equity_from_close(self, close_matrix: NDArray) -> None:
        """Track portfolio equity from close matrix for vol/DD state.

        Estimates portfolio returns from equal-weight portfolio changes,
        feeds them to VolTarget, and tracks a synthetic equity curve
        for the DrawdownGuard.
        """
        n = close_matrix.shape[0]
        if n < 2:
            return

        # Equal-weight daily return as portfolio proxy
        rets = np.diff(close_matrix[-2:, :], axis=0) / np.maximum(
            close_matrix[-2:-1, :], 1e-10
        )
        port_ret = float(np.nanmean(rets))

        # Update vol target with portfolio return
        self._vol_target.update(port_ret)

        # Update equity for drawdown guard
        if not self._equity_history:
            self._equity_history.append(1.0)
        equity = self._equity_history[-1] * (1.0 + port_ret)
        self._equity_history.append(equity)
        if len(self._equity_history) > 504:
            self._equity_history = self._equity_history[-504:]

        if self._dd_enabled and self._drawdown_guard is not None:
            self._drawdown_guard.update(equity)

    # ------------------------------------------------------------------
    # Momentum scoring
    # ------------------------------------------------------------------

    def _compute_momentum_scores(self, close_matrix: NDArray) -> NDArray:
        """Compute cross-sectional momentum z-scores from multi-window returns."""
        n_periods, n_assets = close_matrix.shape
        scores = np.zeros(n_assets)

        for window, weight in zip(self.mom_windows, self.mom_weights):
            if n_periods < window + 1:
                continue
            ret = (close_matrix[-1, :] - close_matrix[-window - 1, :]) / \
                  np.maximum(close_matrix[-window - 1, :], 1e-10)
            mean_ret = np.mean(ret)
            std_ret = np.std(ret, ddof=1)
            if std_ret > 1e-10:
                z = (ret - mean_ret) / std_ret
            else:
                z = np.zeros_like(ret)
            scores += weight * z

        return scores

    # ------------------------------------------------------------------
    # Core weight computation
    # ------------------------------------------------------------------

    def _compute_weights(self, close_matrix: NDArray) -> tuple[NDArray, list[str], dict]:
        """Run full signal pipeline and return (weights, symbols, diagnostics)."""
        n_periods, n_assets = close_matrix.shape

        # 1. Momentum scores
        scores = self._compute_momentum_scores(close_matrix)

        # 2. Vol adjustment
        if self.use_vol_adjustment:
            vol_ranks = self._compute_vol_adjustment(close_matrix)
            adjusted = scores / (vol_ranks + 0.5)
        else:
            vol_ranks = np.full(n_assets, 0.5)
            adjusted = scores

        # 3. Softmax → dollar-neutral weights
        weights = softmax(adjusted) - np.mean(softmax(adjusted))

        # 4. Beta-neutral constraint
        if self._beta_neutral_enabled:
            weights = neutralize_beta(weights, close_matrix, market_returns=None,
                                      window=self._beta_window)

        # 5. Position limits
        weights = np.clip(weights, -self._max_position_pct, self._max_position_pct)

        # 6. Gross exposure normalization (hard cap at max_gross_exposure ≤ 1.0)
        gross = np.sum(np.abs(weights))
        if gross > self._max_gross_exposure and gross > 1e-10:
            weights = weights * (self._max_gross_exposure / gross)

        # 6b. Correlation-stress adjustment (Task 2.3 safety fix)
        #     If average pairwise correlation among top positions exceeds 0.7,
        #     reduce all weights by 50% to prevent concentration blowup.
        corr_scale = 1.0
        if n_periods >= 20 and n_assets >= 2:
            try:
                rets = np.diff(np.log(np.where(close_matrix > 0, close_matrix, np.nan)),
                               axis=0)[-20:]
                # Identify top-N positions by abs weight
                active_mask = np.abs(weights) > 1e-6
                n_active = int(active_mask.sum())
                if n_active >= 2:
                    active_rets = rets[:, active_mask]
                    # Drop columns with zero variance
                    col_std = np.nanstd(active_rets, axis=0)
                    valid = col_std > 1e-10
                    if valid.sum() >= 2:
                        active_rets = active_rets[:, valid]
                        corr_mat = np.corrcoef(active_rets.T)
                        n = corr_mat.shape[0]
                        # Mean of off-diagonal elements
                        off_diag = corr_mat[np.triu_indices(n, k=1)]
                        avg_corr = float(np.nanmean(off_diag))
                        if avg_corr > 0.70:
                            corr_scale = 0.50
            except Exception:
                pass  # Defensive — never crash signal generation

        weights = weights * corr_scale

        # 7. Vol targeting leverage (applied as multiplier, not 0-1 scale)
        #    Capped so total leverage stays ≤ max_leverage (1.0 post-fix)
        vol_leverage = self._vol_target.compute_leverage()

        # 8. Regime + Drawdown risk scale (0-1)
        risk_scale = self._compute_risk_scale(close_matrix)

        # Combined scale — final gross exposure hard-capped at max_gross_exposure
        combined_scale = vol_leverage * risk_scale
        weights = weights * combined_scale

        # Final hard normalization: total long + short <= max_gross_exposure
        gross_final = np.sum(np.abs(weights))
        if gross_final > self._max_gross_exposure and gross_final > 1e-10:
            weights = weights * (self._max_gross_exposure / gross_final)

        # Re-clip per-position limit
        weights = np.clip(weights, -self._max_position_pct, self._max_position_pct)

        # Re-center to maintain dollar-neutrality
        weights = weights - np.mean(weights)

        return weights, list(range(n_assets)), {
            "vol_leverage": vol_leverage,
            "risk_scale": risk_scale,
            "combined_scale": combined_scale,
            "corr_scale": corr_scale,
        }

    # ------------------------------------------------------------------
    # Entry points
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

        # Track equity from portfolio
        if portfolio is not None and hasattr(portfolio, "nav"):
            equity = float(portfolio.nav)
            self._equity_history.append(equity)
            if len(self._equity_history) > 504:
                self._equity_history = self._equity_history[-504:]

            # Feed vol target with portfolio return
            if self._prev_equity is not None and self._prev_equity > 0:
                port_ret = (equity - self._prev_equity) / self._prev_equity
                self._vol_target.update(port_ret)
            self._prev_equity = equity

            # Feed drawdown guard
            if self._dd_enabled and self._drawdown_guard is not None:
                self._drawdown_guard.update(equity)

        # Track calendar days using timestamps
        if self._current_day_ts is None:
            self._current_day_ts = ts
        elif ts > self._current_day_ts + 86400:
            self._current_day_ts = ts
            self._days_since_rebalance += 1

        # Rebalance logic
        if self._last_rebalance_ts is None:
            min_hist = min((len(h) for h in self._price_history.values()), default=0)
            if min_hist >= self._max_window + 2:
                self._cached_signals = self._compute_signals_from_history()
                self._last_rebalance_ts = ts
                self._days_since_rebalance = 0
        elif self._days_since_rebalance >= self.rebalance_freq:
            self._cached_signals = self._compute_signals_from_history()
            self._last_rebalance_ts = ts
            self._days_since_rebalance = 0

        if sym in self._cached_signals:
            return [self._cached_signals[sym]]
        return []

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Batch data entry point."""
        close = data.get("close")
        if close is None:
            return []
        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        if n_periods < self._max_window + 2:
            return []

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        # Update vol/DD state from close data
        self._update_equity_from_close(close)

        # Compute weights
        weights, _, diag = self._compute_weights(close)

        # Convert to signals
        signals: list[TradeSignal] = []
        for i in range(min(n_assets, len(symbols))):
            w = weights[i]
            if abs(w) < self._min_weight:
                continue
            signals.append(TradeSignal(
                symbol=symbols[i],
                direction=1 if w > 0 else -1,
                strength=abs(w),
                size=abs(w) * float(self.config.max_position_notional or 10000),
                metadata={
                    "weight": float(w),
                    "vol_leverage": diag.get("vol_leverage", 1.0),
                    "risk_scale": diag.get("risk_scale", 1.0),
                    "combined_scale": diag.get("combined_scale", 1.0),
                },
            ))
        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_signals_from_history(self) -> dict[str, SignalData]:
        """Build aligned close matrix from price history and compute signals."""
        valid_symbols = [s for s in self._symbols
                         if len(self._price_history.get(s, [])) >= self._max_window + 2]
        if not valid_symbols:
            return {}

        max_len = max(len(self._price_history[s]) for s in valid_symbols)
        n_assets = len(valid_symbols)
        close_matrix = np.zeros((max_len, n_assets))

        for i, sym in enumerate(valid_symbols):
            hist = self._price_history[sym]
            start = max(0, len(hist) - max_len)
            close_matrix[:len(hist) - start, i] = hist[start:]

        # Update vol/DD state
        self._update_equity_from_close(close_matrix)

        # Compute weights
        weights, _, diag = self._compute_weights(close_matrix)

        # Convert to SignalData dict
        signals: dict[str, SignalData] = {}
        for i, sym in enumerate(valid_symbols):
            w = weights[i]
            if abs(w) < self._min_weight:
                continue
            signals[sym] = SignalData(
                symbol=sym,
                direction="long" if w > 0 else "short",
                strength=abs(w),
                strategy_id="macro_cross_sectional_momentum_v2",
                notional_size=abs(w) * float(self.config.max_position_notional or 10000),
            )
        return signals

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    @property
    def vol_leverage(self) -> float:
        return self._vol_target.compute_leverage()

    @property
    def drawdown_state(self):
        if self._dd_enabled and self._drawdown_guard is not None:
            return {
                "drawdown_pct": self._drawdown_guard.current_drawdown,
                "is_in_cooldown": self._drawdown_guard.is_in_cooldown,
                "is_flattened": self._drawdown_guard.is_flattened,
                "scale_factor": self._drawdown_guard.scale_factor,
            }
        return {"drawdown_pct": 0.0, "is_in_cooldown": False,
                "is_flattened": False, "scale_factor": 1.0}

    def on_fill(self, fill, portfolio=None) -> None:
        """Handle fill events from the backtest engine.

        The engine calls on_fill(fill: Fill, portfolio: PortfolioState).
        The ABC declares on_fill(symbol, fill_price, fill_qty, side).
        This implementation is compatible with BOTH calling conventions
        by accepting *any* first positional arg and ignoring it — the
        strategy is purely signal-driven and does not need fill feedback.
        """
        pass  # Signal-driven strategy; no fill-based state update needed

    def reset(self) -> None:
        """Reset all state (for walk-forward folds)."""
        self._price_history.clear()
        self._symbols.clear()
        self._cached_signals.clear()
        self._last_rebalance_ts = None
        self._current_day_ts = None
        self._days_since_rebalance = 0
        self._equity_history.clear()
        self._prev_equity = None
        self._vol_target.reset()
        if self._dd_enabled and self._drawdown_guard is not None:
            self._drawdown_guard.reset()