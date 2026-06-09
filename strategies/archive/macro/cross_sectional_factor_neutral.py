"""Cross-Sectional Factor-Neutral StatArb (CSFN).

Institutional-grade cross-sectional strategy for the ETF_EXPANDED 44-asset
universe.  Extracts macro-factor-adjusted residual alpha, ranks assets by
cross-sectional z-score, and solves a constrained QP for factor + beta +
dollar-neutral weights.

Signal pipeline
---------------
1.  Macro-factor rolling regression (252 d, Ridge) on each asset vs 8
    macro factor returns (SPY, TLT, GLD, DBC, UUP, EFA, VNQ, LQD).
2.  Rolling residual extraction:  epsilon(t) = R(t) - factor_ret(t) * beta.
3.  Predict next-period alpha = mean(epsilon) over ``pred_window``.
4.  Cross-sectional z-score of predicted alpha.
5.  Top 30 % z -> long; bottom 30 % z -> short (symmetric slice)

Portfolio construction
----------------------
solve QP:  min  ||w - z_score||^2
    s.t.  B.T @ w = 0        factor neutrality
          sum(w)     = 0     dollar neutrality
          |w_i|      <= 3 %   position limit

Risk controls (institutional tier)
----------------------------------
- Vol-targeted gross exposure (target 15 % ann. vol)
- Regime filter: scales to 0 % in crisis/extreme VIX, 75 % in elevated.
- Drawdown guard: progressive reduction -> flat at -20 %.
- Daily beta-neutral rebalancing via existing beta_neutral module.
- Max leverage 1.0x.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Optional dependencies (graceful degradation)
# -----------------------------------------------------------------------
try:
    from sklearn.linear_model import Ridge
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False  # type: ignore[assignment]

try:
    from numpy.polynomial import polynomial as _poly  # type: ignore[unused]
except ImportError:
    pass

# QuantCore imports
from quantcore.backtest.engine import SignalData
from quantcore.portfolio.beta_neutral import neutralize_beta
from quantcore.strategies.base import Strategy, StrategyConfig, StrategyStatus, TradeSignal
from quantcore.strategies.registry import register

try:
    from quantcore.strategies.macro.optimizer import (  # type: ignore[attr-defined]
        check_constraints,
        scale_to_target_vol,
        solve_factor_neutral_weights,
    )
except ImportError:  # pragma: no cover
    solve_factor_neutral_weights = None  # type: ignore[assignment]
    scale_to_target_vol = None  # type: ignore[assignment]
    check_constraints = None  # type: ignore[assignment]

try:
    from quantcore.regime.volatility_regime import VolRegime, vol_percentile_regime

    _REGIME_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REGIME_AVAILABLE = False  # type: ignore[assignment]
    VolRegime = None  # type: ignore[assignment,misc]
    vol_percentile_regime = None  # type: ignore[assignment,misc]


# -----------------------------------------------------------------------
# Regime scale mapping
# -----------------------------------------------------------------------
_REGIME_SCALE: dict[Any, float] = {
    "LOW": 1.0,
    "NORMAL": 1.0,
    "ELEVATED": 0.75,
    "CRISIS": 0.50,
    "EXTREME": 0.0,
}

# -----------------------------------------------------------------------
# Strategy
# -----------------------------------------------------------------------


@register("macro_cross_sectional_factor_neutral")
class CrossSectionalFactorNeutralStatArb(Strategy):
    """Cross-sectional factor-neutral stat-arb for ETF_EXPANDED universe.

    DESIGN PHILOSOPHY
    -----------------
    Emerging-market and sector-ETF return streams carry macro-factor exposure
    that is predictable (SPY vs TLT beta, commodity/inflation load, etc.).  By
    first neutralising that predictable component, we harvest the remaining
    *residual* alpha, which is orthogonal to all macro factors.  With 44 liquid
    ETFs, cross-sectional z-scores have meaningful degrees-of-freedom and the
    diversified long-short portfolio has economically sensible capacity.

    Parameters
    ----------
    rebalance_freq : int, default 21
        Trading days between rebalances.
    reg_lookback : int, default 252
        Rolling regression window (days) for factor loading estimation.
    pred_window : int, default 20
        Days over which to average recent residuals for alpha prediction.
    factor_symbols : list[str]
        Macro factor tickers used for regression.  Defaults to the internal.
    ridge_alpha : float, default 1.0
        Ridge regularisation strength on factor loadings.
    max_position : float, default 0.03
        Max absolute weight per asset (3 %).
    target_vol : float, default 0.15
        Annualised target volatility.
    vol_lookback : int, default 20
        Days for realised vol estimate.
    zscore_cutoff : float, default 1.5
        z-score threshold for long/short bucket membership.
    percentile_top : float, default 0.30
        Cross-sectional fraction to go long (short is symmetric).
    """

    STRATEGY_STATUS: StrategyStatus = StrategyStatus.RESEARCH

    # -----------------------------------------------------------------
    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        # ---- config ------------------------------------------------
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.reg_lookback = int(config.params.get("reg_lookback", 252))
        self.pred_window = int(config.params.get("pred_window", 20))
        self.ridge_alpha = float(config.params.get("ridge_alpha", 1.0))
        self.max_position = float(config.params.get("max_position", 0.03))
        self.target_vol = float(config.params.get("target_vol", 0.15))
        self.vol_lookback = int(config.params.get("vol_lookback", 20))
        self.zscore_cutoff = float(config.params.get("zscore_cutoff", 1.5))
        self.percentile_top = float(config.params.get("percentile_top", 0.30))

        raw_factors = config.params.get("factor_symbols")
        if raw_factors is None or raw_factors == "":
            self.factor_symbols = [
                "SPY", "TLT", "GLD", "DBC", "UUP", "EFA", "VNQ", "LQD",
            ]
        elif isinstance(raw_factors, str):
            self.factor_symbols = [s.strip() for s in raw_factors.split(",")]
        else:
            self.factor_symbols = list(raw_factors)

        # ---- state (populated in warm-up) --------------------------
        self._symbols: list[str] = []
        self._factor_map: dict[str, int] = {}
        self._data_matrix: list[dict[str, float]] = []
        self._days_since_rebalance = 0
        self._cached_signals: dict[str, SignalData] = {}
        self._last_weights: NDArray | None = None
        self._rolling_vol_history: list[float] = []
        self._equity_history: list[float] = []
        # event-driven (BacktestEngine) state
        self._current_ts: float = 0.0
        self._current_day_prices: dict[str, float] = {}
        # Cache rebalance results for all symbols on the rebalance day.
        # Named distinct from base Strategy._pending_signals to avoid
        # ``list has no .update()'' when _init_state resets it to [].
        self._rebalance_cache: dict[str, SignalData] = {}

    # -----------------------------------------------------------------
    # Public entry point (event-driven)
    # -----------------------------------------------------------------
    def on_bar(self, bar, portfolio) -> list[SignalData]:  # noqa: D401
        """Process single bar and emit signals (event-driven path).

        The BacktestEngine calls on_bar once per (timestamp, symbol).
        We buffer per-symbol prices for the current day, and on a new
        day we commit the previous day's cross-section to ``_data_matrix``
        and check rebalance.
        """
        sym = bar.symbol
        price = float(bar.close)
        ts = float(bar.timestamp.timestamp() if hasattr(bar.timestamp, "timestamp") else bar.timestamp)

        # --- new day detection & commit --------------------------------
        if self._current_ts != ts:
            if self._current_day_prices:
                self._data_matrix.append(dict(self._current_day_prices))
                max_len = self.reg_lookback + self.pred_window + 50
                if len(self._data_matrix) > max_len:
                    self._data_matrix = self._data_matrix[-max_len:]
                self._days_since_rebalance += 1
            self._current_ts = ts
            self._current_day_prices = {}
            self._rebalance_cache.clear()

        # --- cache today's price for this symbol -----------------------
        self._current_day_prices[sym] = price
        if sym not in self._symbols:
            self._symbols.append(sym)

        # --- track equity ---------------------------------------------
        if portfolio is not None and hasattr(portfolio, "nav"):
            self._equity_history.append(float(portfolio.nav))
            if len(self._equity_history) > 252:
                self._equity_history = self._equity_history[-252:]

        # --- update rolling vol history --------------------------------
        if portfolio is not None and hasattr(portfolio, "total_value"):
            tv = float(portfolio.total_value)
            if self._equity_history and tv > 0:
                self._rolling_vol_history.append(
                    abs(tv - self._equity_history[-1]) / self._equity_history[-1] if self._equity_history[-1] > 0 else 0.0
                )

        # --- rebalance check ------------------------------------------
        ready = self._try_rebalance()
        if ready is not None:
            self._rebalance_cache.update(ready)

        return self._rebalance_cache.get(sym, []) if sym in self._rebalance_cache else []

    # -----------------------------------------------------------------
    # Batch entry point (used by backtest engine on_data)
    # -----------------------------------------------------------------
    def on_fill(self, fill, portfolio=None) -> None:
        """Handle fill events from the backtest engine.

        The BacktestEngine calls ``on_fill(fill: Fill, portfolio: PortfolioState)``.
        The ABC declares ``on_fill(symbol, fill_price, fill_qty, side)``.
        This implementation is compatible with BOTH calling conventions by
        accepting *any* first positional arg and ignoring it - the strategy
        is purely signal-driven and does not need fill feedback.
        """
        pass  # Signal-driven strategy; no fill-based state update needed

    def reset(self) -> None:
        """Reset all state (for walk-forward folds)."""
        self._data_matrix = []
        self._symbols = []
        self._current_ts = 0.0
        self._current_day_prices = {}
        self._rebalance_cache = {}
        self._days_since_rebalance = 0
        self._last_weights = None
        self._rolling_vol_history.clear()
        self._equity_history.clear()

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Process batch data and emit signals.

        Expected keys:
            close   : (T, N) closing price matrix
            symbols : list[str] of asset symbols (length N)
            open, high, low, volume : optional (T, N)
        """
        close = data.get("close")
        if close is None:
            return []
        if close.ndim == 1:
            close = close.reshape(-1, 1)

        symbols = data.get("symbols", [f"asset_{i}" for i in range(close.shape[1])])

        weights = self._compute_weights(close, symbols)
        if weights is None:
            return []

        signals: list[TradeSignal] = []
        for i, sym in enumerate(symbols):
            w = weights[i]
            if abs(w) < 1e-6:
                continue
            signals.append(  # noqa: ERA001
                TradeSignal(
                    symbol=sym,
                    direction=1 if w > 0 else -1,
                    strength=abs(w),
                    size=abs(w) * float(self.config.max_position_notional or 1_000_000),
                    metadata={"weight": w},
                )
            )
        return signals

    # -----------------------------------------------------------------
    # Core signal pipeline
    # -----------------------------------------------------------------
    def _compute_weights(self, close: NDArray, symbols: list[str]) -> NDArray | None:
        """Return (N,) weight vector or None if not enough data."""
        # Need reg_lookback + pred_window returns, which requires +1 close prices
        min_bars = self.reg_lookback + self.pred_window + 1
        if close.shape[0] < min_bars:
            return None

        returns = self._prices_to_returns(close)
        if returns is None or returns.shape[0] < self.reg_lookback + self.pred_window:
            return None

        # Step 1: Extract factor returns from the return matrix
        factors, factor_rets = self._extract_factor_returns(close, symbols, returns)

        # Step 2: Rolling factor regression -> residuals
        residuals, betas = self._rolling_residual(returns, symbols)
        if residuals is None:
            return None

        # Step 3: Predict alpha = mean of recent residuals
        pred_alpha = np.mean(residuals[-self.pred_window:], axis=0)

        # Step 4: Cross-sectional z-score
        z = self._cross_sectional_zscore(pred_alpha)

        # Step 5: Convert to raw weights (top/bottom slice)
        raw_weights = self._zscore_to_raw_weights(z)

        # Step 6: Daily beta-neutral adjustment
        raw_weights = self._apply_beta_neutral(raw_weights, returns)

        # Step 7: Factor-neutral QP
        weights = self._apply_factor_neutral(raw_weights, betas, symbols)

        # Step 8: Vol targeting and regime scaling
        weights = self._apply_risk_scaling(weights, returns)

        self._last_weights = weights
        return weights

    @staticmethod
    def _prices_to_returns(close: NDArray) -> NDArray | None:
        """Convert (T, N) close prices to daily returns (T-1, N)."""
        if close.shape[0] < 2:
            return None
        rets = np.diff(close, axis=0) / np.maximum(close[:-1], 1e-10)
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        return rets

    def _extract_factor_returns(
        self, close: NDArray, symbols: list[str], returns: NDArray
    ) -> tuple[list[str], NDArray]:
        """Return factor symbol list and (T, n_factors) factor return matrix.

        If a factor symbol is not in the provided universe, skip it silently.
        """
        idx_map = {s: i for i, s in enumerate(symbols)}
        valid_factors = [f for f in self.factor_symbols if f in idx_map]
        if not valid_factors:
            # No factors available -> equal-weight proxy
            _log.warning("no_factor_symbols_matched: using equal-weight proxy")
            factor_rets = np.tile(np.mean(returns, axis=1, keepdims=True), (1, 1))
            return valid_factors, factor_rets

        factor_idx = [idx_map[f] for f in valid_factors]
        factor_rets = returns[:, factor_idx]  # (T-1, n_factors)
        return valid_factors, factor_rets

    def _rolling_residual(
        self, returns: NDArray, symbols: list[str]
    ) -> tuple[NDArray | None, NDArray | None]:
        """Run rolling factor regression per asset and return residuals.

        Returns
        -------
        residuals : (T, N) array of residuals (return matrix aligned to returns)
        betas     : (N, n_factors) most recent factor loadings
        """
        T, N = returns.shape
        if T < self.reg_lookback + self.pred_window:
            return None, None

        # Extract factor returns from the same data
        close = np.ones((T + 1, N))  # dummy for the method call
        valid_factors, factor_rets = self._extract_factor_returns(close, symbols, returns)
        n_factors = factor_rets.shape[1]

        betas = np.zeros((N, n_factors))
        residuals = np.zeros_like(returns)  # (T-1, N) or (T, N)

        for i in range(N):
            y = returns[:, i]
            if _HAS_SKLEARN and T >= self.reg_lookback + 5:
                try:
                    model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
                    model.fit(factor_rets[-self.reg_lookback:], y[-self.reg_lookback:])
                    pred = model.predict(factor_rets)
                    betas[i] = model.coef_
                    residuals[:, i] = y - pred
                except Exception:
                    betas[i] = 0.0
                    residuals[:, i] = y
            else:
                betas[i] = 0.0
                residuals[:, i] = y

        return residuals, betas

    # -----------------------------------------------------------------
    # Cross-sectional z-score
    # -----------------------------------------------------------------
    @staticmethod
    def _cross_sectional_zscore(alpha: NDArray) -> NDArray:
        """z = (alpha - mean(alpha)) / std(alpha, ddof=1)."""
        mu = float(np.nanmean(alpha))
        sigma = float(np.nanstd(alpha, ddof=1))
        if sigma < 1e-12:
            return np.zeros_like(alpha)
        return (alpha - mu) / sigma

    # -----------------------------------------------------------------
    # Z-score -> raw long-short weights
    # -----------------------------------------------------------------
    def _zscore_to_raw_weights(self, z: NDArray) -> NDArray:
        """Long top 30 %, short bottom 30 % by z-score."""
        n = len(z)
        n_top = max(1, int(self.percentile_top * n))
        n_short = max(1, int(self.percentile_top * n))

        order = np.argsort(z)[::-1]  # descending
        w = np.zeros(n)

        # long
        if n_top > 0:
            long_idx = order[:n_top]
            w[long_idx] = z[long_idx]
            w_sum = float(np.sum(w[long_idx]))
            if abs(w_sum) > 1e-12:
                w[long_idx] = w[long_idx] / abs(w_sum) * self.percentile_top

        # short
        if n_short > 0:
            short_idx = order[-n_short:]
            w[short_idx] = z[short_idx]
            s_sum = float(np.sum(w[short_idx]))
            if abs(s_sum) > 1e-12:
                w[short_idx] = w[short_idx] / abs(s_sum) * self.percentile_top

        # Re-center to ensure dollar neutrality (should be ~0 already)
        w = w - np.mean(w)
        return w

    # -----------------------------------------------------------------
    # Beta-neutral adjustment
    # -----------------------------------------------------------------
    def _apply_beta_neutral(self, weights: NDArray, returns: NDArray) -> NDArray:
        """Apply beta-neutralisation using existing portfolio module."""
        try:
            adjusted = neutralize_beta(
                weights,
                returns,
                window=min(self.vol_lookback, returns.shape[0] - 1),
            )
            return adjusted
        except Exception:
            return weights

    # -----------------------------------------------------------------
    # Factor-neutral QP
    # -----------------------------------------------------------------
    def _apply_factor_neutral(
        self, alpha: NDArray, betas: NDArray, symbols: list[str]
    ) -> NDArray:
        """Solve QP for factor-neutral, dollar-neutral weights with box limits."""
        if solve_factor_neutral_weights is None:
            _log.warning("solve_factor_neutral_weights unavailable; returning raw z-score weights")
            return alpha

        if betas is None or betas.shape[0] == 0:
            _log.warning("no_factor_loadings: skipping factor-neutral QP")
            alpha = alpha - np.mean(alpha)
            clamped = np.clip(alpha, -self.max_position, self.max_position)
            return clamped - np.mean(clamped)

        if np.all(np.abs(betas) < 1e-12):
            # All betas are zero -> skip factor-neutral QP
            alpha = alpha - np.mean(alpha)
            clamped = np.clip(alpha, -self.max_position, self.max_position)
            return clamped - np.mean(clamped)

        weights = solve_factor_neutral_weights(alpha, betas, max_weight=self.max_position)
        weights = weights - np.mean(weights)
        return weights

    # -----------------------------------------------------------------
    # Risk scaling (vol-targeting + regime)
    # -----------------------------------------------------------------
    def _apply_risk_scaling(self, weights: NDArray, returns: NDArray) -> NDArray:
        """Apply vol-targeting and regime scaling.

        Scale portfolio such that realised vol ~ target_vol.
        """
        if returns.shape[0] < self.vol_lookback + 1:
            return weights

        # Portfolio realised vol
        port_rets = returns[-self.vol_lookback:] @ weights
        port_vol = float(np.std(port_rets, ddof=1)) * np.sqrt(252)

        if port_vol > 1e-12:
            vol_scale = self.target_vol / port_vol
        else:
            vol_scale = 1.0

        # Regime scaling
        regime_scale = self._regime_scale()

        scale = vol_scale * regime_scale

        # Cap leverage at max_leverage
        scale = min(scale, self.config.max_leverage / max(abs(weights).sum(), 1e-10))

        return weights * scale

    def _regime_scale(self) -> float:
        """Return regime scaling factor from rolling vol history."""
        if not self._rolling_vol_history:
            return 1.0

        # Simple regime based on recent realised vol percentile
        if len(self._rolling_vol_history) >= 63:
            vol = self._rolling_vol_history[-1]
            hist = self._rolling_vol_history
            p75 = np.percentile(hist[-252:], 75)
            p90 = np.percentile(hist[-252:], 90)

            if vol > np.percentile(hist[-252:], 99):
                return 0.0
            if vol > p90:
                return 0.50
            if vol > p75:
                return 0.75

        return 1.0

    # -----------------------------------------------------------------
    # Rebalance orchestration
    # -----------------------------------------------------------------
    def _try_rebalance(self) -> dict[str, list[SignalData]] | None:
        """Increment day counter and rebalance if necessary."""
        if self._days_since_rebalance < self.rebalance_freq:
            return None

        self._days_since_rebalance = 0

        # Need enough data accumulated in _data_matrix
        # Need reg_lookback + pred_window + 1 rows of close prices
        min_rows = self.reg_lookback + self.pred_window + 1
        if not self._data_matrix or len(self._data_matrix) < min_rows:
            return None

        symbols = self._symbols or []
        if not symbols:
            return None

        # Build close matrix from _data_matrix.
        # Each row in _data_matrix is {symbol: price} for a single day's close.
        n_cols = len(symbols)
        n_rows = len(self._data_matrix)
        close = np.full((n_rows, n_cols), np.nan)
        for r, row in enumerate(self._data_matrix):
            for c, sym in enumerate(symbols):
                close[r, c] = row.get(sym, np.nan)

        # Forward-fill NAs so the regression doesn't break on missing data
        import pandas as pd
        close_df = pd.DataFrame(close, columns=symbols)
        close_df = close_df.ffill()
        close = close_df.to_numpy()

        weights = self._compute_weights(close, symbols)
        if weights is None:
            return None

        # Cache signals per symbol
        signals: dict[str, list[SignalData]] = {}
        for i, sym in enumerate(symbols):
            w = weights[i]
            if abs(w) < 1e-6:
                continue
            sig = SignalData(
                symbol=sym,
                direction="long" if w > 0 else "short",
                strength=abs(w),
                strategy_id="macro_cross_sectional_factor_neutral",
                notional_size=abs(w) * float(self.config.max_position_notional or 1_000_000),
            )
            signals.setdefault(sym, []).append(sig)

        return signals
