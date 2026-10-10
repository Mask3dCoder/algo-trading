"""Multi-strategy portfolio aggregator — the meta-strategy layer.

Takes signals from any number of independent alpha sources, combines them
into a single market-neutral portfolio, and manages risk dynamically.

The aggregator itself implements the Strategy protocol (on_data / on_bar)
so it can be plugged into the backtest engine as if it were a single strategy.

Combination methods:
  - equal_weight: each strategy gets equal capital allocation.
  - risk_parity: each strategy weighted inversely to trailing realized vol.
  - mean_variance: maximize trailing Sharpe via scipy.optimize.
  - hrp: hierarchical risk parity (López de Prado 2016) — cluster-based,
         more robust to estimation error than mean-variance.

Market neutrality: net beta ≈ 0 vs SPY via rolling beta estimation.
Position limits: no single name > max_position_pct of gross, max leverage cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal

# ── Configuration ──────────────────────────────────────────────────────────


@dataclass
class AggregatorConfig:
    """Configuration for MultiStrategyAggregator."""
    name: str = "multi_strategy_aggregator"
    combination_method: Literal["equal_weight", "risk_parity", "mean_variance", "hrp", "black_litterman"] = "risk_parity"
    weight_update_freq: int = 5           # bars between weight recalculations
    vol_lookback: int = 60                # trailing window for risk parity / MVO
    max_position_pct: float = 0.05        # 5% per name
    max_gross_leverage: float = 2.0       # 2x default
    beta_neutral: bool = True             # enforce net beta ≈ 0 vs SPY
    beta_lookback: int = 60               # rolling window for beta estimation
    market_proxy: str = "SPY"             # benchmark for beta neutrality
    negative_sharpe_cutoff: float = 0.0   # exclude strategies below this Sharpe
    initial_cash: float = 1_000_000.0
    rf_rate: float = 0.04                 # annualized risk-free rate


# ── Sub-strategy wrapper ───────────────────────────────────────────────────


@dataclass
class SubStrategy:
    """A strategy instance bundled with its configuration."""
    strategy_id: str
    strategy_cls: type[Strategy]
    config: StrategyConfig
    instance: Strategy | None = None
    enabled: bool = True
    current_weight: float = 0.0           # set by aggregator

    def instantiate(self) -> Strategy:
        self.instance = self.strategy_cls(self.config)
        self.instance.initialize()
        return self.instance


# ── Multi-strategy aggregator ──────────────────────────────────────────────


class MultiStrategyAggregator:
    """Combine signals from multiple strategies into one market-neutral portfolio.

    Implements on_data() so it behaves like a single Strategy to the backtest
    runner. Internally manages sub-strategy instances, tracks equity curves,
    and periodically re-optimizes strategy weights.

    Usage:
        subs = [
            SubStrategy("strat_a", ClsA, cfg_a),
            SubStrategy("strat_b", ClsB, cfg_b),
        ]
        agg = MultiStrategyAggregator(subs, AggregatorConfig(...))
        signals = agg.on_data(bar_data)   # returns list[TradeSignal]
    """

    def __init__(
        self,
        sub_strategies: list[SubStrategy],
        config: AggregatorConfig | None = None,
    ) -> None:
        self._cfg = config or AggregatorConfig()
        self._sub_strategies = list(sub_strategies)

        # Instantiate all sub-strategies
        for sub in self._sub_strategies:
            if sub.instance is None:
                sub.instantiate()

        n = len(self._sub_strategies)
        self._strategy_weights = np.ones(n) / max(n, 1)

        # Price history for return / beta estimation
        self._close_history: NDArray[np.floating] | None = None
        self._returns_history: NDArray[np.floating] | None = None

        # Per-strategy equity curve tracking
        self._strategy_equity: list[list[float]] = [[] for _ in range(n)]
        self._strategy_prev_weights: list[dict[str, float]] = [{} for _ in range(n)]

        # Aggregator-level tracking
        self._agg_equity_curve: list[float] = [self._cfg.initial_cash]
        self._agg_prev_weights: dict[str, float] = {}

        # Weight update scheduling
        self._bars_since_rebalance = self._cfg.weight_update_freq  # trigger on first bar
        self._n_bars = 0

    # ── Main interface ─────────────────────────────────────────────────────

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Collect signals from all sub-strategies and return combined portfolio.

        This is the main entry point. It:
          1. Feeds data to each sub-strategy.
          2. Collects TradeSignal outputs.
          3. Converts to per-strategy, per-asset weight vectors.
          4. Combines using configured method (equal / risk-parity / MVO).
          5. Enforces beta neutrality and position limits.
          6. Returns combined TradeSignal list.
        """
        close = data.get("close")
        if close is None or not isinstance(close, np.ndarray):
            return []
        if close.ndim < 2 or close.shape[0] < 2:
            return []

        symbols: list[str] = list(data.get("symbols", []))
        n_assets = close.shape[1]
        if len(symbols) != n_assets:
            symbols = [f"asset_{i}" for i in range(n_assets)]

        # Update price history
        self._update_price_history(close)

        # Compute asset returns for this bar (if we have enough history)
        bar_returns = self._compute_bar_returns(close)

        # ── 1. Collect signals from each sub-strategy ──
        strategy_weight_vecs: list[NDArray[np.floating]] = []
        active_indices: list[int] = []

        for i, sub in enumerate(self._sub_strategies):
            if not sub.enabled or sub.instance is None:
                strategy_weight_vecs.append(np.zeros(n_assets))
                continue

            try:
                signals = sub.instance.on_data(data) or []
            except Exception:
                signals = []

            w = np.zeros(n_assets)
            for sig in signals:
                if sig.symbol in symbols:
                    idx = symbols.index(sig.symbol)
                    w[idx] = float(sig.direction) * float(sig.strength)

            # Normalize per-strategy to unit gross exposure
            gross = np.sum(np.abs(w))
            if gross > 1e-10:
                w = w / gross

            strategy_weight_vecs.append(w)
            if gross > 1e-10:
                active_indices.append(i)

            # Track equity for this strategy
            if bar_returns is not None:
                prev_w = self._strategy_prev_weights[i]
                strat_ret = 0.0
                for sym, wt in prev_w.items():
                    if sym in symbols:
                        strat_ret += wt * float(bar_returns[symbols.index(sym)])
                current_eq = self._strategy_equity[i][-1] if self._strategy_equity[i] else self._cfg.initial_cash
                self._strategy_equity[i].append(current_eq * (1.0 + strat_ret))

            # Store current weights for next-bar return computation
            self._strategy_prev_weights[i] = {
                symbols[j]: float(w[j]) for j in range(n_assets) if abs(w[j]) > 1e-10
            }

        # ── 2. Update strategy weights periodically ──
        self._bars_since_rebalance += 1
        if self._bars_since_rebalance >= self._cfg.weight_update_freq:
            self._update_strategy_weights(active_indices)
            self._bars_since_rebalance = 0

        # ── 3. Combine weights ──
        combined = np.zeros(n_assets)
        for i in range(len(self._sub_strategies)):
            combined += self._strategy_weights[i] * strategy_weight_vecs[i]

        # ── 4. Re-center for dollar neutrality ──
        net = np.sum(combined)
        combined -= net / max(n_assets, 1)

        # ── 5. Beta neutralization ──
        if self._cfg.beta_neutral and self._returns_history is not None:
            combined = self._neutralize_beta(combined, symbols)

        # ── 6. Enforce position limits ──
        combined = self._apply_position_limits(combined)

        # ── 7. Convert to TradeSignals ──
        signals = self._weights_to_signals(combined, symbols)

        # ── 8. Track aggregator equity ──
        if bar_returns is not None:
            agg_ret = 0.0
            for sym, wt in self._agg_prev_weights.items():
                if sym in symbols:
                    agg_ret += wt * float(bar_returns[symbols.index(sym)])
            current_eq = self._agg_equity_curve[-1]
            self._agg_equity_curve.append(current_eq * (1.0 + agg_ret))

        self._agg_prev_weights = {
            symbols[j]: float(combined[j])
            for j in range(n_assets) if abs(combined[j]) > 1e-10
        }
        self._n_bars += 1

        return signals

    def on_fill(self, symbol: str, fill_price: float, fill_qty: float, side: str) -> None:
        """Forward fill notifications to sub-strategies."""
        for sub in self._sub_strategies:
            if sub.instance is not None:
                try:
                    sub.instance.on_fill(symbol, fill_price, fill_qty, side)
                except Exception:
                    pass

    def teardown(self) -> None:
        for sub in self._sub_strategies:
            if sub.instance is not None:
                try:
                    sub.instance.teardown()
                except Exception:
                    pass

    # ── Strategy weight computation ────────────────────────────────────────

    def _update_strategy_weights(self, active_indices: list[int]) -> None:
        """Recalculate strategy allocation weights."""
        n = len(self._sub_strategies)
        method = self._cfg.combination_method

        if not active_indices:
            self._strategy_weights = np.zeros(n)
            return

        if method == "equal_weight":
            self._compute_equal_weights(active_indices, n)
        elif method == "risk_parity":
            self._compute_risk_parity_weights(active_indices, n)
        elif method == "mean_variance":
            self._compute_mean_variance_weights(active_indices, n)
        elif method == "hrp":
            self._compute_hrp_weights(active_indices, n)
        elif method == "black_litterman":
            self._compute_bl_weights(active_indices, n)
        else:
            self._compute_equal_weights(active_indices, n)

    def _compute_equal_weights(self, active: list[int], n: int) -> None:
        w = np.zeros(n)
        for i in active:
            w[i] = 1.0 / max(len(active), 1)
        self._strategy_weights = w

    def _compute_risk_parity_weights(self, active: list[int], n: int) -> None:
        """Weight each strategy inversely to trailing realized volatility."""
        lookback = self._cfg.vol_lookback
        vols = np.zeros(n)
        valid = []

        for i in active:
            eq = self._strategy_equity[i]
            if len(eq) < max(lookback, 10):
                vols[i] = 1.0
                valid.append(i)
                continue
            eq_arr = np.array(eq[-lookback:])
            rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-10)
            vol = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1.0
            vols[i] = max(vol, 1e-8)
            valid.append(i)

        # Filter out strategies with negative Sharpe
        filtered = []
        for i in valid:
            sharpe = self._compute_trailing_sharpe(i)
            if sharpe >= self._cfg.negative_sharpe_cutoff:
                filtered.append(i)

        if not filtered:
            filtered = valid  # fallback: keep all

        inv_vols = np.zeros(n)
        for i in filtered:
            inv_vols[i] = 1.0 / vols[i]

        total = np.sum(inv_vols)
        if total > 0:
            self._strategy_weights = inv_vols / total
        else:
            self._compute_equal_weights(filtered, n)

    def _compute_mean_variance_weights(self, active: list[int], n: int) -> None:
        """Maximize trailing Sharpe via scipy constrained optimization."""
        try:
            from scipy.optimize import minimize as sp_minimize
        except ImportError:
            self._compute_risk_parity_weights(active, n)
            return

        lookback = self._cfg.vol_lookback

        # Filter out strategies with negative Sharpe
        filtered = []
        for i in active:
            sharpe = self._compute_trailing_sharpe(i)
            if sharpe >= self._cfg.negative_sharpe_cutoff:
                filtered.append(i)
        if not filtered:
            filtered = list(active)
        if len(filtered) < 2:
            self._compute_equal_weights(filtered, n)
            return

        # Build return matrix
        returns_list = []
        for i in filtered:
            eq = self._strategy_equity[i]
            if len(eq) < max(lookback, 10):
                eq_arr = np.array([self._cfg.initial_cash] * max(lookback, 10))
            else:
                eq_arr = np.array(eq[-lookback:])
            rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-10)
            returns_list.append(rets)

        min_len = min(len(r) for r in returns_list)
        returns_mat = np.column_stack([r[:min_len] for r in returns_list])
        n_strats = returns_mat.shape[1]

        if n_strats < 2 or returns_mat.shape[0] < 10:
            self._compute_equal_weights(filtered, n)
            return

        cov = np.cov(returns_mat, rowvar=False)
        if cov.ndim == 1:
            cov = cov.reshape(1, 1)
        mean_rets = np.mean(returns_mat, axis=0)

        def neg_sharpe(w_sub: NDArray) -> float:
            port_ret = float(w_sub @ mean_rets)
            port_var = float(w_sub @ cov @ w_sub)
            port_vol = np.sqrt(max(port_var, 1e-16))
            return -(port_ret / port_vol) if port_vol > 1e-10 else 0.0

        x0 = np.ones(n_strats) / n_strats
        bounds = [(0.0, 1.0)] * n_strats
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        try:
            result = sp_minimize(
                neg_sharpe, x0, method="SLSQP",
                bounds=bounds, constraints=cons,
                options={"maxiter": 200, "ftol": 1e-12},
            )
            opt_w = result.x if result.success else x0
        except Exception:
            opt_w = x0

        # Map back to full strategy array
        w = np.zeros(n)
        for j, i in enumerate(filtered):
            w[i] = float(opt_w[j])
        self._strategy_weights = w

    def _compute_hrp_weights(self, active: list[int], n: int) -> None:
        """Compute strategy weights via Hierarchical Risk Parity."""
        import pandas as pd

        from quantcore.portfolio.hrp import compute_hrp_weights

        lookback = self._cfg.vol_lookback

        # Filter out strategies with negative Sharpe
        filtered = []
        for i in active:
            sharpe = self._compute_trailing_sharpe(i)
            if sharpe >= self._cfg.negative_sharpe_cutoff:
                filtered.append(i)
        if not filtered:
            filtered = list(active)
        if len(filtered) < 2:
            self._compute_equal_weights(filtered, n)
            return

        # Build return DataFrame
        returns_dict: dict[int, np.ndarray] = {}
        for i in filtered:
            eq = self._strategy_equity[i]
            if len(eq) < max(lookback, 10):
                eq_arr = np.array([self._cfg.initial_cash] * max(lookback, 10))
            else:
                eq_arr = np.array(eq[-lookback:])
            rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-10)
            returns_dict[i] = rets

        min_len = min(len(r) for r in returns_dict.values())
        if min_len < 10:
            self._compute_equal_weights(filtered, n)
            return

        df = pd.DataFrame({i: returns_dict[i][:min_len] for i in filtered})
        weights_dict = compute_hrp_weights(df)

        # Map back to full strategy array
        w = np.zeros(n)
        for i in filtered:
            w[i] = weights_dict.get(i, 0.0)

        total = np.sum(w)
        if total > 0:
            w /= total
        self._strategy_weights = w

    def _compute_bl_weights(self, active: list[int], n: int) -> None:
        """Compute strategy weights via Black-Litterman optimization."""
        import pandas as pd

        from quantcore.portfolio.black_litterman import compute_bl_weights

        lookback = self._cfg.vol_lookback

        # Filter out strategies with negative Sharpe
        filtered = []
        for i in active:
            sharpe = self._compute_trailing_sharpe(i)
            if sharpe >= self._cfg.negative_sharpe_cutoff:
                filtered.append(i)
        if not filtered:
            filtered = list(active)
        if len(filtered) < 2:
            self._compute_equal_weights(filtered, n)
            return

        # Build return DataFrame
        returns_dict: dict[int, np.ndarray] = {}
        for i in filtered:
            eq = self._strategy_equity[i]
            if len(eq) < max(lookback, 10):
                eq_arr = np.array([self._cfg.initial_cash] * max(lookback, 10))
            else:
                eq_arr = np.array(eq[-lookback:])
            rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-10)
            returns_dict[i] = rets

        min_len = min(len(r) for r in returns_dict.values())
        if min_len < 10:
            self._compute_equal_weights(filtered, n)
            return

        df = pd.DataFrame({i: returns_dict[i][:min_len] for i in filtered})
        weights_arr = compute_bl_weights(df)

        # Map back to full strategy array
        w = np.zeros(n)
        for j, i in enumerate(filtered):
            w[i] = float(weights_arr[j])

        total = np.sum(w)
        if total > 0:
            w /= total
        self._strategy_weights = w

    # ── Beta neutralization ────────────────────────────────────────────────

    def _neutralize_beta(
        self, weights: NDArray[np.floating], symbols: list[str],
    ) -> NDArray[np.floating]:
        """Adjust weights so portfolio beta ≈ 0 vs market proxy (SPY)."""
        if self._returns_history is None or self._returns_history.shape[0] < self._cfg.beta_lookback:
            return weights

        lookback = min(self._cfg.beta_lookback, self._returns_history.shape[0])
        rets = self._returns_history[-lookback:]
        proxy = self._cfg.market_proxy

        if proxy not in symbols:
            return weights

        proxy_idx = symbols.index(proxy)
        mkt_returns = rets[:, proxy_idx]

        # Compute asset betas vs market proxy
        mkt_var = float(np.var(mkt_returns, ddof=1))
        if mkt_var < 1e-10:
            return weights

        n_assets = len(weights)
        betas = np.zeros(n_assets)
        for i in range(n_assets):
            if i == proxy_idx:
                betas[i] = 1.0
                continue
            col = rets[:, i]
            valid = ~np.isnan(col) & ~np.isnan(mkt_returns)
            if valid.sum() < 10:
                betas[i] = 0.0
                continue
            cov_val = float(np.cov(col[valid], mkt_returns[valid], ddof=1)[0, 1])
            betas[i] = cov_val / mkt_var

        # Portfolio beta = sum(w_i * beta_i)
        port_beta = float(np.dot(weights, betas))

        if abs(port_beta) < 1e-6:
            return weights

        # Adjust: subtract beta exposure via proxy asset
        # new_w[proxy] -= port_beta / beta[proxy]
        proxy_beta = betas[proxy_idx]
        if abs(proxy_beta) > 1e-6:
            weights[proxy_idx] -= port_beta / proxy_beta
        else:
            # Spread adjustment across all assets
            beta_sq = float(np.dot(betas, betas))
            if beta_sq > 1e-10:
                weights -= betas * (port_beta / beta_sq)

        # Re-center for dollar neutrality
        weights -= np.mean(weights)

        return weights

    # ── Position limits ────────────────────────────────────────────────────

    def _apply_position_limits(self, weights: NDArray[np.floating]) -> NDArray[np.floating]:
        """Enforce max per-name concentration and gross leverage cap."""
        w = np.array(weights, dtype=float).copy()

        # Cap individual positions
        max_pct = self._cfg.max_position_pct
        w = np.clip(w, -max_pct, max_pct)

        # Cap gross leverage
        gross = float(np.sum(np.abs(w)))
        if gross > self._cfg.max_gross_leverage:
            w *= self._cfg.max_gross_leverage / gross

        # Re-center for dollar neutrality
        w -= np.mean(w)

        return w

    # ── Price / return tracking ────────────────────────────────────────────

    def _update_price_history(self, close: NDArray[np.floating]) -> None:
        """Append latest close prices to rolling history."""
        latest = close[-1, :].reshape(1, -1)
        if self._close_history is None:
            self._close_history = latest.copy()
        else:
            self._close_history = np.vstack([self._close_history, latest])

        # Keep only what we need
        max_len = self._cfg.vol_lookback + self._cfg.beta_lookback + 10
        if self._close_history.shape[0] > max_len:
            self._close_history = self._close_history[-max_len:]

    def _compute_bar_returns(self, close: NDArray[np.floating]) -> NDArray[np.floating] | None:
        """Compute bar-level returns and append to history."""
        if close.shape[0] < 2:
            return None

        prev = close[-2, :]
        curr = close[-1, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.where(np.abs(prev) > 1e-10, (curr - prev) / prev, 0.0)
        rets = np.nan_to_num(rets, nan=0.0)

        if self._returns_history is None:
            self._returns_history = rets.reshape(1, -1)
        else:
            self._returns_history = np.vstack([self._returns_history, rets])

        max_len = self._cfg.vol_lookback + self._cfg.beta_lookback + 10
        if self._returns_history.shape[0] > max_len:
            self._returns_history = self._returns_history[-max_len:]

        return rets

    # ── Helpers ────────────────────────────────────────────────────────────

    def _compute_trailing_sharpe(self, strategy_idx: int) -> float:
        """Annualized Sharpe ratio from trailing equity curve."""
        eq = self._strategy_equity[strategy_idx]
        lookback = self._cfg.vol_lookback
        if len(eq) < max(lookback, 10):
            return 0.0
        eq_arr = np.array(eq[-lookback:])
        rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-10)
        if len(rets) < 5:
            return 0.0
        std = float(np.std(rets, ddof=1))
        if std < 1e-10:
            return 0.0
        rf_daily = (1.0 + self._cfg.rf_rate) ** (1.0 / 252.0) - 1.0
        mean_excess = float(np.mean(rets)) - rf_daily
        return (mean_excess / std) * np.sqrt(252.0)

    def _weights_to_signals(
        self, weights: NDArray[np.floating], symbols: list[str],
    ) -> list[TradeSignal]:
        """Convert a weight vector to TradeSignal objects.

        Emits raw combined weights as strength. The backtest runner's
        signal-return computation normalizes by total abs strength,
        so this produces correct unit-gross exposure automatically.
        """
        signals: list[TradeSignal] = []

        for i, sym in enumerate(symbols):
            w = float(weights[i])
            if abs(w) < 1e-6:
                continue

            direction = 1 if w > 0 else -1
            strength = abs(w)

            signals.append(TradeSignal(
                symbol=sym,
                direction=direction,
                strength=strength,
                size=abs(w) * self._cfg.initial_cash,
                metadata={
                    "source": "aggregator",
                    "raw_weight": w,
                    "method": self._cfg.combination_method,
                },
            ))

        return signals

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def strategy_weights(self) -> dict[str, float]:
        """Current allocation weight per sub-strategy."""
        return {
            sub.strategy_id: float(self._strategy_weights[i])
            for i, sub in enumerate(self._sub_strategies)
        }

    @property
    def equity_curve(self) -> list[float]:
        return list(self._agg_equity_curve)

    @property
    def n_bars(self) -> int:
        return self._n_bars

    def get_strategy_equity(self, strategy_id: str) -> list[float]:
        for i, sub in enumerate(self._sub_strategies):
            if sub.strategy_id == strategy_id:
                return list(self._strategy_equity[i])
        return []


# ── Pre-trade exposure checks (used by live runner) ────────────────────────


class CheckVerdict(Enum):
    """Result of a pre-trade aggregator check."""
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class GlobalExposureLimits:
    """Cross-strategy exposure limits for pre-trade checks."""
    max_gross_exposure: float = 20_000_000.0   # max total notional
    max_single_name_pct: float = 0.10          # max 10% in one name
    max_leverage: float = 1.0                  # max gross leverage


class StrategyAggregator:
    """Lightweight pre-trade aggregator for the live runner.

    Validates that a proposed order won't breach global exposure limits.
    This is a simplified version of MultiStrategyAggregator focused on
    the live runner's pre-trade check path.
    """

    def __init__(self, limits: GlobalExposureLimits | None = None) -> None:
        self._limits = limits or GlobalExposureLimits()
        self._gross_exposure: float = 0.0
        self._name_exposure: dict[str, float] = {}
        self._equity: float = 0.0

    def update_exposure(self, symbol: str, notional: float) -> None:
        """Update tracked exposure after a fill."""
        self._name_exposure[symbol] = self._name_exposure.get(symbol, 0.0) + notional
        self._gross_exposure = sum(abs(v) for v in self._name_exposure.values())

    def update_positions(
        self, strategy_id: str, positions: dict[str, object], cash: float
    ) -> None:
        """Refresh tracked exposure from current positions dict."""
        self._name_exposure.clear()
        for sym, pos in positions.items():
            try:
                qty = abs(float(getattr(pos, "quantity", 0)))
                price = float(getattr(pos, "market_price", 0))
                self._name_exposure[sym] = qty * price
            except Exception:
                pass
        self._gross_exposure = sum(abs(v) for v in self._name_exposure.values())

    def update_equity(self, equity: float) -> None:
        """Update equity for percentage-based exposure checks."""
        self._equity = equity

    def check_order(self, order: Any) -> CheckVerdict:
        """Check whether an order would breach global exposure limits.

        Returns CheckVerdict.APPROVED if the order is within limits,
        CheckVerdict.REJECTED if it would breach.
        """
        limits = self._limits

        # Estimate order notional
        price = float(getattr(order, "limit_price", None) or 0.0)
        qty = float(getattr(order, "quantity", 0))
        if price <= 0:
            return CheckVerdict.APPROVED  # can't check without price
        notional = price * qty

        # Check gross exposure
        if self._gross_exposure + notional > limits.max_gross_exposure:
            return CheckVerdict.REJECTED

        # Check single-name concentration
        symbol = str(getattr(order, "symbol", ""))
        current_name = abs(self._name_exposure.get(symbol, 0.0))
        if limits.max_gross_exposure > 0:
            new_name_pct = (current_name + notional) / limits.max_gross_exposure
            if new_name_pct > limits.max_single_name_pct:
                return CheckVerdict.REJECTED

        return CheckVerdict.APPROVED
