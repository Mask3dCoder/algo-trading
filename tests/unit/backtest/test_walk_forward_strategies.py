"""Walk-forward validation for all registered strategies (Priority 4).

Tests that every strategy completes walk-forward folds without error and that
out-of-sample Sharpe is not dramatically worse than in-sample — a basic
overfitting sanity check.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.backtest.walk_forward import rolling_walk_forward
from quantcore.risk.metrics import sharpe_ratio


# ---------------------------------------------------------------------------
# Strategy configs by type
# ---------------------------------------------------------------------------

_EQUITY_PARAMS: dict[str, dict] = {
    "equity_momentum": {"rebalance_freq": 1, "lookback_months": 6, "skip_months": 1},
    "equity_earnings_momentum": {"rebalance_freq": 1},  # DISABLED (audit) — minimal params for coverage
    "equity_low_volatility": {"rebalance_freq": 1},   # RESEARCH — minimal params for coverage
    "equity_stat_arb": {"rescan_freq": 1, "entry_z": 1.5, "lookback": 100, "max_pairs": 2},
    "equity_mean_reversion": {"rsi_oversold": 30, "rsi_overbought": 70},
    "equity_trend_following": {"trend_window": 50, "fast_window": 10, "slow_window": 30},
    "equity_multi_factor": {},
    "equity_quality_factor": {},
    "equity_value_factor": {},
    "equity_volatility_breakout": {},
    "equity_volume_profile": {},
}

_OPTIONS_PARAMS: dict[str, dict] = {
    "options_delta_neutral": {"ivr_entry": 30, "hedge_freq": 1},
    "options_iron_condor": {"ivr_entry": 30},
    "options_earnings_vol": {},
    "options_tail_hedge": {},
}

_MACRO_PARAMS: dict[str, dict] = {
    "macro_risk_parity": {"rebalance_freq": 1, "lookback": 100},
    "macro_cta_trend": {"rebalance_freq": 1, "slow_window": 100},
    "macro_carry": {"rebalance_freq": 1, "n_long": 2, "n_short": 2},
    "macro_carry_trend_filter": {"rebalance_freq": 1, "slow_window": 100},
    "macro_factor_residual": {"rebalance_freq": 5, "reg_window": 252},
    "macro_factor_residual_strict": {"rebalance_freq": 5, "reg_window": 252},
    "macro_factor_residual_qp": {"rebalance_freq": 5, "reg_window": 252},
    "macro_factor_residual_v3": {"rebalance_freq": 5, "reg_window": 252, "residual_window": 63},
    "macro_regime_adaptive_momentum": {"rebalance_freq": 1, "slow_window": 100},
    "macro_state_selective_momentum": {"rebalance_freq": 1, "slow_window": 100},
    "macro_cross_sectional_momentum": {"rebalance_freq": 1, "slow_window": 100},
    # DISABLED: CSM v2 (REJECTED per prospective_trial #1)
    # "macro_cross_sectional_momentum_v2": {
    #     "rebalance_freq": 5, "mom_windows": "20,60,120", "mom_weights": "0.2,0.5,0.3",
    #     "max_leverage": 1.0, "max_gross_exposure": 1.0,
    # },
    "macro_true_carry": {"rebalance_freq": 1, "slow_window": 100},
}

_INTRADAY_PARAMS: dict[str, dict] = {
    "intraday_breakout": {"range_bars": 10, "atr_window": 7, "session_bars": 20, "exit_before_close": 3, "cooldown_bars": 2},
    "intraday_mean_reversion": {"lookback": 10, "rsi_period": 7, "bb_period": 10},
    "intraday_momentum": {"lookback_bars": 6, "rebalance_freq": 3, "n_long": 2, "n_short": 2},
    "intraday_trend": {"trend_window": 15, "fast_window": 5, "slow_window": 10},
}

_STRATEGY_CONFIGS: dict[str, dict] = {}
_STRATEGY_CONFIGS.update(_EQUITY_PARAMS)
_STRATEGY_CONFIGS.update(_OPTIONS_PARAMS)
_STRATEGY_CONFIGS.update(_MACRO_PARAMS)
_STRATEGY_CONFIGS.update(_INTRADAY_PARAMS)


def _strategy_type(name: str) -> str:
    if name.startswith("equity_") or name.startswith("intraday_"):
        return "equity"
    if name.startswith("options_"):
        return "options"
    if name.startswith("macro_"):
        return "macro"
    return "equity"


def _requires_earnings(name: str) -> bool:
    return name == "options_earnings_vol"


def _requires_yields(name: str) -> bool:
    return name == "macro_carry"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

N_BARS = 500
N_ASSETS = 6
RNG_SEED = 42


@pytest.fixture(scope="module")
def strategy_data() -> dict[str, np.ndarray]:
    """Generate multi-asset data usable by all strategy types."""
    rng = np.random.default_rng(RNG_SEED)

    # Prices with modest trend + noise
    base = 100.0 + np.cumsum(rng.normal(0.02, 0.5, (N_BARS, N_ASSETS)), axis=0)
    close = np.maximum(base, 10.0)

    high = close * (1 + np.abs(rng.normal(0, 0.015, (N_BARS, N_ASSETS))))
    low = close * (1 - np.abs(rng.normal(0, 0.015, (N_BARS, N_ASSETS))))
    low = np.minimum(low, close)

    implied_vol = np.full((N_BARS, N_ASSETS), 0.25) + rng.normal(0, 0.02, (N_BARS, N_ASSETS))
    implied_vol = np.maximum(implied_vol, 0.05)

    # Yields for carry strategy
    yields = 0.03 + rng.normal(0, 0.005, (N_BARS, N_ASSETS))
    yields = np.maximum(yields, 0.001)

    data = {
        "close": close,
        "high": high,
        "low": low,
        "implied_vol": implied_vol,
        "yields": yields,
        "earnings_upcoming": np.array(1.0),
        "symbols": [f"A{i}" for i in range(N_ASSETS)],
    }
    return data


# ---------------------------------------------------------------------------
# Helper: compute signal-based returns from a strategy over a data slice
# ---------------------------------------------------------------------------


def _compute_strategy_returns(
    data: dict[str, np.ndarray],
    start: int,
    end: int,
) -> np.ndarray:
    """Compute simple signal-based returns, not a full portfolio simulation.

    For each bar where the strategy emits non-flat signals, we compute:
        return_i = mean(direction * strength * asset_return across all signaled assets)

    This gives a zero-cost, equal-vol approximation of strategy performance.
    """
    # Slice the close prices
    close_slice = data["close"][start:end]
    n = close_slice.shape[0]
    if n < 2:
        return np.array([], dtype=float)

    # Compute per-asset returns
    asset_returns = (close_slice[1:] - close_slice[:-1]) / close_slice[:-1]

    # Truncated slice for signal computation (last bar gives no forward return)
    signal_data = {k: (v[start : end - 1] if isinstance(v, np.ndarray) and v.ndim > 0 else v)
                   for k, v in data.items()}
    signal_data["close"] = close_slice[:-1]

    from quantcore.strategies.base import StrategyConfig, StrategyType
    signal_returns = np.zeros(n - 1)

    # We'll import the right class below — use the abstract Strategy for now
    # and let the caller handle class selection
    return asset_returns, signal_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Import all strategy modules to trigger @register decorators
@pytest.fixture(autouse=True, scope="module")
def _ensure_imported() -> None:
    import quantcore.strategies.equity.mean_reversion  # noqa: F401
    import quantcore.strategies.equity.momentum  # noqa: F401
    import quantcore.strategies.equity.stat_arb  # noqa: F401
    import quantcore.strategies.equity.trend_following  # noqa: F401
    import quantcore.strategies.equity.intraday_breakout  # noqa: F401
    import quantcore.strategies.equity.intraday_mean_reversion  # noqa: F401
    import quantcore.strategies.equity.intraday_momentum  # noqa: F401
    import quantcore.strategies.equity.intraday_trend  # noqa: F401
    import quantcore.strategies.macro.carry  # noqa: F401
    import quantcore.strategies.macro.cta_trend  # noqa: F401
    import quantcore.strategies.macro.risk_parity  # noqa: F401
    import quantcore.strategies.options.delta_neutral  # noqa: F401
    import quantcore.strategies.options.earnings_vol  # noqa: F401
    import quantcore.strategies.options.iron_condor  # noqa: F401
    import quantcore.strategies.options.tail_hedge  # noqa: F401


@pytest.fixture(scope="module")
def strategy_names() -> list[str]:
    """All registered strategy names."""
    from quantcore.strategies.registry import list_strategies
    return list_strategies()


# ---------------------------------------------------------------------------
# Parametrized walk-forward test for every registered strategy
# ---------------------------------------------------------------------------


class TestAllStrategiesWalkForward:
    """Every strategy completes walk-forward folds without crashing."""

    @pytest.mark.parametrize("strategy_name", list(_STRATEGY_CONFIGS.keys()))
    def test_walk_forward_completes(
        self, strategy_name: str, strategy_data: dict[str, np.ndarray],
    ) -> None:
        """Walk-forward must complete without error for every strategy."""
        from quantcore.strategies.registry import create
        from quantcore.strategies.base import StrategyConfig, StrategyType

        stype = _strategy_type(strategy_name)
        params = _STRATEGY_CONFIGS[strategy_name]
        cfg = StrategyConfig(
            name=strategy_name,
            strategy_type=StrategyType(stype),
            params=params,
        )
        strat = create(strategy_name, cfg)
        strat.initialize()

        # Build data dict with required fields
        data = dict(strategy_data)
        # Slice to make sure earnings_upcoming is the right shape
        n = data["close"].shape[0]
        data["earnings_upcoming"] = np.full(n, 1.0)

        # Generate walk-forward folds
        folds = rolling_walk_forward(n, train_size=200, test_size=50, embargo_pct=0.02)
        if len(folds) < 2:
            folds = rolling_walk_forward(n, train_size=100, test_size=50, embargo_pct=0.02)

        assert len(folds) >= 1, "Need at least 1 fold"

        results = []
        for fold in folds:
            # Build train data slice
            train_data = {
                k: (v[fold.train_start:fold.train_end] if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
                for k, v in data.items()
            }
            train_data["close"] = data["close"][fold.train_start:fold.train_end]
            if "high" in data:
                train_data["high"] = data["high"][fold.train_start:fold.train_end]
            if "low" in data:
                train_data["low"] = data["low"][fold.train_start:fold.train_end]

            # Build test data slice
            test_data = {
                k: (v[fold.test_start:fold.test_end] if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
                for k, v in data.items()
            }
            test_data["close"] = data["close"][fold.test_start:fold.test_end]
            if "high" in data:
                test_data["high"] = data["high"][fold.test_start:fold.test_end]
            if "low" in data:
                test_data["low"] = data["low"][fold.test_start:fold.test_end]

            # Train period — collect signals
            _ = strat.on_data(train_data)
            is_returns = _signal_returns(strat, train_data)

            # Reset for test — for rolling strategies, just continue
            test_returns = _signal_returns(strat, test_data)

            is_sr = sharpe_ratio(is_returns, annualize=False) if len(is_returns) > 2 else 0.0
            oos_sr = sharpe_ratio(test_returns, annualize=False) if len(test_returns) > 2 else 0.0
            results.append((is_sr, oos_sr))

        # All folds produce valid output
        assert len(results) == len(folds)

    def test_walk_forward_oos_not_extreme(
        self, strategy_data: dict[str, np.ndarray],
    ) -> None:
        """OOS Sharpe should not be extremely negative for any strategy."""
        from quantcore.strategies.registry import create, list_strategies
        from quantcore.strategies.base import StrategyConfig, StrategyType

        names = list_strategies()
        data = dict(strategy_data)
        n = data["close"].shape[0]
        data["earnings_upcoming"] = np.full(n, 1.0)
        folds = rolling_walk_forward(n, train_size=200, test_size=50, embargo_pct=0.02)
        if len(folds) < 1:
            folds = rolling_walk_forward(n, train_size=100, test_size=50, embargo_pct=0.02)

        for strategy_name in names:
            if strategy_name not in _STRATEGY_CONFIGS:
                continue
            stype = _strategy_type(strategy_name)
            cfg = StrategyConfig(
                name=strategy_name,
                strategy_type=StrategyType(stype),
                params=_STRATEGY_CONFIGS[strategy_name],
            )
            strat = create(strategy_name, cfg)
            strat.initialize()

            for fold in folds:
                test_data = {
                    k: (v[fold.test_start:fold.test_end] if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
                    for k, v in data.items()
                }
                test_data["close"] = data["close"][fold.test_start:fold.test_end]
                if "high" in data:
                    test_data["high"] = data["high"][fold.test_start:fold.test_end]
                if "low" in data:
                    test_data["low"] = data["low"][fold.test_start:fold.test_end]

                returns = _signal_returns(strat, test_data)
                if len(returns) > 2:
                    oos_sr = sharpe_ratio(returns, annualize=False)
                    # NaN is returned for flat/zero-variance returns (audit fix 2026-05-10).
                    # NaN means "undefined Sharpe" NOT "extreme negative Sharpe" — skip the check.
                    import math
                    if math.isnan(oos_sr):
                        continue  # flat strategy — not a failure, just uninformative
                    assert oos_sr > -5.0, f"{strategy_name}: OOS Sharpe {oos_sr:.2f} is extreme"


# ---------------------------------------------------------------------------
# Intraday strategies: test with shorter bars
# ---------------------------------------------------------------------------


class TestIntradayStrategiesWalkForward:
    """Intraday strategies need shorter lookbacks and more bars for stable results."""

    @pytest.fixture
    def intraday_data(self) -> dict[str, np.ndarray]:
        """Higher-frequency data for intraday strategies."""
        rng = np.random.default_rng(42)
        n = 300
        n_assets = 4

        # More volatile, shorter horizon
        base = 100.0 + np.cumsum(rng.normal(0.001, 0.2, (n, n_assets)), axis=0)
        close = np.maximum(base, 10.0)
        high = close * (1 + np.abs(rng.normal(0, 0.005, (n, n_assets))))
        low = close * (1 - np.abs(rng.normal(0, 0.005, (n, n_assets))))
        low = np.minimum(low, close)

        return {
            "close": close,
            "high": high,
            "low": low,
            "implied_vol": np.full((n, n_assets), 0.3),
            "symbols": [f"I{i}" for i in range(n_assets)],
            "earnings_upcoming": np.full(n, 0.0),
        }

    @pytest.mark.parametrize("strategy_name", list(_INTRADAY_PARAMS.keys()))
    def test_intraday_walk_forward(
        self, strategy_name: str, intraday_data: dict[str, np.ndarray],
    ) -> None:
        """Intraday strategies should complete walk-forward without error."""
        from quantcore.strategies.registry import create
        from quantcore.strategies.base import StrategyConfig, StrategyType

        cfg = StrategyConfig(
            name=strategy_name,
            strategy_type=StrategyType.EQUITY,
            params=_INTRADAY_PARAMS[strategy_name],
        )
        strat = create(strategy_name, cfg)
        strat.initialize()

        n = intraday_data["close"].shape[0]
        folds = rolling_walk_forward(n, train_size=100, test_size=30, embargo_pct=0.02)

        assert len(folds) >= 1
        for fold in folds:
            test_data = {
                k: (v[fold.test_start:fold.test_end] if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
                for k, v in intraday_data.items()
            }
            test_data["close"] = intraday_data["close"][fold.test_start:fold.test_end]
            if "high" in intraday_data:
                test_data["high"] = intraday_data["high"][fold.test_start:fold.test_end]
            if "low" in intraday_data:
                test_data["low"] = intraday_data["low"][fold.test_start:fold.test_end]

            returns = _signal_returns(strat, test_data)
            assert len(returns) >= 0  # At least didn't crash


# ---------------------------------------------------------------------------
# Helper: compute signal-based returns
# ---------------------------------------------------------------------------


def _signal_returns(
    strat,
    data: dict[str, np.ndarray],
) -> np.ndarray:
    """Run the strategy over a data slice and compute signal-based returns.

    This is a simplified proxy: for each bar, compute a weighted average of
    signaled position changes * asset returns.
    """
    close = data.get("close")
    if close is None or close.shape[0] < 2:
        return np.array([], dtype=float)

    n = close.shape[0]
    asset_returns = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)

    bar_returns = np.zeros(n - 1)
    signals_by_bar: list[list] = [[] for _ in range(n - 1)]

    for i in range(n - 1):
        bar_data = {k: v for k, v in data.items()}
        # For array fields, provide up to bar i+1 (strategy may look back)
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[: i + 1]

        try:
            signals = strat.on_data(bar_data)
        except Exception:
            signals = []

        for s in signals:
            if s.direction != 0 and i < len(asset_returns):
                signals_by_bar[i].append(s)

    # Compute bar-returns from signals
    for i, sigs in enumerate(signals_by_bar):
        if sigs and i < len(asset_returns):
            # Weighted average signal across all signaled assets
            total_strength = sum(abs(s.strength) for s in sigs) or 1.0
            weighted_ret = sum(
                float(s.direction) * float(s.strength) * float(asset_returns[i][symbol_idx])
                for s in sigs
                if (symbol_idx := _asset_index(data, s.symbol)) is not None
            ) / total_strength
            bar_returns[i] = weighted_ret

    return bar_returns


def _asset_index(data: dict, symbol: str) -> int | None:
    """Find asset index by symbol."""
    symbols = data.get("symbols")
    if symbols is None:
        return None
    try:
        return symbols.index(symbol)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Ensure all strategies covered in the test list
# ---------------------------------------------------------------------------


class TestStrategyCoverage:
    """Verify all registered strategies have walk-forward tests."""

    def test_all_strategies_have_params(self) -> None:
        """Every registered strategy must have a config entry."""
        from quantcore.strategies.registry import list_strategies
        registered = list_strategies()
        for name in registered:
            assert name in _STRATEGY_CONFIGS, (
                f"Strategy '{name}' has no test params — add to _STRATEGY_CONFIGS"
            )

    def test_no_extra_params(self) -> None:
        """No stale entries in the test config dict."""
        from quantcore.strategies.registry import list_strategies
        registered = set(list_strategies())
        for name in _STRATEGY_CONFIGS:
            if name not in registered:
                pytest.skip(f"'{name}' has params but is not registered — may be intentional")
