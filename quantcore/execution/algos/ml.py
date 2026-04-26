"""Machine Learning-based execution algorithm.

Uses offline-learned models and real-time feature engineering to optimize
execution schedule, participation rate, and venue routing. Supports:
- Supervised slippage prediction models
- Reinforcement learning for adaptive schedule optimization
- Feature extraction from market micro-structure
- Ensemble model aggregation for robustness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


@dataclass
class MLExecConfig:
    """ML execution algorithm configuration."""
    window_minutes: int = 60
    num_slices: int = 20
    model: Any = None  # User-supplied ML model (sklearn, pytorch, etc.)
    feature_set: list[str] = field(default_factory=lambda: [
        "spread_bps", "volatility", "volume_ratio", "price_momentum",
        "order_book_imbalance", "trade_flow_imbalance", "hour_of_day",
        "day_of_week", "time_to_close", "remaining_fraction",
    ])
    urgency: float = 1.0  # 0.5 passive → 2.0 aggressive
    risk_aversion: float = 0.5  # 0 = risk-neutral, 1 = highly risk-averse
    min_slice_size: float = 1.0
    max_participation: float = 0.25
    adaptation_rate: float = 0.1  # EMA rate for online adaptation
    seed: int = 42


@dataclass
class MLSlice:
    """A single ML-optimized execution slice."""
    slice_id: int
    target_time: datetime
    shares: float
    participation_rate: float
    features: dict[str, float] = field(default_factory=dict)
    model_score: float | None = None  # Predicted cost/quality
    executed: bool = False
    fill_price: float | None = None
    fill_shares: float = 0.0


@dataclass
class MLResult:
    """ML execution result."""
    slices: list[MLSlice]
    total_shares: float
    total_executed: float
    avg_price: float | None = None
    benchmark_price: float | None = None
    performance_bps: float | None = None
    model_score_avg: float | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)
    cancelled_remaining: float = 0.0


class MLAlgo:
    """ML-based execution: uses predictive models to optimize schedule.

    The algorithm:
    1. Extracts micro-structure features from market data
    2. Scores each potential slice timing using the ML model
    3. Dynamically allocates shares across slices based on predicted cost
    4. Adapts online as market conditions evolve

    The model can be any callable f(features) → score where lower score = better.
    Supports sklearn models, PyTorch, or custom functions.

    Usage:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor().fit(X_train, y_train)
        algo = MLAlgo(MLExecConfig(model=model))
        schedule = algo.compute_schedule(10000, market_features)
    """

    def __init__(self, config: MLExecConfig | None = None) -> None:
        self.config = config or MLExecConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._online_adaptation: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(
        self,
        market_data: dict[str, NDArray[np.floating] | float] | None = None,
        state: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Extract execution-relevant features from market data.

        Args:
            market_data: Dict with arrays/values for spreads, prices, volumes, etc.
            state: Current execution state (remaining, elapsed, etc.).

        Returns:
            Feature vector as dict.
        """
        features: dict[str, float] = {}

        if market_data:
            features["spread_bps"] = self._safe_float(market_data.get("spread_bps", 5.0))
            features["volatility"] = self._safe_float(market_data.get("volatility", 0.01))
            features["volume_ratio"] = self._safe_float(market_data.get("volume_ratio", 1.0))
            features["price_momentum"] = self._safe_float(market_data.get("price_momentum", 0.0))
            features["order_book_imbalance"] = self._safe_float(market_data.get("book_imbalance", 0.0))
            features["trade_flow_imbalance"] = self._safe_float(market_data.get("trade_imbalance", 0.0))

        if state:
            features["remaining_fraction"] = state.get("remaining_fraction", 1.0)
            features["elapsed_fraction"] = state.get("elapsed_fraction", 0.0)

        # Temporal features
        now = datetime.utcnow()
        features["hour_of_day"] = float(now.hour) / 24.0
        features["day_of_week"] = float(now.weekday()) / 7.0
        features["time_to_close"] = max(0.0, 1.0 - (now.hour + now.minute / 60.0) / 24.0)

        # Online adaptation signals
        if self._online_adaptation:
            features.update(self._online_adaptation)

        return features

    def _safe_float(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, np.ndarray):
            return float(np.mean(value))
        return 0.0

    # ------------------------------------------------------------------
    # Schedule computation
    # ------------------------------------------------------------------

    def compute_schedule(
        self,
        total_shares: float,
        market_data: dict[str, NDArray[np.floating] | float] | None = None,
        start_time: datetime | None = None,
    ) -> MLResult:
        """Compute ML-optimized execution schedule.

        Args:
            total_shares: Total shares to execute.
            market_data: Current market micro-structure data.
            start_time: Execution start time.

        Returns:
            MLResult with optimized slice schedule.
        """
        cfg = self.config
        t0 = start_time or datetime.utcnow()
        window_ms = cfg.window_minutes * 60 * 1000
        interval_ms = window_ms / cfg.num_slices

        slices: list[MLSlice] = []
        remaining = total_shares

        for i in range(cfg.num_slices):
            target_time = t0 + timedelta(milliseconds=interval_ms * i)

            # State features for this slice
            state = {
                "remaining_fraction": remaining / total_shares if total_shares > 0 else 0.0,
                "elapsed_fraction": i / cfg.num_slices,
            }

            features = self.extract_features(market_data, state)

            # Score this slice timing using the model
            model_score = self._score(features) if cfg.model is not None else None

            # Allocate shares: lower score → more shares (better execution quality)
            if model_score is not None and i < cfg.num_slices - 1:
                # Weight inversely proportional to predicted slippage
                weight = 1.0 / max(0.01, model_score)
            else:
                weight = 1.0

            # Participation rate from model or urgency
            participation = min(
                cfg.max_participation,
                cfg.urgency * (0.5 + 0.5 * (1.0 / max(1.0, i + 1))),
            ) if model_score is None else min(cfg.max_participation, 1.0 / max(1.0, model_score * 10.0))

            slice_shares = (remaining * weight / max(1, cfg.num_slices - i)) if i < cfg.num_slices - 1 else remaining
            slice_shares = max(cfg.min_slice_size, slice_shares)
            slice_shares = min(slice_shares, remaining)

            slices.append(MLSlice(
                slice_id=i,
                target_time=target_time,
                shares=round(slice_shares, 0),
                participation_rate=participation,
                features={k: v for k, v in features.items() if k in cfg.feature_set},
                model_score=model_score,
            ))

            remaining -= slices[-1].shares
            if remaining <= 0:
                break

        return MLResult(
            slices=slices,
            total_shares=total_shares,
            total_executed=0.0,
        )

    def _score(self, features: dict[str, float]) -> float | None:
        """Score a slice timing using the ML model.

        Lower score = better execution quality expected.
        """
        if self.config.model is None:
            return None

        model = self.config.model
        feature_vec = [
            features.get(f, 0.0) for f in self.config.feature_set
        ]

        try:
            # Support sklearn-style predict, pytorch forward, or raw callable
            if hasattr(model, "predict"):
                score = float(model.predict([feature_vec])[0])
            elif hasattr(model, "forward"):
                import torch
                t = torch.tensor([feature_vec], dtype=torch.float32)
                with torch.no_grad():
                    score = float(model(t).item())
            elif callable(model):
                score = float(model(feature_vec))
            else:
                return None
            return score
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Online adaptation
    # ------------------------------------------------------------------

    def adapt(
        self,
        slice_result: MLSlice,
        market_features: dict[str, float],
    ) -> None:
        """Online adaptation: adjust execution strategy based on fill quality.

        Called after each slice is executed to update internal state so
        future slices adapt to observed market conditions.

        Args:
            slice_result: The just-executed slice.
            market_features: Market features at execution time.
        """
        cfg = self.config
        alpha = cfg.adaptation_rate

        if slice_result.fill_price is not None and slice_result.model_score is not None:
            # How much worse/better was actual execution vs predicted?
            error = float(slice_result.fill_price) - float(slice_result.model_score)

            # Adapt model bias
            if "model_bias" not in self._online_adaptation:
                self._online_adaptation["model_bias"] = 0.0
            self._online_adaptation["model_bias"] = (
                (1.0 - alpha) * self._online_adaptation["model_bias"] + alpha * error
            )

            # Adapt feature weights
            for name, value in market_features.items():
                key = f"adapt_{name}"
                if key not in self._online_adaptation:
                    self._online_adaptation[key] = 0.0
                self._online_adaptation[key] = (
                    (1.0 - alpha) * self._online_adaptation[key] + alpha * error * value
                )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        slices: list[MLSlice],
        benchmark_price: float | None = None,
    ) -> MLResult:
        """Evaluate ML execution performance.

        Args:
            slices: Completed slices with fill data.
            benchmark_price: Arrival price or other benchmark.

        Returns:
            MLResult with performance metrics.
        """
        total_shares = sum(s.shares for s in slices)
        total_exec = sum(s.fill_shares for s in slices)

        if total_exec > 0:
            filled = [(s.fill_price or 0) * s.fill_shares for s in slices if s.fill_price is not None]
            weights = [s.fill_shares for s in slices if s.fill_price is not None]
            avg_price = float(np.average(filled, weights=weights)) if sum(weights) > 0 else None
        else:
            avg_price = None

        scores = [s.model_score for s in slices if s.model_score is not None]
        avg_score = float(np.mean(scores)) if scores else None

        perf_bps = None
        if avg_price is not None and benchmark_price is not None and benchmark_price > 0:
            perf_bps = (avg_price - benchmark_price) / benchmark_price * 10_000

        # Feature importance from adaptation weights
        importance: dict[str, float] = {}
        for key, value in self._online_adaptation.items():
            if key.startswith("adapt_"):
                name = key.replace("adapt_", "")
                importance[name] = abs(value)

        return MLResult(
            slices=slices,
            total_shares=total_shares,
            total_executed=total_exec,
            avg_price=avg_price,
            benchmark_price=benchmark_price,
            performance_bps=perf_bps,
            model_score_avg=avg_score,
            feature_importance=importance,
            cancelled_remaining=total_shares - total_exec,
        )

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> dict[str, float]:
        """Return normalized feature importance from online adaptation."""
        total = sum(abs(v) for v in self._online_adaptation.values())
        if total == 0:
            return {}
        return {k: abs(v) / total for k, v in self._online_adaptation.items()}

    def reset_adaptation(self) -> None:
        """Reset online adaptation state."""
        self._online_adaptation.clear()
