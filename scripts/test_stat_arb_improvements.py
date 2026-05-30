"""Test script for new statistical arbitrage improvements.

Validates that all new modules work correctly and can be integrated
into the existing quantcore system.

Usage:
    python scripts/test_stat_arb_improvements.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_ml_factors():
    """Test ML factor combination."""
    print("\n" + "=" * 60)
    print("Testing ML Factor Combination")
    print("=" * 60)
    
    from quantcore.signals.statistical.ml_factors import (
        GradientBoostingFactorCombiner,
        RandomForestFactorCombiner,
        EnsembleFactorCombiner,
        AdaptiveFactorCombiner,
        evaluate_factor_combiner,
    )
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_factors = 5
    
    X = np.random.randn(n_samples, n_factors)
    # True signal: non-linear combination
    y = 0.3 * X[:, 0] + 0.2 * X[:, 1]**2 - 0.1 * X[:, 2] * X[:, 3] + 0.05 * np.random.randn(n_samples)
    
    # Split into train/test
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Test GradientBoosting
    print("\n1. GradientBoostingFactorCombiner")
    gb = GradientBoostingFactorCombiner(n_estimators=50)
    gb.fit(X_train, y_train)
    result = gb.predict(X_test)
    print(f"   Signal range: [{result.signal.min():.3f}, {result.signal.max():.3f}]")
    print(f"   Feature importance: {result.feature_importance}")
    
    # Test RandomForest
    print("\n2. RandomForestFactorCombiner")
    rf = RandomForestFactorCombiner(n_estimators=50)
    rf.fit(X_train, y_train)
    result = rf.predict(X_test)
    print(f"   Signal range: [{result.signal.min():.3f}, {result.signal.max():.3f}]")
    print(f"   Uncertainty: {result.predictions_std:.3f}")
    
    # Test Ensemble
    print("\n3. EnsembleFactorCombiner")
    ensemble = EnsembleFactorCombiner(models=["gb", "rf"])
    ensemble.fit(X_train, y_train)
    result = ensemble.predict(X_test)
    print(f"   Signal range: [{result.signal.min():.3f}, {result.signal.max():.3f}]")
    
    # Test Adaptive
    print("\n4. AdaptiveFactorCombiner")
    adaptive = AdaptiveFactorCombiner(lookback=100)
    for i in range(200):
        adaptive.update(X_train[i], y_train[i])
    signal = adaptive.combine(X_test)
    print(f"   Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    print(f"   Current weights: {adaptive.current_weights}")
    
    # Evaluate
    print("\n5. Evaluation")
    metrics = evaluate_factor_combiner(gb, X_test, y_test, n_splits=3)
    print(f"   Mean IC: {metrics['mean_ic']:.3f}")
    print(f"   ICIR: {metrics['icir']:.3f}")
    
    print("\n[OK] ML Factor Combination: PASSED")


def test_optimal_execution():
    """Test optimal execution algorithms."""
    print("\n" + "=" * 60)
    print("Testing Optimal Execution")
    print("=" * 60)
    
    from quantcore.execution.optimal_execution import (
        AlmgrenChrissOptimizer,
        VWAPExecutor,
        TWAPExecutor,
        TransactionCostAnalyzer,
        UrgencyLevel,
    )
    
    # Test Almgren-Chriss
    print("\n1. AlmgrenChrissOptimizer")
    ac = AlmgrenChrissOptimizer(sigma=0.20, risk_aversion=1e-6)
    schedule = ac.compute_optimal_trajectory(total_quantity=10000, n_periods=10)
    print(f"   Total quantity: {sum(schedule.quantities):.0f}")
    print(f"   Market impact: {schedule.market_impact:.2f}")
    print(f"   Timing risk: {schedule.timing_risk:.2f}")
    
    # Test with urgency
    schedule_urgent = ac.compute_trajectory_with_urgency(
        total_quantity=10000,
        urgency=UrgencyLevel.HIGH,
        n_periods=10,
    )
    print(f"   Urgent trajectory first slice: {schedule_urgent.quantities[0]:.0f}")
    
    # Test VWAP
    print("\n2. VWAPExecutor")
    vwap = VWAPExecutor(participation_rate=0.10)
    slices = vwap.generate_schedule(
        total_quantity=10000,
        total_volume=1000000,
        n_periods=390,
    )
    print(f"   Number of slices: {len(slices)}")
    print(f"   First slice quantity: {slices[0].quantity:.0f}")
    
    # Test TWAP
    print("\n3. TWAPExecutor")
    twap = TWAPExecutor(n_slices=10)
    slices = twap.generate_schedule(
        total_quantity=10000,
        start_time=pd.Timestamp.now(),
        end_time=pd.Timestamp.now() + pd.Timedelta(hours=6),
    )
    print(f"   Number of slices: {len(slices)}")
    print(f"   Average slice: {np.mean([s.quantity for s in slices]):.0f}")
    
    # Test TCA
    print("\n4. TransactionCostAnalyzer")
    tca = TransactionCostAnalyzer()
    cost = tca.pre_trade_cost(
        quantity=10000,
        adv=1000000,
        price=100.0,
        urgency=UrgencyLevel.MEDIUM,
    )
    print(f"   Total cost: ${cost.total_cost:.2f}")
    print(f"   Cost (bps): {cost.cost_bps:.1f}")
    
    print("\n[OK] Optimal Execution: PASSED")


def test_advanced_risk():
    """Test advanced risk models."""
    print("\n" + "=" * 60)
    print("Testing Advanced Risk Models")
    print("=" * 60)
    
    from quantcore.risk.advanced_risk import (
        PCARiskModel,
        TailRiskModel,
        RiskDecomposer,
        compute_portfolio_var,
        compute_portfolio_es,
    )
    
    # Generate synthetic returns
    np.random.seed(42)
    n_periods = 1000
    n_assets = 10
    
    # Create correlated returns
    factors = np.random.randn(n_periods, 3)
    loadings = np.random.randn(3, n_assets)
    returns = factors @ loadings + 0.1 * np.random.randn(n_periods, n_assets)
    
    # Test PCA
    print("\n1. PCARiskModel")
    pca = PCARiskModel(n_components=3)
    pca.fit(returns)
    loadings_df = pca.get_factor_loadings()
    print(f"   Explained variance: {pca._pca_result.explained_variance_ratio}")
    print(f"   Loadings shape: {loadings_df.shape}")
    
    # Test Tail Risk
    print("\n2. TailRiskModel")
    tail = TailRiskModel()
    tail.fit(returns[:, 0])
    var_95 = tail.compute_var(0.95)
    var_99 = tail.compute_var(0.99)
    es_99 = tail.compute_es(0.99)
    print(f"   VaR 95%: {var_95:.4f}")
    print(f"   VaR 99%: {var_99:.4f}")
    print(f"   ES 99%: {es_99:.4f}")
    
    # Test Risk Decomposition
    print("\n3. RiskDecomposer")
    decomposer = RiskDecomposer(n_factors=3)
    weights = np.ones(n_assets) / n_assets
    decomp = decomposer.decompose(weights, returns)
    print(f"   Total risk: {decomp.total_risk:.4f}")
    print(f"   Systematic risk: {decomp.systematic_risk:.4f}")
    print(f"   Idiosyncratic risk: {decomp.idiosyncratic_risk:.4f}")
    
    # Test Portfolio VaR/ES
    print("\n4. Portfolio VaR/ES")
    var = compute_portfolio_var(weights, returns, confidence=0.99, method="parametric")
    es = compute_portfolio_es(weights, returns, confidence=0.99, method="historical")
    print(f"   Portfolio VaR 99%: {var:.4f}")
    print(f"   Portfolio ES 99%: {es:.4f}")
    
    print("\n[OK] Advanced Risk Models: PASSED")


def test_alternative_data():
    """Test alternative data signals."""
    print("\n" + "=" * 60)
    print("Testing Alternative Data Signals")
    print("=" * 60)
    
    from quantcore.signals.alternative_data import (
        NewsSentimentAnalyzer,
        OptionsFlowAnalyzer,
        AlternativeDataCombiner,
    )
    from datetime import datetime, timedelta
    
    # Test News Sentiment
    print("\n1. NewsSentimentAnalyzer")
    news = NewsSentimentAnalyzer()
    headlines = [
        "Company beats earnings expectations",
        "Stock surges on strong growth",
        "Analysts upgrade rating",
        "Revenue exceeds forecast",
    ]
    timestamps = [datetime.now() - timedelta(days=i) for i in range(4)]
    sentiment = news.analyze_headlines(headlines, timestamps)
    print(f"   Sentiment score: {sentiment.score:.3f}")
    print(f"   Confidence: {sentiment.confidence:.3f}")
    print(f"   Volume: {sentiment.volume}")
    
    # Test Options Flow
    print("\n2. OptionsFlowAnalyzer")
    options = OptionsFlowAnalyzer()
    
    # Synthetic data
    put_volume = np.random.randint(1000, 5000, 20)
    call_volume = np.random.randint(1000, 5000, 20)
    put_oi = np.random.randint(10000, 50000, 20)
    call_oi = np.random.randint(10000, 50000, 20)
    
    put_call_signal = options.compute_put_call_signal(
        put_volume, call_volume, put_oi, call_oi,
    )
    print(f"   Put/Call signal: {put_call_signal:.3f}")
    
    # Test Alternative Data Combiner
    print("\n3. AlternativeDataCombiner")
    combiner = AlternativeDataCombiner()
    combiner.update_signal("news", datetime.now(), sentiment.score)
    combiner.update_signal("options", datetime.now(), put_call_signal)
    
    current_signals = {
        "news": sentiment.score,
        "options": put_call_signal,
    }
    combined = combiner.compute_combined_signal(current_signals, datetime.now())
    print(f"   Combined signal: {combined:.3f}")
    
    print("\n[OK] Alternative Data Signals: PASSED")


def test_advanced_portfolio():
    """Test advanced portfolio construction."""
    print("\n" + "=" * 60)
    print("Testing Advanced Portfolio Construction")
    print("=" * 60)
    
    from quantcore.portfolio.advanced_portfolio import (
        BlackLittermanOptimizer,
        RiskParityOptimizer,
        HierarchicalRiskParity,
        MeanCVaROptimizer,
        compute_portfolio_metrics,
    )
    
    # Generate synthetic data
    np.random.seed(42)
    n_periods = 1000
    n_assets = 10
    
    # Create correlated returns
    factors = np.random.randn(n_periods, 3)
    loadings = np.random.randn(3, n_assets)
    returns = factors @ loadings + 0.1 * np.random.randn(n_periods, n_assets)
    covariance = np.cov(returns.T)
    
    # Test Black-Litterman
    print("\n1. BlackLittermanOptimizer")
    bl = BlackLittermanOptimizer(risk_aversion=2.5)
    market_weights = np.ones(n_assets) / n_assets
    alpha_signals = np.random.randn(n_assets) * 0.01
    result = bl.optimize(market_weights, covariance, alpha_signals=alpha_signals)
    print(f"   Weights: {result.weights[:3]}... (first 3)")
    print(f"   Expected return: {result.expected_return:.4f}")
    print(f"   Sharpe ratio: {result.sharpe_ratio:.2f}")
    
    # Test Risk Parity
    print("\n2. RiskParityOptimizer")
    rp = RiskParityOptimizer()
    result = rp.optimize(covariance)
    print(f"   Weights: {result.weights[:3]}... (first 3)")
    print(f"   Risk: {result.expected_risk:.4f}")
    
    # Test HRP
    print("\n3. HierarchicalRiskParity")
    hrp = HierarchicalRiskParity()
    result = hrp.optimize(returns)
    print(f"   Weights: {result.weights[:3]}... (first 3)")
    print(f"   Risk: {result.expected_risk:.4f}")
    
    # Test Mean-CVaR
    print("\n4. MeanCVaROptimizer")
    cvar = MeanCVaROptimizer(confidence=0.95)
    result = cvar.optimize(returns)
    print(f"   Weights: {result.weights[:3]}... (first 3)")
    print(f"   Sharpe ratio: {result.sharpe_ratio:.2f}")
    
    # Test Portfolio Metrics
    print("\n5. Portfolio Metrics")
    weights = np.ones(n_assets) / n_assets
    metrics = compute_portfolio_metrics(returns, weights)
    print(f"   Ann. Return: {metrics['ann_return']:.2%}")
    print(f"   Ann. Vol: {metrics['ann_vol']:.2%}")
    print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max DD: {metrics['max_drawdown']:.2%}")
    
    print("\n[OK] Advanced Portfolio Construction: PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("STATISTICAL ARBITRAGE IMPROVEMENTS TEST SUITE")
    print("=" * 60)
    
    try:
        test_ml_factors()
        test_optimal_execution()
        test_advanced_risk()
        test_alternative_data()
        test_advanced_portfolio()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED [OK]")
        print("=" * 60)
        
        print("\nNew modules are ready for integration!")
        print("\nNext steps:")
        print("1. Review docs/STAT_ARB_IMPROVEMENTS.md")
        print("2. Integrate ML factors into existing strategies")
        print("3. Add optimal execution to order routing")
        print("4. Deploy risk monitoring")
        print("5. Add alternative data feeds")
        
    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
