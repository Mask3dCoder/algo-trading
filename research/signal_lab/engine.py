"""Signal Analytics Engine v2 — Institutional-Grade Research Platform.

This is the core evaluation tool for Phase 3 alpha discovery.
A signal passes through here BEFORE any backtest, sizing, or execution code is touched.

Usage:
    from research.signal_lab.engine import evaluate_signal, DatasetValidator, SignalExtractor
    
    # 1. Validate dataset quality
    report = DatasetValidator.validate(df)
    
    # 2. Extract the raw signal  
    signal = SignalExtractor.cross_sectional_momentum(df)
    
    # 3. Evaluate predictive power
    result = evaluate_signal(signal, df["forward_returns_1"])
    result.display()
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from quantcore.signals.ml.evaluation import (
    ic_decay,
    information_coefficient,
    quintile_analysis,
    signal_turnover,
)


# ---------------------------------------------------------------------------
# Dataset Validation
# ---------------------------------------------------------------------------

@dataclass
class DatasetValidationReport:
    """Report from validating a research dataset."""
    total_rows: int
    total_symbols: int
    date_range: tuple[str, str]
    missing_pct: float
    has_forward_returns: bool
    has_volume: bool
    has_benchmark: bool
    lookahead_risk: bool  # True = risk detected
    issues: list[str] = field(default_factory=list)
    passed: bool = True

    def display(self):
        status = "PASSED ✓" if self.passed else "FAILED ✗"
        print(f"\n=== Dataset Validation Report [{status}] ===")
        print(f"  Rows:            {self.total_rows:,}")
        print(f"  Symbols:         {self.total_symbols}")
        print(f"  Date Range:      {self.date_range[0]} → {self.date_range[1]}")
        print(f"  Missing Data:    {self.missing_pct:.1%}")
        print(f"  Forward Returns: {'Yes' if self.has_forward_returns else 'No'}")
        print(f"  Volume:          {'Yes' if self.has_volume else 'No'}")
        print(f"  Benchmark:       {'Yes' if self.has_benchmark else 'No'}")
        print(f"  Lookahead Risk:  {'DETECTED ⚠' if self.lookahead_risk else 'None'}")
        if self.issues:
            print("  Issues:")
            for issue in self.issues:
                print(f"    • {issue}")


class DatasetValidator:
    """Validates research datasets before any signal evaluation."""

    REQUIRED_COLUMNS = {"close", "forward_returns_1"}
    RECOMMENDED_COLUMNS = {"volume", "benchmark_returns", "forward_returns_5"}

    @classmethod
    def validate(cls, df: pd.DataFrame) -> DatasetValidationReport:
        issues: list[str] = []
        passed = True

        # Required columns
        missing_required = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing_required:
            issues.append(f"Missing required columns: {missing_required}")
            passed = False

        # Missing data percentage
        missing_pct = float(df.isnull().mean().mean())
        if missing_pct > 0.05:
            issues.append(f"High missing data rate: {missing_pct:.1%}")

        # Forward returns look-ahead check: ensure they are shifted correctly
        lookahead_risk = False
        if "close" in df.columns and "forward_returns_1" in df.columns:
            # If forward returns correlate with CURRENT close changes at lag 0 suspiciously high
            close_ret = df["close"].pct_change().dropna()
            fwd = df["forward_returns_1"].dropna()
            aligned = pd.concat([close_ret, fwd], axis=1).dropna()
            if len(aligned) > 20:
                corr = float(aligned.corr().iloc[0, 1])
                if abs(corr) > 0.85:
                    lookahead_risk = True
                    issues.append(
                        f"Potential look-ahead bias: forward returns correlate {corr:.2f} with current returns."
                    )

        # Date range
        if isinstance(df.index, pd.DatetimeIndex):
            date_range = (str(df.index.min().date()), str(df.index.max().date()))
        else:
            date_range = ("unknown", "unknown")

        # Symbol count
        total_symbols = int(df["symbol"].nunique()) if "symbol" in df.columns else 1

        return DatasetValidationReport(
            total_rows=len(df),
            total_symbols=total_symbols,
            date_range=date_range,
            missing_pct=missing_pct,
            has_forward_returns="forward_returns_1" in df.columns,
            has_volume="volume" in df.columns,
            has_benchmark="benchmark_returns" in df.columns,
            lookahead_risk=lookahead_risk,
            issues=issues,
            passed=passed,
        )


# ---------------------------------------------------------------------------
# Raw Signal Extractors (pure — no execution, no sizing)
# ---------------------------------------------------------------------------

class SignalExtractor:
    """Pure signal extraction from price data.

    All methods return a pd.Series aligned to the input DataFrame index.
    No execution, sizing, or portfolio logic is included here.
    """

    @staticmethod
    def cross_sectional_momentum(
        df: pd.DataFrame,
        lookback: int = 252,
        skip_recent: int = 21,
    ) -> pd.Series:
        """12-1 month cross-sectional momentum (Jegadeesh-Titman).
        
        Signal = rank of trailing return from T-lookback to T-skip_recent.
        Negative correlation with 1-month returns (short-term reversal skipped).
        """
        close = df["close"]
        # 12-1 month: exclude most recent month to avoid reversal contamination
        total_return = close.shift(skip_recent) / close.shift(lookback) - 1.0
        # Cross-sectionally standardize (rank-normalize within each date)
        if "symbol" in df.columns:
            signal = df.groupby(df.index)["close"].transform(
                lambda x: pd.Series(total_return.loc[x.index]).rank(pct=True) - 0.5
            )
        else:
            signal = (total_return - total_return.rolling(252).mean()) / total_return.rolling(252).std()
        return signal.rename("cs_momentum")

    @staticmethod
    def time_series_momentum(
        df: pd.DataFrame,
        lookback: int = 252,
    ) -> pd.Series:
        """Time-series momentum: sign of 12-month trailing return.
        
        Signal = +1 if asset trended up, -1 if trended down.
        Vol-scaled to control for heteroskedasticity.
        """
        close = df["close"]
        ret_12m = close / close.shift(lookback) - 1.0
        daily_ret = close.pct_change()
        ann_vol = daily_ret.rolling(60).std() * np.sqrt(252)
        ann_vol = ann_vol.clip(lower=0.01)
        # Vol-scaled signal: direction / vol
        signal = np.sign(ret_12m) / ann_vol
        return signal.rename("ts_momentum")

    @staticmethod
    def short_term_mean_reversion(
        df: pd.DataFrame,
        zscore_window: int = 20,
    ) -> pd.Series:
        """Short-term reversal: z-score of price vs. rolling mean.
        
        Signal is negatively correlated with next-day return (buy the dip).
        Negative signal = overbought (expect reversal down).
        """
        close = df["close"]
        rolling_mean = close.rolling(zscore_window).mean()
        rolling_std = close.rolling(zscore_window).std()
        zscore = (close - rolling_mean) / rolling_std.clip(lower=1e-8)
        # Negate: high z-score → expect negative returns (reversal)
        signal = -zscore
        return signal.rename("mean_reversion")

    @staticmethod
    def volatility_risk_premium(
        df: pd.DataFrame,
        iv_col: str = "implied_vol",
        hv_window: int = 21,
    ) -> pd.Series:
        """Volatility Risk Premium: IV minus realized HV.
        
        Positive VRP (IV > HV) → sellers receive compensation. 
        Signal for selling options. Missing IV data returns NaN.
        """
        close = df["close"]
        daily_ret = close.pct_change()
        realized_hv = daily_ret.rolling(hv_window).std() * np.sqrt(252)

        if iv_col in df.columns:
            iv = df[iv_col]
        else:
            warnings.warn(
                f"Column '{iv_col}' not found. VRP signal requires options IV data. "
                "Returning NaN series. Obtain CBOE VIX or options chain IV data.",
                UserWarning,
                stacklevel=2,
            )
            return pd.Series(np.nan, index=df.index, name="vrp")

        vrp = iv - realized_hv
        return vrp.rename("vrp")

    @staticmethod
    def earnings_drift(
        df: pd.DataFrame,
        event_col: str = "earnings_surprise",
        drift_window: int = 60,
    ) -> pd.Series:
        """Post-Earnings Announcement Drift (PEAD).
        
        Signal = earnings surprise × decay function.
        Requires earnings_surprise column (SUE: standardized unexpected earnings).
        """
        if event_col not in df.columns:
            warnings.warn(
                f"Column '{event_col}' not found. PEAD requires earnings surprise data.",
                UserWarning,
                stacklevel=2,
            )
            return pd.Series(np.nan, index=df.index, name="earnings_drift")

        surprise = df[event_col].fillna(0.0)
        # Decay the signal over the drift window
        signal = surprise.ewm(span=drift_window).mean()
        return signal.rename("earnings_drift")


# ---------------------------------------------------------------------------
# Signal Evaluation Report
# ---------------------------------------------------------------------------

@dataclass
class SignalReport:
    """Complete statistical evaluation of a raw signal."""
    signal_name: str
    n_observations: int
    ic_pearson: float
    ic_spearman: float
    ic_tstat: float
    ic_pvalue: float
    ic_stderr: float
    rolling_ic_mean: float
    rolling_ic_std: float
    ic_decay_1d: float
    ic_decay_5d: float
    ic_decay_10d: float
    ic_decay_20d: float
    quintile_spread_annualized: float
    is_monotonic: bool
    turnover_mean: float
    hit_rate: float
    coverage: float
    bull_ic: float
    bear_ic: float
    high_vol_ic: float
    low_vol_ic: float
    recommendation: str  # PROMOTE / REFINE / ARCHIVE / REJECT
    fail_reason: Optional[str] = None

    # Promotion gate thresholds
    IC_MIN_THRESHOLD = 0.02
    PVALUE_MAX_THRESHOLD = 0.10

    def passed_gates(self) -> bool:
        return (
            self.ic_pearson > self.IC_MIN_THRESHOLD
            and self.ic_spearman > self.IC_MIN_THRESHOLD
            and self.ic_pvalue < self.PVALUE_MAX_THRESHOLD
            and self.is_monotonic
        )

    def display(self):
        gate = "✓ PASSED" if self.passed_gates() else "✗ FAILED"
        print(f"\n{'='*60}")
        print(f"  Signal: {self.signal_name}  [{self.recommendation}]")
        print(f"{'='*60}")
        print(f"  Gate 1 Check:        {gate}")
        print(f"  N Observations:      {self.n_observations:,}")
        print(f"  Coverage:            {self.coverage:.1%}")
        print(f"  Pearson IC:          {self.ic_pearson:.4f}  (t={self.ic_tstat:.2f}, p={self.ic_pvalue:.4f})")
        print(f"  Spearman Rank IC:    {self.ic_spearman:.4f}")
        print(f"  IC Std Error:        {self.ic_stderr:.4f}")
        print(f"  Rolling IC Mean:     {self.rolling_ic_mean:.4f}  (σ={self.rolling_ic_std:.4f})")
        print(f"  IC Decay @ 5d:       {self.ic_decay_5d:.4f}")
        print(f"  IC Decay @ 20d:      {self.ic_decay_20d:.4f}")
        print(f"  Quintile Spread Ann: {self.quintile_spread_annualized:.2%}")
        print(f"  Monotonic Quintiles: {'Yes' if self.is_monotonic else 'No'}")
        print(f"  Daily Turnover:      {self.turnover_mean:.2%}")
        print(f"  Hit Rate:            {self.hit_rate:.1%}")
        print(f"  Bull IC:             {self.bull_ic:.4f}")
        print(f"  Bear IC:             {self.bear_ic:.4f}")
        print(f"  High Vol IC:         {self.high_vol_ic:.4f}")
        print(f"  Low Vol IC:          {self.low_vol_ic:.4f}")
        if self.fail_reason:
            print(f"  Fail Reason:         {self.fail_reason}")


# ---------------------------------------------------------------------------
# Core Evaluation Function
# ---------------------------------------------------------------------------

def evaluate_signal(
    signal: pd.Series,
    forward_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    signal_name: str = "unnamed",
    rolling_window: int = 126,
) -> SignalReport:
    """Evaluate a raw signal with full institutional analytics.

    Args:
        signal: Raw signal values (higher = more bullish).
        forward_returns: Aligned forward returns (T+1 by default).
        benchmark_returns: Optional benchmark for regime analysis.
        signal_name: Human-readable name for the signal.
        rolling_window: Window for rolling IC calculation.

    Returns:
        SignalReport with complete predictive statistics.
    """
    # Align and drop NaN
    df = pd.concat([signal.rename("s"), forward_returns.rename("r")], axis=1).dropna()
    s_arr = df["s"].values
    r_arr = df["r"].values
    n = len(s_arr)

    if n < 50:
        return SignalReport(
            signal_name=signal_name, n_observations=n,
            ic_pearson=0.0, ic_spearman=0.0, ic_tstat=0.0, ic_pvalue=1.0,
            ic_stderr=0.0, rolling_ic_mean=0.0, rolling_ic_std=0.0,
            ic_decay_1d=0.0, ic_decay_5d=0.0, ic_decay_10d=0.0, ic_decay_20d=0.0,
            quintile_spread_annualized=0.0, is_monotonic=False,
            turnover_mean=0.0, hit_rate=0.0, coverage=0.0,
            bull_ic=0.0, bear_ic=0.0, high_vol_ic=0.0, low_vol_ic=0.0,
            recommendation="REJECT",
            fail_reason=f"Insufficient data: {n} observations (need ≥ 50).",
        )

    # Coverage
    total_signal = len(signal)
    coverage = n / total_signal if total_signal > 0 else 0.0

    # Core IC metrics
    ic_p = information_coefficient(s_arr, r_arr, method="pearson")
    ic_s = information_coefficient(s_arr, r_arr, method="spearman")

    # IC t-stat and p-value (Fisher z-transform or direct t-test)
    ic_stderr = 1.0 / np.sqrt(n)
    ic_tstat = ic_p / ic_stderr if ic_stderr > 0 else 0.0
    ic_pvalue = float(2 * (1 - stats.t.cdf(abs(ic_tstat), df=n - 2)))

    # Rolling IC
    from quantcore.signals.ml.evaluation import rolling_ic as _rolling_ic
    roll_ic_arr = _rolling_ic(s_arr, r_arr, window=rolling_window)
    valid_roll = roll_ic_arr[~np.isnan(roll_ic_arr)]
    rolling_ic_mean = float(np.mean(valid_roll)) if len(valid_roll) > 0 else 0.0
    rolling_ic_std = float(np.std(valid_roll)) if len(valid_roll) > 0 else 0.0

    # IC Decay curve
    decay_arr = ic_decay(s_arr, r_arr, max_lag=20)
    ic_d1 = float(decay_arr[0]) if len(decay_arr) > 0 else 0.0
    ic_d5 = float(decay_arr[4]) if len(decay_arr) > 4 else 0.0
    ic_d10 = float(decay_arr[9]) if len(decay_arr) > 9 else 0.0
    ic_d20 = float(decay_arr[19]) if len(decay_arr) > 19 else 0.0

    # Quintile analysis
    q_result = quintile_analysis(s_arr, r_arr, n_quantiles=5)
    spread_ann = float(q_result.get("spread_annualized", 0.0))
    is_monotonic = bool(q_result.get("is_monotonic", False))

    # Turnover
    turnover_arr = signal_turnover(s_arr)
    valid_to = turnover_arr[~np.isnan(turnover_arr)]
    turnover_mean = float(np.mean(valid_to)) if len(valid_to) > 0 else 0.0

    # Hit rate: fraction of times sign(signal) == sign(return)
    same_sign = np.sign(s_arr) == np.sign(r_arr)
    hit_rate = float(np.mean(same_sign))

    # Regime analysis using benchmark
    bull_ic = bear_ic = high_vol_ic = low_vol_ic = float("nan")
    if benchmark_returns is not None:
        bm = benchmark_returns.reindex(df.index).dropna()
        common_idx = df.index.intersection(bm.index)
        if len(common_idx) > 40:
            bm_arr = bm.loc[common_idx].values
            s_reg = df.loc[common_idx, "s"].values
            r_reg = df.loc[common_idx, "r"].values
            bm_roll = pd.Series(bm_arr).rolling(126).mean().values
            bull_mask = bm_arr > 0
            bear_mask = bm_arr < 0
            if np.sum(bull_mask) > 20:
                bull_ic = information_coefficient(s_reg[bull_mask], r_reg[bull_mask])
            if np.sum(bear_mask) > 20:
                bear_ic = information_coefficient(s_reg[bear_mask], r_reg[bear_mask])
            # Vol regimes
            bm_vol = pd.Series(bm_arr).rolling(21).std().values
            vol_median = np.nanmedian(bm_vol)
            high_vol_mask = bm_vol > vol_median
            low_vol_mask = bm_vol <= vol_median
            if np.sum(high_vol_mask & ~np.isnan(bm_vol)) > 20:
                high_vol_ic = information_coefficient(
                    s_reg[high_vol_mask & ~np.isnan(bm_vol)],
                    r_reg[high_vol_mask & ~np.isnan(bm_vol)],
                )
            if np.sum(low_vol_mask & ~np.isnan(bm_vol)) > 20:
                low_vol_ic = information_coefficient(
                    s_reg[low_vol_mask & ~np.isnan(bm_vol)],
                    r_reg[low_vol_mask & ~np.isnan(bm_vol)],
                )

    # Promotion recommendation
    fail_reason = None
    if ic_p < SignalReport.IC_MIN_THRESHOLD:
        recommendation = "REJECT"
        fail_reason = f"Pearson IC {ic_p:.4f} < {SignalReport.IC_MIN_THRESHOLD}"
    elif ic_pvalue >= SignalReport.PVALUE_MAX_THRESHOLD:
        recommendation = "REJECT"
        fail_reason = f"Not statistically significant: p={ic_pvalue:.4f} ≥ {SignalReport.PVALUE_MAX_THRESHOLD}"
    elif not is_monotonic:
        recommendation = "REFINE"
        fail_reason = "Quintile returns are not monotonic. Investigate signal specification."
    elif ic_s < SignalReport.IC_MIN_THRESHOLD:
        recommendation = "REFINE"
        fail_reason = f"Rank IC {ic_s:.4f} < {SignalReport.IC_MIN_THRESHOLD}"
    else:
        recommendation = "PROMOTE"

    return SignalReport(
        signal_name=signal_name,
        n_observations=n,
        ic_pearson=ic_p,
        ic_spearman=ic_s,
        ic_tstat=ic_tstat,
        ic_pvalue=ic_pvalue,
        ic_stderr=ic_stderr,
        rolling_ic_mean=rolling_ic_mean,
        rolling_ic_std=rolling_ic_std,
        ic_decay_1d=ic_d1,
        ic_decay_5d=ic_d5,
        ic_decay_10d=ic_d10,
        ic_decay_20d=ic_d20,
        quintile_spread_annualized=spread_ann,
        is_monotonic=is_monotonic,
        turnover_mean=turnover_mean,
        hit_rate=hit_rate,
        coverage=coverage,
        bull_ic=bull_ic,
        bear_ic=bear_ic,
        high_vol_ic=high_vol_ic,
        low_vol_ic=low_vol_ic,
        recommendation=recommendation,
        fail_reason=fail_reason,
    )


# ---------------------------------------------------------------------------
# Batch Evaluation (Research Leaderboard)
# ---------------------------------------------------------------------------

def run_research_batch(
    signals: dict[str, pd.Series],
    forward_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Evaluate a batch of signals and return a ranked leaderboard DataFrame."""
    rows = []
    for name, signal in signals.items():
        report = evaluate_signal(
            signal, forward_returns,
            benchmark_returns=benchmark_returns,
            signal_name=name,
        )
        rows.append({
            "Signal": name,
            "IC (Pearson)": round(report.ic_pearson, 4),
            "IC (Rank)": round(report.ic_spearman, 4),
            "t-stat": round(report.ic_tstat, 2),
            "p-value": round(report.ic_pvalue, 4),
            "Rolling IC": round(report.rolling_ic_mean, 4),
            "Q-Spread (Ann)": f"{report.quintile_spread_annualized:.2%}",
            "Monotonic": report.is_monotonic,
            "Turnover": f"{report.turnover_mean:.2%}",
            "Hit Rate": f"{report.hit_rate:.1%}",
            "Bull IC": round(report.bull_ic, 4) if not np.isnan(report.bull_ic) else "N/A",
            "Bear IC": round(report.bear_ic, 4) if not np.isnan(report.bear_ic) else "N/A",
            "Recommendation": report.recommendation,
        })

    df = pd.DataFrame(rows).sort_values("IC (Pearson)", ascending=False)
    return df
