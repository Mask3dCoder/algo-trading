"""Structural tests for the pre-registered factor experiment harness.

These tests assert the harness RUNS and produces the right JSON shape. They
deliberately do not assert on performance values: the smoke panel is triage
only and its numbers are excluded from the gate by protocol.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantcore.research import factor_experiment as fe


def _synthetic_panel(days: int = 1000, symbols: int = 40, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=days)
    close = pd.DataFrame(index=dates)
    for j in range(symbols):
        drift = rng.normal(0.0002, 0.0004)
        rets = rng.normal(drift, 0.015, days)
        # Give a few names a persistent momentum trend.
        if j % 7 == 0:
            rets += np.linspace(0.0, 0.0015, days)
        close[f"S{j:02d}"] = 100.0 * np.cumprod(1.0 + rets)
    return {
        "close": close,
        "dates": close.index,
        "symbols": list(close.columns),
        "corp_actions": pd.DataFrame(),
        "universe": pd.DataFrame(),
        "source": "triage",
        "window": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
        "provenance": "synthetic test panel - triage only",
    }


class TestVariantSet:
    def test_exactly_three_frozen_variants(self) -> None:
        assert fe.VARIANTS == ("momentum", "low_vol", "composite")
        assert len(fe.VARIANTS) == 3


class TestHarnessStructure:
    def test_smoke_run_produces_expected_json_shape(self, tmp_path) -> None:
        panel = _synthetic_panel()
        cfg = fe.ExperimentConfig(source="triage", triage=True, n_bootstrap=30, seed=1)
        results = fe.run_experiment(panel, cfg)

        assert set(results["variants"].keys()) == set(fe.VARIANTS)
        meta = results["meta"]
        assert meta["triage"] is True
        assert meta["protocol_lock_commit"] == "ed52bae"
        assert meta["gate"]["margin"]["dsr_p"] == 0.04

        out = tmp_path / "results.json"
        fe.write_results(results, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert set(loaded["variants"].keys()) == set(fe.VARIANTS)

        for variant in fe.VARIANTS:
            res = loaded["variants"][variant]
            assert res["verdict"] in {"PASS", "FAIL", "FAIL - insufficient margin", "ERROR"}
            for key in (
                "oos_sharpe",
                "oos_sharpe_ci_90",
                "dsr_p",
                "net_total_return",
                "spy_total_return",
                "avg_turnover",
                "trade_count",
                "n_oos_days",
                "oos_return_series",
                "per_year_sharpe",
                "per_fold_sharpe",
                "verdict",
            ):
                assert key in res, f"{variant} missing {key}"
            assert res["n_oos_days"] >= 30
            assert len(res["oos_return_series"]) == res["n_oos_days"]
            assert all(np.isfinite(x) for x in res["oos_return_series"])

    def test_split_adjustment_preserves_momentum(self) -> None:
        """A 20:1 split must not change a pre-split momentum ratio."""
        dates = pd.bdate_range("2021-01-01", periods=600)
        close = pd.DataFrame({"AMZN": np.linspace(2500.0, 2600.0, 600)}, index=dates)
        # Split effective at index 300: a lookback spanning it is distorted raw.
        split_date = dates[300]
        close.loc[dates[dates >= split_date], "AMZN"] /= 20.0
        actions = pd.DataFrame([{
            "action_type": "split", "symbol": "AMZN",
            "effective_date": split_date, "announced_at": pd.NaT, "value": 20.0,
        }])
        factors = fe._split_adjust_factors(close, actions)
        adj = close * factors

        i = 400  # formation whose lookback spans the split (i-273 < 300 < i-21)
        mom_raw = close.iloc[i - 21]["AMZN"] / close.iloc[i - 273]["AMZN"] - 1.0
        mom_adj = adj.iloc[i - 21]["AMZN"] / adj.iloc[i - 273]["AMZN"] - 1.0
        assert mom_raw < -0.9  # raw momentum is a fake -95% from the split gap
        # True split-free ratio: the unmodified linear ramp at the two lookback points.
        p_fast = 2500.0 + 100.0 * (i - 21) / 599.0
        p_slow = 2500.0 + 100.0 * (i - 273) / 599.0
        expected = p_fast / p_slow - 1.0
        assert mom_adj == pytest.approx(expected, rel=1e-6)

    def test_no_crash_on_empty_corp_actions(self) -> None:
        panel = _synthetic_panel(days=800, symbols=20)
        cfg = fe.ExperimentConfig(source="triage", n_bootstrap=20, triage=True)
        results = fe.run_experiment(panel, cfg)
        assert len(results["variants"]) == 3
