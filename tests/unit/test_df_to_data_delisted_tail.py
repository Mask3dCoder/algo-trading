"""Regression test: delisted names must never be zero-filled in the harness.

A zero price after a name's last traded day manufactures a fake -100% bar and
poisons cross-sectional scores. ``_df_to_data`` must mask dead names to NaN,
carry a last-traded-day mask, and ``_compute_signal_returns`` must skip
non-finite contributions.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.validate_all_strategies import _compute_signal_returns, _df_to_data


def _make_panel() -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=40)
    rows: list[dict] = []
    for t, ts in enumerate(idx):
        rows.append(
            {
                "timestamp": ts,
                "symbol": "AAPL",
                "open": 100.0 + t,
                "high": 101.0 + t,
                "low": 99.0 + t,
                "close": 100.5 + t,
                "volume": 1_000_000,
            }
        )
        if t <= 19:  # DEAD delists after bar 19
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": "DEAD",
                    "open": 50.0 + t,
                    "high": 51.0 + t,
                    "low": 49.0 + t,
                    "close": 50.5 + t,
                    "volume": 500_000,
                }
            )
        if t not in (10, 11, 12):  # GAP has a 3-day hole (filled by ffill limit=5)
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": "GAP",
                    "open": 20.0 + t,
                    "high": 21.0 + t,
                    "low": 19.0 + t,
                    "close": 20.5 + t,
                    "volume": 200_000,
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_df_to_data_masks_delisted_names() -> None:
    df = _make_panel()
    data = _df_to_data(df)

    assert "DEAD" in data["symbols"]
    assert "AAPL" in data["symbols"]

    close = data["close"]
    live_mask = data["live_mask"]
    assert close.shape == live_mask.shape
    assert live_mask.shape[0] == df["timestamp"].nunique()

    dead_col = data["symbols"].index("DEAD")
    aapl_col = data["symbols"].index("AAPL")

    # Live through bar 19, NaN afterwards -- never 0.0
    assert live_mask[:20, dead_col].all()
    assert not live_mask[20:, dead_col].any()
    assert np.isnan(close[20:, dead_col]).all()
    assert not np.any(close == 0.0)

    # AAPL fully live
    assert live_mask[:, aapl_col].all()

    assert data["last_traded_index"]["DEAD"] == 19
    assert data["last_traded_index"]["AAPL"] == close.shape[0] - 1

    # The 3-day hole is filled causally; GAP stays finite throughout
    gap_col = data["symbols"].index("GAP")
    assert np.isfinite(close[:, gap_col]).all()


class _StubHoldDead:
    """Emits a persistent long signal on DEAD for every bar."""

    def set_timestamp(self, ts) -> None:  # noqa: D102
        pass

    def on_data(self, bar_data):  # noqa: D102
        return [SimpleNamespace(symbol="DEAD", direction=1.0, strength=1.0)]


def test_compute_signal_returns_no_fake_crash_for_delisted_name() -> None:
    df = _make_panel()
    data = _df_to_data(df)
    timestamps = sorted(df["timestamp"].unique())
    n_bars = data["close"].shape[0]

    returns, trades = _compute_signal_returns(
        _StubHoldDead(),
        data,
        0,
        n_bars,
        timestamps=timestamps,
        tc_bps=5.0,
        spread_bps=5.0,
        impact_bps=10.0,
    )

    assert np.all(np.isfinite(returns)), "NaN leaked into bar returns"
    # The fake -100% bar that zero-filling used to manufacture must not exist.
    assert not np.any(returns <= -0.99)
    # The last-trading-day return is real (bar 18 -> 19) ...
    assert np.isfinite(returns[18]) and returns[18] > 0.0
    # ... and after the last traded day the contribution is 0, not -100%.
    assert returns[20] == pytest.approx(0.0, abs=1e-12)
    assert returns[19] == pytest.approx(0.0, abs=1e-12)

    assert len(trades) == 1
    dead_col = data["symbols"].index("DEAD")
    assert trades[0]["exit_price"] == data["close"][19, dead_col]
    assert np.isfinite(trades[0]["pnl_abs"])
