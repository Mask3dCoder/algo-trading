"""Phase 2 data-pipeline bias guardrail tests."""

from __future__ import annotations

import pandas as pd

from quantcore.data.news_sentiment import aggregate_daily_sentiment
from quantcore.data.point_in_time import (
    FeatureRecord,
    PointInTimeStore,
    detect_corporate_action_lookahead,
    detect_earnings_lookahead,
    detect_macro_release_lag,
    detect_survivorship_bias,
    find_future_leakage,
    get_features_as_of,
)
from quantcore.data.sources.fred_loader import FREDDataLoader


class FakeScorer:
    def score(self, text: str) -> float:
        return 1.0 if "beat" in text.lower() else -1.0


def test_future_leakage_is_blocked_by_point_in_time_query() -> None:
    bar_ts = pd.Timestamp("2026-06-09 09:30:00", tz="UTC")
    store = PointInTimeStore(
        [
            FeatureRecord("fred", "SPY", "2026-06-09", bar_ts - pd.Timedelta(minutes=1), "vix", 18.0),
            FeatureRecord("fred", "SPY", "2026-06-09", bar_ts + pd.Timedelta(minutes=1), "vix", 99.0),
        ]
    )

    features = get_features_as_of("SPY", bar_ts, store=store)

    leaked = find_future_leakage(store.data, bar_ts)

    assert features["feature_name"].tolist() == ["vix"]
    assert features.loc[0, "value"] == 18.0
    assert leaked["feature_name"].tolist() == ["vix"]


def test_survivorship_bias_detects_current_only_names() -> None:
    historical = ["AAPL", "MSFT", "DEAD"]
    current = ["AAPL", "MSFT", "NVDA"]

    bias = detect_survivorship_bias(historical, current)

    assert bias == ["NVDA"]


def test_corporate_action_lookahead_detects_pre_announcement_adjustment() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": [
                pd.Timestamp("2026-06-09 09:30:00", tz="UTC"),
                pd.Timestamp("2026-06-10 09:30:00", tz="UTC"),
            ],
            "close": [190.0, 95.0],
        }
    )
    actions = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2026-06-10", tz="UTC")],
            "valid_from": [pd.Timestamp("2026-06-10 16:05:00", tz="UTC")],
            "action_type": ["split"],
        }
    )

    violations = detect_corporate_action_lookahead(prices, actions, "2026-06-10 09:30:00+00:00")

    assert len(violations) == 1
    assert violations.loc[0, "_future_action_count"] == 1


def test_earnings_lookahead_is_detected_before_announcement() -> None:
    earnings = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "as_of": pd.Timestamp("2026-06-30", tz="UTC"),
                "valid_from": pd.Timestamp("2026-07-31 16:05:00", tz="UTC"),
                "feature_name": "eps_actual",
                "value": 1.40,
            }
        ]
    )

    violations = detect_earnings_lookahead(earnings, "2026-06-30 09:30:00+00:00")

    assert len(violations) == 1
    assert violations.loc[0, "feature_name"] == "eps_actual"


def test_macro_release_lag_is_detected_before_publication() -> None:
    macro = pd.DataFrame(
        [
            {
                "symbol": "CPIAUCSL",
                "as_of": pd.Timestamp("2026-05-01", tz="UTC"),
                "valid_from": pd.Timestamp("2026-06-10 08:30:00", tz="UTC"),
                "feature_name": "cpi",
                "value": 314.0,
            }
        ]
    )

    violations = detect_macro_release_lag(macro, "2026-06-09 16:00:00+00:00")

    assert len(violations) == 1
    assert violations.loc[0, "symbol"] == "CPIAUCSL"


def test_fred_feature_frame_uses_as_of_and_valid_from(tmp_path, monkeypatch) -> None:
    loader = FREDDataLoader(api_key="fake", cache_dir=tmp_path)

    def fake_fetch(series_id: str, start: str, end: str | None) -> pd.Series:
        return pd.Series(
            [1.0, 2.0],
            index=pd.to_datetime(["2026-06-09", "2026-06-10"]),
            name=series_id,
        )

    monkeypatch.setattr(loader, "_fetch_series", fake_fetch)

    frame = loader.get_feature_frame("DGS10", "2026-06-09", "2026-06-10")

    assert frame.columns.tolist() == ["source", "symbol", "as_of", "valid_from", "feature_name", "value"]
    assert frame.loc[0, "source"] == "fred"
    assert frame.loc[0, "valid_from"] == frame.loc[0, "as_of"]


def test_alfred_first_release_feature_frame_uses_realtime_valid_from(tmp_path, monkeypatch) -> None:
    class FakeFred:
        def get_series_first_release(self, series_id: str) -> pd.DataFrame:
            assert series_id == "CPIAUCSL"
            return pd.DataFrame(
                [
                    {
                        "date": "2026-05-01",
                        "realtime_start": "2026-06-10 08:30:00",
                        "value": 314.0,
                    }
                ]
            )

    loader = FREDDataLoader(api_key="fake", cache_dir=tmp_path)
    monkeypatch.setattr(loader, "_get_fred", lambda: FakeFred())

    frame = loader.get_feature_frame("CPIAUCSL", use_first_release=True)

    assert frame.loc[0, "source"] == "alfred_first_release"
    assert frame.loc[0, "as_of"] == pd.Timestamp("2026-05-01", tz="UTC")
    assert frame.loc[0, "valid_from"] == pd.Timestamp("2026-06-10 08:30:00", tz="UTC")


def test_news_aggregation_emits_unified_feature_schema() -> None:
    headlines = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-06-09 14:32:00", tz="UTC"),
                "valid_from": pd.Timestamp("2026-06-09 14:32:00", tz="UTC"),
                "as_of": pd.Timestamp("2026-06-09", tz="UTC"),
                "ticker": "AAPL",
                "symbol": "AAPL",
                "headline": "AAPL earnings beat expectations",
                "body": "AAPL earnings beat expectations",
                "source": "alpaca_news",
            }
        ]
    )

    sentiment = aggregate_daily_sentiment(headlines, scorer=FakeScorer())

    assert sentiment.loc[0, "source"] == "alpaca_news"
    assert sentiment.loc[0, "symbol"] == "AAPL"
    assert sentiment.loc[0, "feature_name"] == "news_sentiment"
    assert sentiment.loc[0, "valid_from"] == pd.Timestamp("2026-06-09 14:32:00", tz="UTC")


def test_options_flow_analyzer_emits_required_opra_signals() -> None:
    from quantcore.alt_data.options_flow.unusual_activity import OptionsFlowAnalyzer

    trades = [
        {
            "underlying": "SPY",
            "symbol": "SPY260619C00520000",
            "option_type": "call",
            "expiration": "2026-06-19",
            "strike": 520.0,
            "spot": 500.0,
            "volume": 1200,
            "price": 2.0,
            "iv": 0.25,
            "timestamp": "2026-06-09 14:32:00+00:00",
        },
        {
            "underlying": "SPY",
            "symbol": "SPY260619P00480000",
            "option_type": "put",
            "expiration": "2026-06-19",
            "strike": 480.0,
            "spot": 500.0,
            "volume": 300,
            "price": 1.0,
            "iv": 0.35,
            "timestamp": "2026-06-09 14:33:00+00:00",
        },
    ]

    frame = OptionsFlowAnalyzer().compute_feature_frame(trades, historical_avg_volume={"SPY": 100.0})

    assert set(frame["feature_name"]) == {
        "unusual_volume_ratio",
        "put_call_ratio_by_expiry",
        "large_order_flag",
        "skew",
        "term_structure",
    }
    assert frame.loc[frame["feature_name"] == "large_order_flag", "value"].iloc[0] == 1.0
    assert frame["valid_from"].max() == pd.Timestamp("2026-06-09 14:33:00", tz="UTC")
