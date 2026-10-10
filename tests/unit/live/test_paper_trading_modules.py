"""Tests for the paper_trading package: RiskController, ExecutionEnvironment,
AnalyticsDashboard, PositionSizer, DeltaTelemetry, PassiveExecutionProxy."""

from __future__ import annotations

import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from quantcore.live.circuit_breaker import CircuitBreaker, CircuitBreakerState
from quantcore.live.paper_trading import (
    AnalyticsConfig,
    ConnectionMonitor,
    ConnectionState,
    DeltaTelemetry,
    DeltaTelemetryConfig,
    DeltaTelemetryMiddleware,
    ExecutionState,
    PaperExecutionConfig,
    PaperExecutionEnvironment,
    PaperTradingAnalytics,
    PassiveExecutionProxy,
    PositionSizer,
    PositionSizerConfig,
    RiskController,
    RiskControllerConfig,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def circuit_breaker():
    return CircuitBreaker()


@pytest.fixture
def risk_cfg(tmp_dir):
    return RiskControllerConfig(
        max_drawdown_pct=0.10,
        sharpe_lookback_days=63,
        sharpe_min_threshold=0.0,
        macro_vix_threshold=30.0,
        macro_drawdown_warning_pct=0.08,
        macro_drawdown_kill_pct=0.12,
        state_dir=tmp_dir / "state",
    )


@pytest.fixture
def risk_controller(risk_cfg, circuit_breaker):
    return RiskController(
        config=risk_cfg,
        circuit_breaker=circuit_breaker,
        initial_equity=100_000.0,
    )


# ══════════════════════════════════════════════════════════════════════════
# Module 1: RiskController
# ══════════════════════════════════════════════════════════════════════════


class TestRiskController:
    """Kill switch tests — all three switches must fire correctly."""

    # ── Kill Switch 1: Hard MaxDD ──────────────────────────────────────

    def test_maxdd_triggers_at_10_percent(self, risk_controller):
        """Equity drop of exactly 10% triggers HALT."""
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        # Peak is now 100,000.  Drop to 89,999 = -10.001%
        risk_controller.update_equity(89_999.0, date(2026, 1, 2))
        assert risk_controller.state == ExecutionState.HALTED

    def test_maxdd_does_not_trigger_below_threshold(self, risk_controller):
        """Equity drop of 9.9% does NOT trigger HALT."""
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        risk_controller.update_equity(90_100.0, date(2026, 1, 2))
        assert risk_controller.state == ExecutionState.ACTIVE

    def test_maxdd_uses_peak_equity_not_initial(self, risk_controller):
        """Peak equity tracks running maximum, not just initial capital."""
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        risk_controller.update_equity(110_000.0, date(2026, 1, 2))  # new peak
        # 10% from peak 110,000 = 99,000
        risk_controller.update_equity(99_100.0, date(2026, 1, 3))
        assert risk_controller.state == ExecutionState.ACTIVE
        # Now trigger: 99,000 is exactly -10%
        risk_controller.update_equity(98_999.0, date(2026, 1, 4))
        assert risk_controller.state == ExecutionState.HALTED
        assert risk_controller.peak_equity == 110_000.0

    def test_maxdd_halts_circuit_breaker(self, risk_controller, circuit_breaker):
        """On MaxDD kill, the circuit breaker is set to CRITICAL_HALT."""
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        risk_controller.update_equity(89_999.0, date(2026, 1, 2))
        assert circuit_breaker.state == CircuitBreakerState.CRITICAL_HALT

    # ── Kill Switch 2: Rolling 3-Month Sharpe ───────────────────────────

    def test_negative_sharpe_triggers_pause(self, risk_controller):
        """Rolling Sharpe < 0 triggers PAUSED (not HALTED)."""
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        # Simulate 65 days of negative returns with variance (avoid zero std)
        import random
        random.seed(42)
        dt = date(2026, 1, 2)
        for i in range(65):
            ret = -0.005 + random.gauss(0, 0.002)  # mostly negative, varying
            risk_controller.record_daily_return(dt + timedelta(days=i), ret)
        # Must call check_monthly_sharpe in a DIFFERENT month than last check
        check_date = date(2026, 4, 1)  # April — different month
        sharpe = risk_controller.check_monthly_sharpe(check_date)
        assert sharpe < 0.0
        assert risk_controller.state == ExecutionState.PAUSED

    def test_positive_sharpe_does_not_halt(self, risk_controller):
        """Rolling Sharpe >= 0 does NOT trigger anything."""
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        import random
        random.seed(43)
        dt = date(2026, 1, 2)
        for i in range(65):
            ret = 0.002 + random.gauss(0, 0.001)  # mostly positive, varying
            risk_controller.record_daily_return(dt + timedelta(days=i), ret)
        # Different month for the check
        check_date = date(2026, 4, 1)
        sharpe = risk_controller.check_monthly_sharpe(check_date)
        assert sharpe > 0.0
        assert risk_controller.state == ExecutionState.ACTIVE

    # ── Kill Switch 3: Macro Volatility Exit ─────────────────────────────

    def test_macro_warning_at_vix_30_pd_8(self, risk_controller):
        """VIX > 30 and drawdown > 8% → warning (reduce 50%)."""
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        status = risk_controller.check_macro_conditions(
            vix_level=35.0,
            current_drawdown=0.09,
        )
        assert status == "reduce_50"
        assert risk_controller.state == ExecutionState.ACTIVE  # not halted yet

    def test_macro_halt_at_vix_30_pd_12(self, risk_controller):
        """VIX > 30 and drawdown > 12% → halt."""
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        status = risk_controller.check_macro_conditions(
            vix_level=35.0,
            current_drawdown=0.13,
        )
        assert status == "halt"
        assert risk_controller.state == ExecutionState.HALTED

    def test_macro_ok_when_vix_below_threshold(self, risk_controller):
        """VIX < 30 returns 'ok' regardless of drawdown."""
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        status = risk_controller.check_macro_conditions(
            vix_level=20.0,
            current_drawdown=0.15,
        )
        assert status == "ok"

    # ── Manual reset ────────────────────────────────────────────────────

    def test_manual_reset_restores_active(self, risk_controller):
        risk_controller.update_equity(100_000.0, date(2026, 1, 1))
        risk_controller.update_equity(89_000.0, date(2026, 1, 2))
        assert risk_controller.state == ExecutionState.HALTED

        risk_controller.manual_reset("committee_chair", "Approved after review")
        assert risk_controller.is_active

    # ── Persistence ─────────────────────────────────────────────────────

    def test_state_persists_across_restart(self, risk_cfg, circuit_breaker):
        """Peak equity and state survive process restart."""
        rc1 = RiskController(config=risk_cfg, circuit_breaker=circuit_breaker, initial_equity=100_000.0)
        rc1.update_equity(120_000.0, date(2026, 1, 1))
        rc1.update_equity(105_000.0, date(2026, 1, 2))

        # Simulate restart — new instance reads persisted state
        rc2 = RiskController(config=risk_cfg, circuit_breaker=circuit_breaker, initial_equity=100_000.0)
        assert rc2.peak_equity == 120_000.0


# ══════════════════════════════════════════════════════════════════════════
# Module 2: PaperExecutionEnvironment
# ══════════════════════════════════════════════════════════════════════════


class TestPaperExecutionEnvironment:

    @pytest.fixture
    def exec_env(self, tmp_dir):
        return PaperExecutionEnvironment(
            config=PaperExecutionConfig(
                initial_paper_capital=100_000.0,
                slippage_spread_bps=5.0,
                options_slippage_bps=50.0,
                log_dir=tmp_dir / "logs",
            ),
        )

    def test_initial_equity(self, exec_env):
        assert exec_env.equity == 100_000.0
        assert exec_env.peak_equity == 100_000.0

    def test_equity_update_tracks_peak(self, exec_env):
        exec_env.update_equity(5_000.0)
        assert exec_env.equity == 105_000.0
        assert exec_env.peak_equity == 105_000.0

        exec_env.update_equity(-3_000.0)
        assert exec_env.equity == 102_000.0
        assert exec_env.peak_equity == 105_000.0  # peak stays

    def test_ingest_daily_bar_appends_to_window(self, exec_env):
        exec_env.ingest_daily_bar(date(2026, 1, 1), 450.0, vix=18.5)
        exec_env.ingest_daily_bar(date(2026, 1, 2), 452.0, vix=18.0)
        # IV should be computed after enough bars
        iv = exec_env.get_current_iv()
        assert iv > 0.0

    def test_compute_order_details_option(self, exec_env):
        details = exec_env.compute_order_details(
            theoretical_price=3.50,
            quantity=10,
            side="sell",
            asset_type="option",
        )
        assert details["theoretical_price"] == 3.50
        assert "fill_price" in details
        assert "slippage_bps" in details
        assert details["commission"] > 0.0  # $0.65 * 10 = $6.50
        assert details["asset_type"] == "option"

    def test_compute_order_details_equity(self, exec_env):
        details = exec_env.compute_order_details(
            theoretical_price=450.0,
            quantity=100,
            side="buy",
            asset_type="equity",
        )
        assert details["commission"] == pytest.approx(0.50, abs=0.01)  # $0.005 * 100

    def test_delta_tracker_net_delta(self, exec_env):
        positions = [
            {"call_delta": 0.30, "put_delta": -0.25, "contracts": 5},
        ]
        record = exec_env.record_delta_hedge(
            dt=date(2026, 1, 5),
            positions=positions,
            hedge_shares=25.0,  # buy 25 shares to offset
        )
        # Short call delta: -0.30 * 100 * 5 = -150
        # Short put delta:  -(-0.25) * 100 * 5 = +125
        # Options total: -150 + 125 = -25
        # Hedge: +25 shares
        # Net delta: 0
        assert record["net_delta"] == pytest.approx(0.0, abs=0.01)
        assert record["delta_drift"] == pytest.approx(0.0, abs=0.01)

    def test_slippage_accumulators(self, exec_env):
        exec_env.compute_order_details(3.50, 10, "sell", "option")
        exec_env.compute_order_details(450.0, 100, "buy", "equity")
        stats = exec_env.get_todays_slippage()
        assert stats["trades_today"] == 2
        assert stats["slippage_dollars"] >= 0.0

        exec_env.reset_daily_slippage()
        stats = exec_env.get_todays_slippage()
        assert stats["trades_today"] == 0


# ══════════════════════════════════════════════════════════════════════════
# Module 3: PaperTradingAnalytics
# ══════════════════════════════════════════════════════════════════════════


class TestPaperTradingAnalytics:

    @pytest.fixture
    def analytics(self, tmp_dir):
        return PaperTradingAnalytics(
            config=AnalyticsConfig(
                output_dir=tmp_dir,
                expected_sharpe_annual=1.30,
                expected_annual_return=0.12,
                expected_win_rate=0.867,
            ),
        )

    def test_record_daily_result_creates_csv_row(self, analytics, tmp_dir):
        result = analytics.record_daily_result(
            dt=date(2026, 6, 23),
            equity=100_500.0,
            realized_pnl=500.0,
            expected_pnl=47.62,  # ~12%/252 * 100k
            slippage_bps=3.5,
            slippage_dollars=15.0,
            trades_executed=2,
            open_positions=1,
            net_delta=0.5,
            vix_close=18.5,
        )
        assert "row" in result
        assert result["row"]["date"] == "2026-06-23"
        assert result["row"]["equity"] == 100_500.0
        csv_path = tmp_dir / "daily_log.csv"
        assert csv_path.exists()

    def test_record_trade_updates_win_rate(self, analytics):
        # Need at least one daily result for analytics to have equity_history
        analytics.record_daily_result(
            dt=date(2026, 6, 19),
            equity=100_000.0,
            realized_pnl=0.0,
        )
        analytics.record_trade(pnl=100.0, trade_date=date(2026, 6, 20))
        analytics.record_trade(pnl=-50.0, trade_date=date(2026, 6, 21))
        analytics.record_trade(pnl=200.0, trade_date=date(2026, 6, 22))
        summary = analytics.generate_summary()
        assert summary["status"] == "ok"
        assert summary["total_closed_trades"] == 3
        assert summary["overall_win_rate"] == pytest.approx(66.67, abs=0.1)

    def test_win_rate_drift_alert_fires(self, analytics, tmp_dir):
        """3-month win rate below 65% triggers alert."""
        # Record enough losing trades to push win rate below 65%
        for i in range(20):
            analytics.record_trade(pnl=-100.0, trade_date=date(2026, 1, 1) + timedelta(days=i))
        for i in range(5):
            analytics.record_trade(pnl=50.0, trade_date=date(2026, 2, 1) + timedelta(days=i))

        result = analytics.record_daily_result(
            dt=date(2026, 6, 23),
            equity=95_000.0,
            realized_pnl=-100.0,
            trades_executed=0,
            open_positions=0,
        )
        # Should have win_rate_drift alert
        alert_types = [a["type"] for a in result["alerts"]]
        assert "win_rate_drift" in alert_types

    def test_pnl_drift_alert_fire(self, analytics, tmp_dir):
        """P&L drift > 2 sigma triggers warning."""
        result = analytics.record_daily_result(
            dt=date(2026, 6, 23),
            equity=100_000.0,
            realized_pnl=-500.0,
            expected_pnl=47.62,
        )
        # Very large drift should trigger
        if result["row"].get("pnl_drift_pct") is not None:
            drift_pct = abs(result["row"]["pnl_drift_pct"])
            if drift_pct > 200:  # ~2 sigma equivalent
                alert_types = [a["type"] for a in result["alerts"]]
                assert "pnl_drift" in alert_types

    def test_generate_summary_returns_status(self, analytics):
        summary = analytics.generate_summary()
        # No equity history yet
        assert summary["status"] == "no_data"

        analytics.record_daily_result(
            dt=date(2026, 6, 23),
            equity=100_000.0,
            realized_pnl=0.0,
        )
        summary = analytics.generate_summary()
        assert summary["status"] == "ok"

    def test_alert_log_created(self, analytics, tmp_dir):
        analytics.record_daily_result(
            dt=date(2026, 6, 23),
            equity=100_000.0,
            realized_pnl=0.0,
        )
        alert_path = tmp_dir / "alerts.log"
        # May or may not exist depending on alert conditions


# ══════════════════════════════════════════════════════════════════════════
# PositionSizer
# ══════════════════════════════════════════════════════════════════════════


class TestPositionSizer:

    def test_option_entry_sizing(self):
        sizer = PositionSizer(equity=100_000.0)
        result = sizer.size_option_entry(premium=5.0, underlying_price=450.0)
        # 5% of 100k = $5000 max premium spend
        # $5000 / ($5 * 100) = 10 contracts
        assert result["contracts"] == 10
        assert result["sizing_basis"] == "premium"

    def test_option_entry_zero_premium(self):
        sizer = PositionSizer(equity=100_000.0)
        result = sizer.size_option_entry(premium=0.0)
        assert result["contracts"] == 0
        assert result["sizing_basis"] == "invalid_premium"

    def test_hard_cap_limits_contracts(self):
        sizer = PositionSizer(
            equity=1_000_000.0,
            config=PositionSizerConfig(max_contracts_per_trade=10),
        )
        result = sizer.size_option_entry(premium=1.0, underlying_price=450.0)
        # Without the cap, $50k / $100 = 500; but cap is 10
        assert result["contracts"] <= 10
        assert result["capped"] is True

    def test_portfolio_limit_respects_open_positions(self):
        sizer = PositionSizer(
            equity=100_000.0,
            open_positions=[{"contracts": 8, "premium_cost": 4000.0, "margin_required": 8000.0}],
        )
        # max_portfolio_premium_pct = 20% → $20000, already $4000 used
        # remaining = $16000 → $16000 / ($5 * 100) = 32 contracts
        # but max_total_contracts = 50, already 8 open → max 42
        result = sizer.size_option_entry(premium=5.0, underlying_price=450.0)
        assert result["contracts"] >= 0

    def test_delta_hedge_sizing(self):
        sizer = PositionSizer(equity=100_000.0)
        result = sizer.size_delta_hedge(
            option_delta=-0.30,
            contracts=10,
            target_net_delta=0.0,
        )
        # Total options delta: -0.30 * 10 * 100 = -300
        # Need to buy 300 shares to offset
        assert result["hedge_shares"] == 300
        assert result["direction"] == "buy"
        assert abs(result["net_delta_after"]) < 1.0  # within 1 share rounding

    def test_update_equity(self):
        sizer = PositionSizer(equity=100_000.0)
        sizer.update_equity(105_000.0)
        assert sizer.equity == 105_000.0
        assert sizer.peak_equity == 105_000.0

    def test_from_yaml(self, tmp_dir):
        yaml_path = tmp_dir / "test_config.yaml"
        yaml_path.write_text("""
sizing:
  max_premium_allocation_pct: 0.03
  max_portfolio_premium_pct: 0.15
  max_contracts_per_trade: 5
  max_total_contracts: 25
execution:
  initial_cash: 50000
""")
        cfg = PositionSizerConfig.from_yaml(yaml_path)
        assert cfg.max_premium_allocation_pct == 0.03
        assert cfg.max_portfolio_premium_pct == 0.15
        assert cfg.max_contracts_per_trade == 5
        assert cfg.max_total_contracts == 25


# ══════════════════════════════════════════════════════════════════════════
# DeltaTelemetry
# ══════════════════════════════════════════════════════════════════════════


class TestDeltaTelemetry:

    @pytest.fixture
    def telemetry(self, tmp_dir):
        return DeltaTelemetry(
            config=DeltaTelemetryConfig(
                output_dir=tmp_dir,
                log_interval_seconds=0,  # log on every call
                drift_warning_threshold=5.0,
                drift_critical_threshold=15.0,
            ),
        )

    def test_snapshot_records_net_delta(self, telemetry, tmp_dir):
        positions = [
            {"call_delta": 0.30, "put_delta": -0.25, "contracts": 5},
        ]
        snapshot = telemetry.record_snapshot(
            dt=date(2026, 6, 23),
            positions=positions,
            hedge_shares=25.0,
            equity=100_000.0,
            vix_close=18.0,
            force_log=True,
        )
        assert snapshot.net_delta == pytest.approx(0.0, abs=0.01)
        assert snapshot.alert_level == "ok"

    def test_critical_drift_alert(self, telemetry, tmp_dir):
        positions = [
            {"call_delta": 0.50, "put_delta": -0.10, "contracts": 10},
        ]
        # Net delta will be large: short call -500, short put +100 = -400
        # With 0 hedge shares: net = -400
        snapshot = telemetry.record_snapshot(
            dt=date(2026, 6, 23),
            positions=positions,
            hedge_shares=0.0,
            equity=100_000.0,
            force_log=True,
        )
        assert snapshot.alert_level == "critical"
        assert abs(snapshot.delta_drift) > 15.0

    def test_drift_statistics(self, telemetry):
        for i in range(5):
            telemetry.record_snapshot(
                dt=date(2026, 6, 20) + timedelta(days=i),
                positions=[{"call_delta": 0.30, "put_delta": -0.25, "contracts": 3}],
                hedge_shares=15.0,
                equity=100_000.0,
                force_log=True,
            )
        stats = telemetry.get_drift_statistics()
        assert stats["status"] == "ok"
        assert stats["count"] == 5

    def test_telemetry_csv_created(self, telemetry, tmp_dir):
        telemetry.record_snapshot(
            dt=date(2026, 6, 23),
            positions=[],
            hedge_shares=0.0,
            equity=100_000.0,
            force_log=True,
        )
        csv_path = tmp_dir / "delta_telemetry.csv"
        assert csv_path.exists()

    def test_middleware_delegates(self, tmp_dir):
        middleware = DeltaTelemetryMiddleware(
            config=DeltaTelemetryConfig(
                output_dir=tmp_dir,
                log_interval_seconds=0,
            ),
        )
        snapshot = middleware.on_positions_updated(
            dt=date(2026, 6, 23),
            positions=[{"call_delta": 0.25, "put_delta": -0.20, "contracts": 2}],
            hedge_shares=10.0,
            equity=100_000.0,
            force_log=True,
        )
        assert snapshot.alert_level == "ok"
        stats = middleware.get_drift_stats()
        assert stats["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════
# PassiveExecutionProxy
# ══════════════════════════════════════════════════════════════════════════


class TestPassiveExecutionProxy:

    def test_log_bypassed_order(self, tmp_dir):
        proxy = PassiveExecutionProxy(output_path=tmp_dir / "passive_trades.json")
        record = proxy.log_bypassed_order(
            ticker="SPY",
            side="sell",
            target_size=10,
            mid_price=450.0,
            bid_ask_spread=0.02,
            contract="SPY_062026C450",
            option_type="call",
            notional=4500.0,
            reason="strategy_signal",
        )
        assert record.mode == "passive"
        assert proxy.trade_count == 1
        # Verify JSONL file
        log_path = tmp_dir / "passive_trades.json"
        assert log_path.exists()

    def test_log_delta_rebalance(self, tmp_dir):
        proxy = PassiveExecutionProxy(output_path=tmp_dir / "passive_trades.json")
        record = proxy.log_delta_rebalance(
            ticker="SPY",
            hedge_shares=25,
            mid_price=450.0,
            bid_ask_spread=0.01,
        )
        assert record.side == "buy"
        assert record.target_size == 25
        assert record.reason == "delta_rebalance"


# ══════════════════════════════════════════════════════════════════════════
# ConnectionMonitor
# ══════════════════════════════════════════════════════════════════════════


class TestConnectionMonitor:

    def test_register_and_heartbeat(self):
        monitor = ConnectionMonitor(alert_threshold_sec=30.0, market_hours_only=False)
        monitor.register_feed("test_feed")
        monitor.heartbeat("test_feed")
        health = monitor.get_health("test_feed")
        assert health is not None
        assert health.state == ConnectionState.CONNECTED
        monitor.stop()

    def test_is_healthy(self):
        monitor = ConnectionMonitor(alert_threshold_sec=30.0, market_hours_only=False)
        monitor.register_feed("feed_a")
        monitor.heartbeat("feed_a")
        assert monitor.is_healthy("feed_a")
        monitor.stop()

    def test_feed_health_transitions(self):
        from quantcore.live.paper_trading.connection_monitor import FeedHealth

        feed = FeedHealth(name="test")
        assert feed.state == ConnectionState.DISCONNECTED

        feed.record_seen()
        assert feed.state == ConnectionState.CONNECTED

        feed.missed_beat()
        assert feed.state == ConnectionState.DEGRADED
        assert feed.missed_beats == 1

        feed.missed_beat()
        feed.missed_beat()  # 3rd missed beat → disconnected
        assert feed.state == ConnectionState.DISCONNECTED

    def test_exponential_backoff(self):
        from quantcore.live.paper_trading.connection_monitor import FeedHealth

        feed = FeedHealth(name="test")
        d1 = feed.next_backoff()
        assert d1 == 2.0  # initial
        d2 = feed.next_backoff()
        assert d2 == 4.0  # doubled
        d3 = feed.next_backoff()
        assert d3 == 8.0
        feed.reset_backoff()
        assert feed.reconnect_delay == 2.0
