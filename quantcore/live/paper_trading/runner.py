"""Delta Neutral VRP Paper Trading Runner.

Wires together the three paper_trading modules with the live data feed and
strategy registry for a complete daily trading loop.

Components
---------
    RiskController         : hard kill switches (maxDD, Sharpe halt, macro exit)
    PaperExecutionEnvironment : zero look-ahead IV + slippage + delta tracking
    PaperTradingAnalytics  : daily P&L logging + drift monitoring
    LiveDataFeed           : Alpaca IEX market data
    strategy               : from registry (options_delta_neutral)
    CircuitBreaker         : production-grade breaker propagated to production

Daily loop (runs at configured trading_time, default 16:15 ET Mon–Fri):

    1. Fetch daily snapshot with VIX
    2. Ingest bar to rolling IV window (strictly historical — no look-ahead)
    3. Run strategy.on_data() → TradeSignal list
    4. RiskController pre-trade check (active? halted?)
    5. PaperExecutionEnvironment.compute_order_details() for each signal → slippage
    6. Record fills: update equity, track positions
    7. PaperTradingAnalytics.record_daily_result() → CSV row + alert evaluation
    8. Record delta hedge via PaperExecutionEnvironment
    9. End-of-day RiskController equity update + monthly Sharpe check

Usage
-----
    # Module invocation (primary)
    python -m quantcore.live.paper_trading.runner

    # Within Python
    from quantcore.live.paper_trading.runner import run_paper_trading
    run_paper_trading()   # blocks — Ctrl+C to stop
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

# schedule — pure stdlib-compatible scheduling (pip install schedule)
try:
    import schedule
except ImportError:  # pragma: no cover — defensive; condaforge makes it available
    raise ImportError("schedule module required: pip install schedule") from None

from quantcore.core.data_validation import validate_live_snapshot
from quantcore.core.types import Side
from quantcore.live.circuit_breaker import CircuitBreaker
from quantcore.live.data_feed import LiveDataFeed
from quantcore.live.execution_state import ExecutionEvent, ExecutionEventState
from quantcore.live.paper_trading.adaptive_hedger import (
    AdaptiveHedger,
    AdaptiveHedgerConfig,
)
from quantcore.live.paper_trading.analytics_dashboard import (
    AnalyticsConfig,
    PaperTradingAnalytics,
)
from quantcore.live.paper_trading.connection_monitor import (
    ConnectionMonitor,
)
from quantcore.live.paper_trading.delta_telemetry import (
    DeltaTelemetryConfig,
    DeltaTelemetryMiddleware,
)
from quantcore.live.paper_trading.execution_environment import (
    PaperExecutionConfig,
    PaperExecutionEnvironment,
)
from quantcore.live.paper_trading.live_connector import (
    PassiveExecutionProxy,
)
from quantcore.live.paper_trading.position_sizer import (
    PositionSizer,
    PositionSizerConfig,
)
from quantcore.live.paper_trading.risk_controller import (
    RiskController,
    RiskControllerConfig,
)
from quantcore.live.paper_trading.tca_engine import TCAConfig, TCAEngine
from quantcore.live.paper_trading.terminal_dashboard import TerminalDashboard
from quantcore.strategies.base import Strategy, StrategyConfig
from quantcore.strategies.registry import create as create_strategy

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_PAPER_TRADING_CONFIG_ENV = "QUANTCORE_PAPER_CONFIG"
_DEFAULT_CONFIG_PATH = Path("config/paper_trading.yaml")


def _load_config(config_path: Path | None = None) -> dict:
    """Load paper trading configuration from YAML."""
    path = config_path or Path(os.environ.get(_PAPER_TRADING_CONFIG_ENV, _DEFAULT_CONFIG_PATH))
    if not path.exists():
        log.warning(
            "Paper trading config not found at %s — using environment-variable overrides "
            "and hardcoded defaults.", path,
        )
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Scale mode (per Investment Committee mandate)
# ---------------------------------------------------------------------------

class ScaleMode(str, Enum):
    """Graduated scale of live deployment — mandated by the Investment Committee."""
    SINGLE_SHARE = "single_share"       # 1 share per signal (minimum viable, debug)
    FRACTIONAL = "fractional"           # 10% notional sizing
    CAPITAL_ALLOCATED = "capital_allocated"  # PositionSizer with premium-based allocation


# ---------------------------------------------------------------------------
# Kill-switch integration
# ---------------------------------------------------------------------------

KILL_SWITCH_PATH = Path(os.environ.get("QUANTCORE_KILL_SWITCH", "PAPER_KILL_SWITCH"))


def _check_kill_switch() -> bool:
    """Return True if kill switch file exists (operator-initiated stop)."""
    return KILL_SWITCH_PATH.exists()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


@dataclass
class PaperTradingConfig:
    """Application-level config — assembled in main(), consumed by PaperTradingRunner."""
    strategy_name: str = "options_delta_neutral"
    scale_mode: ScaleMode = ScaleMode.SINGLE_SHARE
    initial_capital: float = 100_000.0
    trading_time: str = "16:15"
    reconciliation_time: str = "16:30"
    timezone: str = "US/Eastern"
    initial_cash: float = 100_000.0

    # Risk controller thresholds (hardcoded per Investment Committee mandate)
    max_drawdown_pct: float = 0.10      # 10% hard kill
    sharpe_min_threshold: float = 0.0  # halt if Sharpe < 0
    macro_vix_threshold: float = 30.0
    macro_drawdown_kill_pct: float = 0.12
    sharpe_lookback_days: int = 63

    # Slippage model
    slippage_spread_bps: float = 5.0
    options_slippage_bps: float = 50.0

    # Passive tracking (Silent Run)
    passive_tracking_mode: bool = True

    # Benchmark (walk-forward OOS from per-year decomposition)
    expected_sharpe_annual: float = 1.30
    expected_annual_return: float = 0.12
    expected_max_dd: float = 0.1543
    expected_win_rate: float = 0.867

    # Sizing (capital-allocated sizing replaces single_share in production)
    max_premium_allocation_pct: float = 0.05   # 5% of equity per trade cycle
    max_portfolio_premium_pct: float = 0.20    # 20% across all open positions
    max_contracts_per_trade: int = 10
    max_total_contracts: int = 50

    # VIX entry threshold (from delta_neutral alpha params)
    ivr_entry: int = 50
    ivr_exit: int = 25
    dte_min: int = 30
    dte_max: int = 60

    @classmethod
    def from_dict(cls, data: dict) -> PaperTradingConfig:
        """Build from a YAML-loaded dict."""
        strategies = data.get("strategies", [])
        strat_cfg = next(
            (s for s in strategies if s.get("name") == "options_delta_neutral"),
            {},
        )
        params = strat_cfg.get("params", {})
        risk_cfg = data.get("risk", {})

        # Scale mode
        scale_raw = data.get("execution", {}).get("scale_mode", "single_share")
        try:
            scale = ScaleMode(scale_raw)
        except ValueError:
            scale = ScaleMode.SINGLE_SHARE

        return cls(
            strategy_name=strat_cfg.get("name", "options_delta_neutral"),
            scale_mode=scale,
            initial_capital=data.get("execution", {}).get("initial_cash", 100_000.0),
            trading_time=data.get("schedule", {}).get("trading_time", "16:15"),
            reconciliation_time=data.get("schedule", {}).get("reconciliation_time", "16:30"),
            timezone=data.get("schedule", {}).get("timezone", "US/Eastern"),
            initial_cash=data.get("execution", {}).get("initial_cash", 100_000.0),
            max_drawdown_pct=risk_cfg.get("max_drawdown_pct", 0.10),
            macro_vix_threshold=data.get("risk", {}).get("macro_vix_threshold", 30.0),
            macro_drawdown_kill_pct=risk_cfg.get("macro_drawdown_kill_pct", 0.12),
            sharpe_lookback_days=risk_cfg.get("sharpe_lookback_days", 63),
            slippage_spread_bps=data.get("execution", {}).get("slippage_bps", 5.0),
            options_slippage_bps=data.get("execution", {}).get("options_slippage_bps", 50.0),
            expected_sharpe_annual=data.get("benchmark", {}).get("expected_sharpe_annual", 1.30),
            expected_annual_return=data.get("benchmark", {}).get("expected_annual_return", 0.12),
            expected_max_dd=data.get("benchmark", {}).get("expected_max_dd", 0.1543),
            expected_win_rate=data.get("benchmark", {}).get("expected_win_rate", 0.867),
            passive_tracking_mode=data.get("execution", {}).get(
                "passive_tracking_mode", True,
            ),
            ivr_entry=params.get("ivr_entry", 50),
            ivr_exit=params.get("ivr_exit", 25),
            dte_min=params.get("dte_min", 30),
            dte_max=params.get("dte_max", 60),
            # Sizing (from config/paper_trading.yaml "sizing" section)
            max_premium_allocation_pct=data.get("sizing", {}).get(
                "max_premium_allocation_pct", 0.05,
            ),
            max_portfolio_premium_pct=data.get("sizing", {}).get(
                "max_portfolio_premium_pct", 0.20,
            ),
            max_contracts_per_trade=data.get("sizing", {}).get(
                "max_contracts_per_trade", 10,
            ),
            max_total_contracts=data.get("sizing", {}).get(
                "max_total_contracts", 50,
            ),
        )


class PaperTradingRunner:
    """End-to-end paper trading runner for delta_neutral_vrp.

    Combines RiskController, PaperExecutionEnvironment, and PaperTradingAnalytics
    into a single daily loop. Designed to run inside Docker in parallel with the
    production runner (separate process, separate risk state).

    Parameters
    ----------
    config : PaperTradingConfig
        Application configuration.
    data_feed : LiveDataFeed | None
        Injected for testing. Defaults to LiveDataFeed with delta-neutral universe.
    """

    def __init__(
        self,
        config: PaperTradingConfig,
        data_feed: LiveDataFeed | None = None,
    ) -> None:
        self.cfg = config
        self._data_feed = data_feed or LiveDataFeed(
            tickers=["SPY", "QQQ", "IWM", "GLD", "VIX"],
        )

        # ── Production circuit breaker (for kill-switch propagation) ──
        self._circuit_breaker = CircuitBreaker()

        # ── RiskController (hardcoded kill switches) ────────────────────
        risk_cfg = RiskControllerConfig(
            max_drawdown_pct=config.max_drawdown_pct,
            sharpe_lookback_days=config.sharpe_lookback_days,
            sharpe_min_threshold=config.sharpe_min_threshold,
            macro_vix_threshold=config.macro_vix_threshold,
            macro_drawdown_kill_pct=config.macro_drawdown_kill_pct,
        )
        self._risk = RiskController(
            config=risk_cfg,
            circuit_breaker=self._circuit_breaker,
            initial_equity=config.initial_capital,
        )

        # ── Execution environment (zero look-ahead IV, slippage, delta) ─
        exec_cfg = PaperExecutionConfig(
            initial_paper_capital=config.initial_capital,
            slippage_spread_bps=config.slippage_spread_bps,
            options_slippage_bps=config.options_slippage_bps,
        )
        self._exec = PaperExecutionEnvironment(config=exec_cfg)

        # ── PositionSizer (capital-allocated sizing) ────────────────────
        sizer_cfg = PositionSizerConfig(
            max_premium_allocation_pct=config.max_premium_allocation_pct,
            max_portfolio_premium_pct=config.max_portfolio_premium_pct,
            max_contracts_per_trade=config.max_contracts_per_trade,
            max_total_contracts=config.max_total_contracts,
        )
        self._position_sizer = PositionSizer(
            equity=config.initial_capital,
            config=sizer_cfg,
        )

        # ── Delta telemetry (real-time delta monitoring) ──────────────
        delta_telemetry_cfg = DeltaTelemetryConfig(
            output_dir=Path("reports/paper_trading"),
            log_interval_seconds=300,  # 5 min
        )
        self._delta_telemetry = DeltaTelemetryMiddleware(config=delta_telemetry_cfg)

        # ── Analytics (daily logging + drift monitoring) ─────────────────
        ana_cfg = AnalyticsConfig(
            expected_sharpe_annual=config.expected_sharpe_annual,
            expected_annual_return=config.expected_annual_return,
            expected_max_dd=config.expected_max_dd,
            expected_win_rate=config.expected_win_rate,
        )
        self._analytics = PaperTradingAnalytics(config=ana_cfg)

        # ── TCA Engine (active execution quality monitoring) ─────────────
        self._tca = TCAEngine(config=TCAConfig())

        # ── Adaptive Hedger (volatility-adjusted delta threshold) ─────────
        self._hedger = AdaptiveHedger(
            underlying="SPY",
            config=AdaptiveHedgerConfig(),
            on_signal=None,  # runner handles hedge routing; set via .on_hedge_signal()
        )

        # ── Terminal Dashboard (display-only, non-blocking) ──────────────
        # Only start display dashboard in active (non-passive) mode
        self._dashboard: TerminalDashboard | None = None
        if not config.passive_tracking_mode:
            self._dashboard = TerminalDashboard(
                mode="ansi",
                refresh_interval_sec=1.0,
                starting_capital=config.initial_capital,
                risk_controller=self._risk,
                tca_engine=self._tca,
                hedger=self._hedger,
            )

        # ── Passive Tracking (Silent Run) ───────────────────────────────
        if config.passive_tracking_mode:
            log.info("PASSIVE TRACKING MODE enabled — orders will be logged, not sent.")
            self._passive_proxy = PassiveExecutionProxy()
        else:
            self._passive_proxy = None

        # ── Connection Monitor (heartbeat + reconnect) ────────────────────
        self._conn_monitor = ConnectionMonitor(alert_threshold_sec=30.0)
        self._conn_monitor.register_feed("paper_data_feed")
        self._conn_monitor.start()

        # ── Strategy (from registry) ────────────────────────────────────
        strategy_config = StrategyConfig(
            name=config.strategy_name,
            strategy_type="OPTIONS",
            max_position_notional=None,
            params={
                "ivr_entry": config.ivr_entry,
                "ivr_exit": config.ivr_exit,
                "dte_min": config.dte_min,
                "dte_max": config.dte_max,
            },
        )
        self._strategy: Strategy = create_strategy(
            config.strategy_name, strategy_config
        )
        self._strategy.initialize()  # type:ignore[union-attr]

        # ── Equity tracking ──────────────────────────────────────────────
        self._current_equity = config.initial_capital
        self._today_pnl = 0.0

        log.info(
            "PaperTradingRunner: strategy=%s scale=%s capital=$%.2f",
            config.strategy_name,
            config.scale_mode.value,
            config.initial_capital,
        )

    # ------------------------------------------------------------------
    # Properties — useful for monitoring / health endpoints
    # ------------------------------------------------------------------

    @property
    def risk_controller(self) -> RiskController:
        return self._risk

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    @property
    def execution_environment(self) -> PaperExecutionEnvironment:
        return self._exec

    @property
    def analytics(self) -> PaperTradingAnalytics:
        return self._analytics

    @property
    def position_sizer(self) -> PositionSizer:
        return self._position_sizer

    @property
    def delta_telemetry(self) -> DeltaTelemetryMiddleware:
        return self._delta_telemetry

    @property
    def tca(self) -> TCAEngine:
        return self._tca

    @property
    def hedger(self) -> AdaptiveHedger:
        return self._hedger

    @property
    def dashboard(self) -> TerminalDashboard | None:
        return self._dashboard

    @property
    def current_equity(self) -> float:
        return self._current_equity

    @property
    def is_active(self) -> bool:
        return self._risk.is_active

    # ------------------------------------------------------------------
    # Public API — called by scheduling loop
    # ------------------------------------------------------------------

    def run_once(self, as_of: date | None = None) -> dict[str, object]:
        """Execute one full trading day. Called by the scheduler."""
        as_of = as_of or date.today()
        log.info("=== Paper trading day %s ===", as_of)

        # 0. Operator kill switch
        if _check_kill_switch():
            log.critical("Kill switch file detected — halting.")
            self._risk.manual_reset("operator", "kill_switch_file_detected")
            return {"status": "killed", "date": str(as_of)}

        # 1. State check
        if not self._risk.is_active:
            result = self._risk.get_summary()
            log.warning("RiskController not active — skipping day. State: %s", result["state"])
            return {"status": "risk_inactive", "date": str(as_of), "risk_summary": result}

        # 2. Fetch market snapshot
        try:
            snapshot = self._data_feed.get_daily_snapshot(as_of)
            validate_live_snapshot(snapshot)
        except Exception as exc:
            log.error("Data feed failed for %s: %s — skipping day", as_of, exc)
            log.warning("Data feed failure for %s: %s — skipping day", as_of, exc)
            return {"status": "data_error", "date": str(as_of), "error": str(exc)}

        # 3. Ingest bar to rolling IV window (zero look-ahead)
        close = snapshot.get("close")
        symbols = snapshot.get("symbols", [])
        spy_close: float = 0.0
        vix_value: float | None = None
        if close is not None and not isinstance(close, str):
            arr = np.asarray(close)
            if arr.ndim == 2 and arr.shape[0] > 0:
                try:
                    spy_idx = symbols.index("SPY") if "SPY" in symbols else 0
                    spy_close = float(arr[-1, spy_idx])
                except (IndexError, ValueError):
                    log.warning("Could not extract SPY close from snapshot — skipping day")
                    return {"status": "data_error", "date": str(as_of), "error": "missing_spy_close"}
                if "VIX" in symbols:
                    try:
                        vix_value = float(arr[-1, symbols.index("VIX")])
                    except (IndexError, ValueError):
                        pass
        if spy_close == 0.0 and not symbols:
            # Both close lookups failed (e.g., all Alpaca 401s)
            log.warning("No market data available for %s — skipping day", as_of)
            return {"status": "data_error", "date": str(as_of), "error": "no_market_data"}

        self._exec.ingest_daily_bar(as_of, spy_close, vix=vix_value)

        # 4. Mark-to-market revaluation BEFORE signal processing
        # This drives the equity curve — the cash register that was stuck at $100k
        sigma = self._exec.get_current_iv()
        mtm_pnl = self._exec.mark_to_market(as_of, spy_close, sigma)
        self._current_equity += mtm_pnl
        self._exec.update_equity(mtm_pnl)

        # 5. Strategy signal generation
        signals: list = []
        try:
            # LiveDataFeed returns {"close": ndarray, "symbols": [...]} compatible with on_data
            signals = self._strategy.on_data(snapshot)  # type: ignore[union-attr]
        except Exception as exc:
            log.error("Strategy signal generation failed: %s", exc)
            log.warning("Strategy signal error %s: %s", as_of, exc)
            return {"status": "signal_error", "date": str(as_of), "error": str(exc)}

        # 5. Pre-trade risk check
        macro_status = self._risk.check_macro_conditions(
            vix_level=vix_value if vix_value is not None else 0.0,
            current_drawdown=(self._current_equity - self._risk.peak_equity)
            / self._risk.peak_equity if self._risk.peak_equity > 0 else 0.0,
        )
        if macro_status == "halt":
            log.warning("Macro conditions halted — skipping signals.")
            return {"status": "macro_halted", "date": str(as_of)}

        # 6. Process signals → slippage-aware fills
        fills: list[dict] = []
        for sig in signals:
            fill = self._process_signal(sig, as_of)
            if fill:
                fills.append(fill)

        # 7. Update equity: MtM daily revaluation drives equity curve.
        # Fill-level P&L (realized_pnl) comes from signal execution costs only.
        realized_pnl = sum(f.get("pnl", 0.0) for f in fills)
        self._today_pnl = mtm_pnl

        # Record trade P&L with analytics (closed positions only)
        self._exec.update_equity(realized_pnl)

        # Record trade P&L with analytics
        for f in fills:
            if f.get("pnl", 0.0) != 0.0:
                self._analytics.record_trade(pnl=f["pnl"], trade_date=as_of)

        # Record fills with TCA Engine (synthetic ExecutionEvent for paper mode)
        self._record_fills_with_tca(fills, spy_close)

        # Compute position snapshot + net delta for downstream blocks
        # (Moved before dashboard/hedger to fix scope ordering bugs)
        net_delta = 0.0
        positions = self._get_position_snapshot()
        if positions:
            delta_record = self._exec.record_delta_hedge(as_of, positions, hedge_shares=0.0)
            net_delta = delta_record.get("net_delta", 0.0)

        # Feed underlying tick to AdaptiveHedger (vol-adjusted delta threshold)
        if spy_close > 0:
            self._hedger.on_underlying_tick(
                price=spy_close,
                bid=spy_close * (1 - self.cfg.slippage_spread_bps / 10000),
                ask=spy_close * (1 + self.cfg.slippage_spread_bps / 10000),
            )
            self._hedger.on_portfolio_delta_update(
                net_delta=net_delta,
                dollar_gamma=0.0,
            )
            hedge_signal = self._hedger.decide()
            if hedge_signal.is_hedge:
                log.info(
                    "HEDGE SIGNAL: %s %+d shares  reason=%s",
                    hedge_signal.action.value,
                    hedge_signal.shares * hedge_signal.direction,
                    hedge_signal.reason,
                )

        # Update dashboard with live state (non-blocking, fire-and-forget)
        if self._dashboard is not None:
            dd = (self._current_equity - self._risk.peak_equity) / self._risk.peak_equity \
                if self._risk.peak_equity > 0 else 0.0
            self._dashboard.update_portfolio(
                equity=self._current_equity,
                peak_equity=self._risk.peak_equity,
                current_drawdown_pct=dd,
                max_drawdown_pct=abs(min(dd, 0.0)),
                margin_used_pct=self._position_sizer.margin_utilization_pct,
                daily_pnl=realized_pnl,
                open_positions=len(positions),
                net_delta=net_delta,
            )
            self._dashboard.update_greeks(delta=net_delta, gamma=0.0, vega=0.0, theta=0.0)
            self._dashboard.sync_from_risk_controller()
            self._dashboard.sync_from_tca()
            self._dashboard.update_connection(
                status="CONNECTED"
                if self._conn_monitor.is_healthy("paper_data_feed")
                else "DISCONNECTED",
            )

        # Update PositionSizer equity after P&L
        self._position_sizer.update_equity(self._current_equity)

        # Update RiskController equity (triggers maxDD check)
        self._risk.update_equity(self._current_equity, as_of)

        # Record delta telemetry (real-time drift monitoring)
        if positions:
            self._delta_telemetry.on_positions_updated(
                dt=as_of,
                positions=positions,
                hedge_shares=0.0,
                equity=self._current_equity,
                vix_close=vix_value,
                force_log=True,
            )

        # Record daily result to analytics
        result = self._analytics.record_daily_result(
            dt=as_of,
            equity=self._current_equity,
            realized_pnl=realized_pnl,
            expected_pnl=self._expected_daily_pnl(),
            slippage_bps=self._exec.get_todays_slippage()["slippage_bps"],
            slippage_dollars=self._exec.get_todays_slippage()["slippage_dollars"],
            trades_executed=len(fills),
            open_positions=len(positions),
            net_delta=net_delta,
            vix_close=vix_value,
            macro_exit_triggered=macro_status in ("halt", "reduce_50"),
        )

        # Monthly Sharpe check (end/start of month detection)
        self._risk.record_daily_return(
            as_of,
            (realized_pnl / self._current_equity) if self._current_equity > 0 else 0.0,
        )
        if (as_of.month != (as_of - timedelta(days=1)).month) or (as_of.day <= 3):
            self._risk.check_monthly_sharpe(as_of)

        # Reset daily accumulators
        self._exec.reset_daily_slippage()
        self._today_pnl = 0.0

        result["status"] = "ok"
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_signal(self, signal: object, as_of: date) -> dict | None:
        """Apply slippage model and update the option ledger.

        TradeSignal metadata drives all ledger operations:
          - action=delta_neutral_entry  → open new short straddle in ledger
          - action=gamma_scalp         → rebalance hedge shares
          - action=delta_neutral_exit  → close positions in ledger
          - direction=±1 (no action)  → standalone equity/option order
        """
        try:
            if hasattr(signal, "__dataclass_fields__"):
                sig_dict = asdict(signal)
            elif hasattr(signal, "__iter__") and not isinstance(signal, str):
                sig_dict = dict(signal)
            else:
                return None
        except Exception:
            return None

        action = sig_dict.get("metadata", {}).get("action", "")
        sig_date = sig_dict.get("date", as_of)
        direction = sig_dict.get("direction", 0)
        size = sig_dict.get("size", 1)          # notional
        signal_type = sig_dict.get("signal_type", "option")
        asset_type = "option" if signal_type in ("option", "options") else "equity"

        # ── Enforce: no position opens while halted ────────────────────
        if not self._risk.is_active:
            return None

        # ── ENTRY: open short straddle in ledger ───────────────────────
        if action == "delta_neutral_entry":
            metadata = sig_dict.get("metadata", {})
            contracts = metadata.get("n_contracts", 1)
            # iv_rank and hedge_shares_signal are unused — removed

            # Current spot from execution environment
            spot = self._exec.prev_spot
            if spot <= 0:
                spot = self._exec.get_historical_iv(as_of, lookback=5) * 100  # rough fallback

            # Size position using PositionSizer (capital-allocated)
            sigma_est = self._exec.get_current_iv()
            T_rough = (self.cfg.dte_min + self.cfg.dte_max) / 2.0 / 252.0
            from quantcore.alpha.delta_neutral import _bsm_call, _bsm_delta, _bsm_put
            synthetic_premium = _bsm_call(spot, spot, T_rough, 0.05, sigma_est) * 2
            sizing = self._position_sizer.size_option_entry(
                premium=max(synthetic_premium, 0.50),
                underlying_price=spot,
            )
            contracts = max(contracts, sizing["contracts"])

            # Build expiry: midpoint of DTE range
            dte_avg = (self.cfg.dte_min + self.cfg.dte_max) // 2
            expiry = as_of + timedelta(days=dte_avg)

            # ATM strike, rounded to nearest 0.5
            strike = round(spot * 2) / 2

            # Credit = theoretical mid price from strategy (selling straddle = credit received)
            sigma = self._exec.get_current_iv()
            T_entry = dte_avg / 252.0
            call_prem = _bsm_call(strike, strike, T_entry, 0.05, sigma)
            put_prem = _bsm_put(strike, strike, T_entry, 0.05, sigma)
            credit = call_prem + put_prem
            # Apply entry slippage: sell at ~95% of mid (realistic bid/ask)
            credit *= 0.95

            # Initial delta hedge: buy shares to neutralise short straddle delta
            call_delta_init = _bsm_delta(strike, strike, T_entry, 0.05, sigma, True)
            put_delta_init = _bsm_delta(strike, strike, T_entry, 0.05, sigma, False)
            net_delta_per_contract = (-call_delta_init + -put_delta_init) * 100
            init_hedge = -net_delta_per_contract * contracts
            commission_rate = getattr(self._exec._cfg, 'options_commission_per_contract', 0.65)
            hedge_cost = abs(init_hedge) * (
                spot * (self.cfg.slippage_spread_bps / 10000.0) + commission_rate
            )

            # Add to ledger
            pos = self._exec.add_short_straddle(
                entry_date=as_of,
                expiry=expiry,
                call_strike=strike,
                put_strike=strike,
                credit=credit,
                contracts=contracts,
                entry_spot=spot,
                entry_iv=sigma,
                initial_hedge_shares=init_hedge,
                hedge_cost=hedge_cost,
            )
            log.info(
                "LEDGER ENTRY: short straddle K=%.1f credit=%.3f "
                "contracts=%d expiry=%s hedge_shares=%.0f",
                strike, credit, contracts, expiry, init_hedge,
            )
            # Return fill-like dict (ledger is source-of-truth for P&L)
            return {
                "date": str(sig_date),
                "direction": -1,
                "side": "sell",
                "quantity": contracts,
                "theoretical_price": credit,
                "fill_price": credit,
                "slippage_bps": 0.0,
                "slippage_dollars": 0.0,
                "commission": 0.65 * contracts * 2,
                "pnl": 0.0,
                "asset_type": "option",
                "ledger_pos": id(pos),
                "mode": "ledger_entry",
            }

        # ── EXIT: close all open positions in ledger ───────────────────
        if action == "delta_neutral_exit" or direction == 0:
            closed_count = 0
            close_reason_used = sig_dict.get("metadata", {}).get("reason", "signal_exit")
            for pos in self._exec.option_ledger:
                if not pos.closed:
                    pos.closed = True
                    pos.close_date = as_of
                    pos.close_reason = close_reason_used
                    closed_count += 1
            log.info(
                "LEDGER EXIT: closed %d position(s) reason=%s",
                closed_count, close_reason_used,
            )
            spot_exit = self._exec.prev_spot
            return {
                "date": str(sig_date),
                "direction": 0,
                "side": "buy",
                "quantity": closed_count,
                "theoretical_price": spot_exit,
                "fill_price": spot_exit,
                "slippage_bps": 0.0,
                "slippage_dollars": 0.0,
                "commission": 0.0,
                "pnl": 0.0,
                "asset_type": "option",
                "mode": "ledger_exit",
                "closed_count": closed_count,
            }

        # ── GAMMA SCALP / DELTA REBALANCE: update hedge shares ─────────
        if action == "gamma_scalp":
            spot_h = self._exec.prev_spot
            hedge_delta = sig_dict.get("metadata", {}).get("hedge_delta", 0.0)
            for pos in self._exec.option_ledger:
                if not pos.closed:
                    old_shares = pos.hedge_shares
                    pos.hedge_shares += hedge_delta
                    new_shares = pos.hedge_shares
                    # Update cumulative hedge cost with the new trade
                    hedge_trade_cost = abs(hedge_delta) * (
                        spot_h * (self.cfg.slippage_spread_bps / 10000.0)
                        + 0.005
                    )
                    pos.cumulative_hedge_cost += hedge_trade_cost
                    log.info(
                        "HEDGE REBALANCE: shares %.0f → %.0f delta=%.0f cost=%.2f",
                        old_shares, new_shares, hedge_delta, hedge_trade_cost,
                    )
            return {
                "date": str(sig_date),
                "direction": direction,
                "side": "buy" if hedge_delta > 0 else "sell",
                "quantity": abs(hedge_delta),
                "theoretical_price": spot_h if spot_h > 0 else sig_dict.get("price", 450.0),
                "fill_price": spot_h if spot_h > 0 else sig_dict.get("price", 450.0),
                "slippage_bps": 0.0,
                "slippage_dollars": 0.0,
                "commission": 0.0,
                "pnl": 0.0,
                "asset_type": "equity",
                "mode": "hedge_rebalance",
            }

        # ── STANDALONE ORDERS (non-strategy signal, e.g. direct hedge) ─
        price = sig_dict.get("price", 0.0)
        if price <= 0:
            return None

        side = "buy" if direction > 0 else "sell"

        if self._passive_proxy is not None:
            self._passive_proxy.log_bypassed_order(
                ticker=sig_dict.get("ticker", "SPY"),
                side=side,
                target_size=abs(size) if size else 1,
                mid_price=price,
                bid_ask_spread=0.02,
                contract=sig_dict.get("contract"),
                option_type=sig_dict.get("option_type"),
                notional=abs(size) * price if size else price,
                reason="standalone_signal",
            )
            return {
                "date": str(sig_date),
                "direction": direction,
                "side": side,
                "quantity": abs(size) if size else 1,
                "theoretical_price": price,
                "fill_price": price,
                "slippage_bps": 0.0,
                "slippage_dollars": 0.0,
                "commission": 0.0,
                "pnl": 0.0,
                "asset_type": asset_type,
                "mode": "passive",
            }

        # Capital-allocated sizing
        if self.cfg.scale_mode == ScaleMode.SINGLE_SHARE:
            quantity = 1
        elif self.cfg.scale_mode == ScaleMode.CAPITAL_ALLOCATED:
            sizing_result = self._position_sizer.size_option_entry(
                premium=price,
                underlying_price=self._exec._prev_spy_close if self._exec._prev_spy_close > 0 else price,
            )
            quantity = sizing_result["contracts"]
        else:
            notional = self._current_equity * 0.10
            quantity = max(1, int(notional / price / 100))

        fill_details = self._exec.compute_order_details(
            theoretical_price=price,
            quantity=quantity,
            side=side,
            asset_type=asset_type,
        )

        return {
            "date": str(sig_date),
            "direction": direction,
            "side": side,
            "quantity": quantity,
            "theoretical_price": fill_details["theoretical_price"],
            "fill_price": fill_details["fill_price"],
            "slippage_bps": fill_details["slippage_bps"],
            "slippage_dollars": fill_details["slippage_dollars"],
            "commission": fill_details["commission"],
            "pnl": 0.0,
            "asset_type": asset_type,
        }

    def _get_position_snapshot(self) -> list[dict]:
        """Return current open positions — sourced from the MtM option ledger."""
        return self._exec.get_position_snapshot()

    def _expected_daily_pnl(self) -> float | None:
        """Interpolated expected daily P&L from backtest benchmark.

        Uses annual return target / 252 as rough daily expectation.
        """
        if self._current_equity <= 0:
            return None
        ann_return = self.cfg.expected_annual_return
        daily_expected = ann_return / 252 * self._current_equity
        return daily_expected

    def _record_fills_with_tca(
        self,
        fills: list[dict],
        spy_close: float,
    ) -> None:
        """Record paper fills in the TCA Engine for execution quality analysis.

        Synthesizes ``ExecutionEvent`` objects from the paper fill dicts since
        there is no real broker adapter to emit them.  Thread-safe.
        """
        from decimal import Decimal

        for idx, f in enumerate(fills):
            if not f or f.get("mode") == "passive":
                # Passive mode: record benchmark only (no trade executed)
                self._tca.record_benchmark(
                    order_id=f"paper_{f.get('date', '')}_{idx}",
                    symbol="SPY",
                    side=f.get("side", "buy"),
                    asset_type=f.get("asset_type", "option"),
                    arrival_price=f.get("theoretical_price", spy_close),
                    mid_price=spy_close,
                    expected_spread_bps=self.cfg.slippage_spread_bps
                    if f.get("asset_type") == "equity"
                    else self.cfg.options_slippage_bps,
                )
                continue

            # Build synthetic ExecutionEvent from fill dict
            side_str = f.get("side", "buy")
            side = Side.BUY if side_str == "buy" else Side.SELL
            quantity = f.get("quantity", 0)
            fill_price = f.get("fill_price", spy_close)

            event = ExecutionEvent(
                broker_name="paper",
                internal_order_id=f"paper_{f.get('date', '')}_{idx}",
                state=ExecutionEventState.FILLED,
                symbol="SPY",
                side=side,
                quantity=Decimal(str(quantity)),
                filled_quantity=Decimal(str(quantity)),
                price=Decimal(str(fill_price)),
                commission=Decimal(str(f.get("commission", 0.0))),
            )

            self._tca.record_execution(
                event=event,
                arrival_price=f.get("theoretical_price", spy_close),
                mid_price=spy_close,
                is_sweep=False,
                notes="paper_trade",
            )

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _schedule_trading(self) -> None:
        """Register the daily jobs with schedule library (local timezone)."""
        tz = self.cfg.timezone

        schedule.every().Monday.at(self.cfg.trading_time).do(self._run_day_job).tag("paper-trading")
        schedule.every().Tuesday.at(self.cfg.trading_time).do(self._run_day_job).tag("paper-trading")
        schedule.every().Wednesday.at(self.cfg.trading_time).do(self._run_day_job).tag("paper-trading")
        schedule.every().Thursday.at(self.cfg.trading_time).do(self._run_day_job).tag("paper-trading")
        schedule.every().Friday.at(self.cfg.trading_time).do(self._run_day_job).tag("paper-trading")

        # Reconciliation pass 15 min after trading time
        rec_time = _add_minutes(self.cfg.trading_time, 15)
        schedule.every().Monday.at(rec_time).do(self._run_reconcile_job).tag("paper-reconcile")
        schedule.every().Tuesday.at(rec_time).do(self._run_reconcile_job).tag("paper-reconcile")
        schedule.every().Wednesday.at(rec_time).do(self._run_reconcile_job).tag("paper-reconcile")
        schedule.every().Thursday.at(rec_time).do(self._run_reconcile_job).tag("paper-reconcile")
        schedule.every().Friday.at(rec_time).do(self._run_reconcile_job).tag("paper-reconcile")

        log.info(
            "Scheduled paper trading at %s ET, reconciliation at %s ET",
            self.cfg.trading_time,
            rec_time,
        )

    def _run_day_job(self) -> None:
        self.run_once()

    def _run_reconcile_job(self) -> None:
        """End-of-day reconciliation pass."""
        log.info("Running end-of-day reconciliation for %s", date.today())
        # Sync positions from paper broker, update equity, check risk state
        summary = self._risk.get_summary()
        log.info(
            "Equity=$%.2f peak=$%.2f state=%s",
            self._current_equity,
            summary["peak_equity"],
            summary["state"],
        )

    def _shutdown(self) -> None:
        """Graceful shutdown — stop connection monitor and cleanup."""
        log.info("Shutting down PaperTradingRunner...")
        if hasattr(self, "_conn_monitor"):
            self._conn_monitor.stop()
        log.info("Shutdown complete.")

    def run_forever(self) -> None:
        """Run the scheduler loop forever (or until kill switch)."""
        log.info("Starting paper trading loop — press Ctrl+C to stop.")
        self._schedule_trading()
        atexit.register(self._shutdown)

        while True:
            if _check_kill_switch():
                log.critical("Kill switch detected in run loop — shutting down.")
                break
            schedule.run_pending()
            time.sleep(30)


def _add_minutes(time_str: str, minutes: int) -> str:
    """Add minutes to HH:MM string, returning new HH:MM."""
    h, m = map(int, time_str.split(":"))
    total = h * 60 + m + minutes
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def run_paper_trading(config_path: Path | None = None) -> None:
    """Load config and run the paper trading loop (blocking).

    Parameters
    ----------
    config_path : Path | None
        Path to config/paper_trading.yaml. Defaults to config/paper_trading.yaml.
    """
    data = _load_config(config_path)
    cfg = PaperTradingConfig.from_dict(data)

    runner = PaperTradingRunner(cfg)
    runner.run_forever()


def main() -> int:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Delta Neutral VRP Paper Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to paper_trading config YAML (default: config/paper_trading.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the daily loop once with today's data and exit",
    )
    args = parser.parse_args()

    # Quick dependency checks
    try:
        import schedule  # noqa: F401
    except ImportError:
        print("ERROR: schedule module not found. Install: pip install schedule", file=sys.stderr)
        return 1

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    data = _load_config(args.config)
    cfg = PaperTradingConfig.from_dict(data)
    runner = PaperTradingRunner(cfg)

    if args.dry_run:
        result = runner.run_once()
        print(json.dumps(result, indent=2, default=str))
        return 0

    log.info("Starting delta_neutral_vrp paper trading runner...")
    runner.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
