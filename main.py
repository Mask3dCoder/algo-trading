#!/usr/bin/env python
"""QuantCore — Institutional-grade algorithmic trading system.

Entry point for CLI, backtesting, live trading, and API server.

Usage:
    python main.py backtest --strategy momentum --data data.csv
    python main.py live --broker paper
    python main.py api --host 0.0.0.0 --port 8000
    python main.py report --input results.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run a backtest from CLI."""
    from quantcore.backtest.engine import BacktestEngine, PortfolioState
    from quantcore.backtest.analytics import generate_performance_report, generate_html_report
    from quantcore.risk.limits import RiskLimits

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: data file not found: {data_path}")
        return 1

    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} rows from {data_path}")

    # Build a simple moving-average crossover strategy
    class MACrossoverStrategy:
        def __init__(self, symbol: str, fast: int = 20, slow: int = 50):
            self.symbol = symbol
            self.fast = fast
            self.slow = slow
            self._prices: list[float] = []

        def on_bar(self, bar, portfolio):
            from quantcore.backtest.engine import SignalData
            price = float(bar.close)
            self._prices.append(price)
            signals = []
            if len(self._prices) >= self.slow:
                fast_ma = np.mean(self._prices[-self.fast:])
                slow_ma = np.mean(self._prices[-self.slow:])
                if fast_ma > slow_ma and portfolio.cash > 0:
                    signals.append(SignalData(
                        symbol=self.symbol, direction="long",
                        strength=abs(fast_ma - slow_ma) / slow_ma,
                        strategy_id="ma_cross",
                    ))
                elif fast_ma < slow_ma:
                    signals.append(SignalData(
                        symbol=self.symbol, direction="short",
                        strength=abs(fast_ma - slow_ma) / slow_ma,
                        strategy_id="ma_cross",
                    ))
            return signals

        def on_fill(self, fill, portfolio):
            pass

    symbol = data["symbol"].iloc[0] if "symbol" in data.columns else "DEFAULT"
    strategy = MACrossoverStrategy(symbol=symbol)

    # Risk limits
    risk_limits = None
    if args.risk_limits:
        risk_limits = RiskLimits(
            max_concentration_pct=args.max_concentration,
            max_dollar_exposure_per_name=args.max_exposure,
        )

    # Run backtest
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=args.cash,
        slippage_bps=args.slippage,
        commission_per_share=args.commission,
        risk_limits=risk_limits,
        sizing_pct=args.sizing_pct,
    )

    result = engine.run(data)
    print(f"Backtest complete: {len(result)} bars")

    # Generate report
    report = engine.summary()
    print(f"\n=== Performance Summary ===")
    print(f"Total Return:  {report.total_return:.2%}")
    print(f"Annual Return: {report.annual_return:.2%}")
    print(f"Sharpe Ratio:  {report.sharpe:.2f}")
    print(f"Sortino Ratio: {report.sortino:.2f}")
    print(f"Max Drawdown:  {report.max_drawdown:.2%}")
    print(f"Win Rate:      {report.win_rate:.2%}")
    print(f"N Trades:      {report.n_trades}")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nEquity curve saved to {args.output}")

    if args.html:
        html = generate_html_report(report)
        html_path = Path(args.html)
        html_path.write_text(html)
        print(f"HTML report saved to {html_path}")

    if engine._rejected_orders:
        print(f"\nRisk-rejected orders: {len(engine._rejected_orders)}")

    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Start live trading (paper or live)."""
    from quantcore.live import LiveRunner, LiveConfig
    from quantcore.risk.limits import RiskLimits
    from quantcore.risk.drawdown import DrawdownConfig, DrawdownAction
    from quantcore.portfolio.rebalance import RebalanceConfig, RebalanceTrigger

    config = LiveConfig(
        heartbeat_interval_sec=args.heartbeat,
        max_daily_loss_pct=args.max_daily_loss,
        max_position_pct=args.max_position_pct,
        max_leverage=args.max_leverage,
        risk_limits=RiskLimits(),
        drawdown=DrawdownConfig(),
        rebalance=RebalanceConfig(trigger=RebalanceTrigger.SCHEDULE) if args.rebalance else None,
    )

    runner = LiveRunner(config=config)

    def on_state_change(old, new):
        print(f"[{datetime.utcnow().isoformat()}] State: {old.value} → {new.value}")

    def on_dd_action(action, state):
        print(f"[ALERT] Drawdown action: {action.value} at {state.drawdown_pct:.1f}%")

    runner.on_state_change = on_state_change
    runner.on_drawdown_action = on_dd_action

    print(f"Starting live trading (broker={args.broker})...")
    runner.start()

    try:
        import time
        while runner.state.value != "stopped":
            snapshot = runner.tick()
            if args.verbose:
                print(f"  Portfolio: ${snapshot.portfolio_value:,.0f} | "
                      f"DD: {snapshot.drawdown.drawdown_pct:.1f}% | "
                      f"Regime: {snapshot.regime}")
            time.sleep(config.heartbeat_interval_sec)
    except KeyboardInterrupt:
        print("\nShutting down...")
        runner.stop()

    print("Live trading stopped.")
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    """Start the REST API server."""
    try:
        import uvicorn
        from fastapi import FastAPI
        from quantcore.api.routes import (
            risk_router, portfolio_router, execution_router, regime_router,
        )

        app = FastAPI(
            title="QuantCore API",
            description="Institutional-grade algorithmic trading system API",
            version="0.1.0",
        )

        app.include_router(risk_router)
        app.include_router(portfolio_router)
        app.include_router(execution_router)
        app.include_router(regime_router)

        @app.get("/")
        def root():
            return {
                "service": "QuantCore API",
                "version": "0.1.0",
                "docs": "/docs",
                "health": "/health",
            }

        @app.get("/health")
        def health():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

        print(f"Starting QuantCore API on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)

    except ImportError as e:
        print(f"Error: Missing dependency — {e}")
        print("Install API dependencies: pip install fastapi uvicorn")
        return 1

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate a performance report from a CSV of returns or equity curve."""
    from quantcore.backtest.analytics import (
        generate_performance_report, generate_html_report, compute_returns,
    )

    data = pd.read_csv(args.input)

    if "equity" in data.columns:
        equity = data["equity"].values
        returns = compute_returns(equity)
    elif "returns" in data.columns:
        returns = data["returns"].values
        equity = np.cumprod(1 + returns) * (args.initial_value or 1_000_000)
    else:
        print("Error: CSV must have 'equity' or 'returns' column")
        return 1

    report = generate_performance_report(equity, returns)
    print(f"\n=== Performance Report ===")
    print(f"Total Return:  {report.total_return:.2%}")
    print(f"Annual Return: {report.annual_return:.2%}")
    print(f"Sharpe Ratio:  {report.sharpe:.2f}")
    print(f"Sortino Ratio: {report.sortino:.2f}")
    print(f"Calmar Ratio:  {report.calmar:.2f}")
    print(f"Omega Ratio:   {report.omega:.2f}")
    print(f"Max Drawdown:  {report.max_drawdown:.2%}")
    print(f"Win Rate:      {report.win_rate:.2%}")
    print(f"Profit Factor: {report.profit_factor:.2f}")

    if args.html:
        html = generate_html_report(report)
        Path(args.html).write_text(html)
        print(f"\nHTML report saved to {args.html}")

    return 0


def cmd_risk(args: argparse.Namespace) -> int:
    """Run risk analysis on a returns series."""
    from quantcore.risk.var import historical_var, parametric_var, cvar
    from quantcore.risk.metrics import compute_all_metrics

    data = pd.read_csv(args.input)
    if "returns" not in data.columns:
        print("Error: CSV must have 'returns' column")
        return 1

    r = data["returns"].values.astype(float)

    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))

    print("\n=== VaR Analysis ===")
    for conf in [0.95, 0.975, 0.99]:
        hv = historical_var(r, confidence=conf)
        pv = parametric_var(mu, sigma, confidence=conf)
        cv = cvar(r, confidence=conf)
        print(f"  VaR {conf:.0%}: Historical={-hv.var:.4f}, Parametric={-pv.var:.4f}, CVaR={-cv:.4f}")

    print("\n=== Risk Metrics ===")
    metrics = compute_all_metrics(r)
    for key in ["sharpe", "sortino", "calmar", "omega", "max_drawdown", "skewness", "kurtosis"]:
        val = metrics.get(key, 0)
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="QuantCore — Institutional-grade algorithmic trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ----- backtest -----
    bt = sub.add_parser("backtest", help="Run a backtest")
    bt.add_argument("--data", "-d", required=True, help="Path to CSV data file")
    bt.add_argument("--cash", type=float, default=1_000_000, help="Initial cash (default: 1,000,000)")
    bt.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps (default: 5)")
    bt.add_argument("--commission", type=float, default=0.005, help="Commission per share (default: 0.005)")
    bt.add_argument("--sizing-pct", type=float, default=0.10, help="Position size as fraction of cash (default: 0.10)")
    bt.add_argument("--risk-limits", action="store_true", help="Enable pre-trade risk limit checks")
    bt.add_argument("--max-concentration", type=float, default=0.25, help="Max position concentration")
    bt.add_argument("--max-exposure", type=float, default=1_000_000, help="Max dollar exposure per name")
    bt.add_argument("--output", "-o", help="Save equity curve CSV")
    bt.add_argument("--html", help="Save HTML performance report")

    # ----- live -----
    live = sub.add_parser("live", help="Start live trading")
    live.add_argument("--broker", default="paper", choices=("paper", "alpaca", "ibkr"), help="Broker (default: paper)")
    live.add_argument("--heartbeat", type=float, default=1.0, help="Heartbeat interval in seconds")
    live.add_argument("--max-daily-loss", type=float, default=0.03, help="Max daily loss fraction (default: 0.03)")
    live.add_argument("--max-position-pct", type=float, default=0.10, help="Max position size fraction")
    live.add_argument("--max-leverage", type=float, default=1.0, help="Max leverage")
    live.add_argument("--rebalance", action="store_true", help="Enable scheduled rebalancing")
    live.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # ----- api -----
    api = sub.add_parser("api", help="Start the REST API server")
    api.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    api.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    api.add_argument("--log-level", default="info", choices=("debug", "info", "warning", "error"))

    # ----- report -----
    rep = sub.add_parser("report", help="Generate performance report from CSV")
    rep.add_argument("--input", "-i", required=True, help="CSV with 'equity' or 'returns' column")
    rep.add_argument("--initial-value", type=float, help="Initial portfolio value (for returns input)")
    rep.add_argument("--html", help="Save HTML report to file")

    # ----- risk -----
    risk = sub.add_parser("risk", help="Run risk analysis on returns")
    risk.add_argument("--input", "-i", required=True, help="CSV with 'returns' column")

    args = parser.parse_args()

    if args.command == "backtest":
        return cmd_backtest(args)
    elif args.command == "live":
        return cmd_live(args)
    elif args.command == "api":
        return cmd_api(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "risk":
        return cmd_risk(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
