#!/usr/bin/env python3
"""Launch script for Phase 3/4 live paper trading.

Usage::

    # With environment variables set:
    export ALPACA_API_KEY=PKXXXXXXXX
    export ALPACA_API_SECRET=xxxxxxxx
    export ALPACA_BASE_URL=https://paper-api.alpaca.markets   # default

    python run_paper_trading.py

    # Specify strategy (default: macro_cross_sectional_momentum):
    python run_paper_trading.py --strategy macro_state_selective_momentum

    # Custom initial cash:
    python run_paper_trading.py --cash 250000

    # Graduated scale modes
    python run_paper_trading.py --scale single_share     # 1 share per signal
    python run_paper_trading.py --scale fractional     # 10% notional sizing
    python run_paper_trading.py --scale full           # full strategy notional (default)

    # Check Alpaca connectivity without starting the trader:
    python run_paper_trading.py --check

    # FULL LIVE MODE (REAL MONEY AT RISK):
    python run_paper_trading.py --live --scale single_share

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Banner (printed immediately — before any import errors)
# ---------------------------------------------------------------------------
BANNER = r"""
╔═══════════════════════════════════════════════════════════════════╗
║         QUANTCORE PHASE 3/4 — PAPER TRADING LAUNCH                ║
║                                                                   ║
║   PAPER TRADING STARTED — no real money is at risk.               ║
║   All orders are routed to Alpaca PAPER BROKER.                   ║
║   Press Ctrl+C to stop gracefully.                                ║
╚═══════════════════════════════════════════════════════════════════╝
"""

LIVE_BANNER = r"""
╔═══════════════════════════════════════════════════════════════════╗
║  ⚠️  QUANTCORE — LIVE TRADING MODE  ⚠️                            ║
║                                                                   ║
║    REAL MONEY IS AT RISK.                                         ║
║    Orders route to Alpaca LIVE broker.                            ║
║    Press Ctrl+C to stop gracefully.                               ║
║    Touch KILL_SWITCH to emergency halt.                           ║
╚═══════════════════════════════════════════════════════════════════╝
"""

DEPLOY_STRATEGIES = {
    "macro_cross_sectional_momentum",
    "macro_state_selective_momentum",
}

DEFAULT_STRATEGY = "macro_cross_sectional_momentum"
DEFAULT_CASH = 100_000.0


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def _check_api_keys() -> tuple[str | None, str | None]:
    key    = os.environ.get("ALPACA_API_KEY",    "").strip()
    secret = os.environ.get("ALPACA_API_SECRET",  "").strip()
    if key and secret:
        return key, secret
    return None, None


def _check_connectivity(api_key: str, api_secret: str) -> bool:
    """Ping Alpaca account endpoint to verify credentials."""
    try:
        import urllib.request
        base = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        headers = {
            "APCA-API-KEY-ID":     api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        acct_url = base.rstrip("/") + "/v2/account"
        req = urllib.request.Request(acct_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return True
    except Exception as exc:
        logging.error("Alpaca connectivity check failed: %s", exc)
    return False


def _check(args: argparse.Namespace) -> int:
    """Check mode: verify credentials and connectivity only."""
    key, secret = _check_api_keys()
    if not key or not secret:
        print("ERROR: ALPACA_API_KEY and ALPACA_API_SECRET must be set.", file=sys.stderr)
        print("Set them as environment variables before running:", file=sys.stderr)
        print("  export ALPACA_API_KEY=PKXXXXXXXX", file=sys.stderr)
        print("  export ALPACA_API_SECRET=xxxxxxxx", file=sys.stderr)
        return 1

    print(f"API Key : {key[:8]}...", file=sys.stderr)
    print(f"Base URL: {os.environ.get('ALPACA_BASE_URL', '(default paper)')}", file=sys.stderr)
    print("Checking connectivity...", end=" ", file=sys.stderr)
    sys.stderr.flush()

    if _check_connectivity(key, secret):
        print("OK", file=sys.stderr)
        return 0
    else:
        print("FAILED", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    """Full launch: initialise all components and start the trader."""
    _setup_logging()

    # Validate strategy name
    strategy_name = args.strategy or DEFAULT_STRATEGY
    if strategy_name not in DEPLOY_STRATEGIES:
        print(
            f"ERROR: Unknown strategy {strategy_name!r}.  Valid DEPLOY strategies:",
            file=sys.stderr,
        )
        for s in sorted(DEPLOY_STRATEGIES):
            print(f"  - {s}", file=sys.stderr)
        return 1

    # Check API keys
    key, secret = _check_api_keys()
    if not key or not secret:
        print("ERROR: ALPACA_API_KEY and ALPACA_API_SECRET must be set.", file=sys.stderr)
        return 1

    # Verify connectivity
    print("Checking Alpaca connectivity...", end=" ", file=sys.stderr)
    sys.stderr.flush()
    if not _check_connectivity(key, secret):
        print("FAILED — check your API key/secret and try again.", file=sys.stderr)
        return 1
    print("OK", file=sys.stderr)

    # Live mode guard rails
    live_mode = bool(args.live)
    scale_mode = args.scale or "full"
    if live_mode:
        if scale_mode == "full":
            print(
                "\n⚠️  ERROR: --live with --scale full is prohibited.\n"
                "   Graduate: single_share → fractional → full (on paper first).\n"
                "   Use --scale single_share or --scale fractional with --live.\n",
                file=sys.stderr,
            )
            return 1
        # Enforce live base URL
        os.environ["ALPACA_BASE_URL"] = "https://api.alpaca.markets"
        print(LIVE_BANNER, file=sys.stderr)
    else:
        print(BANNER, file=sys.stderr)

    # Instantiate components via factory
    from quantcore.live.orchestrator import make_paper_trader, ScaleMode

    initial_cash = float(args.cash) if args.cash else DEFAULT_CASH
    tickers = args.tickers.split(",") if args.tickers else None

    trader = make_paper_trader(
        strategy_name=strategy_name,
        tickers=tickers,           # None = defaults to 6-asset macro universe
        initial_cash=initial_cash,
        scale_mode=scale_mode,
    )

    mode_label = "LIVE" if live_mode else "PAPER"
    print(
        f"Mode      : {mode_label}",
        f"Scale     : {scale_mode}",
        f"Strategy  : {strategy_name}",
        f"Assets    : {trader.feed.tickers}",
        f"Initial $ : {initial_cash:,.0f}",
        f"Events    : {trader.events_path}",
        f"Equity    : {trader.equity_path}",
        sep="\n",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    try:
        trader.run()   # blocks
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down.", file=sys.stderr)
    finally:
        trader.stop()

    return 0


def _run_once(args: argparse.Namespace) -> int:
    """One-shot rehearsal: snapshot, signals, orders, summary, then exit.

    Same validation and component wiring as _run() but executes a single
    trading tick instead of entering the continuous schedule loop.
    """
    _setup_logging()

    # Validate strategy name
    strategy_name = args.strategy or DEFAULT_STRATEGY
    if strategy_name not in DEPLOY_STRATEGIES:
        print(
            f"ERROR: Unknown strategy {strategy_name!r}.  Valid DEPLOY strategies:",
            file=sys.stderr,
        )
        for s in sorted(DEPLOY_STRATEGIES):
            print(f"  - {s}", file=sys.stderr)
        return 1

    # Check API keys
    key, secret = _check_api_keys()
    if not key or not secret:
        print("ERROR: ALPACA_API_KEY and ALPACA_API_SECRET must be set.", file=sys.stderr)
        return 1

    # Verify connectivity
    print("Checking Alpaca connectivity...", end=" ", file=sys.stderr)
    sys.stderr.flush()
    if not _check_connectivity(key, secret):
        print("FAILED — check your API key/secret and try again.", file=sys.stderr)
        return 1
    print("OK", file=sys.stderr)

    scale_mode = args.scale or "full"

    # Instantiate components via factory
    from quantcore.live.orchestrator import make_paper_trader, ScaleMode

    initial_cash = float(args.cash) if args.cash else DEFAULT_CASH
    tickers = args.tickers.split(",") if args.tickers else None

    trader = make_paper_trader(
        strategy_name=strategy_name,
        tickers=tickers,
        initial_cash=initial_cash,
        scale_mode=scale_mode,
    )

    print(
        f"REHEARSAL MODE (--once)",
        f"Scale     : {scale_mode}",
        f"Strategy  : {strategy_name}",
        f"Assets    : {trader.feed.tickers}",
        f"Initial $ : {initial_cash:,.0f}",
        sep="\n",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    # Determine the most recent completed trading day
    today = date.today()
    if trader.feed.is_market_open():
        trade_date = today
    else:
        trade_date = today - timedelta(days=1)
        # Walk back to a weekday (skip Sat/Sun)
        while trade_date.weekday() >= 5:
            trade_date -= timedelta(days=1)

    print(f"Trade date: {trade_date}", file=sys.stderr)
    print(f"Market open today: {trader.feed.is_market_open()}", file=sys.stderr)
    sys.stderr.flush()

    # Execute one full trading tick: snapshot → signals → risk → orders → log
    trader._trading_tick(trade_date)

    # Collect summary data from broker and risk manager
    acct = trader.broker.get_account()
    positions = trader.broker.get_positions()
    cash = float(acct.get("cash", 0))
    pos_value = sum(
        float(abs(p.quantity)) * float(p.market_price)
        for p in positions.values()
    )
    total_eq = cash + pos_value

    # Count paper order outcomes from the broker's internal records
    total_orders = len(trader.broker._paper_orders)
    filled = sum(
        1 for r in trader.broker._paper_orders.values()
        if r.status.value == "filled"
    )
    rejected = sum(
        1 for r in trader.broker._paper_orders.values()
        if r.status.value == "rejected"
    )
    pending = total_orders - filled - rejected

    # Read signal count from the event log (last tick's events)
    signals_generated = 0
    try:
        events_path = trader.events_path
        if Path(events_path).exists():
            with open(events_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ev = json.loads(line)
                        if ev.get("event") == "submit":
                            signals_generated += 1
    except Exception:
        pass

    # Summary banner
    print("", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("  REHEARSAL COMPLETE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Trade date      : {trade_date}", file=sys.stderr)
    print(f"  Strategy        : {strategy_name}", file=sys.stderr)
    print(f"  Signals generated : {signals_generated}", file=sys.stderr)
    print(f"  Orders submitted   : {total_orders}", file=sys.stderr)
    print(f"  Fills             : {filled}", file=sys.stderr)
    print(f"  Rejected          : {rejected}", file=sys.stderr)
    print(f"  Pending           : {pending}", file=sys.stderr)
    print(f"  Cash              : ${cash:,.2f}", file=sys.stderr)
    print(f"  Positions value   : ${pos_value:,.2f}", file=sys.stderr)
    print(f"  Total equity      : ${total_eq:,.2f}", file=sys.stderr)
    print(f"  P&L               : ${total_eq - initial_cash:+,.2f}", file=sys.stderr)
    print(f"  Circuit breaker   : {trader.circuit.state.value}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4 Paper/Live Trading Launch Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy", "-s",
        default=os.environ.get("QUANTCORE_STRATEGY", DEFAULT_STRATEGY),
        choices=sorted(DEPLOY_STRATEGIES),
        help="DEPLOY strategy to run (default: macro_cross_sectional_momentum)",
    )
    parser.add_argument(
        "--cash", "-c",
        default=os.environ.get("QUANTCORE_INITIAL_CASH"),
        help="Initial paper account cash in dollars (default: 100000)",
    )
    parser.add_argument(
        "--tickers", "-t",
        default=os.environ.get("QUANTCORE_TICKERS"),
        help="Comma-separated ticker list (default: SVXY,SPXL,TMF,TLT,SPY,QQQ)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="One-shot rehearsal: snapshot, signals, orders, summary, exit",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Only verify Alpaca API connectivity and exit",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Switch to LIVE broker (REAL MONEY). Requires --scale single_share or fractional.\n"
             "Paper remains default.",
    )
    parser.add_argument(
        "--scale",
        choices=["single_share", "fractional", "full"],
        default=os.environ.get("QUANTCORE_SCALE", "full"),
        help="Notional scaling: single_share (1/share), fractional (10%%), full (100%%). Default: full",
    )
    args = parser.parse_args()

    if args.check:
        return _check(args)
    if args.once:
        return _run_once(args)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
