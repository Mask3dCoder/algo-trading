#!/usr/bin/env python
"""QuantCore — Alpaca Connection Diagnostic.

Run before every paper or live trading session to verify all systems are operational.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime

from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = ["ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"]
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(label: str, fn) -> bool:
    """Run a diagnostic check. Returns True on success, False on failure."""
    try:
        result = fn()
        print(f"  {PASS} {label}: {result}")
        return True
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        return False


def main() -> None:
    print("\nQuantCore -- Alpaca Diagnostic")
    print("=" * 50)
    failures = 0

    # ---- Check 1: Environment variables ----
    print("\n[1] Environment Variables")
    for var in REQUIRED_VARS:
        val = os.environ.get(var, "")
        if not val:
            print(f"  {FAIL} {var}: NOT SET")
            failures += 1
        elif var == "ALPACA_BASE_URL":
            mode = "PAPER" if "paper" in val else "LIVE"
            print(f"  {PASS} {var}: {val} ({mode} mode)")
            if mode == "LIVE":
                print(f"  {WARN} WARNING: LIVE mode -- real money at risk")
        else:
            masked = val[:4] + "***" + val[-4:] if len(val) > 8 else "***"
            print(f"  {PASS} {var}: {masked}")

    if failures > 0:
        print(f"\n{FAIL} Cannot continue -- {failures} required environment variable(s) missing")
        print("  Add them to your .env file. See .env.example for the template.")
        sys.exit(1)

    # ---- Check 2: Trading account ----
    print("\n[2] Trading Account")
    from alpaca.trading.client import TradingClient

    paper = "paper" in os.environ.get("ALPACA_BASE_URL", "")
    client = TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_API_SECRET"],
        paper=paper,
    )

    ok = True
    ok &= check("Account status", lambda: client.get_account().status)
    ok &= check(
        "Buying power",
        lambda: f"${float(client.get_account().buying_power):,.2f}",
    )
    ok &= check(
        "Portfolio value",
        lambda: f"${float(client.get_account().portfolio_value):,.2f}",
    )
    ok &= check(
        "Daytrade count",
        lambda: f"{client.get_account().daytrade_count} (PDT limit: 3 per 5 days)",
    )
    if not ok:
        failures += 1

    # ---- Check 3: Market status ----
    print("\n[3] Market Status")
    clock = client.get_clock()
    check(
        "Market is open",
        lambda: "YES" if clock.is_open else f"NO -- next open: {clock.next_open.strftime('%Y-%m-%d %H:%M ET')}",
    )
    check(
        "Current time (ET)",
        lambda: clock.timestamp.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )

    # ---- Check 4: Market data feed ----
    print("\n[4] Market Data Feed")
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestBarRequest

    data_client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_API_SECRET"],
    )

    test_symbols = ["SPY", "QQQ", "AAPL"]
    for sym in test_symbols:
        ok &= check(
            f"Latest bar ({sym})",
            lambda s=sym: (
                lambda b: f"close=${b.close:.2f}, volume={b.volume:,}, "
                f"age={int((datetime.now(UTC) - b.timestamp).total_seconds())}s"
            )(data_client.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=[s]))[s]),
        )

    # ---- Check 5: Order management ----
    print("\n[5] Order Management")
    check("Open orders", lambda: f"{len(client.get_orders())} open orders")
    check("Open positions", lambda: f"{len(client.get_all_positions())} open positions")

    # ---- Check 6: API latency ----
    print("\n[6] API Latency")
    times: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        client.get_account()
        times.append((time.perf_counter() - t0) * 1000)
    avg_latency = sum(times) / len(times)
    check(
        "API round-trip (avg of 3)",
        lambda: f"{avg_latency:.1f}ms {'OK' if avg_latency < 500 else 'SLOW -- check connection'}",
    )

    # ---- Check 7: Telegram alerting ----
    print("\n[7] Telegram Alerts")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print(f"  {WARN} Telegram not configured -- alerts will be disabled")
        print("      To enable: see .env.example for TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    else:
        try:
            from quantcore.alerts.telegram_alert import send_alert
            ok_telegram = send_alert(
                "QuantCore diagnostic check -- Telegram alerts working", level="INFO"
            )
            if ok_telegram:
                print(f"  {PASS} Telegram: test message sent successfully")
            else:
                print(f"  {FAIL} Telegram: failed to send test message -- check token and chat ID")
                failures += 1
        except ImportError:
            print(f"  {WARN} Telegram module not yet available -- will be added in Section 5")

    # Check 8: IBKR availability (optional)
    print("\n[8] IBKR Connection (optional -- only if using IBKR broker)")
    ibkr_port = int(os.environ.get("IBKR_PORT", "7497"))
    ibkr_host = os.environ.get("IBKR_HOST", "127.0.0.1")
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((ibkr_host, ibkr_port))
    sock.close()
    if result == 0:
        print(f"  {PASS} TWS/Gateway port {ibkr_port} is open -- IBKR connection available")
        print(f"       Run python scripts/verify_ibkr.py for full IBKR diagnostics")
    else:
        print(f"  {WARN} TWS/Gateway not detected at {ibkr_host}:{ibkr_port}")
        print(f"       This is OK if you are using --broker alpaca only")
        print(f"       To use IBKR: start TWS, enable API, set port to {ibkr_port}")

    # ---- Summary ----
    print("\n" + "=" * 50)
    if failures == 0:
        print(f"{PASS} All checks passed -- QuantCore is ready to trade")
    else:
        print(f"{FAIL} {failures} check(s) failed -- resolve before trading")
        sys.exit(1)
    print()


if __name__ == "__main__":
    main()
